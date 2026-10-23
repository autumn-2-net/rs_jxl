// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG VarDCT entropy orchestration from enc_frame.cc::EncodeGlobalACInfo,
//! enc_group.cc::EncodeGroupTokenizedCoefficients and enc_coeff_order.cc.
//! Reuses the ported entropy/LZ engines, without coefficient or pixel transforms.

use super::coefficient_order::{Dct8Orders, PERMUTATION_CONTEXTS};
use crate::entropy::{
    options::{EntropyOptions, SpeedTier},
    stream::{Coding, MAX_RAW_CONTEXTS, Model, Token},
};
use crate::tools::lz77::{self, Literal, SourceConfig};
use crate::{Error, Result, bit_writer::BitWriter, codestream::write_choice};

#[derive(Clone, Copy)]
pub enum LzPolicy<'a> {
    Disabled,
    Rle,
    Explicit(&'a SourceConfig),
}

/// One shared entropy model, with a separately terminated token stream per
/// AC group. Acceptance of LZ is retained independently of actual copy count.
pub struct CoefficientEntropy {
    model: Model,
    streams: Vec<Vec<Token>>,
    lz_enabled: bool,
}

impl CoefficientEntropy {
    pub fn source(
        streams: Vec<Vec<Token>>,
        contexts: usize,
        tier: SpeedTier,
        decoding_speed: u8,
        max_tokens: usize,
    ) -> Result<Self> {
        let mut options = EntropyOptions::for_speed_tier(tier);
        if decoding_speed >= 1 {
            options.max_histograms = 6;
        }
        let lz = if tier > SpeedTier::Tortoise {
            LzPolicy::Disabled
        } else {
            LzPolicy::Rle
        };
        Self::new(streams, contexts, options, lz, max_tokens)
    }

    pub fn new(
        mut streams: Vec<Vec<Token>>,
        contexts: usize,
        options: EntropyOptions,
        lz: LzPolicy<'_>,
        max_tokens: usize,
    ) -> Result<Self> {
        options.validate()?;
        if streams.is_empty() || contexts == 0 || contexts >= MAX_RAW_CONTEXTS {
            return Err(Error::InvalidInput(
                "invalid JPEG entropy stream/context count".into(),
            ));
        }
        let mut total = 0usize;
        for stream in &streams {
            total = total
                .checked_add(stream.len())
                .filter(|&n| n <= max_tokens)
                .ok_or(Error::Limit("JPEG coefficient tokens"))?;
            if stream.iter().any(|t| t.context >= contexts || t.is_length) {
                return Err(Error::InvalidInput(
                    "JPEG raw coefficient token is not a valid literal".into(),
                ));
            }
        }
        let mut lz_enabled = false;
        if !matches!(lz, LzPolicy::Disabled) {
            let config = match lz {
                LzPolicy::Explicit(config) => config.clone(),
                _ => SourceConfig {
                    length_uint_config: options.lz77_length_config,
                    min_symbol: Some(options.lz77_min_symbol),
                    ..Default::default()
                },
            };
            if config.length_uint_config != options.lz77_length_config
                || config.min_symbol() != options.lz77_min_symbol
            {
                return Err(Error::InvalidInput(
                    "JPEG LZ cost/wire configuration differs".into(),
                ));
            }
            let literals: Vec<Vec<Literal>> = streams
                .iter()
                .map(|stream| {
                    stream
                        .iter()
                        .map(|t| Literal {
                            context: t.context as u32,
                            value: t.value,
                        })
                        .collect()
                })
                .collect();
            let slices: Vec<_> = literals.iter().map(Vec::as_slice).collect();
            let widths = vec![0; slices.len()]; // VarDCT entropy has no spatial distance multiplier.
            let parsed = if matches!(lz, LzPolicy::Rle) {
                lz77::parse_source_rle_streams_with_state(&slices, &widths, &config)?
            } else {
                lz77::parse_source_streams_with_state(&slices, &widths, &config)?
            };
            lz_enabled = parsed.enabled;
            streams = parsed
                .commands
                .into_iter()
                .map(|commands| {
                    crate::modular::encode::commands_to_tokens(commands, contexts, 0)
                        .map(|(tokens, _)| tokens)
                })
                .collect::<Result<_>>()?;
        }
        let model = Model::build_with_options(
            &streams,
            contexts + usize::from(lz_enabled),
            lz_enabled,
            Coding::Source,
            &options,
        )?;
        Ok(Self {
            model,
            streams,
            lz_enabled,
        })
    }

    pub fn group_count(&self) -> usize {
        self.streams.len()
    }
    pub fn lz_enabled(&self) -> bool {
        self.lz_enabled
    }
    pub fn write_model(&self, writer: &mut BitWriter) -> Result<()> {
        self.model.write_header(writer)
    }
    pub fn write_group(&self, index: usize, writer: &mut BitWriter) -> Result<()> {
        let stream = self
            .streams
            .get(index)
            .ok_or_else(|| Error::InvalidInput("JPEG AC group index out of range".into()))?;
        // One histogram group, so there is no per-group selector field.
        self.model.write_stream(stream, writer)
    }
}

/// Source uses a no-op non-XYB correlation base for JPEG reconstruction. The
/// generic default bit would select a nonzero Y-to-B base and is NOT valid here.
pub fn write_jpeg_color_correlation(writer: &mut BitWriter) -> Result<()> {
    writer.write(1, 0)?;
    writer.write(2, 0)?; // Color factor 84.
    writer.write(16, 0)?; // Zero F16 base X.
    writer.write(16, 0)?; // Zero F16 base B.
    writer.write(8, 128)?; // Signed DC factor zero.
    writer.write(8, 128)?;
    Ok(())
}

/// Writes the used-order field, optional permutation entropy model and tokens.
/// It does not insert byte padding before the following coefficient model.
pub fn write_orders(orders: &Dct8Orders, writer: &mut BitWriter) -> Result<()> {
    write_choice(
        writer,
        u32::from(orders.used_orders()),
        [(0x5f, 0), (0x13, 0), (0, 0), (0, 13)],
    )?;
    if orders.used_orders() != 0 {
        let entropy = CoefficientEntropy::new(
            vec![orders.tokens()],
            PERMUTATION_CONTEXTS,
            EntropyOptions::for_speed_tier(SpeedTier::Glacier),
            LzPolicy::Rle,
            3 * 64,
        )?;
        entropy.write_model(writer)?;
        entropy.write_group(0, writer)?;
    }
    Ok(())
}

/// One histogram-group selector is encoded using ceil(log2(num_ac_groups)) bits.
pub fn write_single_histogram_group(writer: &mut BitWriter, groups: usize) -> Result<()> {
    if groups == 0 {
        return Err(Error::InvalidInput(
            "JPEG AC group count must be nonzero".into(),
        ));
    }
    let bits = usize::BITS - (groups - 1).leading_zeros();
    writer.write(
        u8::try_from(bits).map_err(|_| Error::Limit("JPEG AC group count"))?,
        0,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_noop_correlation_and_empty_order_fields_at_arbitrary_alignment() {
        let mut writer = BitWriter::new(100);
        writer.write(3, 5).unwrap();
        write_jpeg_color_correlation(&mut writer).unwrap();
        assert_eq!(writer.bits_written(), 54);
        write_orders(&Dct8Orders::default(), &mut writer).unwrap();
        assert_eq!(writer.bits_written(), 56);
        let bytes = writer.into_bytes().unwrap();
        let read = |start: usize, n: usize| {
            (0..n).fold(0u64, |v, i| {
                v | u64::from((bytes[(start + i) / 8] >> ((start + i) % 8)) & 1) << i
            })
        };
        assert_eq!(read(0, 3), 5);
        assert_eq!(read(3, 35), 0);
        assert_eq!(read(38, 8), 128);
        assert_eq!(read(46, 8), 128);
        assert_eq!(read(54, 2), 2);
        for (groups, bits) in [(1, 0), (2, 1), (3, 2), (4, 2), (65536, 16)] {
            let mut w = BitWriter::new(8);
            write_single_histogram_group(&mut w, groups).unwrap();
            assert_eq!(w.bits_written(), bits);
        }
    }

    #[test]
    fn source_routes_share_model_and_bound_each_output() {
        let streams = vec![
            vec![Token::literal(0, 0); 4096],
            (0..64).map(|i| Token::literal(1, i)).collect(),
        ];
        for tier in [SpeedTier::Tortoise, SpeedTier::Squirrel] {
            let model = CoefficientEntropy::source(streams.clone(), 2, tier, 0, 5000).unwrap();
            assert_eq!(model.group_count(), 2);
            if tier == SpeedTier::Squirrel {
                assert!(!model.lz_enabled());
            }
            let mut writer = BitWriter::new(1 << 20);
            model.write_model(&mut writer).unwrap();
            for group in 0..2 {
                model.write_group(group, &mut writer).unwrap();
            }
            assert!(writer.bits_written() > 0);
            let mut limited = BitWriter::new(0);
            assert!(model.write_model(&mut limited).is_err());
            assert!(model.write_group(2, &mut writer).is_err());
        }
    }

    #[test]
    fn malformed_raw_tokens_and_configuration_fail_before_model_emission() {
        let options = EntropyOptions::default();
        assert!(CoefficientEntropy::new(vec![], 1, options, LzPolicy::Disabled, 1).is_err());
        assert!(
            CoefficientEntropy::new(
                vec![vec![Token::literal(1, 0)]],
                1,
                options,
                LzPolicy::Disabled,
                1
            )
            .is_err()
        );
        assert!(
            CoefficientEntropy::new(
                vec![vec![Token::literal(0, 0); 2]],
                1,
                options,
                LzPolicy::Disabled,
                1
            )
            .is_err()
        );
        assert!(
            CoefficientEntropy::new(
                vec![vec![Token {
                    context: 0,
                    value: 0,
                    is_length: true
                }]],
                1,
                options,
                LzPolicy::Disabled,
                1
            )
            .is_err()
        );
        assert!(
            CoefficientEntropy::new(
                vec![vec![]],
                1,
                options,
                LzPolicy::Explicit(&SourceConfig::default()),
                1
            )
            .is_err()
        );
        let mut writer = BitWriter::new(100);
        assert!(write_single_histogram_group(&mut writer, 0).is_err());
        assert_eq!(writer.bits_written(), 0);
    }
}
