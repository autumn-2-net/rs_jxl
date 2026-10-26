//! Byte-reversible JPEG input API, composed entirely from ported Rust tools.
//! This is independent of PNG/JPEG pixel input and never falls back to it.

use std::io::Write;

use super::{
    cfl::CflMode,
    coefficient_order::OrderPolicy,
    coefficient_reader::{
        CoefficientReadError, CoefficientReadLimits, read_coefficients_with_limits,
    },
    data::{JpegData, JpegError, JpegRecord},
    transcode::{Options as CoefficientOptions, encode_coefficients},
};
use crate::{Error, Result, entropy::options::SpeedTier, scheduler::WorkerLimits};

/// Optional caller budgets; `usize::MAX` (the default) imposes no extra ceiling.
#[derive(Clone, Debug)]
pub struct TranscodeLimits {
    /// Structural, coefficient-read and scan-work limits, checked before
    /// coefficient allocations. Includes input byte and metadata limits.
    pub input: CoefficientReadLimits,
    pub max_output_bytes: usize,
    /// Maximum i32 coefficients in the full three-plane JXL working layout,
    /// including MCU padding and the zero tails of subsampled AC groups.
    pub max_coefficients: usize,
    pub max_tokens: usize,
    /// Hard selected-sample ceiling across auxiliary MA tree domains.
    /// `usize::MAX` keeps all samples selected by the effort's learning policy.
    pub sample_limit: usize,
}

impl Default for TranscodeLimits {
    fn default() -> Self {
        Self {
            input: CoefficientReadLimits::default(),
            max_output_bytes: usize::MAX,
            max_coefficients: usize::MAX,
            max_tokens: usize::MAX,
            sample_limit: usize::MAX,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TranscodeOptions {
    /// Direct coefficient backend effort, 1..=11. The JPEG frame entry maps
    /// effort 1 to backend 2 and 11 to 10, as in the source VarDCT path.
    pub effort: u8,
    /// 1 stays on the caller; 2..=256 uses a bounded per-call worker pool.
    pub threads: usize,
    /// Optional source cooperative JPEG MA split execution. None keeps domain
    /// parallelism only. Limits bound live learners, not the pool's thread count.
    pub cooperative_ma: Option<WorkerLimits>,
    pub decoding_speed: u8,
    pub cfl: CflMode,
    pub order: OrderPolicy,
    /// Explicit Brotli quality 0..=11. None selects source effort+1,
    /// capped at Brotli's maximum of 11.
    pub reconstruction_quality: Option<u8>,
    pub limits: TranscodeLimits,
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        Self {
            effort: 3,
            threads: 1,
            cooperative_ma: None,
            decoding_speed: 0,
            cfl: CflMode::Enabled,
            order: OrderPolicy::Source,
            reconstruction_quality: None,
            limits: TranscodeLimits::default(),
        }
    }
}

impl TranscodeOptions {
    pub(crate) fn resolve(&self) -> Result<CoefficientOptions> {
        if let Some(limits) = self.cooperative_ma {
            limits.validate()?;
        }
        if !(1..=11).contains(&self.effort)
            || !(1..=256).contains(&self.threads)
            || self.decoding_speed > 4
            || self.reconstruction_quality.is_some_and(|q| q > 11)
        {
            return Err(Error::InvalidInput(
                "invalid JPEG transcoding options".into(),
            ));
        }
        if self.limits.max_output_bytes == 0
            || self.limits.max_coefficients == 0
            || self.limits.max_tokens == 0
            || self.limits.sample_limit == 0
        {
            return Err(Error::Limit("JPEG transcode budgets must be nonzero"));
        }
        Ok(CoefficientOptions {
            tier: SpeedTier::try_from(10 - self.effort.clamp(2, 10) as i8)?,
            threads: self.threads,
            cooperative_ma: self.cooperative_ma,
            decoding_speed: self.decoding_speed,
            sample_limit: self.limits.sample_limit,
            reconstruction_quality: self
                .reconstruction_quality
                .unwrap_or((self.effort + 1).min(11)),
            cfl: self.cfl,
            order: self.order,
            max_output_bytes: self.limits.max_output_bytes,
            max_coefficients: self.limits.max_coefficients,
            max_tokens: self.limits.max_tokens,
        })
    }
}

/// Encode an 8-bit Huffman DCT JPEG as a standard JXL container with `jbrd`.
/// Original scan ordering, metadata, padding and trailing bytes are retained
/// for exact JPEG reconstruction. No IDCT, pixel decoding or re-quantization
/// occurs. Unsupported JPEG structures or exhausted budgets return an error.
/// ICC metadata is validated strictly; malformed profiles are not discarded.
pub fn transcode(input: &[u8], options: &TranscodeOptions) -> Result<Vec<u8>> {
    let settings = options.resolve()?;
    let jpeg = read_input(input, &options.limits.input)?;
    encode_coefficients(&jpeg, &settings)
}

pub(crate) fn read_input(input: &[u8], limits: &CoefficientReadLimits) -> Result<JpegData> {
    let mut jpeg = read_coefficients_with_limits(input, limits).map_err(|e| match e {
        CoefficientReadError::Jpeg(JpegError::LimitExceeded { resource, .. }) => {
            Error::Limit(resource)
        }
        other => Error::InvalidInput(other.to_string()),
    })?;
    // Encoding/reconstruction must be based on coefficients, not cached scan
    // bytes. Release the reader's archival copy before JXL model allocation.
    for record in &mut jpeg.records {
        if let JpegRecord::Entropy(entropy) = record {
            entropy.bytes = Vec::new();
        }
    }
    Ok(jpeg)
}

/// Encode completely before writing to a caller-owned sink. Encoding failures
/// write nothing. An I/O failure may leave partial output, as with Write::write_all.
pub fn transcode_to(
    input: &[u8],
    options: &TranscodeOptions,
    output: &mut impl Write,
) -> Result<()> {
    output.write_all(&transcode(input, options)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jpeg() -> Vec<u8> {
        // One gray 1x1 baseline block: DC zero and EOB each use the code 0.
        let mut bytes = vec![0xff, 0xd8, 0xff, 0xdb, 0, 67, 0];
        bytes.extend([16; 64]);
        bytes.extend([
            0xff, 0xc0, 0, 11, 8, 0, 1, 0, 1, 1, 1, 0x11, 0, 0xff, 0xc4, 0, 38,
        ]);
        for selector in [0, 0x10] {
            bytes.extend([selector, 1]);
            bytes.extend([0; 15]);
            bytes.push(0);
        }
        bytes.extend([0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0, 0x3f, 0xff, 0xd9]);
        bytes
    }

    #[test]
    fn public_one_byte_output_budget_is_a_limit_and_leaves_sink_unchanged() {
        let jpeg = tiny_jpeg();
        assert!(
            !transcode(&jpeg, &TranscodeOptions::default())
                .unwrap()
                .is_empty()
        );
        let options = TranscodeOptions {
            limits: TranscodeLimits {
                max_output_bytes: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let error = transcode(&jpeg, &options).unwrap_err();
        assert!(matches!(error, Error::Limit(_)), "{error:?}");
        let mut sink = vec![19, 87];
        let error = transcode_to(&jpeg, &options, &mut sink).unwrap_err();
        assert!(matches!(error, Error::Limit(_)), "{error:?}");
        assert_eq!(sink, [19, 87]);
    }

    #[test]
    fn public_sample_budget_is_opt_in_and_does_not_reduce_sampling() {
        let jpeg = tiny_jpeg();
        let mut options = TranscodeOptions {
            effort: 8,
            ..Default::default()
        };
        let uncapped = transcode(&jpeg, &options).unwrap();
        options.limits.sample_limit = 1;
        assert!(matches!(transcode(&jpeg, &options), Err(Error::Limit(_))));
        options.limits.sample_limit = 1 << 20;
        assert_eq!(transcode(&jpeg, &options).unwrap(), uncapped);
    }

    #[test]
    fn public_coefficient_and_token_budgets_remain_limits() {
        let jpeg = tiny_jpeg();
        for limits in [
            TranscodeLimits {
                max_coefficients: 1,
                ..Default::default()
            },
            TranscodeLimits {
                max_tokens: 1,
                ..Default::default()
            },
            TranscodeLimits {
                input: CoefficientReadLimits {
                    max_coefficient_bytes: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        ] {
            let options = TranscodeOptions {
                limits,
                ..Default::default()
            };
            let error = transcode(&jpeg, &options).unwrap_err();
            assert!(matches!(error, Error::Limit(_)), "{error:?}");
        }
    }

    #[test]
    fn public_malformed_and_unsupported_jpegs_remain_invalid_input() {
        let mut invalid_quant = tiny_jpeg();
        invalid_quant[7] = 0;
        let mut unsupported = tiny_jpeg();
        let sof = unsupported
            .windows(2)
            .position(|w| w == [0xff, 0xc0])
            .unwrap();
        unsupported[sof + 1] = 0xc3;
        for jpeg in [b"not a JPEG".to_vec(), invalid_quant, unsupported] {
            let error = transcode(&jpeg, &TranscodeOptions::default()).unwrap_err();
            assert!(matches!(error, Error::InvalidInput(_)), "{error:?}");
        }
    }

    #[test]
    fn all_efforts_resolve_and_invalid_options_reject_before_input() {
        for effort in 1..=11 {
            let settings = TranscodeOptions {
                effort,
                ..Default::default()
            }
            .resolve()
            .unwrap();
            assert_eq!(settings.tier as i8, 10 - effort.clamp(2, 10) as i8);
            assert_eq!(settings.reconstruction_quality, (effort + 1).min(11));
        }
        for effort in [0, 12, 255] {
            assert!(
                TranscodeOptions {
                    effort,
                    ..Default::default()
                }
                .resolve()
                .is_err()
            );
        }
        for threads in [0, 257, usize::MAX] {
            assert!(
                TranscodeOptions {
                    threads,
                    ..Default::default()
                }
                .resolve()
                .is_err()
            );
        }
        assert!(
            TranscodeOptions {
                decoding_speed: 5,
                ..Default::default()
            }
            .resolve()
            .is_err()
        );
        assert!(
            TranscodeOptions {
                reconstruction_quality: Some(12),
                ..Default::default()
            }
            .resolve()
            .is_err()
        );
    }

    #[test]
    fn failed_encoding_never_writes_to_sink() {
        let mut sink = vec![19, 87];
        assert!(transcode_to(b"not JPEG", &TranscodeOptions::default(), &mut sink).is_err());
        assert_eq!(sink, [19, 87]);
        let options = TranscodeOptions {
            limits: TranscodeLimits {
                max_output_bytes: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(transcode_to(&[], &options, &mut sink).is_err());
        assert_eq!(sink, [19, 87]);
    }

    #[test]
    fn cooperative_limits_resolve_early_and_keep_search_outputs_and_order() {
        use crate::jpeg::search::{SearchLimits, transcode_search};
        assert!(
            TranscodeOptions::default()
                .resolve()
                .unwrap()
                .cooperative_ma
                .is_none()
        );
        for limits in [
            WorkerLimits {
                tasks: 0,
                scratch_bytes: 1,
            },
            WorkerLimits {
                tasks: 65,
                scratch_bytes: 1,
            },
            WorkerLimits {
                tasks: 1,
                scratch_bytes: 0,
            },
        ] {
            let options = TranscodeOptions {
                cooperative_ma: Some(limits),
                ..Default::default()
            };
            let mut sink = vec![19, 87];
            let error = transcode_to(&[], &options, &mut sink).unwrap_err();
            assert!(matches!(error, Error::InvalidInput(message) if message.contains("worker")));
            assert_eq!(sink, [19, 87]);
            assert!(
                matches!(transcode_search(&[], &options, SearchLimits::default()), Err(Error::InvalidInput(message)) if message.contains("worker"))
            );
        }
        let input = tiny_jpeg();
        let base = TranscodeOptions {
            effort: 9,
            ..Default::default()
        };
        let encoded = transcode(&input, &base).unwrap();
        let search = transcode_search(&input, &base, SearchLimits::default()).unwrap();
        let trials = |result: &crate::jpeg::search::SearchResult| {
            result
                .trials
                .iter()
                .map(|trial| (trial.stage, trial.backend_effort, trial.bytes))
                .collect::<Vec<_>>()
        };
        for threads in [1, 3] {
            let options = TranscodeOptions {
                threads,
                cooperative_ma: Some(WorkerLimits {
                    tasks: 1,
                    scratch_bytes: 1,
                }),
                ..base.clone()
            };
            let resolved = options.resolve().unwrap().cooperative_ma.unwrap();
            assert_eq!((resolved.tasks, resolved.scratch_bytes), (1, 1));
            assert_eq!(transcode(&input, &options).unwrap(), encoded);
            let actual = transcode_search(&input, &options, SearchLimits::default()).unwrap();
            assert_eq!(actual.data, search.data);
            assert_eq!(
                (
                    actual.backend_effort,
                    actual.winner_index,
                    actual.full_pixels
                ),
                (
                    search.backend_effort,
                    search.winner_index,
                    search.full_pixels
                )
            );
            assert_eq!(trials(&actual), trials(&search));
        }
    }
}
