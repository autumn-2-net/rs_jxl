// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG integer auxiliary-stream policy from frozen
//! b866b3defec13494bfbaf705db93581cda118e37: enc_modular.cc::{Init,
//! AddVarDCTDC, AddACMetadata, AddQuantTable, ComputeTree}, enc_ans.cc::
//! HistogramParams::ForModular, and enc_encoding.cc::PredefinedTree.
//!
//! Only nl_dc=false, jpeg_transcode=true, default predictor selection, zero
//! previous-channel properties, and non-streaming JPEG frames are represented.
//! Prepared DC channels stay in Modular Y/X/B order (JXL 1/0/2). AC metadata
//! stays YToX, YToB, [strategy; raw_quant-1], EPF. Quant tables keep their input
//! channel order. No color transform, palette, squeeze, or WP search is added.
//!
//! Pass the returned options to modular::encode::write_prepared_stream with
//! bits=8 and the real stream ID. A shared-frame composer must instead retain
//! source tree domains and use options_for_global_entropy for BOTH the merged
//! tree and shared residual model. Local quant-table RLE is supported for both
//! tree and data tokens by the local Modular writer; it is not a global policy.

use crate::entropy::options::{EntropyOptions, ModularEntropyParams, SpeedTier};
use crate::entropy::stream::Coding;
use crate::modular::encode::{FixedTree, ModularOptions};
use crate::modular::fixed::{self, FixedTreeMode};
use crate::modular::ma_train::MAX_LEAVES;
use crate::modular::predictor::Predictor;
use crate::modular::sample::{SamplingPolicy, SourcePolicy, TreeMode};
use crate::tools::lz77::{Lz77Mode, SourceConfig, SourceLz77Method};
use crate::{Error, Result};

/// Frame facts not inferable from encoder effort. No process environment is read.
#[derive(Clone, Copy, Debug)]
pub struct StreamContext {
    /// All Modular slots, including empty ones. Source: 1 + 3*DC_groups + 17 +
    /// AC_groups*passes. The default 22 is one DC group, AC group, and pass.
    pub num_streams: usize,
    /// Unscaled source nb_repeats; effort scaling is applied exactly once.
    pub learning_fraction: f32,
    /// Source slow-tier override, true by default in CompressParams. This flag,
    /// not the presence of nonzero/eligible CfL maps, selects metadata learning.
    pub force_cfl_jpeg_recompression: bool,
    /// Sum of all channel areas in the shared DC tree domain. Some builds the
    /// source ComputeTree template at bitdepth 8 with prevprop=0. None defers
    /// population counting to the local writer; callers must still pass bits=8.
    pub dc_tree_pixels: Option<usize>,
}

impl Default for StreamContext {
    fn default() -> Self {
        Self {
            num_streams: 22,
            learning_fraction: 0.5,
            force_cfl_jpeg_recompression: true,
            dc_tree_pixels: None,
        }
    }
}

/// AddVarDCTDC(nl_dc=false, jpeg_transcode=true), for default frame context.
/// `tier` is common.h::SpeedTier (-1..=9), not effort; decode_speed is 0..=4.
pub fn options_for_dc(tier: i8, decode_speed: u8) -> Result<ModularOptions> {
    StreamContext::default().options_for_dc(tier, decode_speed)
}

/// AddACMetadata(jpeg_transcode=true), with source-default forced CfL enabled.
/// Below Squirrel this learns a tree even when the prepared CfL maps are zero.
pub fn options_for_ac_metadata(tier: i8, decode_speed: u8) -> Result<ModularOptions> {
    StreamContext::default().options_for_ac_metadata(tier, decode_speed)
}

/// AddQuantTable: inherited Init learning policy and default local histogram
/// policy, not the standalone EncodeQuantTable fallback's unscaled defaults.
pub fn options_for_quant(tier: i8, decode_speed: u8) -> Result<ModularOptions> {
    StreamContext::default().options_for_quant(tier, decode_speed)
}

/// EncodeGlobalInfo's non-streaming JPEG entropy policy, shared by merged tree
/// tokens and every DC/AC-metadata/quant-table residual stream. Source nl_dc is
/// false, so extra DC precision never enables the Fast hybrid-uint search.
///
/// This is an entropy/resource carrier, NOT a fourth tree domain: sampling,
/// fixed_tree and custom_tree are absent. Build each domain with its own local
/// policy and actual IDs/population, then give this value to the shared writer.
/// In particular, never substitute options_for_quant: its local RLE and default
/// histogram limits do not describe the shared frame. Existing resource fields
/// (including sample_limit) remain explicit caller-adjustable hard ceilings.
pub fn options_for_global_entropy(tier: i8, decode_speed: u8) -> Result<ModularOptions> {
    // Reuse the DC/AC ForModular mapping before any fixed-tree attachment.
    let mut options = StreamContext::default().options(tier, decode_speed, StreamKind::Dc)?;
    options.predictor = Predictor::Gradient;
    options.sampling = None;
    options.validate()?;
    Ok(options)
}

impl StreamContext {
    pub fn options_for_dc(&self, tier: i8, decode_speed: u8) -> Result<ModularOptions> {
        let mut options = self.options(tier, decode_speed, StreamKind::Dc)?;
        if decode_speed >= 1 || tier >= SpeedTier::Squirrel as i8 {
            let gradient = decode_speed >= 1;
            if let Some(total_pixels) = self.dc_tree_pixels {
                let mode = if gradient {
                    FixedTreeMode::Gradient {
                        total_pixels,
                        previous_channels: 0,
                    }
                } else {
                    FixedTreeMode::Weighted { total_pixels }
                };
                options.custom_tree = Some(fixed::build(8, mode)?);
            } else {
                options.fixed_tree = Some(if gradient {
                    FixedTree::Gradient
                } else {
                    FixedTree::Weighted
                });
            }
        }
        options.validate()?;
        Ok(options)
    }

    pub fn options_for_ac_metadata(&self, tier: i8, decode_speed: u8) -> Result<ModularOptions> {
        let mut options = self.options(tier, decode_speed, StreamKind::AcMetadata)?;
        if !(tier < SpeedTier::Squirrel as i8 && self.force_cfl_jpeg_recompression) {
            // kJpegTranscodeACMeta is one Zero leaf, NOT kACMeta's channel tree.
            options.custom_tree = Some(fixed::build(8, FixedTreeMode::Zero)?);
        }
        options.validate()?;
        Ok(options)
    }

    pub fn options_for_quant(&self, tier: i8, decode_speed: u8) -> Result<ModularOptions> {
        let options = self.options(tier, decode_speed, StreamKind::Quant)?;
        options.validate()?;
        Ok(options)
    }

    fn options(&self, tier: i8, decode_speed: u8, kind: StreamKind) -> Result<ModularOptions> {
        let tier = SpeedTier::try_from(tier)?;
        if decode_speed > 4 {
            return Err(Error::InvalidInput(
                "JPEG Modular decoding speed must be 0..=4".into(),
            ));
        }
        // Init's IsLossless() is false for JPEG/VarDCT, whereas its
        // ModularPartIsLossless() is true. Thus its default is Gradient, not
        // the pixel-lossless Best/Variable predictor choice.
        let (predictor, tree_mode, max_channel_size) = match kind {
            StreamKind::Dc => {
                if tier >= SpeedTier::Squirrel {
                    (Predictor::Weighted, TreeMode::WeightedOnly, 0x00ff_ffff)
                } else {
                    (
                        if tier < SpeedTier::Kitten {
                            Predictor::Variable
                        } else {
                            Predictor::Best
                        },
                        TreeMode::Default,
                        0x00ff_ffff,
                    )
                }
            }
            StreamKind::AcMetadata => (Predictor::Gradient, TreeMode::NoWeighted, 0x00ff_ffff),
            StreamKind::Quant => (
                Predictor::Gradient,
                match decode_speed {
                    0 => TreeMode::Default,
                    1 | 2 => TreeMode::NoWeighted,
                    _ => TreeMode::GradientOnly,
                },
                // GetGroupSizeShift always returns 1 for non-Modular frames.
                256,
            ),
        };
        // AddVarDCTDC and AddACMetadata override Init's tree mode AFTER its
        // decoding-speed restrictions. Resolve properties using that final
        // mode; retain Init's sample-zero and threshold effects separately.
        let mut sampling = SamplingPolicy::from_source(&SourcePolicy {
            speed_tier: tier as i8,
            decoding_speed_tier: 0,
            predictor,
            num_streams: self.num_streams,
            learning_fraction: self.learning_fraction,
            tree_mode,
            max_channel_size,
            ..SourcePolicy::default()
        })?;
        if decode_speed >= 4 {
            sampling.nb_repeats = 0.0;
        }
        sampling.node_threshold += 10.0 * f32::from(decode_speed);
        sampling.fast_decode_multiplier = 1.0;

        let (entropy_options, source_method) = if matches!(kind, StreamKind::Quant) {
            // AddQuantTable leaves stream_options_[id].histogram_params alone:
            // HistogramParams() is Best/Precise/Best + RLE + length uint000.
            (EntropyOptions::for_speed_tier(SpeedTier::Tortoise), None)
        } else {
            (
                EntropyOptions::for_modular(ModularEntropyParams {
                    speed_tier: tier,
                    modular_mode: false,
                    lossless: false,
                    decoding_speed_tier: decode_speed,
                    ..ModularEntropyParams::default()
                }),
                if tier <= SpeedTier::Tortoise {
                    Some(SourceLz77Method::Optc256)
                } else if tier == SpeedTier::Kitten {
                    Some(SourceLz77Method::Lz77b3w3t)
                } else {
                    None
                },
            )
        };
        let source_lz77 = source_method.map(|method| SourceConfig {
            method,
            length_uint_config: entropy_options.lz77_length_config,
            min_symbol: Some(entropy_options.lz77_min_symbol),
            ..SourceConfig::default()
        });
        Ok(ModularOptions {
            predictor,
            rct: None,
            max_tree_leaves: MAX_LEAVES,
            split_threshold: f64::from(sampling.node_threshold),
            entropy: Coding::Source,
            lz77: matches!(kind, StreamKind::Quant) || source_method.is_some(),
            lz77_mode: match source_method {
                Some(SourceLz77Method::Optc256) => Lz77Mode::Optimal,
                Some(_) => Lz77Mode::Greedy,
                None => Lz77Mode::Rle,
            },
            sampling: Some(sampling),
            entropy_options,
            source_lz77,
            source_tree_entropy: true,
            ..ModularOptions::default()
        })
    }
}

#[derive(Clone, Copy)]
enum StreamKind {
    Dc,
    AcMetadata,
    Quant,
}

#[cfg(test)]
#[path = "modular_policy_tests.rs"]
mod tests;
