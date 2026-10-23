// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Entropy policies from enc_ans_params.h::HistogramParams and
//! enc_ans.cc::HistogramParams::ForModular at
//! b866b3defec13494bfbaf705db93581cda118e37.
//!
//! The general constructor and Modular mapping are intentionally separate.
//! LZ77 parsing/selection remains the caller's responsibility: these policies
//! neither create nor remove length/distance tokens. Process environment,
//! audit/replay and worker scheduling are not policies of this module.
//! Missing-symbol/fixed-histogram injection and streaming model reuse are
//! implemented in stream.rs. Environment-derived overrides must be
//! supplied explicitly, so they participate in the caller's candidate identity.
//!
//! ANS alphabet logarithms span 5..=8; prefix alphabets span 1..=32768 symbols
//! and signal logarithm 15. Adaptive uint search follows source population
//! costs, then selects the smallest supported ANS alphabet for the result.

pub use super::ans::AnsHistogramStrategy;
pub use super::cluster::ClusterStrategy;
use super::hybrid_uint::HybridUintConfig;
use crate::{Error, Result};

pub const MIN_LOG_ALPHA_SIZE: u8 = 5;
pub const MAX_LOG_ALPHA_SIZE: u8 = 8;
pub const PREFIX_LOG_ALPHA_SIZE: u8 = 15;
pub const MAX_HISTOGRAMS: usize = 256;

/// Numeric values are common.h::SpeedTier, not encoder effort values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i8)]
pub enum SpeedTier {
    TectonicPlate = -1,
    Glacier = 0,
    Tortoise = 1,
    Kitten = 2,
    Squirrel = 3,
    Wombat = 4,
    Hare = 5,
    Cheetah = 6,
    Falcon = 7,
    Thunder = 8,
    Lightning = 9,
}

impl TryFrom<i8> for SpeedTier {
    type Error = Error;

    fn try_from(value: i8) -> Result<Self> {
        match value {
            -1 => Ok(Self::TectonicPlate),
            0 => Ok(Self::Glacier),
            1 => Ok(Self::Tortoise),
            2 => Ok(Self::Kitten),
            3 => Ok(Self::Squirrel),
            4 => Ok(Self::Wombat),
            5 => Ok(Self::Hare),
            6 => Ok(Self::Cheetah),
            7 => Ok(Self::Falcon),
            8 => Ok(Self::Thunder),
            9 => Ok(Self::Lightning),
            _ => Err(Error::InvalidInput(
                "source speed tier must be -1..=9".into(),
            )),
        }
    }
}

impl TryFrom<u8> for SpeedTier {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        let value = i8::try_from(value)
            .map_err(|_| Error::InvalidInput("source speed tier must be 0..=9".into()))?;
        Self::try_from(value)
    }
}

/// Fixed implements source kNone=(4,2,0), k000=(0,0,0),
/// kContextMap=(2,0,1), or an explicit validated configuration. All mapped
/// histograms use it; length tokens retain their separate lz77_length_config.
/// Fast/Best search independently for every mapped literal/distance histogram.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UintConfigStrategy {
    Fixed(HybridUintConfig),
    Fast,
    #[default]
    Best,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyOptions {
    pub uint_config: UintConfigStrategy,
    pub clustering: ClusterStrategy,
    pub ans_histogram: AnsHistogramStrategy,
    /// 1..=256 new histograms, independent of the number of raw contexts.
    /// Streaming append permits zero for a frozen-table refit. The total
    /// including existing/fixed histograms is still at most 256.
    pub max_histograms: usize,
    /// Inclusive ANS alphabet range. Prefix coding always uses logarithm 15.
    pub min_log_alpha_size: u8,
    pub max_log_alpha_size: u8,
    /// Source streaming uint policy: skip adaptive search and use default420
    /// (or an explicit Fixed config), with ANS logarithm 8. Also retains encoded
    /// histogram bits for Model::append_with_options / deferred header output.
    pub streaming_mode: bool,
    /// Add one observation of each symbol 0..255 to EVERY raw context, even
    /// already-present symbols and the LZ77 distance context. This is the
    /// EncodeGlobalInfo streaming-global-tree support guarantee, not a default
    /// for independent local Modular streams or for tree-token entropy.
    pub add_missing_symbols: bool,
    /// Install the source's flat 256-symbol/4096-weight coding table before
    /// initial clustering. Requires streaming mode; never repeat on append.
    pub add_fixed_histograms: bool,
    /// Separate HybridUint config for length-minus-three, signaled with log 8.
    pub lz77_length_config: HybridUintConfig,
    /// Reserved length-symbol start: source presets 224, 512, 4096, or any
    /// value in 8..=32775 using the 15-bit offset form. Literal/distance tokens
    /// must stay below it. Prefix coding is required when length symbols >255.
    pub lz77_min_symbol: u32,
    /// Source force_huffman singleton-distance guard during clustering.
    /// Applied to prefix candidates only. False preserves Model::build's policy.
    pub prefix_singleton_guard: bool,
}

impl Default for EntropyOptions {
    fn default() -> Self {
        Self {
            uint_config: UintConfigStrategy::Best,
            clustering: ClusterStrategy::Best,
            ans_histogram: AnsHistogramStrategy::Precise,
            max_histograms: MAX_HISTOGRAMS,
            min_log_alpha_size: MIN_LOG_ALPHA_SIZE,
            max_log_alpha_size: MAX_LOG_ALPHA_SIZE,
            streaming_mode: false,
            add_missing_symbols: false,
            add_fixed_histograms: false,
            lz77_length_config: HybridUintConfig::new(4, 0, 0).expect("constant"),
            lz77_min_symbol: 224,
            prefix_singleton_guard: false,
        }
    }
}

/// Explicit equivalent of JXL_LOSSLESS_PROBE_ENTROPY's "1" and "2".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProbeEntropy {
    #[default]
    Default,
    FastUint,
    FastUintAndHistograms,
}

/// Entropy-relevant inputs to HistogramParams::ForModular. This function is
/// also used for VarDCT's Modular/DC streams, hence the modular_mode switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModularEntropyParams {
    pub speed_tier: SpeedTier,
    pub modular_mode: bool,
    pub lossless: bool,
    /// The explicit flag OR the source environment switch, before lossless gating.
    pub precise_entropy: bool,
    /// Source !extra_dc_precision.empty() && extra_dc_precision[0] != 0.
    pub has_extra_dc_precision: bool,
    pub decoding_speed_tier: u8,
    pub disposable_probe: bool,
    pub probe_entropy: ProbeEntropy,
}

impl Default for ModularEntropyParams {
    fn default() -> Self {
        Self {
            speed_tier: SpeedTier::Squirrel,
            modular_mode: true,
            lossless: true,
            precise_entropy: false,
            has_extra_dc_precision: false,
            decoding_speed_tier: 0,
            disposable_probe: false,
            probe_entropy: ProbeEntropy::Default,
        }
    }
}

impl EntropyOptions {
    /// General HistogramParams(tier, num_ctx), NOT ForModular. num_ctx is
    /// unused in the source constructor; the Rust model caps histograms at 256.
    /// Source LZ77 RLE/None selection is intentionally left to the caller.
    pub fn for_speed_tier(tier: SpeedTier) -> Self {
        let mut options = Self {
            lz77_length_config: HybridUintConfig::new(0, 0, 0).expect("constant"),
            ..Self::default()
        };
        if tier > SpeedTier::Falcon {
            options.clustering = ClusterStrategy::Fastest;
        } else if tier > SpeedTier::Tortoise {
            options.clustering = ClusterStrategy::Fast;
        }
        if tier > SpeedTier::Tortoise {
            options.uint_config = UintConfigStrategy::Fixed(HybridUintConfig::default());
        }
        if tier >= SpeedTier::Squirrel {
            options.ans_histogram = AnsHistogramStrategy::Approximate;
        }
        options
    }

    /// The entropy-policy portion of HistogramParams::ForModular, including
    /// lossless precise-entropy and disposable-probe override precedence.
    /// Predictor/LZ77, image widths and parser cost reuse belong to the caller.
    pub fn for_modular(params: ModularEntropyParams) -> Self {
        let mut options = Self {
            lz77_length_config: HybridUintConfig::new(0, 0, 0).expect("constant"),
            ..Self::default()
        };
        let tier = if params.lossless && params.precise_entropy {
            SpeedTier::Tortoise
        } else {
            params.speed_tier
        };
        if tier > SpeedTier::Kitten {
            options.clustering = ClusterStrategy::Fast;
            options.ans_histogram = if tier > SpeedTier::Thunder {
                AnsHistogramStrategy::Fast
            } else {
                AnsHistogramStrategy::Approximate
            };
            options.uint_config = if params.has_extra_dc_precision
                || (params.modular_mode && tier < SpeedTier::Cheetah)
            {
                UintConfigStrategy::Fast
            } else {
                UintConfigStrategy::Fixed(HybridUintConfig::default())
            };
        }
        if params.decoding_speed_tier >= 2 {
            options.max_histograms = 12;
        }
        if params.lossless && params.disposable_probe {
            match params.probe_entropy {
                ProbeEntropy::Default => {}
                ProbeEntropy::FastUint => options.uint_config = UintConfigStrategy::Fast,
                ProbeEntropy::FastUintAndHistograms => {
                    options.uint_config = UintConfigStrategy::Fast;
                    options.clustering = ClusterStrategy::Fast;
                    options.ans_histogram = AnsHistogramStrategy::Approximate;
                }
            }
        }
        options
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_histograms > MAX_HISTOGRAMS
            || (self.max_histograms == 0 && !self.streaming_mode)
        {
            return Err(Error::InvalidInput(
                "maximum histograms must be 1..=256".into(),
            ));
        }
        if self.add_fixed_histograms && !self.streaming_mode {
            return Err(Error::InvalidInput(
                "fixed histogram injection requires streaming mode".into(),
            ));
        }
        if self.min_log_alpha_size < MIN_LOG_ALPHA_SIZE
            || self.max_log_alpha_size > MAX_LOG_ALPHA_SIZE
            || self.min_log_alpha_size > self.max_log_alpha_size
        {
            return Err(Error::InvalidInput(
                "ANS log-alpha range must be within 5..=8".into(),
            ));
        }
        if !(8..=32775).contains(&self.lz77_min_symbol) {
            return Err(Error::InvalidInput(
                "LZ77 min-symbol must be in 8..=32775".into(),
            ));
        }
        let length = self.lz77_length_config;
        if length.split_exponent() > 8
            || (length.split_exponent() == 8
                && (length.msb_in_token() != 0 || length.lsb_in_token() != 0))
        {
            return Err(Error::InvalidInput(
                "LZ77 length config must fit its log-alpha 8 header".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_defaults_and_wire_domain_are_explicit() {
        let legacy = EntropyOptions::default();
        assert_eq!(
            legacy.lz77_length_config,
            HybridUintConfig::new(4, 0, 0).unwrap()
        );
        assert_eq!(legacy.lz77_min_symbol, 224);
        let modular = EntropyOptions::for_modular(ModularEntropyParams::default());
        assert_eq!(
            modular.lz77_length_config,
            HybridUintConfig::new(0, 0, 0).unwrap()
        );
        assert_eq!(modular.lz77_min_symbol, 224);
        let mut options = legacy;
        for minimum in [8, 224, 512, 4096, 32775] {
            options.lz77_min_symbol = minimum;
            options.validate().unwrap();
        }
        for minimum in [0, 7, 32776, u32::MAX] {
            options.lz77_min_symbol = minimum;
            assert!(options.validate().is_err());
        }
        options.lz77_min_symbol = 224;
        for config in [
            HybridUintConfig::new(9, 0, 0).unwrap(),
            HybridUintConfig::new(8, 1, 0).unwrap(),
        ] {
            options.lz77_length_config = config;
            assert!(options.validate().is_err());
        }
        options.lz77_length_config = HybridUintConfig::new(8, 0, 0).unwrap();
        options.validate().unwrap();
    }

    #[test]
    fn general_constructor_matches_every_source_tier() {
        use AnsHistogramStrategy::{Approximate, Precise};
        use ClusterStrategy::{Best, Fast, Fastest};
        let expected = [
            (Best, false, Precise),
            (Best, false, Precise),
            (Best, false, Precise),
            (Fast, true, Precise),
            (Fast, true, Approximate),
            (Fast, true, Approximate),
            (Fast, true, Approximate),
            (Fast, true, Approximate),
            (Fast, true, Approximate),
            (Fastest, true, Approximate),
            (Fastest, true, Approximate),
        ];
        for (tier, (clustering, fixed, ans)) in (-1i8..=9).zip(expected) {
            let options = EntropyOptions::for_speed_tier(SpeedTier::try_from(tier).unwrap());
            assert_eq!(options.clustering, clustering, "tier {tier}");
            assert_eq!(options.ans_histogram, ans, "tier {tier}");
            assert_eq!(
                options.uint_config,
                if fixed {
                    UintConfigStrategy::Fixed(HybridUintConfig::default())
                } else {
                    UintConfigStrategy::Best
                },
                "tier {tier}"
            );
            assert_eq!(options.max_histograms, 256);
            options.validate().unwrap();
        }
    }

    #[test]
    fn modular_mapping_matches_every_source_tier() {
        use AnsHistogramStrategy::{Approximate, Fast, Precise};
        let fixed = UintConfigStrategy::Fixed(HybridUintConfig::default());
        let expected = [
            (ClusterStrategy::Best, UintConfigStrategy::Best, Precise),
            (ClusterStrategy::Best, UintConfigStrategy::Best, Precise),
            (ClusterStrategy::Best, UintConfigStrategy::Best, Precise),
            (ClusterStrategy::Best, UintConfigStrategy::Best, Precise),
            (ClusterStrategy::Fast, UintConfigStrategy::Fast, Approximate),
            (ClusterStrategy::Fast, UintConfigStrategy::Fast, Approximate),
            (ClusterStrategy::Fast, UintConfigStrategy::Fast, Approximate),
            (ClusterStrategy::Fast, fixed, Approximate),
            (ClusterStrategy::Fast, fixed, Approximate),
            (ClusterStrategy::Fast, fixed, Approximate),
            (ClusterStrategy::Fast, fixed, Fast),
        ];
        for (tier, (clustering, uint_config, ans)) in (-1i8..=9).zip(expected) {
            let options = EntropyOptions::for_modular(ModularEntropyParams {
                speed_tier: SpeedTier::try_from(tier).unwrap(),
                ..ModularEntropyParams::default()
            });
            assert_eq!(options.clustering, clustering, "tier {tier}");
            assert_eq!(options.uint_config, uint_config, "tier {tier}");
            assert_eq!(options.ans_histogram, ans, "tier {tier}");
        }
    }

    #[test]
    fn precise_entropy_requires_lossless_and_probes_override_last() {
        let mut params = ModularEntropyParams {
            speed_tier: SpeedTier::Lightning,
            precise_entropy: true,
            decoding_speed_tier: 2,
            ..ModularEntropyParams::default()
        };
        let precise = EntropyOptions::for_modular(params);
        assert_eq!(precise.uint_config, UintConfigStrategy::Best);
        assert_eq!(precise.clustering, ClusterStrategy::Best);
        assert_eq!(precise.ans_histogram, AnsHistogramStrategy::Precise);
        assert_eq!(precise.max_histograms, 12);
        params.probe_entropy = ProbeEntropy::FastUintAndHistograms;
        assert_eq!(EntropyOptions::for_modular(params), precise);
        params.disposable_probe = true;
        params.probe_entropy = ProbeEntropy::FastUint;
        let probe = EntropyOptions::for_modular(params);
        assert_eq!(probe.uint_config, UintConfigStrategy::Fast);
        assert_eq!(probe.clustering, ClusterStrategy::Best);
        assert_eq!(probe.ans_histogram, AnsHistogramStrategy::Precise);
        params.probe_entropy = ProbeEntropy::FastUintAndHistograms;
        let probe = EntropyOptions::for_modular(params);
        assert_eq!(probe.clustering, ClusterStrategy::Fast);
        assert_eq!(probe.ans_histogram, AnsHistogramStrategy::Approximate);
        assert_eq!(probe.max_histograms, 12);
        params.lossless = false;
        let lossy = EntropyOptions::for_modular(params);
        assert_eq!(
            lossy.uint_config,
            UintConfigStrategy::Fixed(HybridUintConfig::default())
        );
        assert_eq!(lossy.ans_histogram, AnsHistogramStrategy::Fast);
    }

    #[test]
    fn nonmodular_dc_and_decoding_speed_boundaries() {
        for tier in [
            SpeedTier::Squirrel,
            SpeedTier::Cheetah,
            SpeedTier::Lightning,
        ] {
            let mut params = ModularEntropyParams {
                speed_tier: tier,
                modular_mode: false,
                decoding_speed_tier: 1,
                ..ModularEntropyParams::default()
            };
            let options = EntropyOptions::for_modular(params);
            assert_eq!(
                options.uint_config,
                UintConfigStrategy::Fixed(HybridUintConfig::default())
            );
            assert_eq!(options.max_histograms, 256);
            params.has_extra_dc_precision = true;
            params.decoding_speed_tier = 2;
            let options = EntropyOptions::for_modular(params);
            assert_eq!(options.uint_config, UintConfigStrategy::Fast);
            assert_eq!(options.max_histograms, 12);
        }
    }

    #[test]
    fn policy_validation_and_tier_numbers() {
        assert_eq!(MIN_LOG_ALPHA_SIZE, 5);
        assert_eq!(MAX_LOG_ALPHA_SIZE, 8);
        assert_eq!(SpeedTier::try_from(9u8).unwrap(), SpeedTier::Lightning);
        assert!(SpeedTier::try_from(10u8).is_err());
        assert!(SpeedTier::try_from(255u8).is_err());
        assert!(SpeedTier::try_from(-2i8).is_err());
        let mut options = EntropyOptions::default();
        options.validate().unwrap();
        options.max_histograms = 0;
        assert!(options.validate().is_err());
        options.max_histograms = 257;
        assert!(options.validate().is_err());
        options.max_histograms = 1;
        for split in 0..=15 {
            options.uint_config =
                UintConfigStrategy::Fixed(HybridUintConfig::new(split, 0, 0).unwrap());
            options.validate().unwrap();
        }
        options.min_log_alpha_size = 4;
        assert!(options.validate().is_err());
        options.min_log_alpha_size = 8;
        options.max_log_alpha_size = 7;
        assert!(options.validate().is_err());
        options.max_log_alpha_size = 9;
        assert!(options.validate().is_err());
    }
}
