// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Candidate identity and MakePlan from `enc_lossless_probe.cc`; actual backend
//! decisions from `enc_modular.cc` and `enc_ans.cc::HistogramParams::ForModular`.

pub use crate::modular::sample::TreeMode;
pub use crate::modular::source_options::SourceBuffering;
use crate::{Error, Result, modular::predictor::Predictor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Override {
    Default,
    Off,
    On,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DeltaPalette {
    #[default]
    Disabled,
    ExactGradient,
    ExactLeft,
    ExactTop,
    ZeroGradient,
    ZeroLeft,
    ZeroTop,
}
impl DeltaPalette {
    /// `enc_palette.cc`: JXL_LOSSLESS_DELTA_PALETTE values 1..=6.
    pub fn from_source_setting(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Disabled),
            1 => Ok(Self::ExactGradient),
            2 => Ok(Self::ExactLeft),
            3 => Ok(Self::ExactTop),
            4 => Ok(Self::ZeroGradient),
            5 => Ok(Self::ZeroLeft),
            6 => Ok(Self::ZeroTop),
            _ => Err(Error::InvalidInput(
                "delta palette setting must be 0..=6".into(),
            )),
        }
    }
    pub fn predictor(self) -> Option<Predictor> {
        match self {
            Self::Disabled => None,
            Self::ExactGradient | Self::ZeroGradient => Some(Predictor::Gradient),
            Self::ExactLeft | Self::ZeroLeft => Some(Predictor::Left),
            Self::ExactTop | Self::ZeroTop => Some(Predictor::Top),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LzMethod {
    None,
    Lz77b1w3t,
    Lz77b3w3t,
    Optc8,
    Optc256,
}

/// Source parameters, not a guessed mapping onto `ModularOptions`.
/// None is the C++ auto sentinel for predictor/group/RCT, not "disabled".
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateParams {
    /// `common.h::SpeedTier`: TectonicPlate=-1, Glacier=0, ... Lightning=9.
    /// This is an explicit backend policy. The enc_frame.cc effort11 trial
    /// tournament is a separate orchestration route, not an alias for tier 0.
    pub speed_tier: i8,
    /// Source auto (-1) is distinct from forced full-frame (0). Included in
    /// candidate equality/cache identity, even when a small probe uses neither.
    pub buffering: SourceBuffering,
    /// Source responsive=1: the whole-frame default squeeze script, before
    /// DC/AC partitioning. Does not request progressive passes or local squeeze.
    /// False represents the pixel-lossless auto/default (responsive <=0).
    pub responsive: bool,
    pub group_size_shift: Option<u8>,
    pub rct: Option<u32>,
    pub palette_colors: u32,
    pub channel_colors_percent: f32,
    pub channel_colors_pre_transform_percent: f32,
    pub patches: Override,
    pub predictor: Option<Predictor>,
    /// Source options.wp_tree_mode, independent of the explicit predictor and
    /// weighted-predictor parameter search. Part of candidate/cache identity.
    pub tree_mode: TreeMode,
    pub learning_fraction: f32,
    pub max_properties: u8,
    pub lz77_careful_first_pass: bool,
    pub bucket_matcher: bool,
    pub precise_entropy: bool,
    pub probe_lz77: u8,
    /// Previously process-global extension switches are explicit cache keys.
    pub delta_palette: DeltaPalette,
    /// Explicit local Rust directions, NOT source responsive or its global
    /// default/chroma/range script. Source projection rejects nonempty lists.
    pub squeeze: Vec<bool>,
}
impl Default for CandidateParams {
    fn default() -> Self {
        Self {
            speed_tier: 3,
            buffering: SourceBuffering::Auto,
            responsive: false,
            group_size_shift: None,
            rct: None,
            palette_colors: 1024,
            channel_colors_percent: 80.0,
            channel_colors_pre_transform_percent: 95.0,
            patches: Override::Default,
            predictor: None,
            tree_mode: TreeMode::Default,
            learning_fraction: 0.5,
            max_properties: 0,
            lz77_careful_first_pass: false,
            bucket_matcher: false,
            precise_entropy: false,
            probe_lz77: 0,
            delta_palette: DeltaPalette::Disabled,
            squeeze: Vec::new(),
        }
    }
}
impl CandidateParams {
    pub fn validate(&self) -> Result<()> {
        if !(-1..=9).contains(&self.speed_tier)
            || self.group_size_shift.is_some_and(|v| v > 3)
            || self.rct.is_some_and(|v| v >= 42)
            || self.probe_lz77 > 2
            || !self.learning_fraction.is_finite()
            || self.learning_fraction < 0.0
            || !self.channel_colors_percent.is_finite()
            || self.channel_colors_percent < 0.0
            || !self.channel_colors_pre_transform_percent.is_finite()
            || self.channel_colors_pre_transform_percent < 0.0
        {
            return Err(Error::InvalidInput(
                "invalid lossless candidate parameters".into(),
            ));
        }
        Ok(())
    }
    pub fn backend(&self, effort: u8) -> Result<Self> {
        if !(1..=11).contains(&effort) {
            return Err(Error::InvalidInput("source effort must be 1..=11".into()));
        }
        let mut p = self.clone();
        p.speed_tier = 10 - effort as i8;
        Ok(p)
    }
    /// Every input reaching the Modular backend has bypassed FJXL.
    /// EncodeFrameImpl then falls back from Lightning to Thunder (enc_frame.cc
    /// 2924), including nonresponsive floats and explicit legacy options.
    pub fn modular_speed_tier(&self) -> i8 {
        if self.speed_tier == 9 {
            8
        } else {
            self.speed_tier
        }
    }
    pub fn resolved_predictor(&self) -> Predictor {
        self.predictor.unwrap_or(if self.responsive {
            Predictor::Zero
        } else if self.speed_tier <= 0 {
            Predictor::Variable
        } else if self.speed_tier < 7 {
            Predictor::Best
        } else if self.speed_tier == 7 {
            Predictor::Weighted
        } else {
            Predictor::Gradient
        })
    }
    pub fn lz_method(&self) -> LzMethod {
        if self.probe_lz77 == 1 {
            return LzMethod::Lz77b3w3t;
        }
        if self.probe_lz77 == 2 {
            return LzMethod::Optc8;
        }
        let tier = if self.precise_entropy {
            1
        } else {
            self.modular_speed_tier()
        };
        if self.resolved_predictor() == Predictor::Zero {
            if tier >= 7 {
                LzMethod::None
            } else if tier >= 5 {
                LzMethod::Lz77b1w3t
            } else if tier >= 2 {
                LzMethod::Lz77b3w3t
            } else {
                LzMethod::Optc256
            }
        } else if tier > 2 {
            if tier <= 5 {
                LzMethod::Lz77b1w3t
            } else {
                LzMethod::None
            }
        } else if tier <= 1 {
            LzMethod::Optc256
        } else {
            LzMethod::Lz77b3w3t
        }
    }
    /// Exact source order; transform cost estimation belongs to Modular.
    pub fn automatic_rcts(&self) -> &'static [u32] {
        const RCTS: [u32; 19] = [
            0, 6, 5, 10, 26, 40, 12, 19, 8, 4, 9, 15, 16, 17, 32, 33, 2, 1, 3,
        ];
        let count = match self.speed_tier {
            -1..=1 => 19,
            2 => 9,
            3 => 7,
            4 => 5,
            5 => 4,
            _ => 0,
        };
        &RCTS[..count]
    }
    pub fn weighted_modes(&self) -> u8 {
        if !self.resolved_predictor().has_weighted() {
            1
        } else if self.speed_tier <= 1 {
            5
        } else if self.speed_tier <= 2 {
            2
        } else {
            1
        }
    }
    pub fn training_policy(&self, channels: usize, num_streams: usize) -> TrainingPolicy {
        let speed_tier = self.modular_speed_tier();
        let mut order = if !self.responsive {
            vec![0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8]
        } else {
            vec![0, 1, 4, 5, 6, 7, 8, 15, 9, 10, 11, 12, 13, 14, 2, 3]
        };
        if !self.responsive && num_streams < 30 && speed_tier > 1 {
            order.remove(1);
        }
        // Init explicitly lists Glacier/Tortoise, but NOT TectonicPlate.
        // Tier -1 takes the default 3-property, 32-bin, *0.3 branch.
        let (count, values, scale) = match speed_tier {
            5 => (4, 48, 0.5),
            4 => (5, 64, 0.7),
            3 => (7, 96, 1.0),
            2 => (10, 128, 1.1),
            0 | 1 => (order.len(), 256, 1.3),
            _ => (3, 32, 0.3),
        };
        order.truncate(count);
        let previous_channels = usize::from(self.max_properties).min(channels.saturating_sub(1));
        if speed_tier > 1 {
            order.extend((0..previous_channels).map(|i| 16 + i as u32 * 4 + 3));
        } else {
            order.extend((0..previous_channels * 4).map(|i| 16 + i as u32));
        }
        match self.tree_mode {
            TreeMode::Default => {}
            TreeMode::WeightedOnly => order = vec![15],
            TreeMode::NoWeighted => order.retain(|&p| p != 15),
            TreeMode::GradientOnly => order = vec![9],
        }
        TrainingPolicy {
            fraction: (self.learning_fraction * scale).min(1.0),
            properties: order,
            max_property_values: values,
            node_threshold: u32::try_from(75 + 14 * i32::from(speed_tier))
                .expect("validated source tier has a positive threshold"),
            previous_channels,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingPolicy {
    pub fraction: f32,
    pub properties: Vec<u32>,
    pub max_property_values: u32,
    pub node_threshold: u32,
    pub previous_channels: usize,
}
impl TrainingPolicy {
    /// `enc_encoding.cc::TreeSampleFraction`, per transformed channel.
    pub fn sample_fraction(&self, channel_pixels: usize) -> f64 {
        if self.fraction == 0.0 || channel_pixels == 0 {
            return 0.0;
        }
        f64::from(self.fraction).max((1024.0 / channel_pixels as f64).min(1.0))
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub name: &'static str,
    pub params: CandidateParams,
    pub bytes: Option<usize>,
    pub small_bytes: Option<usize>,
    pub context_bytes: Option<usize>,
    pub exploratory: bool,
}
impl Candidate {
    pub fn new(name: &'static str, params: CandidateParams) -> Self {
        Self {
            name,
            params,
            bytes: None,
            small_bytes: None,
            context_bytes: None,
            exploratory: false,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanOptions {
    /// C++ JXL_LOSSLESS_RCT_SCREEN=1.
    pub extended_rct: bool,
}

pub fn lossless_probe_plan(input: &CandidateParams, options: PlanOptions) -> Vec<Candidate> {
    let mut p = input.clone();
    p.speed_tier = 1;
    p.learning_fraction = 0.1;
    p.patches = Override::Off;
    p.group_size_shift = Some(3);
    p.predictor = Some(Predictor::Zero);
    p.channel_colors_percent = 0.0;
    p.palette_colors = 70000;
    let mut result = vec![Candidate::new("large-palette-zero", p.clone())];
    p.predictor = Some(Predictor::Gradient);
    result.push(Candidate::new("large-palette-gradient", p.clone()));
    p.palette_colors = 0;
    p.channel_colors_pre_transform_percent = 0.0;
    for (rct, names) in [
        (0, ["raw-zero", "raw-gradient", "raw-weighted"]),
        (6, ["ycocg-zero", "ycocg-gradient", "ycocg-weighted"]),
    ] {
        p.rct = Some(rct);
        for (predictor, name) in [Predictor::Zero, Predictor::Gradient, Predictor::Weighted]
            .into_iter()
            .zip(names)
        {
            p.predictor = Some(predictor);
            result.push(Candidate::new(name, p.clone()));
        }
    }
    p.predictor = Some(Predictor::Variable);
    p.max_properties = 2;
    for (rct, name) in [(0, "raw-cross-channel"), (6, "ycocg-cross-channel")] {
        p.rct = Some(rct);
        result.push(Candidate::new(name, p.clone()));
    }
    if options.extended_rct {
        for (rct, name) in [
            (10, "rct10-cross-channel"),
            (17, "rct17-cross-channel"),
            (26, "rct26-cross-channel"),
        ] {
            p.rct = Some(rct);
            result.push(Candidate::new(name, p.clone()));
        }
    }
    p.max_properties = 0;
    for (rct, names) in [
        (0, ["raw-select", "raw-left"]),
        (6, ["ycocg-select", "ycocg-left"]),
    ] {
        p.rct = Some(rct);
        for (predictor, name) in [Predictor::Select, Predictor::Left].into_iter().zip(names) {
            p.predictor = Some(predictor);
            result.push(Candidate::new(name, p.clone()));
        }
    }
    p.rct = Some(0);
    p.predictor = Some(Predictor::Top);
    result.push(Candidate::new("raw-top", p));
    result
}

#[cfg(test)]
mod tectonic_tests {
    use super::*;

    #[test]
    fn effort_eleven_is_negative_one_not_glacier() {
        let params = CandidateParams::default().backend(11).unwrap();
        let glacier = params.backend(10).unwrap();
        assert_eq!(params.speed_tier, -1);
        assert_eq!(params.modular_speed_tier(), -1);
        assert_eq!(glacier.speed_tier, 0);
        assert_ne!(params, glacier);
        assert!(params.validate().is_ok());
        assert_eq!(params.resolved_predictor(), Predictor::Variable);
        assert_eq!(params.automatic_rcts(), glacier.automatic_rcts());
        assert_eq!(params.automatic_rcts().len(), 19);
        assert_eq!(params.weighted_modes(), 5);
        assert_eq!(params.lz_method(), LzMethod::Optc256);

        let policy = params.training_policy(3, 22);
        assert_eq!(policy.properties, vec![0, 1, 15]);
        assert_eq!(policy.max_property_values, 32);
        assert_eq!(policy.node_threshold, 61);
        assert_eq!(policy.fraction, 0.5f32 * 0.3f32);
        let glacier_policy = glacier.training_policy(3, 22);
        assert_eq!(glacier_policy.properties.len(), 16);
        assert_eq!(glacier_policy.max_property_values, 256);
        assert_eq!(glacier_policy.node_threshold, 75);
        assert_eq!(glacier_policy.fraction, 0.5f32 * 1.3f32);
    }

    #[test]
    fn tectonic_responsive_and_explicit_overrides_keep_source_precedence() {
        let mut params = CandidateParams {
            speed_tier: -1,
            responsive: true,
            max_properties: 2,
            ..Default::default()
        };
        assert_eq!(params.resolved_predictor(), Predictor::Zero);
        assert_eq!(params.modular_speed_tier(), -1);
        let policy = params.training_policy(3, 22);
        assert_eq!(&policy.properties[..3], &[0, 1, 4]);
        assert_eq!(&policy.properties[3..], &(16..24).collect::<Vec<_>>());
        assert_eq!(policy.node_threshold, 61);
        params.predictor = Some(Predictor::Weighted);
        assert_eq!(params.resolved_predictor(), Predictor::Weighted);
        assert_eq!(params.weighted_modes(), 5);
        params.precise_entropy = true;
        assert_eq!(params.lz_method(), LzMethod::Optc256);
        params.probe_lz77 = 1;
        assert_eq!(params.lz_method(), LzMethod::Lz77b3w3t);
        params.probe_lz77 = 2;
        assert_eq!(params.lz_method(), LzMethod::Optc8);
    }

    #[test]
    fn signed_tier_and_effort_boundaries_are_validated_before_projection() {
        for speed_tier in [i8::MIN, -2, 10, i8::MAX] {
            assert!(
                CandidateParams {
                    speed_tier,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        for effort in [0, 12, u8::MAX] {
            assert!(CandidateParams::default().backend(effort).is_err());
        }
        for effort in 1..=11 {
            let params = CandidateParams::default().backend(effort).unwrap();
            assert_eq!(10 - i16::from(params.speed_tier), i16::from(effort));
        }
    }
}
