// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Lossless palette policy from libjxl b866b3d:
//! - enc_modular.cc::try_palettes / maybe_do_transform and their global/local
//!   eligibility gates; enc_params.h defaults (1024, 95%, 80%, speed tier 3).
//! - enc_palette.cc::FwdPalette orchestration, through apply_palette_range_with_options
//!   and the existing transforms.rs sample kernels, including exact-zero deltas.
//! - Cost is cost.rs::estimate_cost, the port of enc_modular_simd.cc::EstimateCost,
//!   NOT encoded-byte trials, distinct-color ratios, or replacement entropy.
//!
//! Preserved details: all/all-minus-one/single order; signed palette limit picks
//! image order; stage-specific speed inequalities; strictly-greater cost rejection;
//! original per-channel cost baseline; latest meta-prefix offset after insertion;
//! source f32/f64 arithmetic and maxval/CeilLog2(max) bookkeeping. Source uses
//! channel[0]'s size for the pixel-count heuristic even if it is a previous table.
//!
//! Explicit settings replace environment reads. This is the integer lossless
//! branch only: no lossy/JXL-art quantization, worker/audit counters or replay.
//! Optional ordering/mapping tools are selected by PaletteOptions::tools. Default luma
//! tie order and >24-bit multichannel limitations remain those documented in
//! transforms.rs. Cost ISA rounding remains documented in cost.rs. Overflowing
//! source int conversions are errors, never saturating threshold substitutes.

#![forbid(unsafe_code)]

use super::cost::{self, ChannelView};
use super::group::{self, Channel, Transform};
use super::predictor::Predictor;
use super::transforms;
pub use super::transforms::{PaletteGeometry, PaletteMapping, PaletteOrder, PaletteTools};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PaletteMode {
    #[default]
    Absolute,
    /// FwdExactDeltaPalette first; ordinary palette if it declines.
    ExactDelta(Predictor),
    /// Ordinary palette plus AddExactZeroDelta from an original-sample mask.
    ExactZero(Predictor),
}

impl PaletteMode {
    pub fn validate(self) -> Result<()> {
        if let Self::ExactDelta(p) | Self::ExactZero(p) = self
            && (p.has_weighted() || (p as u32) >= 14)
        {
            return Err(Error::InvalidInput(
                "palette deltas require a concrete stateless predictor".into(),
            ));
        }
        Ok(())
    }

    /// Exact mapping of JXL_LOSSLESS_DELTA_PALETTE modes, without reading env.
    pub fn from_source_mode(mode: u8) -> Result<Self> {
        match mode {
            0 => Ok(Self::Absolute),
            1 => Ok(Self::ExactDelta(Predictor::Gradient)),
            2 => Ok(Self::ExactDelta(Predictor::Left)),
            3 => Ok(Self::ExactDelta(Predictor::Top)),
            4 => Ok(Self::ExactZero(Predictor::Gradient)),
            5 => Ok(Self::ExactZero(Predictor::Left)),
            6 => Ok(Self::ExactZero(Predictor::Top)),
            _ => Err(Error::InvalidInput(
                "source delta-palette mode must be 0..6".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaletteStage {
    /// Global palette gate from ComputeEncodingData, normally before RCT.
    PreRct,
    /// Local palette gate from PrepareStreamParams, after global transforms.
    /// The parent may still search/apply a local RCT after this stage. Source
    /// calls this only for nonzero streams, not a second time on global stream 0.
    PostRct,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaletteOptions {
    /// 0 disables multichannel palettes, NOT single-channel compaction.
    /// Negative selects image order instead of the default luma/frequency order.
    pub palette_colors: i32,
    pub channel_colors_pre_transform_percent: f32,
    pub channel_colors_percent: f32,
    /// Numeric libjxl SpeedTier, -1..9; 3 is the source default.
    pub speed_tier: i8,
    pub responsive: bool,
    /// Global palettes are skipped by the source streaming/groupwise route.
    /// encode_layout treats true as an explicit buffering=2 request, subject to
    /// source eligibility; rejected requests use full-frame transforms instead.
    pub groupwise: bool,
    pub do_color: bool,
    /// Source force-palette switch; bypasses multi-channel cap/cost rejection,
    /// but does NOT force the single-channel compaction phase.
    pub force_palette: bool,
    /// Source zero_tokens bypass of multi-channel cost rejection only.
    pub zero_tokens: bool,
    /// Source direct-audit branch: one positive-limit all-channel attempt only.
    pub direct: bool,
    pub mode: PaletteMode,
    /// Optional frozen-source experiments; defaults retain ordinary ordering.
    pub tools: PaletteTools,
}

impl Default for PaletteOptions {
    fn default() -> Self {
        Self {
            palette_colors: 1024,
            channel_colors_pre_transform_percent: 95.0,
            channel_colors_percent: 80.0,
            speed_tier: 3,
            responsive: false,
            groupwise: false,
            do_color: true,
            force_palette: false,
            zero_tokens: false,
            direct: false,
            mode: PaletteMode::Absolute,
            tools: PaletteTools::default(),
        }
    }
}

impl PaletteOptions {
    pub fn validate(&self) -> Result<()> {
        self.mode.validate()?;
        if !(-1..=9).contains(&self.speed_tier) || self.palette_colors.unsigned_abs() > 70911 {
            return Err(Error::InvalidInput(
                "palette speed tier or color limit exceeds supported range".into(),
            ));
        }
        for percent in [
            self.channel_colors_percent,
            self.channel_colors_pre_transform_percent,
        ] {
            if !percent.is_finite() || !(0.0..=100.0).contains(&percent) {
                return Err(Error::InvalidInput(
                    "channel palette percentage must be finite and 0..100".into(),
                ));
            }
        }
        if self.force_palette && self.palette_colors <= 0 {
            return Err(Error::InvalidInput(
                "forced palette requires a positive limit".into(),
            ));
        }
        Ok(())
    }
}

/// Source nb_meta_channels, max_bitdepth and maxval carried across stages/RCT.
/// This is transform-policy bookkeeping, NOT ImageMetadata's sample depth.
/// The parent updates max_bitdepth for an intervening RCT as the source does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PaletteState {
    pub meta_channels: usize,
    pub max_bitdepth: i32,
    pub max_value: i32,
}

impl PaletteState {
    /// Integer color samples include the source's implicit sign bit. Before
    /// global transforms, raise max_bitdepth to max(extra_bits + 1) as needed.
    /// Source local streams instead initialize the unused depth/value to zero.
    pub fn new(bits: u8) -> Result<Self> {
        if !(1..=31).contains(&bits) {
            return Err(Error::InvalidInput("palette depth must be 1..31".into()));
        }
        Ok(Self {
            meta_channels: 0,
            max_bitdepth: i32::from(bits) + 1,
            max_value: ((1u32 << bits) - 1) as i32,
        })
    }
}

/// FwdPalette with explicit mapping/ordering settings and actual channel
/// geometry. Errors/declines leave inputs unchanged. Exact-delta success keeps
/// its own residual-frequency order; only its absolute fallback uses tools.
/// The legacy group entry point delegates here with default tools.
#[allow(clippy::too_many_arguments)]
pub fn apply_palette_range_with_options(
    channels: &mut Vec<Channel>,
    meta: &mut usize,
    bits: u8,
    begin: usize,
    count: usize,
    max_colors: u32,
    ordered: bool,
    mode: PaletteMode,
    tools: PaletteTools,
) -> Result<Option<Transform>> {
    mode.validate()?;
    if !(1..=32).contains(&bits) {
        return Err(Error::InvalidInput(
            "palette bit depth must be 1..32".into(),
        ));
    }
    group::validate_layout(channels, *meta)?;
    let end = begin
        .checked_add(count)
        .filter(|&end| count != 0 && begin >= *meta && end <= channels.len())
        .ok_or_else(|| {
            Error::InvalidInput("transform must select an existing non-meta channel range".into())
        })?;
    let first = &channels[begin];
    let (width, height, hshift, vshift) = (first.width, first.height, first.hshift, first.vshift);
    if width == 0
        || height == 0
        || channels[begin..end]
            .iter()
            .any(|c| (c.width, c.height, c.hshift, c.vshift) != (width, height, hshift, vshift))
    {
        return Ok(None);
    }
    let planes: Vec<_> = channels[begin..end]
        .iter()
        .map(|c| c.samples.clone())
        .collect();
    let mut output = if let PaletteMode::ExactDelta(p) = mode {
        transforms::delta_palette_forward(
            &planes,
            0,
            count,
            width,
            height,
            max_colors,
            u32::from(bits),
            p,
        )?
    } else {
        None
    };
    if output.is_none() {
        let mask = if let PaletteMode::ExactZero(p) = mode {
            transforms::exact_zero_delta_mask(&planes, 0, count, width, height, u32::from(bits), p)?
        } else {
            None
        };
        let Some(absolute) = transforms::palette_forward_with_options(
            &planes,
            0,
            count,
            width,
            height,
            max_colors,
            u32::from(bits),
            ordered,
            tools,
        )?
        else {
            return Ok(None);
        };
        output = Some(
            if let (PaletteMode::ExactZero(p), Some(mask)) = (mode, mask) {
                transforms::add_exact_zero_delta(&absolute, p, &mask)?
            } else {
                transforms::DeltaPaletteOutput {
                    num_colors: absolute.palette[0].len() as u32,
                    num_deltas: 0,
                    predictor: Predictor::Zero,
                    palette: absolute.palette,
                    indices: absolute.indices,
                }
            },
        );
    }
    let palette = output.expect("absolute or exact palette assigned");
    let transform = Transform::Palette {
        begin,
        channels: count,
        colors: palette.num_colors as usize,
        deltas: palette.num_deltas as usize,
        predictor: palette.predictor,
    };
    group::validate_transform(&transform)?;
    if (channels.len() - count)
        .checked_add(2)
        .is_none_or(|n| n > i32::MAX as usize)
    {
        return Err(Error::Limit("palette channel count"));
    }
    let table = Channel {
        width: palette.palette[0].len(),
        height: count,
        hshift: -1,
        vshift: -1,
        samples: palette.palette.into_iter().flatten().collect(),
    };
    channels
        .try_reserve(1)
        .map_err(|_| Error::Limit("palette channel insertion"))?;
    channels[begin] = Channel {
        width,
        height,
        hshift,
        vshift,
        samples: palette.indices,
    };
    channels.drain(begin + 1..end);
    channels.insert(0, table);
    *meta += 1;
    Ok(Some(transform))
}

/// Parent application helper. All accepted transforms are returned in wire order.
/// On error, channels and state are unchanged; a policy decline is Ok(empty).
/// `bits` is Modular Image::bitdepth (the main image's original sample depth),
/// NOT max_bitdepth or the deepest extra channel; it stays unchanged after RCT.
/// A 32-bit float image is accepted but declines palettes, as in the source.
/// Even an empty script may update source depth/value bookkeeping.
pub fn apply_palettes(
    channels: &mut Vec<Channel>,
    state: &mut PaletteState,
    bits: u8,
    stage: PaletteStage,
    options: &PaletteOptions,
) -> Result<Vec<Transform>> {
    options.validate()?;
    if !(1..=32).contains(&bits) || !(0..=32).contains(&state.max_bitdepth) {
        return Err(Error::InvalidInput(
            "invalid palette input/effective bit depth".into(),
        ));
    }
    group::validate_layout(channels, state.meta_channels)?;
    // FwdPaletteIteration declines all 32-bit inputs before inspecting samples.
    // Avoid cost/range arithmetic for a transform that cannot be applied.
    if bits == 32 {
        return Ok(Vec::new());
    }
    if channels.len() == state.meta_channels {
        return Ok(Vec::new());
    }
    let percent = match stage {
        PaletteStage::PreRct => {
            if options.groupwise || options.responsive {
                return Ok(Vec::new());
            }
            if options.speed_tier <= 8 || (options.do_color && bits > 8) {
                options.channel_colors_pre_transform_percent
            } else {
                0.0
            }
        }
        PaletteStage::PostRct => {
            if options.responsive || options.speed_tier >= 6 {
                return Ok(Vec::new());
            }
            options.channel_colors_percent
        }
    };
    let mut candidate = channels.clone();
    let mut next = *state;
    let transforms = try_palettes(&mut candidate, &mut next, bits, percent, options)?;
    *channels = candidate;
    *state = next;
    Ok(transforms)
}

fn estimate(channels: &[Channel]) -> Result<f32> {
    let views: Vec<_> = channels
        .iter()
        .map(|c| ChannelView {
            pixels: &c.samples,
            width: c.width,
            height: c.height,
            stride: c.width,
        })
        .collect();
    cost::estimate_cost(&views)
}

// Source casts to int (towards zero). Reject undefined out-of-int-range casts.
fn source_int(value: f64) -> Result<u32> {
    if !value.is_finite() || !(0.0..2147483648.0).contains(&value) {
        return Err(Error::Limit(
            "palette heuristic exceeds source signed int range",
        ));
    }
    Ok(value as u32)
}

fn multichannel_limit(cost: f32, pixels: f32, options: &PaletteOptions) -> Result<u32> {
    if options.force_palette {
        return Ok(options.palette_colors as u32);
    }
    Ok(source_int(f64::from(
        cost * 0.0005_f32 + pixels / 128.0_f32 + 128.0_f32,
    ))?
    .min(options.palette_colors.unsigned_abs()))
}

fn channel_limit(pixels: f32, min: i32, max: i32, percent: f32) -> Result<u32> {
    let range = i64::from(max) - i64::from(min) + 1;
    Ok(source_int(f64::from(pixels / 16.0_f32))?
        .min(source_int(f64::from(percent) / 100.0 * range as f64)?))
}

fn minmax(channel: &Channel) -> Option<(i32, i32)> {
    let &first = channel.samples.first()?;
    Some(
        channel
            .samples
            .iter()
            .fold((first, first), |(min, max), &x| (min.min(x), max.max(x))),
    )
}

#[allow(clippy::too_many_arguments)]
fn attempt(
    channels: &mut Vec<Channel>,
    state: &mut PaletteState,
    transforms: &mut Vec<Transform>,
    bits: u8,
    begin: usize,
    count: usize,
    limit: u32,
    ordered: bool,
    mode: PaletteMode,
    tools: PaletteTools,
    cost_before: f32,
    accept_without_cost: bool,
) -> Result<bool> {
    let mut candidate = channels.clone();
    let mut meta = state.meta_channels;
    let Some(transform) = apply_palette_range_with_options(
        &mut candidate,
        &mut meta,
        bits,
        begin,
        count,
        limit,
        ordered,
        mode,
        tools,
    )?
    else {
        return Ok(false);
    };
    if !accept_without_cost && estimate(&candidate)? > cost_before {
        return Ok(false);
    }
    if transforms.len() == 273 {
        return Err(Error::Limit("palette transform count"));
    }
    transforms.push(transform);
    *channels = candidate;
    state.meta_channels = meta;
    Ok(true)
}

fn try_palettes(
    channels: &mut Vec<Channel>,
    state: &mut PaletteState,
    bits: u8,
    percent: f32,
    options: &PaletteOptions,
) -> Result<Vec<Transform>> {
    let mut transforms = Vec::new();
    let count = channels.len() - state.meta_channels;
    if options.direct {
        if options.palette_colors > 0 {
            let begin = state.meta_channels;
            attempt(
                channels,
                state,
                &mut transforms,
                bits,
                begin,
                count,
                options.palette_colors as u32,
                true,
                options.mode,
                options.tools,
                0.0,
                true,
            )?;
        }
        return Ok(transforms);
    }
    let pixels = channels[0].samples.len() as f32;
    let mut cost_before = 0.0;
    let mut did_palette = false;
    if options.palette_colors != 0 {
        cost_before = if options.speed_tier <= 3 {
            estimate(channels)?
        } else {
            pixels * (0.2_f32 * f32::from(bits) * count as f32)
        };
        let ordered = options.palette_colors >= 0;
        let bypass = options.force_palette || options.zero_tokens || options.speed_tier >= 3;
        let begin = state.meta_channels;
        if count > 1 {
            let limit = multichannel_limit(cost_before, pixels, options)?;
            did_palette = attempt(
                channels,
                state,
                &mut transforms,
                bits,
                begin,
                count,
                limit,
                ordered,
                options.mode,
                options.tools,
                cost_before,
                bypass,
            )?;
        }
        if !did_palette && count > 3 {
            let limit = multichannel_limit(cost_before, pixels, options)?;
            did_palette = attempt(
                channels,
                state,
                &mut transforms,
                bits,
                begin,
                count - 1,
                limit,
                ordered,
                options.mode,
                options.tools,
                cost_before,
                bypass,
            )?;
        }
    }
    if percent > 0.0 {
        let first = usize::from(did_palette); // skip the new tuple-index channel
        let remaining = channels.len() - state.meta_channels - first;
        let original_depth = state.max_bitdepth;
        state.max_bitdepth = 0;
        if remaining > 0 && (did_palette || cost_before == 0.0) {
            cost_before = if options.speed_tier < 3 {
                estimate(channels)?
            } else {
                0.0
            };
        }
        for i in first..remaining + first {
            // Each accepted palette inserts a new table at zero. Recompute the
            // physical index from the logical non-meta index on every iteration.
            let begin = state.meta_channels + i;
            let Some((min, max)) = minmax(&channels[begin]) else {
                state.max_bitdepth = original_depth;
                continue;
            };
            let limit = channel_limit(pixels, min, max, percent)?;
            if attempt(
                channels,
                state,
                &mut transforms,
                bits,
                begin,
                1,
                limit,
                true,
                options.mode,
                options.tools,
                cost_before,
                options.speed_tier >= 3,
            )? {
                let (_, max) =
                    minmax(&channels[state.meta_channels + i]).expect("nonempty accepted indices");
                if max < state.max_value {
                    state.max_value = max;
                }
                let depth = if max > 0 {
                    (32 - ((max as u32) - 1).leading_zeros()) as i32
                } else {
                    0
                };
                state.max_bitdepth = state.max_bitdepth.max(depth);
            } else {
                state.max_bitdepth = original_depth;
            }
        }
    }
    Ok(transforms)
}

#[cfg(test)]
mod tests {
    use super::super::group::undo_palette_for_test;
    use super::*;

    fn channel(pattern: &[i32], pixels: usize, width: usize) -> Channel {
        assert_eq!(pixels % width, 0);
        Channel {
            width,
            height: pixels / width,
            hshift: 0,
            vshift: 0,
            samples: pattern.iter().copied().cycle().take(pixels).collect(),
        }
    }

    #[test]
    fn explicit_tools_reach_policy_and_direct_range_entries() {
        let original = vec![channel(&[120, 120, 10, 10, 90, 90], 24, 6); 3];
        let tools = PaletteTools {
            order: PaletteOrder::Morton,
            mapping: PaletteMapping::Flat,
            ..PaletteTools::default()
        };
        for stage in [PaletteStage::PreRct, PaletteStage::PostRct] {
            for direct in [false, true] {
                let mut channels = original.clone();
                let mut state = PaletteState::new(8).unwrap();
                let options = PaletteOptions {
                    palette_colors: 8,
                    direct,
                    tools,
                    channel_colors_pre_transform_percent: 0.0,
                    channel_colors_percent: 0.0,
                    ..PaletteOptions::default()
                };
                let script = apply_palettes(&mut channels, &mut state, 8, stage, &options).unwrap();
                assert_eq!(script.len(), 1);
                assert_eq!(state.meta_channels, 1);
                assert_eq!(&channels[0].samples[..3], [10, 90, 120]);
                undo_palette_for_test(&mut channels, script[0].clone(), 8);
                assert_eq!(channels, original);
            }
        }
        for (mode, bits, deltas) in [
            (PaletteMode::Absolute, 8, 0),
            (PaletteMode::ExactZero(Predictor::Left), 8, 1),
            (PaletteMode::ExactDelta(Predictor::Left), 24, 0),
        ] {
            let mut channels = original.clone();
            let mut meta = 0;
            let transform = apply_palette_range_with_options(
                &mut channels,
                &mut meta,
                bits,
                0,
                3,
                8,
                true,
                mode,
                tools,
            )
            .unwrap()
            .unwrap();
            assert!(matches!(transform, Transform::Palette { deltas: d, .. } if d == deltas));
            assert_eq!(channels[0].samples[deltas], 10);
            undo_palette_for_test(&mut channels, transform, bits);
            assert_eq!(channels, original);
        }
        // Successful exact residual palettes ignore ordinary palette order.
        let mut default = original.clone();
        let mut selected = original.clone();
        let (mut dm, mut sm) = (0, 0);
        let dt = apply_palette_range_with_options(
            &mut default,
            &mut dm,
            8,
            0,
            3,
            8,
            true,
            PaletteMode::ExactDelta(Predictor::Left),
            PaletteTools::default(),
        )
        .unwrap()
        .unwrap();
        let st = apply_palette_range_with_options(
            &mut selected,
            &mut sm,
            8,
            0,
            3,
            8,
            true,
            PaletteMode::ExactDelta(Predictor::Left),
            tools,
        )
        .unwrap()
        .unwrap();
        assert!(matches!(dt, Transform::Palette { deltas, .. } if deltas > 0));
        assert_eq!(dt, st);
        assert_eq!(default, selected);
        undo_palette_for_test(&mut selected, st, 8);
        assert_eq!(selected, original);
    }

    #[test]
    fn optional_range_preserves_meta_prefix_and_rolls_back_declines() {
        let sentinel = Channel {
            width: 1,
            height: 1,
            hshift: -1,
            vshift: -1,
            samples: vec![777],
        };
        let mut original = vec![sentinel, channel(&[555], 4, 2)];
        original.extend(vec![channel(&[10, 20, 30, 40], 4, 2); 3]);
        original.push(channel(&[0], 4, 2));
        let tools = PaletteTools {
            order: PaletteOrder::Adjacency,
            mapping: PaletteMapping::Flat,
            ..PaletteTools::default()
        };
        let mut channels = original.clone();
        let mut meta = 1;
        let transform = apply_palette_range_with_options(
            &mut channels,
            &mut meta,
            8,
            2,
            3,
            4,
            true,
            PaletteMode::Absolute,
            tools,
        )
        .unwrap()
        .unwrap();
        assert_eq!(meta, 2);
        assert_eq!(&channels[0].samples[..4], [30, 10, 20, 40]);
        assert_eq!(channels[1], original[0]);
        assert_eq!(channels[2], original[1]);
        assert_eq!(channels[4], original[5]);
        undo_palette_for_test(&mut channels, transform, 8);
        assert_eq!(channels, original);
        for mutation in 0..4 {
            let mut channels = original.clone();
            match mutation {
                1 => {
                    channels[3].width = 4;
                    channels[3].height = 1;
                }
                2 => channels[3].hshift = 1,
                3 => {
                    channels[3].samples.pop();
                }
                _ => (),
            }
            let before = channels.clone();
            let mut meta = 1;
            let outcome = apply_palette_range_with_options(
                &mut channels,
                &mut meta,
                8,
                2,
                3,
                2,
                true,
                PaletteMode::Absolute,
                tools,
            );
            if mutation == 3 {
                assert!(outcome.is_err());
            } else {
                assert!(outcome.unwrap().is_none());
            }
            assert_eq!(channels, before);
            assert_eq!(meta, 1);
        }
    }

    fn singles() -> PaletteOptions {
        PaletteOptions {
            palette_colors: 0,
            ..PaletteOptions::default()
        }
    }

    // Deterministic U8/U16 oracle inputs, also specified in PALETTE_PORT_STATUS.
    // This matrix exercises policy plumbing; the parent owns official decoding.
    fn palette_oracle_fixture(bits: u8, count: usize) -> Vec<Channel> {
        let colors = [
            [0, 0, 0, 0],
            [255, 255, 255, 255],
            [1, 2, 3, 0],
            [7, 11, 13, 0],
            [7, 11, 13, 255],
            [90, 10, 120, 127],
            [32, 32, 32, 0],
            [254, 1, 200, 255],
        ];
        let scale = if bits == 16 { 257 } else { 1 };
        (0..count)
            .map(|c| Channel {
                width: 32,
                height: 16,
                hshift: 0,
                vshift: 0,
                samples: (0..512)
                    .map(|pos| {
                        let slot = (pos + (pos / 32) * 5) % 64;
                        let id = if pos < 8 {
                            pos
                        } else {
                            match slot {
                                0..=31 => 0,
                                32..=47 => 2,
                                48..=55 => 3,
                                56..=59 => 4,
                                60..=61 => 5,
                                62 => 6,
                                _ => 1,
                            }
                        };
                        colors[id][c] * scale
                    })
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn all_palette_tools_policy_matrix_matches_direct_ranges_and_expands_samples() {
        let selectors: Vec<_> = (0..=9)
            .map(|order| (order, 0))
            .chain([(9, 1), (9, 2)])
            .collect();
        for bits in [8, 16] {
            for count in [3, 4] {
                let original = palette_oracle_fixture(bits, count);
                for &(source_order, source_geometry) in &selectors {
                    for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
                        let tools = PaletteTools {
                            order: PaletteOrder::from_source_mode(source_order).unwrap(),
                            geometry: PaletteGeometry::from_source_mode(source_geometry).unwrap(),
                            mapping,
                        };
                        for mode in [
                            PaletteMode::Absolute,
                            PaletteMode::ExactZero(Predictor::Left),
                            PaletteMode::ExactDelta(Predictor::Left),
                        ] {
                            let mut expected = original.clone();
                            let mut expected_meta = 0;
                            let transform = apply_palette_range_with_options(
                                &mut expected,
                                &mut expected_meta,
                                bits,
                                0,
                                count,
                                16,
                                true,
                                mode,
                                tools,
                            )
                            .unwrap()
                            .unwrap();
                            for stage in [PaletteStage::PreRct, PaletteStage::PostRct] {
                                for direct in [false, true] {
                                    let options = PaletteOptions {
                                        palette_colors: 16,
                                        direct,
                                        mode,
                                        tools,
                                        channel_colors_pre_transform_percent: 0.0,
                                        channel_colors_percent: 0.0,
                                        ..PaletteOptions::default()
                                    };
                                    let mut channels = original.clone();
                                    let mut state = PaletteState::new(bits).unwrap();
                                    let script = apply_palettes(
                                        &mut channels,
                                        &mut state,
                                        bits,
                                        stage,
                                        &options,
                                    )
                                    .unwrap();
                                    assert_eq!(
                                        script,
                                        std::slice::from_ref(&transform),
                                        "bits={bits} channels={count} order={source_order} geometry={source_geometry} map={mapping:?} mode={mode:?} stage={stage:?} direct={direct}"
                                    );
                                    assert_eq!(channels, expected);
                                    assert_eq!(state.meta_channels, expected_meta);
                                    undo_palette_for_test(&mut channels, script[0].clone(), bits);
                                    assert_eq!(channels, original);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply(
        channels: &mut Vec<Channel>,
        bits: u8,
        options: &PaletteOptions,
    ) -> (PaletteState, Vec<Transform>) {
        let mut state = PaletteState::new(bits).unwrap();
        let transforms =
            apply_palettes(channels, &mut state, bits, PaletteStage::PreRct, options).unwrap();
        (state, transforms)
    }

    #[test]
    fn source_32bit_palette_decline_keeps_signed_float_words_and_state() {
        let original = vec![channel(&[i32::MIN, i32::MAX, 0], 32, 8)];
        for stage in [PaletteStage::PreRct, PaletteStage::PostRct] {
            let mut channels = original.clone();
            let mut state = PaletteState {
                meta_channels: 0,
                max_bitdepth: 32,
                max_value: 0,
            };
            let before = state;
            assert!(
                apply_palettes(
                    &mut channels,
                    &mut state,
                    32,
                    stage,
                    &PaletteOptions::default()
                )
                .unwrap()
                .is_empty()
            );
            assert_eq!(channels, original);
            assert_eq!(state, before);
        }
    }

    #[test]
    fn source_threshold_casts_limits_and_mode_mapping() {
        assert_eq!(PaletteState::new(8).unwrap().max_bitdepth, 9);
        assert_eq!(PaletteState::new(31).unwrap().max_bitdepth, 32);
        let mut options = PaletteOptions::default();
        assert_eq!(multichannel_limit(0.0, 0.0, &options).unwrap(), 128);
        assert_eq!(multichannel_limit(2000.0, 128.0, &options).unwrap(), 130);
        options.palette_colors = -129;
        assert_eq!(multichannel_limit(2000.0, 128.0, &options).unwrap(), 129);
        options.palette_colors = 65536;
        options.force_palette = true;
        assert_eq!(multichannel_limit(0.0, 0.0, &options).unwrap(), 65536);
        assert_eq!(channel_limit(31.0, -10, 10, 95.0).unwrap(), 1);
        assert_eq!(channel_limit(32.0, -10, 10, 95.0).unwrap(), 2);
        assert_eq!(channel_limit(128.0, -3, 1, 59.0).unwrap(), 2);
        assert_eq!(channel_limit(128.0, -3, 1, 60.0).unwrap(), 3);
        assert_eq!(channel_limit(128.0, 7, 7, 100.0).unwrap(), 1);
        assert!(channel_limit(32.0, i32::MIN, i32::MAX, 100.0).is_err());
        assert!(source_int(2147483648.0).is_err());
        assert_eq!(source_int(2147483647.99).unwrap(), i32::MAX as u32);
        for (i, expected) in [
            PaletteMode::Absolute,
            PaletteMode::ExactDelta(Predictor::Gradient),
            PaletteMode::ExactDelta(Predictor::Left),
            PaletteMode::ExactDelta(Predictor::Top),
            PaletteMode::ExactZero(Predictor::Gradient),
            PaletteMode::ExactZero(Predictor::Left),
            PaletteMode::ExactZero(Predictor::Top),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(PaletteMode::from_source_mode(i as u8).unwrap(), expected);
        }
        assert!(PaletteMode::from_source_mode(7).is_err());
    }

    #[test]
    fn single_channel_pixel_and_range_thresholds_are_source_gates() {
        for pixels in [31, 32] {
            let original = vec![channel(&[-9, 40], pixels, pixels)];
            let mut channels = original.clone();
            let (state, transforms) = apply(&mut channels, 8, &singles());
            assert_eq!(transforms.len(), usize::from(pixels == 32));
            if pixels == 32 {
                assert_eq!(state.meta_channels, 1);
                assert_eq!(state.max_value, 1);
                assert_eq!(state.max_bitdepth, 0); // source uses CeilLog2(max), not max+1
                assert_eq!(channels[0].samples, [-9, 40]);
                undo_palette_for_test(&mut channels, transforms[0].clone(), 8);
            }
            assert_eq!(channels, original);
        }
        for pattern in [&[0, 1][..], &[7][..]] {
            let original = vec![channel(pattern, 32, 8)];
            let mut channels = original.clone();
            assert!(apply(&mut channels, 8, &singles()).1.is_empty());
            assert_eq!(channels, original); // range percentage truncates to <=1
        }
    }

    #[test]
    fn single_channel_lut_and_tree_cutover_keep_policy_layout_and_dictionary() {
        let options = singles();
        for span in [65536, 65537] {
            for min in [i32::MIN, -32769, i32::MAX - (span - 1)] {
                let max = i32::try_from(i64::from(min) + i64::from(span) - 1).unwrap();
                let original = vec![channel(&[min + 100, max, min, min + 9, min], 112, 7)];
                for stage in [PaletteStage::PreRct, PaletteStage::PostRct] {
                    let mut channels = original.clone();
                    let mut state = PaletteState::new(31).unwrap();
                    let script =
                        apply_palettes(&mut channels, &mut state, 31, stage, &options).unwrap();
                    assert_eq!(
                        script,
                        [Transform::Palette {
                            begin: 0,
                            channels: 1,
                            colors: 4,
                            deltas: 0,
                            predictor: Predictor::Zero,
                        }]
                    );
                    assert_eq!(
                        channels[0],
                        Channel {
                            width: 4,
                            height: 1,
                            hshift: -1,
                            vshift: -1,
                            samples: vec![min, min + 9, min + 100, max],
                        }
                    );
                    assert_eq!((channels[1].width, channels[1].height), (7, 16));
                    assert_eq!(
                        state,
                        PaletteState {
                            meta_channels: 1,
                            max_value: 3,
                            max_bitdepth: 2
                        }
                    );
                    undo_palette_for_test(&mut channels, script[0].clone(), 31);
                    assert_eq!(channels, original);
                }
            }
        }
    }

    #[test]
    fn individual_palettes_recompute_begin_after_each_table_insertion() {
        let original = vec![
            channel(&[-100, 100], 80, 16),
            channel(&[1000, 2000, 3000], 80, 16),
            channel(&[0, 10, 20, 30, 40], 80, 16),
        ];
        let mut channels = original.clone();
        let (state, transforms) = apply(&mut channels, 16, &singles());
        assert_eq!(
            state,
            PaletteState {
                meta_channels: 3,
                max_value: 1,
                max_bitdepth: 2
            }
        );
        assert_eq!(
            transforms
                .iter()
                .map(|t| match *t {
                    Transform::Palette {
                        begin, channels, ..
                    } => (begin, channels),
                    _ => panic!(),
                })
                .collect::<Vec<_>>(),
            [(0, 1), (2, 1), (4, 1)]
        );
        assert_eq!(channels[0].samples, [0, 10, 20, 30, 40]);
        assert_eq!(channels[1].samples, [1000, 2000, 3000]);
        assert_eq!(channels[2].samples, [-100, 100]);
        let transformed = channels.clone();
        let mut next_state = state;
        // try_palettes takes the current channel[0] size (five), NOT image size.
        assert!(
            apply_palettes(
                &mut channels,
                &mut next_state,
                16,
                PaletteStage::PostRct,
                &singles()
            )
            .unwrap()
            .is_empty()
        );
        assert_eq!(channels, transformed);
        for t in transforms.into_iter().rev() {
            undo_palette_for_test(&mut channels, t, 16);
        }
        assert_eq!(channels, original);
    }

    #[test]
    fn rgba_decline_then_rgb_and_alpha_use_original_pixel_count() {
        let original = vec![
            channel(&[1, 7], 32, 8),
            channel(&[2, 11], 32, 8),
            channel(&[3, 13], 32, 8),
            channel(&[0, 0, 255, 255], 32, 8),
        ];
        for mode in [
            PaletteMode::Absolute,
            PaletteMode::ExactZero(Predictor::Left),
        ] {
            let options = PaletteOptions {
                palette_colors: 2,
                mode,
                ..PaletteOptions::default()
            };
            let mut channels = original.clone();
            let (state, transforms) = apply(&mut channels, 8, &options);
            assert_eq!(state.meta_channels, 2);
            assert_eq!(state.max_bitdepth, 0);
            assert_eq!(state.max_value, 1);
            assert_eq!(transforms.len(), 2);
            assert!(matches!(
                transforms[0],
                Transform::Palette {
                    begin: 0,
                    channels: 3,
                    ..
                }
            ));
            assert!(matches!(
                transforms[1],
                Transform::Palette {
                    begin: 2,
                    channels: 1,
                    deltas: 0,
                    ..
                }
            ));
            assert_eq!(channels[0].samples, [0, 255]);
            for t in transforms.into_iter().rev() {
                undo_palette_for_test(&mut channels, t, 8);
            }
            assert_eq!(channels, original);
        }
    }

    #[test]
    fn actual_estimate_cost_rejects_only_strict_increase() {
        let original = vec![channel(&[0, 1], 32, 8)];
        let mut candidate = original.clone();
        let mut meta = 0;
        let transform = group::apply_palette_range(
            &mut candidate,
            &mut meta,
            8,
            0,
            1,
            2,
            true,
            PaletteMode::Absolute,
        )
        .unwrap()
        .unwrap();
        let after = estimate(&candidate).unwrap();
        assert!(after > estimate(&original).unwrap()); // unchanged index samples + nonzero table cost
        for (baseline, bypass, accepted) in [
            (after - 1.0, false, false),
            (after, false, true),
            (after + 1.0, false, true),
            (0.0, true, true),
        ] {
            let mut channels = original.clone();
            let mut state = PaletteState::new(8).unwrap();
            let mut transforms = Vec::new();
            assert_eq!(
                attempt(
                    &mut channels,
                    &mut state,
                    &mut transforms,
                    8,
                    0,
                    1,
                    2,
                    true,
                    PaletteMode::Absolute,
                    PaletteTools::default(),
                    baseline,
                    bypass
                )
                .unwrap(),
                accepted
            );
            assert_eq!(
                channels,
                if accepted {
                    candidate.clone()
                } else {
                    original.clone()
                }
            );
            assert_eq!(state.meta_channels, usize::from(accepted));
            assert_eq!(
                transforms,
                if accepted {
                    vec![transform.clone()]
                } else {
                    vec![]
                }
            );
        }
        for (force_palette, zero_tokens) in [(false, false), (true, false), (false, true)] {
            let mut channels = original.clone();
            let options = PaletteOptions {
                speed_tier: 2,
                force_palette,
                zero_tokens,
                channel_colors_pre_transform_percent: 100.0,
                ..PaletteOptions::default()
            };
            assert!(apply(&mut channels, 8, &options).1.is_empty());
            assert_eq!(channels, original); // these switches bypass MULTI-channel cost only
        }
        let options = PaletteOptions {
            speed_tier: 3,
            channel_colors_pre_transform_percent: 100.0,
            ..singles()
        };
        assert_eq!(apply(&mut original.clone(), 8, &options).1.len(), 1);
    }

    #[test]
    fn slow_per_channel_reference_keeps_one_original_cost_baseline() {
        let original = vec![
            channel(&[0, 1], 80, 16),
            channel(&[-100, 100], 80, 16),
            channel(&[0, 1, 2], 80, 16),
            channel(&[10000, 20000], 80, 16),
        ];
        let options = PaletteOptions {
            speed_tier: 2,
            channel_colors_pre_transform_percent: 100.0,
            ..singles()
        };
        let mut expected = original.clone();
        let mut meta = 0;
        let mut script = Vec::new();
        let baseline = estimate(&original).unwrap();
        // Literal per-channel source loop: all candidates compare with this
        // baseline, including after a previous candidate was accepted.
        for i in 0..original.len() {
            let begin = meta + i;
            let mut candidate = expected.clone();
            let mut next_meta = meta;
            if let Some(t) = group::apply_palette_range(
                &mut candidate,
                &mut next_meta,
                16,
                begin,
                1,
                5,
                true,
                PaletteMode::Absolute,
            )
            .unwrap()
                && estimate(&candidate).unwrap() <= baseline
            {
                expected = candidate;
                meta = next_meta;
                script.push(t);
            }
        }
        let mut channels = original;
        let (state, transforms) = apply(&mut channels, 16, &options);
        assert!(!script.is_empty());
        assert_eq!(transforms, script);
        assert_eq!(channels, expected);
        assert_eq!(state.meta_channels, meta);
    }

    #[test]
    fn source_stage_speed_responsive_and_groupwise_gates() {
        let original = vec![channel(&[0, 200], 32, 8)];
        for (stage, speed, responsive, groupwise, do_color, bits, accepted) in [
            (PaletteStage::PreRct, 8, false, false, false, 8, true),
            (PaletteStage::PreRct, 9, false, false, true, 8, false),
            (PaletteStage::PreRct, 9, false, false, true, 16, true),
            (PaletteStage::PreRct, 9, false, false, false, 16, false),
            (PaletteStage::PreRct, 3, true, false, true, 8, false),
            (PaletteStage::PreRct, 3, false, true, true, 8, false),
            (PaletteStage::PostRct, 5, false, true, true, 8, true),
            (PaletteStage::PostRct, 6, false, false, true, 8, false),
            (PaletteStage::PostRct, 3, true, false, true, 8, false),
        ] {
            let options = PaletteOptions {
                speed_tier: speed,
                responsive,
                groupwise,
                do_color,
                ..singles()
            };
            let mut channels = original.clone();
            let mut state = PaletteState::new(bits).unwrap();
            let transforms =
                apply_palettes(&mut channels, &mut state, bits, stage, &options).unwrap();
            assert_eq!(
                !transforms.is_empty(),
                accepted,
                "{stage:?} speed={speed} bits={bits}"
            );
            if !accepted {
                assert_eq!(channels, original);
                assert_eq!(state, PaletteState::new(bits).unwrap());
            }
        }
    }

    #[test]
    fn signed_order_and_direct_branch_are_explicit_settings() {
        let original = vec![channel(&[120, 10], 2, 2); 3];
        for (limit, first) in [(2, 10), (-2, 120)] {
            let options = PaletteOptions {
                palette_colors: limit,
                channel_colors_pre_transform_percent: 0.0,
                ..PaletteOptions::default()
            };
            let mut channels = original.clone();
            let (_, transforms) = apply(&mut channels, 8, &options);
            assert_eq!(transforms.len(), 1);
            assert_eq!(channels[0].samples[0], first);
            undo_palette_for_test(&mut channels, transforms[0].clone(), 8);
            assert_eq!(channels, original);
        }
        for limit in [-2, 0, 2] {
            let mut channels = vec![channel(&[-9, 40], 32, 8)];
            let options = PaletteOptions {
                direct: true,
                palette_colors: limit,
                ..PaletteOptions::default()
            };
            let (state, transforms) = apply(&mut channels, 8, &options);
            assert_eq!(transforms.len(), usize::from(limit > 0));
            assert_eq!(state.max_bitdepth, 9);
            assert_eq!(state.max_value, 255);
        }
        let mut channels = vec![
            channel(&[1, 7], 32, 8),
            channel(&[2, 11], 32, 8),
            channel(&[3, 13], 32, 8),
            channel(&[0, 0, 255, 255], 32, 8),
        ];
        let original = channels.clone();
        let options = PaletteOptions {
            direct: true,
            palette_colors: 2,
            ..PaletteOptions::default()
        };
        assert!(apply(&mut channels, 8, &options).1.is_empty());
        assert_eq!(channels, original); // no all-minus-one or singles in direct branch
    }

    #[test]
    fn error_rolls_back_earlier_successful_channel_palettes() {
        let original = vec![
            channel(&[0, 100], 32, 8),
            channel(&[i32::MIN, i32::MAX], 32, 8),
        ];
        let mut channels = original.clone();
        let mut state = PaletteState::new(31).unwrap();
        assert!(
            apply_palettes(
                &mut channels,
                &mut state,
                31,
                PaletteStage::PreRct,
                &singles()
            )
            .is_err()
        );
        assert_eq!(channels, original);
        assert_eq!(state, PaletteState::new(31).unwrap());
        // First channel alone is eligible; only the second range overflows the
        // source's signed int threshold, after the first candidate was accepted.
        assert_eq!(
            apply(&mut vec![original[0].clone()], 31, &singles())
                .1
                .len(),
            1
        );
        for bad in [
            PaletteOptions {
                palette_colors: i32::MIN,
                ..singles()
            },
            PaletteOptions {
                speed_tier: 10,
                ..singles()
            },
            PaletteOptions {
                channel_colors_percent: f32::NAN,
                ..singles()
            },
            PaletteOptions {
                channel_colors_pre_transform_percent: -1.0,
                ..singles()
            },
            PaletteOptions {
                force_palette: true,
                ..singles()
            },
            PaletteOptions {
                mode: PaletteMode::ExactZero(Predictor::Weighted),
                ..singles()
            },
        ] {
            assert!(
                apply_palettes(&mut channels, &mut state, 31, PaletteStage::PreRct, &bad).is_err()
            );
            assert_eq!(channels, original);
            assert_eq!(state, PaletteState::new(31).unwrap());
        }
    }

    #[test]
    fn pre_rct_and_post_rct_scripts_roundtrip_multiple_meta_channels() {
        let original = vec![
            channel(&[10, 50, 90], 96, 16),
            channel(&[20, 60, 100], 96, 16),
            channel(&[30, 70, 110], 96, 16),
        ];
        let mut channels = original.clone();
        let (mut state, pre) = apply(&mut channels, 8, &singles());
        assert_eq!(pre.len(), 3);
        assert_eq!(state.meta_channels, 3);
        let begin = state.meta_channels;
        let rct = group::apply_rct(&mut channels, state.meta_channels, begin, 6)
            .unwrap()
            .unwrap();
        state.max_bitdepth += 1;
        assert_eq!(rct, Transform::RctAt { begin: 3, kind: 6 });
        let options = PaletteOptions {
            palette_colors: 8,
            mode: PaletteMode::ExactZero(Predictor::Left),
            ..PaletteOptions::default()
        };
        let post = apply_palettes(
            &mut channels,
            &mut state,
            8,
            PaletteStage::PostRct,
            &options,
        )
        .unwrap();
        assert_eq!(post.len(), 1);
        assert_eq!(state.meta_channels, 4);
        assert!(matches!(
            post[0],
            Transform::Palette {
                begin: 3,
                channels: 3,
                deltas: 1,
                ..
            }
        ));
        for t in post.into_iter().rev() {
            undo_palette_for_test(&mut channels, t, 8);
        }
        // Independent source inverse for RCT 6 (YCoCg), after post-palettes.
        for i in 0..channels[begin].samples.len() {
            let y = channels[begin].samples[i];
            let co = channels[begin + 1].samples[i];
            let cg = channels[begin + 2].samples[i];
            let tmp = y.wrapping_sub(cg >> 1);
            let b = tmp.wrapping_sub(co >> 1);
            channels[begin].samples[i] = b.wrapping_add(co);
            channels[begin + 1].samples[i] = tmp.wrapping_add(cg);
            channels[begin + 2].samples[i] = b;
        }
        for t in pre.into_iter().rev() {
            undo_palette_for_test(&mut channels, t, 8);
        }
        assert_eq!(channels, original);
    }
}
