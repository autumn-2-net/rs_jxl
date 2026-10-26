// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; the complete upstream notice is in docs/PORTING_KERNELS.md.

//! Source MA sampling and shared-population property quantization.
//!
//! Translated from frozen b866b3defec13494bfbaf705db93581cda118e37:
//! enc_modular.cc's property/fraction policy; enc_encoding.cc::TreeSampleFraction,
//! GatherTreeData and LearnTree; enc_ma.cc::CollectPixelSamples,
//! PreQuantizeProperties, QuantizeSamples and QuantizeHistogram; base/random.h.
//!
//! Integration: register `pub mod sample`; resolve SamplingPolicy from source
//! parameters (or populate it from the scheduler's already-resolved policy).
//! Collect pixel quantiles BEFORE skipping verified constant channels. Visit
//! training pixels with a fresh ChannelSampler for EVERY channel/group, advancing
//! weighted/property state on ALL pixels, even rejected samples. Supplied IDs
//! are retained: group zero is not substituted for real Modular stream IDs.
//! Feed policy.apply_to_train's quantization/order/cost into ma_train::train.
//! The integrated entry point is training::learn; standalone callers use
//! collect_pixel_samples(...).quantization(...) and pass the result through
//! TrainOptions::property_quantization. ChannelSampler yields one bool per
//! visible pixel; selected pixels retain weight 1, not inverse-probability weights.
//!
//! Neither sampler allocates an image-sized mask. Pixel/difference pre-samples
//! are accumulated in four 1025-bin histograms; static populations use sparse
//! maps, not allocations proportional to the largest stream ID. SampleLimits
//! rejects an exceeded caller budget; it never substitutes midpoint/reservoir
//! sampling, truncates samples, or changes their weights. Caller-owned Sample
//! storage and source batch/wave deduplication are separate from this module.
//!
//! Training RNG arithmetic and float-to-integer boundaries are explicit. The
//! geometric sampler uses f32 logarithms like the source; libm implementations
//! may differ at an integer boundary. Signed difference narrowing wraps to i32;
//! abs(i32::MIN), undefined in C++, is safely clipped using unsigned magnitude.

use std::collections::BTreeMap;
use std::ops::Range;

use super::ma_train::{ChannelView, PropertyQuantization, TrainOptions};
use super::predictor::Predictor;
use crate::{Error, Result};

pub const SOURCE_GATHER_BUDGET_BYTES: u64 = 1 << 30;
pub const SOURCE_MAX_GATHER_STRIPES: usize = 16;
const QUANTILE_RANGE: i32 = 512;
const HISTOGRAM_SIZE: usize = 1025;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GatherMode {
    /// Parallelize the whole batch only when its source estimate fits the budget.
    #[default]
    Channels,
    /// Append bounded, ordered waves when the whole batch exceeds the budget.
    Waves,
}

/// Gather scheduling/traversal: never changes sample sites, fraction or quantiles.
/// Explicit striping may change learning-only WP state through source warm-up.
/// The byte budget uses the source compact-record estimate, not a strict RSS
/// bound. A single oversized channel runs serially without an extra batch.
#[derive(Clone, Debug)]
pub struct GatherOptions {
    pub mode: GatherMode,
    /// Maximum independent gather chunks, matching LosslessWorkerLimits::tasks.
    pub workers: usize,
    pub batch_budget_bytes: u64,
    /// Source JXL_MA_GATHER_STRIPES, 1..=16. Values above one approximate
    /// learning-only WP state with sixteen warm-up rows; coding is unchanged.
    pub stripes: usize,
    /// Source JXL_MA_SPARSE_GATHER: discard unused property calculations while
    /// still advancing every RNG draw, WP prediction/error and gradient state.
    pub sparse: bool,
    /// Opt-in full-hash cache, bounded to 16 MiB per collector.
    pub cache_hashes: bool,
    /// Experimental terminal concat for one parallel wave: at least 99% rows
    /// per selected weight and at most 64 MiB of column/count payload.
    pub concat_batches: bool,
}

impl Default for GatherOptions {
    fn default() -> Self {
        Self {
            mode: GatherMode::Channels,
            workers: 8,
            batch_budget_bytes: SOURCE_GATHER_BUDGET_BYTES,
            stripes: 1,
            sparse: false,
            cache_hashes: false,
            concat_batches: false,
        }
    }
}

pub(super) struct GatherWave {
    pub tasks: Range<usize>,
    pub chunks: Vec<Range<usize>>,
    pub estimated_bytes: f64,
}

impl GatherOptions {
    pub fn validate(&self) -> Result<()> {
        if !(1..=64).contains(&self.workers)
            || self.batch_budget_bytes == 0
            || !(1..=SOURCE_MAX_GATHER_STRIPES).contains(&self.stripes)
        {
            return Err(Error::InvalidInput("invalid MA gather limits".into()));
        }
        Ok(())
    }

    /// LearnTree's source stripe gates and integer row boundaries.
    pub(super) fn rows(&self, width: usize, height: usize) -> Result<Vec<Range<usize>>> {
        self.validate()?;
        let area = width
            .checked_mul(height)
            .ok_or(Error::Limit("MA stripe area overflow"))?;
        let parts = if height >= 512 && area >= 1 << 18 {
            self.stripes.min(height / 128)
        } else {
            1
        };
        Ok((0..parts)
            .map(|part| {
                // Equivalent to height * part / parts without intermediate overflow.
                let boundary = |p| height / parts * p + height % parts * p / parts;
                boundary(part)..boundary(part + 1)
            })
            .collect())
    }

    /// LearnTree's ordered waves and MaTreeGroupBoundaries channel work cuts.
    /// `expected` includes the source +64 allowance per nonempty channel/stripe.
    pub(super) fn plan(
        &self,
        work: &[u64],
        expected: &[f64],
        predictors: usize,
        properties: &[usize],
        parallel: bool,
    ) -> Result<Vec<GatherWave>> {
        self.validate()?;
        if work.len() != expected.len() || expected.iter().any(|n| !n.is_finite() || *n < 0.0) {
            return Err(Error::InvalidInput(
                "invalid MA gather work estimates".into(),
            ));
        }
        let static_properties = properties.iter().filter(|&&p| p < 2).count();
        let sample_bytes = 2 * predictors + 3 * static_properties + properties.len() + 2;
        // Master and worker batches each retain up to 16 bytes of full hashes
        // per record. Charge all records even beyond the bounded cache prefix.
        let hash_bytes = if self.cache_hashes { 32 } else { 0 };
        let record_bytes = (2 * sample_bytes + 16 + hash_bytes) as f64;
        let total_samples: f64 = expected.iter().sum();
        let total_bytes = total_samples * record_bytes;
        if !total_bytes.is_finite() {
            return Err(Error::Limit("MA gather estimate overflow"));
        }
        let budget = self.batch_budget_bytes as f64;
        let parallel = parallel && self.workers > 1 && total_samples >= 8192.0;
        let waves = parallel && self.mode == GatherMode::Waves && total_bytes > budget;
        let mut result = Vec::new();
        let mut first = 0;
        while first < work.len() {
            let mut last = work.len();
            let mut bytes = total_bytes;
            if waves {
                last = first;
                bytes = 0.0;
                while last < work.len() {
                    let next = expected[last] * record_bytes;
                    if last > first && bytes + next > budget {
                        break;
                    }
                    bytes += next;
                    last += 1;
                }
            }
            let bounds = if parallel && bytes <= budget {
                super::tree_groups::boundaries(
                    &work[first..last],
                    self.workers,
                    65_536,
                    self.workers,
                )?
            } else {
                vec![0, last - first]
            };
            result.push(GatherWave {
                tasks: first..last,
                chunks: bounds
                    .windows(2)
                    .map(|p| first + p[0]..first + p[1])
                    .collect(),
                estimated_bytes: bytes,
            });
            first = last;
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TreeMode {
    #[default]
    Default,
    WeightedOnly,
    NoWeighted,
    GradientOnly,
}

/// Inputs to the pixel-lossless enc_modular.cc policy, before its fraction scale.
/// `num_streams` includes all Modular stream slots, not only nonempty groups.
#[derive(Clone, Debug)]
pub struct SourcePolicy {
    /// common.h::SpeedTier, -1..=9 (source effort = 10 - speed_tier).
    pub speed_tier: i8,
    pub decoding_speed_tier: u8,
    /// Explicit predictor choice before decoding-speed restrictions. Keep this
    /// equal to ModularOptions.predictor when resolving an integrated policy.
    pub predictor: Predictor,
    pub responsive: bool,
    pub num_streams: usize,
    pub learning_fraction: f32,
    pub max_previous_channels: usize,
    pub num_extra_channels: usize,
    pub is_gray: bool,
    pub tree_mode: TreeMode,
    pub max_channel_size: usize,
}

impl Default for SourcePolicy {
    fn default() -> Self {
        Self {
            speed_tier: 3,
            decoding_speed_tier: 0,
            predictor: Predictor::Best,
            responsive: false,
            num_streams: 1,
            learning_fraction: 0.5,
            max_previous_channels: 0,
            num_extra_channels: 0,
            is_gray: false,
            tree_mode: TreeMode::Default,
            max_channel_size: 0x00ff_ffff,
        }
    }
}

/// Resolved policy. `nb_repeats` is already effort-scaled; do not scale it twice
/// when translating scheduler::params::TrainingPolicy into this struct.
#[derive(Clone, Debug)]
pub struct SamplingPolicy {
    pub nb_repeats: f32,
    pub properties: Vec<usize>,
    pub max_property_values: usize,
    pub max_channel_size: usize,
    pub node_threshold: f32,
    pub previous_channels: usize,
    pub tree_mode: TreeMode,
    pub fast_decode_multiplier: f64,
    /// enc_modular.cc sets Gradient at decoding speed >= 3 independently of
    /// TreeMode, which otherwise restricts properties, not predictor candidates.
    pub predictor_override: Option<Predictor>,
}

impl SamplingPolicy {
    pub fn from_source(source: &SourcePolicy) -> Result<Self> {
        validate_fraction(source.learning_fraction)?;
        if !(-1..=9).contains(&source.speed_tier) || source.num_streams == 0 {
            return Err(Error::InvalidInput(
                "invalid MA source policy tier/stream count".into(),
            ));
        }
        let mut properties = if source.responsive {
            vec![0, 1, 4, 5, 6, 7, 8, 15, 9, 10, 11, 12, 13, 14, 2, 3]
        } else {
            vec![0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8]
        };
        if !source.responsive && source.num_streams < 30 && source.speed_tier > 1 {
            properties.remove(1);
        }
        // TectonicPlate is absent from Init's Glacier/Tortoise case and must
        // retain the default 3 properties, 32 bins and *0.3 sample fraction.
        let (count, max_property_values, scale): (usize, usize, f32) = match source.speed_tier {
            5 => (4, 48, 0.5),
            4 => (5, 64, 0.7),
            3 => (7, 96, 1.0),
            2 => (10, 128, 1.1),
            0 | 1 => (properties.len(), 256, 1.3),
            _ => (3, 32, 0.3),
        };
        properties.truncate(count);
        let available = source
            .num_extra_channels
            .checked_add(if source.is_gray { 0 } else { 2 })
            .ok_or(Error::Limit("MA reference channel count"))?;
        let previous_channels = source.max_previous_channels.min(available);
        if previous_channels > 60 {
            return Err(Error::Limit("MA reference properties exceed property IDs"));
        }
        if source.speed_tier > 1 {
            properties.extend((0..previous_channels).map(|i| 16 + 4 * i + 3));
        } else {
            properties.extend(16..16 + 4 * previous_channels);
        }
        let tree_mode = match source.decoding_speed_tier {
            0 => source.tree_mode,
            1 | 2 => TreeMode::NoWeighted,
            _ => TreeMode::GradientOnly,
        };
        match tree_mode {
            TreeMode::WeightedOnly => properties = vec![15],
            TreeMode::GradientOnly => properties = vec![9],
            TreeMode::NoWeighted => properties.retain(|&p| p != 15),
            TreeMode::Default => {}
        }
        let nb_repeats = if source.decoding_speed_tier >= 4 {
            0.0
        } else {
            (source.learning_fraction * scale).min(1.0)
        };
        let result = Self {
            nb_repeats,
            properties,
            max_property_values,
            max_channel_size: source.max_channel_size,
            node_threshold: (75
                + 14 * i32::from(source.speed_tier)
                + 10 * i32::from(source.decoding_speed_tier)) as f32,
            previous_channels,
            tree_mode,
            predictor_override: (source.decoding_speed_tier >= 3
                || (matches!(source.decoding_speed_tier, 1 | 2)
                    && source.predictor == Predictor::Weighted))
                .then_some(Predictor::Gradient),
            fast_decode_multiplier: f64::from(match source.decoding_speed_tier {
                0 => 1.001f32,
                1 => 1.005,
                2 => 1.015,
                _ => 1.01,
            }),
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<()> {
        validate_fraction(self.nb_repeats)?;
        validate_properties(&self.properties, self.max_property_values)?;
        if !self.node_threshold.is_finite()
            || self.node_threshold < 0.0
            || !self.fast_decode_multiplier.is_finite()
            || self.fast_decode_multiplier < 1.0
            || self.previous_channels > 60
            || self.predictor_override.is_some_and(|p| p as u32 >= 14)
        {
            return Err(Error::InvalidInput("invalid MA sampling policy".into()));
        }
        Ok(())
    }

    /// Does not change predictors, leaf/resource limits, constants, or multipliers.
    pub fn apply_to_train(
        &self,
        quantiles: &PropertySamples,
        statistics: SamplingStats,
        train: &mut TrainOptions,
    ) -> Result<()> {
        self.validate()?;
        let quantization = quantiles.quantization(&self.properties, self.max_property_values)?;
        let split_threshold = effective_split_threshold(
            self.node_threshold,
            statistics.selected_samples,
            statistics.source_total_pixels,
        )?;
        train.properties = self.properties.clone();
        train.max_property_values = self.max_property_values;
        train.split_threshold = split_threshold;
        train.fast_decode_multiplier = self.fast_decode_multiplier;
        train.property_quantization = Some(quantization);
        Ok(())
    }
}

fn validate_fraction(fraction: f32) -> Result<()> {
    if !fraction.is_finite() || fraction < 0.0 {
        return Err(Error::InvalidInput(
            "MA sample fraction must be finite and nonnegative".into(),
        ));
    }
    Ok(())
}

fn validate_properties(properties: &[usize], max_values: usize) -> Result<()> {
    if !(2..=256).contains(&max_values)
        || properties.is_empty()
        || properties.len() >= 64
        || properties
            .iter()
            .enumerate()
            .any(|(i, &p)| p >= 256 || properties[..i].contains(&p))
    {
        return Err(Error::InvalidInput(
            "invalid MA sampling property order/bin count".into(),
        ));
    }
    Ok(())
}

/// TreeSampleFraction: a positive fraction samples at least 1024 EXPECTED sites
/// per channel, capped at the whole channel. Zero disables training.
pub fn tree_sample_fraction(channel_pixels: usize, nb_repeats: f32) -> Result<f64> {
    validate_fraction(nb_repeats)?;
    if nb_repeats == 0.0 || channel_pixels == 0 {
        return Ok(0.0);
    }
    Ok(f64::from(nb_repeats.min(1.0)).max((1024.0 / channel_pixels as f64).min(1.0)))
}

/// Source LearnTree's f32/double rounding and sampled-population cost adjustment.
pub fn effective_split_threshold(
    node_threshold: f32,
    selected: u64,
    total_pixels: u64,
) -> Result<f64> {
    if !node_threshold.is_finite()
        || node_threshold < 0.0
        || total_pixels == 0
        || selected > total_pixels
    {
        return Err(Error::InvalidInput(
            "invalid MA sampled split-cost population".into(),
        ));
    }
    let fraction = selected as f32 / total_pixels as f32;
    let required_cost = (f64::from(fraction) * 0.9 + 0.1) as f32;
    Ok(f64::from(node_threshold * required_cost))
}

#[derive(Clone, Debug)]
struct SourceRng {
    state: [u64; 2],
}

impl SourceRng {
    fn new(seed: u64) -> Self {
        Self {
            state: [
                0x94d0_49bb_1331_11eb,
                0xbf58_476d_1ce4_e5b9u64.wrapping_add(seed),
            ],
        }
    }

    fn next(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        let bits = s1.wrapping_add(s0);
        self.state[0] = s0;
        s1 ^= s1 << 23;
        s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
        self.state[1] = s1;
        bits
    }

    fn uniform_f32(&mut self) -> f32 {
        f32::from_bits((self.next() >> 41) as u32 | 0x3f80_0000) - 1.0
    }

    fn geometric(&mut self, inverse_log: f32) -> u32 {
        ((1.0f32 - self.uniform_f32()).ln() * inverse_log) as u32
    }
}

/// O(1) sampler. One Iterator item per visible raster-order pixel, NOT per
/// selected pixel. Construct afresh for each channel, independently of IDs.
#[derive(Clone, Debug)]
pub struct ChannelSampler {
    rng: SourceRng,
    remaining: usize,
    fraction: f64,
    threshold: u64,
}

impl ChannelSampler {
    pub fn new(width: usize, height: usize, nb_repeats: f32) -> Result<Self> {
        let pixels = width
            .checked_mul(height)
            .ok_or(Error::Limit("MA sample dimensions overflow"))?;
        let fraction = tree_sample_fraction(pixels, nb_repeats)?;
        Ok(Self {
            rng: SourceRng::new(0),
            remaining: pixels,
            fraction,
            threshold: (f64::from(u32::MAX) * fraction) as u64,
        })
    }

    pub fn fraction(&self) -> f64 {
        self.fraction
    }

    /// Advance RNG for preceding rows when a caller partitions work into stripes.
    /// This preserves sample locations, not the caller's weighted predictor state.
    pub fn skip_pixels(&mut self, count: usize) -> Result<()> {
        if count > self.remaining {
            return Err(Error::InvalidInput("MA sample skip exceeds channel".into()));
        }
        for _ in 0..count {
            let _ = self.next();
        }
        Ok(())
    }
}

impl Iterator for ChannelSampler {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let draw = self.rng.next() >> 32;
        Some(self.fraction != 0.0 && draw <= self.threshold)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for ChannelSampler {}
impl std::iter::FusedIterator for ChannelSampler {}

#[derive(Clone, Copy, Debug)]
pub struct SampleLimits {
    /// Explicit error limit, not a sampling target. Never drops selected pixels.
    pub max_selected_samples: u64,
    /// First proven constant channel; None gathers all eligible channels.
    pub variable_channels: Option<u32>,
}
impl Default for SampleLimits {
    fn default() -> Self {
        Self {
            max_selected_samples: u64::MAX,
            variable_channels: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SamplingStats {
    pub visited_pixels: u64,
    pub selected_samples: u64,
    /// Frozen LearnTree initializes to max(first-stream pixels, 1), then ADDS
    /// every gathered pixel (including the first stream again). Retain this for
    /// source-equivalent threshold scaling; it is not a unique-pixel count.
    pub source_total_pixels: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampleSite {
    pub view_index: usize,
    pub channel: u32,
    pub group: u32,
    pub x: usize,
    pub y: usize,
    pub index: usize,
}

/// Invoke callback on EVERY eligible training pixel, with its selection flag.
/// Update weighted/property state even when selected is false. An error aborts
/// the whole collection, never yields a silently truncated training population.
pub fn visit_training_pixels(
    channels: &[ChannelView<'_>],
    policy: &SamplingPolicy,
    limits: SampleLimits,
    mut visit: impl FnMut(SampleSite, bool) -> Result<()>,
) -> Result<SamplingStats> {
    policy.validate()?;
    let eligible = eligible_views(channels, policy.max_channel_size)?;
    let first_group = channels.iter().map(|v| v.group).min();
    let mut initial = 0u64;
    for &index in &eligible {
        let view = &channels[index];
        if Some(view.group) == first_group
            && limits
                .variable_channels
                .is_none_or(|end| view.channel < end)
        {
            initial = initial
                .checked_add(view.width as u64 * view.height as u64)
                .ok_or(Error::Limit("MA population overflow"))?;
        }
    }
    let mut stats = SamplingStats {
        source_total_pixels: initial.max(1),
        ..SamplingStats::default()
    };
    for index in eligible {
        let view = &channels[index];
        if limits
            .variable_channels
            .is_some_and(|end| view.channel >= end)
        {
            continue;
        }
        let sampler = ChannelSampler::new(view.width, view.height, policy.nb_repeats)?;
        for (position, selected) in sampler.enumerate() {
            if selected && stats.selected_samples == limits.max_selected_samples {
                return Err(Error::Limit("source MA selected-sample budget exceeded"));
            }
            stats.selected_samples += u64::from(selected);
            stats.visited_pixels = stats
                .visited_pixels
                .checked_add(1)
                .ok_or(Error::Limit("MA population overflow"))?;
            stats.source_total_pixels = stats
                .source_total_pixels
                .checked_add(1)
                .ok_or(Error::Limit("MA population overflow"))?;
            visit(
                SampleSite {
                    view_index: index,
                    channel: view.channel,
                    group: view.group,
                    x: position % view.width,
                    y: position / view.width,
                    index: position,
                },
                selected,
            )?;
        }
    }
    Ok(stats)
}

fn eligible_views(channels: &[ChannelView<'_>], max_channel_size: usize) -> Result<Vec<usize>> {
    let mut order: Vec<_> = (0..channels.len()).collect();
    order.sort_unstable_by_key(|&i| (channels[i].group, channels[i].channel));
    let mut previous = None;
    let mut blocked_group = None;
    let mut eligible = Vec::new();
    for index in order {
        let view = &channels[index];
        let id = (view.group, view.channel);
        if view.channel > i32::MAX as u32 || view.group > i32::MAX as u32 || previous == Some(id) {
            return Err(Error::InvalidInput(
                "invalid/duplicate MA channel or stream ID".into(),
            ));
        }
        previous = Some(id);
        if blocked_group == Some(view.group) {
            continue;
        }
        if !view.is_meta && (view.width > max_channel_size || view.height > max_channel_size) {
            blocked_group = Some(view.group);
            continue;
        }
        if view.width == 0 || view.height == 0 {
            continue;
        }
        let end = (view.height - 1)
            .checked_mul(view.stride)
            .and_then(|n| n.checked_add(view.width))
            .ok_or(Error::Limit("MA channel addressing overflow"))?;
        view.width
            .checked_mul(view.height)
            .ok_or(Error::Limit("MA channel population overflow"))?;
        if view.stride < view.width || end > view.pixels.len() {
            return Err(Error::InvalidInput(
                "invalid MA channel plane/stride".into(),
            ));
        }
        eligible.push(index);
    }
    Ok(eligible)
}

/// Sufficient statistics of exactly the source pre-quantization sample sites.
/// Raw samples need not be retained: clipping and magnitudes are counted BEFORE
/// any family is mutated by PreQuantizeProperties' property-order traversal.
#[derive(Clone, Debug)]
pub struct PropertySamples {
    pixel: [u64; HISTOGRAM_SIZE],
    abs_pixel: [u64; HISTOGRAM_SIZE],
    difference: [u64; HISTOGRAM_SIZE],
    abs_difference: [u64; HISTOGRAM_SIZE],
    pub channel_pixel_count: BTreeMap<u32, u64>,
    pub group_pixel_count: BTreeMap<u32, u64>,
    pub sampled_pixels: u64,
}

impl Default for PropertySamples {
    fn default() -> Self {
        Self {
            pixel: [0; HISTOGRAM_SIZE],
            abs_pixel: [0; HISTOGRAM_SIZE],
            difference: [0; HISTOGRAM_SIZE],
            abs_difference: [0; HISTOGRAM_SIZE],
            channel_pixel_count: BTreeMap::new(),
            group_pixel_count: BTreeMap::new(),
            sampled_pixels: 0,
        }
    }
}

impl PropertySamples {
    fn add(&mut self, pixel: i32, difference: i32) -> Result<()> {
        self.sampled_pixels = self
            .sampled_pixels
            .checked_add(1)
            .ok_or(Error::Limit("MA pre-sample population overflow"))?;
        self.pixel[(pixel.clamp(-512, 512) + 512) as usize] += 1;
        self.abs_pixel[(pixel.unsigned_abs().min(512) + 512) as usize] += 1;
        self.difference[(difference.clamp(-512, 512) + 512) as usize] += 1;
        self.abs_difference[(difference.unsigned_abs().min(512) + 512) as usize] += 1;
        Ok(())
    }

    /// PreQuantizeProperties family mapping, including its lazy signed-cache
    /// initialization before abs mutates a population. Empty caches are retried
    /// in property order exactly as in the source (including rare skewed cases).
    pub fn quantization(
        &self,
        properties: &[usize],
        max_values: usize,
    ) -> Result<PropertyQuantization> {
        validate_properties(properties, max_values)?;
        if self
            .channel_pixel_count
            .keys()
            .chain(self.group_pixel_count.keys())
            .any(|&id| id > i32::MAX as u32)
        {
            return Err(Error::InvalidInput(
                "MA static quantile ID exceeds i32".into(),
            ));
        }
        let mut pixel = FamilyQuantiles::new(&self.pixel, &self.abs_pixel, max_values);
        let mut difference =
            FamilyQuantiles::new(&self.difference, &self.abs_difference, max_values);
        let mut thresholds = Vec::with_capacity(properties.len());
        for &property in properties {
            let bins = match property {
                0 => quantize_counts(
                    self.channel_pixel_count
                        .iter()
                        .map(|(&id, &count)| (id as i32, count)),
                    max_values,
                ),
                1 => quantize_counts(
                    self.group_pixel_count
                        .iter()
                        .map(|(&id, &count)| (id as i32, count)),
                    max_values,
                ),
                2 | 3 => (1..max_values)
                    .map(|i| (i * 256 / max_values) as i32 - 1)
                    .collect(),
                4 | 5 => pixel.absolute(),
                6..=8 => pixel.signed(),
                15 => weighted_thresholds(max_values),
                p if p >= 16 => match (p - 16) % 4 {
                    0 => pixel.absolute(),
                    1 => pixel.signed(),
                    2 => difference.absolute(),
                    _ => difference.signed(),
                },
                _ => difference.signed(),
            };
            thresholds.push(bins);
        }
        Ok(PropertyQuantization {
            properties: properties.to_vec(),
            thresholds,
        })
    }
}

/// CollectPixelSamples across ALL provided streams, ordered by original group
/// and channel IDs. Width-1 channels do not enter this pre-pass or static counts;
/// they DO enter visit_training_pixels. No constant-channel filtering here.
pub fn collect_pixel_samples(
    channels: &[ChannelView<'_>],
    policy: &SamplingPolicy,
) -> Result<PropertySamples> {
    policy.validate()?;
    let mut result = PropertySamples::default();
    if policy.nb_repeats == 0.0 {
        return Ok(result);
    }
    let fraction = (f64::from(policy.nb_repeats) * 0.1).min(0.99) as f32;
    let inverse_log = (1.0f64 / f64::from((1.0f32 - fraction).ln())) as f32;
    if !inverse_log.is_finite() {
        return Err(Error::InvalidInput(
            "MA pre-quantization fraction is below source f32 resolution".into(),
        ));
    }
    let eligible = eligible_views(channels, policy.max_channel_size)?;
    let mut begin = 0;
    while begin < eligible.len() {
        let group = channels[eligible[begin]].group;
        let mut end = begin + 1;
        while end < eligible.len() && channels[eligible[end]].group == group {
            end += 1;
        }
        let mut rng = SourceRng::new(u64::from(group));
        let mut position = u64::from(rng.geometric(inverse_log));
        for &index in &eligible[begin..end] {
            let view = &channels[index];
            if view.width <= 1 {
                continue;
            }
            let pixels = view.width as u64 * view.height as u64;
            add_count(&mut result.channel_pixel_count, view.channel, pixels)?;
            add_count(&mut result.group_pixel_count, view.group, pixels)?;
            while position < pixels {
                let x = (position % view.width as u64) as usize;
                let y = (position / view.width as u64) as usize;
                let row = &view.pixels[y * view.stride..y * view.stride + view.width];
                let xp = if x == 0 { 1 } else { x - 1 };
                // CollectPixelSamples stores its wide difference in pixel_type
                // before quantile clipping, so +/-4294967295 become -/+1.
                result.add(row[x], row[x].wrapping_sub(row[xp]))?;
                position = position
                    .checked_add(u64::from(rng.geometric(inverse_log)) + 1)
                    .ok_or(Error::Limit("MA geometric sample position overflow"))?;
            }
            position -= pixels;
        }
        begin = end;
    }
    Ok(result)
}

fn add_count(counts: &mut BTreeMap<u32, u64>, id: u32, count: u64) -> Result<()> {
    let entry = counts.entry(id).or_default();
    *entry = entry
        .checked_add(count)
        .ok_or(Error::Limit("MA static population overflow"))?;
    Ok(())
}

fn quantize_counts(counts: impl Iterator<Item = (i32, u64)> + Clone, chunks: usize) -> Vec<i32> {
    let total: u128 = counts.clone().map(|(_, count)| u128::from(count)).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut cumulative = 0u128;
    let mut next = 1u128;
    let mut result = Vec::new();
    for (value, count) in counts {
        cumulative += u128::from(count);
        let scaled = cumulative * chunks as u128;
        if scaled >= next * total {
            result.push(value);
            next = scaled / total + 1;
        }
    }
    result.pop();
    result
}

fn quantize_histogram(histogram: &[u64; HISTOGRAM_SIZE], chunks: usize) -> Vec<i32> {
    quantize_counts(
        histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| (i as i32 - QUANTILE_RANGE, count)),
        chunks,
    )
}

struct FamilyQuantiles<'a> {
    signed_histogram: &'a [u64; HISTOGRAM_SIZE],
    absolute_histogram: &'a [u64; HISTOGRAM_SIZE],
    signed: Vec<i32>,
    absolute: Vec<i32>,
    mutated: bool,
    chunks: usize,
}

impl<'a> FamilyQuantiles<'a> {
    fn new(
        signed: &'a [u64; HISTOGRAM_SIZE],
        absolute: &'a [u64; HISTOGRAM_SIZE],
        chunks: usize,
    ) -> Self {
        Self {
            signed_histogram: signed,
            absolute_histogram: absolute,
            signed: Vec::new(),
            absolute: Vec::new(),
            mutated: false,
            chunks,
        }
    }
    fn signed(&mut self) -> Vec<i32> {
        if self.signed.is_empty() {
            self.signed = quantize_histogram(
                if self.mutated {
                    self.absolute_histogram
                } else {
                    self.signed_histogram
                },
                self.chunks,
            );
        }
        self.signed.clone()
    }
    fn absolute(&mut self) -> Vec<i32> {
        if self.absolute.is_empty() {
            let _ = self.signed();
            self.mutated = true;
            self.absolute = quantize_histogram(self.absolute_histogram, self.chunks);
        }
        self.absolute.clone()
    }
}

fn weighted_thresholds(max_values: usize) -> Vec<i32> {
    if max_values < 32 {
        vec![-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127]
    } else if max_values < 64 {
        vec![
            -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -5, -3, -1, 0, 1, 3, 5, 7, 11,
            15, 23, 31, 47, 63, 95, 127, 191, 255,
        ]
    } else {
        vec![
            -255, -223, -191, -159, -127, -111, -95, -79, -63, -55, -47, -39, -31, -27, -23, -19,
            -15, -13, -11, -9, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15,
            19, 23, 27, 31, 39, 47, 55, 63, 79, 95, 111, 127, 159, 191, 223, 255,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_stripes_use_source_gates_and_checked_row_cuts() {
        let mut options = GatherOptions::default();
        assert_eq!(options.stripes, 1);
        assert!(!options.sparse);
        assert!(!options.cache_hashes);
        assert!(!options.concat_batches);
        assert_eq!(
            options.rows(512, 1024).unwrap(),
            std::iter::once(0..1024).collect::<Vec<_>>()
        );
        options.stripes = 16;
        for (width, height) in [(4096, 511), (511, 512), (0, 1024)] {
            assert_eq!(
                options.rows(width, height).unwrap(),
                std::iter::once(0..height).collect::<Vec<_>>()
            );
        }
        assert_eq!(
            options.rows(512, 512).unwrap(),
            [0..128, 128..256, 256..384, 384..512]
        );
        options.stripes = 3;
        assert_eq!(
            options.rows(512, 513).unwrap(),
            [0..171, 171..342, 342..513]
        );
        assert!(options.rows(usize::MAX, 512).is_err());
        for stripes in [0, 17, usize::MAX] {
            options.stripes = stripes;
            assert!(options.validate().is_err());
        }
    }

    #[test]
    fn stripe_rng_warmup_does_not_change_selected_coordinates() {
        for fraction in [0.0, 0.01, 0.25, 1.0] {
            let whole: Vec<_> = ChannelSampler::new(512, 513, fraction).unwrap().collect();
            let options = GatherOptions {
                stripes: 3,
                ..GatherOptions::default()
            };
            let mut striped = Vec::new();
            for rows in options.rows(512, 513).unwrap() {
                let warm = rows.start.saturating_sub(16);
                let mut sampler = ChannelSampler::new(512, 513, fraction).unwrap();
                sampler.skip_pixels(warm * 512).unwrap();
                for y in warm..rows.end {
                    for _ in 0..512 {
                        let selected = sampler.next().unwrap();
                        if y >= rows.start {
                            striped.push(selected);
                        }
                    }
                }
            }
            assert_eq!(striped, whole);
        }
    }

    #[test]
    fn gather_wave_plan_uses_source_estimate_and_ordered_work_cuts() {
        let mut options = GatherOptions {
            mode: GatherMode::Waves,
            batch_budget_bytes: 1 << 20,
            ..GatherOptions::default()
        };
        let work = [65_536; 8];
        let expected = [65_536.0 * 0.1 + 64.0; 8];
        let properties = [0, 1, 15, 9, 19, 23];
        let waves = options
            .plan(&work, &expected, 2, &properties, true)
            .unwrap();
        assert_eq!(
            waves.iter().map(|w| w.tasks.clone()).collect::<Vec<_>>(),
            [0..3, 3..6, 6..8]
        );
        for wave in &waves {
            assert!(wave.estimated_bytes <= options.batch_budget_bytes as f64);
            assert_eq!(wave.chunks.len(), wave.tasks.len());
            assert_eq!(
                wave.chunks
                    .iter()
                    .flat_map(|r| r.clone())
                    .collect::<Vec<_>>(),
                wave.tasks.clone().collect::<Vec<_>>()
            );
        }
        options.mode = GatherMode::Channels;
        let serial = options
            .plan(&work, &expected, 2, &properties, true)
            .unwrap();
        assert_eq!(serial.len(), 1);
        assert_eq!(serial[0].chunks.len(), 1);
        assert_eq!(serial[0].chunks[0], 0..8);
        options.batch_budget_bytes = SOURCE_GATHER_BUDGET_BYTES;
        let parallel = options
            .plan(&work, &expected, 2, &properties, true)
            .unwrap();
        assert_eq!(parallel.len(), 1);
        assert_eq!(parallel[0].chunks.len(), 8);
        let no_pool = options
            .plan(&work, &expected, 2, &properties, false)
            .unwrap();
        assert_eq!(no_pool[0].chunks.len(), 1);
        assert_eq!(no_pool[0].chunks[0], 0..8);
        let small = options
            .plan(&work, &[1000.0; 8], 2, &properties, true)
            .unwrap();
        assert_eq!(small[0].chunks.len(), 1);
        assert_eq!(small[0].chunks[0], 0..8);
    }

    #[test]
    fn gather_hash_budget_charges_both_copies_and_preserves_wave_order() {
        for (property_count, bytes) in [(6, 84), (16, 104), (24, 120)] {
            let properties: Vec<_> = (0..property_count).collect();
            let per_task = 10_000 * bytes;
            let mut options = GatherOptions {
                mode: GatherMode::Waves,
                batch_budget_bytes: 2 * per_task,
                cache_hashes: true,
                ..GatherOptions::default()
            };
            let waves = options
                .plan(&[65_536; 4], &[10_000.0; 4], 2, &properties, true)
                .unwrap();
            assert_eq!(
                waves.iter().map(|w| w.tasks.clone()).collect::<Vec<_>>(),
                [0..2, 2..4]
            );
            for wave in waves {
                assert_eq!(wave.estimated_bytes, (2 * per_task) as f64);
                assert_eq!(wave.chunks.len(), 2);
            }
            options.batch_budget_bytes -= 1;
            let waves = options
                .plan(&[65_536; 4], &[10_000.0; 4], 2, &properties, true)
                .unwrap();
            assert_eq!(
                waves.iter().map(|w| w.tasks.clone()).collect::<Vec<_>>(),
                [0..1, 1..2, 2..3, 3..4]
            );
            // The estimate remains conservative above each collector's cache
            // cap, and the serial path retains every task regardless of budget.
            let large = (1 << 20) as f64 + 1.0;
            let serial = options
                .plan(&[65_536; 4], &[large; 4], 2, &properties, false)
                .unwrap();
            assert_eq!(serial.len(), 1);
            assert_eq!(serial[0].tasks, 0..4);
            assert_eq!(serial[0].chunks.len(), 1);
            assert_eq!(serial[0].chunks[0], 0..4);
            assert_eq!(serial[0].estimated_bytes, 4.0 * large * bytes as f64);
            options.cache_hashes = false;
            let uncached = options
                .plan(&[65_536; 4], &[large; 4], 2, &properties, false)
                .unwrap();
            assert_eq!(
                uncached[0].estimated_bytes,
                4.0 * large * (bytes - 32) as f64
            );
            assert_eq!(uncached[0].tasks, serial[0].tasks);
            assert_eq!(uncached[0].chunks, serial[0].chunks);
        }
    }

    #[test]
    fn gather_limits_and_single_oversized_tasks_do_not_truncate() {
        let mut options = GatherOptions {
            mode: GatherMode::Waves,
            batch_budget_bytes: 1,
            ..GatherOptions::default()
        };
        let waves = options
            .plan(&[65_536; 3], &[10_000.0; 3], 2, &[0, 15], true)
            .unwrap();
        assert_eq!(waves.len(), 3);
        for (i, wave) in waves.iter().enumerate() {
            assert_eq!(wave.tasks, i..i + 1);
            assert_eq!(wave.chunks.len(), 1);
            assert_eq!(wave.chunks[0], i..i + 1);
            assert!(wave.estimated_bytes > 1.0);
        }
        options.workers = 1;
        let serial = options
            .plan(&[65_536; 3], &[10_000.0; 3], 2, &[0, 15], true)
            .unwrap();
        assert_eq!(serial.len(), 1);
        assert_eq!(serial[0].chunks.len(), 1);
        assert_eq!(serial[0].chunks[0], 0..3);
        assert!(
            options
                .plan(&[], &[], 2, &[0, 15], true)
                .unwrap()
                .is_empty()
        );
        for workers in [0, 65] {
            options.workers = workers;
            assert!(options.validate().is_err());
        }
        options.workers = 8;
        options.batch_budget_bytes = 0;
        assert!(options.validate().is_err());
    }

    fn policy() -> SamplingPolicy {
        SamplingPolicy::from_source(&SourcePolicy::default()).unwrap()
    }
    fn view(channel: u32, group: u32, pixels: &[i32]) -> ChannelView<'_> {
        ChannelView {
            channel,
            group,
            width: pixels.len(),
            height: 1,
            stride: pixels.len(),
            pixels,
            is_meta: false,
        }
    }

    #[test]
    fn source_effort_fraction_and_property_tables() {
        let expected = [
            (-1, 3, 32, 0.15),
            (0, 16, 256, 0.65),
            (1, 16, 256, 0.65),
            (2, 10, 128, 0.55),
            (3, 7, 96, 0.5),
            (4, 5, 64, 0.35),
            (5, 4, 48, 0.25),
            (6, 3, 32, 0.15),
            (7, 3, 32, 0.15),
            (8, 3, 32, 0.15),
            (9, 3, 32, 0.15),
        ];
        for (tier, count, bins, fraction) in expected {
            let p = SamplingPolicy::from_source(&SourcePolicy {
                speed_tier: tier,
                ..SourcePolicy::default()
            })
            .unwrap();
            assert_eq!((p.properties.len(), p.max_property_values), (count, bins));
            assert!((p.nb_repeats - fraction).abs() < 1e-7);
            assert_eq!(p.properties.contains(&1), tier <= 1);
            assert_eq!(p.node_threshold, (75 + 14 * i32::from(tier)) as f32);
        }
        let p = SamplingPolicy::from_source(&SourcePolicy {
            responsive: true,
            speed_tier: 5,
            ..SourcePolicy::default()
        })
        .unwrap();
        assert_eq!(p.properties, [0, 1, 4, 5]);
        let p = SamplingPolicy::from_source(&SourcePolicy {
            num_streams: 30,
            speed_tier: 5,
            ..SourcePolicy::default()
        })
        .unwrap();
        assert_eq!(p.properties, [0, 1, 15, 9]);
        let p = SamplingPolicy::from_source(&SourcePolicy {
            learning_fraction: 10.0,
            ..SourcePolicy::default()
        })
        .unwrap();
        assert_eq!(p.nb_repeats, 1.0);
    }

    #[test]
    fn source_reference_caps_modes_and_decode_cost() {
        let mut source = SourcePolicy {
            max_previous_channels: 20,
            num_extra_channels: 1,
            ..SourcePolicy::default()
        };
        let p = SamplingPolicy::from_source(&source).unwrap();
        assert_eq!(p.previous_channels, 3);
        assert!(p.properties.ends_with(&[19, 23, 27]));
        source.speed_tier = 1;
        assert!(
            SamplingPolicy::from_source(&source)
                .unwrap()
                .properties
                .ends_with(&(16..28).collect::<Vec<_>>())
        );
        source.is_gray = true;
        assert_eq!(
            SamplingPolicy::from_source(&source)
                .unwrap()
                .previous_channels,
            1
        );
        source.decoding_speed_tier = 1;
        let p = SamplingPolicy::from_source(&source).unwrap();
        assert!(!p.properties.contains(&15));
        assert_eq!(p.node_threshold, 99.0);
        source.predictor = Predictor::Weighted;
        assert_eq!(
            SamplingPolicy::from_source(&source)
                .unwrap()
                .predictor_override,
            Some(Predictor::Gradient)
        );
        source.decoding_speed_tier = 3;
        assert_eq!(
            SamplingPolicy::from_source(&source).unwrap().properties,
            [9]
        );
        assert_eq!(
            SamplingPolicy::from_source(&source)
                .unwrap()
                .predictor_override,
            Some(Predictor::Gradient)
        );
        source.decoding_speed_tier = 4;
        assert_eq!(
            SamplingPolicy::from_source(&source).unwrap().nb_repeats,
            0.0
        );
    }

    #[test]
    fn tectonic_uses_default_ma_switch_but_full_previous_channel_properties() {
        for responsive in [false, true] {
            for num_streams in [1, 22, 165] {
                let source = SourcePolicy {
                    speed_tier: -1,
                    responsive,
                    num_streams,
                    max_previous_channels: 2,
                    ..Default::default()
                };
                let policy = SamplingPolicy::from_source(&source).unwrap();
                assert_eq!(
                    &policy.properties[..3],
                    if responsive { &[0, 1, 4] } else { &[0, 1, 15] }
                );
                assert_eq!(&policy.properties[3..], &(16..24).collect::<Vec<_>>());
                assert_eq!(policy.previous_channels, 2);
                assert_eq!(policy.max_property_values, 32);
                assert_eq!(policy.nb_repeats, 0.5f32 * 0.3f32);
                assert_eq!(policy.node_threshold, 61.0);
                let decoding = SamplingPolicy::from_source(&SourcePolicy {
                    decoding_speed_tier: 1,
                    ..source
                })
                .unwrap();
                assert_eq!(decoding.node_threshold, 71.0);
                assert!(!decoding.properties.contains(&15));
                assert_eq!(decoding.nb_repeats, policy.nb_repeats);
            }
        }
    }

    #[test]
    fn fraction_floor_disabled_empty_and_streaming_state() {
        assert_eq!(tree_sample_fraction(0, 0.5).unwrap(), 0.0);
        assert_eq!(tree_sample_fraction(20, 0.0).unwrap(), 0.0);
        assert_eq!(tree_sample_fraction(1024, 0.01).unwrap(), 1.0);
        assert_eq!(tree_sample_fraction(4096, 0.01).unwrap(), 0.25);
        assert_eq!(tree_sample_fraction(4096, 0.5).unwrap(), 0.5);
        assert_eq!(
            ChannelSampler::new(32, 32, 0.01)
                .unwrap()
                .filter(|&s| s)
                .count(),
            1024
        );
        assert_eq!(
            ChannelSampler::new(32, 32, 0.0)
                .unwrap()
                .filter(|&s| s)
                .count(),
            0
        );
        let full: Vec<_> = ChannelSampler::new(128, 64, 0.25).unwrap().collect();
        let mut stripe = ChannelSampler::new(128, 64, 0.25).unwrap();
        stripe.skip_pixels(128 * 17).unwrap();
        assert_eq!(stripe.collect::<Vec<_>>(), full[128 * 17..]);
        assert!(ChannelSampler::new(usize::MAX, 2, 0.5).is_err());
    }

    #[test]
    fn training_rng_matches_independent_u128_arithmetic_and_resets_per_channel() {
        let mut a = 0x94d0_49bb_1331_11ebu128;
        let mut b = 0xbf58_476d_1ce4_e5b9u128;
        let mask = (1u128 << 64) - 1;
        let mut rng = SourceRng::new(0);
        for _ in 0..1024 {
            let expected = ((a + b) & mask) as u64;
            let mut next = a ^ ((a << 23) & mask);
            next ^= b ^ (next >> 18) ^ (b >> 5);
            a = b;
            b = next & mask;
            assert_eq!(rng.next(), expected);
        }
        let pixels = vec![0; 4096];
        let channels = [view(4, 1_000_000, &pixels), view(0, 25, &pixels)];
        let mut flags = [Vec::new(), Vec::new()];
        let stats = visit_training_pixels(
            &channels,
            &policy(),
            SampleLimits::default(),
            |site, selected| {
                flags[site.view_index].push(selected);
                assert_eq!(site.group, channels[site.view_index].group);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(flags[0], flags[1]);
        assert_eq!(stats.visited_pixels, 8192);
        assert_eq!(stats.source_total_pixels, 12288);
    }

    #[test]
    fn callback_visits_rejections_but_resource_cap_never_truncates_successfully() {
        let pixels = vec![0; 4096];
        let channels = [view(0, 0, &pixels)];
        let mut calls = 0;
        let result = visit_training_pixels(
            &channels,
            &policy(),
            SampleLimits {
                max_selected_samples: 10,
                ..SampleLimits::default()
            },
            |_, _| {
                calls += 1;
                Ok(())
            },
        );
        assert!(matches!(result, Err(Error::Limit(_))));
        assert!((10..4096).contains(&calls));
        let mut p = policy();
        p.nb_repeats = 0.0;
        let mut calls = 0;
        let stats = visit_training_pixels(&channels, &p, SampleLimits::default(), |_, selected| {
            assert!(!selected);
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 4096);
        assert_eq!(stats.selected_samples, 0);
    }

    #[test]
    fn metadata_width_one_and_oversized_prefix_have_source_eligibility() {
        let mut p = policy();
        p.max_channel_size = 2;
        let wide = [1, 2, 3, 4];
        let narrow = [9];
        let pair = [5, 6];
        let channels = [
            ChannelView {
                is_meta: true,
                ..view(0, 88, &wide)
            },
            view(1, 88, &narrow),
            view(2, 88, &wide),
            view(3, 88, &pair),
            view(0, 99, &pair),
        ];
        let stats = visit_training_pixels(&channels, &p, SampleLimits::default(), |site, _| {
            assert!(!(site.group == 88 && site.channel >= 2));
            Ok(())
        })
        .unwrap();
        assert_eq!(stats.visited_pixels, 7);
        let quantiles = collect_pixel_samples(&channels, &p).unwrap();
        assert_eq!(quantiles.channel_pixel_count.get(&0), Some(&6));
        assert!(!quantiles.channel_pixel_count.contains_key(&1));
        assert_eq!(quantiles.group_pixel_count.get(&88), Some(&4));
        assert_eq!(quantiles.group_pixel_count.get(&99), Some(&2));
    }

    #[test]
    fn geometric_collection_matches_raw_vector_traversal_with_row_padding() {
        let data: Vec<i32> = (0..1000).map(|i| i - 500).collect();
        let channels = [
            ChannelView {
                width: 20,
                height: 25,
                stride: 21,
                ..view(2, 91, &data)
            },
            ChannelView {
                width: 10,
                height: 10,
                stride: 11,
                ..view(5, 91, &data)
            },
        ];
        let p = policy();
        let collected = collect_pixel_samples(&channels, &p).unwrap();
        let mut oracle = PropertySamples::default();
        let mut population = Vec::new();
        for channel in &channels {
            for y in 0..channel.height {
                for x in 0..channel.width {
                    let xp = if x == 0 { 1 } else { x - 1 };
                    population.push((
                        channel.pixels[y * channel.stride + x],
                        channel.pixels[y * channel.stride + x]
                            .wrapping_sub(channel.pixels[y * channel.stride + xp]),
                    ));
                }
            }
        }
        let fraction = (f64::from(p.nb_repeats) * 0.1).min(0.99) as f32;
        let inverse = (1.0f64 / f64::from((1.0f32 - fraction).ln())) as f32;
        let mut rng = SourceRng::new(91);
        let mut position = rng.geometric(inverse) as usize;
        while position < population.len() {
            let (pixel, difference) = population[position];
            oracle.add(pixel, difference).unwrap();
            position += rng.geometric(inverse) as usize + 1;
        }
        assert_eq!(collected.pixel, oracle.pixel);
        assert_eq!(collected.difference, oracle.difference);
        assert_eq!(collected.abs_pixel, oracle.abs_pixel);
        assert!(collected.sampled_pixels > 0);
        assert_eq!(collected.group_pixel_count.get(&91), Some(&600));
    }

    #[test]
    fn quantile_families_share_pixels_differences_and_previous_channel_order() {
        let mut samples = PropertySamples::default();
        for value in [-8, -4, 2, 6] {
            samples.add(value, value * 3).unwrap();
        }
        let q = samples
            .quantization(&[6, 7, 8, 4, 5, 9, 10, 16, 17, 18, 19, 2, 15], 4)
            .unwrap();
        assert_eq!(q.thresholds[0], [-8, -4, 2]);
        assert_eq!(q.thresholds[0], q.thresholds[1]);
        assert_eq!(q.thresholds[0], q.thresholds[2]);
        assert_eq!(q.thresholds[3], [2, 4, 6]);
        assert_eq!(q.thresholds[3], q.thresholds[7]);
        assert_eq!(q.thresholds[0], q.thresholds[8]);
        assert_eq!(q.thresholds[5], [-24, -12, 6]);
        assert_eq!(q.thresholds[9], [6, 12, 18]);
        assert_eq!(q.thresholds[5], q.thresholds[10]);
        assert_eq!(q.thresholds[11], [63, 127, 191]);
        assert_eq!(q.thresholds[12].len(), 15);
    }

    #[test]
    fn quantile_empty_cache_retry_preserves_source_abs_mutation_order() {
        let mut samples = PropertySamples::default();
        samples.add(-1000, 0).unwrap();
        for _ in 0..1000 {
            samples.add(-1, 0).unwrap();
        }
        let q = samples.quantization(&[6, 4, 7], 32).unwrap();
        assert!(q.thresholds[0].is_empty());
        assert_eq!(q.thresholds[1], [1]);
        assert_eq!(q.thresholds[2], [1]);
    }

    #[test]
    fn sparse_static_ids_and_signed_extremes_do_not_alias_or_overflow() {
        let mut samples = PropertySamples::default();
        samples.channel_pixel_count.insert(7, 10);
        samples.channel_pixel_count.insert(800, 10);
        samples.group_pixel_count.insert(1_000_000, 10);
        samples.group_pixel_count.insert(2_000_000, 10);
        samples
            .add(i32::MIN, i32::MAX.wrapping_sub(i32::MIN))
            .unwrap();
        samples.add(0, 0).unwrap();
        let q = samples.quantization(&[0, 1, 6, 4, 9], 256).unwrap();
        assert_eq!(q.thresholds[0], [7]);
        assert_eq!(q.thresholds[1], [1_000_000]);
        assert_eq!(q.thresholds[2], [-512]);
        assert_eq!(q.thresholds[3], [0]);
        assert_eq!(q.thresholds[4], [-1]);
    }

    #[test]
    fn pixel_prepass_narrows_wide_differences_before_quantile_clipping() {
        let pixels: Vec<_> = (0..4096)
            .map(|i| if i % 2 == 0 { i32::MIN } else { i32::MAX })
            .collect();
        let mut channel = view(0, 21, &pixels);
        channel.width = 64;
        channel.height = 64;
        channel.stride = 64;
        let mut policy = policy();
        policy.nb_repeats = 1.0;
        let samples = collect_pixel_samples(&[channel], &policy).unwrap();
        assert!(samples.difference[511] > 0);
        assert!(samples.difference[513] > 0);
        assert_eq!(
            samples.difference[511] + samples.difference[513],
            samples.sampled_pixels
        );
        assert_eq!(samples.abs_difference[513], samples.sampled_pixels);
        assert_eq!(
            samples.pixel[0] + samples.pixel[1024],
            samples.sampled_pixels
        );
        assert_eq!(samples.group_pixel_count.get(&21), Some(&4096));
        let q = samples.quantization(&[6, 9], 32).unwrap();
        assert_eq!(q.thresholds, [vec![-512], vec![-1]]);
    }

    #[test]
    fn policy_installs_quantiles_and_source_fraction_scaled_cost() {
        let p = policy();
        let mut train = TrainOptions::default();
        let statistics = SamplingStats {
            visited_pixels: 1024,
            selected_samples: 1024,
            source_total_pixels: 2048,
        };
        p.apply_to_train(&PropertySamples::default(), statistics, &mut train)
            .unwrap();
        assert_eq!(train.properties, p.properties);
        assert_eq!(
            train.property_quantization.as_ref().unwrap().properties,
            p.properties
        );
        assert_eq!(train.split_threshold, f64::from(p.node_threshold * 0.55f32));
        assert_eq!(train.max_leaves, 64);
    }

    #[test]
    fn invalid_views_probabilities_and_property_counts_are_rejected() {
        for fraction in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(ChannelSampler::new(10, 10, fraction).is_err());
        }
        let pixels = [1, 2];
        let channel = view(0, 0, &pixels);
        assert!(collect_pixel_samples(&[channel, channel], &policy()).is_err());
        assert!(
            collect_pixel_samples(
                &[ChannelView {
                    stride: 1,
                    ..channel
                }],
                &policy()
            )
            .is_err()
        );
        assert!(
            PropertySamples::default()
                .quantization(&[6, 6], 32)
                .is_err()
        );
        assert!(PropertySamples::default().quantization(&[0], 257).is_err());
        for speed_tier in [i8::MIN, -2, 10, i8::MAX] {
            assert!(
                SamplingPolicy::from_source(&SourcePolicy {
                    speed_tier,
                    ..SourcePolicy::default()
                })
                .is_err()
            );
        }
        let mut p = policy();
        p.nb_repeats = f32::MIN_POSITIVE;
        assert!(collect_pixel_samples(&[channel], &p).is_err());
    }
}
