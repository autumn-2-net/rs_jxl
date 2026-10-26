// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Adaptive MA learning from the frozen libjxl b866b3defec13494bfbaf705db93581cda118e37
//! fork: enc_ma.h/cc::TreeSamples, QuantizeHistogram, QuantizeSamples,
//! PreQuantizeProperties, CollectExtraBitsIncrease, FindBestSplitImpl,
//! SplitTreeSamples, and ComputeBestTree. Source-default f32 probability-floored
//! scoring and fused histogram moves reuse native SIMD in tools::ma_simd.
//!
//! The caller samples pixels and computes EVERY predictor in `predictors` order,
//! including Weighted with its full raster-order state. No further sampling is
//! done here. Weights are multiplicities, not inverse sampling probabilities.
//! Deduplication compares quantized properties and (token, extra-bit COUNT), not
//! raw residuals or extra-bit payloads, exactly as TreeSamples::IsSameSample.
//! Residual samples retain the low 32 bits of the wide predictor difference,
//! matching TreeSamples::AddSample. They are not wide arithmetic evidence for
//! multiplier divisibility; the encoder must check that before narrowing.
//!
//! Supply sample::collect_pixel_samples's PropertyQuantization for the source
//! shared pixel/difference populations and whole-image static populations.
//! Omitting it retains the compatibility fallback of per-property weighted
//! sample quantiles. Clipping, coordinate bins, WP bins, and strict `>`
//! boundaries are retained. External u32 weights split into source saturated
//! u16 records without changing histogram multiplicity. The root keeps predictor
//! index 0 in the supplied order, as ComputeBestTree does. A leaf budget adds a
//! deterministic stop to the source's greater-child-first depth-first search.
//!
//! Multiplier ranges use enc_ma.cc::BoxIntersects and its forced static splits.
//! Constant-channel offsets use enc_encoding.cc::LearnTree's full-plane scan
//! and trailing-channel splice. The source does NOT fit general offsets or GCD
//! multipliers from residual samples: enc_modular.cc::ComputeTree supplies the
//! latter from channel quantization metadata. No lossy quantization is done here.
//! Constraints require full-population evidence from the caller; training samples
//! cannot prove divisibility or constancy. See find_constant_channels.
//! Narrowed samples can disprove divisibility only by the multiplier's power-of-2
//! factor; odd factors need the original wide differences at encoding time.
//!
//! Mandatory splits are built before adaptive splits to reserve their leaf
//! budget, including unsampled ranges. Static constraint thresholds are exact,
//! not clipped to 511 or narrowed to byte-sized bin indices. Adaptive scoring
//! retains the source's original residuals (not residual/multiplier histograms).
//!
//! Reuses bounded per-property histograms down the larger child, subtracts
//! smaller-sibling parent histograms, and evaluates ready children concurrently
//! before committing their decisions in the original depth-first order.
//! enc_ma_histograms.h's single-pass parent collection, lane-preserving small-
//! node compaction, and uncached adjacent-predictor collection are active.
//! Leaf termination only collects the current predictor; cache population and
//! remaining parent histograms use the existing bounded execution API.
//! Initial cache reservation precedes dispatch; each property task fills and
//! scores its own entry without a separate population barrier. Only successful
//! sweeps publish the cache, including after deferred inherited subtraction.
//! Invocation-owned scratch leases retain buffers across node dispatches without
//! thread-local aliases. Inherited cache subtraction runs inside each property's
//! scoring task; pending sibling ranges are consumed once, not copied eagerly.
//! Source-sized batches of 128 swap pairs move all prepared sample columns at
//! each split, preserving their alignment and subsequent contiguous traversal.
//! Dynamic quantizers with at most 256 bins use byte columns; static and wider
//! legacy quantizers retain u32 columns. Property tasks dispatch once to typed
//! collection/subtraction/scoring loops without widening sample buffers.
//! Prepared samples use the source default two-choice index table, rebuilt from
//! empty slots on resize and excluding saturated records. Collision misses stay
//! as records; no global key map, weight clamp or dropped sample.
//! Optional predictor deduplication verifies equal penalties, parent counts,
//! weighted extra bits and every residual token/count pair before reusing an
//! earlier predictor. Node-local proofs are rebuilt after each split; copied
//! cache entries remain usable when a child has different predictor penalties.
//! Optional property deduplication verifies every live quantized value after
//! matching marginal signatures and static/WP classes. Serial sweeps inherit
//! verified redundant-property masks; parallel sweeps only add constant masks.
//! Independent stream sharding and property-1 merge live in tree_groups.
//! Optional histogram shards, padded/dense layouts, packed property bins,
//! sparse entropy blocks and paired cache subtraction preserve weighted mass.
//! Optional source count-log, Shannon, clipped/cursor, description and
//! race/quantile/refinement modes are explicit typed controls, disabled by
//! default. Channel roots and optional whole subtrees use disjoint column loans
//! on the caller's bounded cooperative pool. The default is
//! the full property/predictor histogram sweep, not a fixed-tree heuristic.
//! Opt-in live subtree forks and histogram handoff share one cache accounting
//! domain, without sharing mutable sample columns or limiting tree learning.

#[cfg(test)]
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use crate::execution;

use crate::entropy::HybridUintConfig;
use crate::tools::ma_simd::{DefaultScorer, HistogramSplit, MAX_HISTOGRAM_TOTAL};
use crate::{Error, Result};

use super::predictor::Predictor;
use super::tree::{Node, Tree, pack_signed};

#[path = "ma_histograms.rs"]
mod ma_histograms;
pub use ma_histograms::{DenseHistogram, PropertyBins};
use ma_histograms::{
    HistogramShards, SparseHistogramPlan, collect_compact_pair, collect_mapped, histogram_stride,
    prepare_property_bins,
};

#[path = "ma_events.rs"]
mod ma_events;
use ma_events::EventScratch;

#[path = "ma_dominance.rs"]
mod ma_dominance;
#[path = "ma_partition_proofs.rs"]
mod ma_partition_proofs;

#[path = "ma_columns.rs"]
mod columns;
use columns::{PropertyColumn, PropertyValue, swap_column};

#[path = "ma_ranges.rs"]
mod ma_ranges;
use ma_ranges::{SampleStorage, Samples, TreeSamples};

#[path = "ma_samples.rs"]
mod sample_storage;

#[path = "ma_prepared.rs"]
mod prepared_storage;

#[path = "ma_scoring.rs"]
mod ma_scoring;
use ma_scoring::ScoreScratch;

#[path = "ma_binning.rs"]
mod ma_binning;
#[cfg(test)]
#[path = "ma_binning_tests.rs"]
mod ma_binning_tests;
#[path = "ma_search.rs"]
mod ma_search;
pub use ma_scoring::{ClippedEntropyMode, DescriptionCost, ScoringOptions};
pub use ma_search::{QuantileSplits, SearchOptions};

#[path = "ma_channel.rs"]
mod ma_channel;
pub use ma_channel::ChannelTreeMode;

#[path = "ma_parallel.rs"]
mod ma_parallel;
pub use ma_parallel::ParallelOptions;

#[path = "ma_train_options.rs"]
mod train_options;
pub use train_options::{HistogramOptions, MaBudgetMode, TrainOptions};

#[path = "ma_frontier.rs"]
mod ma_frontier;

#[cfg(test)]
#[path = "ma_frontier_tests.rs"]
mod frontier_tests;

#[cfg(test)]
#[path = "ma_handoff_tests.rs"]
mod handoff_tests;

// Monomorphize the entire property task, not a per-record enum accessor.
macro_rules! with_property_values {
    ($column:expr, |$values:ident| $body:block) => {
        match $column {
            PropertyColumn::Packed($values) => $body,
            PropertyColumn::Wide($values) => $body,
        }
    };
}

const PROPERTY_RANGE: i32 = 511;
const QUANTILE_RANGE: i32 = 512;
const WP_PROPERTY: usize = 15;
/// Maximum representable tree leaves, without the former 16,384 resource cap.
/// Raw MA contexts may exceed 256; entropy clustering assigns final histogram IDs.
pub const MAX_LEAVES: usize = MAX_TREE_LEAVES;

/// enc_ma.cc/dec_ma.cc kMaxTreeSize counts nodes, not leaf contexts.
pub const MAX_TREE_NODES: usize = 1 << 22;
pub const MAX_TREE_LEAVES: usize = MAX_TREE_NODES / 2;

/// dec_modular.cc's global-tree bound uses the original frame dimensions,
/// before palette/squeeze transforms. Wide arithmetic also handles large metadata.
pub fn global_tree_leaf_limit(width: usize, height: usize, channels: usize) -> usize {
    let samples = (width as u128)
        .saturating_mul(height as u128)
        .saturating_mul(channels as u128);
    let nodes = (1024 + samples / 16).min(MAX_TREE_NODES as u128) as usize;
    nodes.div_ceil(2)
}

pub(super) fn local_tree_leaf_limit(samples: usize) -> usize {
    let nodes = samples.saturating_add(1024).min(1 << 20);
    nodes.div_ceil(2)
}

fn cache_pair14_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RS_JXL_MA_CACHE_PAIR14").as_deref() == Ok("1"))
}

// Per-predictor timers can dominate small sweeps. RS_JXL_PROFILE=1 keeps
// broad/cache timings; these detail scopes additionally require
// RS_JXL_PROFILE_DETAIL=1, read once before the first detail scope.
fn histogram_detail_scope(name: &'static str) -> Option<crate::profile::Scope> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled =
        *ENABLED.get_or_init(|| std::env::var("RS_JXL_PROFILE_DETAIL").as_deref() == Ok("1"));
    (enabled && crate::profile::enabled()).then(|| crate::profile::scope(name))
}

#[derive(Clone, Debug)]
pub struct Sample {
    /// Indexed by the context_predict.h property ID, not options.properties.
    pub properties: Vec<i32>,
    /// `(i64::from(pixel) - prediction) as i32`, in TrainOptions.predictors order.
    /// Source AddSample narrows before tokenization, including on overflow.
    /// Do not use this narrowed value to divide a wide encoding residual.
    pub residuals: Vec<i32>,
    pub weight: u32,
}

/// Half-open ranges of static properties 0 (channel) and 1 (stream/group ID).
/// Endpoints may reach i32::MAX + 1, but every represented ID must fit i32.
pub type StaticPropRange = [[u32; 2]; 2];

/// enc_ma.h/options.h::ModularMultiplierInfo. The caller guarantees exact
/// residual divisibility throughout this rectangle, not merely at sample sites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiplierInfo {
    pub range: StaticPropRange,
    pub multiplier: u32,
}

/// A full-population-proven constant, not a constant inferred from MA samples.
/// TrainOptions requires an ascending contiguous suffix of the channel range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstantChannel {
    pub channel: u32,
    pub value: i32,
}

/// PreQuantizeProperties output in exactly the selected property order.
/// Each sorted threshold is an inclusive upper bound of its quantized bin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PropertyQuantization {
    pub properties: Vec<usize>,
    pub thresholds: Vec<Vec<i32>>,
}

/// Read-only visible channel plane. Supply every present channel/group view,
/// including metadata channels and channels too large for this stream.
#[derive(Clone, Copy, Debug)]
pub struct ChannelView<'a> {
    pub channel: u32,
    pub group: u32,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub pixels: &'a [i32],
    pub is_meta: bool,
}

/// Port of enc_encoding.cc::LearnTree's constant trailing-channel scan.
/// Missing/empty channel instances are skipped, but a channel absent everywhere
/// ends the suffix. Metadata collisions and oversized instances prevent bypass.
/// Different-sized planes and row padding are supported; every visible pixel
/// is checked. The caller must supply the complete group population in `range`.
pub fn find_constant_channels(
    channels: &[ChannelView<'_>],
    range: StaticPropRange,
    max_channel_size: usize,
) -> Result<Vec<ConstantChannel>> {
    validate_range(range)?;
    let mut seen = std::collections::HashSet::new();
    for view in channels {
        if !range_contains(range, [view.channel, view.group])
            || !seen.insert((view.channel, view.group))
        {
            return Err(Error::InvalidInput(
                "invalid or duplicate MA channel view".into(),
            ));
        }
        if view.width == 0 || view.height == 0 {
            continue;
        }
        let end = (view.height - 1)
            .checked_mul(view.stride)
            .and_then(|offset| offset.checked_add(view.width))
            .ok_or(Error::Limit("MA channel view dimensions overflow"))?;
        if view.stride < view.width || end > view.pixels.len() {
            return Err(Error::InvalidInput(
                "invalid MA channel view buffer/stride".into(),
            ));
        }
    }
    let mut constants = Vec::new();
    for channel in (range[0][0]..range[0][1]).rev() {
        let mut value = None;
        let mut constant = true;
        for view in channels.iter().filter(|view| view.channel == channel) {
            if view.width == 0 || view.height == 0 {
                continue;
            }
            if view.is_meta || view.width > max_channel_size || view.height > max_channel_size {
                constant = false;
                break;
            }
            let expected = *value.get_or_insert(view.pixels[0]);
            for y in 0..view.height {
                let row = &view.pixels[y * view.stride..y * view.stride + view.width];
                if row.iter().any(|&pixel| pixel != expected) {
                    constant = false;
                    break;
                }
            }
            if !constant {
                break;
            }
        }
        match (constant, value) {
            (true, Some(value)) => constants.push(ConstantChannel { channel, value }),
            _ => break,
        }
    }
    constants.reverse();
    Ok(constants)
}

fn validate_range(range: StaticPropRange) -> Result<()> {
    if range
        .iter()
        .any(|&[begin, end]| begin >= end || end > i32::MAX as u32 + 1)
    {
        return Err(Error::InvalidInput(
            "invalid MA static property range".into(),
        ));
    }
    Ok(())
}

fn range_contains(range: StaticPropRange, point: [u32; 2]) -> bool {
    (0..2).all(|axis| range[axis][0] <= point[axis] && point[axis] < range[axis][1])
}

fn intersect_ranges(a: StaticPropRange, b: StaticPropRange) -> Option<StaticPropRange> {
    let intersection = [
        [a[0][0].max(b[0][0]), a[0][1].min(b[0][1])],
        [a[1][0].max(b[1][0]), a[1][1].min(b[1][1])],
    ];
    intersection
        .iter()
        .all(|&[begin, end]| begin < end)
        .then_some(intersection)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Intersection {
    None,
    Inside,
    Partial { axis: usize, threshold: i32 },
}

/// enc_ma.cc::BoxIntersects. The last partially intersecting axis wins.
fn box_intersects(needle: StaticPropRange, haystack: StaticPropRange) -> Intersection {
    let mut result = Intersection::Inside;
    for axis in 0..2 {
        if haystack[axis][0] >= needle[axis][1] || haystack[axis][1] <= needle[axis][0] {
            return Intersection::None;
        }
        if haystack[axis][0] <= needle[axis][0] && haystack[axis][1] >= needle[axis][1] {
            continue;
        }
        let boundary = if haystack[axis][0] > needle[axis][0] {
            haystack[axis][0]
        } else {
            haystack[axis][1]
        };
        result = Intersection::Partial {
            axis,
            threshold: (boundary - 1) as i32,
        };
    }
    result
}

fn validate_samples<'a>(
    samples: impl IntoIterator<Item = &'a Sample>,
    options: &TrainOptions,
) -> Result<()> {
    let mut total = 0u32;
    for sample in samples {
        if sample.residuals.len() != options.predictors.len()
            || options
                .properties
                .iter()
                .any(|&p| p >= sample.properties.len())
        {
            return Err(Error::InvalidInput(
                "MA sample does not match predictors/properties".into(),
            ));
        }
        if let Some(range) = options.static_property_range
            && sample.weight != 0
            && (sample.properties.len() < 2
                || sample.properties[0] < 0
                || sample.properties[1] < 0
                || !range_contains(
                    range,
                    [sample.properties[0] as u32, sample.properties[1] as u32],
                ))
        {
            return Err(Error::InvalidInput(
                "MA sample is outside the static property range".into(),
            ));
        }
        total = total
            .checked_add(sample.weight)
            .filter(|&total| total <= MAX_HISTOGRAM_TOTAL)
            .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
    }
    Ok(())
}

fn constraint_thresholds(options: &TrainOptions, property: usize) -> Option<Vec<i32>> {
    if property >= 2 {
        return None;
    }
    let range = options.static_property_range?;
    let mut thresholds = Vec::new();
    for info in &options.multiplier_info {
        let Some(overlap) = intersect_ranges(range, info.range) else {
            continue;
        };
        if overlap[property][0] != range[property][0] {
            thresholds.push((overlap[property][0] - 1) as i32);
        }
        if overlap[property][1] != range[property][1] {
            thresholds.push((overlap[property][1] - 1) as i32);
        }
    }
    thresholds.sort_unstable();
    thresholds.dedup();
    (!thresholds.is_empty()).then_some(thresholds)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ResidualToken {
    token: u8,
    nbits: u8,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SampleKey {
    properties: Vec<u32>,
    residuals: Vec<ResidualToken>,
}

#[cfg(test)]
fn quantized_key(
    sample: &Sample,
    properties: &[usize],
    thresholds: &[Vec<i32>],
    exact_static: bool,
    config: &HybridUintConfig,
) -> SampleKey {
    let mut key = SampleKey {
        properties: Vec::new(),
        residuals: Vec::new(),
    };
    quantized_key_into(
        &mut key,
        sample,
        properties,
        thresholds,
        exact_static,
        config,
    );
    key
}

fn quantized_key_into(
    key: &mut SampleKey,
    sample: &Sample,
    properties: &[usize],
    thresholds: &[Vec<i32>],
    exact_static: bool,
    config: &HybridUintConfig,
) {
    key.properties.clear();
    key.properties
        .extend(properties.iter().enumerate().map(|(p, &id)| {
            let value = if id < 2 && exact_static {
                sample.properties[id]
            } else {
                sample.properties[id].clamp(-PROPERTY_RANGE, PROPERTY_RANGE)
            };
            thresholds[p].partition_point(|&threshold| value > threshold) as u32
        }));
    key.residuals.clear();
    key.residuals
        .extend(sample.residuals.iter().map(|&residual| {
            let encoded = config.encode(pack_signed(residual));
            ResidualToken {
                token: encoded.token as u8,
                nbits: encoded.nbits as u8,
            }
        }));
}

/// TreeSamples::FlushSampleBatch/HashBatch/AddWeightedToTableAndMerge. The
/// cache is two-choice and bounded: a collision retains the sample, never loses
/// it. Quantization is shared across all waves/streams. Source 16-bit saturation
/// is retained here and by the prepared TreeSamples count columns.
/// This collector is for unconstrained gather only: it must not hide raw
/// residual divisibility checks required by caller-supplied multipliers.
#[cfg(test)]
struct SampleBatchCollector {
    properties: Vec<usize>,
    thresholds: Vec<Vec<i32>>,
    predictors: usize,
    config: HybridUintConfig,
    pending: Vec<Sample>,
    samples: Vec<Sample>,
    keys: Vec<SampleKey>,
    table: Vec<u32>,
    total: u32,
}

#[cfg(test)]
impl SampleBatchCollector {
    const BATCH: usize = 256;
    const MAX_SLOTS: usize = 1 << 20;
    const UNUSED: u32 = u32::MAX;
    const MAX_WEIGHT: u32 = u16::MAX as u32;

    pub(super) fn new(options: &TrainOptions) -> Result<Self> {
        options.validate()?;
        if !options.multiplier_info.is_empty() || !options.constant_channels.is_empty() {
            return Err(Error::InvalidInput(
                "MA batch gather requires unconstrained variable samples".into(),
            ));
        }
        let quantization = options.property_quantization.as_ref().ok_or_else(|| {
            Error::InvalidInput("MA batch gather requires shared population quantiles".into())
        })?;
        Ok(Self {
            properties: options.properties.clone(),
            thresholds: quantization.thresholds.clone(),
            predictors: options.predictors.len(),
            config: HybridUintConfig::new(4, 1, 2).map_err(|e| Error::Encoding(e.to_string()))?,
            pending: Vec::with_capacity(Self::BATCH),
            samples: Vec::new(),
            keys: Vec::new(),
            table: Vec::new(),
            total: 0,
        })
    }

    pub(super) fn push(&mut self, sample: Sample) -> Result<()> {
        if sample.residuals.len() != self.predictors
            || self
                .properties
                .iter()
                .any(|&p| p >= sample.properties.len())
        {
            return Err(Error::InvalidInput(
                "invalid MA batch sample dimensions".into(),
            ));
        }
        self.total = self
            .total
            .checked_add(sample.weight)
            .ok_or(Error::Limit("MA sample weight overflow"))?;
        if sample.weight == 0 {
            return Ok(());
        }
        self.pending.push(sample);
        if self.pending.len() == Self::BATCH {
            self.flush()?;
        }
        Ok(())
    }

    fn positions(key: &SampleKey, mask: usize) -> [usize; 2] {
        const A: u64 = 0x1e35_a7bd;
        const B: u64 = 0x1e35_a7bd_1e35_a7bd;
        let (mut first, mut second) = (A, B);
        for r in &key.residuals {
            first = first
                .wrapping_mul(A)
                .wrapping_add(u64::from(r.token))
                .wrapping_mul(A)
                .wrapping_add(u64::from(r.nbits));
        }
        for &property in &key.properties {
            first = first.wrapping_mul(A).wrapping_add(u64::from(property));
            second = second.wrapping_mul(B) ^ u64::from(property);
        }
        for r in &key.residuals {
            second =
                (second.wrapping_mul(B) ^ u64::from(r.token)).wrapping_mul(B) ^ u64::from(r.nbits);
        }
        [
            (first >> 16) as usize & mask,
            (second >> 16) as usize & mask,
        ]
    }

    fn hash_batch(keys: &[SampleKey]) -> Vec<(u64, u64)> {
        const A: u64 = 0x1e35_a7bd;
        const B: u64 = 0x1e35_a7bd_1e35_a7bd;
        let mut hashes = vec![(A, B); keys.len()];
        let Some(first) = keys.first() else {
            return hashes;
        };
        for predictor in 0..first.residuals.len() {
            for (key, hash) in keys.iter().zip(&mut hashes) {
                let r = key.residuals[predictor];
                hash.0 = hash
                    .0
                    .wrapping_mul(A)
                    .wrapping_add(u64::from(r.token))
                    .wrapping_mul(A)
                    .wrapping_add(u64::from(r.nbits));
            }
        }
        for property in 0..first.properties.len() {
            for (key, hash) in keys.iter().zip(&mut hashes) {
                let value = u64::from(key.properties[property]);
                hash.0 = hash.0.wrapping_mul(A).wrapping_add(value);
                hash.1 = hash.1.wrapping_mul(B) ^ value;
            }
        }
        for predictor in 0..first.residuals.len() {
            for (key, hash) in keys.iter().zip(&mut hashes) {
                let r = key.residuals[predictor];
                hash.1 = (hash.1.wrapping_mul(B) ^ u64::from(r.token)).wrapping_mul(B)
                    ^ u64::from(r.nbits);
            }
        }
        hashes
    }

    fn grow_table(&mut self, additional: usize) {
        let target = self
            .samples
            .len()
            .saturating_add(additional)
            .saturating_mul(3)
            .div_ceil(2)
            .clamp(512, Self::MAX_SLOTS)
            .next_power_of_two();
        if target <= self.table.len() {
            return;
        }
        // Rebuild from empty slots so saturation cannot leave a stale alias.
        self.table = vec![Self::UNUSED; target];
        for index in 0..self.samples.len() {
            if self.samples[index].weight == Self::MAX_WEIGHT {
                continue;
            }
            for position in Self::positions(&self.keys[index], target - 1) {
                if self.table[position] == Self::UNUSED {
                    self.table[position] = index as u32;
                    break;
                }
            }
        }
    }

    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.grow_table(self.pending.len());
        let keys: Vec<_> = self
            .pending
            .iter()
            .map(|sample| {
                quantized_key(
                    sample,
                    &self.properties,
                    &self.thresholds,
                    true,
                    &self.config,
                )
            })
            .collect();
        let hashes = Self::hash_batch(&keys);
        let pending = std::mem::replace(&mut self.pending, Vec::with_capacity(Self::BATCH));
        let mask = self.table.len() - 1;
        for ((mut sample, key), (first, second)) in pending.into_iter().zip(keys).zip(hashes) {
            let positions = [
                (first >> 16) as usize & mask,
                (second >> 16) as usize & mask,
            ];
            for &position in &positions {
                let previous = self.table[position];
                if previous == Self::UNUSED || self.keys[previous as usize] != key {
                    continue;
                }
                let previous = previous as usize;
                let take = sample
                    .weight
                    .min(Self::MAX_WEIGHT - self.samples[previous].weight);
                self.samples[previous].weight += take;
                sample.weight -= take;
                if self.samples[previous].weight == Self::MAX_WEIGHT {
                    for &slot in &positions {
                        if self.table[slot] == previous as u32 {
                            self.table[slot] = Self::UNUSED;
                        }
                    }
                }
                if sample.weight == 0 {
                    break;
                }
            }
            while sample.weight != 0 {
                let weight = sample.weight.min(Self::MAX_WEIGHT);
                sample.weight -= weight;
                let index = self.samples.len();
                self.samples
                    .try_reserve(1)
                    .map_err(|_| Error::Limit("MA batch samples allocation"))?;
                self.keys
                    .try_reserve(1)
                    .map_err(|_| Error::Limit("MA batch keys allocation"))?;
                if weight != Self::MAX_WEIGHT {
                    for &position in &positions {
                        if self.table[position] == Self::UNUSED {
                            self.table[position] = index as u32;
                            break;
                        }
                    }
                }
                if sample.weight == 0 {
                    sample.weight = weight;
                    self.samples.push(sample);
                    self.keys.push(key);
                    break;
                }
                self.samples.push(Sample {
                    weight,
                    ..sample.clone()
                });
                self.keys.push(key.clone());
            }
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Vec<Sample>> {
        self.flush()?;
        debug_assert_eq!(
            self.samples
                .iter()
                .map(|s| u64::from(s.weight))
                .sum::<u64>(),
            u64::from(self.total)
        );
        Ok(self.samples)
    }
}

impl TreeSamples {
    fn new<'a, I>(samples: I, options: &TrainOptions) -> Result<Self>
    where
        I: IntoIterator<Item = &'a Sample>,
        I::IntoIter: Clone,
    {
        let _scope = crate::profile::scope("ma.prepare_samples");
        let samples = samples.into_iter();
        validate_samples(samples.clone(), options)?;
        let thresholds: Vec<_> = options
            .properties
            .iter()
            .enumerate()
            .map(|(index, &property)| {
                constraint_thresholds(options, property)
                    .or_else(|| {
                        options
                            .property_quantization
                            .as_ref()
                            .map(|q| q.thresholds[index].clone())
                    })
                    .unwrap_or_else(|| {
                        property_thresholds(samples.clone(), property, options.max_property_values)
                    })
            })
            .collect();
        let mut result = Self {
            properties: options
                .properties
                .iter()
                .zip(&thresholds)
                .map(|(&property, thresholds)| PropertyColumn::new(property, thresholds.len()))
                .collect(),
            residuals: vec![Vec::new(); options.predictors.len()],
            counts: Vec::new(),
            thresholds,
        };
        let config = HybridUintConfig::new(4, 1, 2).map_err(|e| Error::Encoding(e.to_string()))?;
        let capacity = sample_storage::split_record_capacity(samples.clone())?;
        let mut dedup = sample_storage::DedupTable::new(&options.properties);
        dedup.prepare(&mut result, capacity)?;
        let mut key = SampleKey {
            properties: Vec::with_capacity(options.properties.len()),
            residuals: Vec::with_capacity(options.predictors.len()),
        };
        let mut input_records = 0usize;
        for sample in samples {
            input_records += 1;
            if sample.weight == 0 {
                continue;
            }
            quantized_key_into(
                &mut key,
                sample,
                &options.properties,
                &result.thresholds,
                !options.multiplier_info.is_empty() || options.property_quantization.is_some(),
                &config,
            );
            dedup.insert(&mut result, &key, sample.weight)?;
        }
        crate::profile::note(
            "ma.sample_counts",
            format_args!(
                "input_records={} distinct={} predictors={} properties={}",
                input_records,
                result.counts.len(),
                result.residuals.len(),
                result.properties.len(),
            ),
        );
        Ok(result)
    }
}

/// QuantizeHistogram, including skipping repeated quantiles and removing the
/// terminal threshold. u64 arithmetic retains exact sample multiplicities.
fn quantize_histogram(histogram: &[u64], num_chunks: usize, bias: i32) -> Vec<i32> {
    let sum: u64 = histogram.iter().sum();
    if sum == 0 {
        return Vec::new();
    }
    let mut thresholds = Vec::new();
    let mut cumsum = 0u64;
    let mut threshold = 1u64;
    for (i, &count) in histogram.iter().enumerate() {
        cumsum += count;
        if cumsum * num_chunks as u64 >= threshold * sum {
            thresholds.push(i as i32 + bias);
            while cumsum * num_chunks as u64 >= threshold * sum {
                threshold += 1;
            }
        }
    }
    thresholds.pop();
    thresholds
}

fn property_thresholds<'a>(
    samples: impl IntoIterator<Item = &'a Sample>,
    property: usize,
    max_values: usize,
) -> Vec<i32> {
    if property == 2 || property == 3 {
        return (1..max_values)
            .map(|i| (i * 256 / max_values) as i32 - 1)
            .collect();
    }
    if property == WP_PROPERTY {
        return if max_values < 32 {
            vec![-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127]
        } else if max_values < 64 {
            vec![
                -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -5, -3, -1, 0, 1, 3, 5, 7,
                11, 15, 23, 31, 47, 63, 95, 127, 191, 255,
            ]
        } else {
            vec![
                -255, -223, -191, -159, -127, -111, -95, -79, -63, -55, -47, -39, -31, -27, -23,
                -19, -15, -13, -11, -9, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11,
                13, 15, 19, 23, 27, 31, 39, 47, 55, 63, 79, 95, 111, 127, 159, 191, 223, 255,
            ]
        };
    }
    let range = if property < 2 {
        PROPERTY_RANGE
    } else {
        QUANTILE_RANGE
    };
    let mut histogram = vec![0u64; (2 * range + 1) as usize];
    for sample in samples {
        let value = sample.properties[property].clamp(-range, range);
        histogram[(value + range) as usize] += u64::from(sample.weight);
    }
    quantize_histogram(&histogram, max_values, -range)
}

#[derive(Clone, Debug, PartialEq)]
struct ParentHistogram {
    counts: Vec<u32>,
    extra_bits: u64,
}

fn parent_histograms(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
) -> Vec<ParentHistogram> {
    (0..samples.residuals.len())
        .map(|predictor| parent_histogram(samples, indices, predictor))
        .collect()
}

/// enc_ma_histograms.h::CollectMaParentHistogram: tokens fit a fixed 256-bin
/// accumulator, so finding the highest symbol needs no separate sample scan.
fn parent_histogram(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    predictor: usize,
) -> ParentHistogram {
    let mut counts = [0u32; 256];
    let mut extra_bits = 0;
    let tokens = &samples.residuals[predictor];
    for &i in indices {
        let token = tokens[i];
        let weight = u32::from(samples.counts[i]);
        counts[usize::from(token.token)] += weight;
        extra_bits += u64::from(token.nbits) * u64::from(weight);
    }
    let symbols = counts
        .iter()
        .rposition(|&count| count != 0)
        .map_or(1, |s| s + 1);
    ParentHistogram {
        counts: counts[..symbols].to_vec(),
        extra_bits,
    }
}

struct CompactHistograms {
    parents: Vec<ParentHistogram>,
    mapping: [u8; 256],
}

/// Source CompactMaHistograms's default, not its opt-in dense remapping.
/// The union of parent-active tokens preserves every possible child token.
#[cfg(test)]
fn compact_parent_histograms(
    parents: &[ParentHistogram],
    lanes: usize,
) -> Option<CompactHistograms> {
    compact_parent_histograms_with_layout(parents, lanes, DenseHistogram::Disabled, false)
}

fn compact_parent_histograms_with_layout(
    parents: &[ParentHistogram],
    lanes: usize,
    dense: DenseHistogram,
    padded: bool,
) -> Option<CompactHistograms> {
    if parents.is_empty() || !matches!(lanes, 1 | 4 | 8) {
        return None;
    }
    let symbols = parents.iter().map(|p| p.counts.len()).max()?;
    let stride = histogram_stride(symbols, lanes, padded);
    if !(32..=256).contains(&stride) {
        return None;
    }
    let mut used = [false; 256];
    for parent in parents {
        for (symbol, &count) in parent.counts.iter().enumerate() {
            used[symbol] |= count != 0;
        }
    }
    let mut next = [0usize; 8];
    let mut mapping = [0u8; 256];
    let mut compact_stride = 0;
    let mut dense_symbols = 0usize;
    for symbol in 0..stride {
        if used[symbol] {
            let lane = symbol % lanes;
            if dense != DenseHistogram::Disabled {
                mapping[symbol] = dense_symbols as u8;
                dense_symbols += 1;
                compact_stride = dense_symbols.div_ceil(lanes) * lanes;
            } else {
                mapping[symbol] = (next[lane] * lanes + lane) as u8;
                next[lane] += 1;
                compact_stride = compact_stride.max(next[lane] * lanes);
            }
        }
    }
    if compact_stride == 0 || compact_stride * 4 > stride * 3 {
        return None;
    }
    let parents = parents
        .iter()
        .map(|parent| {
            let mut counts = vec![0; compact_stride];
            for (symbol, &count) in parent.counts.iter().enumerate() {
                if used[symbol] {
                    counts[usize::from(mapping[symbol])] = count;
                }
            }
            // The default trim-entropy path is independent of storage stride.
            while counts.len() > 1 && counts.last() == Some(&0) {
                counts.pop();
            }
            ParentHistogram {
                counts,
                extra_bits: parent.extra_bits,
            }
        })
        .collect();
    Some(CompactHistograms { parents, mapping })
}

fn subtract_parent_histograms(
    mut parent: Vec<ParentHistogram>,
    smaller: &[ParentHistogram],
) -> Result<Vec<ParentHistogram>> {
    for (parent, smaller) in parent.iter_mut().zip(smaller) {
        for (symbol, &count) in smaller.counts.iter().enumerate() {
            parent.counts[symbol] = parent.counts[symbol].checked_sub(count).ok_or_else(|| {
                Error::Encoding("MA parent histogram subtraction underflow".into())
            })?;
        }
        parent.extra_bits = parent
            .extra_bits
            .checked_sub(smaller.extra_bits)
            .ok_or_else(|| Error::Encoding("MA parent extra-bit subtraction underflow".into()))?;
        while parent.counts.len() > 1 && parent.counts.last() == Some(&0) {
            parent.counts.pop();
        }
    }
    Ok(parent)
}

#[derive(Default)]
struct TrainingStats {
    detached_subtrees: AtomicUsize,
    dynamic_forks: AtomicUsize,
    histogram_handoffs: AtomicUsize,
    channel_roots: AtomicUsize,
    race_nodes: AtomicUsize,
    race_draws: AtomicUsize,
    race_pruned: AtomicUsize,
    compact_nodes: AtomicUsize,
    paired_cache_subtractions: AtomicUsize,
    parent_reuses: AtomicUsize,
    cache_builds: AtomicUsize,
    cache_reuses: AtomicUsize,
    cache_subtractions: AtomicUsize,
    ready_nodes: AtomicUsize,
    equivalent_predictors: AtomicUsize,
    constant_properties: AtomicUsize,
    duplicate_properties: AtomicUsize,
    inherited_properties: AtomicUsize,
    dominated_predictors: AtomicUsize,
    #[cfg(test)]
    initial_cache_hook: Option<Box<InitialCacheHook>>,
    #[cfg(test)]
    race_ranges: Mutex<Vec<(usize, usize)>>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InitialCacheStage {
    Filled,
    Scored,
}

#[cfg(test)]
type InitialCacheHook = dyn Fn(usize, InitialCacheStage) -> Result<()> + Send + Sync;

struct HistogramBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
}

impl HistogramBudget {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }
    fn reserve(self: &Arc<Self>, bytes: usize) -> Option<HistogramLease> {
        let previous = self
            .used
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(bytes).filter(|&next| next <= self.limit)
            })
            .ok()?;
        self.peak.fetch_max(previous + bytes, Ordering::Relaxed);
        Some(HistogramLease {
            budget: Arc::clone(self),
            bytes,
        })
    }
}

struct HistogramLease {
    budget: Arc<HistogramBudget>,
    bytes: usize,
}
impl Drop for HistogramLease {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

struct BinHistogram {
    symbols: usize,
    counts: Vec<u32>,
    extra: Vec<u64>,
}

impl BinHistogram {
    #[inline]
    fn subtract(&mut self, residual: ResidualToken, bin: usize, weight: u32) -> Result<()> {
        let cell = &mut self.counts[bin * self.symbols + usize::from(residual.token)];
        *cell = cell
            .checked_sub(weight)
            .ok_or_else(|| Error::Encoding("MA cached token subtraction underflow".into()))?;
        self.extra[bin] = self.extra[bin]
            .checked_sub(u64::from(weight) * u64::from(residual.nbits))
            .ok_or_else(|| Error::Encoding("MA cached extra-bit subtraction underflow".into()))?;
        Ok(())
    }

    fn copy_predictor_from(&mut self, source: &Self) -> Result<()> {
        debug_assert_eq!(self.extra.len(), source.extra.len());
        let shared = self.symbols.min(source.symbols);
        for (destination, source) in self
            .counts
            .chunks_exact_mut(self.symbols)
            .zip(source.counts.chunks_exact(source.symbols))
        {
            // Source uses a common stride; Rust retains per-predictor strides.
            // After subtraction, equal live tokens imply zero discarded tails.
            // Keep both allocations unchanged under the original cache lease.
            if source[shared..].iter().any(|&count| count != 0) {
                return Err(Error::Encoding(
                    "MA predictor cache copy has a nonzero tail".into(),
                ));
            }
            destination[..shared].copy_from_slice(&source[..shared]);
            destination[shared..].fill(0);
        }
        self.extra.copy_from_slice(&source.extra);
        Ok(())
    }
}
#[derive(Default)]
struct PropertyHistogram {
    used: Vec<usize>,
    weights: Vec<u32>,
    predictors: Vec<BinHistogram>,
    valid: bool,
}

fn collect_property_marginals<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    bins: usize,
    used: &mut Vec<usize>,
    weights: &mut Vec<u32>,
) {
    used.clear();
    used.resize(bins, 0);
    weights.clear();
    weights.resize(bins, 0);
    for &i in indices {
        let bin = values[i].index();
        used[bin] += 1;
        weights[bin] += u32::from(samples.counts[i]);
    }
}

impl PropertyHistogram {
    #[cfg(test)]
    fn collect(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        property: usize,
    ) -> Self {
        with_property_values!(&samples.properties[property], |values| {
            Self::collect_values(
                samples,
                indices,
                parents,
                property,
                values,
                &PredictorPlan::identity(parents.len()),
            )
            .expect("identity predictor cache collection")
        })
    }

    #[cfg(test)]
    fn collect_values<T: PropertyValue>(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        property: usize,
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<Self> {
        Self::collect_values_with_stride(samples, indices, parents, property, values, plan, None)
    }

    fn collect_values_with_stride<T: PropertyValue>(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        property: usize,
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
    ) -> Result<Self> {
        let mut histogram = Self::collect_marginals_values(samples, indices, property, values);
        histogram.collect_predictors_values(samples, indices, parents, values, plan, stride)?;
        Ok(histogram)
    }

    fn collect_marginals_values<T: PropertyValue>(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        property: usize,
        values: &[T],
    ) -> Self {
        let bins = samples.thresholds[property].len() + 1;
        let mut histogram = Self::default();
        collect_property_marginals(
            samples,
            indices,
            values,
            bins,
            &mut histogram.used,
            &mut histogram.weights,
        );
        histogram
    }

    fn collect_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
    ) -> Result<()> {
        self.collect_predictors_values_with_pair14(
            samples,
            indices,
            parents,
            values,
            plan,
            stride,
            parents.len() == 14 && cache_pair14_enabled(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_predictors_values_with_pair14<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
        pair14: bool,
    ) -> Result<()> {
        if !self.allocate_predictors(parents, stride) {
            return Ok(());
        }
        if pair14 && plan.cache_pair14_eligible(self.predictors.len()) {
            // Touch two histogram columns per pass without extra allocations.
            for predictor in (0..14).step_by(2) {
                let (first, rest) = self.predictors.split_at_mut(predictor + 1);
                let (first, second) = (&mut first[predictor], &mut rest[0]);
                let (a, b) = (
                    &samples.residuals[predictor][..],
                    &samples.residuals[predictor + 1][..],
                );
                for &i in indices {
                    let bin = values[i].index();
                    let weight = u32::from(samples.counts[i]);
                    let residual = a[i];
                    first.counts[bin * first.symbols + usize::from(residual.token)] += weight;
                    first.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                    let residual = b[i];
                    second.counts[bin * second.symbols + usize::from(residual.token)] += weight;
                    second.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                }
            }
        } else if self.predictors.len() == 2 && plan.active() == [0, 1] {
            // Hoist the two column/histogram lookups out of the record loop.
            let (first, rest) = self.predictors.split_at_mut(1);
            let (first, second) = (&mut first[0], &mut rest[0]);
            let (a, b) = (&samples.residuals[0][..], &samples.residuals[1][..]);
            for &i in indices {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                let residual = a[i];
                first.counts[bin * first.symbols + usize::from(residual.token)] += weight;
                first.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                let residual = b[i];
                second.counts[bin * second.symbols + usize::from(residual.token)] += weight;
                second.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
            }
        } else {
            // Source fused collection: share property/weight loads.
            for &i in indices {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                for &predictor in plan.active() {
                    let histogram = &mut self.predictors[predictor];
                    let residual = samples.residuals[predictor][i];
                    histogram.counts[bin * histogram.symbols + usize::from(residual.token)] +=
                        weight;
                    histogram.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
                }
            }
        }
        self.copy_predictors(plan)?;
        self.valid = true;
        Ok(())
    }

    fn allocate_predictors(&mut self, parents: &[ParentHistogram], stride: Option<usize>) -> bool {
        // A subset of a constant property stays constant. Retain only its
        // partition counts; neither this node nor descendants need tokens.
        if self.used.iter().filter(|&&n| n != 0).take(2).count() < 2 {
            self.valid = true;
            return false;
        }
        let bins = self.used.len();
        self.predictors = parents
            .iter()
            .map(|parent| BinHistogram {
                symbols: stride.unwrap_or(parent.counts.len()),
                counts: vec![0; bins * stride.unwrap_or(parent.counts.len())],
                extra: vec![0; bins],
            })
            .collect();
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_sharded_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        values: &[T],
        plan: &PredictorPlan,
        stride: Option<usize>,
        node_stride: usize,
        scratch: &mut HistogramShards,
    ) -> Result<usize> {
        if !self.allocate_predictors(parents, stride) {
            return Ok(0);
        }
        let bins = self.used.len();
        let mut sharded = 0;
        for &predictor in plan.active() {
            if indices.len() >= 256 && plan.pair_available(predictor, parents.len()) {
                if predictor % 2 == 1 {
                    continue;
                }
                let (first, rest) = self.predictors.split_at_mut(predictor + 1);
                let (first, second) = (&mut first[predictor], &mut rest[0]);
                for &i in indices {
                    let bin = values[i].index();
                    let weight = u32::from(samples.counts[i]);
                    let a = samples.residuals[predictor][i];
                    let b = samples.residuals[predictor + 1][i];
                    first.counts[bin * first.symbols + usize::from(a.token)] += weight;
                    second.counts[bin * second.symbols + usize::from(b.token)] += weight;
                    first.extra[bin] += u64::from(weight) * u64::from(a.nbits);
                    second.extra[bin] += u64::from(weight) * u64::from(b.nbits);
                }
            } else {
                let histogram = &mut self.predictors[predictor];
                if indices.len() >= 4096 && indices.len() >= 4 * bins * node_stride {
                    scratch.collect(
                        samples,
                        indices,
                        values,
                        &samples.residuals[predictor],
                        histogram.symbols,
                        bins,
                        &mut histogram.counts,
                        &mut histogram.extra,
                    )?;
                    sharded += 1;
                } else {
                    collect_mapped::<T, false, false>(
                        samples,
                        indices,
                        values,
                        &samples.residuals[predictor],
                        histogram.symbols,
                        &[],
                        None,
                        &mut histogram.counts,
                        &mut histogram.extra,
                    );
                }
            }
        }
        self.copy_predictors(plan)?;
        self.valid = true;
        Ok(sharded)
    }

    #[cfg(test)]
    fn subtract(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        property: usize,
        removed: &[usize],
    ) -> Result<()> {
        with_property_values!(&samples.properties[property], |values| {
            self.subtract_values(
                samples,
                removed,
                values,
                &PredictorPlan::identity(samples.residuals.len()),
            )
        })
    }

    fn subtract_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<()> {
        if self.predictors.len() == 2 && plan.active() == [0, 1] {
            let (first, rest) = self.predictors.split_at_mut(1);
            let (first, second) = (&mut first[0], &mut rest[0]);
            let (a, b) = (&samples.residuals[0][..], &samples.residuals[1][..]);
            for &i in removed {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                // Preserve record/weight/first/second failure order without
                // introducing the separate marginal pass of paired_cache.
                self.used[bin] = self.used[bin].checked_sub(1).ok_or_else(|| {
                    Error::Encoding("MA cached record subtraction underflow".into())
                })?;
                self.weights[bin] = self.weights[bin].checked_sub(weight).ok_or_else(|| {
                    Error::Encoding("MA cached weight subtraction underflow".into())
                })?;
                first.subtract(a[i], bin, weight)?;
                second.subtract(b[i], bin, weight)?;
            }
        } else {
            for &i in removed {
                let bin = values[i].index();
                let weight = u32::from(samples.counts[i]);
                self.subtract_marginal(bin, weight)?;
                self.subtract_predictors(samples, i, bin, weight, plan)?;
            }
        }
        self.copy_predictors(plan)?;
        Ok(())
    }

    fn subtract_marginal(&mut self, bin: usize, weight: u32) -> Result<()> {
        self.used[bin] = self.used[bin]
            .checked_sub(1)
            .ok_or_else(|| Error::Encoding("MA cached record subtraction underflow".into()))?;
        self.weights[bin] = self.weights[bin]
            .checked_sub(weight)
            .ok_or_else(|| Error::Encoding("MA cached weight subtraction underflow".into()))?;
        Ok(())
    }

    fn subtract_predictors(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        i: usize,
        bin: usize,
        weight: u32,
        plan: &PredictorPlan,
    ) -> Result<()> {
        let predictor_count = self.predictors.len();
        for &predictor in plan.active().iter().filter(|&&p| p < predictor_count) {
            self.predictors[predictor].subtract(samples.residuals[predictor][i], bin, weight)?;
        }
        Ok(())
    }

    fn subtract_marginals_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
    ) -> Result<()> {
        for &i in removed {
            self.subtract_marginal(values[i].index(), u32::from(samples.counts[i]))?;
        }
        Ok(())
    }

    fn subtract_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<()> {
        for &i in removed {
            self.subtract_predictors(
                samples,
                i,
                values[i].index(),
                u32::from(samples.counts[i]),
                plan,
            )?;
        }
        self.copy_predictors(plan)?;
        Ok(())
    }

    fn subtract_paired_predictors_values<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        values: &[T],
        plan: &PredictorPlan,
    ) -> Result<usize> {
        let count = self.predictors.len();
        let mut pairs = 0;
        for &predictor in plan.active().iter().filter(|&&p| p < count) {
            if plan.pair_available(predictor, count) {
                if predictor % 2 == 1 {
                    continue;
                }
                let (first, rest) = self.predictors.split_at_mut(predictor + 1);
                let (first, second) = (&mut first[predictor], &mut rest[0]);
                for &i in removed {
                    let bin = values[i].index();
                    let weight = u32::from(samples.counts[i]);
                    first.subtract(samples.residuals[predictor][i], bin, weight)?;
                    second.subtract(samples.residuals[predictor + 1][i], bin, weight)?;
                }
                pairs += 1;
            } else {
                for &i in removed {
                    self.predictors[predictor].subtract(
                        samples.residuals[predictor][i],
                        values[i].index(),
                        u32::from(samples.counts[i]),
                    )?;
                }
            }
        }
        self.copy_predictors(plan)?;
        Ok(pairs)
    }

    fn copy_predictors(&mut self, plan: &PredictorPlan) -> Result<()> {
        for predictor in 0..self.predictors.len() {
            let representative = plan.representative[predictor];
            if representative != predictor {
                let (previous, current) = self.predictors.split_at_mut(predictor);
                current[0].copy_predictor_from(&previous[representative])?;
            }
        }
        Ok(())
    }
}

struct HistogramCache {
    properties: Vec<PropertyHistogram>,
    _lease: HistogramLease,
    stride: Option<usize>,
}

impl HistogramCache {
    // Reserve the complete conservative lease before any property allocation.
    // Entries stay empty until their scoring tasks fill them. This is separate
    // from find_best_split's record-count creation gate (8192 by default).
    #[cfg(test)]
    fn reserve(
        samples: &Samples<impl SampleStorage>,
        parents: &[ParentHistogram],
        budget: &Arc<HistogramBudget>,
    ) -> Option<Self> {
        Self::reserve_with_stride(samples, parents, budget, None)
    }

    fn reserve_with_stride(
        samples: &Samples<impl SampleStorage>,
        parents: &[ParentHistogram],
        budget: &Arc<HistogramBudget>,
        stride: Option<usize>,
    ) -> Option<Self> {
        if budget.limit == 0 {
            return None;
        }
        let mut bytes = std::mem::size_of::<Self>();
        for thresholds in samples.thresholds.iter() {
            let bins = thresholds.len() + 1;
            bytes = bytes
                .checked_add(std::mem::size_of::<PropertyHistogram>())?
                .checked_add(bins.checked_mul(std::mem::size_of::<usize>() + 4)?)?;
            for parent in parents {
                bytes = bytes
                    .checked_add(std::mem::size_of::<BinHistogram>())?
                    .checked_add(
                        bins.checked_mul(
                            stride
                                .unwrap_or(parent.counts.len())
                                .checked_mul(4)?
                                .checked_add(8)?,
                        )?,
                    )?;
            }
        }
        let lease = budget.reserve(bytes)?;
        Some(Self {
            properties: (0..samples.thresholds.len())
                .map(|_| PropertyHistogram::default())
                .collect(),
            _lease: lease,
            stride,
        })
    }

    // Eager population is retained only as a test reference for the fused path
    // and for constructing inherited-cache fixtures below the creation gate.
    #[cfg(test)]
    fn create(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        parents: &[ParentHistogram],
        budget: &Arc<HistogramBudget>,
        parallel: bool,
    ) -> Option<Self> {
        let mut cache = Self::reserve(samples, parents, budget)?;
        let work = indices
            .len()
            .saturating_mul(parents.len())
            .saturating_mul(samples.thresholds.len());
        if parallel && indices.len() >= 1024 && work >= 65536 {
            let jobs: Vec<_> = cache.properties.iter_mut().map(Mutex::new).collect();
            let _: Vec<()> = execution::map_with_state(
                jobs.len(),
                || (),
                |_, property| {
                    **jobs[property].lock().expect("MA property cache poisoned") =
                        PropertyHistogram::collect(samples, indices, parents, property);
                },
            );
        } else {
            for (property, histogram) in cache.properties.iter_mut().enumerate() {
                *histogram = PropertyHistogram::collect(samples, indices, parents, property);
            }
        }
        Some(cache)
    }

    #[cfg(test)]
    fn subtract(&mut self, samples: &Samples<impl SampleStorage>, removed: &[usize]) -> Result<()> {
        for (property, histogram) in self.properties.iter_mut().enumerate() {
            histogram.subtract(samples, property, removed)?;
        }
        Ok(())
    }

    fn normalize_for_handoff(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        removed: &[usize],
        options: &TrainOptions,
        redundant: &[u8],
    ) -> Result<()> {
        let plan = PredictorPlan::identity(samples.residuals.len());
        let subtract = |property: usize, histogram: &mut PropertyHistogram| {
            // Redundant entries may contain only updated marginals, or token
            // tables from an ancestor. They must never be subtracted/reused.
            if options.property_dedup && redundant.get(property).is_some_and(|&v| v != 0) {
                histogram.valid = false;
            }
            if histogram.valid {
                with_property_values!(&samples.properties[property], |values| {
                    histogram.subtract_values(samples, removed, values, &plan)
                })
            } else {
                Ok(())
            }
        };
        if options.parallel && removed.len() >= 1024 {
            for result in execution::map_mut(&mut self.properties, subtract) {
                result?;
            }
        } else {
            for (property, histogram) in self.properties.iter_mut().enumerate() {
                subtract(property, histogram)?;
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct PropertyScratch {
    dominated_predictors: usize,
    splits: Vec<HistogramSplit>,
    score_mask: Vec<u8>,
    saved_histograms: Vec<BinHistogram>,
    property_bins: Vec<usize>,
    shards: HistogramShards,
    scoring: ScoreScratch,
    events: EventScratch,
    #[cfg(test)]
    sharded_predictors: Vec<usize>,
    #[cfg(test)]
    sparse_predictors: Vec<usize>,
    #[cfg(test)]
    packed_bins: usize,
    used: Vec<usize>,
    weights: Vec<u32>,
    below: Vec<Cost>,
    above: Vec<Cost>,
    increase: Vec<u32>,
    extra: Vec<u64>,
    paired_increase: Vec<u32>,
    paired_extra: Vec<u64>,
    counts_above: Vec<u32>,
    counts_below: Vec<u32>,
    #[cfg(test)]
    scored_predictors: Vec<usize>,
}

/// Source PropertyWorkspace lifetime, without indexing scratch by Rayon thread
/// ID: nested ready-child jobs can suspend a worker while another job runs on it.
/// A lease owns its buffers exclusively and holds no lock during the sweep.
struct ScratchPool {
    idle: Mutex<Vec<PropertyScratch>>,
    retained_limit: usize,
    created: AtomicUsize,
    reused: AtomicUsize,
    // Reset at learner entry, set only by actual parent/property dispatches.
    dispatch_slots: AtomicUsize,
}

impl ScratchPool {
    fn new(retained_limit: usize) -> Self {
        Self {
            idle: Mutex::new(Vec::with_capacity(retained_limit)),
            retained_limit,
            created: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
            dispatch_slots: AtomicUsize::new(0),
        }
    }

    fn acquire(&self) -> ScratchLease<'_> {
        let available = self.idle.lock().expect("MA scratch pool poisoned").pop();
        let scratch = if let Some(scratch) = available {
            self.reused.fetch_add(1, Ordering::Relaxed);
            scratch
        } else {
            self.created.fetch_add(1, Ordering::Relaxed);
            PropertyScratch::default()
        };
        ScratchLease {
            pool: self,
            scratch,
        }
    }
}

struct ScratchLease<'a> {
    pool: &'a ScratchPool,
    scratch: PropertyScratch,
}

impl Drop for ScratchLease<'_> {
    fn drop(&mut self) {
        let mut idle = self.pool.idle.lock().expect("MA scratch pool poisoned");
        if idle.len() < self.pool.retained_limit {
            idle.push(std::mem::take(&mut self.scratch));
        }
    }
}

#[derive(Clone, Copy)]
struct Cost {
    bits: f32,
    penalty: f32,
    predictor: usize,
}

impl Default for Cost {
    fn default() -> Self {
        Self {
            bits: f32::INFINITY,
            penalty: 0.0,
            predictor: 0,
        }
    }
}

fn predictor_penalty(predictor: Predictor, parent: Predictor, threshold: f32) -> f32 {
    let mut penalty = if predictor != parent && parent != Predictor::Weighted {
        800.0 / (100.0 + threshold)
    } else {
        0.0
    };
    if predictor == Predictor::Weighted {
        penalty += 1e-8;
    }
    if predictor == Predictor::Zero {
        penalty -= 1e-8;
    }
    penalty
}

/// enc_ma.cc::FindBestSplitImpl representative/penalty construction. This proof
/// belongs to one node, not to retained worker scratch or an inherited cache.
struct PredictorPlan {
    representative: [usize; 14],
    penalty: [f32; 14],
    active: [usize; 14],
    active_count: usize,
}

impl PredictorPlan {
    fn identity(predictors: usize) -> Self {
        debug_assert!(predictors <= 14);
        Self {
            representative: std::array::from_fn(|i| i),
            penalty: [0.0; 14],
            active: std::array::from_fn(|i| i),
            active_count: predictors,
        }
    }

    fn new(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        options: &TrainOptions,
        parents: &[ParentHistogram],
        parent_predictor: usize,
    ) -> Self {
        let predictors = parents.len();
        let mut plan = Self::identity(predictors);
        plan.active_count = 0;
        for predictor in 0..predictors {
            let penalty = predictor_penalty(
                options.predictors[predictor],
                options.predictors[parent_predictor],
                options.split_threshold as f32,
            );
            plan.penalty[predictor] = penalty;
            if options.predictor_dedup && predictors > 2 {
                for previous in 0..predictor {
                    if plan.representative[previous] != previous
                        || plan.penalty[previous] != penalty
                        || !Self::same_parent(&parents[previous], &parents[predictor])
                    {
                        continue;
                    }
                    if indices
                        .iter()
                        .all(|&i| samples.residuals[previous][i] == samples.residuals[predictor][i])
                    {
                        plan.representative[predictor] = previous;
                        break;
                    }
                }
            }
            if plan.representative[predictor] == predictor {
                plan.active[plan.active_count] = predictor;
                plan.active_count += 1;
            }
        }
        plan
    }

    fn same_parent(a: &ParentHistogram, b: &ParentHistogram) -> bool {
        // Match source memcmp at the common zero-padded stride, including
        // inherited histograms whose trimmed tails differ in length.
        let shared = a.counts.len().min(b.counts.len());
        a.extra_bits == b.extra_bits
            && a.counts[..shared] == b.counts[..shared]
            && a.counts[shared..]
                .iter()
                .chain(&b.counts[shared..])
                .all(|&n| n == 0)
    }

    fn active(&self) -> &[usize] {
        &self.active[..self.active_count]
    }

    fn cache_pair14_eligible(&self, predictors: usize) -> bool {
        predictors == 14
            && self.active().iter().copied().eq(0..14)
            && self.representative.iter().copied().eq(0..14)
    }

    fn pair_available(&self, predictor: usize, predictors: usize) -> bool {
        let start = predictor & !1;
        start + 1 < predictors
            && self.representative[start] == start
            && self.representative[start + 1] == start + 1
    }
}

#[derive(Clone, Copy, Debug)]
struct Split {
    property: usize,
    bin: usize,
    below_count: usize,
    below_predictor: usize,
    above_predictor: usize,
    bits: f32,
}

struct SeenProperty {
    signature: u64,
    property: usize,
}

fn property_signature(used: &[usize]) -> u64 {
    let mut signature = 14695981039346656037u64;
    for (value, &count) in used.iter().enumerate() {
        if count != 0 {
            signature = (signature ^ value as u64).wrapping_mul(1099511628211);
            signature = (signature ^ count as u64).wrapping_mul(1099511628211);
        }
    }
    signature
}

// enc_ma.cc::SeenProperty / enc_ma.h::SameProperty. Signatures only filter
// candidates; equality must cover every quantized value in the live node.
#[allow(clippy::too_many_arguments)]
fn duplicate_property<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    property: usize,
    values: &[T],
    used: &[usize],
    seen: &mut Vec<SeenProperty>,
    remap: &mut Vec<usize>,
) -> bool {
    // Search experiments may select thresholds by their original bin index.
    // Only a complete sweep permits ignoring strictly monotone bin relabeling.
    let relabel = options.search == SearchOptions::default();
    let signature = if relabel {
        ma_partition_proofs::signature(used)
    } else {
        property_signature(used)
    };
    let id = options.properties[property];
    for previous in seen.iter() {
        let previous_id = options.properties[previous.property];
        if previous.signature != signature
            || (previous_id < 2) != (id < 2)
            || (previous_id == WP_PROPERTY) != (id == WP_PROPERTY)
        {
            continue;
        }
        let same = with_property_values!(&samples.properties[previous.property], |other| {
            if relabel {
                ma_partition_proofs::same_partition(indices, values, other, used.len(), remap)
            } else {
                indices
                    .iter()
                    .all(|&i| values[i].index() == other[i].index())
            }
        });
        if same {
            return true;
        }
    }
    seen.push(SeenProperty {
        signature,
        property,
    });
    false
}

type Candidates = [Option<Split>; 4];

fn keep_better(best: &mut Option<Split>, candidate: Split) {
    if best.is_none_or(|old| candidate.bits < old.bits) {
        *best = Some(candidate);
    }
}

/// CollectExtraBitsIncrease + the full FindBestSplit per-property sweep.
// Keep the source sweep inputs explicit alongside its reusable cache and scratch.
#[allow(clippy::too_many_arguments)]
fn evaluate_property_values<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    parents: &[ParentHistogram],
    plan: &PredictorPlan,
    sparse_plan: &SparseHistogramPlan,
    total: u32,
    property: usize,
    values: &[T],
    cached: Option<&PropertyHistogram>,
    marginals_ready: bool,
    mapping: Option<&[u8; 256]>,
    scratch: &mut PropertyScratch,
) -> Result<Candidates> {
    debug_assert!(cached.is_none() || mapping.is_none());
    debug_assert!(cached.is_none_or(|histogram| histogram.valid));
    scratch.score_mask.clear();
    scratch.dominated_predictors = 0;
    #[cfg(test)]
    {
        scratch.scored_predictors.clear();
        scratch.sharded_predictors.clear();
        scratch.sparse_predictors.clear();
        scratch.packed_bins = 0;
        scratch.events.reset_trace();
    }
    let bins = samples.thresholds[property].len() + 1;
    if cached.is_none() && !marginals_ready {
        collect_property_marginals(
            samples,
            indices,
            values,
            bins,
            &mut scratch.used,
            &mut scratch.weights,
        );
    }
    let used = cached.map_or(scratch.used.as_slice(), |histogram| {
        histogram.used.as_slice()
    });
    let weights = cached.map_or(scratch.weights.as_slice(), |histogram| {
        histogram.weights.as_slice()
    });
    let first = used.iter().position(|&n| n != 0).unwrap_or(0);
    let last = used.iter().rposition(|&n| n != 0).unwrap_or(0);
    if first == last {
        return Ok([None; 4]);
    }
    scratch.below.clear();
    scratch.below.resize(bins, Cost::default());
    scratch.above.clear();
    scratch.above.resize(bins, Cost::default());
    let coarse_rows = if cached.is_none() {
        ma_binning::prepare(
            options.search.histogram_bins,
            options.properties[property],
            used,
            weights,
            total,
            &mut scratch.property_bins,
            &mut scratch.splits,
            &mut scratch.score_mask,
        )?
    } else {
        None
    };
    let coarsened = coarse_rows.is_some();
    let _binning_scope = coarsened.then(|| crate::profile::scope("ma.adaptive_histogram"));
    if !coarsened
        && ma_events::eligible(
            options,
            total,
            indices.len(),
            used,
            parents,
            plan,
            sparse_plan,
            cached.is_some(),
        )
    {
        scratch.events.evaluate(
            samples,
            indices,
            values,
            used,
            weights,
            parents,
            plan,
            options,
            total,
            mapping,
            &mut scratch.below,
            &mut scratch.above,
        )?;
        #[cfg(test)]
        scratch.scored_predictors.extend_from_slice(plan.active());
        return Ok(property_candidates(
            options,
            property,
            used,
            (&scratch.below, &scratch.above),
            first..last,
            &[],
        ));
    }
    let rows = coarse_rows.unwrap_or_else(|| {
        prepare_property_bins(
            used,
            options.property_bins,
            indices.len(),
            mapping.is_some(),
            cached.is_some(),
            &mut scratch.property_bins,
        )
    });
    let packed = rows != bins;
    // Property weights and live thresholds do not depend on the predictor.
    // Small descendants often occupy only a handful of the root's 256 bins.
    // One/two predictors do not amortize materializing a separate split table.
    let reuse_splits = coarsened
        || plan.active_count >= 4
        || options.scoring != ScoringOptions::default()
        || !options.search.default_sweep();
    if !coarsened {
        scratch.splits.clear();
    }
    if reuse_splits && !coarsened {
        let mut weight_below = 0;
        for bin in first..last {
            if used[bin] != 0 {
                weight_below += weights[bin];
                scratch.splits.push(HistogramSplit {
                    bin,
                    row: if packed {
                        scratch.property_bins[bin]
                    } else {
                        bin
                    },
                    totals: (total - weight_below, weight_below),
                });
            }
        }
    }
    #[cfg(test)]
    {
        scratch.packed_bins = if packed { rows } else { 0 };
    }
    let below_costs = &mut scratch.below;
    let above_costs = &mut scratch.above;
    let scorer = DefaultScorer::default();
    let node_stride = histogram_stride(
        parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
        scorer.lanes(),
        options.padded_histogram && mapping.is_none(),
    );
    let refine = !coarsened
        && ma_search::prepare_mask(
            options.search,
            options.scoring,
            indices.len(),
            options.properties[property],
            used,
            weights,
            total,
            first,
            last,
            rows.checked_mul(node_stride)
                .ok_or(Error::Limit("MA refinement histogram size"))?,
            parents.len(),
            &mut scratch.score_mask,
        )?;
    let saved_count = if refine && cached.is_none() {
        parents.len()
    } else {
        0
    };
    scratch.saved_histograms.truncate(saved_count);
    scratch
        .saved_histograms
        .resize_with(saved_count, || BinHistogram {
            symbols: 0,
            counts: Vec::new(),
            extra: Vec::new(),
        });
    for pass in 0..1 + usize::from(refine) {
        if pass != 0 {
            ma_search::refine_mask_by(&mut scratch.score_mask, |offset| {
                let bin = first + offset;
                (below_costs[bin].bits + below_costs[bin].penalty)
                    + (above_costs[bin].bits + above_costs[bin].penalty)
            });
        }
        for &predictor in plan.active() {
            let parent = &parents[predictor];
            #[cfg(test)]
            scratch.scored_predictors.push(predictor);
            let symbols = parent.counts.len();
            let storage_stride = if options.padded_histogram {
                node_stride
            } else {
                symbols
            };
            let collect_scope = histogram_detail_scope(if cached.is_some() {
                "ma.histogram_cached"
            } else if pass != 0 {
                "ma.histogram_saved"
            } else {
                "ma.histogram_collect"
            });
            let (increase, extra, stride) = if let Some(cached) = cached {
                let histogram = &cached.predictors[predictor];
                (
                    histogram.counts.as_slice(),
                    histogram.extra.as_slice(),
                    histogram.symbols,
                )
            } else if pass != 0 {
                let saved = &scratch.saved_histograms[predictor];
                (
                    saved.counts.as_slice(),
                    saved.extra.as_slice(),
                    saved.symbols,
                )
            } else {
                let paired = !packed
                    && plan.pair_available(predictor, parents.len())
                    && if mapping.is_some() {
                        // Every active adjacent pair can share the compact
                        // property/weight scan, including Variable's 14 tools.
                        indices.len() <= 256
                    } else {
                        indices.len() >= 256
                    };
                if paired && predictor % 2 == 1 {
                    // The preceding predictor populated both exact histograms.
                    // Swapping preserves capacity and handles unequal strides.
                    std::mem::swap(&mut scratch.increase, &mut scratch.paired_increase);
                    std::mem::swap(&mut scratch.extra, &mut scratch.paired_extra);
                } else {
                    scratch.increase.clear();
                    scratch.increase.resize(rows * storage_stride, 0);
                    scratch.extra.clear();
                    scratch.extra.resize(rows, 0);
                    let tokens = &samples.residuals[predictor];
                    if paired {
                        let other_symbols = if options.padded_histogram {
                            node_stride
                        } else {
                            parents[predictor + 1].counts.len()
                        };
                        let other_tokens = &samples.residuals[predictor + 1];
                        scratch.paired_increase.clear();
                        scratch.paired_increase.resize(bins * other_symbols, 0);
                        scratch.paired_extra.clear();
                        scratch.paired_extra.resize(bins, 0);
                        if let Some(mapping) = mapping {
                            collect_compact_pair(
                                samples,
                                indices,
                                values,
                                [tokens, other_tokens],
                                [storage_stride, other_symbols],
                                mapping,
                                [&mut scratch.increase, &mut scratch.paired_increase],
                                [&mut scratch.extra, &mut scratch.paired_extra],
                            );
                        } else {
                            for &i in indices {
                                let bin = values[i].index();
                                let weight = u32::from(samples.counts[i]);
                                let a = tokens[i];
                                let b = other_tokens[i];
                                scratch.increase[bin * storage_stride + usize::from(a.token)] +=
                                    weight;
                                scratch.paired_increase
                                    [bin * other_symbols + usize::from(b.token)] += weight;
                                scratch.extra[bin] += u64::from(weight) * u64::from(a.nbits);
                                scratch.paired_extra[bin] += u64::from(weight) * u64::from(b.nbits);
                            }
                        }
                    } else if options.histogram_shards
                        && !packed
                        && mapping.is_none()
                        && indices.len() >= 4096
                        && indices.len() >= 4 * bins * node_stride
                    {
                        scratch.shards.collect(
                            samples,
                            indices,
                            values,
                            tokens,
                            storage_stride,
                            bins,
                            &mut scratch.increase,
                            &mut scratch.extra,
                        )?;
                        #[cfg(test)]
                        scratch.sharded_predictors.push(predictor);
                    } else {
                        match (packed, mapping.is_some()) {
                            (false, false) => collect_mapped::<T, false, false>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &[],
                                None,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                            (false, true) => collect_mapped::<T, false, true>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &[],
                                mapping,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                            (true, false) => collect_mapped::<T, true, false>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &scratch.property_bins,
                                None,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                            (true, true) => collect_mapped::<T, true, true>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &scratch.property_bins,
                                mapping,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                        }
                    }
                }
                if refine {
                    let saved = &mut scratch.saved_histograms[predictor];
                    saved.symbols = storage_stride;
                    saved
                        .counts
                        .try_reserve(scratch.increase.len().saturating_sub(saved.counts.len()))
                        .map_err(|_| Error::Limit("MA refinement counts allocation"))?;
                    saved
                        .extra
                        .try_reserve(scratch.extra.len().saturating_sub(saved.extra.len()))
                        .map_err(|_| Error::Limit("MA refinement extra allocation"))?;
                    saved.counts.clear();
                    saved.counts.extend_from_slice(&scratch.increase);
                    saved.extra.clear();
                    saved.extra.extend_from_slice(&scratch.extra);
                }
                (
                    scratch.increase.as_slice(),
                    scratch.extra.as_slice(),
                    storage_stride,
                )
            };
            drop(collect_scope);
            // Both child populations use the nonnegative <=256 contribution
            // table. This lower bound keeps the exact f32 extra-bit order;
            // it is local to this node and must not be inherited by children.
            if options.split_dominance
                && total <= 256
                && options.scoring == ScoringOptions::default()
                && options.search == SearchOptions::default()
                && ma_dominance::predictor_cannot_improve(
                    (first..last).filter(|&bin| used[bin] != 0).map(|bin| {
                        (
                            bin,
                            if packed {
                                scratch.property_bins[bin]
                            } else {
                                bin
                            },
                        )
                    }),
                    extra,
                    parent.extra_bits,
                    plan.penalty[predictor],
                    below_costs,
                    above_costs,
                )
            {
                scratch.dominated_predictors += 1;
                continue;
            }
            let sparse = sparse_plan.blocks(predictor);
            #[cfg(test)]
            if sparse.is_some() {
                scratch.sparse_predictors.push(predictor);
            }
            scratch.counts_above.clone_from(&parent.counts);
            scratch.counts_below.clear();
            scratch.counts_below.resize(symbols, 0);
            scratch
                .scoring
                .reset(options.scoring, &parent.counts, total, indices.len())?;
            let penalty = plan.penalty[predictor];
            let mut extra_below = 0u64;
            let _score_scope = histogram_detail_scope("ma.histogram_score");
            if options.scoring == ScoringOptions::default()
                && (scratch.score_mask.is_empty() || coarsened)
            {
                let mut weight_below = 0;
                let count = if reuse_splits {
                    scratch.splits.len()
                } else {
                    last - first
                };
                let splits = (0..count).filter_map(|offset| {
                    if reuse_splits {
                        return Some(scratch.splits[offset]);
                    }
                    let bin = first + offset;
                    if used[bin] == 0 {
                        return None;
                    }
                    weight_below += weights[bin];
                    Some(HistogramSplit {
                        bin,
                        row: if packed {
                            scratch.property_bins[bin]
                        } else {
                            bin
                        },
                        totals: (total - weight_below, weight_below),
                    })
                });
                scorer.split_sweep(
                    &mut scratch.counts_above,
                    &mut scratch.counts_below,
                    increase,
                    stride,
                    total,
                    sparse,
                    splits,
                    |split, rbits, lbits| {
                        extra_below += extra[split.row];
                        // Keep the same f32 extra-bit and strict tie order as
                        // the configurable per-threshold scoring path below.
                        update_cost(
                            &mut below_costs[split.bin],
                            lbits + extra_below as f32,
                            penalty,
                            predictor,
                        );
                        update_cost(
                            &mut above_costs[split.bin],
                            (rbits + parent.extra_bits as f32) - extra_below as f32,
                            penalty,
                            predictor,
                        );
                    },
                );
                continue;
            }
            for split in &scratch.splits {
                let bin = split.bin;
                let row = split.row;
                extra_below += extra[row];
                // Validated samples and exact cached subsets bound every move and
                // signed SIMD conversion. Keep the cache immutable for child reuse.
                let delta = &increase[row * stride..row * stride + symbols];
                if !scratch.score_mask.is_empty() && scratch.score_mask[bin - first] == 0 {
                    scorer.move_counts(&mut scratch.counts_above, &mut scratch.counts_below, delta);
                    continue;
                }
                let totals = split.totals;
                let (rbits, lbits) = scratch.scoring.move_and_score(
                    options.scoring,
                    &scorer,
                    &mut scratch.counts_above,
                    &mut scratch.counts_below,
                    delta,
                    totals,
                    sparse,
                );
                let mut below_bits = lbits + extra_below as f32;
                // Preserve source f32 addition/subtraction order, not an integer
                // difference followed by one conversion.
                let mut above_bits = (rbits + parent.extra_bits as f32) - extra_below as f32;
                if options.scoring.description != DescriptionCost::Disabled {
                    below_bits +=
                        options
                            .scoring
                            .description
                            .cost(&scorer, &scratch.counts_below, totals.1);
                    above_bits +=
                        options
                            .scoring
                            .description
                            .cost(&scorer, &scratch.counts_above, totals.0);
                }
                update_cost(&mut below_costs[bin], below_bits, penalty, predictor);
                update_cost(&mut above_costs[bin], above_bits, penalty, predictor);
            }
        }
    }
    Ok(property_candidates(
        options,
        property,
        used,
        (below_costs, above_costs),
        first..last,
        &scratch.score_mask,
    ))
}

#[inline]
fn update_cost(best: &mut Cost, bits: f32, penalty: f32, predictor: usize) {
    if bits + penalty < best.bits + best.penalty {
        *best = Cost {
            bits,
            penalty,
            predictor,
        };
    }
}

fn property_candidates(
    options: &TrainOptions,
    property: usize,
    used: &[usize],
    costs: (&[Cost], &[Cost]),
    range: std::ops::Range<usize>,
    score_mask: &[u8],
) -> Candidates {
    let (below_costs, above_costs) = costs;
    let first = range.start;
    let mut result = [None; 4];
    let mut below_count = 0;
    for bin in range {
        if used[bin] == 0 {
            continue;
        }
        below_count += used[bin];
        if !score_mask.is_empty() && score_mask[bin - first] == 0 {
            continue;
        }
        let below = below_costs[bin];
        let above = above_costs[bin];
        let uses_wp = options.properties[property] == WP_PROPERTY
            || options.predictors[below.predictor] == Predictor::Weighted
            || options.predictors[above.predictor] == Predictor::Weighted;
        let class = if options.properties[property] < 2 {
            if below.bits == 0.0 || above.bits == 0.0 {
                0
            } else {
                1
            }
        } else if uses_wp {
            2
        } else {
            3
        };
        // Source penalties select predictors but are NOT part of split cost.
        keep_better(
            &mut result[class],
            Split {
                property,
                bin,
                below_count,
                below_predictor: below.predictor,
                above_predictor: above.predictor,
                bits: below.bits + above.bits,
            },
        );
    }
    result
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn evaluate_property_cached(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    parents: &[ParentHistogram],
    parent_predictor: usize,
    total: u32,
    property: usize,
    cached: Option<&PropertyHistogram>,
    mapping: Option<&[u8; 256]>,
    scratch: &mut PropertyScratch,
) -> Result<Candidates> {
    let plan = PredictorPlan::new(samples, indices, options, parents, parent_predictor);
    let sparse_plan = SparseHistogramPlan::new(
        parents,
        plan.active(),
        DefaultScorer::new().lanes(),
        options.sparse_entropy && !options.scoring.fast_entropy,
    );
    with_property_values!(&samples.properties[property], |values| {
        evaluate_property_values(
            samples,
            indices,
            options,
            parents,
            &plan,
            &sparse_plan,
            total,
            property,
            values,
            cached,
            false,
            mapping,
            scratch,
        )
    })
}

#[cfg(test)]
fn evaluate_property(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    parents: &[ParentHistogram],
    parent_predictor: usize,
    total: u32,
    property: usize,
) -> Result<Candidates> {
    evaluate_property_cached(
        samples,
        indices,
        options,
        parents,
        parent_predictor,
        total,
        property,
        None,
        None,
        &mut PropertyScratch::default(),
    )
}

fn choose_split(candidates: Candidates, base_bits: f32, options: &TrainOptions) -> Option<Split> {
    let threshold = options.split_threshold as f32;
    let mut best = candidates[2];
    for candidate in [candidates[3], candidates[1]].into_iter().flatten() {
        if candidate.bits + threshold < base_bits
            && candidate.bits
                <= options.fast_decode_multiplier as f32 * best.map_or(f32::INFINITY, |b| b.bits)
        {
            best = Some(candidate);
        }
    }
    if let Some(candidate) = candidates[0]
        && candidate.bits + threshold < base_bits
    {
        best = Some(candidate);
    }
    best.filter(|candidate| candidate.bits + threshold < base_bits)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn find_best_split(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parent_predictor: usize,
    options: &TrainOptions,
    parents: &mut Option<Vec<ParentHistogram>>,
    cache: &mut Option<HistogramCache>,
    budget: &Arc<HistogramBudget>,
    stats: &TrainingStats,
) -> Result<Option<Split>> {
    find_best_split_with_scratch(
        samples,
        indices,
        parent_predictor,
        options,
        parents,
        cache,
        budget,
        stats,
        &ScratchPool::new(1),
        &[],
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn find_best_split_with_scratch(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parent_predictor: usize,
    options: &TrainOptions,
    parents: &mut Option<Vec<ParentHistogram>>,
    cache: &mut Option<HistogramCache>,
    budget: &Arc<HistogramBudget>,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
    removed: &[usize],
) -> Result<Option<Split>> {
    find_best_split_with_properties(
        samples,
        indices,
        parent_predictor,
        options,
        parents,
        cache,
        budget,
        stats,
        scratch_pool,
        removed,
        &mut Vec::new(),
    )
}

// FindBestSplit keeps node proofs separate from invocation-owned scratch/cache.
#[allow(clippy::too_many_arguments)]
fn find_best_split_with_properties(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parent_predictor: usize,
    options: &TrainOptions,
    parents: &mut Option<Vec<ParentHistogram>>,
    cache: &mut Option<HistogramCache>,
    budget: &Arc<HistogramBudget>,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
    removed: &[usize],
    redundant_properties: &mut Vec<u8>,
) -> Result<Option<Split>> {
    let _scope = crate::profile::scope("ma.find_best_split");
    debug_assert!(removed.is_empty() || cache.is_some());
    if indices.len() < 2 {
        *cache = None;
        return Ok(None);
    }
    let mut leaf_parent = None;
    if parents.is_none() {
        let _scope = crate::profile::scope("ma.leaf_histogram");
        leaf_parent = Some(parent_histogram(samples, indices, parent_predictor));
    }
    let parent = parents.as_ref().map_or_else(
        || leaf_parent.as_ref().expect("uncached leaf histogram"),
        |parents| &parents[parent_predictor],
    );
    let total: u32 = parent.counts.iter().sum();
    let scorer = DefaultScorer::default();
    let base_bits = options.scoring.leaf_bits(&scorer, &parent.counts, total)
        + parent.extra_bits as f32
        + options
            .scoring
            .description
            .cost(&scorer, &parent.counts, total);
    if base_bits <= options.split_threshold as f32 {
        *cache = None;
        return Ok(None);
    }
    let parents = {
        let _scope = crate::profile::scope("ma.parent_histograms");
        parents.get_or_insert_with(|| {
            let leaf_parent = leaf_parent.expect("uncached leaf histogram");
            let collect = |predictor| {
                if predictor == parent_predictor {
                    leaf_parent.clone()
                } else {
                    parent_histogram(samples, indices, predictor)
                }
            };
            let predictors = samples.residuals.len();
            if options.parallel
                && predictors > 2
                && indices.len() >= 8192
                && indices.len().saturating_mul(predictors - 1) >= 65536
            {
                scratch_pool
                    .dispatch_slots
                    .store(execution::worker_count(), Ordering::Relaxed);
                execution::map_with_state(predictors, || (), |_, p| collect(p))
            } else {
                (0..predictors).map(collect).collect()
            }
        })
    };
    let plan = PredictorPlan::new(samples, indices, options, parents, parent_predictor);
    stats
        .equivalent_predictors
        .fetch_add(parents.len() - plan.active_count, Ordering::Relaxed);
    // Only publish a fully evaluated cache. A task error or unwind drops the
    // local cache/lease after all borrowed worker tasks have finished, leaving
    // no partially filled or subtracted cache visible to the caller.
    let mut working_cache = cache.take();
    #[cfg(test)]
    let inherited_cache = working_cache.is_some();
    if working_cache.is_some() {
        stats.cache_reuses.fetch_add(1, Ordering::Relaxed);
    } else if options.search.histogram_bins == 0
        && indices.len() >= options.histogram_cache_thresholds().0
    {
        let _scope = crate::profile::scope("ma.histogram_cache");
        let stride = options.padded_histogram.then(|| {
            histogram_stride(
                parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
                scorer.lanes(),
                true,
            )
        });
        working_cache = HistogramCache::reserve_with_stride(samples, parents, budget, stride);
        if working_cache.is_some() {
            stats.cache_builds.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[cfg(test)]
    let initialize_cache = !inherited_cache && working_cache.is_some();
    let _sweep_scope = crate::profile::scope("ma.split_sweep");
    let work = indices
        .len()
        .saturating_mul(options.predictors.len())
        .saturating_mul(options.properties.len());
    let parallel =
        options.parallel && indices.len() >= 1024 && work >= 65536 && execution::worker_count() > 1;
    // Publish proofs only when every property succeeds, alongside its cache.
    let mut working_redundant = if options.property_dedup {
        if redundant_properties.is_empty() {
            vec![0; options.properties.len()]
        } else {
            debug_assert_eq!(redundant_properties.len(), options.properties.len());
            redundant_properties.clone()
        }
    } else {
        Vec::new()
    };
    let compact_limit = if options.dense_histogram == DenseHistogram::Extended {
        4096
    } else {
        256
    };
    let compact = if !options.scoring.fast_entropy
        && working_cache.is_none()
        && indices.len() <= compact_limit
    {
        compact_parent_histograms_with_layout(
            parents,
            scorer.lanes(),
            options.dense_histogram,
            options.padded_histogram,
        )
    } else {
        None
    };
    if compact.is_some() {
        stats.compact_nodes.fetch_add(1, Ordering::Relaxed);
    }
    // Preserve raw parents for child subtraction and raw-token cache lookup.
    let scoring_parents = compact
        .as_ref()
        .map_or(parents.as_slice(), |c| c.parents.as_slice());
    let sparse_plan = SparseHistogramPlan::new(
        scoring_parents,
        plan.active(),
        scorer.lanes(),
        options.sparse_entropy && !options.scoring.fast_entropy,
    );
    let race = ma_search::race(
        samples,
        indices,
        options,
        histogram_stride(
            parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
            scorer.lanes(),
            options.padded_histogram,
        ),
        working_cache
            .as_ref()
            .is_some_and(|cache| cache.properties.iter().any(|p| p.valid)),
    )?;
    if !race.keep.is_empty() {
        #[cfg(test)]
        stats.race_ranges.lock().unwrap().push((
            samples.base() + indices[0],
            samples.base() + indices[0] + indices.len(),
        ));
        stats.race_nodes.fetch_add(1, Ordering::Relaxed);
        stats.race_draws.fetch_add(race.draws, Ordering::Relaxed);
        stats.race_pruned.fetch_add(
            race.keep.iter().filter(|&&keep| keep == 0).count(),
            Ordering::Relaxed,
        );
        if let Some(cache) = &mut working_cache {
            for (histogram, &keep) in cache.properties.iter_mut().zip(&race.keep) {
                if keep == 0 {
                    histogram.valid = false;
                }
            }
        }
    }
    let mut winners = [None; 4];
    let candidates: Vec<Result<(Candidates, bool)>> = {
        let cache_stride = working_cache.as_ref().and_then(|cache| cache.stride);
        let collection_stride = histogram_stride(
            parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
            scorer.lanes(),
            options.padded_histogram,
        );
        // Each indexed task owns a different mutable property. These temporary
        // mutexes adapt disjoint borrows to execution's shared Fn callback;
        // no cache-wide lock, copying, or new scheduler is needed.
        let property_jobs: Option<Vec<_>> = working_cache
            .as_mut()
            .map(|cache| cache.properties.iter_mut().map(Mutex::new).collect());
        let evaluate = |scratch: &mut PropertyScratch,
                        property: usize,
                        seen: Option<&mut Vec<SeenProperty>>| {
            if options.property_dedup && working_redundant[property] != 0 {
                stats.inherited_properties.fetch_add(1, Ordering::Relaxed);
                return Ok(([None; 4], true));
            }
            if !race.keep.is_empty() && race.keep[property] == 0 {
                return Ok(([None; 4], false));
            }
            with_property_values!(&samples.properties[property], |values| {
                let mut histogram = property_jobs
                    .as_ref()
                    .map(|jobs| jobs[property].lock().expect("MA property cache poisoned"));
                let cached_property = histogram.as_ref().is_some_and(|h| h.valid);
                let cache_scope = histogram.as_ref().map(|_| {
                    crate::profile::scope(if cached_property {
                        "ma.cache_subtract"
                    } else {
                        "ma.cache_collect"
                    })
                });
                if options.property_dedup {
                    if let Some(histogram) = histogram.as_mut() {
                        if cached_property {
                            histogram.subtract_marginals_values(samples, removed, values)?;
                        } else {
                            ***histogram = PropertyHistogram::collect_marginals_values(
                                samples, indices, property, values,
                            );
                        }
                    } else {
                        collect_property_marginals(
                            samples,
                            indices,
                            values,
                            samples.thresholds[property].len() + 1,
                            &mut scratch.used,
                            &mut scratch.weights,
                        );
                    }
                    let used = histogram
                        .as_ref()
                        .map_or(scratch.used.as_slice(), |h| h.used.as_slice());
                    if used.iter().filter(|&&n| n != 0).take(2).count() < 2 {
                        stats.constant_properties.fetch_add(1, Ordering::Relaxed);
                        return Ok(([None; 4], true));
                    }
                    if indices.len() >= 64
                        && let Some(seen) = seen
                        && duplicate_property(
                            samples,
                            indices,
                            options,
                            property,
                            values,
                            used,
                            seen,
                            &mut scratch.property_bins,
                        )
                    {
                        // Source Invalidate retains storage but forbids reuse of
                        // token tables that still describe the parent range.
                        if let Some(histogram) = histogram.as_mut() {
                            histogram.valid = false;
                        }
                        stats.duplicate_properties.fetch_add(1, Ordering::Relaxed);
                        return Ok(([None; 4], true));
                    }
                    if let Some(histogram) = histogram.as_mut() {
                        if cached_property {
                            if options.paired_cache && indices.len() >= 256 {
                                let pairs = histogram.subtract_paired_predictors_values(
                                    samples, removed, values, &plan,
                                )?;
                                stats
                                    .paired_cache_subtractions
                                    .fetch_add(pairs, Ordering::Relaxed);
                            } else {
                                histogram
                                    .subtract_predictors_values(samples, removed, values, &plan)?;
                            }
                        } else if options.histogram_shards {
                            histogram.collect_sharded_predictors_values(
                                samples,
                                indices,
                                parents,
                                values,
                                &plan,
                                cache_stride,
                                collection_stride,
                                &mut scratch.shards,
                            )?;
                        } else {
                            histogram.collect_predictors_values(
                                samples,
                                indices,
                                parents,
                                values,
                                &plan,
                                cache_stride,
                            )?;
                        }
                    }
                } else if let Some(histogram) = histogram.as_mut() {
                    if !cached_property {
                        // enc_ma.cc fills and scores on the same property worker.
                        // No second dispatch or all-properties fill barrier.
                        if options.histogram_shards {
                            ***histogram = PropertyHistogram::collect_marginals_values(
                                samples, indices, property, values,
                            );
                            histogram.collect_sharded_predictors_values(
                                samples,
                                indices,
                                parents,
                                values,
                                &plan,
                                cache_stride,
                                collection_stride,
                                &mut scratch.shards,
                            )?;
                        } else {
                            ***histogram = PropertyHistogram::collect_values_with_stride(
                                samples,
                                indices,
                                parents,
                                property,
                                values,
                                &plan,
                                cache_stride,
                            )?;
                        }
                    } else {
                        if options.paired_cache && indices.len() >= 256 {
                            histogram.subtract_marginals_values(samples, removed, values)?;
                            let pairs = histogram.subtract_paired_predictors_values(
                                samples, removed, values, &plan,
                            )?;
                            stats
                                .paired_cache_subtractions
                                .fetch_add(pairs, Ordering::Relaxed);
                        } else {
                            histogram.subtract_values(samples, removed, values, &plan)?;
                        }
                    }
                }
                drop(cache_scope);
                if cached_property && !removed.is_empty() {
                    stats.cache_subtractions.fetch_add(1, Ordering::Relaxed);
                }
                #[cfg(test)]
                if initialize_cache && let Some(hook) = &stats.initial_cache_hook {
                    hook(property, InitialCacheStage::Filled)?;
                }
                let candidates = evaluate_property_values(
                    samples,
                    indices,
                    options,
                    scoring_parents,
                    &plan,
                    &sparse_plan,
                    total,
                    property,
                    values,
                    histogram.as_ref().map(|histogram| &***histogram),
                    options.property_dedup,
                    compact.as_ref().map(|c| &c.mapping),
                    scratch,
                )?;
                if scratch.dominated_predictors != 0 {
                    stats
                        .dominated_predictors
                        .fetch_add(scratch.dominated_predictors, Ordering::Relaxed);
                }
                #[cfg(test)]
                if initialize_cache && let Some(hook) = &stats.initial_cache_hook {
                    hook(property, InitialCacheStage::Scored)?;
                }
                Ok((candidates, false))
            })
        };
        if parallel {
            scratch_pool
                .dispatch_slots
                .store(execution::worker_count(), Ordering::Relaxed);
            execution::map_with_state(
                options.properties.len(),
                || scratch_pool.acquire(),
                |lease, property| evaluate(&mut lease.scratch, property, None),
            )
        } else if !options.property_dedup {
            // Tiny serial nodes dominate the tree count. Reduce their results
            // immediately instead of allocating a vector of four-way candidates.
            let mut lease = scratch_pool.acquire();
            let mut first_error = None;
            for property in 0..options.properties.len() {
                match evaluate(&mut lease.scratch, property, None) {
                    Ok((candidates, _)) => {
                        for (best, candidate) in winners.iter_mut().zip(candidates) {
                            if let Some(candidate) = candidate {
                                keep_better(best, candidate);
                            }
                        }
                    }
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                }
            }
            // Like indexed parallel collection, drain the batch before
            // returning its first ordered error; never publish partial caches.
            if let Some(error) = first_error {
                return Err(error);
            }
            Vec::new()
        } else {
            let mut lease = scratch_pool.acquire();
            let mut seen = Vec::new();
            (0..options.properties.len())
                .map(|property| evaluate(&mut lease.scratch, property, Some(&mut seen)))
                .collect()
        }
    };
    // Indexed collection followed by strict ordered reduction preserves ties
    // independently of Rayon completion order. No worker mutates samples.
    for (property, candidates) in candidates.into_iter().enumerate() {
        let (candidates, redundant) = candidates?;
        if options.property_dedup {
            working_redundant[property] = u8::from(redundant);
        }
        for (best, candidate) in winners.iter_mut().zip(candidates) {
            if let Some(candidate) = candidate {
                keep_better(best, candidate);
            }
        }
    }
    let decision = choose_split(winners, base_bits, options);
    *cache = working_cache;
    *redundant_properties = working_redundant;
    Ok(decision)
}

/// Scalar index-only reference for SplitTreeSamples. The production learner
/// instead moves the columns, leaving its range indices contiguous.
fn partition_samples<T: PropertyValue>(
    indices: &mut [usize],
    values: &[T],
    bin: usize,
    below_count: usize,
) {
    let mut below = 0;
    let mut above = below_count;
    while below < below_count && above < indices.len() {
        while below < below_count && values[indices[below]].index() <= bin {
            below += 1;
        }
        while above < indices.len() && values[indices[above]].index() > bin {
            above += 1;
        }
        if below < below_count && above < indices.len() {
            indices.swap(below, above);
            below += 1;
            above += 1;
        }
    }
}

/// enc_ma.cc::SplitTreeSamples<..., true>. REORDER=false is instantiated only
/// by tests to compare complete searches against the former index-only path.
/// The expected left size is the distinct-record count, NOT the sample weight.
fn partition_node<const REORDER: bool>(
    samples: &mut Samples<impl SampleStorage>,
    indices: &mut [usize],
    property: usize,
    bin: usize,
    below_count: usize,
) {
    // Borrow the split column separately so its type dispatch stays outside
    // both scanning and swapping, without widening or moving its allocation.
    let (before, tail) = samples.properties.split_at_mut(property);
    let (column, after) = tail.split_first_mut().expect("MA split property");
    let mut swap_others = |left: &[usize], right: &[usize]| {
        for column in &mut samples.residuals {
            swap_column(column, left, right);
        }
        for column in before.iter_mut().chain(after.iter_mut()) {
            column.swap_pairs(left, right);
        }
        swap_column(&mut samples.counts, left, right);
    };
    with_property_values!(column, |values| {
        partition_values::<REORDER, _>(values, indices, bin, below_count, &mut swap_others);
    });
}

fn partition_values<const REORDER: bool, T: PropertyValue>(
    values: &mut [T],
    indices: &mut [usize],
    bin: usize,
    below_count: usize,
    mut swap_others: impl FnMut(&[usize], &[usize]),
) {
    if !REORDER {
        partition_samples(indices, values, bin, below_count);
        return;
    }
    const BATCH: usize = 128;
    let mut left = [0; BATCH];
    let mut right = [0; BATCH];
    let mut pairs = 0;
    let mut below = 0;
    let mut above = below_count;
    // Both cursors only advance, so deferred swap pairs are disjoint and
    // cannot affect any predicate still to be inspected in this partition.
    while below < below_count && above < indices.len() {
        while below < below_count && values[indices[below]].index() <= bin {
            below += 1;
        }
        while above < indices.len() && values[indices[above]].index() > bin {
            above += 1;
        }
        if below < below_count && above < indices.len() {
            left[pairs] = indices[below];
            right[pairs] = indices[above];
            pairs += 1;
            if pairs == BATCH {
                swap_column(values, &left, &right);
                swap_others(&left, &right);
                pairs = 0;
            }
            below += 1;
            above += 1;
        }
    }
    swap_column(values, &left[..pairs], &right[..pairs]);
    swap_others(&left[..pairs], &right[..pairs]);
}

fn leaf(predictor: Predictor) -> Node {
    scaled_leaf(predictor, 1)
}

fn scaled_leaf(predictor: Predictor, multiplier: u32) -> Node {
    Node::Leaf {
        predictor,
        offset: 0,
        multiplier,
    }
}

#[derive(Clone, Copy)]
struct TrainingNode {
    node: usize,
    begin: usize,
    end: usize,
    predictor: usize,
    multiplier: u32,
}

struct PendingNode {
    job: TrainingNode,
    range: StaticPropRange,
    parents: Option<Vec<ParentHistogram>>,
    cache: Option<HistogramCache>,
    removed: Option<std::ops::Range<usize>>,
    decision: Option<Option<Split>>,
    redundant_properties: Vec<u8>,
}

impl PendingNode {
    fn new(job: TrainingNode) -> Self {
        Self {
            job,
            range: [[0, u32::MAX]; 2],
            parents: None,
            cache: None,
            removed: None,
            decision: None,
            redundant_properties: Vec::new(),
        }
    }
    fn evaluate(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        options: &TrainOptions,
        budget: &Arc<HistogramBudget>,
        stats: &TrainingStats,
        scratch_pool: &ScratchPool,
    ) -> Result<()> {
        if self.decision.is_none() {
            let removed = self.removed.take().map_or(&[][..], |range| &indices[range]);
            self.decision = Some(find_best_split_with_properties(
                samples,
                &indices[self.job.begin..self.job.end],
                self.job.predictor,
                options,
                &mut self.parents,
                &mut self.cache,
                budget,
                stats,
                scratch_pool,
                removed,
                &mut self.redundant_properties,
            )?);
        }
        Ok(())
    }
}

/// Source forced splits, isolated from discretionary splits to guarantee that
/// the finite leaf budget cannot silently drop a multiplier on a pending node.
fn multiplier_skeleton<const REORDER: bool>(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    indices: &mut [usize],
    nodes: &mut Vec<Node>,
) -> Result<Vec<TrainingNode>> {
    let root = TrainingNode {
        node: 0,
        begin: 0,
        end: indices.len(),
        predictor: 0,
        multiplier: 1,
    };
    if options.multiplier_info.is_empty() {
        return Ok(vec![root]);
    }
    let range = options
        .static_property_range
        .ok_or_else(|| Error::InvalidInput("MA multipliers need a static range".into()))?;
    let mut pending = vec![(root, range)];
    let mut leaves = Vec::new();
    while let Some((mut job, range)) = pending.pop() {
        let mut intersection = Intersection::None;
        for info in &options.multiplier_info {
            intersection = box_intersects(range, info.range);
            match intersection {
                Intersection::None => continue,
                Intersection::Inside => job.multiplier = info.multiplier,
                Intersection::Partial { .. } => {}
            }
            break;
        }
        let Intersection::Partial { axis, threshold } = intersection else {
            nodes[job.node] = scaled_leaf(options.predictors[job.predictor], job.multiplier);
            leaves.push(job);
            continue;
        };
        if nodes.len().div_ceil(2) >= options.max_leaves {
            return Err(Error::Limit(
                "required MA multiplier splits exceed leaf budget",
            ));
        }
        let property = options
            .properties
            .iter()
            .position(|&p| p == axis)
            .ok_or_else(|| Error::InvalidInput("missing MA multiplier split property".into()))?;
        let bin = samples.thresholds[property]
            .binary_search(&threshold)
            .map_err(|_| Error::Encoding("missing exact MA multiplier threshold".into()))?;
        let below_count = samples.properties[property].count_le(&indices[job.begin..job.end], bin);
        partition_node::<REORDER>(
            samples,
            &mut indices[job.begin..job.end],
            property,
            bin,
            below_count,
        );
        let cut = job.begin + below_count;
        let greater = nodes.len();
        let less_equal = greater + 1;
        nodes[job.node] = Node::Split {
            property: axis,
            threshold,
            greater,
            less_equal,
        };
        nodes.push(leaf(options.predictors[job.predictor]));
        nodes.push(leaf(options.predictors[job.predictor]));
        let mut below_range = range;
        let mut above_range = range;
        below_range[axis][1] = threshold as u32 + 1;
        above_range[axis][0] = threshold as u32 + 1;
        pending.push((
            TrainingNode {
                node: less_equal,
                end: cut,
                ..job
            },
            below_range,
        ));
        pending.push((
            TrainingNode {
                node: greater,
                begin: cut,
                ..job
            },
            above_range,
        ));
    }
    Ok(leaves)
}

fn train_nodes(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
) -> Result<Vec<Node>> {
    train_nodes_with_stats(samples, options, &TrainingStats::default())
}

fn train_nodes_with_stats(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
) -> Result<Vec<Node>> {
    let slots = if options.parallel {
        options.properties.len().min(execution::worker_count())
            * if options.parallel_nodes { 2 } else { 1 }
    } else {
        1
    };
    train_nodes_with_scratch(samples, options, stats, &ScratchPool::new(slots))
}

fn train_nodes_with_scratch(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
) -> Result<Vec<Node>> {
    // Resolve once, at the invocation root. Descendant learners receive local
    // allowances from this budget; they must not reset to the global ceiling.
    let resolved;
    let options = if let Some(limit) = options.max_merged_leaves {
        resolved = TrainOptions {
            max_leaves: limit,
            max_merged_leaves: None,
            ..options.clone()
        };
        &resolved
    } else {
        options
    };
    if let Some(nodes) = ma_channel::train(samples, options, stats, scratch_pool)? {
        return Ok(nodes);
    }
    train_nodes_with_layout::<true>(samples, options, stats, scratch_pool)
}

fn train_nodes_with_layout<const REORDER: bool>(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
) -> Result<Vec<Node>> {
    let mut indices: Vec<usize> = (0..samples.counts.len()).collect();
    let mut nodes = vec![leaf(options.predictors[0])];
    let roots = multiplier_skeleton::<REORDER>(samples, options, &mut indices, &mut nodes)?;
    train_roots_with_layout::<REORDER>(
        samples,
        options,
        stats,
        scratch_pool,
        &mut indices,
        nodes,
        roots,
    )
}

fn train_roots_with_layout<const REORDER: bool>(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
    indices: &mut [usize],
    nodes: Vec<Node>,
    roots: Vec<TrainingNode>,
) -> Result<Vec<Node>> {
    let ranges = ma_parallel::leaf_ranges(
        &nodes,
        options.static_property_range.unwrap_or([[0, u32::MAX]; 2]),
    );
    let pending = roots
        .into_iter()
        .map(|job| {
            let mut pending = PendingNode::new(job);
            pending.range = ranges[job.node];
            pending
        })
        .collect();
    train_pending_with_layout::<REORDER>(
        samples,
        options,
        stats,
        scratch_pool,
        indices,
        nodes,
        pending,
    )
}

fn train_pending_with_layout<const REORDER: bool>(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
    indices: &mut [usize],
    nodes: Vec<Node>,
    pending: Vec<PendingNode>,
) -> Result<Vec<Node>> {
    let frontier = REORDER
        .then(|| ma_frontier::Frontier::new(options, samples.counts.len()))
        .flatten();
    train_pending_in_frontier::<REORDER>(
        samples,
        options,
        stats,
        scratch_pool,
        indices,
        nodes,
        pending,
        &HistogramBudget::new(options.histogram_cache_bytes),
        frontier.as_ref(),
    )
}

#[allow(clippy::too_many_arguments)]
fn train_pending_in_frontier<const REORDER: bool>(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
    indices: &mut [usize],
    mut nodes: Vec<Node>,
    mut pending: Vec<PendingNode>,
    budget: &Arc<HistogramBudget>,
    frontier: Option<&ma_frontier::Frontier>,
) -> Result<Vec<Node>> {
    let _scope = crate::profile::scope("ma.train_nodes");
    let records: usize = pending
        .iter()
        .map(|root| root.job.end - root.job.begin)
        .sum();
    let detach =
        REORDER && frontier.is_none() && options.parallel_tools.enabled(records, options.parallel);
    let mut subtrees = Vec::new();
    if frontier.is_none() {
        scratch_pool.dispatch_slots.store(0, Ordering::Relaxed);
    }
    // Include finished/reserved seed leaves, not just the ready work queue.
    let mut leaves = nodes.len().div_ceil(2);
    if leaves > options.max_leaves {
        return Err(Error::Limit("MA skeleton exceeds leaf budget"));
    }
    pending.reverse();
    while let Some(mut state) = pending.pop() {
        let job = state.job;
        let TrainingNode {
            node,
            begin,
            end,
            multiplier,
            ..
        } = job;
        if leaves == options.max_leaves && !options.complete_search() {
            break;
        }
        if begin == end {
            continue;
        }
        let slots = scratch_pool.dispatch_slots.load(Ordering::Relaxed);
        if detach && slots > 1 && end - begin <= options.parallel_tools.grain(records, slots) {
            let history = options
                .parallel_tools
                .inherit_histograms
                .then(|| ma_frontier::History::take(&mut state, samples, indices, options, stats))
                .transpose()?;
            subtrees
                .try_reserve(1)
                .map_err(|_| Error::Limit("MA subtree job allocation"))?;
            subtrees.push(ma_parallel::Detached {
                destination: node,
                records: begin..end,
                range: state.range,
                redundant: std::mem::take(&mut state.redundant_properties),
                seed: nodes[node].clone(),
                predictor: job.predictor,
                multiplier,
                history,
            });
            stats.detached_subtrees.fetch_add(1, Ordering::Relaxed);
            // Parent/cache state can refer outside this range. Drop it before
            // any disjoint task starts physically swapping its own columns.
            continue;
        }
        state.evaluate(samples, indices, options, budget, stats, scratch_pool)?;
        let Some(split) = state.decision.flatten() else {
            continue;
        };
        if leaves == options.max_leaves {
            return Err(Error::Limit(
                "MA leaf budget exhausted with beneficial splits remaining",
            ));
        }
        let cut = begin + split.below_count;
        partition_node::<REORDER>(
            samples,
            &mut indices[begin..end],
            split.property,
            split.bin,
            split.below_count,
        );
        let greater = nodes.len();
        let less_equal = greater + 1;
        nodes.push(scaled_leaf(
            options.predictors[split.above_predictor],
            multiplier,
        ));
        nodes.push(scaled_leaf(
            options.predictors[split.below_predictor],
            multiplier,
        ));
        nodes[node] = Node::Split {
            property: options.properties[split.property],
            threshold: samples.thresholds[split.property][split.bin],
            greater,
            less_equal,
        };
        leaves += 1;
        let mut below = PendingNode::new(TrainingNode {
            node: less_equal,
            end: cut,
            predictor: split.below_predictor,
            ..job
        });
        let mut above = PendingNode::new(TrainingNode {
            node: greater,
            begin: cut,
            predictor: split.above_predictor,
            ..job
        });
        below.redundant_properties = state.redundant_properties.clone();
        above.redundant_properties = std::mem::take(&mut state.redundant_properties);
        (above.range, below.range) = ma_parallel::children(
            state.range,
            options.properties[split.property],
            samples.thresholds[split.property][split.bin],
        );
        if leaves == options.max_leaves && !options.complete_search() {
            break;
        }
        let low = cut - begin;
        let high = end - cut;
        let unbalanced = low.min(high) <= (end - begin - 1) / 3;
        let slots = scratch_pool.dispatch_slots.load(Ordering::Relaxed);
        let subtree_grain = options.parallel_tools.grain(records, slots);
        let detach_children = detach && slots > 1 && low.max(high) <= subtree_grain;
        let (smaller, larger, removed) = if low <= high {
            (&mut below, &mut above, &indices[begin..cut])
        } else {
            (&mut above, &mut below, &indices[cut..end])
        };
        let inherit = options.parallel_tools.inherit_histograms;
        if options.reuse_parent_histograms
            && (unbalanced || inherit)
            && low.max(high) >= 512
            && (!detach_children || inherit)
        {
            let smaller_parents = parent_histograms(samples, removed);
            larger.parents = Some(subtract_parent_histograms(
                state.parents.take().expect("split has parent histograms"),
                &smaller_parents,
            )?);
            smaller.parents = Some(smaller_parents);
            stats.parent_reuses.fetch_add(1, Ordering::Relaxed);
        }
        if (unbalanced || inherit)
            && low.max(high) >= options.histogram_cache_thresholds().1
            && let Some(cache) = state.cache.take()
        {
            larger.cache = Some(cache);
            // Descendant partitioning can reorder this interval, not change
            // its membership. Subtract in each property's later scoring task.
            larger.removed = Some(smaller.job.begin..smaller.job.end);
        }
        // Release a non-inherited cache before reserving any ready-child work.
        drop(state);
        if let Some(frontier) = frontier
            && let Some(permit) = frontier.acquire(low, high)
        {
            ma_frontier::fork(
                samples,
                indices,
                options,
                stats,
                scratch_pool,
                budget,
                frontier,
                permit,
                &mut nodes,
                above,
                below,
            )?;
            leaves = nodes.len().div_ceil(2);
            continue;
        }
        let work = (end - begin)
            .saturating_mul(options.predictors.len())
            .saturating_mul(options.properties.len());
        if options.parallel
            && options.parallel_nodes
            && !(detach && slots > 1 && low.min(high) <= subtree_grain)
            && execution::worker_count() > 1
            && low.min(high) >= 512
            && low.max(high) <= 16_384
            && work >= 65_536
        {
            // Both read-only child searches finish before the next column move.
            let shared_samples = &*samples;
            let (high_result, low_result) = execution::join(
                || {
                    above.evaluate(
                        shared_samples,
                        indices,
                        options,
                        budget,
                        stats,
                        scratch_pool,
                    )
                },
                || {
                    below.evaluate(
                        shared_samples,
                        indices,
                        options,
                        budget,
                        stats,
                        scratch_pool,
                    )
                },
            );
            high_result?;
            low_result?;
            stats.ready_nodes.fetch_add(2, Ordering::Relaxed);
        }
        // Search completion order never changes the leaf-budget commit order.
        pending.push(below);
        pending.push(above);
    }
    drop(pending);
    ma_parallel::finish(
        samples,
        options,
        stats,
        &mut nodes,
        subtrees,
        scratch_pool.dispatch_slots.load(Ordering::Relaxed),
        budget,
    )?;
    leaves = nodes.len().div_ceil(2);
    if options.parallel_tools.dynamic_subtrees || options.parallel_tools.inherit_histograms {
        crate::profile::note(
            "ma.frontier",
            format_args!(
                "dynamic_forks={} histogram_handoffs={}",
                stats.dynamic_forks.load(Ordering::Relaxed),
                stats.histogram_handoffs.load(Ordering::Relaxed),
            ),
        );
    }
    crate::profile::note(
        "ma.search_counts",
        format_args!(
            "distinct={} nodes={} leaves={} cache_peak_bytes={} cache_builds={} cache_reuses={} cache_subtractions={} parent_reuses={} ready_nodes={} scratch_created={} scratch_reused={} equivalent_predictors={} constant_properties={} duplicate_properties={} inherited_properties={} compact_nodes={} paired_cache_subtractions={} race_nodes={} race_draws={} race_pruned={} dominated_predictors={}",
            records,
            nodes.len(),
            leaves,
            budget.peak.load(Ordering::Relaxed),
            stats.cache_builds.load(Ordering::Relaxed),
            stats.cache_reuses.load(Ordering::Relaxed),
            stats.cache_subtractions.load(Ordering::Relaxed),
            stats.parent_reuses.load(Ordering::Relaxed),
            stats.ready_nodes.load(Ordering::Relaxed),
            scratch_pool.created.load(Ordering::Relaxed),
            scratch_pool.reused.load(Ordering::Relaxed),
            stats.equivalent_predictors.load(Ordering::Relaxed),
            stats.constant_properties.load(Ordering::Relaxed),
            stats.duplicate_properties.load(Ordering::Relaxed),
            stats.inherited_properties.load(Ordering::Relaxed),
            stats.compact_nodes.load(Ordering::Relaxed),
            stats.paired_cache_subtractions.load(Ordering::Relaxed),
            stats.race_nodes.load(Ordering::Relaxed),
            stats.race_draws.load(Ordering::Relaxed),
            stats.race_pruned.load(Ordering::Relaxed),
            stats.dominated_predictors.load(Ordering::Relaxed),
        ),
    );
    Ok(nodes)
}

fn constant_leaf(value: i32) -> Node {
    Node::Leaf {
        predictor: Predictor::Zero,
        offset: value,
        multiplier: 1,
    }
}

/// enc_encoding.cc::LearnTree's trailing-channel splice, in source order.
fn append_constants(nodes: &mut Vec<Node>, constants: &[ConstantChannel]) {
    for constant in constants {
        for node in nodes.iter_mut() {
            if let Node::Split {
                greater,
                less_equal,
                ..
            } = node
            {
                *greater += 2;
                *less_equal += 2;
            }
        }
        nodes.insert(0, constant_leaf(constant.value));
        nodes.insert(
            0,
            Node::Split {
                property: 0,
                threshold: constant.channel as i32 - 1,
                greater: 1,
                less_equal: 2,
            },
        );
    }
}

fn validate_sample_constraints(
    tree: &Tree,
    samples: &[Sample],
    options: &TrainOptions,
) -> Result<()> {
    if options.multiplier_info.is_empty() && options.constant_channels.is_empty() {
        return Ok(());
    }
    for sample in samples.iter().filter(|s| s.weight != 0) {
        let (_, predictor, offset, multiplier) = tree.lookup(&sample.properties)?;
        let index = options.predictors.iter().position(|&p| p == predictor);
        if let Some(constant) = options
            .constant_channels
            .iter()
            .find(|constant| constant.channel == sample.properties[0] as u32)
        {
            // Zero residuals are the pixel values; other predictors alone do not
            // reconstruct a sample's pixel. Full-plane proof remains mandatory.
            if let Some(index) = index
                && sample.residuals[index] != constant.value
            {
                return Err(Error::InvalidInput(
                    "MA constant contradicts a sampled pixel".into(),
                ));
            }
            continue;
        }
        let index = index
            .ok_or_else(|| Error::Encoding("MA leaf predictor missing from samples".into()))?;
        let residual = i64::from(sample.residuals[index]) - i64::from(offset);
        // AddSample has lost multiples of 2^32. Divisibility by m implies only
        // divisibility of this residue by gcd(m, 2^32), not by m itself.
        let common_factor = 1i64 << multiplier.trailing_zeros();
        if residual % common_factor != 0 {
            return Err(Error::InvalidInput(
                "MA multiplier is incompatible with a narrowed sampled residual".into(),
            ));
        }
    }
    Ok(())
}

fn train_with_constants(
    options: &TrainOptions,
    train_variable: impl FnOnce(&TrainOptions) -> Result<Vec<Node>>,
) -> Result<Tree> {
    let leaf_limit = options.tree_leaf_limit();
    let mut nodes;
    let mut constants = options.constant_channels.as_slice();
    if let Some(first) = constants.first() {
        let mut range = options
            .static_property_range
            .ok_or_else(|| Error::InvalidInput("MA constants need a static range".into()))?;
        if first.channel == range[0][0] {
            nodes = vec![constant_leaf(first.value)];
            constants = &constants[1..];
        } else {
            range[0][1] = first.channel;
            let remaining = leaf_limit
                .checked_sub(constants.len())
                .filter(|&limit| limit != 0)
                .ok_or(Error::Limit("MA constant channels exceed leaf budget"))?;
            let learning = TrainOptions {
                max_leaves: options.max_leaves.min(remaining),
                max_merged_leaves: options.max_merged_leaves.map(|_| remaining),
                static_property_range: Some(range),
                constant_channels: Vec::new(),
                ..options.clone()
            };
            nodes = train_variable(&learning)?;
        }
    } else {
        nodes = train_variable(options)?;
    }
    if nodes.len().div_ceil(2).saturating_add(constants.len()) > leaf_limit {
        return Err(Error::Limit("MA constant channels exceed leaf budget"));
    }
    append_constants(&mut nodes, constants);
    Tree::new(nodes)
}

/// Consume source gather columns without requantizing or deduplicating records.
/// The gatherer must exclude the proven constant suffix before collecting;
/// quantized columns cannot establish raw channel IDs or multiplier evidence.
pub(crate) fn train_prepared(
    samples: super::training::gather_samples::PreparedSamples,
    options: &TrainOptions,
) -> Result<Tree> {
    prepared_storage::train(samples.into(), options)
}

/// Learn an adaptive tree, honoring proven constants and supplied multipliers.
/// No constraints means the existing source-seed behavior on empty samples.
/// With constraints, even unsampled static regions get their required leaves.
/// The encoder must check `(pixel - prediction - offset) % multiplier == 0`
/// in wide arithmetic on EVERY emitted pixel, then divide before narrowing the
/// quotient. Sample checks here cannot establish that invariant or odd factors.
pub fn train(samples: &[Sample], options: &TrainOptions) -> Result<Tree> {
    let _scope = crate::profile::scope("ma.train");
    crate::profile::note(
        "ma.train_input",
        format_args!(
            "records={} constants={} multipliers={} max_leaves={} complete_search={}",
            samples.len(),
            options.constant_channels.len(),
            options.multiplier_info.len(),
            options.max_leaves,
            options.complete_search(),
        ),
    );
    options.validate()?;
    validate_samples(samples, options)?;
    let tree = train_with_constants(options, |learning| {
        let variable_samples = samples.iter().filter(|s| {
            options
                .constant_channels
                .first()
                .is_none_or(|first| s.weight != 0 && (s.properties[0] as u32) < first.channel)
        });
        let mut prepared = TreeSamples::new(variable_samples, learning)?;
        train_nodes(&mut prepared, learning)
    })?;
    validate_sample_constraints(&tree, samples, options)?;
    Ok(tree)
}

#[cfg(test)]
#[path = "ma_budget_mode_tests.rs"]
mod budget_mode_tests;

#[cfg(test)]
#[path = "ma_cache_pair_tests.rs"]
mod cache_pair_tests;

#[cfg(test)]
#[path = "ma_cache_admission_tests.rs"]
mod cache_admission_tests;

#[cfg(test)]
#[path = "ma_cache_pair14_tests.rs"]
mod cache_pair14_tests;

#[cfg(test)]
#[path = "ma_pruning_tests.rs"]
mod pruning_tests;

#[cfg(test)]
#[path = "ma_train_tests.rs"]
mod tests;
