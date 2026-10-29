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
//!
//! This module owns the public facade, constraint handling and configuration
//! reexports. Private children separate sample preparation, parent histograms,
//! histogram caches, scratch leases, split scoring/search and tree orchestration.
//! Existing sibling helpers and test modules retain their qualified paths.

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

// Keep private compatibility names here for the existing helpers and tests.
mod preparation;
use preparation::{ResidualToken, SampleKey, validate_samples};
#[cfg(test)]
use preparation::{SampleBatchCollector, property_thresholds, quantize_histogram, quantized_key};

mod parent_histograms;
#[cfg(test)]
use parent_histograms::compact_parent_histograms;
use parent_histograms::{
    ParentHistogram, compact_parent_histograms_with_layout, parent_histogram, parent_histograms,
    subtract_parent_histograms,
};

mod cache;
use cache::{
    BinHistogram, HistogramBudget, HistogramCache, PropertyHistogram, collect_property_marginals,
};

mod scratch;
use scratch::{PropertyScratch, ScratchPool};

mod split_scoring;
use split_scoring::{
    Candidates, Cost, PredictorPlan, Split, choose_split, evaluate_property_values, keep_better,
};
#[cfg(test)]
use split_scoring::{evaluate_property, evaluate_property_cached, predictor_penalty};

mod split_search;
use split_search::find_best_split_with_properties;
#[cfg(test)]
use split_search::{
    SeenProperty, duplicate_property, find_best_split, find_best_split_with_scratch,
    property_signature,
};

mod learner;
use learner::{
    PendingNode, TrainingNode, leaf, partition_node, train_nodes, train_pending_in_frontier,
    train_roots_with_layout,
};
#[cfg(test)]
use learner::{
    partition_samples, train_nodes_with_layout, train_nodes_with_scratch, train_nodes_with_stats,
    train_pending_with_layout,
};

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
