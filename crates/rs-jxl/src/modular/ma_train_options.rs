// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! MA training controls, defaults and validation.
//! Source: libjxl enc_ma.h/cc and options.h at
//! b866b3defec13494bfbaf705db93581cda118e37.

use super::{
    ChannelTreeMode, ConstantChannel, DenseHistogram, MAX_LEAVES, MAX_TREE_LEAVES, MultiplierInfo,
    ParallelOptions, Predictor, PropertyBins, PropertyQuantization, QuantileSplits, ScoringOptions,
    SearchOptions, StaticPropRange, intersect_ranges, validate_range,
};
use crate::{Error, Result};

/// Behavior when an MA learner runs out of leaf capacity. Required multiplier,
/// channel and constant partitions must fit in either mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaBudgetMode {
    /// Fail if a beneficial adaptive split remains at the ceiling.
    Strict,
    /// Stop adaptive splitting at the ceiling, retaining valid lossless leaves.
    Bounded,
}

/// Explicit histogram experiments; all are disabled by default.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HistogramOptions {
    pub shards: bool,
    pub padded: bool,
    pub dense: DenseHistogram,
    pub property_bins: PropertyBins,
    pub sparse: bool,
    pub paired: bool,
    /// Sparse property events with the same clipped entropy objective. Uncached
    /// full sweeps only, weighted total <=65536; dense work keeps the SIMD path.
    pub events: bool,
    /// Earlier bounded cache creation/inheritance for fourteen candidates only.
    pub early_cache_14: bool,
}

impl HistogramOptions {
    /// Forward only histogram experiments, leaving search and budgets unchanged.
    pub fn apply(&self, options: &mut TrainOptions) {
        options.histogram_shards = self.shards;
        options.padded_histogram = self.padded;
        options.dense_histogram = self.dense;
        options.property_bins = self.property_bins;
        options.sparse_entropy = self.sparse;
        options.paired_cache = self.paired;
        options.histogram_events = self.events;
        options.early_cache_14 = self.early_cache_14;
    }
}

#[derive(Clone, Debug)]
pub struct TrainOptions {
    /// Bitstream predictors only; Best/Variable are modes, not candidates.
    pub predictors: Vec<Predictor>,
    /// Property IDs to search, in stable tie-breaking order. 1 is opt-in.
    pub properties: Vec<usize>,
    /// Legacy invocation ceiling, 1..=MAX_LEAVES. Within a resolved learner this
    /// is its local allowance, including its seed leaves.
    pub max_leaves: usize,
    /// Optional whole-tree resource ceiling, at most MAX_TREE_LEAVES. None
    /// retains max_leaves as the whole-tree ceiling. Source callers resolve this
    /// separately from the legacy preset and clamp it to decoder geometry limits.
    /// Constants and all channel/subtree grafts consume this same budget.
    pub max_merged_leaves: Option<usize>,
    /// At the resource ceiling, fail if any beneficial split remains instead
    /// of returning a truncated tree. Source integration enables this; explicit
    /// legacy leaf-budget searches retain the default false behavior.
    pub require_complete_search: bool,
    /// Overrides require_complete_search. None preserves the caller's existing
    /// policy (bounded for direct training, strict for source integration).
    pub budget_mode: Option<MaBudgetMode>,
    /// Required improvement in sampled bits; equality does not split.
    /// Scoring rounds this to f32, matching the source's default decision path.
    pub split_threshold: f64,
    /// 2..=256 quantile/coordinate bins; WP retains its source preset bins.
    pub max_property_values: usize,
    /// Source preference for static/non-WP splits; finite and at least 1.
    pub fast_decode_multiplier: f64,
    /// Parallel property sweeps above the source work/grain thresholds.
    pub parallel: bool,
    /// Reconstruct sufficiently unbalanced siblings from exact parent counts.
    pub reuse_parent_histograms: bool,
    /// Optional enc_ma.cc JXL_MA_PREDICTOR_DEDUP=1 verified predictor reuse.
    /// Defaults off; like the source, only active with more than two candidates.
    /// Rechecked per node because the current leaf changes predictor penalties.
    pub predictor_dedup: bool,
    /// Optional JXL_MA_PROPERTY_DEDUP=1 exact partition reuse, default off.
    /// Constants and verified redundant properties stay masked in descendants;
    /// cross-property proofs require a serial sweep with at least 64 records.
    pub property_dedup: bool,
    /// Skip predictor sweeps proved unable to improve either child. Limited to
    /// default scoring/search and weighted populations at most 256.
    pub split_dominance: bool,
    /// JXL_MA_HISTOGRAM_SHARDS=4; four independent uncached accumulators.
    pub histogram_shards: bool,
    /// JXL_MA_PADDED_HISTOGRAM=1; SIMD-aligned cache-set-avoiding strides.
    pub padded_histogram: bool,
    pub dense_histogram: DenseHistogram,
    pub property_bins: PropertyBins,
    /// JXL_MA_SPARSE_ENTROPY=1; preserve lanes while skipping parent-zero blocks.
    pub sparse_entropy: bool,
    /// JXL_MA_PAIRED_CACHE=1; adjacent active predictors share deferred loads.
    pub paired_cache: bool,
    /// Opt-in event-bucket sweep. Takes precedence over clipped/count-log only
    /// when eligible; Shannon and nondefault search options keep their routes.
    pub histogram_events: bool,
    /// Explicit source scoring experiments; default scoring remains unchanged.
    pub scoring: ScoringOptions,
    /// Default-off source property race and weighted split proposals.
    pub search: SearchOptions,
    /// Resolved JXL_MA_CHANNEL_TREE mode. Caller disables ineligible metadata,
    /// JPEG, probe or effort routes before training; learner gates still apply.
    pub channel_tree: ChannelTreeMode,
    /// Default-off whole-subtree tools and shared channel-root worker limits.
    pub parallel_tools: ParallelOptions,
    /// Shared cache ceiling per training invocation; zero disables retention.
    /// Source MaHistogramBudget::kLimit is 32 MiB, independent of raw leaf count.
    pub histogram_cache_bytes: usize,
    /// Opt-in 2048/1024-record cache creation/inheritance with fourteen
    /// configured predictors. Does not change the byte budget or split search.
    pub early_cache_14: bool,
    /// Evaluate ready siblings concurrently while preserving DFS commit order.
    /// Also requires parallel=true and the source work/grain thresholds.
    pub parallel_nodes: bool,
    /// Explicit population domain, required for multiplier/constant constraints.
    /// Sample properties 0 and 1 must use these same IDs, including after transforms.
    pub static_property_range: Option<StaticPropRange>,
    /// Source-supplied exact multipliers. Empty means ordinary multiplier 1.
    pub multiplier_info: Vec<MultiplierInfo>,
    /// Verified constants, preferably returned by find_constant_channels.
    /// These produce Zero-predictor leaves even when Zero is not a candidate.
    pub constant_channels: Vec<ConstantChannel>,
    /// Source shared-population quantiles. None selects the legacy fallback.
    /// Forced multiplier boundaries take precedence over these static bins.
    pub property_quantization: Option<PropertyQuantization>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            predictors: vec![Predictor::Weighted, Predictor::Gradient],
            properties: std::iter::once(0).chain(2..=15).collect(),
            max_leaves: 64,
            max_merged_leaves: None,
            require_complete_search: false,
            budget_mode: None,
            split_threshold: 16.0,
            max_property_values: 256,
            fast_decode_multiplier: 1.0,
            parallel: true,
            reuse_parent_histograms: true,
            predictor_dedup: false,
            property_dedup: false,
            split_dominance: false,
            histogram_shards: false,
            padded_histogram: false,
            dense_histogram: DenseHistogram::Disabled,
            property_bins: PropertyBins::Disabled,
            sparse_entropy: false,
            paired_cache: false,
            histogram_events: false,
            scoring: ScoringOptions::default(),
            search: SearchOptions::default(),
            channel_tree: ChannelTreeMode::Disabled,
            parallel_tools: ParallelOptions::default(),
            histogram_cache_bytes: 32 * 1024 * 1024,
            early_cache_14: false,
            parallel_nodes: true,
            static_property_range: None,
            multiplier_info: Vec::new(),
            constant_channels: Vec::new(),
            property_quantization: None,
        }
    }
}

impl TrainOptions {
    pub(in super::super) fn complete_search(&self) -> bool {
        self.budget_mode
            .map_or(self.require_complete_search, |mode| {
                mode == MaBudgetMode::Strict
            })
    }

    pub(in super::super) fn tree_leaf_limit(&self) -> usize {
        self.max_merged_leaves.unwrap_or(self.max_leaves)
    }

    /// SetPredictor(Variable): enum order with Weighted/Zero and Gradient/Left
    /// exchanged, not a fresh ordering after the first two entries.
    pub fn variable() -> Self {
        Self {
            predictors: vec![
                Predictor::Weighted,
                Predictor::Gradient,
                Predictor::Top,
                Predictor::Average0,
                Predictor::Select,
                Predictor::Left,
                Predictor::Zero,
                Predictor::TopRight,
                Predictor::TopLeft,
                Predictor::LeftLeft,
                Predictor::Average1,
                Predictor::Average2,
                Predictor::Average3,
                Predictor::Average4,
            ],
            ..Self::default()
        }
    }

    pub(super) fn histogram_cache_thresholds(&self) -> (usize, usize) {
        if self.early_cache_14 && self.predictors.len() == 14 {
            (2048, 1024)
        } else {
            (8192, 4096)
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        self.parallel_tools.workers.validate()?;
        if self.search.histogram_bins != 0
            && (!(2..=256).contains(&self.search.histogram_bins)
                || self.search.refine_quantiles
                || self.search.quantile_splits != QuantileSplits::Disabled
                || self.search.statistical_race)
        {
            return Err(Error::InvalidInput(
                "MA histogram bins must be 2..=256 and cannot combine with other split proposals"
                    .into(),
            ));
        }
        if self.predictors.is_empty()
            || self.predictors.len() > 14
            || self.predictors.iter().any(|&p| p as u32 >= 14)
            || self
                .predictors
                .iter()
                .enumerate()
                .any(|(i, p)| self.predictors[..i].contains(p))
        {
            return Err(Error::InvalidInput("invalid MA training predictors".into()));
        }
        if self.properties.is_empty()
            || self.properties.len() >= 64
            || self.properties.iter().any(|&p| p >= 256)
            || self
                .properties
                .iter()
                .enumerate()
                .any(|(i, p)| self.properties[..i].contains(p))
        {
            return Err(Error::InvalidInput("invalid MA training properties".into()));
        }
        if !(1..=MAX_LEAVES).contains(&self.max_leaves)
            || self
                .max_merged_leaves
                .is_some_and(|limit| !(1..=MAX_TREE_LEAVES).contains(&limit))
            || !(2..=256).contains(&self.max_property_values)
            || !self.split_threshold.is_finite()
            || !(self.split_threshold as f32).is_finite()
            || self.split_threshold < 0.0
            || !self.fast_decode_multiplier.is_finite()
            || !(self.fast_decode_multiplier as f32).is_finite()
            || self.fast_decode_multiplier < 1.0
        {
            return Err(Error::InvalidInput(
                "invalid MA training limits or costs".into(),
            ));
        }
        if let Some(quantization) = &self.property_quantization {
            if quantization.properties != self.properties
                || quantization.thresholds.len() != self.properties.len()
            {
                return Err(Error::InvalidInput(
                    "MA quantization property order mismatch".into(),
                ));
            }
            for (&property, thresholds) in self.properties.iter().zip(&quantization.thresholds) {
                let max_thresholds = if property < 2 { MAX_LEAVES - 1 } else { 255 };
                if thresholds.len() > max_thresholds
                    || thresholds.windows(2).any(|w| w[0] >= w[1])
                    || (property >= 2 && thresholds.iter().any(|&v| !(-512..=511).contains(&v)))
                {
                    return Err(Error::InvalidInput(
                        "invalid MA property quantization".into(),
                    ));
                }
            }
        }
        self.validate_constraints()
    }

    fn validate_constraints(&self) -> Result<()> {
        let Some(range) = self.static_property_range else {
            return if self.multiplier_info.is_empty() && self.constant_channels.is_empty() {
                Ok(())
            } else {
                Err(Error::InvalidInput(
                    "MA constraints require an explicit static property range".into(),
                ))
            };
        };
        validate_range(range)?;
        let count = u32::try_from(self.constant_channels.len())
            .map_err(|_| Error::Limit("MA constant channel count"))?;
        if count > range[0][1] - range[0][0] {
            return Err(Error::InvalidInput(
                "MA constants exceed channel range".into(),
            ));
        }
        let variable_end = range[0][1] - count;
        if self
            .constant_channels
            .iter()
            .enumerate()
            .any(|(i, constant)| constant.channel != variable_end + i as u32)
        {
            return Err(Error::InvalidInput(
                "MA constants must form an ascending channel suffix".into(),
            ));
        }
        if self.constant_channels.len() + usize::from(variable_end > range[0][0])
            > self.tree_leaf_limit()
        {
            return Err(Error::Limit("MA constant channels exceed leaf budget"));
        }
        let variable_range = [[range[0][0], variable_end], range[1]];
        for (i, info) in self.multiplier_info.iter().enumerate() {
            validate_range(info.range)?;
            if info.multiplier == 0 || info.multiplier > i32::MAX as u32 {
                return Err(Error::InvalidInput("invalid MA multiplier".into()));
            }
            let Some(overlap) = intersect_ranges(variable_range, info.range) else {
                continue;
            };
            for axis in 0..2 {
                if overlap[axis] != variable_range[axis] && !self.properties.contains(&axis) {
                    return Err(Error::InvalidInput(format!(
                        "MA multiplier boundary requires static property {axis}"
                    )));
                }
            }
            if self.multiplier_info[..i].iter().any(|previous| {
                previous.multiplier != info.multiplier
                    && intersect_ranges(overlap, previous.range).is_some()
            }) {
                return Err(Error::InvalidInput(
                    "conflicting overlapping MA multipliers".into(),
                ));
            }
        }
        Ok(())
    }
}
