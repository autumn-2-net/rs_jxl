//! Independent, encoder-only block decisions compiled to standard JXL MA.
//!
//! The conceptual reference is QLIC's tiled predictor selection, not its
//! predictors, arithmetic, map coding or bitstream. Prediction here comes from
//! the existing Rust port of libjxl context_predict.h at
//! b866b3defec13494bfbaf705db93581cda118e37. No foreign codec code is used.
//!
//! Each terminal rectangle scores all its original transformed samples with
//! Zero/Left/Top/Gradient (wire IDs 0/1/2/5). For residual r we estimate
//! H(HybridUint(PackSigned(r))) + extra bits + histogram/leaf description cost,
//! using fast_context's existing estimator. Parents compare a pooled predictor
//! against their children plus a split-description estimate; children coalesce
//! only when pooling costs less. Equal predictors may retain distinct entropy
//! contexts. Strict comparisons give deterministic candidate-order
//! ties and prefer the unsplit plan on a rate tie.
//!
//! Only properties 0 (original channel index), 1 (actual stream ID), 2 (local y)
//! and 3 (local x) are emitted. Tiles never reset causal neighbors. Metadata is
//! neither omitted nor renumbered. A single bounded plan spans the supplied
//! learning domain, including when tree-group scheduling was requested: optional
//! sharding is coarsened rather than imposing a minimum number of leaves.
//!
//! Limitations: this is a balanced tile-aligned hierarchy, not an exhaustive
//! partition optimizer. Budget is divided before rate pruning, without reclaiming
//! pruned allocations. At most max_plan_leaves terminal candidates are explored;
//! larger grids are coarsened, never rejected. Coordinates whose i32 property
//! would wrap are not split on that axis. The estimate does not model final
//! histogram clustering, selected UintConfig, LZ77, ANS normalization or actual
//! tree entropy, so smaller estimates do not guarantee smaller codestreams.

mod plan;

// Reuse the native estimator without changing fast_context's ownership/API.
#[allow(dead_code)]
#[path = "fast_context/stats.rs"]
mod stats;

use std::sync::{Arc, Mutex};

use super::{ma_train::ChannelView, predictor::Predictor, tree::Tree};
use crate::{Error, Result};

/// Stable receipt order; these are standard JXL IDs 0, 1, 2, and 5.
pub const PREDICTORS: [Predictor; 4] = [
    Predictor::Zero,
    Predictor::Left,
    Predictor::Top,
    Predictor::Gradient,
];

/// Optional planning-work bound, not an input size or normative format limit.
pub const DEFAULT_PLAN_LEAVES: usize = 4096;

#[derive(Clone, Debug)]
pub struct Options {
    /// Requested tile edge, exactly 8 or 16; budgets may coarsen the plan.
    pub tile_size: usize,
    /// Configurable planning-work ceiling; excess regions coarsen, not fail.
    pub max_plan_leaves: usize,
    pub receipt: Option<Arc<Receipt>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tile_size: 16,
            max_plan_leaves: DEFAULT_PLAN_LEAVES,
            receipt: None,
        }
    }
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.max_plan_leaves == 0 {
            return Err(Error::InvalidInput(
                "zero block-context planning budget".into(),
            ));
        }
        if !matches!(self.tile_size, 8 | 16) {
            return Err(Error::InvalidInput(
                "block_context tile_size must be 8 or 16".into(),
            ));
        }
        Ok(())
    }
}

/// Cumulative successful learner invocations. Read after encoding joins; probes
/// also count when the caller deliberately shares a collector with them.
#[derive(Debug, Default)]
pub struct Receipt(Mutex<Snapshot>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub trees: usize,
    /// Every original transformed sample, including metadata and hidden RGB.
    pub pixels: usize,
    /// Nonempty transformed channel views, not distinct channel numbers.
    pub views: usize,
    pub requested_tiles: usize,
    pub leaves: usize,
    pub spatial_splits: usize,
    pub channel_splits: usize,
    pub stream_splits: usize,
    /// Views not independently explored at every requested tile because of a
    /// leaf/work budget or wrapped coordinate axis. Rate pruning is excluded.
    pub coarsened_views: usize,
    /// Emitted leaf counts in PREDICTORS order (not dense wire-ID order).
    pub predictor_leaves: [usize; 4],
}

impl Receipt {
    pub fn snapshot(&self) -> Snapshot {
        *self.0.lock().expect("block-context receipt poisoned")
    }

    fn record(&self, value: Snapshot) {
        let mut total = self.0.lock().expect("block-context receipt poisoned");
        total.trees += value.trees;
        total.pixels += value.pixels;
        total.views += value.views;
        total.requested_tiles += value.requested_tiles;
        total.leaves += value.leaves;
        total.spatial_splits += value.spatial_splits;
        total.channel_splits += value.channel_splits;
        total.stream_splits += value.stream_splits;
        total.coarsened_views += value.coarsened_views;
        for (to, from) in total
            .predictor_leaves
            .iter_mut()
            .zip(value.predictor_leaves)
        {
            *to += from;
        }
    }
}

/// The shared training entry point has validated views and original IDs.
pub(super) fn learn(views: &[ChannelView<'_>], limit: usize, options: &Options) -> Result<Tree> {
    let _scope = crate::profile::scope("ma.block_context");
    options.validate()?;
    if limit == 0 {
        return Err(Error::InvalidInput("zero block-context leaf budget".into()));
    }
    let mut ordered: Vec<_> = views
        .iter()
        .filter(|v| v.width != 0 && v.height != 0)
        .collect();
    ordered.sort_unstable_by_key(|v| (v.group, v.channel));
    let mut snapshot = Snapshot {
        trees: 1,
        views: ordered.len(),
        ..Default::default()
    };
    for view in &ordered {
        let area = view
            .width
            .checked_mul(view.height)
            .ok_or(Error::Limit("block-context population overflow"))?;
        snapshot.pixels = snapshot
            .pixels
            .checked_add(area)
            .ok_or(Error::Limit("block-context population overflow"))?;
        // Tile count is no larger than the checked sample population.
        snapshot.requested_tiles +=
            view.width.div_ceil(options.tile_size) * view.height.div_ceil(options.tile_size);
    }
    let candidate = plan::domain(
        &ordered,
        limit.min(options.max_plan_leaves),
        options.tile_size,
    )?;
    snapshot.coarsened_views = candidate.coarsened_views;
    let mut nodes = Vec::new();
    candidate.plan.append(&mut nodes);
    let tree = Tree::new(nodes)?;
    snapshot.leaves = tree.leaves();
    for node in tree.nodes() {
        match node {
            super::tree::Node::Split { property: 0, .. } => snapshot.channel_splits += 1,
            super::tree::Node::Split { property: 1, .. } => snapshot.stream_splits += 1,
            super::tree::Node::Split { .. } => snapshot.spatial_splits += 1,
            super::tree::Node::Leaf { predictor, .. } => {
                let p = PREDICTORS
                    .iter()
                    .position(|p| p == predictor)
                    .expect("block-context emits only standard basic predictors");
                snapshot.predictor_leaves[p] += 1;
            }
        }
    }
    debug_assert!(tree.leaves() <= limit.min(options.max_plan_leaves));
    if let Some(receipt) = &options.receipt {
        receipt.record(snapshot);
    }
    Ok(tree)
}

#[cfg(test)]
mod tests;
