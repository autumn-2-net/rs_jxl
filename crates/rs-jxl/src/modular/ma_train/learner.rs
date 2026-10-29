// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Required multiplier skeletons, column partitioning and depth-first tree training.
//! This layer owns node/leaf budgets, child commit order and histogram handoff scheduling.

use super::{
    Arc, HistogramBudget, HistogramCache, Intersection, Node, Ordering, ParentHistogram, Predictor,
    PropertyColumn, PropertyValue, SampleStorage, Samples, ScratchPool, Split, StaticPropRange,
    TrainOptions, TrainingStats, box_intersects, execution, find_best_split_with_properties,
    ma_channel, ma_frontier, ma_parallel, parent_histograms, subtract_parent_histograms,
    swap_column,
};
use crate::{Error, Result};

/// Scalar index-only reference for SplitTreeSamples. The production learner
/// instead moves the columns, leaving its range indices contiguous.
pub(super) fn partition_samples<T: PropertyValue>(
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
pub(super) fn partition_node<const REORDER: bool>(
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

pub(super) fn leaf(predictor: Predictor) -> Node {
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
pub(super) struct TrainingNode {
    pub(super) node: usize,
    pub(super) begin: usize,
    pub(super) end: usize,
    pub(super) predictor: usize,
    pub(super) multiplier: u32,
}

pub(super) struct PendingNode {
    pub(super) job: TrainingNode,
    pub(super) range: StaticPropRange,
    pub(super) parents: Option<Vec<ParentHistogram>>,
    pub(super) cache: Option<HistogramCache>,
    pub(super) removed: Option<std::ops::Range<usize>>,
    pub(super) decision: Option<Option<Split>>,
    pub(super) redundant_properties: Vec<u8>,
}

impl PendingNode {
    pub(super) fn new(job: TrainingNode) -> Self {
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
    pub(super) fn evaluate(
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

pub(super) fn train_nodes(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
) -> Result<Vec<Node>> {
    train_nodes_with_stats(samples, options, &TrainingStats::default())
}

pub(super) fn train_nodes_with_stats(
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

pub(super) fn train_nodes_with_scratch(
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

pub(super) fn train_nodes_with_layout<const REORDER: bool>(
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

pub(super) fn train_roots_with_layout<const REORDER: bool>(
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

pub(super) fn train_pending_with_layout<const REORDER: bool>(
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
pub(super) fn train_pending_in_frontier<const REORDER: bool>(
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
