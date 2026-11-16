// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; notice in docs/PORTING_KERNELS.md.
//! Coarse native greedy initialization and a single unfinished-frontier handoff.
//! Split/child handling follows ma_train/learner.rs (enc_ma.cc, b866b3d).

use super::super::{
    ChannelTreeMode, ClippedEntropyMode, HistogramBudget, Node, Ordering, PendingNode,
    QuantileSplits, SampleStorage, Samples, ScratchPool, SearchOptions, TrainOptions, TrainingNode,
    TrainingStats, TreeLearning, learner, ma_frontier, ma_parallel, parent_histograms,
    subtract_parent_histograms,
};
use crate::{Error, Result};
use std::sync::Arc;

// Initialization depth only: the ordinary learner finishes every unresolved root.
const COARSE_DEPTH: u8 = 3;

pub(super) fn eligible(options: &TrainOptions, records: usize) -> bool {
    options.seed.is_none()
        && options.lookahead.is_none()
        && options.channel_tree == ChannelTreeMode::Disabled
        && options.max_leaves > 4
        && records >= 512
}

fn coarse_options(options: &TrainOptions) -> TrainOptions {
    let mut coarse = options.clone();
    // Deliberately cheaper than e9's PR5007 interval/cost enumeration. Only the
    // initial topology uses Legacy; resume keeps the caller's complete policy.
    coarse.tree_learning = TreeLearning::Legacy;
    coarse.tree_learning_receipt = None;
    if options.tree_learning.enabled() {
        // Legacy cross-property partition proofs ignore PR5007 syntax costs
        // and symmetric eligibility; never transfer them to native learning.
        coarse.property_dedup = false;
    }
    if options.search == SearchOptions::default()
        && !options.scoring.fast_entropy
        && options.scoring.clipped == ClippedEntropyMode::Disabled
    {
        coarse.search.quantile_splits = QuantileSplits::Twelve;
    }
    coarse
}

#[derive(Default)]
struct Receipt {
    searched: usize,
    splits: usize,
    saturated: usize,
    fine_checks: usize,
    fine_decisions: usize,
    parent_handoffs: usize,
    cache_handoffs: usize,
}

struct Continuation {
    indices: Vec<usize>,
    nodes: Vec<Node>,
    pending: Vec<PendingNode>,
    budget: Arc<HistogramBudget>,
    receipt: Receipt,
}

fn build(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch: &ScratchPool,
    depth: u8,
) -> Result<Continuation> {
    let _scope = crate::profile::scope("ma.coarse_continue.build");
    let mut indices: Vec<_> = (0..samples.counts.len()).collect();
    let mut nodes = vec![learner::leaf(options.predictors[0])];
    let roots = learner::multiplier_skeleton::<true>(samples, options, &mut indices, &mut nodes)?;
    let ranges = ma_parallel::leaf_ranges(
        &nodes,
        options.static_property_range.unwrap_or([[0, u32::MAX]; 2]),
    );
    let mut work: Vec<_> = roots
        .into_iter()
        .rev()
        .map(|job| {
            let mut state = PendingNode::new(job);
            state.range = ranges[job.node];
            (state, depth)
        })
        .collect();
    let coarse = coarse_options(options);
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let mut pending = Vec::new();
    let mut receipt = Receipt::default();
    let mut leaves = nodes.len().div_ceil(2);
    if leaves > options.max_leaves {
        return Err(Error::Limit("MA coarse skeleton exceeds leaf budget"));
    }
    while let Some((mut state, depth)) = work.pop() {
        if leaves == options.max_leaves && !options.complete_search() {
            break;
        }
        let job = state.job;
        let records = job.end - job.begin;
        if records == 0 {
            continue;
        }
        if depth == 0 {
            pending.push(state);
            continue;
        }
        // prepare_mask is inactive for fewer than 512 records and for small
        // property alphabets. Use the exact route there, avoiding a second sweep.
        let restricted = coarse.tree_learning != options.tree_learning
            || (coarse.search != options.search
                && records >= 512
                && options
                    .properties
                    .iter()
                    .enumerate()
                    .any(|(p, &id)| id >= 2 && samples.thresholds[p].len() > 12));
        let search = if restricted { &coarse } else { options };
        receipt.searched += 1;
        state.evaluate(samples, &indices, search, &budget, stats, scratch)?;
        let Some(split) = state.decision.flatten() else {
            if restricted {
                // A coarse rejection is not a saturation proof. Retain raw
                // parent/cache/proof state and reprice all native candidates.
                receipt.fine_checks += 1;
                state.decision = None;
                state.evaluate(samples, &indices, options, &budget, stats, scratch)?;
                if state.decision.flatten().is_some() {
                    receipt.fine_decisions += 1;
                    pending.push(state);
                    continue;
                }
            }
            // Terminal decisions never enter the continuation queue.
            receipt.saturated += 1;
            continue;
        };
        debug_assert!(split.symmetric.is_none());
        if leaves == options.max_leaves {
            if options.complete_search() {
                return Err(Error::Limit(
                    "MA coarse leaf budget exhausted with beneficial splits remaining",
                ));
            }
            continue;
        }
        receipt.splits += 1;
        leaves += 1;
        let cut = job.begin + split.below_count;
        learner::partition_node::<true>(
            samples,
            &mut indices[job.begin..job.end],
            split.property,
            split.bin,
            split.below_count,
        );
        let greater = nodes.len();
        let less_equal = greater + 1;
        nodes.push(learner::scaled_leaf(
            options.predictors[split.above_predictor],
            job.multiplier,
        ));
        nodes.push(learner::scaled_leaf(
            options.predictors[split.below_predictor],
            job.multiplier,
        ));
        nodes[job.node] = Node::Split {
            property: options.properties[split.property],
            threshold: samples.thresholds[split.property][split.bin],
            greater,
            less_equal,
        };
        let mut above = PendingNode::new(TrainingNode {
            node: greater,
            begin: cut,
            predictor: split.above_predictor,
            ..job
        });
        let mut below = PendingNode::new(TrainingNode {
            node: less_equal,
            end: cut,
            predictor: split.below_predictor,
            ..job
        });
        above.redundant_properties = state.redundant_properties.clone();
        below.redundant_properties = std::mem::take(&mut state.redundant_properties);
        (above.range, below.range) = ma_parallel::children(
            state.range,
            options.properties[split.property],
            samples.thresholds[split.property][split.bin],
        );
        let (smaller, larger) = if cut - job.begin <= job.end - cut {
            (&mut below, &mut above)
        } else {
            (&mut above, &mut below)
        };
        if options.reuse_parent_histograms {
            let small = parent_histograms(samples, &indices[smaller.job.begin..smaller.job.end]);
            larger.parents = Some(subtract_parent_histograms(
                state
                    .parents
                    .take()
                    .expect("coarse split parent histograms"),
                &small,
            )?);
            smaller.parents = Some(small);
            receipt.parent_handoffs += 2;
            stats.parent_reuses.fetch_add(1, Ordering::Relaxed);
        }
        let unbalanced = smaller.job.end - smaller.job.begin <= (records - 1) / 3;
        if (unbalanced || options.parallel_tools.inherit_histograms)
            && larger.job.end - larger.job.begin >= options.histogram_cache_thresholds().1
            && let Some(cache) = state.cache.take()
        {
            larger.cache = Some(cache);
            larger.removed = Some(smaller.job.begin..smaller.job.end);
            receipt.cache_handoffs += 1;
        }
        drop(state);
        work.push((below, depth - 1));
        work.push((above, depth - 1));
    }
    Ok(Continuation {
        indices,
        nodes,
        pending,
        budget,
        receipt,
    })
}

fn resume(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch: &ScratchPool,
    mut continuation: Continuation,
) -> Result<Vec<Node>> {
    let _scope = crate::profile::scope("ma.coarse_continue.resume");
    if continuation.pending.is_empty() {
        return Ok(continuation.nodes);
    }
    let frontier = ma_frontier::Frontier::new(options, samples.counts.len());
    // Enter below train_nodes_with_scratch: no environment hook, preparation,
    // root predictor reset, multiplier rebuild, or second histogram budget.
    learner::train_pending_in_frontier::<true>(
        samples,
        options,
        stats,
        scratch,
        &mut continuation.indices,
        continuation.nodes,
        continuation.pending,
        &continuation.budget,
        frontier.as_ref(),
    )
}

pub(super) fn train(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch: &ScratchPool,
) -> Result<Vec<Node>> {
    let _scope = crate::profile::scope("ma.coarse_continue");
    let continuation = build(samples, options, stats, scratch, COARSE_DEPTH)?;
    let receipt = &continuation.receipt;
    crate::profile::note(
        "ma.coarse_continue.receipt",
        format_args!(
            "policy=legacy12 records={} coarse_nodes={} coarse_splits={} saturated_skipped={} fine_checks={} cached_fine_decisions={} unfinished_roots={} parent_handoffs={} cache_handoffs={}",
            samples.counts.len(),
            receipt.searched,
            receipt.splits,
            receipt.saturated,
            receipt.fine_checks,
            receipt.fine_decisions,
            continuation.pending.len(),
            receipt.parent_handoffs,
            receipt.cache_handoffs,
        ),
    );
    resume(samples, options, stats, scratch, continuation)
}

#[cfg(test)]
mod tests;
