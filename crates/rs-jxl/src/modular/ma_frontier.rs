// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! Experimental scheduling of required MA branches; no sample/search pruning.

use super::{
    HistogramBudget, HistogramCache, Node, ParentHistogram, PendingNode, SampleStorage, Samples,
    ScratchPool, Split, TrainOptions, TrainingStats,
};
use crate::{Error, Result, execution};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

pub(super) struct Frontier {
    free: AtomicUsize,
    grain: usize,
    slots: usize,
}

impl Frontier {
    pub(super) fn new(options: &TrainOptions, records: usize) -> Option<Self> {
        let tools = options.parallel_tools;
        if !tools.dynamic_subtrees
            || !options.complete_search()
            || !tools.enabled(records, options.parallel)
        {
            return None;
        }
        let slots = tools.roots(execution::worker_count());
        (slots > 1).then(|| Self {
            free: AtomicUsize::new(slots - 1),
            grain: tools.grain(records, slots),
            slots,
        })
    }

    fn parallel_properties(&self) -> bool {
        let active = self.slots.saturating_sub(self.free.load(Ordering::Relaxed));
        // Once several independent branches are runnable, keep their inner
        // sweeps local instead of issuing another full property batch each.
        active < (self.slots / 4).max(2)
    }

    pub(super) fn acquire(&self, low: usize, high: usize) -> Option<Permit<'_>> {
        if low.min(high) < self.grain {
            return None;
        }
        self.free
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .ok()
            .map(|_| Permit {
                frontier: self,
                released: AtomicBool::new(false),
            })
    }
}

pub(super) struct Permit<'a> {
    frontier: &'a Frontier,
    released: AtomicBool,
}

impl Permit<'_> {
    fn release(&self) {
        if !self.released.swap(true, Ordering::Relaxed) {
            self.frontier.free.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn completion(&self) -> Completion<'_, '_> {
        Completion(self)
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

struct Completion<'a, 'f>(&'a Permit<'f>);

impl Drop for Completion<'_, '_> {
    fn drop(&mut self) {
        // Only one additional branch was admitted. Return its slot when the
        // first side finishes, not after the slower side finishes as well.
        self.0.release();
    }
}

pub(super) struct History {
    parents: Option<Vec<ParentHistogram>>,
    cache: Option<HistogramCache>,
    decision: Option<Option<Split>>,
}

impl History {
    pub(super) fn take(
        state: &mut PendingNode,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        options: &TrainOptions,
        stats: &TrainingStats,
    ) -> Result<Self> {
        let _scope = crate::profile::scope("ma.histogram_handoff");
        // A deferred sibling range belongs to the parent view. Consume it
        // before lending either child mutable columns with rebased indices.
        if let Some(removed) = state.removed.take()
            && let Some(mut cache) = state.cache.take()
        {
            let _scope = crate::profile::scope("ma.handoff_normalize");
            cache.normalize_for_handoff(
                samples,
                &indices[removed],
                options,
                &state.redundant_properties,
            )?;
            state.cache = Some(cache);
            stats.cache_subtractions.fetch_add(1, Ordering::Relaxed);
        }
        if state.cache.is_some() {
            stats.histogram_handoffs.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Self {
            parents: state.parents.take(),
            cache: state.cache.take(),
            decision: state.decision.take(),
        })
    }

    pub(super) fn restore(self, state: &mut PendingNode) {
        state.parents = self.parents;
        state.cache = self.cache;
        state.decision = self.decision;
        state.removed = None;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fork(
    samples: &mut Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch: &ScratchPool,
    budget: &Arc<HistogramBudget>,
    frontier: &Frontier,
    permit: Permit<'_>,
    nodes: &mut Vec<Node>,
    mut above: PendingNode,
    mut below: PendingNode,
) -> Result<()> {
    let _scope = crate::profile::scope("ma.dynamic_fork");
    let allowance = super::ma_channel::graft_leaf_allowance(nodes.len(), options.max_leaves)?;
    let high_destination = above.job.node;
    let low_destination = below.job.node;
    let high_seed = nodes[high_destination].clone();
    let low_seed = nodes[low_destination].clone();
    let ranges = [
        below.job.begin..below.job.end,
        above.job.begin..above.job.end,
    ];
    for state in [&mut above, &mut below] {
        if options.parallel_tools.inherit_histograms {
            History::take(state, samples, indices, options, stats)?.restore(state);
        } else {
            state.parents = None;
            state.cache = None;
            state.removed = None;
            state.decision = None;
        }
        state.job.end -= state.job.begin;
        state.job.begin = 0;
        state.job.node = 0;
    }
    let mut views = samples.ranges_mut(&ranges)?;
    let mut high = views.pop().expect("MA high child view");
    let mut low = views.pop().expect("MA low child view");
    let learn = |view: &mut super::ma_ranges::SampleView<'_>, state: PendingNode, seed: Node| {
        let learning = TrainOptions {
            max_leaves: allowance,
            max_merged_leaves: None,
            static_property_range: Some(state.range),
            parallel: frontier.parallel_properties(),
            ..options.clone()
        };
        let mut local_indices = Vec::new();
        local_indices
            .try_reserve_exact(view.counts.len())
            .map_err(|_| Error::Limit("MA live subtree indices allocation"))?;
        local_indices.extend(0..view.counts.len());
        super::train_pending_in_frontier::<true>(
            view,
            &learning,
            stats,
            scratch,
            &mut local_indices,
            vec![seed],
            vec![state],
            budget,
            Some(frontier),
        )
    };
    stats.dynamic_forks.fetch_add(1, Ordering::Relaxed);
    let (high_result, low_result) = execution::join(
        || {
            let _completion = permit.completion();
            learn(&mut high, above, high_seed)
        },
        || {
            let _completion = permit.completion();
            learn(&mut low, below, low_seed)
        },
    );
    // Join both jobs even on error, then publish only complete child trees.
    let high_nodes = high_result?;
    let low_nodes = low_result?;
    super::ma_channel::graft(nodes, high_destination, high_nodes, options.max_leaves)?;
    super::ma_channel::graft(nodes, low_destination, low_nodes, options.max_leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_permit_returns_once_on_first_completion_or_unwind() {
        let frontier = Frontier {
            free: AtomicUsize::new(1),
            grain: 1,
            slots: 2,
        };
        let permit = frontier.acquire(1, 1).unwrap();
        assert!(frontier.acquire(1, 1).is_none());
        let first = permit.completion();
        let second = permit.completion();
        drop(first);
        assert_eq!(frontier.free.load(Ordering::Relaxed), 1);
        let next = frontier.acquire(1, 1).unwrap();
        drop(second);
        drop(permit);
        assert_eq!(frontier.free.load(Ordering::Relaxed), 0);
        drop(next);
        assert_eq!(frontier.free.load(Ordering::Relaxed), 1);
        let failed = std::panic::catch_unwind(|| {
            let permit = frontier.acquire(1, 1).unwrap();
            let _completion = permit.completion();
            panic!("injected branch panic");
        });
        assert!(failed.is_err());
        assert_eq!(frontier.free.load(Ordering::Relaxed), 1);
    }
}
