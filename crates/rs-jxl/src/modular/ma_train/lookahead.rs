//! Bounded two-ply split selection. Only initial learner roots are probed;
//! speculative partitions own indices and never move the training columns.

use super::{
    Arc, DefaultScorer, HistogramBudget, Ordering, ParentHistogram, PendingNode, PropertyColumn,
    PropertyValue, SampleStorage, Samples, ScratchPool, Split, TrainOptions, TrainingStats,
    TreeLearning, execution, parent_histograms,
};
use crate::{Error, Result};
use std::sync::{Mutex, atomic::AtomicUsize};

#[derive(Clone, Debug)]
pub struct LookaheadOptions {
    /// Total proposals, including the greedy choice. Two descendant leaves are
    /// evaluated for each proposal, without recursive beam expansion.
    pub width: usize,
    /// Distinct training records, not summed weights.
    pub min_records: usize,
    /// Additional one-step cost allowed relative to the best local proposal.
    pub relative_window: f32,
    pub receipt: Option<Arc<LookaheadReceipt>>,
}

impl Default for LookaheadOptions {
    fn default() -> Self {
        Self {
            width: 3,
            min_records: 8192,
            relative_window: 0.01,
            receipt: None,
        }
    }
}

impl LookaheadOptions {
    pub fn validate(&self) -> Result<()> {
        if !(2..=4).contains(&self.width)
            || self.min_records < 4
            || !self.relative_window.is_finite()
            || !(0.0..=1.0).contains(&self.relative_window)
        {
            return Err(Error::InvalidInput(
                "invalid MA lookahead configuration".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct LookaheadReceipt {
    roots: AtomicUsize,
    eligible: AtomicUsize,
    probed_candidates: AtomicUsize,
    child_searches: AtomicUsize,
    changed: AtomicUsize,
    estimated_saved_bits: Mutex<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LookaheadSnapshot {
    pub roots: usize,
    pub eligible: usize,
    pub probed_candidates: usize,
    pub child_searches: usize,
    pub changed: usize,
    /// Local estimated two-ply score improvement, NOT final codestream savings.
    pub estimated_saved_bits: f64,
}

impl LookaheadReceipt {
    pub fn snapshot(&self) -> LookaheadSnapshot {
        LookaheadSnapshot {
            roots: self.roots.load(Ordering::Relaxed),
            eligible: self.eligible.load(Ordering::Relaxed),
            probed_candidates: self.probed_candidates.load(Ordering::Relaxed),
            child_searches: self.child_searches.load(Ordering::Relaxed),
            changed: self.changed.load(Ordering::Relaxed),
            estimated_saved_bits: *self
                .estimated_saved_bits
                .lock()
                .expect("lookahead receipt poisoned"),
        }
    }
}

fn same(a: Split, b: Split) -> bool {
    (a.property, a.bin, a.below_predictor, a.above_predictor)
        == (b.property, b.bin, b.below_predictor, b.above_predictor)
}

pub(super) fn retain_best(candidates: &mut Vec<Split>, candidate: Split, width: usize) {
    if width == 0
        || !candidate.cost().is_finite()
        || candidates.iter().any(|&other| same(other, candidate))
    {
        return;
    }
    // Strict ordered insertion keeps property/bin tie-breaking deterministic.
    let position = candidates.partition_point(|other| other.cost() <= candidate.cost());
    if position < width {
        candidates.insert(position, candidate);
        candidates.truncate(width);
    }
}

fn leaf_cost(parent: &ParentHistogram, options: &TrainOptions) -> f32 {
    let total = parent.counts.iter().sum();
    let scorer = DefaultScorer::default();
    options.scoring.leaf_bits(&scorer, &parent.counts, total)
        + parent.extra_bits as f32
        + options
            .scoring
            .description
            .cost(&scorer, &parent.counts, total)
}

fn branch_cost(split: Split, options: &TrainOptions) -> f32 {
    if options.tree_learning == TreeLearning::Legacy {
        // Legacy's split threshold is its per-branch model overhead proxy.
        options.split_threshold as f32
    } else {
        split.split_cost
    }
}

fn child_cost(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    predictor: usize,
    options: &TrainOptions,
    stats: &TrainingStats,
    pool: &ScratchPool,
    budget: &Arc<HistogramBudget>,
) -> Result<f32> {
    let parents = parent_histograms(samples, indices);
    let leaf = leaf_cost(&parents[predictor], options);
    let split = super::split_search::find_best_split_with_properties(
        samples,
        indices,
        predictor,
        options,
        &mut Some(parents),
        &mut None,
        budget,
        stats,
        pool,
        &[],
        &mut Vec::new(),
    )?;
    Ok(split.map_or(leaf, |split| {
        leaf.min(split.bits + branch_cost(split, options))
    }))
}

fn score(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    split: Split,
    options: &TrainOptions,
    stats: &TrainingStats,
    pool: &ScratchPool,
    budget: &Arc<HistogramBudget>,
) -> Result<f32> {
    let mut below = Vec::with_capacity(split.below_count);
    let mut above = Vec::with_capacity(indices.len() - split.below_count);
    with_property_values!(&samples.properties[split.property], |values| {
        for &index in indices {
            if values[index].index() <= split.bin {
                below.push(index);
            } else {
                above.push(index);
            }
        }
    });
    debug_assert_eq!(below.len(), split.below_count);
    let low = || {
        child_cost(
            samples,
            &below,
            split.below_predictor,
            options,
            stats,
            pool,
            budget,
        )
    };
    let high = || {
        child_cost(
            samples,
            &above,
            split.above_predictor,
            options,
            stats,
            pool,
            budget,
        )
    };
    let (a, b) = if options.parallel && execution::worker_count() > 1 {
        execution::join(low, high)
    } else {
        (low(), high())
    };
    Ok((a? + b?) + branch_cost(split, options))
}

pub(super) fn initialize(
    state: &mut PendingNode,
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    stats: &TrainingStats,
    pool: &ScratchPool,
    available_leaves: usize,
) -> Result<()> {
    let Some(config) = &options.lookahead else {
        return Ok(());
    };
    if let Some(r) = &config.receipt {
        r.roots.fetch_add(1, Ordering::Relaxed);
    }
    let indices = &indices[state.job.begin..state.job.end];
    // A two-ply proposal can require three additional leaves. Never speculate
    // on a tree that cannot even represent it within its configured allowance.
    if indices.len() < config.min_records || available_leaves < 3 {
        return Ok(());
    }
    let _scope = crate::profile::scope("ma.lookahead");
    if let Some(r) = &config.receipt {
        r.eligible.fetch_add(1, Ordering::Relaxed);
    }
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let mut alternatives = Vec::new();
    let greedy = super::split_search::find_best_split_collect(
        samples,
        indices,
        state.job.predictor,
        options,
        &mut state.parents,
        &mut state.cache,
        &budget,
        stats,
        pool,
        &[],
        &mut state.redundant_properties,
        config.width,
        &mut alternatives,
    )?;
    state.decision = Some(greedy);
    // Initial roots are evaluated before the main queue: do not accumulate one
    // retained full cache per channel or bypass the main queue's shared budget.
    state.cache = None;
    let Some(first) = alternatives.first() else {
        return Ok(());
    };
    let cutoff =
        first.cost() + first.cost().abs() * config.relative_window + options.split_threshold as f32;
    alternatives.retain(|candidate| candidate.cost() <= cutoff);
    if let Some(greedy) = greedy {
        alternatives.retain(|&candidate| !same(candidate, greedy));
        alternatives.insert(0, greedy);
        alternatives.truncate(config.width);
    }
    if alternatives.len() < 2 {
        return Ok(());
    }
    let base = leaf_cost(
        &state.parents.as_ref().expect("scored root parents")[state.job.predictor],
        options,
    );
    let mut best = greedy;
    let mut best_cost = base;
    let mut original_cost = base;
    for (i, &candidate) in alternatives.iter().enumerate() {
        let cost = score(samples, indices, candidate, options, stats, pool, &budget)?;
        if let Some(r) = &config.receipt {
            r.probed_candidates.fetch_add(1, Ordering::Relaxed);
            r.child_searches.fetch_add(2, Ordering::Relaxed);
        }
        if i == 0 && greedy.is_some() {
            original_cost = cost;
            best_cost = cost;
        } else if cost < best_cost {
            best_cost = cost;
            best = Some(candidate);
        }
    }
    let changed = match (greedy, best) {
        (Some(a), Some(b)) => !same(a, b),
        (None, Some(_)) => true,
        _ => false,
    };
    if changed && let Some(r) = &config.receipt {
        r.changed.fetch_add(1, Ordering::Relaxed);
        *r.estimated_saved_bits
            .lock()
            .expect("lookahead receipt poisoned") += f64::from(original_cost - best_cost);
    }
    state.decision = Some(best);
    Ok(())
}

#[cfg(test)]
mod tests;
