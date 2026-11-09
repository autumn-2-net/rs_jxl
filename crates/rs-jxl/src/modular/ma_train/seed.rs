//! Experimental shared-preparation MA warm start. The seed uses existing
//! quantized properties and residual columns; refinement never gathers pixels
//! again. Only standard tree splits and predictors are emitted.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::{
    DefaultScorer, HistogramBudget, HistogramCache, Node, ParentHistogram, PendingNode,
    PredictorPlan, PropertyColumn, PropertyHistogram, QuantileSplits, SampleStorage, Samples,
    ScratchPool, SearchOptions, SparseHistogramPlan, Split, TrainOptions, TrainingNode,
    TrainingStats, choose_split, evaluate_property_values, keep_better, learner, ma_parallel,
    parent_histograms, subtract_parent_histograms,
};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SeedMode {
    Only,
    #[default]
    Refine,
    Repair,
}

#[derive(Clone, Debug)]
pub struct SeedOptions {
    pub mode: SeedMode,
    /// Initial depth only; does not cap the refined tree.
    pub depth: u8,
    /// Approximate sampled-bit overhead per seed split, not final wire bits.
    pub split_bits: f32,
    pub receipt: Option<Arc<SeedReceipt>>,
}

impl Default for SeedOptions {
    fn default() -> Self {
        Self {
            mode: SeedMode::Refine,
            depth: 3,
            split_bits: 24.0,
            receipt: None,
        }
    }
}

impl SeedOptions {
    pub fn validate(&self) -> Result<()> {
        if !(1..=8).contains(&self.depth) || !self.split_bits.is_finite() || self.split_bits < 0.0 {
            return Err(Error::InvalidInput(
                "invalid MA seed depth/model cost".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct SeedReceipt {
    learners: AtomicUsize,
    prepared_records: AtomicUsize,
    seed_splits: AtomicUsize,
    refined_leaves: AtomicUsize,
    repaired_branches: AtomicUsize,
    merged_branches: AtomicUsize,
    predictor_changes: AtomicUsize,
    histogram_reuses: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SeedSnapshot {
    pub learners: usize,
    pub prepared_records: usize,
    pub seed_splits: usize,
    pub refined_leaves: usize,
    pub repaired_branches: usize,
    pub merged_branches: usize,
    pub predictor_changes: usize,
    pub histogram_reuses: usize,
}

impl SeedReceipt {
    pub fn snapshot(&self) -> SeedSnapshot {
        SeedSnapshot {
            learners: self.learners.load(Ordering::Relaxed),
            prepared_records: self.prepared_records.load(Ordering::Relaxed),
            seed_splits: self.seed_splits.load(Ordering::Relaxed),
            refined_leaves: self.refined_leaves.load(Ordering::Relaxed),
            repaired_branches: self.repaired_branches.load(Ordering::Relaxed),
            merged_branches: self.merged_branches.load(Ordering::Relaxed),
            predictor_changes: self.predictor_changes.load(Ordering::Relaxed),
            histogram_reuses: self.histogram_reuses.load(Ordering::Relaxed),
        }
    }
}

fn leaf(state: &PendingNode, options: &TrainOptions) -> Node {
    Node::Leaf {
        predictor: options.predictors[state.job.predictor],
        offset: 0,
        multiplier: state.job.multiplier,
    }
}

fn bits(parent: &ParentHistogram, options: &TrainOptions) -> f32 {
    let total = parent.counts.iter().sum();
    let scorer = DefaultScorer::default();
    options.scoring.leaf_bits(&scorer, &parent.counts, total)
        + parent.extra_bits as f32
        + options
            .scoring
            .description
            .cost(&scorer, &parent.counts, total)
}

struct Seed {
    state: PendingNode,
    children: Option<Box<(Split, Seed, Seed)>>,
    leaves: usize,
    cost: f32,
}

struct Work<'a> {
    options: &'a TrainOptions,
    coarse: TrainOptions,
    seed: &'a SeedOptions,
    receipt: &'a SeedReceipt,
    stats: &'a TrainingStats,
    scratch: &'a ScratchPool,
    budget: Arc<HistogramBudget>,
    leaves: usize,
}

impl Work<'_> {
    fn best_leaf(
        &self,
        state: &mut PendingNode,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
    ) -> f32 {
        let parents = state.parents.get_or_insert_with(|| {
            parent_histograms(samples, &indices[state.job.begin..state.job.end])
        });
        let old = state.job.predictor;
        let mut cost = bits(&parents[old], self.options);
        for (p, histogram) in parents.iter().enumerate() {
            let candidate = bits(histogram, self.options);
            if candidate < cost {
                cost = candidate;
                state.job.predictor = p;
            }
        }
        if state.job.predictor != old {
            self.receipt
                .predictor_changes
                .fetch_add(1, Ordering::Relaxed);
        }
        cost
    }

    fn propose(
        &self,
        state: &mut PendingNode,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        cost: f32,
    ) -> Result<Option<Split>> {
        let indices = &indices[state.job.begin..state.job.end];
        if indices.len() < 2 || cost <= self.coarse.split_threshold as f32 {
            return Ok(None);
        }
        let parents = state.parents.as_ref().expect("seed parent histograms");
        let plan = PredictorPlan::new(samples, indices, &self.coarse, parents, state.job.predictor);
        let scorer = DefaultScorer::default();
        let sparse = SparseHistogramPlan::new(parents, plan.active(), scorer.lanes(), false);
        let total = parents[0].counts.iter().sum();
        if indices.len() >= self.options.histogram_cache_thresholds().0 {
            state.cache = HistogramCache::reserve_with_stride(samples, parents, &self.budget, None);
        }
        let mut winners = [None; 6];
        let mut scratch = self.scratch.acquire();
        for (property, &id) in self.options.properties.iter().enumerate() {
            if ![0, 10, 11].contains(&id) {
                continue;
            }
            let candidates = with_property_values!(&samples.properties[property], |values| {
                let cached = if let Some(cache) = &mut state.cache {
                    cache.properties[property] = PropertyHistogram::collect_values_with_stride(
                        samples, indices, parents, property, values, &plan, None,
                    )?;
                    Some(&cache.properties[property])
                } else {
                    None
                };
                evaluate_property_values(
                    samples,
                    indices,
                    &self.coarse,
                    parents,
                    &plan,
                    &sparse,
                    total,
                    property,
                    values,
                    cached,
                    false,
                    None,
                    &mut scratch.scratch,
                )
            })?;
            for (best, candidate) in winners.iter_mut().zip(candidates) {
                if let Some(candidate) = candidate {
                    keep_better(best, candidate);
                }
            }
        }
        Ok(choose_split(winners, cost, &self.coarse))
    }

    fn build(
        &mut self,
        mut state: PendingNode,
        samples: &mut Samples<impl SampleStorage>,
        indices: &mut [usize],
        depth: u8,
    ) -> Result<Seed> {
        let cost = self.best_leaf(&mut state, samples, indices);
        let mut result = Seed {
            state,
            children: None,
            leaves: 1,
            cost,
        };
        if depth == 0 {
            return Ok(result);
        }
        let Some(split) = self.propose(&mut result.state, samples, indices, cost)? else {
            return Ok(result);
        };
        self.receipt.seed_splits.fetch_add(1, Ordering::Relaxed);
        // Include the seed model cost before declaring a strict budget failure.
        // An unprofitable proposal does not require another leaf slot.
        if split.bits + self.seed.split_bits >= cost {
            self.receipt.merged_branches.fetch_add(1, Ordering::Relaxed);
            return Ok(result);
        }
        if self.leaves == self.options.max_leaves {
            if self.options.complete_search() {
                return Err(Error::Limit("MA seed exhausted explicit leaf budget"));
            }
            return Ok(result);
        }
        self.leaves += 1;
        let job = result.state.job;
        let cut = job.begin + split.below_count;
        learner::partition_node::<true>(
            samples,
            &mut indices[job.begin..job.end],
            split.property,
            split.bin,
            split.below_count,
        );
        let mut above = PendingNode::new(TrainingNode {
            begin: cut,
            predictor: split.above_predictor,
            ..job
        });
        let mut below = PendingNode::new(TrainingNode {
            end: cut,
            predictor: split.below_predictor,
            ..job
        });
        (above.range, below.range) = ma_parallel::children(
            result.state.range,
            self.options.properties[split.property],
            samples.thresholds[split.property][split.bin],
        );
        let parents = result
            .state
            .parents
            .as_ref()
            .expect("seed parent histograms");
        let (smaller, larger) = if job.end - cut < cut - job.begin {
            (&mut above, &mut below)
        } else {
            (&mut below, &mut above)
        };
        let small = parent_histograms(samples, &indices[smaller.job.begin..smaller.job.end]);
        larger.parents = Some(subtract_parent_histograms(parents.clone(), &small)?);
        smaller.parents = Some(small);
        self.receipt
            .histogram_reuses
            .fetch_add(2, Ordering::Relaxed);
        // Keeping a branch cache allows repair to reprice the existing bins.
        // Refine-only never visits this branch again; free its lease now.
        if self.seed.mode != SeedMode::Repair {
            result.state.cache = None;
        }
        let above = self.build(above, samples, indices, depth - 1)?;
        let below = self.build(below, samples, indices, depth - 1)?;
        let split_cost = above.cost + below.cost + self.seed.split_bits;
        if split_cost >= cost {
            self.leaves -= above.leaves + below.leaves - 1;
            self.receipt.merged_branches.fetch_add(1, Ordering::Relaxed);
        } else {
            result.cost = split_cost;
            result.leaves = above.leaves + below.leaves;
            result.children = Some(Box::new((split, above, below)));
        }
        Ok(result)
    }

    fn refine(
        &mut self,
        mut state: PendingNode,
        samples: &mut Samples<impl SampleStorage>,
        indices: &mut [usize],
    ) -> Result<Vec<Node>> {
        let _scope = crate::profile::scope("ma.seed.refine");
        self.receipt.refined_leaves.fetch_add(1, Ordering::Relaxed);
        self.receipt
            .histogram_reuses
            .fetch_add(usize::from(state.parents.is_some()), Ordering::Relaxed);
        let mut options = self.options.clone();
        options.seed = None;
        options.max_merged_leaves = None;
        options.max_leaves = self.options.max_leaves - self.leaves + 1;
        options.static_property_range = Some(state.range);
        state.job.node = 0;
        let root = leaf(&state, &options);
        let nodes = learner::train_pending_in_frontier::<true>(
            samples,
            &options,
            self.stats,
            self.scratch,
            indices,
            vec![root],
            vec![state],
            &self.budget,
            None,
        )?;
        self.leaves += nodes.len().div_ceil(2) - 1;
        Ok(nodes)
    }

    fn finish(
        &mut self,
        mut seed: Seed,
        samples: &mut Samples<impl SampleStorage>,
        indices: &mut [usize],
    ) -> Result<Vec<Node>> {
        let Some(children) = seed.children.take() else {
            return if self.seed.mode == SeedMode::Only {
                Ok(vec![leaf(&seed.state, self.options)])
            } else {
                self.refine(seed.state, samples, indices)
            };
        };
        if self.seed.mode == SeedMode::Repair {
            let _scope = crate::profile::scope("ma.seed.audit");
            self.receipt
                .histogram_reuses
                .fetch_add(usize::from(seed.state.cache.is_some()), Ordering::Relaxed);
            seed.state.evaluate(
                samples,
                indices,
                self.options,
                &self.budget,
                self.stats,
                self.scratch,
            )?;
            // Rebuild only if a fine split already beats the ENTIRE shallow
            // subtree estimate, not merely its first coarse split.
            if let Some(fine) = seed.state.decision.flatten()
                && fine.bits + self.seed.split_bits < seed.cost
            {
                self.leaves -= seed.leaves - 1;
                drop(children);
                self.receipt
                    .repaired_branches
                    .fetch_add(1, Ordering::Relaxed);
                return self.refine(seed.state, samples, indices);
            }
        }
        seed.state.cache = None;
        let (split, above, below) = *children;
        let above = self.finish(above, samples, indices)?;
        let below = self.finish(below, samples, indices)?;
        let mut nodes = vec![
            Node::Split {
                property: self.options.properties[split.property],
                threshold: samples.thresholds[split.property][split.bin],
                greater: 1,
                less_equal: 2,
            },
            leaf(&seed.state, self.options),
            leaf(&seed.state, self.options),
        ];
        graft(&mut nodes, 1, above);
        graft(&mut nodes, 2, below);
        Ok(nodes)
    }
}

fn graft(nodes: &mut Vec<Node>, destination: usize, subtree: Vec<Node>) {
    let start = nodes.len();
    let remap = |index: usize| {
        if index == 0 {
            destination
        } else {
            start + index - 1
        }
    };
    for (index, mut node) in subtree.into_iter().enumerate() {
        if let Node::Split {
            greater,
            less_equal,
            ..
        } = &mut node
        {
            *greater = remap(*greater);
            *less_equal = remap(*less_equal);
        }
        if index == 0 {
            nodes[destination] = node;
        } else {
            nodes.push(node);
        }
    }
}

pub(super) fn train(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch: &ScratchPool,
) -> Result<Vec<Node>> {
    let _scope = crate::profile::scope("ma.seed");
    let seed = options.seed.as_ref().expect("explicit seed");
    let default_receipt = SeedReceipt::default();
    let receipt = seed.receipt.as_deref().unwrap_or(&default_receipt);
    receipt.learners.fetch_add(1, Ordering::Relaxed);
    receipt
        .prepared_records
        .fetch_add(samples.counts.len(), Ordering::Relaxed);
    let mut indices: Vec<_> = (0..samples.counts.len()).collect();
    let mut nodes = vec![learner::leaf(options.predictors[0])];
    let roots = learner::multiplier_skeleton::<true>(samples, options, &mut indices, &mut nodes)?;
    let ranges = ma_parallel::leaf_ranges(
        &nodes,
        options.static_property_range.unwrap_or([[0, u32::MAX]; 2]),
    );
    let mut coarse = options.clone();
    coarse.seed = None;
    coarse.search = SearchOptions {
        quantile_splits: QuantileSplits::Twelve,
        ..Default::default()
    };
    let mut work = Work {
        options,
        coarse,
        seed,
        receipt,
        stats,
        scratch,
        budget: HistogramBudget::new(options.histogram_cache_bytes),
        leaves: roots.len(),
    };
    let mut seeds = Vec::new();
    {
        let _scope = crate::profile::scope("ma.seed.build");
        for root in roots {
            let mut state = PendingNode::new(root);
            state.range = ranges[root.node];
            seeds.push((
                root.node,
                work.build(state, samples, &mut indices, seed.depth)?,
            ));
        }
    }
    for (destination, seed) in seeds {
        graft(
            &mut nodes,
            destination,
            work.finish(seed, samples, &mut indices)?,
        );
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests;
