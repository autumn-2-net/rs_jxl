// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! enc_ma.cc channel roots and detached whole-subtree scheduling, b866b3d.

use super::{Node, StaticPropRange, TrainOptions};
use crate::scheduler::worker_pool::WorkerLimits;
use crate::{Error, Result, execution};
use std::sync::{Mutex, OnceLock};

pub(super) const ROOT_SCRATCH_BYTES: u64 = 32 << 20;

/// Explicit source scheduling switches. These never reduce the search space.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParallelOptions {
    /// JXL_MA_SUBTREE_PARALLEL=1.
    pub subtrees: bool,
    /// JXL_MA_SUBTREE_BALANCED=1, effective only with subtrees.
    pub balanced: bool,
    /// JXL_MA_SMALL_SUBTREES=1: lower the invocation gate to 8192 records.
    pub small_subtrees: bool,
    /// Source lossless worker/task scratch limits, also used by channel roots.
    pub workers: WorkerLimits,
    /// Source ARENA_QUIET, forwarded to the caller's cooperative arena.
    pub quiet: bool,
}

impl ParallelOptions {
    pub fn apply(&self, options: &mut TrainOptions) {
        options.parallel_tools = *self;
    }

    pub(super) fn enabled(self, records: usize, parallel: bool) -> bool {
        self.subtrees
            && parallel
            && !execution::is_serial()
            && rayon::current_thread_index().is_some()
            && records >= if self.small_subtrees { 8192 } else { 65536 }
    }

    pub(super) fn grain(self, records: usize, slots: usize) -> usize {
        if self.balanced && slots != 0 {
            (records / (4 * slots)).clamp(1024, 65536)
        } else {
            65536
        }
    }

    pub(super) fn roots(self, jobs: usize) -> usize {
        self.workers.concurrency(jobs, ROOT_SCRATCH_BYTES)
    }
}

/// Take ownership of disjoint loans only after the task starts. Locks are never
/// held during training or helping descendants, and all tasks join on failure.
pub(super) fn run<J: Send>(
    jobs: Vec<J>,
    order: &[usize],
    workers: usize,
    max_roots: usize,
    quiet: bool,
    work: impl Fn(J) -> Result<Vec<Node>> + Sync,
) -> Result<Vec<Vec<Node>>> {
    let inputs: Vec<_> = jobs.into_iter().map(|job| Mutex::new(Some(job))).collect();
    let outputs: Vec<OnceLock<Result<Vec<Node>>>> =
        (0..inputs.len()).map(|_| OnceLock::new()).collect();
    let complete = execution::cooperative(order.len(), workers, max_roots, quiet, |position| {
        let index = order[position];
        let job = inputs[index]
            .lock()
            .expect("MA root input poisoned")
            .take()
            .expect("MA root dispatched twice");
        let result = work(job);
        let success = result.is_ok();
        assert!(
            outputs[index].set(result).is_ok(),
            "MA root completed twice"
        );
        success
    });
    let mut learned = Vec::new();
    learned
        .try_reserve_exact(outputs.len())
        .map_err(|_| Error::Limit("MA root result allocation"))?;
    // A failed root can leave later jobs unstarted; report its actual error,
    // not a missing-result error from a skipped lower-index job in LPT order.
    let mut missing = !complete;
    for output in outputs {
        match output.into_inner() {
            Some(result) => learned.push(result?),
            None => missing = true,
        }
    }
    if missing {
        return Err(Error::Encoding("MA root dispatch incomplete".into()));
    }
    Ok(learned)
}

pub(super) fn children(
    range: StaticPropRange,
    property: usize,
    threshold: i32,
) -> (StaticPropRange, StaticPropRange) {
    let mut high = range;
    let mut low = range;
    if property < 2 {
        let boundary = threshold.wrapping_add(1) as u32;
        high[property][0] = boundary;
        low[property][1] = boundary;
    }
    (high, low)
}

pub(super) fn leaf_ranges(nodes: &[Node], range: StaticPropRange) -> Vec<StaticPropRange> {
    let mut ranges = vec![range; nodes.len()];
    let mut pending = vec![0];
    while let Some(index) = pending.pop() {
        if let Node::Split {
            property,
            threshold,
            greater,
            less_equal,
        } = nodes[index]
        {
            let (high, low) = children(ranges[index], property, threshold);
            ranges[greater] = high;
            ranges[less_equal] = low;
            pending.push(less_equal);
            pending.push(greater);
        }
    }
    ranges
}

pub(super) fn longest_first(sizes: &[usize]) -> Vec<usize> {
    let mut order: Vec<_> = (0..sizes.len()).collect();
    order.sort_by_key(|&index| std::cmp::Reverse(sizes[index]));
    order
}

pub(super) struct Detached {
    pub(super) destination: usize,
    pub(super) records: std::ops::Range<usize>,
    pub(super) range: StaticPropRange,
    pub(super) redundant: Vec<u8>,
    pub(super) seed: Node,
    pub(super) predictor: usize,
    pub(super) multiplier: u32,
}

pub(super) fn finish(
    samples: &mut super::Samples<impl super::SampleStorage>,
    options: &TrainOptions,
    stats: &super::TrainingStats,
    nodes: &mut Vec<Node>,
    jobs: Vec<Detached>,
    slots: usize,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    if options.budget_mode == Some(super::MaBudgetMode::Bounded) {
        // Each destination already reserves one seed leaf. Recompute the
        // remaining allowance after every graft instead of training all jobs
        // against the same capacity or truncating their finished trees.
        let mut indices: Vec<_> = (0..samples.counts.len()).collect();
        for job in jobs {
            let learning = TrainOptions {
                max_leaves: super::ma_channel::graft_leaf_allowance(
                    nodes.len(),
                    options.max_leaves,
                )?,
                max_merged_leaves: None,
                parallel: false,
                static_property_range: Some(job.range),
                ..options.clone()
            };
            let mut root = super::PendingNode::new(super::TrainingNode {
                node: 0,
                begin: job.records.start,
                end: job.records.end,
                predictor: job.predictor,
                multiplier: job.multiplier,
            });
            root.range = job.range;
            root.redundant_properties = job.redundant;
            let learned = super::train_pending_with_layout::<true>(
                samples,
                &learning,
                stats,
                &super::ScratchPool::new(1),
                &mut indices,
                vec![job.seed],
                vec![root],
            )?;
            super::ma_channel::graft(nodes, job.destination, learned, options.max_leaves)?;
        }
        return Ok(());
    }
    let local_limit = super::ma_channel::graft_leaf_allowance(nodes.len(), options.max_leaves)?;
    let _scope = crate::profile::scope("ma.subtrees");
    let sizes: Vec<_> = jobs.iter().map(|job| job.records.len()).collect();
    let order = longest_first(&sizes);
    let destinations: Vec<_> = jobs.iter().map(|job| job.destination).collect();
    let tasks = slots.min(options.parallel_tools.roots(jobs.len()));
    crate::profile::note(
        "ma.subtrees",
        format_args!(
            "jobs={} worker_tasks={} max_samples={} balanced={} cache_budget_bytes={}",
            jobs.len(),
            tasks,
            sizes[order[0]],
            options.parallel_tools.balanced,
            tasks as u64 * ROOT_SCRATCH_BYTES,
        ),
    );
    let mut spatial_order: Vec<_> = jobs.into_iter().enumerate().collect();
    spatial_order.sort_by_key(|(_, job)| job.records.start);
    let ranges: Vec<_> = spatial_order
        .iter()
        .map(|(_, job)| job.records.clone())
        .collect();
    let views = samples.ranges_mut(&ranges)?;
    let mut jobs: Vec<_> = spatial_order
        .into_iter()
        .zip(views)
        .map(|((index, job), view)| (index, job, view))
        .collect();
    jobs.sort_by_key(|(index, _, _)| *index);
    let learned = run(
        jobs,
        &order,
        slots,
        tasks,
        options.parallel_tools.quiet,
        |(_, job, mut view)| {
            execution::serial(|| {
                // Source passes nullptr and subtree=true. No recursive dispatch,
                // cache inherited from a sibling, or re-created multiplier skeleton.
                let learning = TrainOptions {
                    max_leaves: local_limit,
                    parallel: false,
                    static_property_range: Some(job.range),
                    ..options.clone()
                };
                let mut indices = Vec::new();
                indices
                    .try_reserve_exact(view.counts.len())
                    .map_err(|_| Error::Limit("MA subtree local indices allocation"))?;
                indices.extend(0..view.counts.len());
                let mut root = super::PendingNode::new(super::TrainingNode {
                    node: 0,
                    begin: 0,
                    end: view.counts.len(),
                    predictor: job.predictor,
                    multiplier: job.multiplier,
                });
                root.range = job.range;
                root.redundant_properties = job.redundant;
                super::train_pending_with_layout::<true>(
                    &mut view,
                    &learning,
                    stats,
                    &super::ScratchPool::new(1),
                    &mut indices,
                    vec![job.seed],
                    vec![root],
                )
            })
        },
    )?;
    let before = nodes.len();
    for (destination, learned) in destinations.into_iter().zip(learned) {
        super::ma_channel::graft(nodes, destination, learned, options.max_leaves)?;
    }
    crate::profile::note(
        "ma.subtrees",
        format_args!("added_nodes={}", nodes.len() - before),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        Predictor, PropertyColumn, ResidualToken, TrainingStats, Tree, TreeSamples, leaf,
        train_nodes_with_stats,
    };
    use super::*;
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    fn fixture(records: usize) -> (TreeSamples, TrainOptions) {
        let properties = (0..8)
            .map(|bit| {
                PropertyColumn::Packed(
                    (0..records)
                        .map(|i| {
                            if bit == 7 {
                                0
                            } else {
                                ((i >> (bit % 6)) & 1) as u8
                            }
                        })
                        .collect(),
                )
            })
            .collect();
        let residuals: Vec<_> = (0..records)
            .map(|i| ResidualToken {
                token: (i % 64) as u8,
                nbits: 0,
            })
            .collect();
        (
            TreeSamples {
                properties,
                residuals: vec![residuals.clone(), residuals],
                counts: (0..records).map(|i| 1 + ((i / 64) % 3) as u16).collect(),
                thresholds: vec![vec![0]; 8],
            },
            TrainOptions {
                predictors: vec![Predictor::Zero, Predictor::Left],
                properties: vec![2, 6, 7, 8, 9, 10, 11, 12],
                split_threshold: 1.0,
                max_leaves: 128,
                require_complete_search: true,
                parallel: true,
                property_dedup: true,
                parallel_tools: ParallelOptions {
                    subtrees: true,
                    ..Default::default()
                },
                ..TrainOptions::default()
            },
        )
    }

    fn signature(nodes: Vec<Node>) -> Vec<(usize, u32)> {
        Tree::new(nodes)
            .unwrap()
            .tokens()
            .into_iter()
            .map(|t| (t.context, t.value))
            .collect()
    }

    #[test]
    fn ma_subtrees_source_gates_grain_record_boundaries_and_stable_lpt() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut tools = ParallelOptions::default();
        assert!(!pool.install(|| tools.enabled(1 << 20, true)));
        tools.subtrees = true;
        pool.install(|| {
            assert!(!tools.enabled(65535, true));
            assert!(tools.enabled(65536, true));
            assert!(!tools.enabled(65536, false));
            assert!(!execution::serial(|| tools.enabled(65536, true)));
        });
        tools.small_subtrees = true;
        assert!(!pool.install(|| tools.enabled(8191, true)));
        assert!(pool.install(|| tools.enabled(8192, true)));
        assert_eq!(tools.grain(8192, 4), 65536);
        tools.balanced = true;
        for (records, slots, expected) in [
            (8192, 0, 65536),
            (8192, 4, 1024),
            (65536, 4, 4096),
            (1048576, 4, 65536),
            (2097152, 4, 65536),
        ] {
            assert_eq!(tools.grain(records, slots), expected);
        }
        assert_eq!(longest_first(&[9, 2, 9, 1, 4]), [0, 2, 4, 1, 3]);
        tools.workers = WorkerLimits {
            tasks: 4,
            scratch_bytes: ROOT_SCRATCH_BYTES - 1,
        };
        assert_eq!(tools.roots(9), 1);
        tools.workers.scratch_bytes = 3 * ROOT_SCRATCH_BYTES;
        assert_eq!(tools.roots(9), 3);
        assert_eq!(tools.roots(2), 2);
        assert_eq!(tools.roots(0), 0);
    }

    #[test]
    fn ma_roots_cooperative_bound_child_work_order_failure_and_null_scope() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for quiet in [false, true] {
            let live = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let barrier = Barrier::new(2);
            let order: Vec<_> = (0..6).rev().collect();
            let output = pool
                .install(|| {
                    run((0..6).collect(), &order, 4, 2, quiet, |index| {
                        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        if index >= 4 {
                            barrier.wait();
                        }
                        let children = execution::map_with_state(11, Vec::new, |scratch, i| {
                            scratch.push(i);
                            i + index
                        });
                        assert_eq!(children, (0..11).map(|i| i + index).collect::<Vec<_>>());
                        live.fetch_sub(1, Ordering::SeqCst);
                        Ok(vec![Node::Leaf {
                            predictor: Predictor::Zero,
                            offset: index as i32,
                            multiplier: 1,
                        }])
                    })
                })
                .unwrap();
            assert_eq!(peak.load(Ordering::SeqCst), 2);
            assert_eq!(live.load(Ordering::SeqCst), 0);
            for (index, nodes) in output.into_iter().enumerate() {
                assert!(matches!(nodes[0], Node::Leaf { offset, .. } if offset == index as i32));
            }
        }
        let calls = AtomicUsize::new(0);
        let error = pool.install(|| {
            execution::serial(|| {
                run(vec![0, 1, 2], &[2, 0, 1], 4, 2, false, |index| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    assert!(execution::is_serial());
                    assert_eq!(index, 2);
                    Err(Error::Limit("root failure before lower-index tasks"))
                })
            })
        });
        assert!(matches!(
            error,
            Err(Error::Limit("root failure before lower-index tasks"))
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(
            run::<()>(Vec::new(), &[], 1, 1, false, |_| unreachable!())
                .unwrap()
                .is_empty()
        );

        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let jobs = (0..8).map(|_| Guard(Arc::clone(&dropped))).collect();
        let result = pool.install(|| {
            run(jobs, &(0..8).collect::<Vec<_>>(), 4, 2, true, |_guard| {
                Err(Error::Limit("injected root error"))
            })
        });
        assert!(result.is_err());
        assert_eq!(dropped.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn ma_subtrees_whole_learning_matches_serial_balanced_small_and_bounded_cache() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (records, small, balanced) in [
            (65536, false, false),
            (65536, false, true),
            (8192, true, false),
            (8192, true, true),
        ] {
            let (samples, mut options) = fixture(records);
            options.parallel_tools.small_subtrees = small;
            options.parallel_tools.balanced = balanced;
            options.parallel_tools.workers = WorkerLimits {
                tasks: 3,
                scratch_bytes: 2 * ROOT_SCRATCH_BYTES,
            };
            let expected = signature(
                pool.install(|| {
                    execution::serial(|| {
                        train_nodes_with_stats(
                            &mut samples.clone(),
                            &options,
                            &TrainingStats::default(),
                        )
                    })
                })
                .unwrap(),
            );
            for cache in [0, 32 << 20] {
                options.histogram_cache_bytes = cache;
                let stats = TrainingStats::default();
                let actual = pool
                    .install(|| train_nodes_with_stats(&mut samples.clone(), &options, &stats))
                    .unwrap();
                let jobs = stats.detached_subtrees.load(Ordering::Relaxed);
                assert!(
                    jobs >= 2,
                    "no whole-subtree activation for {records}/{balanced}"
                );
                let leaves = actual.len().div_ceil(2);
                assert!(leaves > jobs, "jobs must learn beyond their seed leaves");
                assert_eq!(signature(actual), expected);
            }
        }
    }

    #[test]
    fn ma_subtrees_weight_does_not_enable_gate_and_pool_initialization_is_invocation_local() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (mut samples, mut options) = fixture(8191);
        options.parallel_tools.small_subtrees = true;
        samples.counts.fill(1024);
        let stats = TrainingStats::default();
        pool.install(|| train_nodes_with_stats(&mut samples, &options, &stats))
            .unwrap();
        assert_eq!(stats.detached_subtrees.load(Ordering::Relaxed), 0);
        let (samples, mut options) = fixture(8192);
        options.parallel_tools.small_subtrees = true;
        let scratch = super::super::ScratchPool::new(4);
        scratch.dispatch_slots.store(4, Ordering::Relaxed);
        // Gate passes but leaf termination occurs before a worker dispatch.
        options.split_threshold = 1e15;
        pool.install(|| {
            super::super::train_nodes_with_scratch(&mut samples.clone(), &options, &stats, &scratch)
        })
        .unwrap();
        assert_eq!(stats.detached_subtrees.load(Ordering::Relaxed), 0);
        assert_eq!(scratch.dispatch_slots.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn ma_subtrees_inherit_seed_mask_static_ranges_and_merge_budget_without_quotas() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let (samples, mut options) = fixture(128);
        options.parallel_tools.workers.tasks = 2;
        let range = [[3, 7], [10, 20]];
        let seed = Node::Leaf {
            predictor: Predictor::Left,
            offset: 9,
            multiplier: 7,
        };
        let stats = TrainingStats::default();
        let mut nodes = vec![seed.clone()];
        pool.install(|| {
            finish(
                &mut samples.clone(),
                &options,
                &stats,
                &mut nodes,
                vec![Detached {
                    destination: 0,
                    records: 0..128,
                    range,
                    redundant: vec![1; 8],
                    seed: seed.clone(),
                    predictor: 1,
                    multiplier: 7,
                }],
                3,
            )
        })
        .unwrap();
        assert_eq!(nodes.len(), 1);
        assert!(matches!(
            nodes[0],
            Node::Leaf {
                predictor: Predictor::Left,
                offset: 9,
                multiplier: 7
            }
        ));
        assert_eq!(stats.inherited_properties.load(Ordering::Relaxed), 8);
        assert_eq!(stats.detached_subtrees.load(Ordering::Relaxed), 0);
        assert_eq!(stats.ready_nodes.load(Ordering::Relaxed), 0);
        let skeleton = vec![
            Node::Split {
                property: 0,
                threshold: 4,
                greater: 1,
                less_equal: 2,
            },
            Node::Split {
                property: 1,
                threshold: 14,
                greater: 3,
                less_equal: 4,
            },
            leaf(Predictor::Zero),
            leaf(Predictor::Zero),
            leaf(Predictor::Zero),
        ];
        let ranges = leaf_ranges(&skeleton, range);
        assert_eq!(ranges[2], [[3, 5], [10, 20]]);
        assert_eq!(ranges[3], [[5, 7], [15, 20]]);
        assert_eq!(ranges[4], [[5, 7], [10, 15]]);

        let (samples, mut options) = fixture(8192);
        options.parallel_tools.small_subtrees = true;
        options.max_leaves = 63;
        let stats = TrainingStats::default();
        assert!(
            pool.install(|| train_nodes_with_stats(&mut samples.clone(), &options, &stats))
                .is_err()
        );
        assert!(stats.detached_subtrees.load(Ordering::Relaxed) > 0);
        options.max_leaves = 64;
        assert_eq!(
            pool.install(|| train_nodes_with_stats(
                &mut samples.clone(),
                &options,
                &TrainingStats::default()
            ))
            .unwrap()
            .len(),
            127
        );
    }
}
