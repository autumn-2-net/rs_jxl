// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! b866b3d enc_lossless_workers.h and enc_lossless_probe.cc::Rounds/Score.
//! Cooperative execution ports enc_lossless_probe_pool.h's scoped child queues
//! onto the caller's Rayon workers, including bounded roots and own-group help.
//! No environment variables, new pools, or global Rayon executor are used here.

use std::sync::{Mutex, OnceLock};

use super::batch::ProbeTask;
pub use super::driver::WorkerLimits;
use crate::{Error, Result, execution, modular::predictor::Predictor};

/// Source cooperative probe estimate, not an allocation or RSS guarantee.
pub const PROBE_SCRATCH_BYTES: u64 = 32 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeBatchKind {
    /// One region across candidates; RunProbeTasks supplies a null runner.
    Round,
    /// Several required regions: optional LPT and cooperative dispatch.
    Rounds,
    /// Missing regions of ONE candidate: also eligible for data parallelism.
    Score,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerMode {
    /// Independent encoders; each runs with the null runner.
    Null,
    /// Bounded roots and their child algorithms share the supplied pool.
    Cooperative,
    /// Sequential roots; each can use the entire supplied pool.
    DataParallel,
}

/// Explicit equivalents of the source WORKER_*/PROBE_* execution settings.
/// Defaults keep independent null-runner encodes. These flags only schedule
/// already-required work; they do not change candidates, training, or budgets.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerExecution {
    pub limits: WorkerLimits,
    /// PROBE_WORKERS override. Only RunProbeTasks uses this, not cooperative.
    pub probe_workers: Option<usize>,
    pub cooperative: bool,
    /// Source ARENA_QUIET: wake waiting workers only on group completion.
    pub arena_quiet: bool,
    pub data_parallel: bool,
    pub longest_first: bool,
}

impl WorkerExecution {
    pub fn validate(self) -> Result<()> {
        self.limits.validate()?;
        if self.probe_workers.is_some_and(|n| !(1..=64).contains(&n)) {
            return Err(Error::InvalidInput(
                "source probe worker slots must be 1..=64".into(),
            ));
        }
        Ok(())
    }

    /// Tasks must be the source's deduplicated pending cache cells, in canonical
    /// (candidate, region) order. Stable LPT ties retain that input order. Score
    /// accepts only one candidate; do not infer it from arbitrary batch length.
    pub fn plan_probes(
        self,
        tasks: &[ProbeTask<'_>],
        kind: ProbeBatchKind,
        has_pool: bool,
    ) -> Result<WorkerPlan> {
        self.validate()?;
        if kind == ProbeBatchKind::Score
            && tasks
                .first()
                .is_some_and(|first| tasks.iter().any(|t| t.params != first.params))
        {
            return Err(Error::InvalidInput(
                "source Score dispatch requires one candidate".into(),
            ));
        }
        let large = tasks
            .iter()
            .any(|t| u64::from(t.image.width()) * u64::from(t.image.height()) >= 65536);
        let mut plan = if has_pool && self.cooperative && kind != ProbeBatchKind::Round {
            WorkerPlan::cooperative(tasks.len(), self.limits, PROBE_SCRATCH_BYTES)?
                .with_quiet(self.arena_quiet)
        } else if has_pool
            && self.data_parallel
            && kind == ProbeBatchKind::Score
            && large
            && tasks.len() <= 4
        {
            WorkerPlan::data_parallel(tasks.len())
        } else {
            WorkerPlan::independent(tasks.len(), self.probe_workers.unwrap_or(self.limits.tasks))?
        };
        if self.longest_first && kind == ProbeBatchKind::Rounds {
            let costs = tasks.iter().map(probe_work).collect::<Result<Vec<_>>>()?;
            plan.order.sort_by(|&a, &b| costs[b].total_cmp(&costs[a]));
        }
        Ok(plan)
    }
}

/// Rounds' literal LPT expression. In particular, auto means TWO predictors,
/// even if that candidate's backend would resolve auto to Variable, and the
/// fraction is the original candidate's, not a later probe-training cap.
pub fn probe_work(task: &ProbeTask<'_>) -> Result<f64> {
    let fraction = task.params.learning_fraction;
    if !fraction.is_finite() || fraction < 0.0 {
        return Err(Error::InvalidInput(
            "LPT requires a finite nonnegative learning fraction".into(),
        ));
    }
    let predictors = match task.params.predictor {
        Some(Predictor::Variable) => 8.0,
        None | Some(Predictor::Best) => 2.0,
        Some(_) => 1.0,
    };
    Ok(f64::from(task.image.width())
        * f64::from(task.image.height())
        * (1.0 + predictors * f64::from(fraction) * 10.0))
}

/// Immutable dispatch plan. Output indices always refer to the original input,
/// never LPT order or completion order. An empty plan invokes no runner.
#[derive(Clone, Debug)]
pub struct WorkerPlan {
    mode: WorkerMode,
    worker_slots: usize,
    max_roots: usize,
    quiet: bool,
    order: Vec<usize>,
}

impl WorkerPlan {
    pub fn independent(count: usize, worker_slots: usize) -> Result<Self> {
        if !(1..=64).contains(&worker_slots) {
            return Err(Error::InvalidInput(
                "source worker slots must be 1..=64".into(),
            ));
        }
        Ok(Self {
            mode: WorkerMode::Null,
            worker_slots: count.min(worker_slots),
            max_roots: count.min(worker_slots),
            quiet: false,
            order: (0..count).collect(),
        })
    }

    /// Generic source root/scratch cap, also usable for full-frame finalists.
    /// A single oversized job remains eligible. Caller supplies the source's
    /// scratch estimate for that route; probes use PROBE_SCRATCH_BYTES.
    pub fn cooperative(count: usize, limits: WorkerLimits, scratch_per_task: u64) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            mode: WorkerMode::Cooperative,
            worker_slots: limits.tasks,
            max_roots: limits.concurrency(count, scratch_per_task),
            quiet: false,
            order: (0..count).collect(),
        })
    }

    pub fn data_parallel(count: usize) -> Self {
        Self {
            mode: WorkerMode::DataParallel,
            worker_slots: usize::from(count != 0),
            max_roots: usize::from(count != 0),
            quiet: false,
            order: (0..count).collect(),
        }
    }

    pub fn mode(&self) -> WorkerMode {
        self.mode
    }

    pub fn with_quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// Source logical worker setting, NOT the supplied pool's physical width.
    /// Cooperative execution caps participating workers to this many slots.
    pub fn worker_slots(&self) -> usize {
        self.worker_slots
    }

    pub fn max_roots(&self) -> usize {
        self.max_roots
    }

    pub fn order(&self) -> &[usize] {
        &self.order
    }
}

/// The callback must use encode_serial for Null, or encode_with_pool with this
/// exact borrowed pool for Pool. It must not build another pool or spawn work
/// that outlives the callback. Null is also enforced by the execution scope.
/// Cooperative callbacks must not directly schedule/yield through Rayon or
/// install a different pool: use the supplied encoder entry points, whose
/// internal task dispatch participates in the active cooperative arena.
#[derive(Clone, Copy)]
pub enum TaskRunner<'a> {
    Null,
    Pool(&'a rayon::ThreadPool),
}

/// Execute only on the supplied pool, preserving input-order result ownership.
/// Fixed root loops bound live encoders even when Rayon steals another root
/// while a parent waits for its children. No permits are held across blocking
/// waits; scratch belongs to each Rust invocation, not the current thread ID.
/// Cooperative failure stops new root claims and drains active invocations.
/// Independent mode retains the existing Rust batch contract: attempt every
/// cell and return the first input-order error, not C++'s lane-local early exit.
/// Cooperative child tasks prefer descendants; suspended parents only help
/// their own group. Errors/panics drain active callbacks before returning.
pub fn run_plan<'a, T: Send + Sync>(
    pool: Option<&'a rayon::ThreadPool>,
    plan: &WorkerPlan,
    task: impl Fn(usize, TaskRunner<'a>) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let results: Vec<OnceLock<Result<T>>> =
        (0..plan.order.len()).map(|_| OnceLock::new()).collect();
    if plan.order.is_empty() {
        return Ok(Vec::new());
    }
    let publish = |index: usize, result| {
        assert!(results[index].set(result).is_ok());
    };
    if execution::is_serial() || pool.is_none() {
        // A null scope must never install a pool: its TLS cannot follow stolen
        // Rayon tasks. This branch does not yield or dispatch children.
        for &index in &plan.order {
            let result = execution::serial(|| task(index, TaskRunner::Null));
            let failed = result.is_err();
            publish(index, result);
            if failed && plan.mode != WorkerMode::Null {
                break;
            }
        }
    } else if let Some(pool) = pool {
        execution::with_pool(pool, || -> Result<()> {
            if plan.mode == WorkerMode::DataParallel {
                for &index in &plan.order {
                    publish(index, Ok(task(index, TaskRunner::Pool(pool))?));
                }
                return Ok(());
            }
            if plan.mode == WorkerMode::Cooperative {
                execution::cooperative(
                    plan.order.len(),
                    plan.worker_slots,
                    plan.max_roots,
                    plan.quiet,
                    |slot| {
                        let index = plan.order[slot];
                        let result = task(index, TaskRunner::Pool(pool));
                        let ok = result.is_ok();
                        publish(index, result);
                        ok
                    },
                );
                return Ok(());
            }
            struct Queue {
                next: usize,
                failed: bool,
            }
            let queue = Mutex::new(Queue {
                next: 0,
                failed: false,
            });
            let owners: Vec<_> = (0..plan.max_roots).collect();
            execution::map(&owners, |_| {
                loop {
                    let index = {
                        let mut queue = queue.lock().expect("root queue poisoned");
                        if queue.failed || queue.next == plan.order.len() {
                            break;
                        }
                        let index = plan.order[queue.next];
                        queue.next += 1;
                        index
                    };
                    let result = if plan.mode == WorkerMode::Null {
                        execution::serial(|| task(index, TaskRunner::Null))
                    } else {
                        task(index, TaskRunner::Pool(pool))
                    };
                    let failed = result.is_err();
                    publish(index, result);
                    if failed && plan.mode == WorkerMode::Cooperative {
                        queue.lock().expect("root queue poisoned").failed = true;
                        break;
                    }
                }
            });
            Ok(())
        })?;
    }
    let mut output = Vec::with_capacity(results.len());
    let mut missing = false;
    for result in results {
        match result.into_inner() {
            Some(value) => output.push(value?),
            None => missing = true,
        }
    }
    if missing {
        return Err(Error::Encoding("source worker omitted a task".into()));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Image, PixelData, PixelLayout, scheduler::params::CandidateParams};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn image(width: u32, height: u32) -> Image {
        Image::new(
            width,
            height,
            PixelLayout::Gray,
            PixelData::U8(vec![0; width as usize * height as usize]),
        )
        .unwrap()
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    #[test]
    fn source_root_caps_include_scratch_zero_jobs_and_oversized_jobs() {
        let defaults = WorkerLimits::default();
        assert_eq!((defaults.tasks, defaults.scratch_bytes), (8, 1 << 30));
        let limits = WorkerLimits {
            tasks: 64,
            scratch_bytes: 64 << 20,
        };
        for (jobs, scratch, expected) in [
            (100, PROBE_SCRATCH_BYTES, 2),
            (1, PROBE_SCRATCH_BYTES, 1),
            (0, PROBE_SCRATCH_BYTES, 0),
            (100, 128 << 20, 1),
            (100, u64::MAX, 1),
            (100, 0, 64),
        ] {
            let plan = WorkerPlan::cooperative(jobs, limits, scratch).unwrap();
            assert_eq!(plan.max_roots(), expected);
            assert_eq!(plan.worker_slots(), 64);
        }
        for tasks in [0, 65, usize::MAX] {
            assert!(WorkerPlan::cooperative(1, WorkerLimits { tasks, ..limits }, 0).is_err());
            assert!(WorkerPlan::independent(1, tasks).is_err());
        }
        assert!(
            WorkerPlan::cooperative(
                1,
                WorkerLimits {
                    scratch_bytes: 0,
                    ..limits
                },
                0
            )
            .is_err()
        );
        for probe_workers in [Some(0), Some(65)] {
            assert!(
                WorkerExecution {
                    probe_workers,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }

    #[test]
    fn source_dispatch_gates_are_specific_to_round_rounds_and_score() {
        let params = CandidateParams::default();
        let small = image(255, 257); // 65535 pixels, just below the gate.
        let large = image(256, 256);
        let config = WorkerExecution {
            cooperative: true,
            data_parallel: true,
            probe_workers: Some(1),
            limits: WorkerLimits {
                tasks: 8,
                scratch_bytes: 64 << 20,
            },
            ..Default::default()
        };
        let tasks = [ProbeTask {
            params: &params,
            image: &large,
        }; 3];
        let round = config
            .plan_probes(&tasks, ProbeBatchKind::Round, true)
            .unwrap();
        assert_eq!((round.mode(), round.max_roots()), (WorkerMode::Null, 1));
        for kind in [ProbeBatchKind::Rounds, ProbeBatchKind::Score] {
            let plan = config.plan_probes(&tasks, kind, true).unwrap();
            assert_eq!(
                (plan.mode(), plan.max_roots()),
                (WorkerMode::Cooperative, 2)
            );
            assert_eq!(plan.worker_slots(), 8); // Cooperative ignores PROBE_WORKERS.
            assert_eq!(
                config.plan_probes(&tasks, kind, false).unwrap().mode(),
                WorkerMode::Null
            );
        }
        let config = WorkerExecution {
            cooperative: false,
            ..config
        };
        for count in 1..=5 {
            let mut tasks = vec![
                ProbeTask {
                    params: &params,
                    image: &small
                };
                count
            ];
            assert_eq!(
                config
                    .plan_probes(&tasks, ProbeBatchKind::Score, true)
                    .unwrap()
                    .mode(),
                WorkerMode::Null
            );
            tasks[count - 1].image = &large; // ANY large pending sample is enough.
            let plan = config
                .plan_probes(&tasks, ProbeBatchKind::Score, true)
                .unwrap();
            assert_eq!(
                plan.mode(),
                if count <= 4 {
                    WorkerMode::DataParallel
                } else {
                    WorkerMode::Null
                }
            );
            assert_eq!(
                config
                    .plan_probes(&tasks, ProbeBatchKind::Rounds, true)
                    .unwrap()
                    .mode(),
                WorkerMode::Null
            );
        }
        let mut other = params.clone();
        other.rct = Some(0);
        let mixed = [
            tasks[0],
            ProbeTask {
                params: &other,
                image: &large,
            },
        ];
        assert!(
            config
                .plan_probes(&mixed, ProbeBatchKind::Score, true)
                .is_err()
        );
        let plan = WorkerExecution::default()
            .plan_probes(&[tasks[0]; 20], ProbeBatchKind::Rounds, true)
            .unwrap();
        assert_eq!((plan.mode(), plan.max_roots()), (WorkerMode::Null, 8));
    }

    #[test]
    fn lpt_uses_raw_predictors_original_fraction_and_stable_ties() {
        let image = image(8, 8);
        let params: Vec<_> = [
            Some(Predictor::Left),
            None,
            Some(Predictor::Variable),
            Some(Predictor::Best),
        ]
        .into_iter()
        .map(|predictor| CandidateParams {
            speed_tier: -1,
            predictor,
            learning_fraction: 0.5,
            ..Default::default()
        })
        .collect();
        let tasks: Vec<_> = params
            .iter()
            .map(|params| ProbeTask {
                params,
                image: &image,
            })
            .collect();
        let work: Vec<_> = tasks.iter().map(|t| probe_work(t).unwrap()).collect();
        assert_eq!(work, [384.0, 704.0, 2624.0, 704.0]);
        let config = WorkerExecution {
            longest_first: true,
            ..Default::default()
        };
        let plan = config
            .plan_probes(&tasks, ProbeBatchKind::Rounds, true)
            .unwrap();
        assert_eq!(plan.order(), [2, 1, 3, 0]);
        assert_eq!(
            config
                .plan_probes(&tasks, ProbeBatchKind::Round, true)
                .unwrap()
                .order(),
            [0, 1, 2, 3]
        );
        let pool = pool(1);
        let visits = Mutex::new(Vec::new());
        let values = run_plan(Some(&pool), &plan, |i, runner| {
            assert!(matches!(runner, TaskRunner::Null));
            assert!(execution::is_serial());
            visits.lock().unwrap().push(i);
            Ok(i * 7)
        })
        .unwrap();
        assert_eq!(*visits.lock().unwrap(), [2, 1, 3, 0]);
        assert_eq!(values, [0, 7, 14, 21]);
        for learning_fraction in [f32::NAN, f32::INFINITY, -0.01] {
            let invalid = CandidateParams {
                learning_fraction,
                ..Default::default()
            };
            assert!(
                probe_work(&ProbeTask {
                    params: &invalid,
                    image: &image
                })
                .is_err()
            );
        }
    }

    struct ActiveRoot<'a>(&'a AtomicUsize);
    impl Drop for ActiveRoot<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn cooperative_roots_and_nested_scratch_stay_bounded_on_the_same_pool() {
        let limits = WorkerLimits {
            tasks: 8,
            scratch_bytes: 64 << 20,
        };
        let plan = WorkerPlan::cooperative(29, limits, PROBE_SCRATCH_BYTES).unwrap();
        for threads in [1, 2, 4] {
            let pool = pool(threads);
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let visits: Vec<_> = (0..29).map(|_| AtomicUsize::new(0)).collect();
            let values = run_plan(Some(&pool), &plan, |i, runner| {
                let TaskRunner::Pool(borrowed) = runner else {
                    panic!("missing cooperative runner")
                };
                assert!(std::ptr::eq(borrowed, &pool));
                assert!(pool.current_thread_index().is_some());
                assert!(!execution::is_serial());
                assert_eq!(visits[i].fetch_add(1, Ordering::Relaxed), 0);
                let live = active.fetch_add(1, Ordering::SeqCst) + 1;
                let _root = ActiveRoot(&active);
                peak.fetch_max(live, Ordering::Relaxed);
                assert!(live <= 2);
                // The mutable scratch is per invocation/job, never a shared
                // slot indexed by a worker that could execute a nested root.
                let sums = execution::with_pool(borrowed, || {
                    execution::map_with_state(31, Vec::new, |scratch: &mut Vec<usize>, j| {
                        assert!(pool.current_thread_index().is_some());
                        scratch.clear();
                        scratch.extend([i, j, i + j]);
                        let (a, b) = execution::join(|| scratch[0] + scratch[1], || scratch[2]);
                        a + b
                    })
                });
                Ok(sums.into_iter().sum::<usize>())
            })
            .unwrap();
            assert_eq!(
                values,
                (0..29)
                    .map(|i| (0..31).map(|j| 2 * (i + j)).sum::<usize>())
                    .collect::<Vec<_>>()
            );
            assert_eq!(active.load(Ordering::SeqCst), 0);
            assert!((1..=2).contains(&peak.load(Ordering::Relaxed)));
            assert!(visits.iter().all(|n| n.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn data_parallel_runs_sequential_roots_with_parallel_child_semantics() {
        let pool = pool(3);
        let plan = WorkerPlan::data_parallel(4);
        let roots = AtomicUsize::new(0);
        let order = Mutex::new(Vec::new());
        let result = run_plan(Some(&pool), &plan, |i, runner| {
            assert!(matches!(runner, TaskRunner::Pool(p) if std::ptr::eq(p, &pool)));
            assert_eq!(roots.fetch_add(1, Ordering::SeqCst), 0);
            let _root = ActiveRoot(&roots);
            order.lock().unwrap().push(i);
            execution::try_map(&[2, 3, 5, 7], |_, n| {
                assert!(pool.current_thread_index().is_some());
                assert!(!execution::is_serial());
                Ok(i * n)
            })
        })
        .unwrap();
        assert_eq!(*order.lock().unwrap(), [0, 1, 2, 3]);
        assert_eq!(
            result,
            (0..4)
                .map(|i| vec![2 * i, 3 * i, 5 * i, 7 * i])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn outer_null_scope_overrides_all_plans_without_pool_entry_or_tls_loss() {
        let outer = pool(2);
        let unused = pool(3);
        outer.install(|| {
            execution::serial(|| {
                let caller = std::thread::current().id();
                let plans = [
                    WorkerPlan::independent(7, 3).unwrap(),
                    WorkerPlan::cooperative(7, WorkerLimits::default(), PROBE_SCRATCH_BYTES)
                        .unwrap(),
                    WorkerPlan::data_parallel(7),
                ];
                for plan in &plans {
                    for supplied in [None, Some(&unused)] {
                        let values = run_plan(supplied, plan, |i, runner| {
                            assert!(matches!(runner, TaskRunner::Null));
                            assert!(execution::is_serial());
                            assert_eq!(execution::worker_count(), 1);
                            assert_eq!(std::thread::current().id(), caller);
                            assert!(unused.current_thread_index().is_none());
                            execution::with_pool(&unused, || {
                                assert_eq!(std::thread::current().id(), caller);
                                Ok(execution::map(&[i], |v| v * v)[0])
                            })
                        })
                        .unwrap();
                        assert_eq!(values, [0, 1, 4, 9, 16, 25, 36]);
                    }
                }
                assert!(execution::is_serial());
            })
        });
        assert!(!execution::is_serial());
    }

    #[test]
    fn cooperative_errors_stop_new_roots_and_preserve_the_actual_error() {
        // One root makes cancellation deterministic even with a wide pool.
        let pool = pool(4);
        let mut plan = WorkerPlan::cooperative(
            8,
            WorkerLimits {
                tasks: 8,
                scratch_bytes: 1,
            },
            PROBE_SCRATCH_BYTES,
        )
        .unwrap();
        plan.order.reverse();
        let visited = AtomicUsize::new(0);
        let error = run_plan(Some(&pool), &plan, |i, _| -> Result<usize> {
            visited.fetch_add(1, Ordering::Relaxed);
            assert_eq!(i, 7);
            Err(Error::Limit("cooperative resource failure"))
        });
        // Missing lower-index cells must not mask the actual higher-index error.
        assert!(matches!(
            error,
            Err(Error::Limit("cooperative resource failure"))
        ));
        assert_eq!(visited.load(Ordering::Relaxed), 1);
        let result = run_plan(Some(&pool), &plan, |i, _| Ok(i)).unwrap();
        assert_eq!(result, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn independent_errors_still_drain_every_input_even_with_one_root() {
        let pool = pool(3);
        for slots in [1, 8] {
            let plan = WorkerPlan::independent(12, slots).unwrap();
            for supplied in [None, Some(&pool)] {
                let completed = AtomicUsize::new(0);
                let result = run_plan(supplied, &plan, |i, runner| {
                    assert!(matches!(runner, TaskRunner::Null));
                    assert!(execution::is_serial());
                    completed.fetch_add(1, Ordering::Relaxed);
                    if i == 2 || i == 8 {
                        Err(Error::Encoding(format!("job {i}")))
                    } else {
                        Ok(i)
                    }
                });
                assert!(matches!(result, Err(Error::Encoding(message)) if message == "job 2"));
                assert_eq!(completed.load(Ordering::Relaxed), 12);
            }
        }
    }

    #[test]
    fn absent_pool_uses_the_caller_and_restores_its_execution_policy() {
        assert!(!execution::is_serial());
        let caller = std::thread::current().id();
        let plan =
            WorkerPlan::cooperative(4, WorkerLimits::default(), PROBE_SCRATCH_BYTES).unwrap();
        let result = run_plan(None, &plan, |i, runner| {
            assert!(matches!(runner, TaskRunner::Null));
            assert!(execution::is_serial());
            assert_eq!(std::thread::current().id(), caller);
            let (a, b) = execution::join(|| i, || i * 2);
            Ok(a + b)
        })
        .unwrap();
        assert_eq!(result, [0, 3, 6, 9]);
        assert!(!execution::is_serial());
    }

    #[test]
    fn panic_drains_active_roots_and_does_not_poison_future_dispatch() {
        let pool = pool(3);
        let plan =
            WorkerPlan::cooperative(19, WorkerLimits::default(), PROBE_SCRATCH_BYTES).unwrap();
        let active = AtomicUsize::new(0);
        let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = run_plan(Some(&pool), &plan, |i, _| {
                active.fetch_add(1, Ordering::SeqCst);
                let _root = ActiveRoot(&active);
                execution::join(
                    || {
                        if i == 0 {
                            panic!("test child panic");
                        }
                    },
                    || (),
                );
                Ok(i)
            });
        }));
        assert!(error.is_err());
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(
            run_plan(Some(&pool), &plan, |i, _| Ok(i)).unwrap().len(),
            19
        );
    }

    #[test]
    fn empty_plans_never_invoke_callback_or_require_a_pool() {
        for plan in [
            WorkerPlan::independent(0, 8).unwrap(),
            WorkerPlan::cooperative(0, WorkerLimits::default(), PROBE_SCRATCH_BYTES).unwrap(),
            WorkerPlan::data_parallel(0),
        ] {
            assert_eq!(plan.max_roots(), 0);
            assert!(
                run_plan(None, &plan, |_, _| -> Result<usize> {
                    panic!("empty batch")
                })
                .unwrap()
                .is_empty()
            );
        }
    }
}
