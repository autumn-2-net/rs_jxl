// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lossless_workers.h and enc_lz77.cc stream dispatch at
//! b866b3defec13494bfbaf705db93581cda118e37. Explicit limits replace getenv;
//! the caller's installed pool is the only executor. No global pool is started.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{Literal, SourceConfig, SourceIndexMode, SourceLz77Method, reserved};
use crate::{Error, Result, execution};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceParallelism {
    /// Source logical task limit, independent of the caller pool's width.
    /// Parser/seed partitions must not change when fewer workers execute them.
    pub max_workers: usize,
    /// Source scratch estimate limits concurrent owners, not total memory,
    /// input, retained output or model. One oversized stream must still run.
    pub scratch_bytes: usize,
    /// JXL_LOSSLESS_LZ77_FAST_PARALLEL opt-in. Careful lazy seeds stay serial.
    pub fast_streams: bool,
}

impl Default for SourceParallelism {
    fn default() -> Self {
        Self {
            max_workers: 8,
            scratch_bytes: 1 << 30,
            fast_streams: false,
        }
    }
}

impl SourceParallelism {
    fn tasks(self, jobs: usize, scratch: usize) -> usize {
        let memory = self
            .scratch_bytes
            .checked_div(scratch)
            .map_or(self.max_workers, |slots| slots.max(1));
        jobs.min(self.max_workers).min(memory)
    }
}

// Source logical jobs, not physical workers: these counts also select parser
// and cold/warm seed boundaries. Keep null/serial gates even for SMALL_INDEX.
pub(super) fn index_tasks(config: &SourceConfig, jobs: usize, scratch: usize) -> usize {
    if !execution::is_serial()
        && rayon::current_thread_index().is_some()
        && let Some(limits) = config.parallelism
    {
        limits.tasks(jobs, scratch).max(1)
    } else {
        1
    }
}

// enc_lz77_seed.h computes this BEFORE RunOnPool, with no runner gate. None
// disables execution parallelism, not the source's default logical partitions.
pub(super) fn indexed_seed_tasks(config: &SourceConfig, size: usize) -> usize {
    config
        .parallelism
        .unwrap_or_default()
        .tasks((size / 65536).max(1), 4 << 20)
        .max(1)
}

// Execution-only limit. Call after logical partitions are fixed so pool width
// or additional Rust parser scratch cannot silently change the source cuts.
pub(super) fn index_workers(config: &SourceConfig, jobs: usize, scratch: usize) -> usize {
    let tasks = index_tasks(config, jobs, scratch);
    if tasks > 1 {
        tasks.min(execution::worker_count())
    } else {
        // index_tasks checked the installed-pool/serial gates. Do not query
        // Rayon's worker count on a null path, which could start its global pool.
        1
    }
}

pub(super) fn each<T: Send>(parts: &mut [T], work: impl Fn(usize, &mut T) + Send + Sync) {
    let visit = |i, part: &mut T| execution::serial(|| work(i, part));
    if parts.len() > 1 && !execution::is_serial() && rayon::current_thread_index().is_some() {
        execution::for_each_mut(parts, visit);
    } else {
        for (i, part) in parts.iter_mut().enumerate() {
            visit(i, part);
        }
    }
}

/// enc_lz77.cc::ApplyLZ77_Optimal STREAM_ARENA dispatch. The executor owns
/// descendant scheduling; this plan owns source gates, LPT and root limits.
pub(super) struct StreamArena {
    order: Vec<usize>,
    slots: usize,
    pub(super) roots: usize,
    quiet: bool,
}

impl StreamArena {
    pub(super) fn new(
        inputs: &[&[Literal]],
        config: &SourceConfig,
        reuse_index: bool,
    ) -> Result<Option<Self>> {
        let Some(limits) = config.parallelism else {
            return Ok(None);
        };
        if !config.stream_arena
            || reuse_index
            || execution::is_serial()
            || rayon::current_thread_index().is_none()
            || !super::index::enabled(inputs, config)
        {
            return Ok(None);
        }
        let count = inputs.iter().filter(|input| !input.is_empty()).count();
        if count <= 1 {
            return Ok(None);
        }
        let mut order = reserved(count)?;
        order.extend((0..inputs.len()).filter(|&i| !inputs[i].is_empty()));
        order.sort_by_key(|&i| std::cmp::Reverse(inputs[i].len()));
        let largest = inputs[order[0]].len();
        // Indexed eligibility implies compact float nodes and match-cost 2:
        // 12-byte step + 4-byte cost/value/distance/distance-cost arrays. This
        // is the source estimate, not a hard RSS ceiling; one oversized root
        // remains eligible. Child partitions retain their own scratch limits.
        let scratch = largest
            .saturating_add(1)
            .saturating_mul(28)
            .saturating_add(21 * super::WINDOW_SIZE);
        Ok(Some(Self {
            roots: limits.tasks(count, scratch).max(1),
            slots: limits.max_workers,
            order,
            quiet: config.stream_arena_quiet,
        }))
    }

    pub(super) fn collect<T: Send + Sync>(
        &self,
        count: usize,
        work: impl Fn(usize) -> Result<T> + Sync,
    ) -> Result<Vec<Option<T>>> {
        let mut results = reserved(count)?;
        results.resize_with(count, OnceLock::new);
        let success = execution::cooperative(
            self.order.len(),
            self.slots,
            self.roots,
            self.quiet,
            |job| {
                let index = self.order[job];
                // Unlike independent streams, this root retains its runner:
                // index builds, batches and block jobs cooperate in this arena.
                let result = work(index);
                let success = result.is_ok();
                assert!(results[index].set(result).is_ok());
                success
            },
        );
        let mut output = reserved(count)?;
        let mut completed = 0;
        for cell in results {
            output.push(match cell.into_inner() {
                Some(result) => {
                    completed += 1;
                    Some(result?)
                }
                None => None,
            });
        }
        // Inspect every published result first: cancellation can leave an
        // earlier input unclaimed, which must not hide the actual later error.
        if !success || completed != self.order.len() {
            return Err(Error::Encoding("source LZ77 arena omitted a stream".into()));
        }
        Ok(output)
    }
}

pub(super) struct Schedule {
    order: Vec<usize>,
    pub(super) tasks: usize,
}

impl Schedule {
    pub(super) fn jobs(count: usize, tasks: usize) -> Self {
        Self {
            order: (0..count).collect(),
            tasks: tasks.max(1).min(count.max(1)),
        }
    }

    pub(super) fn new(inputs: &[&[Literal]], config: &SourceConfig, fast: bool) -> Self {
        let total: usize = inputs.iter().map(|input| input.len()).sum();
        let mut order: Vec<_> = (0..inputs.len())
            .filter(|&i| !inputs[i].is_empty())
            .collect();
        let largest = order.iter().map(|&i| inputs[i].len()).max().unwrap_or(0);
        let mut tasks = 1;
        if let Some(limits) = config.parallelism {
            // Older history modes and range modes are serial in enc_lz77.cc.
            let supported = if fast {
                limits.fast_streams
            } else {
                !config.match_cost.range()
                    && config.index_mode == SourceIndexMode::Disabled
                    && (config.method != SourceLz77Method::Optc256
                        || config.bucket_route().parallel)
            };
            if supported
                && total >= 65_536
                && !execution::is_serial()
                && rayon::current_thread_index().is_some()
            {
                // Rust stores usize positions and uses a dyadic DP tree. Do
                // not apply C++'s smaller compact-node byte estimate to it.
                // These bounds include ring/cache arrays, parser, cost vectors,
                // allocator capacity slack and fast prefix/anchor scratch.
                // Fast's fixed allowance covers Rust's 31-entry usize rings
                // plus the source i8 distance table (at most window + 1).
                let scratch = largest
                    .saturating_add(1)
                    .saturating_mul(if fast { 16 } else { 64 })
                    .saturating_add(if fast { 8 << 20 } else { 40 << 20 });
                tasks = limits
                    .tasks(order.len(), scratch)
                    .min(execution::worker_count())
                    .max(1);
            }
        }
        order.sort_by_key(|&i| std::cmp::Reverse(inputs[i].len()));
        Self { order, tasks }
    }

    pub(super) fn collect<T: Send>(
        &self,
        count: usize,
        work: impl Fn(usize) -> Result<T> + Send + Sync,
    ) -> Result<Vec<Option<T>>> {
        let mut ordered = reserved(count)?;
        ordered.resize_with(count, || None);
        if self.tasks <= 1 {
            for (index, slot) in ordered.iter_mut().enumerate() {
                *slot = Some(work(index)?);
            }
            return Ok(ordered);
        }
        let next = AtomicUsize::new(0);
        let owners: Vec<_> = (0..self.tasks).collect();
        let results = execution::map(&owners, |_| {
            let mut results = Vec::new();
            loop {
                let slot = next.fetch_add(1, Ordering::Relaxed);
                let Some(&index) = self.order.get(slot) else {
                    break;
                };
                // Match history is local and no nested pool may be entered.
                results.push((index, execution::serial(|| work(index))));
            }
            results
        });
        let mut merged: Vec<_> = results.into_iter().flatten().collect();
        merged.sort_by_key(|(index, _)| *index);
        for (index, result) in merged {
            ordered[index] = Some(result?);
        }
        Ok(ordered)
    }
}

#[cfg(test)]
mod arena_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn limits_bound_scratch_owners_but_allow_one_oversized_job() {
        let limits = SourceParallelism {
            max_workers: 8,
            scratch_bytes: 100,
            fast_streams: true,
        };
        assert_eq!(limits.tasks(20, 30), 3);
        assert_eq!(limits.tasks(20, 101), 1);
        assert_eq!(limits.tasks(0, 101), 0);
        assert_eq!(limits.tasks(20, 0), 8);
    }

    #[test]
    fn logical_jobs_keep_source_limits_while_execution_uses_two_workers() {
        let mut config = SourceConfig {
            parallelism: Some(SourceParallelism::default()),
            ..SourceConfig::default()
        };
        assert_eq!(index_tasks(&config, 8, 8 << 20), 1);
        assert_eq!(index_workers(&config, 8, 8 << 20), 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(index_tasks(&config, 8, 8 << 20), 8);
            assert_eq!(index_workers(&config, 8, 8 << 20), 2);
            let visits: Vec<_> = (0..8).map(|_| AtomicUsize::new(0)).collect();
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let schedule = Schedule::jobs(8, index_workers(&config, 8, 8 << 20));
            let result = schedule
                .collect(8, |job| {
                    assert!(execution::is_serial());
                    assert!(pool.current_thread_index().unwrap() < 2);
                    let live = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(live, Ordering::SeqCst);
                    visits[job].fetch_add(1, Ordering::SeqCst);
                    std::thread::yield_now();
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok(job)
                })
                .unwrap();
            assert_eq!(result, (0..8).map(Some).collect::<Vec<_>>());
            assert!(visits.iter().all(|v| v.load(Ordering::SeqCst) == 1));
            assert!((1..=2).contains(&peak.load(Ordering::SeqCst)));
            assert_eq!(active.load(Ordering::SeqCst), 0);
            execution::serial(|| {
                assert_eq!(index_tasks(&config, 8, 8 << 20), 1);
                assert_eq!(index_workers(&config, 8, 8 << 20), 1);
            });
        });
        for (budget, expected) in [(1, 1), (24 << 20, 3), (64 << 20, 8)] {
            config.parallelism.as_mut().unwrap().scratch_bytes = budget;
            pool.install(|| {
                assert_eq!(index_tasks(&config, 8, 8 << 20), expected);
                assert_eq!(index_workers(&config, 8, 8 << 20), expected.min(2));
                // Execution-only parser scratch must not alter the job count.
                assert_eq!(index_workers(&config, expected, usize::MAX), 1);
                assert_eq!(index_tasks(&config, 8, 8 << 20), expected);
            });
        }
        config.parallelism.as_mut().unwrap().max_workers = 64;
        pool.install(|| assert_eq!(index_tasks(&config, usize::MAX, 0), 64));
        config.parallelism = None;
        pool.install(|| {
            assert_eq!(index_tasks(&config, 8, 8 << 20), 1);
            assert_eq!(index_workers(&config, 8, 8 << 20), 1);
        });
    }

    #[test]
    fn dispatch_uses_installed_pool_input_order_and_serial_scope() {
        let input = vec![
            Literal {
                value: 0,
                context: 0
            };
            32768
        ];
        let inputs = [&input[..], &[], &input[..], &input[..]];
        let config = SourceConfig {
            parallelism: Some(SourceParallelism::default()),
            ..SourceConfig::default()
        };
        assert_eq!(Schedule::new(&inputs, &config, false).tasks, 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let schedule = Schedule::new(&inputs, &config, false);
            assert_eq!(schedule.tasks, 2);
            let results = schedule
                .collect(inputs.len(), |i| {
                    assert!(execution::is_serial());
                    assert_eq!(rayon::current_num_threads(), 2);
                    Ok(i * 7)
                })
                .unwrap();
            assert_eq!(results, [Some(0), None, Some(14), Some(21)]);
            let result: Result<Vec<Option<usize>>> = schedule.collect(inputs.len(), |i| {
                if i == 2 {
                    Err(Error::Limit("ordered stream error"))
                } else {
                    Ok(i)
                }
            });
            assert!(result.is_err());
            execution::serial(|| assert_eq!(Schedule::new(&inputs, &config, false).tasks, 1));
            let mut parts = [0; 4];
            each(&mut parts, |i, part| {
                assert!(pool.current_thread_index().is_some());
                assert!(execution::is_serial());
                *part = i;
            });
            assert_eq!(parts, [0, 1, 2, 3]);
            let caller = std::thread::current().id();
            execution::serial(|| {
                each(&mut parts, |_, _| {
                    assert_eq!(std::thread::current().id(), caller);
                    assert!(execution::is_serial());
                })
            });
            super::super::index::tests::check_installed_pool();
        });
    }
}
