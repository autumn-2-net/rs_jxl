// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Entropy-local port of enc_lossless_workers.h and the atomic job loops in
//! enc_ans.cc / enc_cluster.cc, baseline b866b3defec13494bfbaf705db93581cda118e37.
//! Bounds independent scratch owners, not the caller's pool thread count.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    static IN_ROOT: Cell<bool> = const { Cell::new(false) };
}

fn root_scope<R>(work: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_ROOT.set(self.0);
        }
    }
    let _restore = Restore(IN_ROOT.replace(true));
    work()
}

pub(super) fn task_count(jobs: usize, workers: usize, scratch_per_task: usize) -> usize {
    if crate::execution::is_serial() || IN_ROOT.get() || rayon::current_thread_index().is_none() {
        return usize::from(jobs != 0);
    }
    bounded_tasks(jobs, workers, scratch_per_task)
}

fn bounded_tasks(jobs: usize, workers: usize, scratch_per_task: usize) -> usize {
    let slots = (1usize << 30)
        .checked_div(scratch_per_task)
        .unwrap_or(8)
        .max(1);
    jobs.min(workers.max(1)).min(8).min(slots)
}

/// Independent task bodies consume the supplied stable order. Collect every
/// result (including errors) in that order; never nest another entropy runner.
pub(super) fn map<R: Send>(
    order: &[usize],
    workers: usize,
    scratch_per_task: usize,
    work: impl Fn(usize) -> R + Sync,
) -> Vec<R> {
    let tasks = task_count(order.len(), workers, scratch_per_task);
    if tasks <= 1 {
        return root_scope(|| order.iter().map(|&i| work(i)).collect());
    }
    let next = AtomicUsize::new(0);
    let owners: Vec<_> = (0..tasks).collect();
    let roots: Vec<Vec<_>> = crate::execution::map(&owners, |_| {
        root_scope(|| {
            let mut results = Vec::new();
            loop {
                let position = next.fetch_add(1, Ordering::Relaxed);
                if position >= order.len() {
                    break;
                }
                results.push((position, work(order[position])));
            }
            results
        })
    });
    let mut results: Vec<_> = roots.into_iter().flatten().collect();
    results.sort_unstable_by_key(|(position, _)| *position);
    results.into_iter().map(|(_, result)| result).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_default_scratch_limits_bound_roots_not_pool_size() {
        assert_eq!(bounded_tasks(100, 64, 0), 8);
        assert_eq!(bounded_tasks(3, 64, 0), 3);
        assert_eq!(bounded_tasks(100, 2, 0), 2);
        assert_eq!(bounded_tasks(100, 64, 1 << 29), 2);
        assert_eq!(bounded_tasks(100, 64, (1 << 30) + 1), 1);
        assert_eq!(bounded_tasks(0, 64, 1), 0);
    }

    #[test]
    fn jobs_preserve_order_limit_scratch_and_do_not_dispatch_recursively() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build()
            .unwrap();
        pool.install(|| {
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let order: Vec<_> = (0..129).rev().collect();
            let results = map(&order, 12, 1 << 29, |i| {
                let n = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(n, Ordering::SeqCst);
                assert_eq!(task_count(100, 12, 0), 1);
                let caller = std::thread::current().id();
                let nested = map(&[i, i + 1], 12, 0, |n| {
                    assert_eq!(std::thread::current().id(), caller);
                    n * n
                });
                active.fetch_sub(1, Ordering::SeqCst);
                nested[0]
            });
            assert!(peak.load(Ordering::SeqCst) <= 2);
            assert_eq!(results, order.iter().map(|i| i * i).collect::<Vec<_>>());
            assert!(!IN_ROOT.get());
            map(&[0], 12, 1 << 30, |_| assert_eq!(task_count(100, 12, 0), 1));
            assert!(!IN_ROOT.get());
            crate::execution::serial(|| {
                let caller = std::thread::current().id();
                map(&order, 12, 0, |_| {
                    assert_eq!(std::thread::current().id(), caller)
                });
            });
        });
    }
}
