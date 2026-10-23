// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Core execution policy, matching b866b3d base/data_parallel.h::RunOnPool's
//! null ThreadPool::Run branch. A serial scope is the equivalent of a null runner:
//! no pool installation, global Rayon access, or child-task dispatch. Source
//! sampling/search policy is deliberately not part of this module.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::Result;

mod arena;

thread_local! {
    static SERIAL: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn is_serial() -> bool {
    SERIAL.get()
}

pub(crate) fn serial<R>(work: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SERIAL.set(self.0);
        }
    }
    let _restore = Restore(SERIAL.replace(true));
    work()
}

pub(crate) fn with_pool<R: Send>(pool: &rayon::ThreadPool, work: impl FnOnce() -> R + Send) -> R {
    // Cooperative roots already occupy this pool. Direct entry avoids another
    // install/yield point, while serial scopes remain strictly caller-local.
    if !is_serial() && arena::worker_count().is_some() {
        // Cross-pool install yields the current worker. It could steal another
        // arena loop which waits for this suspended root, forming a deadlock.
        assert!(
            pool.current_thread_index().is_some(),
            "cooperative callbacks cannot enter a different thread pool"
        );
    }
    if is_serial() || pool.current_thread_index().is_some() {
        work()
    } else {
        pool.install(work)
    }
}

pub(crate) fn worker_count() -> usize {
    if is_serial() {
        1
    } else if let Some(workers) = arena::worker_count() {
        workers
    } else {
        rayon::current_num_threads()
    }
}

/// Source cooperative arena on the caller's installed pool. A serial/null
/// runner remains caller-local. Nested invocations share the current arena.
pub(crate) fn cooperative(
    count: usize,
    workers: usize,
    max_roots: usize,
    quiet: bool,
    work: impl Fn(usize) -> bool + Sync,
) -> bool {
    if count == 0 {
        return true;
    }
    assert!(workers > 0 && max_roots > 0);
    if is_serial() || rayon::current_thread_index().is_none() {
        return serial(|| (0..count).all(work));
    }
    arena::run(count, workers, max_roots, quiet, work)
}

fn arena_map<R: Send>(count: usize, work: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let results: Vec<Mutex<Option<R>>> = (0..count).map(|_| Mutex::new(None)).collect();
    assert!(arena::for_each(count, |i| {
        let value = work(i);
        *results[i].lock().expect("arena output poisoned") = Some(value);
    }));
    results
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("arena output poisoned")
                .expect("arena omitted a task")
        })
        .collect()
}

pub(crate) fn for_each_mut<T: Send>(items: &mut [T], work: impl Fn(usize, &mut T) + Send + Sync) {
    if is_serial() || rayon::current_thread_index().is_none() {
        items
            .iter_mut()
            .enumerate()
            .for_each(|(i, item)| work(i, item));
    } else if arena::worker_count().is_some() {
        let slots: Vec<_> = items
            .iter_mut()
            .map(|item| Mutex::new(Some(item)))
            .collect();
        assert!(arena::for_each(slots.len(), |i| {
            let item = slots[i]
                .lock()
                .expect("arena input poisoned")
                .take()
                .expect("arena repeated a task");
            work(i, item);
        }));
    } else {
        items
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, item)| work(i, item));
    }
}

pub(crate) fn map_mut<T: Send, R: Send>(
    items: &mut [T],
    work: impl Fn(usize, &mut T) -> R + Send + Sync,
) -> Vec<R> {
    if is_serial() || rayon::current_thread_index().is_none() {
        items
            .iter_mut()
            .enumerate()
            .map(|(i, item)| work(i, item))
            .collect()
    } else if arena::worker_count().is_some() {
        let slots: Vec<_> = items
            .iter_mut()
            .map(|item| Mutex::new(Some(item)))
            .collect();
        arena_map(slots.len(), |i| {
            let item = slots[i]
                .lock()
                .expect("arena input poisoned")
                .take()
                .expect("arena repeated a task");
            work(i, item)
        })
    } else {
        items
            .par_iter_mut()
            .enumerate()
            .map(|(i, item)| work(i, item))
            .collect()
    }
}

/// Ordered collection; the serial branch short-circuits on the first error.
pub(crate) fn try_map<T: Sync, R: Send>(
    items: &[T],
    work: impl Fn(usize, &T) -> Result<R> + Send + Sync,
) -> Result<Vec<R>> {
    if is_serial() {
        items
            .iter()
            .enumerate()
            .map(|(i, item)| work(i, item))
            .collect()
    } else if arena::worker_count().is_some() {
        arena_map(items.len(), |i| work(i, &items[i]))
            .into_iter()
            .collect()
    } else {
        items
            .par_iter()
            .enumerate()
            .map(|(i, item)| work(i, item))
            .collect()
    }
}

/// Unlike try_map, retains every result, including errors, for ordered merging.
pub(crate) fn map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Send + Sync) -> Vec<R> {
    if is_serial() {
        items.iter().map(work).collect()
    } else if arena::worker_count().is_some() {
        arena_map(items.len(), |i| work(&items[i]))
    } else {
        items.par_iter().map(work).collect()
    }
}

pub(crate) fn map_with_state<S, R: Send>(
    count: usize,
    init: impl Fn() -> S + Send + Sync,
    work: impl Fn(&mut S, usize) -> R + Send + Sync,
) -> Vec<R> {
    if count == 0 {
        Vec::new()
    } else if is_serial() || worker_count() == 1 || count == 1 {
        let mut state = init();
        (0..count).map(|i| work(&mut state, i)).collect()
    } else {
        // Source RunOnPool owns scratch per worker. Rayon map_init instead
        // creates it per split job. Retain one private state in each bounded
        // worker loop; nested tasks cannot alias a suspended worker's state.
        let workers = count.min(worker_count());
        let next = AtomicUsize::new(0);
        let owners: Vec<_> = (0..workers).collect();
        let completed: Vec<Vec<(usize, R)>> = map(&owners, |_| {
            let mut state = init();
            let mut results = Vec::new();
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= count {
                    break;
                }
                results.push((index, work(&mut state, index)));
            }
            results
        });
        let mut ordered: Vec<Option<R>> = (0..count).map(|_| None).collect();
        for (index, value) in completed.into_iter().flatten() {
            ordered[index] = Some(value);
        }
        ordered
            .into_iter()
            .map(|value| value.expect("stateful worker omitted a task"))
            .collect()
    }
}

pub(crate) fn join<A: Send, B: Send>(
    left: impl FnOnce() -> A + Send,
    right: impl FnOnce() -> B + Send,
) -> (A, B) {
    if is_serial() {
        (left(), right())
    } else if arena::worker_count().is_some() {
        let left = Mutex::new(Some(left));
        let right = Mutex::new(Some(right));
        let a = Mutex::new(None);
        let b = Mutex::new(None);
        assert!(arena::for_each(2, |i| {
            if i == 0 {
                let work = left.lock().expect("arena join poisoned").take().unwrap();
                let value = work();
                *a.lock().expect("arena join output poisoned") = Some(value);
            } else {
                let work = right.lock().expect("arena join poisoned").take().unwrap();
                let value = work();
                *b.lock().expect("arena join output poisoned") = Some(value);
            }
        }));
        (
            a.into_inner().unwrap().unwrap(),
            b.into_inner().unwrap().unwrap(),
        )
    } else {
        rayon::join(left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    #[test]
    fn nested_serial_scopes_restore_after_success_error_and_panic() {
        assert!(!is_serial());
        serial(|| {
            assert!(is_serial());
            assert_eq!(worker_count(), 1);
            serial(|| assert!(is_serial()));
            let error: Result<()> = serial(|| Err(crate::Error::Limit("test")));
            assert!(error.is_err());
            assert!(std::panic::catch_unwind(|| serial(|| panic!("test"))).is_err());
            assert!(is_serial());
        });
        assert!(!is_serial());
        assert!(std::panic::catch_unwind(|| serial(|| panic!("test"))).is_err());
        assert!(!is_serial());
    }

    fn check_local_primitives() {
        let caller = thread::current().id();
        let inputs: Vec<_> = (0..4096).collect();
        let expected: Vec<_> = inputs.iter().map(|v| v * v).collect();
        let result = try_map(&inputs, |i, v| {
            assert_eq!(thread::current().id(), caller);
            assert!(is_serial());
            assert_eq!(i, *v);
            Ok(serial(|| map(&[*v], |n| n * n)[0]))
        })
        .unwrap();
        assert_eq!(result, expected);
        let result = map_with_state(inputs.len(), Vec::new, |scratch, i| {
            assert_eq!(thread::current().id(), caller);
            scratch.push(i);
            i * i
        });
        assert_eq!(result, expected);
        let (a, b) = join(|| thread::current().id(), || thread::current().id());
        assert_eq!((a, b), (caller, caller));
        let visited = AtomicUsize::new(0);
        assert!(
            try_map(&inputs, |i, _| {
                visited.fetch_add(1, Ordering::Relaxed);
                if i == 7 {
                    Err(crate::Error::Limit("test"))
                } else {
                    Ok(i)
                }
            })
            .is_err()
        );
        assert_eq!(visited.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn serial_primitives_stay_on_caller_without_a_pool() {
        serial(check_local_primitives);
    }

    #[test]
    fn stateful_workers_bound_allocations_and_keep_order_under_nested_work() {
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for count in [0, 1, 3, 257] {
                let initialized = AtomicUsize::new(0);
                let visited: Vec<_> = (0..count).map(|_| AtomicUsize::new(0)).collect();
                let values = pool.install(|| {
                    map_with_state(
                        count,
                        || {
                            let id = initialized.fetch_add(1, Ordering::Relaxed);
                            (id, Vec::<usize>::new())
                        },
                        |(id, scratch), i| {
                            assert_eq!(visited[i].fetch_add(1, Ordering::Relaxed), 0);
                            scratch.clear();
                            scratch.extend([*id, i, i * i]);
                            let saved = scratch.clone();
                            let (left, right) = join(
                                || {
                                    map_with_state(7, Vec::new, |inner, j| {
                                        inner.clear();
                                        inner.push(i + j);
                                        inner[0]
                                    })
                                },
                                || scratch[2],
                            );
                            assert_eq!(*scratch, saved);
                            assert_eq!(left, (0..7).map(|j| i + j).collect::<Vec<_>>());
                            right
                        },
                    )
                });
                assert_eq!(values, (0..count).map(|i| i * i).collect::<Vec<_>>());
                assert_eq!(initialized.load(Ordering::Relaxed), count.min(threads));
                assert!(visited.iter().all(|n| n.load(Ordering::Relaxed) == 1));
            }
        }
    }

    #[test]
    fn stateful_workers_drop_scratch_after_panic_and_remain_reusable() {
        struct Scratch<'a>(&'a AtomicUsize);
        impl Drop for Scratch<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let live = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.install(|| {
                map_with_state(
                    101,
                    || {
                        live.fetch_add(1, Ordering::Relaxed);
                        Scratch(&live)
                    },
                    |_, i| {
                        assert_ne!(i, 5, "test stateful worker panic");
                        i
                    },
                )
            })
        }));
        assert!(result.is_err());
        assert_eq!(live.load(Ordering::Relaxed), 0);
        assert_eq!(
            pool.install(|| map_with_state(17, || (), |_, i| i)),
            (0..17).collect::<Vec<_>>()
        );
    }

    #[test]
    fn same_pool_entry_is_inline_but_a_different_pool_is_still_honored() {
        let first = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let second = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        first.install(|| {
            let caller = thread::current().id();
            let index = first.current_thread_index();
            assert!(index.is_some());
            with_pool(&first, || {
                assert_eq!(thread::current().id(), caller);
                assert_eq!(first.current_thread_index(), index);
                assert!(!is_serial());
                assert_eq!(worker_count(), 2);
            });
            with_pool(&second, || {
                assert!(first.current_thread_index().is_none());
                assert!(second.current_thread_index().is_some());
                let caller = thread::current().id();
                with_pool(&second, || assert_eq!(thread::current().id(), caller));
                assert_eq!(worker_count(), 3);
            });
            assert_eq!(thread::current().id(), caller);
            assert_eq!(first.current_thread_index(), index);
        });
    }

    #[test]
    fn serial_scope_inside_workers_does_not_leak_or_install_another_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let other = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let actual = map(&[2, 3, 5, 7], |&n| {
                assert!(!is_serial());
                let caller = thread::current().id();
                serial(|| {
                    check_local_primitives();
                    with_pool(&other, || {
                        assert_eq!(thread::current().id(), caller);
                        assert_eq!(worker_count(), 1);
                    });
                });
                assert!(!is_serial());
                assert_eq!(worker_count(), 3);
                n * n
            });
            assert_eq!(actual, [4, 9, 25, 49]);
        });
    }
}
