use super::*;
use crate::execution;
use std::sync::atomic::{AtomicUsize, Ordering};

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

struct Live<'a>(&'a AtomicUsize);
impl Drop for Live<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn nested_groups_keep_borrowed_data_order_and_root_scratch_limits() {
    for threads in [1, 2, 4] {
        for quiet in [false, true] {
            let pool = pool(threads);
            let visits: Vec<_> = (0..23).map(|_| AtomicUsize::new(0)).collect();
            let roots = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            pool.install(|| {
                assert!(execution::cooperative(23, 8, 2, quiet, |root| {
                    assert!(pool.current_thread_index().is_some());
                    assert_eq!(execution::worker_count(), threads);
                    let active = roots.fetch_add(1, Ordering::SeqCst) + 1;
                    let _live = Live(&roots);
                    peak.fetch_max(active, Ordering::SeqCst);
                    assert!(active <= 2);
                    assert_eq!(visits[root].fetch_add(1, Ordering::Relaxed), 0);
                    let borrowed: Vec<_> = (0..97).map(|i| root + i).collect();
                    let actual = execution::map(&borrowed, |value| {
                        let (a, b) = execution::join(|| value * 2, || value * 3);
                        let nested = execution::map_with_state(7, Vec::new, |scratch, i| {
                            scratch.clear();
                            scratch.push(a + b + i);
                            scratch[0]
                        });
                        nested.into_iter().sum::<usize>()
                    });
                    assert_eq!(
                        actual,
                        borrowed
                            .iter()
                            .map(|v| (0..7).map(|i| v * 5 + i).sum::<usize>())
                            .collect::<Vec<_>>()
                    );
                    true
                }))
            });
            assert_eq!(roots.load(Ordering::SeqCst), 0);
            assert!((1..=2).contains(&peak.load(Ordering::SeqCst)));
            assert!(visits.iter().all(|v| v.load(Ordering::Relaxed) == 1));
            assert_eq!(worker_count(), None);
        }
    }
}

#[test]
fn nested_cooperative_roots_share_arena_and_preserve_limits() {
    let pool = pool(4);
    for quiet in [false, true] {
        pool.install(|| {
            assert!(execution::cooperative(3, 3, 2, quiet, |_| {
                let live = AtomicUsize::new(0);
                let visited = AtomicUsize::new(0);
                assert!(execution::cooperative(19, 64, 1, !quiet, |i| {
                    assert_eq!(execution::worker_count(), 3);
                    assert_eq!(live.fetch_add(1, Ordering::SeqCst), 0);
                    let _live = Live(&live);
                    let result = execution::map(&[1, 2, 3], |v| v * i);
                    assert_eq!(result, [i, 2 * i, 3 * i]);
                    visited.fetch_add(1, Ordering::Relaxed);
                    true
                }));
                assert_eq!(visited.load(Ordering::Relaxed), 19);
                true
            }))
        });
    }
}

#[test]
fn mutable_jobs_and_send_only_outputs_are_scoped() {
    let pool = pool(4);
    pool.install(|| {
        assert!(execution::cooperative(4, 4, 2, false, |root| {
            let mut values: Vec<_> = (0..513).collect();
            execution::for_each_mut(&mut values, |i, value| {
                let (a, b) = execution::join(|| i, || root);
                *value = a + b;
            });
            let result = execution::map_mut(&mut values, |i, value| {
                *value *= 2;
                Cell::new(*value + i)
            });
            assert_eq!(values, (0..513).map(|i| 2 * (i + root)).collect::<Vec<_>>());
            assert!(
                result
                    .iter()
                    .enumerate()
                    .all(|(i, value)| value.get() == 3 * i + 2 * root)
            );
            true
        }))
    });
}

#[test]
fn panic_drains_scoped_borrows_and_executor_is_reusable() {
    let pool = pool(4);
    for quiet in [false, true] {
        for nested in [false, true] {
            let live = AtomicUsize::new(0);
            let error = catch_unwind(AssertUnwindSafe(|| {
                pool.install(|| {
                    execution::cooperative(31, 4, 2, quiet, |root| {
                        live.fetch_add(1, Ordering::SeqCst);
                        let _live = Live(&live);
                        let visit = |i| {
                            live.fetch_add(1, Ordering::SeqCst);
                            let _live = Live(&live);
                            if root == 0 && i == 3 {
                                panic!("arena scoped child panic");
                            }
                            let _ = execution::join(|| i + root, || root * 2);
                            true
                        };
                        if nested {
                            execution::cooperative(29, 4, 2, quiet, visit);
                        } else {
                            execution::map(&(0..29).collect::<Vec<_>>(), |&i| visit(i));
                        }
                        true
                    });
                })
            }));
            assert!(error.is_err());
            assert_eq!(live.load(Ordering::SeqCst), 0);
            pool.install(|| {
                assert_eq!(worker_count(), None);
                assert!(execution::cooperative(11, 4, 2, quiet, |_| true));
                assert_eq!(worker_count(), None);
            });
        }
    }
}

#[test]
fn errors_stop_new_roots_after_failure_and_empty_never_enters_pool() {
    assert!(execution::cooperative(0, 0, 0, false, |_| panic!("empty")));
    let pool = pool(3);
    for quiet in [false, true] {
        let visited = AtomicUsize::new(0);
        pool.install(|| {
            assert!(!execution::cooperative(100, 3, 1, quiet, |i| {
                assert_eq!(i, 0);
                visited.fetch_add(1, Ordering::Relaxed);
                execution::map(&[1, 2, 3], |v| v * v);
                false
            }))
        });
        assert_eq!(visited.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn source_queue_prefers_latest_group_and_respects_quantum() {
    let work = |_| {};
    let make = |count, quantum| Pending {
        group: Arc::new(Group {
            callback: Callback::borrowed(&work),
            panic: Mutex::new(None),
        }),
        next: 0,
        end: count,
        remaining: count,
        quantum,
    };
    let mut state = State {
        groups: vec![make(5, 2), make(7, 3)],
        ..Default::default()
    };
    let newer = Arc::clone(&state.groups[1].group);
    for (begin, end) in [(0, 3), (3, 6), (6, 7)] {
        let job = state.take_descendant().unwrap();
        assert!(Arc::ptr_eq(&job.group, &newer));
        assert_eq!((job.begin, job.end), (begin, end));
    }
    let job = state.take_descendant().unwrap();
    assert!(!Arc::ptr_eq(&job.group, &newer));
    assert_eq!((job.begin, job.end), (0, 2));
}

#[test]
fn suspended_callbacks_only_help_their_own_root_descendants() {
    thread_local! { static OWNER: Cell<Option<usize>> = const { Cell::new(None) }; }
    struct Owner(Option<usize>);
    impl Owner {
        fn enter(root: usize) -> Self {
            let previous = OWNER.get();
            assert!(previous.is_none_or(|previous| previous == root));
            Self(OWNER.replace(Some(root)))
        }
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            OWNER.set(self.0);
        }
    }
    let pool = pool(4);
    for quiet in [false, true] {
        pool.install(|| {
            assert!(execution::cooperative(19, 4, 3, quiet, |root| {
                let _owner = Owner::enter(root);
                execution::map(&(0..23).collect::<Vec<_>>(), |&child| {
                    let _owner = Owner::enter(root);
                    execution::map_with_state(17, Vec::new, |scratch, inner| {
                        let _owner = Owner::enter(root);
                        scratch.clear();
                        scratch.push(root + child + inner);
                        execution::join(
                            || scratch[0],
                            || {
                                let _owner = Owner::enter(root);
                                scratch[0]
                            },
                        )
                    })
                });
                true
            }))
        });
    }
}

#[test]
fn null_scope_never_dispatches_children_or_changes_caller() {
    let caller = std::thread::current().id();
    assert!(execution::cooperative(5, 8, 8, true, |_| {
        assert!(execution::is_serial());
        assert_eq!(std::thread::current().id(), caller);
        assert_eq!(worker_count(), None);
        true
    }));
    let pool = pool(4);
    pool.install(|| {
        execution::serial(|| {
            let caller = std::thread::current().id();
            assert!(execution::cooperative(5, 8, 8, false, |_| {
                assert_eq!(std::thread::current().id(), caller);
                assert_eq!(execution::worker_count(), 1);
                true
            }));
        })
    });
}

#[test]
fn cross_pool_install_is_rejected_before_yield_and_does_not_poison_arena() {
    let other = pool(2);
    let entered = AtomicUsize::new(0);
    for threads in [1, 2, 4] {
        let caller = pool(threads);
        caller.install(|| {
            assert!(execution::cooperative(7, threads, 2, true, |_| {
                let rejected = catch_unwind(AssertUnwindSafe(|| {
                    execution::with_pool(&other, || entered.fetch_add(1, Ordering::Relaxed))
                }));
                assert!(rejected.is_err());
                let thread = std::thread::current().id();
                execution::with_pool(&caller, || assert_eq!(std::thread::current().id(), thread));
                execution::serial(|| {
                    execution::with_pool(&other, || {
                        assert_eq!(std::thread::current().id(), thread);
                        assert!(other.current_thread_index().is_none());
                    })
                });
                let values = execution::map(&[1, 2, 3], |value| value * value);
                assert_eq!(values, [1, 4, 9]);
                true
            }))
        });
    }
    assert_eq!(entered.load(Ordering::Relaxed), 0);
}

#[test]
fn secondary_panicking_payloads_do_not_unwind_borrowed_work() {
    struct DropPanic(Arc<AtomicUsize>);
    impl Drop for DropPanic {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
            panic!("deliberate panic payload destructor");
        }
    }
    let pool = pool(3);
    let drops = Arc::new(AtomicUsize::new(0));
    let visited = AtomicUsize::new(0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        pool.install(|| {
            execution::cooperative(1, 3, 1, false, |_| {
                execution::map(&[0, 1, 2], |_| {
                    visited.fetch_add(1, Ordering::Relaxed);
                    std::panic::panic_any(DropPanic(Arc::clone(&drops)));
                });
                true
            });
        })
    }));
    let payload = result.expect_err("child panic must propagate after draining");
    assert!(payload.is::<DropPanic>());
    assert_eq!(visited.load(Ordering::Relaxed), 3);
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    assert!(catch_unwind(AssertUnwindSafe(|| drop(payload))).is_err());
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    pool.install(|| {
        assert_eq!(worker_count(), None);
        assert!(execution::cooperative(3, 3, 1, true, |_| true));
    });
}
