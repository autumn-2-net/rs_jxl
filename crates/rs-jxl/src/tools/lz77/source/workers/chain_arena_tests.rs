// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Source contracts from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::super::{
    Command, Costs, SourceMatchCost, Work, chain_batch, optimal_stream_replay, prices,
};
use super::*;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

fn config() -> SourceConfig {
    SourceConfig {
        method: SourceLz77Method::Optc256,
        chain_batch: true,
        parallelism: Some(SourceParallelism::default()),
        require_global_savings: false,
        ..SourceConfig::default()
    }
}

fn input(size: usize) -> Vec<Literal> {
    let mut state = 0x163277bdu32;
    let mut input: Vec<Literal> = Vec::with_capacity(size);
    for p in 0..size {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        input.push(Literal {
            value: if p % 113 >= 81 && p >= 113 {
                input[p - 113].value
            } else {
                state
            },
            context: (p % 3) as u32,
        });
    }
    input
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn scratch(inputs: &[&[Literal]], eligible: &[bool], config: &SourceConfig) -> usize {
    inputs
        .iter()
        .zip(eligible)
        .filter(|(input, _)| !input.is_empty())
        .map(|(input, &batch)| {
            (input.len() + 1) * 64
                + (40 << 20)
                + if batch {
                    chain_batch::scratch_bytes(input.len(), config).unwrap_or(0)
                } else {
                    0
                }
        })
        .max()
        .unwrap_or(0)
}

// Bound the wait so a scheduling regression fails instead of hanging Cargo.
fn rendezvous(started: &(Mutex<usize>, Condvar)) {
    let mut count = started.0.lock().unwrap();
    *count += 1;
    started.1.notify_all();
    let (count, _) = started
        .1
        .wait_timeout_while(count, Duration::from_secs(5), |count| *count < 2)
        .unwrap();
    assert!(*count >= 2, "independent work did not overlap");
}

#[test]
fn chain_arena_gates_nonempty_count_stable_lpt_and_input_order() {
    let a = input(65536);
    let b = input(65537);
    let inputs = [&a[..], &[], &b[..], &a[..]];
    let eligible = [true, false, false, true];
    let config = config();
    assert!(SourceConfig::default().chain_batch_streams);
    assert!(!config.stream_arena);
    assert!(
        StreamArena::chain_batches(&inputs, &eligible, &config)
            .unwrap()
            .is_none()
    );
    pool(1).install(|| {
        let plan = StreamArena::chain_batches(&inputs, &eligible, &config)
            .unwrap()
            .unwrap();
        assert_eq!(plan.order, [2, 0, 3]);
        assert_eq!((plan.slots, plan.roots), (1, 1));
        let visited = Mutex::new(Vec::new());
        let output = plan
            .collect(inputs.len(), |i| {
                assert!(!execution::is_serial());
                visited.lock().unwrap().push(i);
                Ok(i)
            })
            .unwrap();
        assert_eq!(*visited.lock().unwrap(), [2, 0, 3]);
        assert_eq!(output, [Some(0), None, Some(2), Some(3)]);
        assert!(StreamArena::chain_batches(&inputs, &eligible[..3], &config).is_err());
        for mask in [[false; 4], [false, true, false, false]] {
            assert!(
                StreamArena::chain_batches(&inputs, &mask, &config)
                    .unwrap()
                    .is_none()
            );
        }
        for (inputs, mask) in [
            ([&a[..], &[][..]], [true, false]),
            ([&[][..], &[][..]], [true, true]),
        ] {
            assert!(
                StreamArena::chain_batches(&inputs, &mask, &config)
                    .unwrap()
                    .is_none()
            );
        }
        for variant in [
            SourceConfig {
                chain_batch: false,
                ..config.clone()
            },
            SourceConfig {
                chain_batch_streams: false,
                ..config.clone()
            },
            SourceConfig {
                parallelism: None,
                ..config.clone()
            },
            SourceConfig {
                method: SourceLz77Method::Lz77b1w3f,
                ..config.clone()
            },
            SourceConfig {
                bucket_matcher: true,
                ..config.clone()
            },
            SourceConfig {
                index_mode: SourceIndexMode::All,
                ..config.clone()
            },
            SourceConfig {
                match_cost: SourceMatchCost::Frontier,
                ..config.clone()
            },
        ] {
            assert!(
                StreamArena::chain_batches(&inputs, &eligible, &variant)
                    .unwrap()
                    .is_none()
            );
        }
        execution::serial(|| {
            assert!(
                StreamArena::chain_batches(&inputs, &eligible, &config)
                    .unwrap()
                    .is_none()
            );
        });
    });
}

#[test]
fn chain_arena_scratch_adds_batch_state_and_bounds_all_nonempty_roots() {
    let a = input(65536);
    let b = input(131072);
    let inputs = [&a[..], &[], &b[..], &a[..]];
    let eligible = [true, false, false, false];
    let config = config();
    let bound = scratch(&inputs, &eligible, &config);
    let parser_only = (b.len() + 1) * 64 + (40 << 20);
    // A smaller eligible root can cost more than the largest fallback root.
    assert!(bound > parser_only);
    pool(3).install(|| {
        for (budget, expected) in [
            (1, 1),
            (parser_only * 2, 1),
            (bound * 2 - 1, 1),
            (bound * 2, 2),
            (bound * 3, 3),
        ] {
            let mut limited = config.clone();
            limited.parallelism.as_mut().unwrap().scratch_bytes = budget;
            let plan = StreamArena::chain_batches(&inputs, &eligible, &limited)
                .unwrap()
                .unwrap();
            assert_eq!(plan.roots, expected, "budget={budget}");
            assert_eq!(
                plan.collect(inputs.len(), Ok).unwrap(),
                [Some(0), None, Some(2), Some(3)]
            );
        }
        for workers in [1, 2, 64] {
            let mut limited = config.clone();
            limited.parallelism.as_mut().unwrap().max_workers = workers;
            let plan = StreamArena::chain_batches(&inputs, &eligible, &limited)
                .unwrap()
                .unwrap();
            assert_eq!((plan.slots, plan.roots), (workers.min(3), workers.min(3)));
        }
        let mut invalid = config.clone();
        invalid.parallelism.as_mut().unwrap().max_workers = 0;
        assert!(
            StreamArena::chain_batches(&inputs, &eligible, &invalid)
                .unwrap()
                .is_none()
        );
    });
}

#[test]
fn chain_arena_batch_and_ineligible_roots_overlap_on_the_caller_pool() {
    let a = input(65536);
    let inputs = [&a[..], &[], &a[..], &a[..7]];
    let eligible = [true, false, false, false];
    let pool = pool(2);
    for quiet in [false, true] {
        let mut config = config();
        config.stream_arena_quiet = quiet;
        config.parallelism.as_mut().unwrap().max_workers = 2;
        pool.install(|| {
            let plan = StreamArena::chain_batches(&inputs, &eligible, &config)
                .unwrap()
                .unwrap();
            assert_eq!((plan.roots, plan.quiet), (2, quiet));
            let started = (Mutex::new(0), Condvar::new());
            let output = plan
                .collect(inputs.len(), |i| {
                    assert!(pool.current_thread_index().is_some());
                    assert!(!execution::is_serial());
                    assert_eq!(execution::worker_count(), 2);
                    if i != 3 {
                        rendezvous(&started);
                    }
                    Ok(i)
                })
                .unwrap();
            assert_eq!(output, [Some(0), None, Some(2), Some(3)]);
        });
    }
}

#[test]
fn chain_arena_one_oversized_root_keeps_children_and_nested_roots_bounded() {
    let a = input(65536);
    let inputs = [&a[..], &[], &a[..], &a[..7]];
    let eligible = [true, false, false, false];
    let pool = pool(4);
    for quiet in [false, true] {
        let mut config = config();
        config.stream_arena_quiet = quiet;
        config.parallelism.as_mut().unwrap().max_workers = 2;
        config.parallelism.as_mut().unwrap().scratch_bytes =
            scratch(&inputs, &eligible, &config) - 1;
        pool.install(|| {
            let plan = StreamArena::chain_batches(&inputs, &eligible, &config)
                .unwrap()
                .unwrap();
            assert_eq!((plan.slots, plan.roots), (2, 1));
            let active = AtomicUsize::new(0);
            let output = plan
                .collect(inputs.len(), |i| {
                    assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                    assert!(pool.current_thread_index().is_some());
                    assert!(!execution::is_serial());
                    assert_eq!(index_workers(&config, 16, 1024), 2);
                    let started = (Mutex::new(0), Condvar::new());
                    let mut children = [0; 2];
                    each(&mut children, |j, child| {
                        assert!(pool.current_thread_index().is_some());
                        assert!(execution::is_serial());
                        rendezvous(&started);
                        *child = i + j;
                    });
                    assert_eq!(children, [i, i + 1]);
                    let nested = StreamArena::chain_batches(&inputs, &eligible, &config)
                        .unwrap()
                        .unwrap();
                    let nested_active = AtomicUsize::new(0);
                    let nested_output = nested.collect(inputs.len(), |j| {
                        assert!(!execution::is_serial());
                        assert_eq!(execution::worker_count(), 2);
                        assert_eq!(nested_active.fetch_add(1, Ordering::SeqCst), 0);
                        let (a, b) = execution::join(|| i + j, || i * j);
                        assert_eq!(nested_active.fetch_sub(1, Ordering::SeqCst), 1);
                        Ok(a + b)
                    })?;
                    assert_eq!(
                        nested_output,
                        [Some(i), None, Some(i + 2 + 2 * i), Some(i + 3 + 3 * i)]
                    );
                    assert_eq!(active.fetch_sub(1, Ordering::SeqCst), 1);
                    Ok(i)
                })
                .unwrap();
            assert_eq!(output, [Some(0), None, Some(2), Some(3)]);
        });
    }
}

#[test]
fn chain_arena_propagates_later_input_errors_and_drains_panicking_children() {
    let a = input(65536);
    let b = input(65537);
    let inputs = [&a[..], &[], &b[..], &a[..7]];
    let eligible = [true, false, true, false];
    let mut config = config();
    config.parallelism.as_mut().unwrap().scratch_bytes = scratch(&inputs, &eligible, &config);
    pool(2).install(|| {
        let plan = StreamArena::chain_batches(&inputs, &eligible, &config)
            .unwrap()
            .unwrap();
        assert_eq!(plan.roots, 1);
        let visited = AtomicUsize::new(0);
        let error = plan.collect(inputs.len(), |i| -> Result<usize> {
            assert_eq!(i, 2);
            visited.fetch_add(1, Ordering::SeqCst);
            let nested = StreamArena::chain_batches(&inputs, &eligible, &config)?.unwrap();
            nested.collect(inputs.len(), |_| -> Result<usize> {
                Err(Error::Limit("chain arena nested failure"))
            })?;
            Ok(i)
        });
        assert!(matches!(
            error,
            Err(Error::Limit("chain arena nested failure"))
        ));
        assert_eq!(visited.load(Ordering::SeqCst), 1);
        let active = AtomicUsize::new(0);
        struct Active<'a>(&'a AtomicUsize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.collect(inputs.len(), |i| -> Result<usize> {
                active.fetch_add(1, Ordering::SeqCst);
                let _active = Active(&active);
                let _ = execution::join(|| -> usize { panic!("chain arena child panic") }, || i);
                Ok(i)
            });
        }));
        assert!(panic.is_err());
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(
            plan.collect(inputs.len(), Ok).unwrap(),
            [Some(0), None, Some(2), Some(3)]
        );
    });
}

#[test]
fn chain_arena_real_mixed_parsers_keep_commands_and_empty_results() {
    let a = input(65536);
    let inputs = [&a[..], &[], &a[..], &a[..1024]];
    let eligible = [true, false, false, false];
    let seeds = [a.len(), 0, 0, 1024];
    let widths = [37, 0, 1, 0];
    let config = config();
    let values: Vec<Vec<u32>> = inputs
        .iter()
        .map(|input| input.iter().map(|v| v.value).collect())
        .collect();
    let costs = Costs::raw(&inputs, &config);
    let prices = prices::Prices::with_lengths(&costs, &config).unwrap();
    let baseline = SourceConfig {
        chain_batch: false,
        ..config.clone()
    };
    let expected: Vec<Vec<Command>> = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| {
            execution::serial(|| {
                optimal_stream_replay(
                    input,
                    &values[i],
                    widths[i],
                    &baseline,
                    &prices,
                    seeds[i],
                    &mut Work::default(),
                    None,
                )
            })
            .unwrap()
        })
        .collect();
    for threads in [1, 3] {
        pool(threads).install(|| {
            for quiet in [false, true] {
                let config = SourceConfig {
                    stream_arena_quiet: quiet,
                    ..config.clone()
                };
                let plan = StreamArena::chain_batches(&inputs, &eligible, &config)
                    .unwrap()
                    .unwrap();
                let results = plan
                    .collect(inputs.len(), |i| {
                        let mut work = Work::default();
                        let commands = optimal_stream_replay(
                            inputs[i], &values[i], widths[i], &config, &prices, seeds[i],
                            &mut work, None,
                        )?;
                        Ok((commands, work.chain_batch_streams))
                    })
                    .unwrap();
                for (i, result) in results.into_iter().enumerate() {
                    if inputs[i].is_empty() {
                        assert!(result.is_none());
                        assert!(expected[i].is_empty());
                    } else {
                        let (commands, batches) = result.unwrap();
                        assert_eq!(
                            commands, expected[i],
                            "threads={threads} quiet={quiet} stream={i}"
                        );
                        assert_eq!(batches, usize::from(eligible[i]));
                    }
                }
            }
        });
    }
}
