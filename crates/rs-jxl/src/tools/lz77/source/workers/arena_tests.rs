// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Source contracts from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::super::{
    Command, SourceBlockMode, SourceBucketMode, SourceMatchCost, SourceSuffixMode, Work, run,
};
use super::*;

fn config() -> SourceConfig {
    SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_matcher: true,
        index_mode: SourceIndexMode::All,
        match_cost: SourceMatchCost::Frontier,
        small_index: true,
        parallelism: Some(SourceParallelism::default()),
        stream_arena: true,
        max_match_length: 257,
        require_global_savings: false,
        ..SourceConfig::default()
    }
}

fn input(size: usize, offset: u32) -> Vec<Literal> {
    (0..size)
        .map(|i| Literal {
            value: 65536 + offset + (i % 37) as u32,
            context: (i % 3) as u32,
        })
        .collect()
}

fn make_pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn decode(input: &[Literal], commands: &[Command]) {
    let mut values = Vec::with_capacity(input.len());
    for command in commands {
        let position = values.len();
        assert!(position < input.len());
        match *command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[position]);
                values.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[position].context);
                assert!(length >= 3 && position + length as usize <= input.len());
                assert!(distance != 0 && distance as usize <= position);
                for _ in 0..length {
                    values.push(values[values.len() - distance as usize]);
                }
            }
        }
    }
    assert_eq!(values, input.iter().map(|v| v.value).collect::<Vec<_>>());
}

#[test]
fn stream_arena_source_gates_stable_lpt_and_logical_root_limits() {
    assert!(!SourceConfig::default().stream_arena);
    assert!(!SourceConfig::default().stream_arena_quiet);
    let a = input(8192, 0);
    let b = input(16384, 1);
    let inputs = [&a[..], &[], &b[..], &a[..]];
    let config = config();
    assert!(StreamArena::new(&inputs, &config, false).unwrap().is_none());
    let pool = make_pool(2);
    pool.install(|| {
        let plan = StreamArena::new(&inputs, &config, false).unwrap().unwrap();
        assert_eq!(plan.order, [2, 0, 3]);
        assert_eq!((plan.slots, plan.roots, plan.quiet), (8, 3, false));
        // No total-token >=65536 gate: SMALL_INDEX permits this source case.
        let scratch = (b.len() + 1) * 28 + (21 << 20);
        for (budget, roots) in [(1, 1), (scratch * 2 - 1, 1), (scratch * 2, 2)] {
            let mut limited = config.clone();
            limited.parallelism.as_mut().unwrap().scratch_bytes = budget;
            let plan = StreamArena::new(&inputs, &limited, false).unwrap().unwrap();
            assert_eq!((plan.slots, plan.roots), (8, roots));
        }
        let mut limited = config.clone();
        limited.parallelism.as_mut().unwrap().max_workers = 1;
        let plan = StreamArena::new(&inputs, &limited, false).unwrap().unwrap();
        assert_eq!((plan.slots, plan.roots), (1, 1));
        assert!(StreamArena::new(&inputs, &config, true).unwrap().is_none());
        for variant in [
            SourceConfig {
                stream_arena: false,
                ..config.clone()
            },
            SourceConfig {
                parallelism: None,
                ..config.clone()
            },
            SourceConfig {
                small_index: false,
                ..config.clone()
            },
            SourceConfig {
                index_mode: SourceIndexMode::Disabled,
                ..config.clone()
            },
            SourceConfig {
                index_mode: SourceIndexMode::Dominant,
                ..config.clone()
            },
            SourceConfig {
                match_cost: SourceMatchCost::Refit,
                ..config.clone()
            },
            SourceConfig {
                match_cost: SourceMatchCost::Range,
                ..config.clone()
            },
            SourceConfig {
                bucket_matcher: false,
                bucket_mode: SourceBucketMode::History,
                ..config.clone()
            },
        ] {
            assert!(
                StreamArena::new(&inputs, &variant, false)
                    .unwrap()
                    .is_none()
            );
        }
        let excluded: [[&[Literal]; 2]; 3] = [[&a, &[]], [&[], &[]], [&a[..8191], &a[..8191]]];
        for inputs in excluded {
            assert!(StreamArena::new(&inputs, &config, false).unwrap().is_none());
        }
        execution::serial(|| {
            assert!(StreamArena::new(&inputs, &config, false).unwrap().is_none());
        });
    });
    // A single physical worker makes the stable LPT visitation observable.
    make_pool(1).install(|| {
        let plan = StreamArena::new(&inputs, &config, false).unwrap().unwrap();
        let visited = std::sync::Mutex::new(Vec::new());
        let result = plan
            .collect(inputs.len(), |i| {
                visited.lock().unwrap().push(i);
                Ok(i)
            })
            .unwrap();
        assert_eq!(*visited.lock().unwrap(), [2, 0, 3]);
        assert_eq!(result, [Some(0), None, Some(2), Some(3)]);
    });
}

struct Active<'a>(&'a AtomicUsize);

impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn stream_arena_shares_pool_and_bounds_roots_through_nested_children() {
    let a = input(8192, 0);
    let b = input(16384, 1);
    let inputs = [&a[..], &[], &b[..], &a[..]];
    let pool = make_pool(2);
    for quiet in [false, true] {
        let mut config = config();
        config.stream_arena_quiet = quiet;
        config.parallelism.as_mut().unwrap().scratch_bytes = 2 * ((b.len() + 1) * 28 + (21 << 20));
        pool.install(|| {
            let plan = StreamArena::new(&inputs, &config, false).unwrap().unwrap();
            assert_eq!(plan.roots, 2);
            assert_eq!(plan.quiet, quiet);
            let active = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            let visits: Vec<_> = (0..inputs.len()).map(|_| AtomicUsize::new(0)).collect();
            let result = plan
                .collect(inputs.len(), |i| {
                    assert!(pool.current_thread_index().is_some());
                    assert!(!execution::is_serial());
                    assert_eq!(index_tasks(&config, 8, 4 << 20), 8);
                    assert_eq!(index_workers(&config, 8, 4 << 20), 2);
                    let live = active.fetch_add(1, Ordering::SeqCst) + 1;
                    let _active = Active(&active);
                    peak.fetch_max(live, Ordering::SeqCst);
                    assert!(live <= plan.roots);
                    assert_eq!(visits[i].fetch_add(1, Ordering::SeqCst), 0);
                    let mut parts = [0usize; 17];
                    each(&mut parts, |j, part| {
                        assert!(pool.current_thread_index().is_some());
                        assert!(execution::is_serial());
                        *part = i + j;
                    });
                    let sum = execution::map(&parts, |&n| {
                        assert!(pool.current_thread_index().is_some());
                        assert!(!execution::is_serial());
                        let (a, b) = execution::join(|| n, || n * 2);
                        a + b
                    })
                    .into_iter()
                    .sum::<usize>();
                    let children: Vec<_> = (0..3).map(|_| AtomicUsize::new(0)).collect();
                    assert!(execution::cooperative(3, 8, 2, quiet, |j| {
                        assert!(pool.current_thread_index().is_some());
                        assert!(!execution::is_serial());
                        children[j].fetch_add(1, Ordering::SeqCst);
                        true
                    }));
                    assert!(children.iter().all(|n| n.load(Ordering::SeqCst) == 1));
                    Ok(sum)
                })
                .unwrap();
            let expected: Vec<_> = (0..inputs.len())
                .map(|i| {
                    (!inputs[i].is_empty()).then(|| (0..17).map(|j| 3 * (i + j)).sum::<usize>())
                })
                .collect();
            assert_eq!(result, expected);
            assert_eq!(active.load(Ordering::SeqCst), 0);
            assert!((1..=2).contains(&peak.load(Ordering::SeqCst)));
            for (i, count) in visits.iter().enumerate() {
                assert_eq!(
                    count.load(Ordering::SeqCst),
                    usize::from(!inputs[i].is_empty())
                );
            }
        });
    }
}

#[test]
fn stream_arena_failure_preserves_error_and_panic_drains_before_reuse() {
    let a = input(8192, 0);
    let b = input(16384, 1);
    let inputs = [&a[..], &[], &b[..], &a[..]];
    let pool = make_pool(2);
    let mut config = config();
    config.parallelism.as_mut().unwrap().scratch_bytes = 1;
    pool.install(|| {
        let plan = StreamArena::new(&inputs, &config, false).unwrap().unwrap();
        assert_eq!(plan.roots, 1);
        let visits = AtomicUsize::new(0);
        let error = plan.collect(inputs.len(), |i| -> Result<usize> {
            assert_eq!(i, 2);
            visits.fetch_add(1, Ordering::SeqCst);
            Err(Error::Limit("stream arena test failure"))
        });
        assert!(matches!(
            error,
            Err(Error::Limit("stream arena test failure"))
        ));
        assert_eq!(visits.load(Ordering::SeqCst), 1);
        let active = AtomicUsize::new(0);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.collect(inputs.len(), |i| -> Result<usize> {
                active.fetch_add(1, Ordering::SeqCst);
                let _active = Active(&active);
                let _ = execution::join(|| -> usize { panic!("arena child test panic") }, || i);
                Ok(i)
            });
        }));
        assert!(panic.is_err());
        assert_eq!(active.load(Ordering::SeqCst), 0);
        let result = plan.collect(inputs.len(), Ok).unwrap();
        assert_eq!(result, [Some(0), None, Some(2), Some(3)]);
    });
}

#[test]
fn stream_arena_real_index_block_suffix_and_batch_routes_decode_independently() {
    let a = input(65536, 0);
    let b = input(131072, 9);
    let inputs = [&a[..], &[], &b[..], &a[..37]];
    let widths = [37, 0, 19, 0];
    let pool = make_pool(2);
    for (block_mode, suffix_mode) in [
        (SourceBlockMode::Disabled, SourceSuffixMode::Disabled),
        (SourceBlockMode::Endpoints, SourceSuffixMode::Disabled),
        (SourceBlockMode::Intervals, SourceSuffixMode::Disabled),
        (SourceBlockMode::Intervals, SourceSuffixMode::SlidingLcp),
    ] {
        let config = SourceConfig {
            small_index: false,
            block_mode,
            suffix_mode,
            parallel_index_build: true,
            index_match_cache: true,
            index_compact_values: true,
            index_coordinates: true,
            ..config()
        };
        pool.install(|| {
            let sequential = SourceConfig {
                stream_arena: false,
                ..config.clone()
            };
            let expected = run(&inputs, &widths, &sequential, &mut Work::default()).unwrap();
            for quiet in [false, true] {
                let config = SourceConfig {
                    stream_arena_quiet: quiet,
                    ..config.clone()
                };
                let mut work = Work::default();
                let actual = run(&inputs, &widths, &config, &mut work).unwrap();
                assert_eq!((work.arena_passes, work.arena_roots), (1, 3));
                assert_eq!(work.indexed_streams, 2);
                assert_eq!(
                    work.block_streams,
                    if block_mode == SourceBlockMode::Disabled {
                        0
                    } else {
                        2
                    }
                );
                assert_eq!(
                    work.suffix_streams,
                    if suffix_mode == SourceSuffixMode::Disabled {
                        0
                    } else {
                        2
                    }
                );
                assert_eq!(
                    actual, expected,
                    "{block_mode:?} {suffix_mode:?} quiet={quiet}"
                );
                assert!(actual.enabled);
                for (input, commands) in inputs.iter().zip(&actual.commands) {
                    decode(input, commands);
                }
            }
        });
    }
}

#[test]
fn stream_arena_null_shared_seed_and_rejection_gates_preserve_acceptance() {
    let a = input(131072, 0);
    let inputs = [&a[..65536], &[], &a[..]];
    let config = SourceConfig {
        block_mode: SourceBlockMode::Intervals,
        shared_index: true,
        ..config()
    };
    let check = || {
        let mut work = Work::default();
        let actual = run(&inputs, &[37], &config, &mut work).unwrap();
        assert_eq!(work.shared_index_builds, 1);
        assert_eq!(work.arena_passes, 0);
        assert!(actual.enabled);
        for (input, commands) in inputs.iter().zip(&actual.commands) {
            decode(input, commands);
        }
        actual
    };
    let null = check();
    let pool = make_pool(2);
    pool.install(|| {
        assert_eq!(execution::serial(check), null);
        // A pooled parse may have different block cuts, but shared-index reuse
        // must still exclude the arena and its output must decode exactly.
        check();
        let zero = vec![
            Literal {
                value: 0,
                context: 0
            };
            8192
        ];
        let rejected = SourceConfig {
            shared_index: false,
            require_global_savings: true,
            ..config.clone()
        };
        let mut work = Work::default();
        let result = run(&[&zero, &zero], &[], &rejected, &mut work).unwrap();
        assert!(!result.enabled);
        assert_eq!(work.arena_passes, 0);
        for commands in &result.commands {
            decode(&zero, commands);
            assert!(commands.iter().all(|c| matches!(c, Command::Literal(_))));
        }
    });
}
