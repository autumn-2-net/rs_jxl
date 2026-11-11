use super::super::index::Dominance;
use super::*;
use crate::execution;
use crate::tools::lz77::source::{Literal, SourceParallelism, Work, run};

fn config() -> SourceConfig {
    SourceConfig {
        method: SourceLz77Method::Optc256,
        chain_batch: true,
        chain_batch_parallel_only: true,
        require_global_savings: false,
        parallelism: Some(SourceParallelism::default()),
        ..Default::default()
    }
}

#[test]
fn parallel_only_admission_uses_effective_workers_and_preserves_existing_gates() {
    let size = 65536;
    let config = config();
    assert!(!SourceConfig::default().chain_batch_parallel_only);
    assert!(!available(&config));
    assert!(!enabled(size, size, &config));
    assert!(enabled(
        size,
        size,
        &SourceConfig {
            chain_batch_parallel_only: false,
            ..config.clone()
        }
    ));
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(available(&config), threads > 1);
            assert_eq!(enabled(size, size, &config), threads > 1);
            for max_workers in [0, 1, 2, 8] {
                let limited = SourceConfig {
                    parallelism: Some(SourceParallelism {
                        max_workers,
                        ..Default::default()
                    }),
                    ..config.clone()
                };
                assert_eq!(
                    enabled(size, size, &limited),
                    threads > 1 && max_workers > 1
                );
                assert_eq!(available(&limited), threads > 1 && max_workers > 1);
            }
            execution::serial(|| {
                assert!(!available(&config));
                assert!(!enabled(size, size, &config));
            });
            for disabled in [
                SourceConfig {
                    chain_batch: false,
                    ..config.clone()
                },
                SourceConfig {
                    parallelism: None,
                    ..config.clone()
                },
                SourceConfig {
                    method: SourceLz77Method::Optc8,
                    ..config.clone()
                },
                SourceConfig {
                    chain_batch_bytes: 1,
                    ..config.clone()
                },
                SourceConfig {
                    index_mode: SourceIndexMode::All,
                    ..config.clone()
                },
                SourceConfig {
                    bucket_matcher: true,
                    ..config.clone()
                },
                SourceConfig {
                    match_cost: SourceMatchCost::Frontier,
                    ..config.clone()
                },
            ] {
                assert!(!enabled(size, size, &disabled));
            }
            assert!(!enabled(size - 1, size, &config));
            assert!(!enabled(size, size / 4, &config));
        });
    }
}

#[test]
fn chain_batch_frontier_clear_keeps_stale_entries_out_of_queries_and_overflow() {
    let mut reused = Dominance::new();
    // Alternate populated, empty, overflowing and small frontiers in one slot.
    for count in [7, 0, 17, 2, 0, 16, 1] {
        reused.clear();
        assert_eq!(reused.size, 0);
        assert!(!reused.overflowed);
        assert_eq!(reused.required(f32::INFINITY, u32::MAX), MIN_MATCH);
        let mut fresh = Dominance::new();
        for i in 0..count {
            let length = MIN_MATCH + i;
            let symbol = i as u32;
            reused.add(length, symbol, symbol as f32);
            fresh.add(length, symbol, symbol as f32);
        }
        assert_eq!(reused.size, fresh.size);
        assert_eq!(reused.overflowed, fresh.overflowed);
        for (a, b) in reused.entries[..reused.size]
            .iter()
            .zip(&fresh.entries[..fresh.size])
        {
            assert_eq!((a.length, a.symbol, a.cost), (b.length, b.symbol, b.cost));
        }
        for symbol in 0..20 {
            assert_eq!(
                reused.required(symbol as f32, symbol),
                fresh.required(symbol as f32, symbol)
            );
        }
    }
}

#[test]
fn parallel_only_parser_preserves_commands_and_avoids_single_worker_batch_allocation() {
    let mut input: Vec<Literal> = Vec::new();
    for p in 0..65540 {
        let value = if p > 113 && p % 113 >= 81 {
            input[p - 113].value
        } else {
            (p as u32).wrapping_mul(2654435761)
        };
        input.push(Literal {
            value,
            context: (p % 3) as u32,
        });
    }
    let ordinary = SourceConfig {
        chain_batch: false,
        ..config()
    };
    let expected = run(&[&input], &[37], &ordinary, &mut Work::default()).unwrap();
    assert!(expected.enabled);
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut work = Work::default();
            let actual = run(&[&input], &[37], &config(), &mut work).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(work.chain_batch_streams, usize::from(threads > 1));
            assert_eq!(work.arena_passes, 0);
            execution::serial(|| {
                let mut work = Work::default();
                assert_eq!(
                    run(&[&input], &[37], &config(), &mut work).unwrap(),
                    expected
                );
                assert_eq!(work.chain_batch_streams, 0);
            });
        });
    }
}
