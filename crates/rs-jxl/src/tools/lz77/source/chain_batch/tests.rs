use super::super::matchers::Chain;
use super::super::{Command, Literal, SourceParallelism, Work};
use super::*;

fn config() -> SourceConfig {
    SourceConfig {
        method: SourceLz77Method::Optc256,
        chain_batch: true,
        require_global_savings: false,
        ..SourceConfig::default()
    }
}

fn layout(window: usize, refined: bool) -> Layout {
    Layout {
        window,
        fine_buckets: if refined { 1 << 20 } else { 0 },
        bytes: 0,
    }
}

fn fixture(size: usize) -> Vec<u32> {
    let mut state = 0x163277bdu32;
    let mut values = Vec::new();
    for p in 0..size {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values.push(if p % 311 < 37 {
            0
        } else if p % 113 >= 81 && p >= 113 {
            values[p - 113]
        } else {
            state
        });
    }
    values
}

#[test]
fn immutable_chain_preserves_candidates_across_empty_tail_wrap_and_zero_chains() {
    let mut mixed = fixture(777);
    mixed.extend([0; 279]);
    mixed.extend([u32::MAX; 73]);
    for values in [Vec::new(), vec![7], vec![7, 7], mixed] {
        for window in [1, 4, 16, 64, 256] {
            for cap in [1, 8, 256] {
                for (width, maximum) in [(0, 3), (1, 31), (37, usize::MAX)] {
                    let config = SourceConfig {
                        chain_limit: Some(cap),
                        max_match_length: maximum,
                        ..config()
                    };
                    let plain = Index::new(&values, width, &config, layout(window, false)).unwrap();
                    let refined =
                        Index::new(&values, width, &config, layout(window, true)).unwrap();
                    let mut source =
                        Chain::with_window(&values, width, cap as usize, maximum, window).unwrap();
                    for p in 0..values.len() {
                        source.update(p);
                        let mut expected = Vec::new();
                        source.find_matches(p, false, |m| expected.push(m));
                        for index in [&plain, &refined] {
                            let mut actual = Vec::new();
                            index.find(p, |m| actual.push(m));
                            assert_eq!(
                                actual, expected,
                                "p={p} window={window} cap={cap} width={width} max={maximum}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn refined_prefixes_count_skipped_collisions_against_the_original_chain_budget() {
    let prefix = [7, 8, 9];
    let target = hash(&prefix, 0);
    let mut values = vec![7, 8, 9, 777];
    let mut collisions = 0;
    for last in 10..2_000_000u32 {
        let candidate = [7, 8, last];
        let key = hash(&candidate, 0);
        if key & 32767 == target & 32767 && key & ((1 << 20) - 1) != target & ((1 << 20) - 1) {
            values.extend(candidate);
            values.push(100000 + collisions);
            collisions += 1;
            if collisions == 12 {
                break;
            }
        }
    }
    assert_eq!(collisions, 12);
    let position = values.len();
    values.extend(prefix);
    for cap in [8, 256] {
        let config = SourceConfig {
            chain_limit: Some(cap),
            ..config()
        };
        let index = Index::new(&values, 0, &config, layout(128, true)).unwrap();
        let mut source = Chain::with_window(&values, 0, cap as usize, usize::MAX, 128).unwrap();
        source.update_range(0, position + 1);
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        source.find_matches(position, false, |m| expected.push(m));
        index.find(position, |m| actual.push(m));
        assert_eq!(actual, expected);
        assert_eq!(actual.iter().any(|m| m.distance == position), cap == 256);
    }
}

fn batch<'a>(values: &'a [u32], width: usize, workers: usize) -> Batch<'a> {
    Batch {
        index: Index::new(values, width, &config(), layout(window(values.len()), true)).unwrap(),
        frontiers: filled(values.len().min(BATCH), Dominance::new()).unwrap(),
        begin: 0,
        end: 0,
        workers,
        width,
        computed: 0,
        consumed: 0,
        overflows: 0,
        demand_only: false,
    }
}

fn frontier(find: impl FnOnce(&mut dyn FnMut(Match))) -> Vec<Match> {
    let mut result = super::super::Frontier::new(256).unwrap();
    find(&mut |m: Match| result.add(m, f64::from(m.symbol)));
    result.entries.iter().map(|entry| entry.matched).collect()
}

#[test]
fn bounded_batches_preserve_skipped_origins_refills_tail_and_parallel_frontiers() {
    let values = fixture(BATCH * 2 + 37);
    for width in [0, 1, 37] {
        let mut expected = batch(&values, width, 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        pool.install(|| {
            let mut actual = batch(&values, width, 3);
            for p in (0..BATCH)
                .step_by(3)
                .chain([BATCH + 81, BATCH * 2 + 35, values.len() - 1])
            {
                let reference = frontier(|visit| expected.index.find(p, visit));
                assert_eq!(frontier(|visit| expected.find_matches(p, visit)), reference);
                assert_eq!(frontier(|visit| actual.find_matches(p, visit)), reference);
            }
            assert_eq!(actual.computed, expected.computed);
            assert!(actual.computed > actual.consumed);
            assert!(!actual.demand_only);
            assert_eq!(actual.computed, values.len() - 81);
            assert!(actual.frontiers.len() <= BATCH);
        });
    }
}

#[test]
fn repeated_skips_disable_further_batch_work_without_changing_matches() {
    let values = fixture(BATCH * 2 + 37);
    let mut batch = batch(&values, 37, 1);
    for position in [0, BATCH + 81, values.len() - 1] {
        let expected = frontier(|visit| batch.index.find(position, visit));
        assert_eq!(
            frontier(|visit| batch.find_matches(position, visit)),
            expected
        );
    }
    assert!(batch.demand_only);
    assert_eq!(batch.computed, BATCH);
    assert_eq!(batch.consumed, 3);
}

#[test]
fn overflowing_symbol_frontiers_requery_without_losing_longer_matches() {
    let prefix: Vec<u32> = (100..140).collect();
    let mut values = Vec::new();
    for length in (3..=24).rev() {
        values.extend_from_slice(&prefix[..length]);
        values.push(10000 + length as u32);
    }
    let position = values.len();
    values.extend_from_slice(&prefix);
    let mut batch = batch(&values, 0, 1);
    let reference = frontier(|visit| batch.index.find(position, visit));
    assert!(reference.len() > 16);
    assert_eq!(
        frontier(|visit| batch.find_matches(position, visit)),
        reference
    );
    assert_eq!(batch.overflows, 1);
}

#[test]
fn speculative_length_cap_defers_long_copies_without_truncating_consumed_frontiers() {
    let mut values = fixture(300);
    values.extend(std::iter::repeat_n(17, BATCH + 500));
    values.extend(std::iter::repeat_n(0, 500));
    let mut batch = batch(&values, 37, 1);
    for position in [0, 301, 349, 600, BATCH + 350, values.len() - 400] {
        let expected = frontier(|visit| batch.index.find(position, visit));
        assert_eq!(
            frontier(|visit| batch.find_matches(position, visit)),
            expected
        );
    }
    assert!(batch.overflows > 0);
    let index = &batch.index;
    let mut limited = Vec::new();
    index.find_limited(301, SPECULATIVE_MATCH, |m| limited.push(m));
    assert!(limited.iter().any(|m| m.length == SPECULATIVE_MATCH));
    assert!(limited.iter().all(|m| m.length <= SPECULATIVE_MATCH));
    assert_eq!(limited.len(), 1);
    let position = values.len() - 400;
    let mut zero_batch = self::batch(&values, 37, 1);
    let expected = frontier(|visit| zero_batch.index.find(position, visit));
    assert_eq!(
        frontier(|visit| zero_batch.find_matches(position, visit)),
        expected
    );
    assert_eq!(zero_batch.overflows, 1);
    for limit in [127, 128, 129] {
        let mut bounded =
            Index::new(&values, 37, &config(), layout(window(values.len()), true)).unwrap();
        bounded.maximum = limit;
        for position in [
            301,
            values.len() - 127,
            values.len() - 128,
            values.len() - 129,
        ] {
            let full = frontier(|visit| bounded.find(position, visit));
            let mut truncated = false;
            let prefix = frontier(|visit| {
                bounded.find_limited(position, SPECULATIVE_MATCH, |m| {
                    truncated |= limit.min(values.len() - position) > SPECULATIVE_MATCH
                        && m.length == SPECULATIVE_MATCH;
                    visit(m);
                })
            });
            if !truncated {
                assert_eq!(prefix, full);
            }
        }
    }
}

#[test]
fn chain_batch_route_and_memory_gates_leave_other_methods_and_repetition_alone() {
    let base = config();
    assert!(!SourceConfig::default().chain_batch);
    assert!(enabled(65536, 65536, &base));
    for size in [0, 1, 2, 65535, (1 << 24) + 1] {
        assert!(!enabled(size, size, &base));
    }
    assert!(!enabled(65536, 16384, &base));
    assert!(enabled(65536, 16385, &base));
    for config in [
        SourceConfig {
            chain_batch: false,
            ..base.clone()
        },
        SourceConfig {
            chain_batch_bytes: 0,
            ..base.clone()
        },
        SourceConfig {
            bucket_matcher: true,
            ..base.clone()
        },
        SourceConfig {
            method: SourceLz77Method::Optc8,
            ..base.clone()
        },
        SourceConfig {
            match_cost: SourceMatchCost::Frontier,
            ..base.clone()
        },
        SourceConfig {
            match_cost: SourceMatchCost::RangeReplay,
            ..base.clone()
        },
        SourceConfig {
            index_mode: SourceIndexMode::All,
            ..base.clone()
        },
        SourceConfig {
            parallelism: Some(SourceParallelism {
                scratch_bytes: 1,
                ..SourceParallelism::default()
            }),
            ..base
        },
    ] {
        assert!(!enabled(65536, 65536, &config));
    }
    let full = Layout::new(1 << 20, usize::MAX).unwrap();
    assert_ne!(full.fine_buckets, 0);
    let coarse = Layout::new(1 << 20, full.bytes - 1).unwrap();
    assert_eq!(coarse.fine_buckets, 0);
    assert!(Layout::new(1 << 20, coarse.bytes - 1).is_none());
    assert_eq!(
        Layout::new(1 << 20, coarse.bytes).unwrap().bytes,
        coarse.bytes
    );
}

fn replay(input: &[Literal], commands: &[Command]) {
    let mut decoded = Vec::new();
    for &command in commands {
        match command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[decoded.len()]);
                decoded.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[decoded.len()].context);
                assert!(length >= 3 && distance > 0 && distance as usize <= decoded.len());
                for _ in 0..length {
                    decoded.push(decoded[decoded.len() - distance as usize]);
                }
            }
        }
    }
    assert_eq!(decoded, input.iter().map(|t| t.value).collect::<Vec<_>>());
}

#[test]
fn chain_batches_preserve_global_dp_commands_acceptance_contexts_and_pool_sizes() {
    let input: Vec<_> = fixture(65536)
        .into_iter()
        .enumerate()
        .map(|(p, value)| Literal {
            value,
            context: [0, 7, u32::MAX][p % 3],
        })
        .collect();
    let inputs = [&input[..], &[], &input[..1024]];
    for compact_steps in [false, true] {
        let baseline = SourceConfig {
            chain_batch: false,
            compact_steps,
            ..config()
        };
        let expected =
            super::super::run(&inputs, &[37, 0, 1], &baseline, &mut Work::default()).unwrap();
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let config = SourceConfig {
                chain_batch: true,
                parallelism: Some(SourceParallelism::default()),
                ..baseline.clone()
            };
            let mut work = Work::default();
            let actual = pool
                .install(|| super::super::run(&inputs, &[37, 0, 1], &config, &mut work))
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(work.chain_batch_streams, 1);
            for (input, commands) in inputs.iter().zip(&actual.commands) {
                replay(input, commands);
            }
        }
    }
}

#[test]
fn immutable_chains_preserve_the_full_dictionary_window_boundary() {
    let window = super::super::WINDOW_SIZE;
    let mut values = fixture(window + 67);
    for i in 0..67 {
        values[window + i] = values[i + 1];
    }
    let config = SourceConfig {
        max_match_length: 97,
        ..config()
    };
    let index = Index::new(&values, 37, &config, layout(window, true)).unwrap();
    let mut source = Chain::new(&values, 37, 256, 97).unwrap();
    source.update_range(0, window - 7);
    for p in window - 7..values.len() {
        source.update(p);
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        source.find_matches(p, false, |m| expected.push(m));
        index.find(p, |m| actual.push(m));
        assert_eq!(actual, expected, "p={p}");
        assert!(actual.iter().all(|m| m.distance < window));
    }
}
