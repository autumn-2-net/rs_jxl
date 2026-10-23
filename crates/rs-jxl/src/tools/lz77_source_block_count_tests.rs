// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Source contracts from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::{SourceParallelism, StaticIndex};
use super::*;
use crate::execution;

const SIZE: usize = 8 * 65536;
const MAXIMUM: usize = 4093;

fn fixture() -> Vec<Literal> {
    (0..SIZE)
        .map(|position| Literal {
            value: 65536,
            context: (position % 3) as u32,
        })
        .collect()
}

fn config() -> SourceConfig {
    SourceConfig {
        block_mode: SourceBlockMode::Intervals,
        max_match_length: MAXIMUM,
        parallelism: Some(SourceParallelism::default()),
        ..SourceConfig::default()
    }
}

fn expected_cuts(parts: usize) -> Vec<usize> {
    // Every seed token is a literal, so source LZ77ParseCuts lands exactly at
    // each integer target, independently of the implementation's cut builder.
    (0..=parts).map(|part| SIZE * part / parts).collect()
}

fn nominate(position: usize, maximum: usize, visit: &mut dyn FnMut(Match)) {
    if position != 0 && maximum >= MIN_MATCH {
        visit(Match {
            length: maximum.min(MAXIMUM),
            distance: 1,
            symbol: 0,
        });
    }
}

fn decode(input: &[Literal], commands: &[Command], cuts: &[usize]) {
    let mut decoded = Vec::with_capacity(input.len());
    let mut copies = 0;
    for command in commands {
        let position = decoded.len();
        assert!(position < input.len());
        let end = cuts[cuts.partition_point(|&cut| cut <= position)];
        match *command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[position]);
                decoded.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[position].context);
                assert!(length as usize >= MIN_MATCH);
                assert!(distance != 0 && distance as usize <= position);
                assert!(position + length as usize <= end);
                copies += 1;
                for _ in 0..length {
                    decoded.push(decoded[decoded.len() - distance as usize]);
                }
            }
        }
    }
    assert!(copies > 0);
    assert_eq!(decoded.len(), input.len());
    assert!(
        decoded
            .iter()
            .zip(input)
            .all(|(&v, literal)| v == literal.value)
    );
}

#[test]
fn two_threads_keep_eight_logical_blocks_and_grain_with_scalar_cut_oracle() {
    let input = fixture();
    let seed: Vec<_> = input.iter().copied().map(Command::Literal).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for (mode, grain) in [
        (SourceBlockMode::Endpoints, 1),
        (SourceBlockMode::Intervals, 1),
        (SourceBlockMode::Intervals, 3),
    ] {
        let config = SourceConfig {
            block_mode: mode,
            block_grain: grain,
            ..config()
        };
        let cuts = expected_cuts(8 * grain);
        let costs = Costs::raw(&[&input], &config);
        let prices = DistancePrices::new(window(SIZE), 0, &config, |d| costs.distance(d)).unwrap();
        let expected = execution::serial(|| {
            let mut commands = Vec::new();
            for cut in cuts.windows(2) {
                commands.extend(
                    parse_part(
                        &input,
                        cut[0],
                        cut[1],
                        0,
                        &config,
                        &costs,
                        &prices,
                        &[],
                        &|p, query: Query<'_>, _, _, visit| nominate(p, query.maximum, visit),
                    )
                    .unwrap(),
                );
            }
            commands
        });
        let observed: Vec<_> = (0..cuts.len() - 1).map(|_| AtomicUsize::new(0)).collect();
        let actual = pool.install(|| {
            assert_eq!(workers::index_tasks(&config, 8, 8 << 20), 8);
            assert_eq!(workers::index_workers(&config, 8, 8 << 20), 2);
            parse(
                &input,
                &seed,
                0,
                &config,
                &costs,
                |p, query, _, _, visit| {
                    assert!(execution::is_serial());
                    assert!(pool.current_thread_index().unwrap() < 2);
                    let block = cuts.binary_search(&(p + query.maximum)).unwrap() - 1;
                    assert!((cuts[block]..cuts[block + 1]).contains(&p));
                    observed[block].fetch_add(1, Ordering::Relaxed);
                    nominate(p, query.maximum, visit);
                },
            )
            .unwrap()
        });
        assert!(observed.iter().all(|n| n.load(Ordering::Relaxed) > 0));
        assert_eq!(actual, expected, "mode={mode:?} grain={grain}");
        decode(&input, &actual, &cuts);
    }
}

#[test]
fn null_serial_and_scratch_limited_block_routes_keep_one_partition() {
    let input = fixture();
    let seed: Vec<_> = input.iter().copied().map(Command::Literal).collect();
    let config = config();
    let costs = Costs::raw(&[&input], &config);
    let check = |config: &SourceConfig| {
        let caller = std::thread::current().id();
        let commands = parse(&input, &seed, 0, config, &costs, |p, query, _, _, visit| {
            assert_eq!(std::thread::current().id(), caller);
            assert_eq!(p + query.maximum, SIZE);
            nominate(p, query.maximum, visit);
        })
        .unwrap();
        decode(&input, &commands, &[0, SIZE]);
        commands
    };
    assert!(rayon::current_thread_index().is_none());
    let without_pool = check(&config);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        execution::serial(|| assert_eq!(check(&config), without_pool));
        let mut disabled = config.clone();
        disabled.parallelism = None;
        assert_eq!(check(&disabled), without_pool);
        disabled.parallelism = Some(SourceParallelism {
            scratch_bytes: 1,
            ..SourceParallelism::default()
        });
        assert_eq!(check(&disabled), without_pool);
    });
}

#[test]
fn seed_partitions_also_keep_logical_counts_on_a_two_thread_pool() {
    let input = fixture();
    let values: Vec<_> = input.iter().map(|literal| literal.value).collect();
    let config = SourceConfig {
        max_match_length: 37,
        ..config()
    };
    let costs = Costs::raw(&[&input], &config);
    let index = StaticIndex::new(&values, 0, &config).unwrap();
    let cuts = expected_cuts(8);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for seed_blocks in [
            SourceSeedBlocks::Cold,
            SourceSeedBlocks::Warm { history: 17 },
        ] {
            let config = SourceConfig {
                seed_blocks,
                ..config.clone()
            };
            let parts = seed_parts(&[&input], &config, true).unwrap();
            assert_eq!(parts.len(), 8);
            for (part, cut) in parts.iter().zip(cuts.windows(2)) {
                let prefix = if seed_blocks == SourceSeedBlocks::Cold {
                    0
                } else {
                    cut[0].min(17)
                };
                assert_eq!(
                    (part.owner, part.begin, part.end, part.prefix),
                    (0, cut[0] - prefix, cut[1], prefix)
                );
            }
            execution::serial(|| {
                assert_eq!(seed_parts(&[&input], &config, true).unwrap().len(), 1)
            });
        }
        let (commands, saving, raw) = indexed_seed(&input, &index, &config, &costs).unwrap();
        assert!(saving > 0.0 && raw > saving);
        decode(&input, &commands, &cuts);
    });
}

#[test]
fn indexed_seed_null_runner_keeps_source_partitions_and_double_reduction() {
    let input = fixture();
    let values: Vec<_> = input.iter().map(|literal| literal.value).collect();
    let config = SourceConfig {
        max_match_length: 37,
        ..config()
    };
    let costs = Costs::raw(&[&input], &config);
    let index = StaticIndex::new(&values, 0, &config).unwrap();
    assert!(rayon::current_thread_index().is_none());
    assert_eq!(workers::indexed_seed_tasks(&config, input.len()), 8);
    let null = indexed_seed(&input, &index, &config, &costs).unwrap();
    decode(&input, &null.0, &expected_cuts(8));
    let default_limits = SourceConfig {
        parallelism: None,
        ..config.clone()
    };
    assert_eq!(workers::indexed_seed_tasks(&default_limits, input.len()), 8);
    assert_eq!(
        indexed_seed(&input, &index, &default_limits, &costs).unwrap(),
        null
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(indexed_seed(&input, &index, &config, &costs).unwrap(), null);
        execution::serial(|| {
            assert_eq!(workers::indexed_seed_tasks(&config, input.len()), 8);
            assert_eq!(workers::index_workers(&config, 8, 4 << 20), 1);
            assert_eq!(indexed_seed(&input, &index, &config, &costs).unwrap(), null);
        });
    });
    for (tasks, scratch, parts) in [(3, 1 << 30, 3), (8, 12 << 20, 3), (8, 1, 1)] {
        let config = SourceConfig {
            parallelism: Some(SourceParallelism {
                max_workers: tasks,
                scratch_bytes: scratch,
                fast_streams: false,
            }),
            ..config.clone()
        };
        assert_eq!(workers::indexed_seed_tasks(&config, input.len()), parts);
        let null = indexed_seed(&input, &index, &config, &costs).unwrap();
        decode(&input, &null.0, &expected_cuts(parts));
        assert_eq!(
            pool.install(|| indexed_seed(&input, &index, &config, &costs))
                .unwrap(),
            null
        );
    }
}
