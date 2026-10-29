use super::super::{
    Command, ContextCost, Costs, DistanceCodes, HybridUintConfig, Literal, MIN_MATCH, SourceConfig,
    WINDOW_SIZE,
};
use super::*;

fn reference_raw(inputs: &[&[Literal]], config: &SourceConfig) -> Costs {
    let mut counts = Counts::new();
    for input in inputs {
        for literal in *input {
            let symbol = config.literal_uint_config.encode(literal.value).token;
            *counts
                .entry(literal.context)
                .or_default()
                .entry(symbol)
                .or_default() += 1;
        }
    }
    Costs::from_counts(counts, BTreeMap::new(), config)
}

fn reference_fitted(seed: &[Vec<Command>], widths: &[usize], config: &SourceConfig) -> Costs {
    let mut counts = Counts::new();
    let mut distances = BTreeMap::new();
    for (index, stream) in seed.iter().enumerate() {
        let codes = DistanceCodes::new(widths.get(index).copied().unwrap_or(0), WINDOW_SIZE);
        for &command in stream {
            let (context, symbol) = match command {
                Command::Literal(literal) => (
                    literal.context,
                    config.literal_uint_config.encode(literal.value).token,
                ),
                Command::Match {
                    context,
                    length,
                    distance,
                } => {
                    let token = config
                        .literal_uint_config
                        .encode(codes.symbol(distance as usize))
                        .token;
                    *distances.entry(token).or_default() += 1;
                    (
                        context,
                        config.min_symbol()
                            + config
                                .length_uint_config
                                .encode(length - MIN_MATCH as u32)
                                .token,
                    )
                }
            };
            *counts
                .entry(context)
                .or_default()
                .entry(symbol)
                .or_default() += 1;
        }
    }
    Costs::from_counts(counts, distances, config)
}

fn assert_row(actual: &ContextCost, expected: &ContextCost) {
    assert_eq!(actual.penalty.to_bits(), expected.penalty.to_bits());
    assert!(actual.bits.keys().eq(expected.bits.keys()));
    for (&symbol, &cost) in &expected.bits {
        assert_eq!(
            actual.bits[&symbol].to_bits(),
            cost.to_bits(),
            "symbol={symbol}"
        );
    }
}

fn assert_costs(actual: &Costs, expected: &Costs) {
    assert!(actual.literals.keys().eq(expected.literals.keys()));
    for (&context, row) in &expected.literals {
        assert_row(&actual.literals[&context], row);
    }
    assert_row(&actual.distances, &expected.distances);
}

fn configs() -> Vec<SourceConfig> {
    let mut configs = Vec::new();
    for fused_cost_math in [false, true] {
        for force_huffman in [false, true] {
            for (literal_uint_config, length_uint_config, min_symbol) in [
                (
                    HybridUintConfig::default(),
                    HybridUintConfig::new(0, 0, 0).unwrap(),
                    None,
                ),
                (
                    HybridUintConfig::new(0, 0, 0).unwrap(),
                    HybridUintConfig::new(4, 0, 0).unwrap(),
                    None,
                ),
                (
                    HybridUintConfig::new(8, 4, 4).unwrap(),
                    HybridUintConfig::new(4, 2, 0).unwrap(),
                    Some(1 << 20),
                ),
                (
                    HybridUintConfig::new(15, 15, 0).unwrap(),
                    HybridUintConfig::new(15, 15, 0).unwrap(),
                    Some(1 << 20),
                ),
            ] {
                configs.push(SourceConfig {
                    literal_uint_config,
                    length_uint_config,
                    min_symbol,
                    force_huffman,
                    fused_cost_math,
                    ..SourceConfig::default()
                });
            }
        }
    }
    configs
}

fn input(size: usize) -> Vec<Literal> {
    let mut state = 0xabcdef01u32;
    (0..size)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            Literal {
                context: [0, 1, 37, 255, 65535, 65536, u32::MAX][i % 7],
                value: match i % 11 {
                    0..=2 => 0,
                    3 => 255,
                    4 => 256,
                    5 => 65535,
                    6 => 65536,
                    7 => u32::MAX,
                    _ => state,
                },
            }
        })
        .collect()
}

#[test]
fn context_counts_promote_only_repeated_small_symbols() {
    let mut counts = ContextCounts::new(MIN_DENSE_TOKENS);
    let mut expected = Counts::new();
    for context in 0..1000 {
        counts.increment(context, 0);
        expected.entry(context).or_default().insert(0, 1);
    }
    assert!(counts.rows.is_empty());
    for _ in 1..PROMOTE_AFTER {
        counts.increment(7, 255);
        *expected.entry(7).or_default().entry(255).or_default() += 1;
    }
    for _ in 0..100 {
        counts.increment(7, 256);
        *expected.entry(7).or_default().entry(256).or_default() += 1;
    }
    assert_eq!(counts.contexts[7], READY);
    assert!(counts.rows.is_empty());
    counts.increment(7, 255);
    *expected.entry(7).or_default().entry(255).or_default() += 1;
    assert_eq!(counts.rows.len(), 1);
    assert_eq!(counts.rows[0].counts[255], 1);
    assert_eq!(counts.into_counts(), expected);
}

#[test]
fn context_counts_merge_dense_and_sparse_keys_in_order() {
    let mut counts = ContextCounts::new(MIN_DENSE_TOKENS);
    let mut expected = Counts::new();
    for round in 0..100 {
        for context in [u32::MAX, 65536, 65535, 37, 1, 0] {
            for symbol in [u32::MAX, 65536, 256, 255, 224, 7, 0] {
                for _ in 0..1 + round % 3 {
                    counts.increment(context, symbol);
                    *expected
                        .entry(context)
                        .or_default()
                        .entry(symbol)
                        .or_default() += 1;
                }
            }
        }
    }
    assert_eq!(counts.rows.len(), 4);
    assert_eq!(counts.contexts.len(), CONTEXTS);
    let actual = counts.into_counts();
    assert_eq!(actual, expected);
    for config in configs() {
        assert_costs(
            &Costs::from_counts(actual.clone(), BTreeMap::new(), &config),
            &Costs::from_counts(expected.clone(), BTreeMap::new(), &config),
        );
    }
}

#[test]
fn context_counts_row_budget_falls_back_without_losing_counts() {
    let mut counts = ContextCounts::new(MIN_DENSE_TOKENS);
    let mut expected = Counts::new();
    for context in 0..(ROWS + 3) as u32 {
        for _ in 0..usize::from(PROMOTE_AFTER) + 7 {
            counts.increment(context, 1);
            *expected.entry(context).or_default().entry(1).or_default() += 1;
        }
        counts.increment(context, 256);
        expected.entry(context).or_default().insert(256, 1);
    }
    assert_eq!(counts.rows.len(), ROWS);
    assert!(counts.rows.len() * std::mem::size_of::<[usize; SYMBOLS]>() <= 8 << 20);
    assert_eq!(counts.contexts[ROWS - 1], (ROWS - 1) as u16);
    assert_eq!(counts.contexts[ROWS], READY);
    assert_eq!(counts.into_counts(), expected);
}

#[test]
fn raw_narrow_rows_preserve_high_symbols_and_sparse_contexts() {
    let mut counts = RawCounts::new(MIN_DENSE_TOKENS);
    let mut expected = Counts::new();
    for context in [0, 37, 65535, 65536, u32::MAX] {
        for round in 0..100 {
            for symbol in [0, 1, 63, 64, 127, 224, 255, 256, u32::MAX] {
                for _ in 0..1 + round % 3 {
                    counts.increment(context, symbol);
                    *expected
                        .entry(context)
                        .or_default()
                        .entry(symbol)
                        .or_default() += 1;
                }
            }
        }
    }
    assert_eq!(counts.rows.len(), 3);
    assert!(counts.rows.iter().all(|row| row.counts.len() == 64));
    assert_eq!(counts.into_counts(), expected);
}

#[test]
fn raw_row_budget_covers_16384_contexts_at_the_same_bin_memory_limit() {
    let mut counts = RawCounts::new(MIN_DENSE_TOKENS);
    let mut expected = Counts::new();
    for context in 0..16387 {
        for _ in 0..usize::from(PROMOTE_AFTER) + 3 {
            counts.increment(context, 63);
            *expected.entry(context).or_default().entry(63).or_default() += 1;
        }
        counts.increment(context, 64);
        expected.entry(context).or_default().insert(64, 1);
    }
    assert_eq!(counts.rows.len(), 16384);
    assert_eq!(counts.rows.len() * 64, ROWS * SYMBOLS);
    assert_eq!(counts.contexts[16383], 16383);
    assert_eq!(counts.contexts[16384], READY);
    assert_eq!(counts.into_counts(), expected);
}

#[test]
fn raw_known_domain_counts_every_context_densely_from_first_use() {
    let input: Vec<_> = (0..16384u32)
        .rev()
        .map(|context| Literal {
            context,
            value: context % 64,
        })
        .collect();
    let inputs = [&input[..1], &[], &input[1..]];
    let mut counts = RawCounts::for_inputs(&inputs, input.len());
    assert_eq!(counts.promote_after, 0);
    assert_eq!(counts.contexts.len(), 16384);
    let mut expected = Counts::new();
    for literal in &input {
        counts.increment(literal.context, literal.value);
        expected
            .entry(literal.context)
            .or_default()
            .insert(literal.value, 1);
    }
    assert_eq!(counts.rows.len(), 16384);
    assert!(counts.sparse.is_empty());
    assert_eq!(counts.rows.len() * 64, ROWS * SYMBOLS);
    assert_eq!(counts.into_counts(), expected);
}

#[test]
fn raw_known_domain_keeps_high_symbol_counts_sparse_and_ordered() {
    let input: Vec<_> = (0..MIN_DENSE_TOKENS)
        .map(|i| Literal {
            context: [16383, 37, 0][i % 3],
            value: [0, 63, 64, 255, 256, u32::MAX][i % 6],
        })
        .collect();
    let mut counts = RawCounts::for_inputs(&[&input], input.len());
    let mut expected = Counts::new();
    for literal in &input {
        counts.increment(literal.context, literal.value);
        *expected
            .entry(literal.context)
            .or_default()
            .entry(literal.value)
            .or_default() += 1;
    }
    assert_eq!(counts.promote_after, 0);
    assert!(
        counts
            .sparse
            .values()
            .all(|row| row.keys().all(|&s| s >= 64))
    );
    assert_eq!(counts.into_counts(), expected);
}

#[test]
fn raw_domain_preflight_checks_all_streams_before_skipping_probation() {
    let warm = vec![
        Literal {
            context: 7,
            value: 0
        };
        MIN_DENSE_TOKENS
    ];
    for context in [16383, 16384, 65535, 65536, u32::MAX] {
        let tail = [Literal { context, value: 1 }];
        let inputs = [&warm[..], &[], &tail[..]];
        let mut counts = RawCounts::for_inputs(&inputs, warm.len() + 1);
        assert_eq!(
            counts.promote_after,
            if context < 16384 { 0 } else { PROMOTE_AFTER }
        );
        let mut expected = Counts::new();
        for input in inputs {
            for literal in input {
                counts.increment(literal.context, literal.value);
                *expected
                    .entry(literal.context)
                    .or_default()
                    .entry(literal.value)
                    .or_default() += 1;
            }
        }
        if context >= 16384 {
            assert_eq!(counts.sparse[&7][&0], PROMOTE_AFTER as usize);
        }
        assert_eq!(counts.into_counts(), expected);
    }
    for size in [0, 1, MIN_DENSE_TOKENS - 1] {
        let counts = RawCounts::for_inputs(&[&warm[..size]], size);
        assert!(counts.contexts.is_empty());
        assert_eq!(counts.promote_after, PROMOTE_AFTER);
    }
}

#[test]
fn raw_first_use_domain_preserves_all_cost_bits_and_weighted_order() {
    let mut input = input(16387);
    for (i, literal) in input.iter_mut().enumerate() {
        literal.context = if i % 3 == 0 {
            (i % 137) as u32
        } else {
            (i % 16384) as u32
        };
    }
    let inputs = [&input[..73], &input[73..8192], &[], &input[8192..]];
    assert_eq!(RawCounts::for_inputs(&inputs, input.len()).promote_after, 0);
    for config in configs() {
        assert_costs(
            &Costs::raw(&inputs, &config),
            &reference_raw(&inputs, &config),
        );
    }
}

#[test]
fn small_models_skip_dense_context_allocations() {
    let mut counts = ContextCounts::new(MIN_DENSE_TOKENS - 1);
    for _ in 0..MIN_DENSE_TOKENS - 1 {
        counts.increment(0, 0);
    }
    assert!(counts.contexts.is_empty());
    assert!(counts.rows.is_empty());
    assert_eq!(
        counts.into_counts(),
        [(0, [(0, MIN_DENSE_TOKENS - 1)].into_iter().collect())]
            .into_iter()
            .collect()
    );
}

#[test]
fn symbol_counts_keep_large_tokens_and_omit_zero_bins() {
    assert!(SymbolCounts::new().into_counts().is_empty());
    let mut counts = SymbolCounts::new();
    let mut expected = BTreeMap::new();
    for (i, symbol) in [u32::MAX, 65536, 256, 255, 1, 0].into_iter().enumerate() {
        for _ in 0..i + 1 {
            counts.increment(symbol);
            *expected.entry(symbol).or_default() += 1;
        }
    }
    assert_eq!(counts.into_counts(), expected);
}

#[test]
fn dense_merge_preserves_usize_weights() {
    let mut dense = [0; SYMBOLS];
    let large = usize::MAX / 4;
    dense[2] = large;
    dense[255] = large + 17;
    let mut sparse = [(2, 37), (256, 19)].into_iter().collect();
    merge_dense(&dense, &mut sparse);
    let expected = [(2, large + 37), (255, large + 17), (256, 19)]
        .into_iter()
        .collect();
    assert_eq!(sparse, expected);
}

#[test]
fn raw_histograms_match_tree_reference_bitwise() {
    let input = input(16387);
    let inputs = [&input[..73], &input[73..8192], &input[8192..], &[]];
    for config in configs() {
        assert_costs(
            &Costs::raw(&inputs, &config),
            &reference_raw(&inputs, &config),
        );
    }
}

#[test]
fn fitted_histograms_match_tree_reference_bitwise() {
    let lengths = [3, 4, 127, 128, 129, 65536, i32::MAX as u32];
    let distances = [1, 2, 37, 38, 4096, WINDOW_SIZE as u32];
    let mut seed = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (i, literal) in input(16387).into_iter().enumerate() {
        let command = if i % 3 == 0 {
            Command::Literal(literal)
        } else {
            Command::Match {
                context: literal.context,
                length: lengths[(i / 7) % lengths.len()],
                distance: distances[(i / 11) % distances.len()],
            }
        };
        seed[i % 3].push(command);
    }
    for config in configs() {
        for widths in [&[0, 37, 4096][..], &[37][..], &[][..]] {
            assert_costs(
                &Costs::fitted(&seed, widths, &config),
                &reference_fitted(&seed, widths, &config),
            );
        }
    }
}

#[test]
fn histogram_empty_and_all_literal_thresholds_are_bitwise_equivalent() {
    let config = SourceConfig::default();
    assert_costs(&Costs::raw(&[], &config), &reference_raw(&[], &config));
    assert_costs(
        &Costs::fitted(&[], &[], &config),
        &reference_fitted(&[], &[], &config),
    );
    for size in [
        0,
        1,
        MIN_DENSE_TOKENS - 1,
        MIN_DENSE_TOKENS,
        MIN_DENSE_TOKENS + 1,
    ] {
        let input = input(size);
        let inputs = [&input[..]];
        let seed = vec![input.iter().copied().map(Command::Literal).collect()];
        let expected = reference_raw(&inputs, &config);
        assert_costs(&Costs::raw(&inputs, &config), &expected);
        assert_costs(&Costs::fitted(&seed, &[], &config), &expected);
    }
}

#[test]
fn fitted_counts_commands_not_match_length_and_keeps_distance_config() {
    for fused_cost_math in [false, true] {
        for force_huffman in [false, true] {
            let config = SourceConfig {
                literal_uint_config: HybridUintConfig::new(0, 0, 0).unwrap(),
                length_uint_config: HybridUintConfig::new(4, 0, 0).unwrap(),
                force_huffman,
                fused_cost_math,
                ..SourceConfig::default()
            };
            let long = i32::MAX as u32;
            let mut stream = vec![
                Command::Literal(Literal {
                    context: 7,
                    value: 0
                });
                4096
            ];
            for length in [MIN_MATCH as u32, long] {
                stream.extend(std::iter::repeat_n(
                    Command::Match {
                        context: 7,
                        length,
                        distance: 32,
                    },
                    1024,
                ));
            }
            let counts = [(
                7,
                [
                    (0, 4096),
                    (config.min_symbol(), 1024),
                    (
                        config.min_symbol()
                            + config
                                .length_uint_config
                                .encode(long - MIN_MATCH as u32)
                                .token,
                        1024,
                    ),
                ]
                .into_iter()
                .collect(),
            )]
            .into_iter()
            .collect();
            let distance = config.literal_uint_config.encode(31).token;
            assert_ne!(distance, HybridUintConfig::default().encode(31).token);
            let distances = [(distance, 2048)].into_iter().collect();
            let expected = Costs::from_counts(counts, distances, &config);
            assert_costs(&Costs::fitted(&[stream], &[0], &config), &expected);
        }
    }
}
