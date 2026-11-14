// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;

fn input(values: impl IntoIterator<Item = u32>) -> Vec<Literal> {
    values
        .into_iter()
        .enumerate()
        .map(|(i, value)| Literal {
            context: (i % 3) as u32,
            value,
        })
        .collect()
}

fn assert_public_literals(input: &[Literal], config: &SourceConfig, enabled: bool) {
    let expected = literals(input).unwrap();
    let owned = parse_source_with_state(input, 0, config).unwrap();
    assert_eq!(owned.enabled, enabled);
    assert_eq!(owned.commands, expected);
    assert_eq!(parse_source(input, 0, config).unwrap(), expected);
    let streams = parse_source_streams_with_state(&[input], &[], config).unwrap();
    assert_eq!(streams.enabled, enabled);
    assert_eq!(streams.commands, vec![expected.clone()]);
    assert_eq!(
        parse_source_streams(&[input], &[], config).unwrap(),
        vec![expected]
    );
}

#[test]
fn borrowed_no_matches_preserve_owned_public_results_and_gate_override() {
    let input = input(0..257);
    for method in [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Lz77b7w3t,
        SourceLz77Method::Lz77b15w3t,
        SourceLz77Method::Lz77b31w3t,
        SourceLz77Method::Lz77b1w3f,
        SourceLz77Method::Lz77b3w3f,
        SourceLz77Method::Lz77b7w3f,
        SourceLz77Method::Lz77b15w3f,
        SourceLz77Method::Lz77b31w3f,
    ] {
        for require_global_savings in [true, false] {
            let config = SourceConfig {
                method,
                require_global_savings,
                ..Default::default()
            };
            for size in [0, 1, 2, input.len()] {
                let input = &input[..size];
                assert_eq!(
                    SourceParseOutcome::parse_borrowed(input, 0, &config).unwrap(),
                    (None, !require_global_savings)
                );
                assert_public_literals(input, &config, !require_global_savings);
            }
        }
    }
}

#[test]
fn borrowed_global_rejection_discards_matched_seed_without_literal_commands() {
    let input = vec![
        Literal {
            context: 0,
            value: 0
        };
        128
    ];
    let config = SourceConfig::new(SourceLz77Method::Lz77b1w3f);
    let forced = SourceConfig {
        require_global_savings: false,
        ..config.clone()
    };
    let (commands, enabled) = SourceParseOutcome::parse_borrowed(&input, 0, &forced).unwrap();
    assert!(enabled);
    assert!(
        commands
            .unwrap()
            .iter()
            .any(|c| matches!(c, Command::Match { .. }))
    );
    assert_eq!(
        SourceParseOutcome::parse_borrowed(&input, 0, &config).unwrap(),
        (None, false)
    );
    assert_public_literals(&input, &config, false);

    let checked = SourceConfig {
        method: SourceLz77Method::Lz77b1w3t,
        ..forced
    };
    // Exact repeats can all fail the per-match price check; still no commands.
    assert_eq!(
        SourceParseOutcome::parse_borrowed(&input, 0, &checked).unwrap(),
        (None, true)
    );
    assert_public_literals(&input, &checked, true);
}

#[test]
fn borrowed_fast_prefix_middle_and_tail_literals_keep_exact_commands() {
    let input = input(
        (0..64)
            .chain(0..64)
            .chain([1000])
            .chain(0..64)
            .chain([1001, 1002]),
    );
    let mut expected = literals(&input[..64]).unwrap();
    expected.push(Command::Match {
        context: input[64].context,
        length: 64,
        distance: 64,
    });
    expected.push(Command::Literal(input[128]));
    expected.push(Command::Match {
        context: input[129].context,
        length: 64,
        distance: 65,
    });
    expected.extend(literals(&input[193..]).unwrap());
    for method in [SourceLz77Method::Lz77b1w3t, SourceLz77Method::Lz77b3w3f] {
        for prefix_filter in [FilterPolicy::Disabled, FilterPolicy::Eager] {
            let config = SourceConfig {
                method,
                prefix_filter,
                require_global_savings: false,
                ..Default::default()
            };
            let (commands, enabled) =
                SourceParseOutcome::parse_borrowed(&input, 64, &config).unwrap();
            assert!(enabled);
            assert_eq!(commands.unwrap(), expected);
            assert_eq!(parse_source(&input, 64, &config).unwrap(), expected);
        }
    }
}

#[test]
fn borrowed_scout_literal_shortcut_preserves_public_commands() {
    let input = input((0..2048).map(|n| 0xf000_0000 + n));
    let receipt = std::sync::Arc::new(SourceScoutReceipt::default());
    let config = SourceConfig {
        method: SourceLz77Method::Optc8,
        scout: SourceScout {
            mode: SourceScoutMode::Route,
            receipt: Some(receipt.clone()),
        },
        ..Default::default()
    };
    assert_eq!(
        SourceParseOutcome::parse_borrowed(&input, 0, &config).unwrap(),
        (None, false)
    );
    assert_eq!(receipt.snapshot().decision, SourceScoutDecision::Literal);
    assert_public_literals(&input, &config, false);
}

#[test]
fn borrowed_streams_preserve_mixed_literal_and_match_streams() {
    let unique = input(0..257);
    let repeated = input((0..64).cycle().take(1024));
    let inputs = [&unique[..], &[][..], &repeated[..]];
    let config = SourceConfig {
        method: SourceLz77Method::Lz77b3w3t,
        require_global_savings: false,
        ..Default::default()
    };
    let outcome = run_borrowed(&inputs, &[], &config, &mut Work::default()).unwrap();
    assert!(outcome.enabled);
    let commands = outcome.commands.unwrap();
    assert_eq!(commands[0], literals(&unique).unwrap());
    assert!(commands[1].is_empty());
    assert!(
        commands[2]
            .iter()
            .any(|c| matches!(c, Command::Match { .. }))
    );
    assert_eq!(
        parse_source_streams(&inputs, &[], &config).unwrap(),
        commands
    );
}

#[test]
fn fast_seed_wire_tokens_match_command_scan_for_matches_and_cost_rejection() {
    let mixed = input(
        (0..64)
            .chain(0..64)
            .chain([1000])
            .chain(0..64)
            .chain([1001, 1002]),
    );
    let unique = input(0..257);
    for method in [SourceLz77Method::Lz77b1w3t, SourceLz77Method::Lz77b1w3f] {
        for max_match_length in [MIN_MATCH, 79] {
            let config = SourceConfig {
                method,
                max_match_length,
                ..Default::default()
            };
            for input in [&[][..], &mixed[..1], &mixed[..2], &unique, &mixed] {
                let values: Vec<_> = input.iter().map(|literal| literal.value).collect();
                let costs = Costs::raw(&[input], &config);
                let penalties = prices::Penalties::new(&costs).unwrap();
                // Zero prices reject every match in the checked method;
                // high prices accept even minimum-length matches.
                for bits in [0.0, 32.0] {
                    let sums: Vec<_> = (0..=input.len()).map(|i| i as f32 * bits).collect();
                    let mut saving = 0.0;
                    let (commands, tokens) = fast_stream_borrowed::<1>(
                        input,
                        &values,
                        64,
                        &config,
                        &penalties,
                        &sums,
                        &mut saving,
                    )
                    .unwrap();
                    let scanned = commands
                        .as_ref()
                        .map_or(input.len(), |commands| wire_token_count(commands));
                    assert_eq!(tokens, scanned);
                    if bits == 0.0 && method.runtime_cost_comparison() {
                        assert!(commands.is_none());
                        assert_eq!(tokens, input.len());
                        assert_eq!(saving, 0.0);
                    } else if input.len() == mixed.len() {
                        assert!(commands.is_some());
                        assert!(tokens < input.len());
                    }
                    let mut owned_saving = 0.0;
                    let owned = fast_stream::<1>(
                        input,
                        &values,
                        64,
                        &config,
                        &penalties,
                        &sums,
                        &mut owned_saving,
                    )
                    .unwrap();
                    assert_eq!(owned, commands.unwrap_or_else(|| literals(input).unwrap()));
                    assert_eq!(owned_saving, saving);
                }
            }
        }
    }
}

#[test]
fn fast_seed_wire_tokens_preserve_multistream_savings_gate_and_parallel_order() {
    let unique = input(0..32768);
    let repeated = input((0..64).cycle().take(32768));
    let inputs = [&unique[..], &[][..], &repeated[..]];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for parallel in [false, true] {
            let config = SourceConfig {
                method: SourceLz77Method::Lz77b1w3t,
                parallelism: parallel.then_some(SourceParallelism {
                    fast_streams: true,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let costs = Costs::raw(&inputs, &config);
            let prices = prices::Prices::new(&costs).unwrap();
            let penalties = prices::Penalties::new(&costs).unwrap();
            let mut expected = Vec::new();
            let (mut saving, mut raw) = (0.0, 0.0);
            for input in inputs {
                let values: Vec<_> = input.iter().map(|literal| literal.value).collect();
                let sums = prices.cumulative(input, &config).unwrap();
                let mut decrease = 0.0;
                let (commands, _) = fast_stream_borrowed::<1>(
                    input,
                    &values,
                    0,
                    &config,
                    &penalties,
                    &sums,
                    if parallel { &mut decrease } else { &mut saving },
                )
                .unwrap();
                if parallel {
                    saving += decrease;
                }
                raw += sums[input.len()];
                expected.push(commands.unwrap_or_else(|| literals(input).unwrap()));
            }
            assert_eq!(expected[0], literals(&unique).unwrap());
            assert!(expected[1].is_empty());
            assert!(
                expected[2]
                    .iter()
                    .any(|c| matches!(c, Command::Match { .. }))
            );
            let tokens: usize = expected.iter().map(|c| wire_token_count(c)).sum();
            let total: usize = inputs.iter().map(|input| input.len()).sum();
            let mut threshold = tokens as f32 * 0.2 + 16.0;
            if raw <= total as f32 * 2.5 {
                threshold = threshold.min(raw * 0.01 + 256.0);
            }
            for require_global_savings in [false, true] {
                let config = SourceConfig {
                    require_global_savings,
                    ..config.clone()
                };
                let enabled = !require_global_savings || saving > threshold;
                let mut work = Work::default();
                let outcome = run_borrowed(&inputs, &[], &config, &mut work).unwrap();
                assert_eq!(work.fast_tasks, if parallel { 2 } else { 1 });
                assert_eq!(outcome.enabled, enabled);
                if enabled {
                    assert_eq!(outcome.commands.unwrap(), expected);
                } else {
                    assert!(outcome.commands.is_none());
                }
                let public = parse_source_streams_with_state(&inputs, &[], &config).unwrap();
                assert_eq!(public.enabled, enabled);
                let fallback: Vec<_> = inputs
                    .iter()
                    .map(|input| literals(input).unwrap())
                    .collect();
                assert_eq!(
                    public.commands,
                    if enabled { expected.clone() } else { fallback }
                );
            }
        }
    });
}

#[test]
fn borrowed_empty_streams_and_validation_keep_source_contract() {
    for require_global_savings in [true, false] {
        let config = SourceConfig {
            require_global_savings,
            ..Default::default()
        };
        let none = run_borrowed(&[], &[], &config, &mut Work::default()).unwrap();
        assert!(none.commands.is_none());
        assert!(!none.enabled);
        let inputs = [&[][..], &[][..]];
        let empty = run_borrowed(&inputs, &[], &config, &mut Work::default()).unwrap();
        assert!(empty.commands.is_none());
        assert_eq!(empty.enabled, !require_global_savings);
        let owned = parse_source_streams_with_state(&inputs, &[], &config).unwrap();
        assert_eq!(owned.commands, vec![vec![], vec![]]);
        assert_eq!(owned.enabled, !require_global_savings);
    }
    let invalid = SourceConfig {
        max_match_length: MIN_MATCH - 1,
        ..Default::default()
    };
    assert!(SourceParseOutcome::parse_borrowed(&[], 0, &invalid).is_err());
    assert!(parse_source_with_state(&[], 0, &invalid).is_err());
}

#[test]
fn borrowed_optimal_literal_only_acceptance_does_not_skip_dp() {
    let input = input(0..257);
    let config = SourceConfig::new(SourceLz77Method::Optc8);
    assert_eq!(
        SourceParseOutcome::parse_borrowed(&input, 0, &config).unwrap(),
        (None, false)
    );
    assert_public_literals(&input, &config, false);
    let forced = SourceConfig {
        require_global_savings: false,
        ..config
    };
    let mut work = Work::default();
    let outcome = run_borrowed(&[&input], &[], &forced, &mut work).unwrap();
    assert!(outcome.enabled);
    assert_eq!(work.passes, 1);
    assert_eq!(outcome.commands.unwrap(), vec![literals(&input).unwrap()]);
    assert_public_literals(&input, &forced, true);
}
