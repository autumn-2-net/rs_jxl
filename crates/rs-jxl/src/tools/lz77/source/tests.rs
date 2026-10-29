// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;

fn tokens(values: &[u32]) -> Vec<Literal> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| Literal {
            context: if i % 7 == 0 { u32::MAX } else { (i % 3) as u32 },
            value,
        })
        .collect()
}

fn replay(input: &[Literal], commands: &[Command], width: usize) {
    let mut decoded = Vec::new();
    for &command in commands {
        let origin = decoded.len();
        assert!(origin < input.len());
        match command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[origin]);
                decoded.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[origin].context);
                assert!(length >= 3);
                let distance = distance as usize;
                assert!((1..=origin.min(WINDOW_SIZE)).contains(&distance));
                assert!(super::super::distance_symbol(distance, width).is_ok());
                assert!(origin + length as usize <= input.len());
                for _ in 0..length {
                    decoded.push(decoded[decoded.len() - distance]);
                }
            }
        }
    }
    assert_eq!(decoded, input.iter().map(|v| v.value).collect::<Vec<_>>());
}

#[test]
fn source_state_rejection_and_compatibility_wrappers() {
    for method in [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Optc8,
        SourceLz77Method::Optc256,
    ] {
        let config = SourceConfig::new(method);
        let flat = vec![
            Literal {
                context: 0,
                value: 0
            };
            128
        ];
        for input in [&[][..], &flat[..]] {
            let outcome = parse_source_with_state(input, 1, &config).unwrap();
            assert!(!outcome.enabled);
            assert_eq!(outcome.commands, literals(input).unwrap());
            assert_eq!(outcome.commands, parse_source(input, 1, &config).unwrap());
        }
        let input = tokens(&[65535; 512]);
        let outcome = parse_source_with_state(&input, 0, &config).unwrap();
        assert!(outcome.enabled);
        assert!(
            outcome
                .commands
                .iter()
                .any(|c| matches!(c, Command::Match { .. }))
        );
        assert_eq!(outcome.commands, parse_source(&input, 0, &config).unwrap());
        replay(&input, &outcome.commands, 0);
    }
}

#[test]
fn shared_state_applies_to_empty_and_literal_only_streams() {
    let a = vec![
        Literal {
            context: 0,
            value: 0
        };
        128
    ];
    let b = vec![
        Literal {
            context: 0,
            value: 1
        };
        128
    ];
    let tail = [Literal {
        context: 7,
        value: 5,
    }];
    for method in [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Optc8,
        SourceLz77Method::Optc256,
    ] {
        let config = SourceConfig::new(method);
        let inputs = [&a[..], &[][..], &b[..], &tail[..]];
        let outcome = parse_source_streams_with_state(&inputs, &[1], &config).unwrap();
        assert!(outcome.enabled);
        assert!(outcome.commands[1].is_empty());
        assert_eq!(outcome.commands[3], literals(&tail).unwrap());
        assert_eq!(
            outcome.commands,
            parse_source_streams(&inputs, &[1], &config).unwrap()
        );
        for (i, input) in inputs.iter().enumerate() {
            replay(input, &outcome.commands[i], usize::from(i == 0));
        }
    }
}

#[test]
fn accepted_optimal_literal_only_output_does_not_clear_state() {
    let input = tokens(&[0, 1, 2, 3, 4, 5, 6]);
    for method in [SourceLz77Method::Optc8, SourceLz77Method::Optc256] {
        for match_cost in [
            SourceMatchCost::SymbolId,
            SourceMatchCost::Refit,
            SourceMatchCost::RangeReplay,
            SourceMatchCost::Range,
            SourceMatchCost::RangeRefit,
            SourceMatchCost::Auto,
        ] {
            // Force acceptance at the existing diagnostic seed gate, then
            // execute real DP/refit. There are no repeatable triples.
            let config = SourceConfig {
                method,
                match_cost,
                require_global_savings: false,
                ..SourceConfig::default()
            };
            let mut work = Work::default();
            let outcome = run(&[&input], &[0], &config, &mut work).unwrap();
            assert!(outcome.enabled);
            assert_eq!(work.passes, match_cost.passes());
            assert_eq!(outcome.commands, vec![literals(&input).unwrap()]);
            assert!(
                !parse_source_streams_with_state(&[], &[], &config)
                    .unwrap()
                    .enabled
            );
        }
    }
}

#[test]
fn source_rle_pools_float_costs_and_preserves_histories_and_contexts() {
    let a = vec![
        Literal {
            context: 0,
            value: 0
        };
        128
    ];
    let b: Vec<_> = (0..128)
        .map(|i| Literal {
            context: if i == 1 { 9 } else { 0 },
            value: 1,
        })
        .collect();
    let config = SourceConfig::default();
    assert!(!parse_source_rle_with_state(&a, 0, &config).unwrap().enabled);
    assert!(!parse_source_rle_with_state(&b, 0, &config).unwrap().enabled);
    let inputs = [&a[..], &[][..], &b[..]];
    let outcome = parse_source_rle_streams_with_state(&inputs, &[], &config).unwrap();
    assert!(outcome.enabled);
    assert_eq!(
        outcome.commands[0],
        vec![
            Command::Literal(a[0]),
            Command::Match {
                context: 0,
                length: 127,
                distance: 1
            }
        ]
    );
    assert_eq!(
        outcome.commands[2],
        vec![
            Command::Literal(b[0]),
            Command::Match {
                context: 9,
                length: 127,
                distance: 1
            }
        ]
    );
    for (input, commands) in inputs.into_iter().zip(&outcome.commands) {
        replay(input, commands, 0);
    }
    // RLE uses default420, NOT params.UintConfig (also for validation).
    let different_literal_config = SourceConfig {
        literal_uint_config: HybridUintConfig::new(15, 0, 0).unwrap(),
        ..config.clone()
    };
    let high = tokens(&[65535; 128]);
    assert_eq!(
        parse_source_rle_with_state(&high, 0, &config).unwrap(),
        parse_source_rle_with_state(&high, 0, &different_literal_config).unwrap()
    );
}

#[test]
fn source_rle_minimum_tails_caps_and_strict_gate() {
    for force_huffman in [false, true] {
        let config = SourceConfig {
            force_huffman,
            ..SourceConfig::default()
        };
        for size in 0..=3 {
            let input = tokens(&vec![u32::MAX; size]);
            let outcome = parse_source_rle_with_state(&input, 0, &config).unwrap();
            assert!(!outcome.enabled);
            assert_eq!(outcome.commands, literals(&input).unwrap());
        }
        let input = tokens(&[u32::MAX; 4]);
        let outcome = parse_source_rle_with_state(&input, 0, &config).unwrap();
        assert!(outcome.enabled);
        assert!(matches!(
            outcome.commands[1],
            Command::Match {
                length: 3,
                distance: 1,
                ..
            }
        ));
        let input = tokens(&[u32::MAX; 17]);
        let capped = SourceConfig {
            max_match_length: 5,
            ..config.clone()
        };
        let outcome = parse_source_rle_with_state(&input, 1, &capped).unwrap();
        assert!(outcome.enabled);
        assert!(
            outcome
                .commands
                .iter()
                .all(|c| !matches!(c, Command::Match { length, .. } if *length > 5))
        );
        assert!(matches!(
            outcome.commands.last().unwrap(),
            Command::Literal(_)
        ));
        replay(&input, &outcome.commands, 1);
    }
    let config = SourceConfig::default();
    // uint420 encodes 16 with two extra bits. Repeated zeros cost zero.
    // Three copies save exactly 3*2-1=5 bits: below the shared gate.
    let input = vec![
        Literal {
            context: 0,
            value: 16
        };
        4
    ];
    let rejected = parse_source_rle_with_state(&input, 0, &config).unwrap();
    assert!(!rejected.enabled);
    assert_eq!(rejected.commands, literals(&input).unwrap());
    let forced = SourceConfig {
        require_global_savings: false,
        ..config
    };
    assert!(
        parse_source_rle_with_state(&input, 0, &forced)
            .unwrap()
            .enabled
    );
    assert!(
        !parse_source_rle_streams_with_state(&[], &[], &forced)
            .unwrap()
            .enabled
    );
}

#[test]
fn source_rle_shared_gate_rejects_exact_equality() {
    let run = vec![
        Literal {
            context: 0,
            value: 16
        };
        40
    ];
    let padding = [Literal {
        context: 1,
        value: 0,
    }];
    let config = SourceConfig::default();
    // Constant uint420 symbol 16 costs two extra bits. Its 39 copies
    // save 39*2-(ceil(log2(39-3+1))+1) = 71 bits exactly. Independent
    // single-token zero streams add no entropy, matches or savings.
    for (extra, accepted) in [(234, true), (235, false), (236, false)] {
        let mut inputs = vec![&run[..]];
        inputs.extend(std::iter::repeat_n(&padding[..], extra));
        let outcome = parse_source_rle_streams_with_state(&inputs, &[], &config).unwrap();
        assert_eq!(outcome.enabled, accepted, "input total={}", 40 + extra);
        assert_eq!(outcome.commands.len(), inputs.len());
        for (input, commands) in inputs.into_iter().zip(&outcome.commands) {
            replay(input, commands, 0);
        }
    }
}

#[test]
fn every_source_route_replays_u32_residuals_sparse_contexts_and_tails() {
    let mut values: Vec<_> = [0, 255, 65535, 131070, u32::MAX, 7, 19, 3]
        .into_iter()
        .cycle()
        .take(515)
        .collect();
    values[211] = 65_536;
    let input = tokens(&values);
    for method in [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Optc8,
        SourceLz77Method::Optc256,
    ] {
        for match_cost in [
            SourceMatchCost::SymbolId,
            SourceMatchCost::Entropy,
            SourceMatchCost::Frontier,
            SourceMatchCost::Refit,
        ] {
            for width in [0, 1, 2, 37] {
                let config = SourceConfig {
                    method,
                    match_cost,
                    bucket_matcher: true,
                    prefix_filter: FilterPolicy::Eager,
                    require_global_savings: false,
                    ..SourceConfig::default()
                };
                let commands = parse_source(&input, width, &config).unwrap();
                assert!(commands.iter().any(|c| matches!(c, Command::Match { .. })));
                replay(&input, &commands, width);
            }
        }
    }
}

#[test]
fn empty_and_incomplete_prefix_inputs_are_valid() {
    for method in [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Optc8,
        SourceLz77Method::Optc256,
    ] {
        let config = SourceConfig {
            method,
            bucket_matcher: true,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        for length in 0..6 {
            let input = tokens(&vec![u32::MAX; length]);
            replay(
                &input,
                &parse_source(&input, usize::MAX, &config).unwrap(),
                usize::MAX,
            );
        }
        assert_eq!(
            parse_source_streams(&[], &[], &config).unwrap(),
            Vec::<Vec<Command>>::new()
        );
    }
}

#[test]
fn shared_histograms_and_gate_do_not_share_match_history() {
    let a = vec![
        Literal {
            context: 9,
            value: 0
        };
        128
    ];
    let b = vec![
        Literal {
            context: 9,
            value: 1
        };
        128
    ];
    let config = SourceConfig::default();
    assert_eq!(parse_source(&a, 1, &config).unwrap(), literals(&a).unwrap());
    assert_eq!(parse_source(&b, 0, &config).unwrap(), literals(&b).unwrap());
    let streams = parse_source_streams(&[&a, &[], &b], &[1], &config).unwrap();
    assert!(
        streams[0]
            .iter()
            .any(|c| matches!(c, Command::Match { .. }))
    );
    assert!(
        streams[2]
            .iter()
            .any(|c| matches!(c, Command::Match { .. }))
    );
    replay(&a, &streams[0], 1);
    replay(&[], &streams[1], 0);
    replay(&b, &streams[2], 0);
    assert!(matches!(streams[2][0], Command::Literal(_)));
}

#[test]
fn combined_token_threshold_and_careful_override_select_the_seed() {
    let input = vec![
        Literal {
            context: 0,
            value: 0
        };
        65_536
    ];
    for (size, forced, expected) in [
        (65_535, false, true),
        (65_536, false, false),
        (65_536, true, true),
    ] {
        let config = SourceConfig {
            method: SourceLz77Method::Optc8,
            careful_first_pass: forced,
            match_cost: SourceMatchCost::Refit,
            ..SourceConfig::default()
        };
        let mut work = Work::default();
        let streams = [&input[..size / 2], &input[size / 2..size]];
        let output = run(&streams, &[], &config, &mut work).unwrap();
        assert_eq!(work.careful_seed, expected);
        // A zero-entropy seed is rejected before either DP/refit pass.
        assert_eq!(work.passes, 0);
        assert!(!output.enabled);
        for (input, commands) in streams.into_iter().zip(&output.commands) {
            replay(input, commands, 0);
        }
    }
}

#[test]
fn automatic_range_selection_preserves_all_integer_boundaries() {
    assert!(!prefer_range(65_535, 1, 65_535, 65_535));
    assert!(!prefer_range(65_536, 0, 65_536, 65_536));
    assert!(!prefer_range(65_536, 1025, 32_768, 16_384));
    assert!(!prefer_range(65_536, 1024, 32_767, 16_384));
    assert!(!prefer_range(65_536, 1024, 32_768, 16_383));
    assert!(prefer_range(65_536, 1024, 32_768, 16_384));
    assert!(!prefer_range(65_537, 1024, 32_768, 16_384));
    assert!(prefer_range(65_537, 1024, 32_769, 16_384));
    let input = tokens(&(0..65_536).map(|i| 65535 + (i % 17)).collect::<Vec<_>>());
    for (total, expected) in [
        (65535, SourceMatchCost::SymbolId),
        (65536, SourceMatchCost::Range),
    ] {
        let inputs = [&input[..total / 2], &input[total / 2..total]];
        let config = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_mode: SourceBucketMode::History,
            match_cost: SourceMatchCost::Auto,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let mut work = Work::default();
        let actual = run(&inputs, &[0, 0], &config, &mut work).unwrap();
        assert_eq!(work.selected_cost, Some(expected));
        assert_eq!(work.passes, 1);
        let forced = SourceConfig {
            match_cost: expected,
            ..config
        };
        assert_eq!(
            actual,
            run(&inputs, &[0, 0], &forced, &mut Work::default()).unwrap()
        );
        for (input, output) in inputs.iter().zip(&actual.commands) {
            replay(input, output, 0);
        }
    }
}

#[test]
fn mode_four_full_replay_avoids_second_pass_queries_and_preserves_acceptance() {
    let input = tokens(&vec![65535; 4096]);
    let config = SourceConfig {
        method: SourceLz77Method::Optc256,
        match_cost: SourceMatchCost::RangeReplay,
        bucket_mode: SourceBucketMode::History,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    let mut replay_work = Work::default();
    let actual = run(&[&input, &[]], &[37], &config, &mut replay_work).unwrap();
    let mut first_work = Work::default();
    run(
        &[&input],
        &[37],
        &SourceConfig {
            match_cost: SourceMatchCost::Range,
            ..config.clone()
        },
        &mut first_work,
    )
    .unwrap();
    assert_eq!(replay_work.passes, 2);
    assert_eq!(replay_work.full_replay_streams, 1);
    assert_eq!(replay_work.partial_replay_streams, 0);
    assert_eq!(replay_work.queries, first_work.queries);
    assert!(replay_work.replayed_positions > 0);
    assert!(replay_work.replay_bytes > 0 && replay_work.replay_bytes <= replay::CACHE_LIMIT);
    assert_eq!(replay_work.optimal_tasks, 1);
    assert!(actual.enabled && actual.commands[1].is_empty());
    replay(&input, &actual.commands[0], 37);
    let empty_matches = tokens(&[0, 1, 2, 3, 4]);
    let mut work = Work::default();
    let outcome = run(&[&empty_matches], &[], &config, &mut work).unwrap();
    assert!(outcome.enabled);
    assert_eq!(work.full_replay_streams, 1);
    assert_eq!(work.replay_bytes, 0);
    assert_eq!(outcome.commands, vec![literals(&empty_matches).unwrap()]);
}

#[test]
fn replay_zero_budget_equals_fresh_refit_and_partial_tail_rebuilds_chain_history() {
    let mut values: Vec<_> = (0..97).map(|i| 65535 + (i * 19) % 31).collect();
    values.extend_from_within(0..83);
    values.extend([65535; 521]);
    values.extend_from_within(13..167);
    let input = tokens(&values);
    let inputs = [&input[..], &input[7..]];
    let data: Vec<Vec<u32>> = inputs
        .iter()
        .map(|s| s.iter().map(|t| t.value).collect())
        .collect();
    let config = SourceConfig {
        method: SourceLz77Method::Optc256,
        match_cost: SourceMatchCost::RangeReplay,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    let initial: Vec<_> = inputs.iter().map(|s| literals(s).unwrap()).collect();
    let mut expected = initial.clone();
    for _ in 0..2 {
        let costs = Costs::fitted(&expected, &[37, 0], &config);
        expected = inputs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                optimal_stream(
                    s,
                    &data[i],
                    [37, 0][i],
                    &config,
                    &costs,
                    0,
                    &mut Work::default(),
                )
                .unwrap()
            })
            .collect();
    }
    let mut work = Work::default();
    let zero = range_replayed(
        &inputs,
        &data,
        &[37, 0],
        &config,
        initial.clone(),
        0,
        &mut work,
    )
    .unwrap();
    assert_eq!(zero.commands, expected);
    assert_eq!(work.replay_bytes, 0);
    assert_eq!(work.full_replay_streams, 0);
    assert_eq!(work.partial_replay_streams, 2);
    for limit in [24, 80, 256] {
        let mut partial = Work::default();
        let output = range_replayed(
            &inputs,
            &data,
            &[37, 0],
            &config,
            initial.clone(),
            limit,
            &mut partial,
        )
        .unwrap();
        assert!(partial.replay_bytes > 0 && partial.replay_bytes <= limit);
        assert!(partial.partial_replay_streams > 0);
        assert!(partial.replayed_positions > 0);
        for (i, commands) in output.commands.iter().enumerate() {
            replay(inputs[i], commands, [37, 0][i]);
        }
    }
}

#[test]
fn range_refit_uses_fresh_chains_and_never_the_requested_history_bucket() {
    let values = vec![65535; 4096];
    let input = tokens(&values);
    let config = SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_mode: SourceBucketMode::WideKey,
        match_cost: SourceMatchCost::RangeRefit,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    assert!(matches!(
        Optimal::new(&values, 0, &config, 2).unwrap(),
        Optimal::Chain(_)
    ));
    let mut work = Work::default();
    let output = run(&[&input], &[], &config, &mut work).unwrap();
    assert_eq!(work.passes, 2);
    assert_eq!(work.cached_streams, 0);
    assert!(work.queries < 80);
    replay(&input, &output.commands[0], 0);
}

#[test]
fn stream_parallelism_runs_real_parsers_and_retains_shared_acceptance() {
    let input = tokens(&(0..32768).map(|i| 65535 + (i % 19)).collect::<Vec<_>>());
    let inputs = [&input[..], &[], &input[..]];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for (bucket_mode, careful_first_pass) in [
        (SourceBucketMode::Default, false),
        (SourceBucketMode::WideRing, true),
    ] {
        let serial = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_mode,
            careful_first_pass,
            ..SourceConfig::default()
        };
        let parallel = SourceConfig {
            parallelism: Some(SourceParallelism {
                fast_streams: true,
                ..SourceParallelism::default()
            }),
            ..serial.clone()
        };
        let expected = run(&inputs, &[0, 0, 37], &serial, &mut Work::default()).unwrap();
        let mut work = Work::default();
        let actual = pool
            .install(|| run(&inputs, &[0, 0, 37], &parallel, &mut work))
            .unwrap();
        assert_eq!(actual, expected);
        assert!(actual.enabled);
        assert_eq!(work.fast_tasks, if careful_first_pass { 1 } else { 2 });
        assert_eq!(work.optimal_tasks, 2);
        let mut serial_work = Work::default();
        let forced = pool
            .install(|| {
                crate::execution::serial(|| run(&inputs, &[0, 0, 37], &parallel, &mut serial_work))
            })
            .unwrap();
        assert_eq!(forced, expected);
        assert_eq!((serial_work.fast_tasks, serial_work.optimal_tasks), (1, 1));
    }
}

#[test]
fn run_key_and_deep_options_reach_the_indexed_parser_and_replay_pixels() {
    let mut pattern: Vec<_> = (0..211).map(|i| u32::MAX - (i * 19) % 37).collect();
    pattern.extend([0; 123]);
    pattern.extend([65535, 65536, u32::MAX]);
    let input = tokens(&pattern.into_iter().cycle().take(8192).collect::<Vec<_>>());
    for (run_keys, deep) in [
        (SourceRunKeys::LengthAndSuffix, SourceDeepPrefix::Disabled),
        (SourceRunKeys::SharedSuffix, SourceDeepPrefix::Disabled),
        (SourceRunKeys::Disabled, SourceDeepPrefix::Four),
        (SourceRunKeys::SharedSuffix, SourceDeepPrefix::Four),
        (SourceRunKeys::SharedSuffix, SourceDeepPrefix::Eight),
    ] {
        let config = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_mode: SourceBucketMode::WideRing,
            match_cost: SourceMatchCost::Frontier,
            index_mode: SourceIndexMode::All,
            small_index: true,
            index_run_keys: run_keys,
            index_deep_prefix: deep,
            prefix_filter: FilterPolicy::Eager,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let mut work = Work::default();
        let output = run(&[&input], &[37], &config, &mut work).unwrap();
        assert_eq!(work.indexed_streams, 1);
        assert!(
            output.commands[0]
                .iter()
                .any(|c| matches!(c, Command::Match { .. }))
        );
        replay(&input, &output.commands[0], 37);
    }
}

#[test]
fn indexed_parser_dispatch_equals_mutable_buckets_and_rejects_ineligible_modes() {
    let input = tokens(&(0..8192).map(|i| 65535 + (i % 23)).collect::<Vec<_>>());
    let config = SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_matcher: true,
        match_cost: SourceMatchCost::Frontier,
        small_index: true,
        index_mode: SourceIndexMode::All,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    for bucket_mode in [SourceBucketMode::Default, SourceBucketMode::WideRing] {
        let config = SourceConfig {
            bucket_mode,
            ..config.clone()
        };
        let mut work = Work::default();
        let actual = run(&[&input], &[37], &config, &mut work).unwrap();
        assert_eq!(work.indexed_streams, 1);
        let mutable = SourceConfig {
            index_mode: SourceIndexMode::Disabled,
            ..config
        };
        let expected = run(&[&input], &[37], &mutable, &mut Work::default()).unwrap();
        assert_eq!(actual, expected);
        replay(&input, &actual.commands[0], 37);
    }
    assert!(!index::enabled(&[&input[..8191]], &config));
    assert!(index::enabled(&[&input, &input], &config));
    assert!(!index::enabled(
        &[&input, &input],
        &SourceConfig {
            index_mode: SourceIndexMode::Dominant,
            ..config.clone()
        }
    ));
    for ineligible in [
        SourceConfig {
            method: SourceLz77Method::Optc8,
            ..config.clone()
        },
        SourceConfig {
            bucket_mode: SourceBucketMode::History,
            ..config.clone()
        },
        SourceConfig {
            match_cost: SourceMatchCost::Refit,
            ..config.clone()
        },
        SourceConfig {
            small_index: false,
            ..config.clone()
        },
    ] {
        assert!(!index::enabled(&[&input], &ineligible));
    }
}

#[test]
fn refit_is_one_extra_pass_and_long_runs_use_bounded_intervals() {
    let input = vec![
        Literal {
            context: u32::MAX,
            value: 65535
        };
        32768
    ];
    for bucket_matcher in [false, true] {
        let config = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_matcher,
            match_cost: SourceMatchCost::Refit,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let mut work = Work::default();
        let output = run(&[&input], &[0], &config, &mut work).unwrap();
        assert_eq!(work.passes, 2);
        assert_eq!(work.cached_streams, if bucket_matcher { 0 } else { 2 });
        assert!(work.queries < 80, "queries={}", work.queries);
        assert!(work.intervals < 4096, "intervals={}", work.intervals);
        assert!(output.enabled);
        assert!(
            output.commands[0].iter().any(
                |c| matches!(c, Command::Match { length, distance: 1, .. } if *length > 32000)
            )
        );
        replay(&input, &output.commands[0], 0);
    }
}

#[test]
fn source_cost_configs_and_distance_histogram_query_distinction() {
    for fused_cost_math in [false, true] {
        let config = SourceConfig {
            fused_cost_math,
            ..SourceConfig::default()
        };
        for value in [1.0f32, 0.5, 0.25, 0.125, 1.0 / 65536.0] {
            assert!((fast_log2(value, fused_cost_math) - value.log2()).abs() < 1e-5);
        }
        let row = context_cost([(7, 65_537)].into_iter().collect(), &config);
        assert_eq!(row.bits[&7], 0.0);
        assert_eq!(row.penalty, 6.0);
    }
    let config = SourceConfig {
        literal_uint_config: HybridUintConfig::new(0, 0, 0).unwrap(),
        length_uint_config: HybridUintConfig::new(4, 0, 0).unwrap(),
        force_huffman: true,
        ..SourceConfig::default()
    };
    assert_eq!(config.min_symbol(), 512);
    let seed = vec![vec![Command::Match {
        context: u32::MAX,
        length: 515,
        distance: 32,
    }]];
    let costs = Costs::fitted(&seed, &[0], &config);
    let length = config.length_uint_config.encode(512);
    assert_eq!(costs.bits(u32::MAX, 512 + length.token), 0.0);
    let hist = config.literal_uint_config.encode(31);
    let query = HybridUintConfig::default().encode(31);
    assert_ne!(hist.token, query.token);
    assert_eq!(costs.distances.bits[&hist.token], 0.0);
    assert_eq!(costs.distance(31), 12.0 + query.nbits as f32);
}

#[test]
fn invalid_alphabets_and_length_configs_fail_before_matching() {
    let input = tokens(&[u32::MAX]);
    for config in [
        SourceConfig {
            max_match_length: 2,
            ..SourceConfig::default()
        },
        SourceConfig {
            min_symbol: Some(u32::MAX),
            ..SourceConfig::default()
        },
        SourceConfig {
            length_uint_config: HybridUintConfig::new(2, 0, 1).unwrap(),
            ..SourceConfig::default()
        },
        SourceConfig {
            literal_uint_config: HybridUintConfig::new(15, 0, 0).unwrap(),
            ..SourceConfig::default()
        },
        SourceConfig {
            parallelism: Some(SourceParallelism {
                max_workers: 0,
                ..SourceParallelism::default()
            }),
            ..SourceConfig::default()
        },
        SourceConfig {
            parallelism: Some(SourceParallelism {
                scratch_bytes: 0,
                ..SourceParallelism::default()
            }),
            ..SourceConfig::default()
        },
    ] {
        assert!(parse_source(&input, 0, &config).is_err());
    }
    replay(
        &input,
        &parse_source(&input, 0, &SourceConfig::default()).unwrap(),
        0,
    );
}

#[test]
fn precise_ring_and_short128_equal_scalar_endpoint_oracle() {
    let mut values: Vec<_> = (0..129).map(|i| 65535 + i * 19 % 37).collect();
    values.extend_from_within(13..109);
    values.extend_from_within(0..99);
    values.extend([0; 101]);
    let input = tokens(&values);
    for bucket_mode in [SourceBucketMode::PreciseRing, SourceBucketMode::Short128] {
        for match_cost in [
            SourceMatchCost::SymbolId,
            SourceMatchCost::Entropy,
            SourceMatchCost::Frontier,
            SourceMatchCost::Refit,
        ] {
            let config = SourceConfig {
                method: SourceLz77Method::Optc256,
                bucket_mode,
                match_cost,
                require_global_savings: false,
                ..SourceConfig::default()
            };
            let raw = Costs::raw(&[&input], &config);
            let sums = raw.cumulative(&input, &config).unwrap();
            let seed = careful_stream(&input, &values, 37, &config, &raw, &sums, &mut 0.0).unwrap();
            let seed_tokens = wire_token_count(&seed);
            let costs = Costs::fitted(&[seed], &[37], &config);
            let actual = optimal_stream(
                &input,
                &values,
                37,
                &config,
                &costs,
                seed_tokens,
                &mut Work::default(),
            )
            .unwrap();
            assert_eq!(
                actual,
                scalar_pass(&input, &values, 37, &config, &costs),
                "{bucket_mode:?} {match_cost:?}"
            );
            replay(&input, &actual, 37);
        }
    }
}

#[test]
fn precise_cumulative_promotes_only_after_source_float_literal_price() {
    let input = tokens(&vec![1; 20001]);
    let costs = Costs {
        literals: BTreeMap::from([(
            0,
            ContextCost {
                bits: BTreeMap::from([(1, 0.1f32)]),
                penalty: 0.0,
            },
        )]),
        distances: ContextCost {
            bits: BTreeMap::new(),
            penalty: 0.0,
        },
    };
    let precise = SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_mode: SourceBucketMode::PreciseRing,
        ..SourceConfig::default()
    };
    let sums = costs.optimal_cumulative(&input, &precise).unwrap();
    let mut expected = 0.0;
    for token in &input {
        let encoded = precise.literal_uint_config.encode(token.value);
        expected += f64::from(costs.bits(token.context, encoded.token) + encoded.nbits as f32);
    }
    assert_eq!(sums.last().copied(), Some(expected));
    let normal = SourceConfig::default();
    let floats = costs.cumulative(&input, &normal).unwrap();
    let ordinary = costs.optimal_cumulative(&input, &normal).unwrap();
    assert_eq!(
        ordinary,
        floats.iter().copied().map(f64::from).collect::<Vec<_>>()
    );
    assert_ne!(expected, f64::from(*floats.last().unwrap()));
}

#[test]
fn block_pruning_controls_preserve_unpruned_static_suffix_and_nonblock_results() {
    assert!(!SourceConfig::default().block_zero_tail);
    assert!(!SourceConfig::default().block_literal_bound);
    let input: Vec<_> = (0..8192)
        .map(|p| Literal {
            value: if p < 6000 { 65536 + p % 37 } else { 7 },
            context: if p < 6000 { 1 + p % 3 } else { 0 },
        })
        .collect();
    for suffix_mode in [SourceSuffixMode::Disabled, SourceSuffixMode::SlidingLcp] {
        for block_mode in [SourceBlockMode::Endpoints, SourceBlockMode::Intervals] {
            let mut baseline = None;
            for flags in 0..4 {
                let config = SourceConfig {
                    method: SourceLz77Method::Optc256,
                    bucket_matcher: true,
                    index_mode: SourceIndexMode::All,
                    small_index: true,
                    match_cost: SourceMatchCost::Frontier,
                    block_mode,
                    suffix_mode,
                    block_zero_tail: flags & 1 != 0,
                    block_literal_bound: flags & 2 != 0,
                    index_word_match: true,
                    block_sparse_cache: true,
                    block_shared_frontier: true,
                    block_frontier_carry: true,
                    block_distance_bound: true,
                    block_distance_table: true,
                    block_distance_classes: true,
                    max_match_length: 257,
                    require_global_savings: false,
                    ..SourceConfig::default()
                };
                let mut work = Work::default();
                let result = run(&[&input, &[]], &[37], &config, &mut work).unwrap();
                assert!(result.enabled);
                assert_eq!(work.block_streams, 1);
                assert_eq!(
                    work.suffix_streams,
                    usize::from(suffix_mode != SourceSuffixMode::Disabled)
                );
                assert!(result.commands[1].is_empty());
                replay(&input, &result.commands[0], 37);
                if let Some(expected) = &baseline {
                    assert_eq!(&result, expected);
                } else {
                    baseline = Some(result);
                }
            }
        }
    }
    let config = SourceConfig {
        require_global_savings: false,
        ..SourceConfig::default()
    };
    let enabled = SourceConfig {
        block_zero_tail: true,
        block_literal_bound: true,
        ..config.clone()
    };
    assert_eq!(
        parse_source_with_state(&input[..511], 37, &config).unwrap(),
        parse_source_with_state(&input[..511], 37, &enabled).unwrap()
    );
}

#[test]
fn guided_lazy_options_dispatch_replay_and_leave_nonblock_routes_unchanged() {
    assert_eq!(
        SourceConfig::default().block_seed_guided,
        SourceSeedGuided::Disabled
    );
    assert!(!SourceConfig::default().block_lazy);
    let values: Vec<_> = (0..8192).map(|i| 65536 + i % 37).collect();
    let input = tokens(&values);
    for suffix_mode in [SourceSuffixMode::Disabled, SourceSuffixMode::SlidingLcp] {
        for block_seed_guided in [
            SourceSeedGuided::Disabled,
            SourceSeedGuided::Hints,
            SourceSeedGuided::SparseOrigins,
        ] {
            for block_lazy in [false, true] {
                let config = SourceConfig {
                    method: SourceLz77Method::Optc256,
                    bucket_matcher: true,
                    index_mode: SourceIndexMode::All,
                    small_index: true,
                    match_cost: SourceMatchCost::Frontier,
                    block_mode: SourceBlockMode::Intervals,
                    suffix_mode,
                    block_seed_guided,
                    block_lazy,
                    index_word_match: true,
                    block_sparse_cache: true,
                    block_shared_frontier: true,
                    block_frontier_carry: true,
                    block_distance_bound: true,
                    block_distance_table: true,
                    block_distance_classes: true,
                    max_match_length: 257,
                    require_global_savings: false,
                    ..SourceConfig::default()
                };
                let mut work = Work::default();
                let result = run(&[&input, &[]], &[37], &config, &mut work).unwrap();
                assert!(result.enabled);
                assert_eq!(work.block_streams, 1);
                assert_eq!(
                    work.suffix_streams,
                    usize::from(suffix_mode != SourceSuffixMode::Disabled)
                );
                assert_eq!(result.commands.len(), 2);
                assert!(result.commands[1].is_empty());
                replay(&input, &result.commands[0], 37);
            }
        }
    }
    let baseline = SourceConfig {
        require_global_savings: false,
        ..SourceConfig::default()
    };
    let enabled = SourceConfig {
        block_seed_guided: SourceSeedGuided::SparseOrigins,
        block_lazy: true,
        ..baseline.clone()
    };
    assert_eq!(
        parse_source_with_state(&input[..511], 37, &baseline).unwrap(),
        parse_source_with_state(&input[..511], 37, &enabled).unwrap()
    );
}

#[test]
fn optimized_block_options_reach_static_and_suffix_dispatch_and_replay() {
    let values: Vec<_> = (0..8192).map(|i| 65536 + i % 37).collect();
    let input = tokens(&values);
    for suffix_mode in [SourceSuffixMode::Disabled, SourceSuffixMode::SlidingLcp] {
        let config = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_matcher: true,
            index_mode: SourceIndexMode::All,
            small_index: true,
            match_cost: SourceMatchCost::Frontier,
            block_mode: SourceBlockMode::Intervals,
            suffix_mode,
            index_word_match: true,
            block_sparse_cache: true,
            block_shared_frontier: true,
            block_frontier_carry: true,
            block_distance_bound: true,
            block_distance_table: true,
            block_distance_classes: true,
            max_match_length: 257,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let mut work = Work::default();
        let result = run(&[&input], &[37], &config, &mut work).unwrap();
        assert!(result.enabled);
        assert_eq!(work.block_streams, 1);
        assert_eq!(
            work.suffix_streams,
            usize::from(suffix_mode != SourceSuffixMode::Disabled)
        );
        replay(&input, &result.commands[0], 37);
    }
}

#[test]
fn suffix_dispatch_replays_and_budget_fallback_keeps_block_contract() {
    let values: Vec<_> = (0..8192).map(|i| 65536 + (i * 19 % 37)).collect();
    let input = tokens(&values);
    let base = SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_matcher: true,
        index_mode: SourceIndexMode::All,
        small_index: true,
        match_cost: SourceMatchCost::Frontier,
        block_mode: SourceBlockMode::Endpoints,
        index_tags: SourceIndexTags::Simd,
        max_match_length: 257,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    for suffix_mode in [
        SourceSuffixMode::Adjacent,
        SourceSuffixMode::Sliding,
        SourceSuffixMode::SlidingLcp,
        SourceSuffixMode::AdjacentLcp,
    ] {
        let config = SourceConfig {
            suffix_mode,
            ..base.clone()
        };
        let mut work = Work::default();
        let result = run(&[&input], &[37], &config, &mut work).unwrap();
        assert!(result.enabled);
        assert_eq!((work.block_streams, work.suffix_streams), (1, 1));
        replay(&input, &result.commands[0], 37);
    }
    let fallback = SourceConfig {
        suffix_mode: SourceSuffixMode::SlidingLcp,
        suffix_scratch_bytes: 0,
        ..base.clone()
    };
    let mut work = Work::default();
    let result = run(&[&input], &[37], &fallback, &mut work).unwrap();
    assert_eq!((work.block_streams, work.suffix_streams), (1, 0));
    assert_eq!(
        result.commands,
        parse_source_streams(&[&input], &[37], &base).unwrap()
    );
    replay(&input, &result.commands[0], 37);
}

#[test]
fn shared_index_seed_reuses_one_build_without_changing_outer_acceptance() {
    let values: Vec<_> = (0..65536).map(|i| 65536 + (i * 19 % 37)).collect();
    let input = tokens(&values);
    let config = SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_matcher: true,
        index_mode: SourceIndexMode::All,
        block_mode: SourceBlockMode::Intervals,
        shared_index: true,
        max_match_length: 1024,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    let mut work = Work::default();
    let result = run(&[&input, &[]], &[37], &config, &mut work).unwrap();
    assert!(result.enabled);
    assert_eq!(result.commands.len(), 2);
    assert!(result.commands[1].is_empty());
    assert_eq!((work.shared_index_builds, work.block_streams), (1, 1));
    replay(&input, &result.commands[0], 37);
    let careful = SourceConfig {
        careful_first_pass: true,
        ..config
    };
    let mut work = Work::default();
    let result = run(&[&input], &[37], &careful, &mut work).unwrap();
    assert_eq!(work.shared_index_builds, 0);
    replay(&input, &result.commands[0], 37);
}

// Deliberately scalar endpoint relaxation, independent of Frontier and
// RangeParser. This is only a small-input oracle for source DP ordering.
fn scalar_pass(
    input: &[Literal],
    values: &[u32],
    width: usize,
    config: &SourceConfig,
    costs: &Costs,
) -> Vec<Command> {
    // Keep this endpoint-DP oracle uncached, independent of the seed gate.
    let mut matcher = Optimal::new(values, width, config, 0).unwrap();
    let mut sums = vec![0.0f64];
    for token in input {
        let encoded = config.literal_uint_config.encode(token.value);
        let price = costs.bits(token.context, encoded.token) + encoded.nbits as f32;
        sums.push(if config.precise() {
            sums.last().unwrap() + f64::from(price)
        } else {
            f64::from(*sums.last().unwrap() as f32 + price)
        });
    }
    let mut steps = vec![(f64::INFINITY, 0usize, 0usize); input.len() + 1];
    steps[0].0 = 0.0;
    let (mut skip, mut repeats, mut previous_end, mut previous_symbol) = (0, 0, 0, 0);
    for position in 0..input.len() {
        matcher.update(position);
        let base = steps[position].0;
        let literal = if config.match_cost.range() {
            let encoded = config.literal_uint_config.encode(input[position].value);
            f64::from(
                base as f32
                    + (costs.bits(input[position].context, encoded.token) + encoded.nbits as f32),
            )
        } else if config.precise() {
            base + sums[position + 1] - sums[position]
        } else {
            f64::from(base as f32 + sums[position + 1] as f32 - sums[position] as f32)
        };
        if literal < steps[position + 1].0 {
            steps[position + 1] = (literal, 1, 0);
        }
        if skip > 0 {
            skip -= 1;
            continue;
        }
        let mut matches = Vec::new();
        matcher.find_matches(position, config.match_cost.keep_short(), |m| {
            matches.push(m)
        });
        if config.match_cost.range() {
            // Independent class reduction, preserving the source's
            // longest match and then smallest symbol within each class.
            let mut classes = BTreeMap::<u32, Match>::new();
            for matched in matches {
                let class = HybridUintConfig::default().encode(matched.symbol).token;
                let kept = classes.entry(class).or_insert(matched);
                if matched.length > kept.length
                    || (matched.length == kept.length && matched.symbol < kept.symbol)
                {
                    *kept = matched;
                }
            }
            matches = classes.into_values().collect();
        }
        let longest = matches.iter().map(|m| m.length).max().unwrap_or(0);
        let mut last = None;
        for length in MIN_MATCH..=longest {
            let best = matches
                .iter()
                .filter(|m| m.length >= length)
                .min_by(|a, b| {
                    if config.match_cost == SourceMatchCost::SymbolId {
                        a.symbol.cmp(&b.symbol)
                    } else {
                        costs
                            .distance(a.symbol)
                            .total_cmp(&costs.distance(b.symbol))
                            .then(a.symbol.cmp(&b.symbol))
                    }
                })
                .unwrap();
            let price =
                costs.length(input[position].context, length, config) + costs.distance(best.symbol);
            let cost = if config.precise() {
                base + f64::from(price)
            } else {
                f64::from(base as f32 + price)
            };
            if cost < steps[position + length].0 {
                steps[position + length] = (cost, length, best.distance);
            }
            last = Some(*best);
        }
        let Some(last) = last else {
            repeats = 0;
            continue;
        };
        let end = position + longest;
        if last.symbol == 0
            || (last.symbol == 1 && width != 0)
            || (longest >= 256 && end == previous_end && last.symbol == previous_symbol)
        {
            repeats += 1;
        } else {
            repeats = 0;
        }
        previous_end = end;
        previous_symbol = last.symbol;
        if repeats >= 8 && longest >= 9 {
            skip = longest - 9;
            repeats = 0;
        }
    }
    let mut output = Vec::new();
    let mut end = input.len();
    while end != 0 {
        let (_, length, distance) = steps[end];
        assert!(length > 0);
        end -= length;
        output.push(if distance == 0 {
            Command::Literal(input[end])
        } else {
            Command::Match {
                context: input[end].context,
                length: length as u32,
                distance: distance as u32,
            }
        });
    }
    output.reverse();
    output
}

#[test]
fn range_parser_cost_equals_independent_scalar_dp_with_and_without_uint000_spans() {
    let mut values: Vec<_> = (0..73).map(|i| 65535 + (i * 19) % 31).collect();
    values.extend_from_within(0..69);
    values.extend_from_within(17..113);
    values.extend([65535; 81]);
    let input = tokens(&values);
    for method in [SourceLz77Method::Optc8, SourceLz77Method::Optc256] {
        for split in [0, 4] {
            let config = SourceConfig {
                method,
                match_cost: SourceMatchCost::Range,
                length_uint_config: HybridUintConfig::new(split, 0, 0).unwrap(),
                require_global_savings: false,
                ..SourceConfig::default()
            };
            let raw = Costs::raw(&[&input], &config);
            let sums = raw.cumulative(&input, &config).unwrap();
            let seed = careful_stream(&input, &values, 37, &config, &raw, &sums, &mut 0.0).unwrap();
            let costs = Costs::fitted(&[seed], &[37], &config);
            let actual = optimal_stream(
                &input,
                &values,
                37,
                &config,
                &costs,
                0,
                &mut Work::default(),
            )
            .unwrap();
            let expected = scalar_pass(&input, &values, 37, &config, &costs);
            let codes = DistanceCodes::new(37, WINDOW_SIZE);
            let score = |commands: &[Command]| {
                commands.iter().fold(0.0f32, |base, command| {
                    base + match *command {
                        Command::Literal(literal) => {
                            let encoded = config.literal_uint_config.encode(literal.value);
                            costs.bits(literal.context, encoded.token) + encoded.nbits as f32
                        }
                        Command::Match {
                            context,
                            length,
                            distance,
                        } => {
                            costs.length(context, length as usize, &config)
                                + costs.distance(codes.symbol(distance as usize))
                        }
                    }
                })
            };
            // Equal-cost queued/explicit ties can select different copies.
            assert_eq!(score(&actual), score(&expected));
            replay(&input, &actual, 37);
        }
    }
}

#[test]
fn compact_step_selection_matches_chain_bucket_index_and_precision_boundaries() {
    assert!(!SourceConfig::default().compact_steps);
    let chain = SourceConfig::new(SourceLz77Method::Optc256);
    assert!(!chain.uses_compact_steps());
    assert!(
        SourceConfig {
            compact_steps: true,
            ..chain.clone()
        }
        .uses_compact_steps()
    );
    assert!(
        SourceConfig {
            bucket_matcher: true,
            ..chain.clone()
        }
        .uses_compact_steps()
    );
    for bucket_mode in [
        SourceBucketMode::History,
        SourceBucketMode::WideKey,
        SourceBucketMode::WideRing,
        SourceBucketMode::PreciseRing,
        SourceBucketMode::Short128,
    ] {
        assert!(
            !SourceConfig {
                compact_steps: true,
                bucket_mode,
                ..chain.clone()
            }
            .uses_compact_steps()
        );
    }
    assert!(
        SourceConfig {
            bucket_mode: SourceBucketMode::WideRing,
            index_mode: SourceIndexMode::All,
            ..chain.clone()
        }
        .uses_compact_steps()
    );
    for method in [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Optc8,
    ] {
        assert!(
            !SourceConfig {
                method,
                compact_steps: true,
                bucket_matcher: true,
                ..chain.clone()
            }
            .uses_compact_steps()
        );
    }
    for match_cost in [
        SourceMatchCost::RangeReplay,
        SourceMatchCost::Range,
        SourceMatchCost::RangeRefit,
    ] {
        assert!(
            !SourceConfig {
                match_cost,
                compact_steps: true,
                bucket_matcher: true,
                ..chain.clone()
            }
            .uses_compact_steps()
        );
    }
}

#[test]
fn compact_chain_refit_preserves_commands_acceptance_contexts_and_physical_replay() {
    let values: Vec<_> = (0..733u32)
        .map(|p| match p % 83 {
            0..=19 => 0,
            20 => u32::MAX,
            p => 65535 + p % 17,
        })
        .collect();
    let input = tokens(&values);
    for width in [0, 1, 37] {
        for match_cost in [
            SourceMatchCost::SymbolId,
            SourceMatchCost::Entropy,
            SourceMatchCost::Frontier,
            SourceMatchCost::Refit,
        ] {
            let baseline = SourceConfig {
                method: SourceLz77Method::Optc256,
                match_cost,
                max_match_length: 127,
                require_global_savings: false,
                ..SourceConfig::default()
            };
            let mut wide_work = Work::default();
            let expected = run(&[&input], &[width], &baseline, &mut wide_work).unwrap();
            let compact = SourceConfig {
                compact_steps: true,
                ..baseline
            };
            let mut work = Work::default();
            let actual = run(&[&input], &[width], &compact, &mut work).unwrap();
            assert_eq!(actual, expected);
            assert!(actual.enabled);
            assert_eq!(work.compact_dp_streams, match_cost.passes());
            assert_eq!(wide_work.compact_dp_streams, 0);
            assert_eq!(
                (work.queries, work.intervals),
                (wide_work.queries, wide_work.intervals)
            );
            replay(&input, &actual.commands[0], width);
        }
    }
    let config = SourceConfig {
        method: SourceLz77Method::Optc256,
        compact_steps: true,
        ..SourceConfig::default()
    };
    let empty = parse_source_with_state(&[], 0, &config).unwrap();
    assert!(!empty.enabled && empty.commands.is_empty());
    let short = parse_source_with_state(&input[..2], 0, &config).unwrap();
    assert!(!short.enabled);
    replay(&input[..2], &short.commands, 0);
}

#[test]
fn compact_nodes_reach_indexed_endpoints_but_not_block_range_or_precise_routes() {
    let values: Vec<_> = (0..8192u32).map(|p| 65536 + p % 37).collect();
    let input = tokens(&values);
    let indexed = SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_mode: SourceBucketMode::WideRing,
        index_mode: SourceIndexMode::All,
        small_index: true,
        match_cost: SourceMatchCost::Frontier,
        max_match_length: 127,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    for block_mode in [
        SourceBlockMode::Disabled,
        SourceBlockMode::Endpoints,
        SourceBlockMode::Intervals,
    ] {
        let config = SourceConfig {
            block_mode,
            ..indexed.clone()
        };
        let mut work = Work::default();
        let actual = run(&[&input], &[37], &config, &mut work).unwrap();
        assert_eq!(
            work.compact_dp_streams,
            usize::from(block_mode == SourceBlockMode::Disabled)
        );
        assert_eq!(work.indexed_streams, 1);
        replay(&input, &actual.commands[0], 37);
    }
    for (bucket_mode, match_cost) in [
        (SourceBucketMode::Default, SourceMatchCost::Range),
        (SourceBucketMode::Default, SourceMatchCost::RangeRefit),
        (SourceBucketMode::PreciseRing, SourceMatchCost::Frontier),
    ] {
        let baseline = SourceConfig {
            bucket_mode,
            match_cost,
            bucket_matcher: true,
            index_mode: SourceIndexMode::Disabled,
            ..indexed.clone()
        };
        let config = SourceConfig {
            compact_steps: true,
            ..baseline.clone()
        };
        let mut work = Work::default();
        let actual = run(&[&input[..511]], &[37], &config, &mut work).unwrap();
        assert_eq!(work.compact_dp_streams, 0);
        assert_eq!(
            actual,
            run(&[&input[..511]], &[37], &baseline, &mut Work::default()).unwrap()
        );
        replay(&input[..511], &actual.commands[0], 37);
    }
}

#[test]
fn interval_parser_equals_scalar_endpoint_dp_including_strict_ties() {
    let mut varied: Vec<_> = (0..83).map(|i| (i * 19) % 29).collect();
    varied.extend_from_within(0..73);
    varied.extend_from_within(13..99);
    let cases = [varied, vec![65535; 599], vec![0; 47]];
    for (index, values) in cases.iter().enumerate() {
        let input = tokens(values);
        for (method, bucket_matcher, compact_steps) in [
            (SourceLz77Method::Optc8, false, false),
            (SourceLz77Method::Optc256, false, false),
            (SourceLz77Method::Optc256, true, false),
            (SourceLz77Method::Optc256, false, true),
        ] {
            for match_cost in [
                SourceMatchCost::SymbolId,
                SourceMatchCost::Entropy,
                SourceMatchCost::Frontier,
                SourceMatchCost::Refit,
            ] {
                let config = SourceConfig {
                    method,
                    bucket_matcher,
                    compact_steps,
                    match_cost,
                    length_uint_config: HybridUintConfig::new(if index == 0 { 4 } else { 0 }, 0, 0)
                        .unwrap(),
                    require_global_savings: false,
                    ..SourceConfig::default()
                };
                let width = [31, 0, 2][index];
                let costs = Costs::raw(&[&input], &config);
                let sums = costs.cumulative(&input, &config).unwrap();
                let seed = careful_stream(&input, values, width, &config, &costs, &sums, &mut 0.0)
                    .unwrap();
                let seed_tokens = wire_token_count(&seed);
                let costs = Costs::fitted(&[seed], &[width], &config);
                let actual = optimal_stream(
                    &input,
                    values,
                    width,
                    &config,
                    &costs,
                    seed_tokens,
                    &mut Work::default(),
                )
                .unwrap();
                assert_eq!(
                    actual,
                    scalar_pass(&input, values, width, &config, &costs),
                    "case={index} method={method:?} mode={match_cost:?} buckets={bucket_matcher} compact={compact_steps}"
                );
                replay(&input, &actual, width);
            }
        }
    }
}
