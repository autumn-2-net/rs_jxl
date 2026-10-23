// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;

const ALIASES: [SourceBucketSelection; 9] = [
    SourceBucketSelection::Unset,
    SourceBucketSelection::Short32,
    SourceBucketSelection::Parallel32,
    SourceBucketSelection::History,
    SourceBucketSelection::WideKey,
    SourceBucketSelection::WideRing,
    SourceBucketSelection::ParallelWide,
    SourceBucketSelection::PreciseRing,
    SourceBucketSelection::Short128,
];

fn replay(input: &[Literal], commands: &[Command]) {
    let mut output = Vec::new();
    let mut copies = 0;
    for command in commands {
        match *command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[output.len()]);
                output.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[output.len()].context);
                assert!(length >= 3 && distance > 0 && distance as usize <= output.len());
                assert!(output.len() + length as usize <= input.len());
                for _ in 0..length {
                    output.push(output[output.len() - distance as usize]);
                }
                copies += 1;
            }
        }
    }
    assert!(copies > 0);
    assert_eq!(output, input.iter().map(|t| t.value).collect::<Vec<_>>());
}

#[test]
fn exact_source_aliases_distinguish_layout_and_parameter_precedence() {
    use SourceBucketSelection as S;
    assert_eq!(SourceConfig::default().bucket_selection, None);
    let base = SourceConfig {
        method: SourceLz77Method::Optc256,
        match_cost: SourceMatchCost::Frontier,
        index_mode: SourceIndexMode::All,
        small_index: true,
        compact_steps: true,
        // Exact source selections ignore this older candidate-only override.
        bucket_mode: SourceBucketMode::PreciseRing,
        ..SourceConfig::default()
    };
    let input = vec![
        Literal {
            value: 7,
            context: 0
        };
        8192
    ];
    for selection in ALIASES {
        let config = SourceConfig {
            bucket_selection: Some(selection),
            ..base.clone()
        };
        let route = config.bucket_route();
        let compact_bucket = matches!(selection, S::Parallel32 | S::ParallelWide);
        assert_eq!(route.enabled, selection != S::Unset);
        assert_eq!(
            config.uses_compact_steps(),
            compact_bucket || selection == S::Unset
        );
        assert_eq!(config.precise(), selection == S::PreciseRing);
        assert_eq!(index::enabled(&[&input], &config), compact_bucket);
        assert_eq!(route.parallel, compact_bucket || selection == S::Unset);
        let expected_mode = match selection {
            S::Unset | S::Short32 | S::Parallel32 => SourceBucketMode::Default,
            S::History => SourceBucketMode::History,
            S::WideKey => SourceBucketMode::WideKey,
            S::WideRing | S::ParallelWide => SourceBucketMode::WideRing,
            S::PreciseRing => SourceBucketMode::PreciseRing,
            S::Short128 => SourceBucketMode::Short128,
        };
        assert_eq!(route.mode, expected_mode);
        let parameter = SourceConfig {
            bucket_matcher: true,
            ..config
        };
        assert_eq!(
            parameter.bucket_route(),
            BucketRoute {
                mode: SourceBucketMode::Default,
                enabled: true,
                compact: true,
                indexed: true,
                parallel: true
            }
        );
        assert!(!parameter.precise());
        assert!(index::enabled(&[&input], &parameter));
    }
    let legacy = SourceConfig {
        bucket_mode: SourceBucketMode::WideRing,
        ..base.clone()
    };
    assert!(index::enabled(&[&input], &legacy));
    assert!(legacy.uses_compact_steps());
    assert!(
        SourceConfig {
            bucket_matcher: true,
            ..base
        }
        .precise()
    );
}

#[test]
fn source_alias_routes_independently_decode_and_use_expected_nodes_and_indexes() {
    use SourceBucketSelection as S;
    let input: Vec<_> = (0..8193)
        .map(|i| Literal {
            context: (i % 3) as u32,
            value: if i % 127 < 32 {
                u32::MAX
            } else {
                65530 + (i % 17) as u32
            },
        })
        .collect();
    for selection in ALIASES {
        let config = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_selection: Some(selection),
            compact_steps: true,
            match_cost: SourceMatchCost::Frontier,
            index_mode: SourceIndexMode::All,
            small_index: true,
            max_match_length: 129,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let mut work = Work::default();
        let result = run(&[&input], &[37], &config, &mut work).unwrap();
        let indexed = matches!(selection, S::Parallel32 | S::ParallelWide);
        assert_eq!(work.indexed_streams, usize::from(indexed), "{selection:?}");
        assert_eq!(
            work.compact_dp_streams,
            usize::from(indexed || selection == S::Unset),
            "{selection:?}"
        );
        assert_eq!(work.block_streams, 0);
        replay(&input, &result.commands[0]);
    }
    let parameter = SourceConfig {
        method: SourceLz77Method::Optc256,
        bucket_matcher: true,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    let expected = parse_source(&input[..513], 37, &parameter).unwrap();
    for selection in ALIASES {
        let config = SourceConfig {
            bucket_selection: Some(selection),
            ..parameter.clone()
        };
        assert_eq!(parse_source(&input[..513], 37, &config).unwrap(), expected);
    }
}

#[test]
fn alias_stream_parallel_and_shared_seed_gates_follow_compact_instantiation() {
    use SourceBucketSelection as S;
    let stream = vec![
        Literal {
            value: 7,
            context: 0
        };
        32768
    ];
    let inputs = [&stream[..], &stream[..], &stream[..]];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for selection in ALIASES {
            let config = SourceConfig {
                method: SourceLz77Method::Optc256,
                bucket_selection: Some(selection),
                parallelism: Some(SourceParallelism::default()),
                ..SourceConfig::default()
            };
            let parallel = matches!(selection, S::Unset | S::Parallel32 | S::ParallelWide);
            assert_eq!(
                Schedule::new(&inputs, &config, false).tasks,
                if parallel { 2 } else { 1 }
            );
            crate::execution::serial(|| {
                assert_eq!(Schedule::new(&inputs, &config, false).tasks, 1)
            });
        }
    });
    let values = [vec![7u32; 65536]];
    for selection in [S::Short32, S::Parallel32, S::WideRing, S::ParallelWide] {
        let config = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_selection: Some(selection),
            shared_index: true,
            index_mode: SourceIndexMode::All,
            block_mode: SourceBlockMode::Intervals,
            ..SourceConfig::default()
        };
        assert_eq!(
            blocks::shared_index(&values, &[37], &config)
                .unwrap()
                .is_some(),
            matches!(selection, S::Parallel32 | S::ParallelWide)
        );
    }
}

#[test]
fn source_aliases_do_not_override_other_methods_or_range_parsers() {
    let input: Vec<_> = (0..257)
        .map(|i| Literal {
            value: i % 11,
            context: i % 3,
        })
        .collect();
    for (method, match_cost) in [
        (SourceLz77Method::Optc3, SourceMatchCost::Frontier),
        (SourceLz77Method::Optc256, SourceMatchCost::Range),
        (SourceLz77Method::Lz77b3w3t, SourceMatchCost::SymbolId),
    ] {
        let base = SourceConfig {
            method,
            match_cost,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let expected = parse_source(&input, 37, &base).unwrap();
        for selection in ALIASES {
            let config = SourceConfig {
                bucket_selection: Some(selection),
                compact_steps: true,
                ..base.clone()
            };
            let commands = parse_source(&input, 37, &config).unwrap();
            assert_eq!(commands, expected);
            assert!(!config.uses_compact_steps());
            replay(&input, &commands);
        }
    }
}
