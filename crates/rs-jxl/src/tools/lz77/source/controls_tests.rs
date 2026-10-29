// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;

fn replay(input: &[Literal], commands: &[Command]) {
    let mut values = Vec::new();
    let mut copies = 0;
    for command in commands {
        match *command {
            Command::Literal(literal) => {
                assert_eq!(literal, input[values.len()]);
                values.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[values.len()].context);
                assert!(length >= 3 && distance > 0 && distance as usize <= values.len());
                assert!(values.len() + length as usize <= input.len());
                for _ in 0..length {
                    values.push(values[values.len() - distance as usize]);
                }
                copies += 1;
            }
        }
    }
    assert!(copies != 0);
    assert_eq!(values, input.iter().map(|t| t.value).collect::<Vec<_>>());
}

#[test]
fn generic_methods_preserve_exact_capacity_cost_veto_and_first_pass_metadata() {
    use super::super::policy::{SourceFirstPass, SourceMatcher, source_config, source_options};
    use SourceLz77Method::*;
    for (unpriced, priced, capacity) in [
        (Lz77b1w3f, Lz77b1w3t, 1),
        (Lz77b3w3f, Lz77b3w3t, 3),
        (Lz77b7w3f, Lz77b7w3t, 7),
        (Lz77b15w3f, Lz77b15w3t, 15),
        (Lz77b31w3f, Lz77b31w3t, 31),
    ] {
        for (method, comparison) in [(unpriced, false), (priced, true)] {
            let source = source_config(method);
            assert_eq!(
                source.matcher,
                SourceMatcher::FastHash {
                    bucket_capacity: capacity
                }
            );
            assert_eq!(source.runtime_cost_comparison, comparison);
            assert_eq!(
                source.special_distances_to_check,
                if capacity <= 3 { 24 } else { 120 }
            );
            assert_eq!(source.first_pass(65536, false), SourceFirstPass::None);
            assert!(!source_options(method).is_source_equivalent());
        }
    }
    for (method, limit) in [(Optc1, 1), (Optc3, 3)] {
        let source = source_config(method);
        assert_eq!(
            source.matcher,
            SourceMatcher::HashChain {
                max_chain_length: limit
            }
        );
        assert_eq!(
            source.first_pass(65535, false),
            SourceFirstPass::LazyChain256
        );
        assert_eq!(
            source.first_pass(65536, false),
            SourceFirstPass::GreedyBucket7
        );
        assert_eq!(
            source.first_pass(65536, true),
            SourceFirstPass::LazyChain256
        );
    }
}

#[test]
fn generic_w3f_bypasses_only_local_veto_not_global_savings() {
    use SourceLz77Method::*;
    let input = vec![
        Literal {
            value: 0,
            context: 0
        };
        128
    ];
    for (unpriced, priced) in [
        (Lz77b1w3f, Lz77b1w3t),
        (Lz77b3w3f, Lz77b3w3t),
        (Lz77b7w3f, Lz77b7w3t),
        (Lz77b15w3f, Lz77b15w3t),
        (Lz77b31w3f, Lz77b31w3t),
    ] {
        let config = SourceConfig {
            method: unpriced,
            max_match_length: 31,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let unpriced = parse_source_with_state(&input, 0, &config).unwrap();
        assert!(unpriced.enabled);
        replay(&input, &unpriced.commands);
        let priced = parse_source_with_state(
            &input,
            0,
            &SourceConfig {
                method: priced,
                ..config.clone()
            },
        )
        .unwrap();
        assert!(priced.enabled);
        assert!(
            priced
                .commands
                .iter()
                .all(|c| matches!(c, Command::Literal(_)))
        );
        let gated = parse_source_with_state(
            &input,
            0,
            &SourceConfig {
                require_global_savings: true,
                ..config
            },
        )
        .unwrap();
        assert!(!gated.enabled);
        assert!(
            gated
                .commands
                .iter()
                .all(|c| matches!(c, Command::Literal(_)))
        );
    }
}

#[test]
fn generic_optimal_and_fast_methods_independently_decode_wide_values_and_refits() {
    use SourceLz77Method::*;
    let input: Vec<_> = (0..513)
        .map(|i| Literal {
            context: (i % 3) as u32,
            value: if i % 97 < 12 {
                u32::MAX
            } else {
                (i % 11) as u32 + 65530
            },
        })
        .collect();
    for method in [
        Lz77b1w3f, Lz77b3w3f, Lz77b7w3f, Lz77b15w3f, Lz77b31w3f, Lz77b1w3t, Lz77b3w3t, Lz77b7w3t,
        Lz77b15w3t, Lz77b31w3t, Optc1, Optc3,
    ] {
        for width in [0, 1, 37] {
            let config = SourceConfig {
                method,
                require_global_savings: false,
                max_match_length: 129,
                ..SourceConfig::default()
            };
            replay(&input, &parse_source(&input, width, &config).unwrap());
        }
    }
    for method in [Optc1, Optc3] {
        for match_cost in [
            SourceMatchCost::Entropy,
            SourceMatchCost::Frontier,
            SourceMatchCost::Refit,
            SourceMatchCost::RangeReplay,
            SourceMatchCost::Range,
            SourceMatchCost::RangeRefit,
            SourceMatchCost::Auto,
        ] {
            let config = SourceConfig {
                method,
                match_cost,
                require_global_savings: false,
                chain_match_cache: SourceChainCache::Enabled,
                bucket_matcher: true,
                compact_steps: true,
                ..SourceConfig::default()
            };
            let mut work = Work::default();
            let result = run(&[&input], &[37], &config, &mut work).unwrap();
            assert!(work.careful_seed && work.compact_dp_streams == 0 && work.indexed_streams == 0);
            replay(&input, &result.commands[0]);
        }
    }
}

#[test]
fn source_interval_two_visits_exactly_first_and_last_eight_after_uint000_splitting() {
    for begin in 3..90 {
        for end in begin + 1..140 {
            for boundaries in [false, true] {
                let mut visited = Vec::new();
                visit_intervals(
                    begin,
                    end,
                    HybridUintConfig::new(0, 0, 0).unwrap(),
                    |first, stop| {
                        visit_endpoint_ranges(first, stop, boundaries, |a, b| visited.extend(a..b));
                    },
                );
                // Independent scalar partition by each length's encoded token.
                let uint = HybridUintConfig::new(0, 0, 0).unwrap();
                let mut expected = Vec::new();
                let mut first = begin;
                while first < end {
                    let token = uint.encode((first - 3) as u32).token;
                    let mut stop = first + 1;
                    while stop < end && uint.encode((stop - 3) as u32).token == token {
                        stop += 1;
                    }
                    for length in first..stop {
                        if !boundaries
                            || stop - first <= 16
                            || length - first < 8
                            || stop - length <= 8
                        {
                            expected.push(length);
                        }
                    }
                    first = stop;
                }
                assert_eq!(visited, expected);
            }
        }
    }
}

#[test]
fn boundary_only_is_an_actual_endpoint_approximation_not_an_exact_tree_alias() {
    let input = vec![
        Literal {
            value: 7,
            context: 3
        };
        128
    ];
    let mut final_costs = Vec::new();
    for boundaries in [false, true] {
        let mut parser = Parser::new(input.len(), false, false, 0).unwrap();
        let mut oracle = vec![f64::INFINITY; input.len() + 1];
        oracle[0] = 0.0;
        for position in 0..input.len() {
            parser.advance(position);
            assert_eq!(parser.cost(position), oracle[position]);
            let base = oracle[position];
            parser.literal(position, base + 3.0);
            oracle[position + 1] = oracle[position + 1].min(base + 3.0);
            let (end, cost) = match position {
                1 => (96, 5.0),
                50 => (79, base + 1.0),
                _ => continue,
            };
            visit_intervals(
                3,
                end,
                HybridUintConfig::new(0, 0, 0).unwrap(),
                |first, stop| {
                    visit_endpoint_ranges(first, stop, boundaries, |a, b| {
                        parser.interval(position, a, b, 1, cost)
                    });
                    for length in first..stop {
                        if !boundaries
                            || stop - first <= 16
                            || length < first + 8
                            || length >= stop - 8
                        {
                            oracle[position + length] = oracle[position + length].min(cost);
                        }
                    }
                },
            );
        }
        parser.advance(input.len());
        assert_eq!(parser.cost(input.len()), oracle[input.len()]);
        final_costs.push(parser.cost(input.len()));
        replay(&input, &parser.backtrack(&input).unwrap());
    }
    assert!(final_costs[1] > final_costs[0]);
}

#[test]
fn independent_decode_of_index_interval_choices_and_chain_bucket_controls() {
    let input: Vec<_> = (0..8193)
        .map(|i| Literal {
            value: if i % 257 < 70 {
                0
            } else {
                65530 + (i % 17) as u32
            },
            context: (i % 3) as u32,
        })
        .collect();
    let baseline = SourceConfig {
        method: SourceLz77Method::Optc256,
        match_cost: SourceMatchCost::Frontier,
        small_index: true,
        max_match_length: 513,
        require_global_savings: false,
        ..SourceConfig::default()
    };
    let mut exact = None;
    for index_interval_dp in [
        SourceIntervalDp::Disabled,
        SourceIntervalDp::AllEndpoints,
        SourceIntervalDp::Boundaries,
    ] {
        let config = SourceConfig {
            bucket_matcher: true,
            index_mode: SourceIndexMode::All,
            index_interval_dp,
            ..baseline.clone()
        };
        let mut work = Work::default();
        let output = run(&[&input], &[37], &config, &mut work).unwrap();
        assert!(output.enabled && work.indexed_streams > 0 && work.block_streams == 0);
        replay(&input, &output.commands[0]);
        if index_interval_dp == SourceIntervalDp::Disabled {
            exact = Some(output.commands);
        } else if index_interval_dp == SourceIntervalDp::AllEndpoints {
            assert_eq!(Some(output.commands), exact);
        }
    }
    for chain_match_cache in [
        SourceChainCache::Auto,
        SourceChainCache::Disabled,
        SourceChainCache::Enabled,
    ] {
        let config = SourceConfig {
            chain_limit: Some(3),
            chain_match_cache,
            ..baseline.clone()
        };
        let output = parse_source(&input, 37, &config).unwrap();
        replay(&input, &output);
    }
    for bucket_mode in [
        SourceBucketMode::Default,
        SourceBucketMode::History,
        SourceBucketMode::WideKey,
        SourceBucketMode::WideRing,
        SourceBucketMode::PreciseRing,
        SourceBucketMode::Short128,
    ] {
        let config = SourceConfig {
            bucket_matcher: true,
            bucket_mode,
            bucket_dominance: true,
            ..baseline.clone()
        };
        replay(&input, &parse_source(&input, 37, &config).unwrap());
    }
}

#[test]
fn independent_overrides_do_not_enable_ineligible_or_default_routes() {
    let defaults = SourceConfig::default();
    assert_eq!(defaults.chain_limit, None);
    assert_eq!(defaults.chain_match_cache, SourceChainCache::Auto);
    assert!(!defaults.bucket_dominance);
    assert_eq!(defaults.index_interval_dp, SourceIntervalDp::Disabled);
    assert_eq!(defaults.fast_ranges, SourceFastRanges::Disabled);
    assert_eq!(defaults.fast_prefix, None);
    let input: Vec<_> = (0..513)
        .map(|i| Literal {
            value: (i % 11) as u32,
            context: 0,
        })
        .collect();
    for method in [SourceLz77Method::Lz77b1w3t, SourceLz77Method::Lz77b3w3t] {
        let config = SourceConfig {
            method,
            require_global_savings: false,
            ..defaults.clone()
        };
        let expected = parse_source(&input, 37, &config).unwrap();
        let overrides = SourceConfig {
            chain_limit: Some(1),
            chain_match_cache: SourceChainCache::Enabled,
            bucket_dominance: true,
            index_interval_dp: SourceIntervalDp::Boundaries,
            ..config
        };
        assert_eq!(parse_source(&input, 37, &overrides).unwrap(), expected);
    }
    let config = SourceConfig {
        method: SourceLz77Method::Optc256,
        match_cost: SourceMatchCost::SymbolId,
        bucket_matcher: true,
        require_global_savings: false,
        ..defaults
    };
    let expected = parse_source(&input, 37, &config).unwrap();
    let overrides = SourceConfig {
        bucket_dominance: true,
        index_interval_dp: SourceIntervalDp::Boundaries,
        ..config
    };
    assert_eq!(parse_source(&input, 37, &overrides).unwrap(), expected);
}

#[test]
fn fast_range_options_preserve_commands_and_decode_with_clipped_tails() {
    let mut input: Vec<_> = (0..513)
        .map(|i| Literal {
            value: if i % 97 < 40 {
                u32::MAX
            } else {
                65530 + i % 11
            },
            context: i % 3,
        })
        .collect();
    input.extend(
        [Literal {
            value: u32::MAX,
            context: 2,
        }; 2],
    );
    for method in [
        SourceLz77Method::Lz77b1w3f,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Lz77b31w3t,
        SourceLz77Method::Optc3,
    ] {
        let base = SourceConfig {
            method,
            max_match_length: 33,
            require_global_savings: false,
            ..SourceConfig::default()
        };
        let expected = parse_source(&input, 37, &base).unwrap();
        for fast_ranges in [
            SourceFastRanges::Disabled,
            SourceFastRanges::RepeatedInsert,
            SourceFastRanges::MatchLength,
            SourceFastRanges::Both,
        ] {
            for fast_prefix in [
                None,
                Some(SourceFastPrefix::Disabled),
                Some(SourceFastPrefix::Scalar),
                Some(SourceFastPrefix::Simd),
            ] {
                let config = SourceConfig {
                    fast_ranges,
                    fast_prefix,
                    ..base.clone()
                };
                let commands = parse_source(&input, 37, &config).unwrap();
                assert_eq!(commands, expected);
                replay(&input, &commands);
            }
        }
    }
}
