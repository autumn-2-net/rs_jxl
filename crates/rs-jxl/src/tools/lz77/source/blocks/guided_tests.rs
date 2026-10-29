// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Source-derived fixtures follow BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;
use std::cell::RefCell;

fn decode_part(input: &[Literal], begin: usize, end: usize, commands: &[Command], maximum: usize) {
    let mut decoded: Vec<_> = input[..begin].iter().map(|t| t.value).collect();
    for command in commands {
        let position = decoded.len();
        match *command {
            Command::Literal(token) => {
                assert_eq!(token, input[position]);
                decoded.push(token.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[position].context);
                assert!((MIN_MATCH..=maximum).contains(&(length as usize)));
                assert!(distance != 0 && distance as usize <= position);
                assert!((distance as usize) < window(input.len()));
                assert!(position + length as usize <= end);
                for _ in 0..length {
                    decoded.push(decoded[decoded.len() - distance as usize]);
                }
            }
        }
    }
    assert_eq!(
        decoded,
        input[..end].iter().map(|t| t.value).collect::<Vec<_>>()
    );
}

#[test]
fn seed_spans_keep_physical_distances_and_exact_repair_boundaries() {
    let literal = Command::Literal(Literal {
        context: 0,
        value: 7,
    });
    let mut seed = vec![literal; 37];
    seed.push(Command::Match {
        context: 0,
        length: 64,
        distance: 37,
    });
    seed.extend([literal; 2]);
    seed.push(Command::Match {
        context: 0,
        length: 96,
        distance: 1,
    });
    assert_eq!(
        seed_spans(&seed, 199).unwrap(),
        [
            SeedSpan {
                begin: 37,
                end: 101,
                distance: 37
            },
            SeedSpan {
                begin: 103,
                end: 199,
                distance: 1
            },
        ]
    );
    for length in [3, 63, 64, 65, 95, 96, 97, 129] {
        let span = SeedSpan {
            begin: 37,
            end: 37 + length,
            distance: 1,
        };
        let expected: Vec<_> = (0..length)
            .filter(|&offset| {
                length < 64 || offset < 8 || offset >= length - 8 || (offset / 32) * 32 == offset
            })
            .map(|offset| 37 + offset)
            .collect();
        let actual: Vec<_> = (span.begin..span.end).filter(|&p| span.search(p)).collect();
        assert_eq!(actual, expected, "length={length}");
    }
    assert!(seed_spans(&[], 0).unwrap().is_empty());
    assert!(seed_spans(&seed, 198).is_err());
    assert!(seed_spans(&seed, 200).is_err());
}

#[test]
fn guidance_traces_containing_spans_repair_origins_and_nonzero_block_start() {
    let input = vec![
        Literal {
            context: 0,
            value: 65536
        };
        300
    ];
    let spans = [
        SeedSpan {
            begin: 1,
            end: 60,
            distance: 1,
        },
        SeedSpan {
            begin: 80,
            end: 240,
            distance: 17,
        },
        SeedSpan {
            begin: 250,
            end: 300,
            distance: 37,
        },
    ];
    for mode in [
        SourceSeedGuided::Disabled,
        SourceSeedGuided::Hints,
        SourceSeedGuided::SparseOrigins,
    ] {
        let config = SourceConfig {
            block_mode: SourceBlockMode::Intervals,
            block_seed_guided: mode,
            ..SourceConfig::default()
        };
        let costs = Costs::raw(&[&input], &config);
        let prices =
            DistancePrices::new(window(input.len()), 37, &config, |d| costs.distance(d)).unwrap();
        let trace = RefCell::new(Vec::new());
        let find = |p,
                    query: Query<'_>,
                    _: &mut MatchCache,
                    _: &dyn Fn(u32) -> f32,
                    _: &mut dyn FnMut(Match)| {
            trace
                .borrow_mut()
                .push((p, query.hint_distance, query.maximum));
        };
        let active = if mode == SourceSeedGuided::Disabled {
            &[][..]
        } else {
            &spans[..]
        };
        let commands = parse_part(
            &input, 100, 300, 37, &config, &costs, &prices, active, &find,
        )
        .unwrap();
        let expected: Vec<_> = (100..300)
            .filter_map(|p| {
                let copy = active.iter().find(|s| s.begin <= p && p < s.end);
                if mode == SourceSeedGuided::SparseOrigins
                    && p < 232
                    && ![112, 144, 176, 208].contains(&p)
                {
                    return None;
                }
                Some((p, copy.map_or(0, |s| s.distance), 300 - p))
            })
            .collect();
        assert_eq!(*trace.borrow(), expected);
        assert!(commands.iter().all(|c| matches!(c, Command::Literal(_))));
        decode_part(&input, 100, 300, &commands, config.max_match_length);
    }
}

#[test]
fn seed_hint_nominates_an_evicted_copy_before_recent_bucket_candidates() {
    let mut values = Vec::new();
    for block in 0..60u32 {
        values.extend([10, 11, 12, 13, 14, 15, 16, 17]);
        values.extend((8..64).map(|i| 10000 + block * 64 + i));
    }
    let begin = values.len();
    values.extend_from_within(0..64);
    let input: Vec<_> = values
        .iter()
        .map(|&value| Literal { value, context: 0 })
        .collect();
    let mut seed: Vec<_> = input[..begin]
        .iter()
        .copied()
        .map(Command::Literal)
        .collect();
    seed.push(Command::Match {
        context: 0,
        length: 64,
        distance: begin as u32,
    });
    let spans = seed_spans(&seed, input.len()).unwrap();
    for mode in [
        SourceSeedGuided::Disabled,
        SourceSeedGuided::Hints,
        SourceSeedGuided::SparseOrigins,
    ] {
        let config = SourceConfig {
            block_mode: SourceBlockMode::Intervals,
            block_seed_guided: mode,
            max_match_length: 64,
            ..SourceConfig::default()
        };
        let costs = Costs::raw(&[&input], &config);
        let prices =
            DistancePrices::new(window(input.len()), 0, &config, |d| costs.distance(d)).unwrap();
        let index = super::super::StaticIndex::new(&values, 0, &config).unwrap();
        let nominations = RefCell::new(Vec::new());
        let find = |p,
                    query: Query<'_>,
                    cache: &mut MatchCache,
                    price: &dyn Fn(u32) -> f32,
                    visit: &mut dyn FnMut(Match)| {
            index.find_query(
                p,
                query,
                price,
                |m| {
                    if p == begin {
                        nominations.borrow_mut().push((m.length, m.distance));
                    }
                    visit(m);
                },
                cache,
            );
        };
        let active = if mode == SourceSeedGuided::Disabled {
            &[][..]
        } else {
            &spans[..]
        };
        let commands = parse_part(
            &input,
            begin,
            input.len(),
            0,
            &config,
            &costs,
            &prices,
            active,
            &find,
        )
        .unwrap();
        if mode == SourceSeedGuided::Disabled {
            assert!(nominations.borrow().iter().all(|&(length, _)| length < 64));
        } else {
            assert_eq!(nominations.borrow()[0], (64, begin));
            assert_eq!(
                commands[0],
                Command::Match {
                    context: 0,
                    length: 64,
                    distance: begin as u32
                }
            );
        }
        decode_part(
            &input,
            begin,
            input.len(),
            &commands,
            config.max_match_length,
        );
    }
}

#[test]
fn lazy_lookahead_reuses_pending_choice_and_bypasses_dp_query_controls() {
    let input = vec![
        Literal {
            context: 0,
            value: 65536
        };
        70
    ];
    let config = SourceConfig {
        block_mode: SourceBlockMode::Endpoints,
        block_lazy: true,
        block_seed_guided: SourceSeedGuided::SparseOrigins,
        block_shared_frontier: true,
        block_frontier_carry: true,
        block_distance_bound: true,
        block_distance_classes: true,
        block_sparse_cache: true,
        ..SourceConfig::default()
    };
    let costs = Costs::raw(&[&input], &config);
    let prices =
        DistancePrices::new(window(input.len()), 0, &config, |d| costs.distance(d)).unwrap();
    let spans = [SeedSpan {
        begin: 1,
        end: 70,
        distance: 1,
    }];
    let trace = RefCell::new(Vec::new());
    let find = |p,
                query: Query<'_>,
                cache: &mut MatchCache,
                _: &dyn Fn(u32) -> f32,
                visit: &mut dyn FnMut(Match)| {
        assert_eq!(query.hint_distance, 0);
        assert!(query.result.is_none() && query.carry.is_none());
        assert_eq!(query.bound, super::super::search::DistanceBound::default());
        assert_eq!(query.maximum, input.len() - p);
        cache.advance(p);
        trace.borrow_mut().push(p);
        if p == 34 {
            visit(Match {
                length: 3,
                distance: 1,
                symbol: 0,
            });
        }
        if p == 35 {
            visit(Match {
                length: 8,
                distance: 2,
                symbol: 1,
            });
        }
    };
    let commands = parse_part(&input, 34, 70, 0, &config, &costs, &prices, &spans, &find).unwrap();
    assert_eq!(commands[0], Command::Literal(input[34]));
    assert_eq!(
        commands[1],
        Command::Match {
            context: 0,
            length: 8,
            distance: 2
        }
    );
    let mut expected = vec![34, 35, 36];
    expected.extend(43..68);
    assert_eq!(*trace.borrow(), expected);
    decode_part(&input, 34, 70, &commands, config.max_match_length);
}

#[test]
fn lazy_fractional_cost_uses_source_integer_interval_type_and_rejects_zero_gain() {
    let input = vec![
        Literal {
            context: 0,
            value: 0
        };
        8
    ];
    let config = SourceConfig {
        block_mode: SourceBlockMode::Intervals,
        block_lazy: true,
        ..SourceConfig::default()
    };
    assert_eq!(lazy_price(1.7, 1.7), 3.0);
    for literal_bits in [1.0, 1.1] {
        let mut costs = Costs::raw(&[&input], &config);
        let row = &mut costs.literals.get_mut(&0).unwrap().bits;
        row.insert(0, literal_bits);
        row.insert(config.min_symbol(), 1.7);
        costs.distances.bits.insert(0, 1.7);
        let prices = DistancePrices::new(8, 0, &config, |d| costs.distance(d)).unwrap();
        let find = |p,
                    _: Query<'_>,
                    _: &mut MatchCache,
                    _: &dyn Fn(u32) -> f32,
                    visit: &mut dyn FnMut(Match)| {
            if p == 4 {
                visit(Match {
                    length: 3,
                    distance: 1,
                    symbol: 0,
                });
            }
        };
        let commands = parse_part(&input, 4, 8, 0, &config, &costs, &prices, &[], &find).unwrap();
        assert_eq!(
            matches!(
                commands[0],
                Command::Match {
                    length: 3,
                    distance: 1,
                    ..
                }
            ),
            literal_bits > 1.0
        );
        decode_part(&input, 4, 8, &commands, config.max_match_length);
    }
}

#[test]
fn lazy_positive_gain_ties_prefer_length_then_first_candidate_not_symbol() {
    let input = vec![
        Literal {
            context: 0,
            value: 0
        };
        12
    ];
    let config = SourceConfig {
        block_mode: SourceBlockMode::Intervals,
        block_lazy: true,
        ..SourceConfig::default()
    };
    let mut costs = Costs::raw(&[&input], &config);
    let row = &mut costs.literals.get_mut(&0).unwrap().bits;
    row.insert(0, 2.0);
    row.insert(config.min_symbol(), 1.0);
    row.insert(config.min_symbol() + 1, 3.0);
    costs.distances.bits.insert(0, 0.0);
    costs.distances.bits.insert(1, 0.0);
    let prices = DistancePrices::new(16, 0, &config, |d| costs.distance(d)).unwrap();
    let find = |p,
                _: Query<'_>,
                _: &mut MatchCache,
                _: &dyn Fn(u32) -> f32,
                visit: &mut dyn FnMut(Match)| {
        if p == 4 {
            visit(Match {
                length: 4,
                distance: 2,
                symbol: 1,
            });
            visit(Match {
                length: 4,
                distance: 1,
                symbol: 0,
            });
        }
    };
    let commands = parse_part(&input, 4, 12, 0, &config, &costs, &prices, &[], &find).unwrap();
    assert_eq!(
        commands[0],
        Command::Match {
            context: 0,
            length: 4,
            distance: 2
        }
    );
    decode_part(&input, 4, 12, &commands, config.max_match_length);
}

#[test]
fn lazy_zero_price_suffix_never_queries_or_copies() {
    let input = vec![
        Literal {
            context: 0,
            value: 0
        };
        100
    ];
    let config = SourceConfig {
        block_lazy: true,
        ..SourceConfig::default()
    };
    let costs = Costs::raw(&[&input], &config);
    let prices = DistancePrices::new(128, 0, &config, |d| costs.distance(d)).unwrap();
    let find = |_: usize,
                _: Query<'_>,
                _: &mut MatchCache,
                _: &dyn Fn(u32) -> f32,
                _: &mut dyn FnMut(Match)| panic!("zero-cost suffix must not query");
    let commands = parse_part(&input, 0, 100, 0, &config, &costs, &prices, &[], &find).unwrap();
    assert_eq!(commands.len(), input.len());
    decode_part(&input, 0, 100, &commands, config.max_match_length);
}

#[test]
fn lazy_static_and_suffix_parses_match_an_all_length_scalar_oracle() {
    use super::super::{SourceSuffixMode, StaticIndex, suffix::SuffixIndex};
    let values: Vec<u32> = (0..271)
        .map(|i| if i % 61 < 13 { 0 } else { 65536 + i % 19 })
        .collect();
    let input: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &value)| Literal {
            value,
            context: (i % 3) as u32,
        })
        .collect();
    for width in [0, 37] {
        for suffix_mode in [SourceSuffixMode::Disabled, SourceSuffixMode::SlidingLcp] {
            let config = SourceConfig {
                block_mode: SourceBlockMode::Intervals,
                block_lazy: true,
                suffix_mode,
                block_sparse_cache: true,
                block_distance_table: true,
                block_distance_classes: true,
                max_match_length: 73,
                ..SourceConfig::default()
            };
            let costs = Costs::raw(&[&input], &config);
            let prices =
                DistancePrices::new(window(input.len()), width, &config, |d| costs.distance(d))
                    .unwrap();
            let index = StaticIndex::new(&values, width, &config).unwrap();
            let suffix = SuffixIndex::new(&values, width, &config).unwrap();
            let find = |p,
                        query: Query<'_>,
                        cache: &mut MatchCache,
                        price: &dyn Fn(u32) -> f32,
                        visit: &mut dyn FnMut(Match)| {
                if let Some(suffix) = &suffix {
                    suffix.find_query(p, query, price, visit, cache);
                } else {
                    index.find_query(p, query, price, visit, cache);
                }
            };
            for cut in [0, 97, 191, input.len()].windows(2) {
                let (begin, end) = (cut[0], cut[1]);
                let commands = parse_part(
                    &input,
                    begin,
                    end,
                    width,
                    &config,
                    &costs,
                    &prices,
                    &[],
                    &find,
                )
                .unwrap();
                let mut prefix = vec![0.0f64];
                for token in &input[begin..end] {
                    let encoded = config.literal_uint_config.encode(token.value);
                    let with_bits = prefix.last().unwrap() + f64::from(encoded.nbits);
                    prefix.push(with_bits + f64::from(costs.bits(token.context, encoded.token)));
                }
                // Enumerate every length, without cost intervals, cached endpoints,
                // pending choices, or the implementation's LazyChoice/price helper.
                let choose = |p: usize| {
                    let mut best = (0usize, 0usize, 0.0f64);
                    find(
                        p,
                        Query {
                            maximum: end - p,
                            ..Query::default()
                        },
                        &mut MatchCache::uncached(),
                        &|d| costs.distance(d),
                        &mut |m| {
                            for length in MIN_MATCH..=m.length {
                                let price = (costs.length(input[p].context, length, &config)
                                    + costs.distance(m.symbol))
                                .trunc();
                                let gain = prefix[p - begin + length]
                                    - prefix[p - begin]
                                    - f64::from(price);
                                if gain > best.2
                                    || (gain == best.2 && gain > 0.0 && length > best.0)
                                {
                                    best = (length, m.distance, gain);
                                }
                            }
                        },
                    );
                    best
                };
                let mut expected = Vec::new();
                let mut p = begin;
                while p < end {
                    let best = choose(p);
                    if best.0 < MIN_MATCH || (p + 1 < end && choose(p + 1).2 > best.2) {
                        expected.push(Command::Literal(input[p]));
                        p += 1;
                    } else {
                        expected.push(Command::Match {
                            context: input[p].context,
                            length: best.0 as u32,
                            distance: best.1 as u32,
                        });
                        p += best.0;
                    }
                }
                assert_eq!(
                    commands, expected,
                    "width={width}, suffix={suffix_mode:?}, begin={begin}"
                );
                decode_part(&input, begin, end, &commands, config.max_match_length);
            }
        }
    }
}

#[test]
fn guided_and_lazy_real_indexes_decode_across_seed_cuts_and_cache_modes() {
    use super::super::{SourceSuffixMode, StaticIndex, suffix::SuffixIndex};
    let values: Vec<u32> = (0..511).map(|i| 65536 + i % 17).collect();
    let input: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &value)| Literal {
            value,
            context: (i % 3) as u32,
        })
        .collect();
    let mut seed: Vec<_> = input[..34].iter().copied().map(Command::Literal).collect();
    for cut in [34, 194, 354, input.len()].windows(2) {
        seed.push(Command::Match {
            context: input[cut[0]].context,
            length: (cut[1] - cut[0]) as u32,
            distance: 17,
        });
    }
    let spans = seed_spans(&seed, input.len()).unwrap();
    for width in [0, 37] {
        for suffix_mode in [SourceSuffixMode::Disabled, SourceSuffixMode::SlidingLcp] {
            for mode in [SourceSeedGuided::Hints, SourceSeedGuided::SparseOrigins] {
                for lazy in [false, true] {
                    for cache_mode in 0..3 {
                        let config = SourceConfig {
                            block_mode: if cache_mode == 0 {
                                SourceBlockMode::Endpoints
                            } else {
                                SourceBlockMode::Intervals
                            },
                            block_seed_guided: mode,
                            block_lazy: lazy,
                            suffix_mode,
                            index_word_match: true,
                            block_shared_frontier: true,
                            block_frontier_carry: true,
                            block_distance_bound: true,
                            block_distance_classes: true,
                            block_distance_table: cache_mode == 1,
                            index_match_cache: cache_mode == 1,
                            block_sparse_cache: cache_mode == 2,
                            max_match_length: 67,
                            ..SourceConfig::default()
                        };
                        let costs = Costs::raw(&[&input], &config);
                        let prices =
                            DistancePrices::new(window(input.len()), width, &config, |d| {
                                costs.distance(d)
                            })
                            .unwrap();
                        let index = StaticIndex::new(&values, width, &config).unwrap();
                        let suffix = SuffixIndex::new(&values, width, &config).unwrap();
                        let find =
                            |p,
                             query: Query<'_>,
                             cache: &mut MatchCache,
                             price: &dyn Fn(u32) -> f32,
                             visit: &mut dyn FnMut(Match)| {
                                if let Some(suffix) = &suffix {
                                    suffix.find_query(p, query, price, visit, cache);
                                } else {
                                    index.find_query(p, query, price, visit, cache);
                                }
                            };
                        for cut in [0, 194, 354, input.len()].windows(2) {
                            let commands = parse_part(
                                &input, cut[0], cut[1], width, &config, &costs, &prices, &spans,
                                &find,
                            )
                            .unwrap();
                            assert!(commands.iter().any(|c| matches!(c, Command::Match { .. })));
                            decode_part(&input, cut[0], cut[1], &commands, config.max_match_length);
                        }
                    }
                }
            }
        }
    }
}
