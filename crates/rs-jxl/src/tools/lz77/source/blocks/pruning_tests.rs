// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Source-derived fixtures follow BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;
use std::cell::RefCell;

fn decode(input: &[Literal], begin: usize, end: usize, commands: &[Command], maximum: usize) {
    let mut values: Vec<_> = input[..begin].iter().map(|t| t.value).collect();
    for command in commands {
        let position = values.len();
        match *command {
            Command::Literal(token) => {
                assert_eq!(token, input[position]);
                values.push(token.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                assert_eq!(context, input[position].context);
                assert!((MIN_MATCH..=maximum).contains(&(length as usize)));
                assert!(distance > 0 && distance as usize <= position);
                assert!((distance as usize) < window(input.len()));
                assert!(position + length as usize <= end);
                for _ in 0..length {
                    values.push(values[values.len() - distance as usize]);
                }
            }
        }
    }
    assert_eq!(
        values,
        input[..end].iter().map(|t| t.value).collect::<Vec<_>>()
    );
}

#[test]
fn zero_tail_requires_exact_zero_bits_and_no_extra_bits_within_this_block() {
    let input = [
        Literal {
            context: 0,
            value: 7,
        },
        Literal {
            context: 1,
            value: 16,
        },
        Literal {
            context: 0,
            value: 7,
        },
        Literal {
            context: 0,
            value: 7,
        },
    ];
    let config = SourceConfig::default();
    let mut costs = Costs::raw(&[&input], &config);
    assert_eq!(
        costs.bits(1, config.literal_uint_config.encode(16).token),
        0.0
    );
    assert_ne!(config.literal_uint_config.encode(16).nbits, 0);
    assert_eq!(zero_tail_end(&input, 0, 4, &config, &costs), 2);
    assert_eq!(zero_tail_end(&input, 3, 4, &config, &costs), 3);
    assert_eq!(zero_tail_end(&input, 4, 4, &config, &costs), 4);
    costs
        .literals
        .get_mut(&0)
        .unwrap()
        .bits
        .insert(7, f32::from_bits(1));
    assert_eq!(zero_tail_end(&input, 0, 4, &config, &costs), 4);
    costs.literals.get_mut(&0).unwrap().bits.insert(7, -0.0);
    assert_eq!(zero_tail_end(&input, 0, 4, &config, &costs), 2);
    let prefix = literal_bound_prefix(&input, &config, &costs).unwrap();
    assert_eq!(prefix, [0.0, 0.0, 2.0, 2.0, 2.0]);
}

#[test]
fn literal_bound_prefix_promotes_prices_before_adding_extra_bits() {
    let input = vec![
        Literal {
            context: 0,
            value: 16
        };
        3
    ];
    let config = SourceConfig::default();
    let mut costs = Costs::raw(&[&input], &config);
    let encoded = config.literal_uint_config.encode(16);
    costs
        .literals
        .get_mut(&0)
        .unwrap()
        .bits
        .insert(encoded.token, 0.1);
    let prefix = literal_bound_prefix(&input, &config, &costs).unwrap();
    let price = f64::from(0.1f32) + f64::from(encoded.nbits);
    assert_eq!(prefix, [0.0, price, price * 2.0, price * 3.0]);
    assert_ne!(prefix[1], f64::from(0.1f32 + encoded.nbits as f32));
}

#[test]
fn literal_bound_keeps_exact_and_near_ties_and_source_mixed_precision_order() {
    let edge = 8.0 * f32::EPSILON;
    assert!(!literal_bound_exceeded(edge, 0.0, 0.0, 0.0, 6));
    assert!(literal_bound_exceeded(
        f32::from_bits(edge.to_bits() + 1),
        0.0,
        0.0,
        0.0,
        6
    ));
    assert!(!literal_bound_exceeded(8.0, 0.0, 8.0, 0.0, 7));
    assert!(!literal_bound_exceeded(
        f32::from_bits(8.0f32.to_bits() + 1),
        0.0,
        8.0,
        0.0,
        7
    ));
    assert!(literal_bound_exceeded(
        f32::from_bits(8.0f32.to_bits() + 32),
        0.0,
        8.0,
        0.0,
        7
    ));

    // At the index size limit, size_t * epsilon rounds in float before promotion.
    let rounded_edge = 2.0 + 4.0 * f32::EPSILON;
    assert!(!literal_bound_exceeded(
        rounded_edge,
        0.0,
        0.0,
        0.0,
        (1 << 24) + 1
    ));
    assert!(literal_bound_exceeded(
        f32::from_bits(rounded_edge.to_bits() + 1),
        0.0,
        0.0,
        0.0,
        (1 << 24) + 1
    ));

    // Do not reassociate (base + prefix_end) - prefix_begin into a difference.
    let base = 2.0f32.powi(-30);
    let cost = 5.0 * f32::EPSILON + base * 0.5;
    assert!(literal_bound_exceeded(
        cost,
        base,
        2.0f64.powi(30),
        2.0f64.powi(30),
        3
    ));
}

#[test]
fn zero_tail_stops_queries_but_keeps_earlier_copies_ending_in_the_suffix() {
    let input: Vec<_> = (0..14)
        .map(|p| Literal {
            context: u32::from(p < 12),
            value: 7,
        })
        .collect();
    for block_mode in [SourceBlockMode::Endpoints, SourceBlockMode::Intervals] {
        let config = SourceConfig {
            block_mode,
            block_zero_tail: true,
            block_literal_bound: true,
            ..SourceConfig::default()
        };
        let mut costs = Costs::raw(&[&input], &config);
        let row = &mut costs.literals.get_mut(&1).unwrap().bits;
        row.insert(7, 5.0);
        for residual in 0..12 {
            row.insert(
                config.min_symbol() + config.length_uint_config.encode(residual).token,
                0.0,
            );
        }
        costs.distances.bits.insert(0, 0.0);
        let prices = DistancePrices::new(16, 0, &config, |d| costs.distance(d)).unwrap();
        let trace = RefCell::new(Vec::new());
        let find = |p,
                    query: Query<'_>,
                    _: &mut MatchCache,
                    _: &dyn Fn(u32) -> f32,
                    visit: &mut dyn FnMut(Match)| {
            trace.borrow_mut().push((p, query.maximum));
            if p == 4 {
                visit(Match {
                    length: 10,
                    distance: 1,
                    symbol: 0,
                });
            }
        };
        let commands = parse_part(&input, 4, 14, 0, &config, &costs, &prices, &[], &find).unwrap();
        assert_eq!(
            *trace.borrow(),
            (4..12).map(|p| (p, 14 - p)).collect::<Vec<_>>()
        );
        assert_eq!(
            commands,
            [Command::Match {
                context: 1,
                length: 10,
                distance: 1
            }]
        );
        decode(&input, 4, 14, &commands, config.max_match_length);
    }
}

fn model(input: &[Literal], config: &SourceConfig) -> Costs {
    let mut costs = Costs::raw(&[input], config);
    for (&context, row) in &mut costs.literals {
        let literal = match context {
            0 => 0.0,
            1 => 1.125,
            2 => 2.25,
            _ => 0.125,
        };
        for symbol in 0..7 {
            row.bits.insert(symbol, literal);
        }
        let length = match context {
            1 => 100.75,
            2 => 0.75,
            _ => 12.0,
        };
        for residual in 0..=config.max_match_length as u32 {
            row.bits.insert(
                config.min_symbol() + config.length_uint_config.encode(residual).token,
                length,
            );
        }
    }
    for distance in 0..512 {
        costs.distances.bits.insert(
            super::super::HybridUintConfig::default()
                .encode(distance)
                .token,
            0.625,
        );
    }
    costs
}

fn candidates(input: &[Literal], p: usize, end: usize, maximum: usize) -> Vec<Match> {
    let limit = maximum.min(end - p);
    let mut matches = Vec::new();
    for distance in 1..=p.min(window(input.len()) - 1) {
        let length = input[p..p + limit]
            .iter()
            .zip(&input[p - distance..p - distance + limit])
            .take_while(|(a, b)| a.value == b.value)
            .count();
        if length >= MIN_MATCH {
            matches.push(Match {
                length,
                distance,
                symbol: (distance - 1) as u32,
            });
        }
    }
    matches
}

// Independent scalar endpoints: no frontier, cost intervals, pruning or heap.
fn scalar_unpruned(
    input: &[Literal],
    begin: usize,
    end: usize,
    config: &SourceConfig,
    costs: &Costs,
) -> Vec<Command> {
    let mut steps = vec![(f32::INFINITY, 0usize, 0usize); end - begin + 1];
    steps[0].0 = 0.0;
    for (local, token) in input[begin..end].iter().enumerate() {
        let p = begin + local;
        let base = steps[local].0;
        let encoded = config.literal_uint_config.encode(token.value);
        let literal = base + costs.bits(token.context, encoded.token) + encoded.nbits as f32;
        if literal < steps[local + 1].0 {
            steps[local + 1] = (literal, 1, 0);
        }
        let matches = candidates(input, p, end, config.max_match_length);
        // This periodic fixture has no distance-1 match, so no source run skip.
        assert!(matches.iter().all(|m| m.symbol != 0));
        for length in MIN_MATCH..=config.max_match_length.min(end - p) {
            let best = matches
                .iter()
                .filter(|m| m.length >= length)
                .min_by(|a, b| {
                    costs
                        .distance(a.symbol)
                        .total_cmp(&costs.distance(b.symbol))
                        .then(a.symbol.cmp(&b.symbol))
                });
            if let Some(best) = best {
                let cost = base
                    + (costs.length(token.context, length, config) + costs.distance(best.symbol));
                if cost < steps[local + length].0 {
                    steps[local + length] = (cost, length, best.distance);
                }
            }
        }
    }
    let mut commands = Vec::new();
    let mut p = end - begin;
    while p != 0 {
        let (_, length, distance) = steps[p];
        assert!(length != 0);
        p -= length;
        commands.push(if distance == 0 {
            Command::Literal(input[begin + p])
        } else {
            Command::Match {
                context: input[begin + p].context,
                length: length as u32,
                distance: distance as u32,
            }
        });
    }
    commands.reverse();
    commands
}

#[test]
fn pruning_matches_scalar_unpruned_endpoint_oracle_with_fractional_context_prices() {
    let input: Vec<_> = (0..153)
        .map(|p| Literal {
            value: (p % 7) as u32,
            context: if p >= 112 { 0 } else { 1 + (p % 3) as u32 },
        })
        .collect();
    for block_mode in [SourceBlockMode::Endpoints, SourceBlockMode::Intervals] {
        for flags in 0..4 {
            let config = SourceConfig {
                block_mode,
                block_zero_tail: flags & 1 != 0,
                block_literal_bound: flags & 2 != 0,
                max_match_length: 31,
                ..SourceConfig::default()
            };
            let costs = model(&input, &config);
            let prices =
                DistancePrices::new(window(input.len()), 0, &config, |d| costs.distance(d))
                    .unwrap();
            for (begin, end) in [(0, 153), (11, 125), (112, 153)] {
                let find = |p,
                            query: Query<'_>,
                            _: &mut MatchCache,
                            _: &dyn Fn(u32) -> f32,
                            visit: &mut dyn FnMut(Match)| {
                    for matched in
                        candidates(&input, p, end, config.max_match_length.min(query.maximum))
                    {
                        visit(matched);
                    }
                };
                let commands =
                    parse_part(&input, begin, end, 0, &config, &costs, &prices, &[], &find)
                        .unwrap();
                assert_eq!(
                    commands,
                    scalar_unpruned(&input, begin, end, &config, &costs),
                    "flags={flags}, mode={block_mode:?}, begin={begin}"
                );
                decode(&input, begin, end, &commands, config.max_match_length);
            }
        }
    }
}

#[test]
fn pruning_all_intervals_does_not_change_longest_match_or_repetition_skips() {
    let input = vec![
        Literal {
            context: 1,
            value: 0
        };
        97
    ];
    for block_mode in [SourceBlockMode::Endpoints, SourceBlockMode::Intervals] {
        let mut baseline = None;
        for bound in [false, true] {
            let config = SourceConfig {
                block_mode,
                block_literal_bound: bound,
                block_frontier_carry: true,
                block_sparse_cache: true,
                max_match_length: 32,
                ..SourceConfig::default()
            };
            let costs = model(&input, &config);
            let prices = DistancePrices::new(128, 0, &config, |d| costs.distance(d)).unwrap();
            let trace = RefCell::new(Vec::new());
            let find = |p,
                        query: Query<'_>,
                        cache: &mut MatchCache,
                        _: &dyn Fn(u32) -> f32,
                        visit: &mut dyn FnMut(Match)| {
                cache.advance(p);
                assert!(query.carry.is_some());
                trace.borrow_mut().push(p);
                let length = query.maximum.min(32);
                if length >= MIN_MATCH {
                    visit(Match {
                        length,
                        symbol: 0,
                        distance: 1,
                    });
                }
            };
            let commands =
                parse_part(&input, 1, 97, 0, &config, &costs, &prices, &[], &find).unwrap();
            assert!(commands.iter().all(|c| matches!(c, Command::Literal(_))));
            assert_eq!(&trace.borrow()[..9], &[1, 2, 3, 4, 5, 6, 7, 8, 32]);
            let result = (commands, trace.into_inner());
            if let Some(expected) = &baseline {
                assert_eq!(&result, expected);
            } else {
                baseline = Some(result);
            }
        }
    }
}

#[test]
fn lazy_bypasses_both_dp_pruning_controls_without_changing_queries() {
    use super::super::StaticIndex;
    let input: Vec<_> = (0..127)
        .map(|p| Literal {
            value: (p % 7) as u32,
            context: if p >= 91 { 0 } else { 2 },
        })
        .collect();
    let values: Vec<_> = input.iter().map(|t| t.value).collect();
    let mut baseline = None;
    for flags in 0..4 {
        let config = SourceConfig {
            block_mode: SourceBlockMode::Intervals,
            block_lazy: true,
            block_zero_tail: flags & 1 != 0,
            block_literal_bound: flags & 2 != 0,
            max_match_length: 31,
            ..SourceConfig::default()
        };
        let costs = model(&input, &config);
        let prices = DistancePrices::new(128, 0, &config, |d| costs.distance(d)).unwrap();
        let index = StaticIndex::new(&values, 0, &config).unwrap();
        let trace = RefCell::new(Vec::new());
        let find = |p,
                    query: Query<'_>,
                    cache: &mut MatchCache,
                    price: &dyn Fn(u32) -> f32,
                    visit: &mut dyn FnMut(Match)| {
            trace.borrow_mut().push((p, query.maximum));
            index.find_query(p, query, price, visit, cache);
        };
        let commands = parse_part(&input, 0, 127, 0, &config, &costs, &prices, &[], &find).unwrap();
        decode(&input, 0, 127, &commands, config.max_match_length);
        let result = (commands, trace.into_inner());
        if let Some(expected) = &baseline {
            assert_eq!(&result, expected);
        } else {
            baseline = Some(result);
        }
    }
}
