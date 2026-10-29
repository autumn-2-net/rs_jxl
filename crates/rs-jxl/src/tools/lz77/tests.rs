// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; the complete upstream notice and license
// are retained in docs/PORTING_KERNELS.md.

use super::*;

const MODES: [Lz77Mode; 3] = [Lz77Mode::Rle, Lz77Mode::Greedy, Lz77Mode::Lazy];

fn input(values: &[u32]) -> Vec<Literal> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| Literal {
            context: (i % 5) as u32,
            value,
        })
        .collect()
}

fn test_options(mode: Lz77Mode) -> Lz77Options {
    Lz77Options {
        mode,
        bucket_bits: 4,
        require_global_savings: false,
        ..Lz77Options::default()
    }
}

// Independent command expansion: copy one already-decoded value at a
// time, so matches longer than their distance exercise actual overlap.
fn expand(commands: &[Command], original: &[Literal], options: &Lz77Options) -> Vec<u32> {
    let mut values = Vec::new();
    for &command in commands {
        let position = values.len();
        match command {
            Command::Literal(literal) => {
                assert_eq!(literal, original[position]);
                values.push(literal.value);
            }
            Command::Match {
                context,
                length,
                distance,
            } => {
                let length = length as usize;
                let distance = distance as usize;
                assert_eq!(context, original[position].context);
                assert!((MIN_MATCH..=options.max_match_length).contains(&length));
                assert!((1..=options.window_size.min(WINDOW_SIZE)).contains(&distance));
                assert!(distance <= position);
                assert!(length <= original.len() - position);
                for _ in 0..length {
                    values.push(values[values.len() - distance]);
                }
            }
        }
    }
    assert_eq!(values, original.iter().map(|l| l.value).collect::<Vec<_>>());
    values
}

fn has_match(commands: &[Command]) -> bool {
    commands
        .iter()
        .any(|command| matches!(command, Command::Match { .. }))
}

#[test]
fn canonical_distance_vectors_and_all_120_codes() {
    for (distance, width, symbol) in [
        (1, 0, 0),
        (2, 0, 1),
        (1, 1, 0),
        (2, 1, 2),
        (1, 2, 1),
        (2, 2, 0),
        (3, 2, 2),
        (3, 3, 0),
        (257, 257, 0),
        (1, 257, 1),
        (258, 257, 2),
        (256, 257, 3),
        (514, 257, 4),
        (2, 257, 5),
        (8, 257, 96),
        (1807, 257, 119),
        (100, 257, 219),
        (1, usize::MAX, 1),
        (8, usize::MAX, 96),
    ] {
        assert_eq!(distance_symbol(distance, width).unwrap(), symbol);
    }
    for width in [1usize, 2, 3, 7, 8, 16, 257, WINDOW_SIZE, usize::MAX] {
        let mut canonical = BTreeMap::new();
        for (symbol, &(x, y)) in SPECIAL_DISTANCES.iter().enumerate() {
            let signed = width as i128 * i128::from(y) + i128::from(x);
            let distance = if signed <= 1 { 1 } else { signed };
            if distance <= WINDOW_SIZE as i128 {
                canonical.entry(distance as usize).or_insert(symbol as u32);
            }
        }
        let codes = DistanceCodes::new(width, WINDOW_SIZE);
        for (distance, symbol) in canonical {
            assert_eq!(distance_symbol(distance, width).unwrap(), symbol);
            assert_eq!(codes.symbol(distance), symbol);
        }
    }
    // With width 257 every table entry denotes a distinct physical distance.
    for (symbol, &(x, y)) in SPECIAL_DISTANCES.iter().enumerate() {
        let distance = (i32::from(x) + 257 * i32::from(y)) as usize;
        assert_eq!(distance_symbol(distance, 257).unwrap(), symbol as u32);
    }
    assert_eq!(
        distance_symbol(WINDOW_SIZE, 0).unwrap(),
        WINDOW_SIZE as u32 - 1
    );
    assert_eq!(distance_symbol(WINDOW_SIZE, WINDOW_SIZE).unwrap(), 0);
    for distance in [0, WINDOW_SIZE + 1, usize::MAX] {
        assert!(distance_symbol(distance, 0).is_err());
        assert!(distance_symbol(distance, usize::MAX).is_err());
    }
}

#[test]
fn empty_short_inputs_and_two_token_tails() {
    let values = [
        u32::MAX,
        u32::MAX - 1,
        u32::MAX - 2,
        u32::MAX,
        u32::MAX - 1,
        u32::MAX - 2,
        79,
        83,
    ];
    for mode in MODES {
        let options = test_options(mode);
        for length in 0..=values.len() {
            let original = input(&values[..length]);
            let commands = parse(&original, 1, &options).unwrap();
            expand(&commands, &original, &options);
            if length <= 3 {
                assert_eq!(commands, literals(&original).unwrap());
            }
        }
    }
    for mode in [Lz77Mode::Greedy, Lz77Mode::Lazy] {
        let original = input(&values);
        let options = test_options(mode);
        let commands = parse(&original, 0, &options).unwrap();
        assert!(commands.contains(&Command::Match {
            context: 3,
            length: 3,
            distance: 3
        }));
        assert_eq!(
            commands[commands.len() - 2..],
            [Command::Literal(original[6]), Command::Literal(original[7]),]
        );
    }
}

#[test]
fn overlapping_runs_and_periods_preserve_first_context_only() {
    for mode in MODES {
        for width in [0, 1, 2, 3, 17, usize::MAX] {
            let options = test_options(mode);
            let original: Vec<_> = (0..1025)
                .map(|i| Literal {
                    context: u32::MAX - i,
                    value: u32::MAX,
                })
                .collect();
            let commands = parse(&original, width, &options).unwrap();
            expand(&commands, &original, &options);
            assert!(commands.iter().any(|command| matches!(command,
                Command::Match { length: 1024, distance: 1, context } if *context == u32::MAX - 1
            )));
        }
    }
    for period in [2, 3, 7, 32] {
        let original = input(
            &(0..4099)
                .map(|i| u32::MAX - (i % period))
                .collect::<Vec<_>>(),
        );
        for mode in [Lz77Mode::Greedy, Lz77Mode::Lazy] {
            let options = test_options(mode);
            let commands = parse(&original, period as usize, &options).unwrap();
            expand(&commands, &original, &options);
            assert!(commands.iter().any(|command| matches!(command,
                Command::Match { length, distance, .. } if *length > *distance && *distance == period
            )));
        }
    }
}

#[test]
fn length_limit_splits_matches_but_does_not_limit_history() {
    for maximum in [3, 4, 7, 16, 255, 256] {
        let original = input(&vec![u32::MAX; 1029]);
        for mode in MODES {
            let options = Lz77Options {
                max_match_length: maximum,
                ..test_options(mode)
            };
            let commands = parse(&original, 1, &options).unwrap();
            expand(&commands, &original, &options);
            assert!(has_match(&commands));
        }
    }
    let values: Vec<_> = (0..19).chain(0..19).map(|v| u32::MAX - v).collect();
    let original = input(&values);
    let options = Lz77Options {
        max_match_length: 3,
        ..test_options(Lz77Mode::Greedy)
    };
    let commands = parse(&original, 19, &options).unwrap();
    expand(&commands, &original, &options);
    assert!(commands.iter().any(|c| matches!(
        c,
        Command::Match {
            distance: 19,
            length: 3,
            ..
        }
    )));
}

#[test]
fn entropy_checks_and_global_gate_are_real() {
    for mode in MODES {
        let options = test_options(mode);
        let zeros = input(&vec![0; 4099]);
        let commands = parse(&zeros, 0, &options).unwrap();
        assert!(!has_match(&commands));
        expand(&commands, &zeros, &options);

        let original = vec![
            Literal {
                context: 0,
                value: 100
            };
            5
        ];
        let commands = parse(&original, 0, &options).unwrap();
        assert!(has_match(&commands));
        expand(&commands, &original, &options);
        let gated = Lz77Options {
            require_global_savings: true,
            ..options.clone()
        };
        assert!(!has_match(&parse(&original, 0, &gated).unwrap()));
        let direct_tokens = Lz77Options {
            literal_uint_config: HybridUintConfig::new(15, 0, 0).unwrap(),
            ..options
        };
        assert!(!has_match(&parse(&original, 0, &direct_tokens).unwrap()));
    }
    let original = [
        Literal {
            context: u32::MAX,
            value: 0,
        },
        Literal {
            context: u32::MAX,
            value: 0,
        },
        Literal {
            context: u32::MAX,
            value: 1,
        },
    ];
    let ans = SymbolCosts::new(&original, &test_options(Lz77Mode::Lazy));
    let huffman = SymbolCosts::new(
        &original,
        &Lz77Options {
            force_huffman: true,
            ..test_options(Lz77Mode::Lazy)
        },
    );
    assert!(ans.bits(u32::MAX, 0) < 1.0);
    assert_eq!(huffman.bits(u32::MAX, 0), 1.0);
    assert_eq!(huffman.bits(u32::MAX, 1), 2.0);
    assert_eq!(ans.bits(0, 0), 12.0);
}

#[test]
fn copy_length_is_independent_of_history_window_and_calls_start_fresh() {
    let original = input(&vec![u32::MAX; 4096]);
    for mode in MODES {
        let options = Lz77Options {
            mode,
            window_size: 1,
            bucket_bits: 4,
            ..Lz77Options::default()
        };
        for _ in 0..2 {
            let commands = parse(&original, 1, &options).unwrap();
            assert_eq!(
                commands,
                [
                    Command::Literal(original[0]),
                    Command::Match {
                        context: original[1].context,
                        length: 4095,
                        distance: 1
                    },
                ]
            );
            expand(&commands, &original, &options);
        }
    }
}

#[test]
fn lazy_lookahead_uses_longer_next_match_without_double_inserting() {
    let values: Vec<_> = [
        11, 12, 13, 99, 12, 13, 14, 15, 16, 17, 98, 11, 12, 13, 14, 15, 16, 17,
    ]
    .iter()
    .map(|&v| 0xf000_0000 + v)
    .collect();
    let original = input(&values);
    let greedy = test_options(Lz77Mode::Greedy);
    let lazy = test_options(Lz77Mode::Lazy);
    let a = parse(&original, 0, &greedy).unwrap();
    let b = parse(&original, 0, &lazy).unwrap();
    expand(&a, &original, &greedy);
    expand(&b, &original, &lazy);
    assert!(matches!(
        a[11],
        Command::Match {
            length: 3,
            distance: 11,
            ..
        }
    ));
    assert_eq!(b[11], Command::Literal(original[11]));
    assert!(matches!(
        b[12],
        Command::Match {
            length: 6,
            distance: 8,
            ..
        }
    ));
}

#[test]
fn decoder_window_inclusive_and_expired_spatial_candidates() {
    for window in [1, 2, 16, WINDOW_SIZE] {
        for extra in [0, 1] {
            let distance = window + extra;
            let mut values = vec![0; distance + 3];
            values[0] = u32::MAX;
            // Distance one/two deliberately exercise overlapping patterns.
            for i in 0..3 {
                values[distance + i] = values[i];
            }
            let options = Lz77Options {
                window_size: window,
                bucket_capacity: 1,
                bucket_bits: 0,
                ..test_options(Lz77Mode::Greedy)
            };
            let mut matcher = Matcher::<1>::new(&values, distance, &options).unwrap();
            matcher.update_range(distance);
            let found = matcher.find(distance);
            if extra == 0 {
                assert_eq!(found.distance, distance);
                assert_eq!(found.length, 3);
            } else {
                assert_eq!(found.length, 0);
            }
        }
    }
}

#[test]
fn fingerprint_collision_is_not_an_exact_match() {
    let mut seen = [None; 256];
    let (a, b) = (0..=256u32)
        .find_map(|middle| {
            let hash = prefix_hash(&[700, middle, 900], 0).unwrap();
            seen[(hash >> 24) as usize]
                .replace(middle)
                .map(|previous| (previous, middle))
        })
        .unwrap();
    let values = [700, a, 900, 700, b, 900];
    assert_eq!(
        prefix_hash(&values, 0).unwrap() >> 24,
        prefix_hash(&values, 3).unwrap() >> 24
    );
    for policy in [
        FilterPolicy::Disabled,
        FilterPolicy::Eager,
        FilterPolicy::Lazy,
    ] {
        let options = Lz77Options {
            bucket_bits: 0,
            bucket_capacity: 4,
            prefix_filter: policy,
            ..test_options(Lz77Mode::Greedy)
        };
        let mut matcher = Matcher::<4>::new(&values, 3, &options).unwrap();
        matcher.update_range(3);
        assert_eq!(matcher.find(3).length, 0);
        let original = input(&values);
        expand(&parse(&original, 3, &options).unwrap(), &original, &options);
    }
}

fn check_matcher<const N: usize>() {
    let mut random = 0x1234_5678u32;
    let values: Vec<_> = (0..385)
        .map(|i| {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if i % 57 < 23 {
                (i % 7) as u32
            } else {
                random >> 29
            }
        })
        .collect();
    for width in [0usize, 1, 2, 3, 7, 16, 257] {
        for policy in [
            FilterPolicy::Disabled,
            FilterPolicy::Eager,
            FilterPolicy::Lazy,
        ] {
            for range_search in [false, true] {
                let options = Lz77Options {
                    bucket_capacity: N,
                    bucket_bits: 2,
                    window_size: 127,
                    prefix_filter: policy,
                    range_search,
                    max_match_length: 37,
                    ..test_options(Lz77Mode::Greedy)
                };
                let mut matcher = Matcher::<N>::new(&values, width, &options).unwrap();
                let mut history = vec![Vec::new(); 4];
                let symbols: Vec<_> = (1..=options.window_size)
                    .map(|distance| distance_symbol(distance, width).unwrap())
                    .collect();
                for position in 0..values.len() {
                    let hash = prefix_hash(&values, position);
                    let mut expected = Match::default();
                    if let Some(hash) = hash {
                        let bucket = &mut history[hash as usize & 3];
                        // Exhaustive scalar reference over retained positions
                        // plus every legal special distance, without filters,
                        // range pruning, or endpoint rejection shortcuts.
                        for previous in position.saturating_sub(options.window_size)..position {
                            let distance = position - previous;
                            let symbol = symbols[distance - 1];
                            if !bucket.contains(&previous) && !(width != 0 && symbol < 120) {
                                continue;
                            }
                            let length = values[position..]
                                .iter()
                                .zip(&values[previous..])
                                .take(options.max_match_length)
                                .take_while(|(a, b)| a == b)
                                .count();
                            if length >= MIN_MATCH
                                && (length > expected.length
                                    || (length == expected.length && symbol < expected.symbol))
                            {
                                expected = Match {
                                    length,
                                    distance,
                                    symbol,
                                };
                            }
                        }
                        bucket.push(position);
                        if bucket.len() > N {
                            bucket.remove(0);
                        }
                    }
                    assert_eq!(
                        matcher.find(position),
                        expected,
                        "N={N}, width={width}, policy={policy:?}, range={range_search}, pos={position}"
                    );
                }
            }
        }
    }
}

#[test]
fn recent_bucket_matcher_matches_exhaustive_reference() {
    check_matcher::<1>();
    check_matcher::<4>();
    check_matcher::<32>();
    check_matcher::<64>();
}

fn check_range_updates<const N: usize>() {
    let values: Vec<_> = (0..523)
        .map(|i| if i % 97 < 81 { 0 } else { (i % 5) as u32 })
        .collect();
    let options = Lz77Options {
        bucket_capacity: N,
        bucket_bits: 1,
        prefix_filter: FilterPolicy::Eager,
        ..test_options(Lz77Mode::Greedy)
    };
    for end in [0, 1, 2, 3, 31, 63, 80, 81, 96, 97, 255, 521, 522, 523] {
        let mut batched = Matcher::<N>::new(&values, 3, &options).unwrap();
        let mut scalar = Matcher::<N>::new(&values, 3, &options).unwrap();
        batched.update_range(end);
        for position in 0..end {
            scalar.insert(position);
        }
        scalar.next_position = end;
        for (a, b) in batched.buckets.iter_mut().zip(&mut scalar.buckets) {
            let mut left = Vec::new();
            let mut right = Vec::new();
            a.visit(None, MIN_MATCH, |p| left.push(p));
            b.visit(None, MIN_MATCH, |p| right.push(p));
            assert_eq!(left, right, "N={N}, end={end}");
            assert_eq!(a.oldest_if_full(), b.oldest_if_full());
        }
        for position in end..values.len() {
            assert_eq!(batched.find(position), scalar.find(position));
        }
    }
}

#[test]
fn repeated_range_updates_preserve_logical_ring_and_tails() {
    check_range_updates::<1>();
    check_range_updates::<2>();
    check_range_updates::<8>();
    check_range_updates::<64>();
}

#[test]
fn deterministic_roundtrips_across_modes_buckets_filters_and_widths() {
    let mut random = 1u32;
    let values: Vec<_> = (0..2071)
        .map(|i| {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match i % 101 {
                0..=30 => 0,
                31..=60 => u32::MAX - (i % 7) as u32,
                _ => random,
            }
        })
        .collect();
    let original = input(&values);
    for mode in MODES {
        for capacity in [1, 2, 4, 8, 16, 32, 64] {
            for width in [0, 1, 2, 3, 16, 17, usize::MAX] {
                let options = Lz77Options {
                    bucket_capacity: capacity,
                    ..test_options(mode)
                };
                let expected = parse(&original, width, &options).unwrap();
                expand(&expected, &original, &options);
                for policy in [FilterPolicy::Disabled, FilterPolicy::Eager] {
                    for range_search in [false, true] {
                        let options = Lz77Options {
                            prefix_filter: policy,
                            range_search,
                            ..options.clone()
                        };
                        let actual = parse(&original, width, &options).unwrap();
                        assert_eq!(actual, expected);
                        expand(&actual, &original, &options);
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_options_are_errors_even_for_empty_input() {
    let defaults = Lz77Options::default();
    for window_size in [0, WINDOW_SIZE + 1, usize::MAX] {
        assert!(
            parse(
                &[],
                0,
                &Lz77Options {
                    window_size,
                    ..defaults.clone()
                }
            )
            .is_err()
        );
    }
    for max_match_length in [0, 1, 2] {
        assert!(
            parse(
                &[],
                0,
                &Lz77Options {
                    max_match_length,
                    ..defaults.clone()
                }
            )
            .is_err()
        );
    }
    if let Some(max_match_length) = (u32::MAX as usize).checked_add(1) {
        assert!(
            parse(
                &[],
                0,
                &Lz77Options {
                    max_match_length,
                    ..defaults.clone()
                }
            )
            .is_err()
        );
    }
    for bucket_capacity in [0, 3, 7, 65, 128, usize::MAX] {
        assert!(
            parse(
                &[],
                0,
                &Lz77Options {
                    bucket_capacity,
                    ..defaults.clone()
                }
            )
            .is_err()
        );
    }
    for bucket_bits in [21, 32, u32::MAX] {
        assert!(
            parse(
                &[],
                0,
                &Lz77Options {
                    bucket_bits,
                    ..defaults.clone()
                }
            )
            .is_err()
        );
    }
}

#[test]
fn current_group_wire_configs_cover_full_width_residuals_and_distances() {
    let literal = HybridUintConfig::default();
    let length = HybridUintConfig::new(4, 0, 0).unwrap();
    let distance = HybridUintConfig::new(0, 0, 0).unwrap();
    for value in [0, 1, 65_535, 65_536, 131_071, 0x8000_0000, u32::MAX] {
        assert!(literal.encode(value).token < 224);
    }
    for width in [0, 1, 2, 256, usize::MAX] {
        for physical in [1, 2, 119, 120, WINDOW_SIZE] {
            let symbol = distance_symbol(physical, width).unwrap();
            assert!(distance.encode(symbol).token < 256);
        }
    }
    for channels in 1..=4u32 {
        // Even a copy covering the entire maximum-sized current group
        // (an upper bound) fits the caller's fixed length alphabet.
        let length_minus_three = 256 * 256 * channels - MIN_MATCH as u32;
        assert!(224 + length.encode(length_minus_three).token < 256);
    }
    assert_eq!(224 + length.encode((1 << 20) - 1).token, 255);
    assert_eq!(224 + length.encode(1 << 20).token, 256);
}
