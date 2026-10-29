// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;

#[derive(Clone)]
struct WideRing<const N: usize> {
    positions: [usize; N],
    next: usize,
    size: usize,
}

impl<const N: usize> WideRing<N> {
    fn new() -> Self {
        Self {
            positions: [0; N],
            next: 0,
            size: 0,
        }
    }

    fn insert(&mut self, position: usize) {
        self.positions[self.next] = position;
        self.next = (self.next + 1) % N;
        self.size = (self.size + 1).min(N);
    }

    fn check(&self, ring: &Ring<N>) {
        assert_eq!(self.positions, ring.positions.map(|p| p as usize));
        assert_eq!(
            (self.next, self.size),
            (ring.next as usize, ring.size as usize)
        );
    }
}

fn check_compact_ring<const N: usize>() {
    for base in [0, u32::MAX as usize - (8 * N + 8)] {
        for filled in [0, 1, N - 1, N, N + 1, 2 * N] {
            for count in [0, 1, N - 1, N, N + 1, 3 * N + 5] {
                let mut expected = WideRing::<N>::new();
                let mut actual = Ring::<N>::new();
                for p in base..base + filled {
                    expected.insert(p);
                    actual.insert(p);
                    expected.check(&actual);
                }
                actual.repeated(base + filled, base + filled + count);
                for p in base + filled..base + filled + count {
                    expected.insert(p);
                }
                expected.check(&actual);
                expected.insert(base + filled + count);
                actual.insert(base + filled + count);
                expected.check(&actual);
            }
        }
    }
}

#[test]
fn compact_ring_layout_and_rotation_match_usize_reference() {
    assert_eq!(std::mem::size_of::<Ring<7>>(), 32);
    assert_eq!(std::mem::size_of::<Ring<31>>(), 128);
    assert_eq!(std::mem::size_of::<Ring<128>>(), 516);
    check_compact_ring::<1>();
    check_compact_ring::<3>();
    check_compact_ring::<7>();
    check_compact_ring::<15>();
    check_compact_ring::<31>();
    check_compact_ring::<128>();
    check_compact_ring::<255>();
}

#[test]
fn compact_ring_preserves_maximum_position_and_rejects_oversized_input() {
    for len in [0, 1, i32::MAX as usize, u32::MAX as usize] {
        assert!(validate_ring_input_len(len).is_ok());
    }
    let maximum = u32::MAX as usize;
    let mut expected = WideRing::<7>::new();
    let mut actual = Ring::<7>::new();
    for p in maximum - 20..=maximum {
        expected.insert(p);
        actual.insert(p);
        expected.check(&actual);
    }
    if let Some(end) = maximum.checked_add(1) {
        assert!(matches!(validate_ring_input_len(end), Err(Error::Limit(_))));
        assert!(matches!(
            validate_ring_input_len(usize::MAX),
            Err(Error::Limit(_))
        ));
        let mut bulk = Ring::<7>::new();
        bulk.repeated(maximum - 20, end);
        expected.check(&bulk);
    }
}

#[test]
fn compact_ring_bulk_rotation_handles_a_full_u32_span() {
    let end = u32::MAX as usize;
    let mut actual = Ring::<31>::new();
    actual.insert(0);
    actual.repeated(1, end);
    assert_eq!(actual.next as usize, end % 31);
    assert_eq!(actual.size, 31);
    for i in 0..31 {
        let slot = (actual.next as usize + i) % 31;
        assert_eq!(actual.positions[slot] as usize, end - 31 + i);
    }
}

#[test]
fn short128_compact_ring_keeps_newest_first_and_oldest_boundary() {
    let mut buckets = ShortBuckets::Large(vec![Ring::<128>::new()]);
    let mut expected = WideRing::<128>::new();
    let mut visited = Vec::new();
    assert_eq!(buckets.visit(0, false, |p| visited.push(p)), 0);
    assert!(visited.is_empty());
    let maximum = u32::MAX as usize;
    for p in maximum - 300..=maximum {
        buckets.insert(p, 0);
        expected.insert(p);
        visited.clear();
        let oldest = buckets.visit(0, false, |p| visited.push(p));
        let order: Vec<_> = (0..expected.size)
            .map(|i| expected.positions[expected.next.wrapping_sub(i + 1) & 127])
            .collect();
        assert_eq!(visited, order);
        assert_eq!(
            oldest,
            if expected.size == 128 {
                expected.positions[expected.next]
            } else {
                0
            }
        );
    }
}

fn check_scratch<const N: usize>(values: &[u32], width: usize, config: &SourceConfig) {
    let mut reused = Fast::<N>::new(values, width, config).unwrap();
    let mut cleared = Fast::<N>::new(values, width, config).unwrap();
    for matcher in [&mut reused, &mut cleared] {
        matcher
            .prefix
            .record_candidate_work(matcher.prefix.activation_threshold());
    }
    let mut position = 0;
    while position < values.len() {
        reused.candidates.fill((usize::MAX, u32::MAX));
        cleared.candidates.fill((0, 0));
        let actual = reused.find(position);
        let expected = cleared.find(position);
        assert_eq!(
            (actual.length, actual.distance, actual.symbol),
            (expected.length, expected.distance, expected.symbol),
            "N={N} width={width} position={position}"
        );
        if N <= 3 || width == 0 {
            assert_eq!(reused.candidates[119], (usize::MAX, u32::MAX));
        }
        let next = (position + 1 + position % 17).min(values.len());
        reused.update_range(position + 1, next);
        cleared.update_range(position + 1, next);
        position = next;
    }
    reused.update_range(values.len(), values.len());
    cleared.update_range(values.len(), values.len());
    for (actual, expected) in reused.buckets.iter().zip(&cleared.buckets) {
        assert_eq!(actual.positions, expected.positions);
        assert_eq!((actual.next, actual.size), (expected.next, expected.size));
    }
}

#[test]
fn fast_scratch_handles_empty_inputs_and_incomplete_tails() {
    for len in 0..=5 {
        for width in [0, 1, 37] {
            check_scratch::<1>(&[u32::MAX; 5][..len], width, &SourceConfig::default());
            check_scratch::<7>(&[0; 5][..len], width, &SourceConfig::default());
        }
    }
}

#[test]
fn fast_scratch_skipped_ranges_equal_a_fresh_query_buffer() {
    let mut values = vec![0; 137];
    values.extend((0..193).map(|i| if i % 31 < 9 { u32::MAX } else { i % 17 }));
    values.extend_from_within(7..104);
    values.extend([u32::MAX; 79]);
    values.extend([0, u32::MAX]);
    for fast_ranges in [SourceFastRanges::Disabled, SourceFastRanges::Both] {
        for fast_prefix in [
            SourceFastPrefix::Disabled,
            SourceFastPrefix::Scalar,
            SourceFastPrefix::Simd,
        ] {
            for range_search in [false, true] {
                for width in [0, 1, 37, usize::MAX] {
                    let config = SourceConfig {
                        max_match_length: 47,
                        fast_ranges,
                        fast_prefix: Some(fast_prefix),
                        range_search,
                        ..SourceConfig::default()
                    };
                    check_scratch::<1>(&values, width, &config);
                    check_scratch::<3>(&values, width, &config);
                    check_scratch::<7>(&values, width, &config);
                    check_scratch::<31>(&values, width, &config);
                }
            }
        }
    }
}

fn reference_consider(
    values: &[u32],
    position: usize,
    limit: usize,
    distance: usize,
    symbol: u32,
    best: &mut Match,
) {
    let previous = position - distance;
    let required = if best.length == 0 {
        MIN_MATCH
    } else {
        best.length + usize::from(symbol >= best.symbol)
    };
    if required > limit
        || values[position] != values[previous]
        || values[position + required - 1] != values[previous + required - 1]
    {
        return;
    }
    let mut length = 0;
    while length < limit && values[position + length] == values[previous + length] {
        length += 1;
    }
    if length >= required {
        *best = Match {
            length,
            distance,
            symbol,
        };
    }
}

fn match_tuple(matched: Match) -> (usize, usize, u32) {
    (matched.length, matched.distance, matched.symbol)
}

#[test]
fn mandatory_prefix_rejection_does_not_resolve_distance_symbols() {
    for ranges in [SourceFastRanges::Disabled, SourceFastRanges::Both] {
        for mismatch in 0..MIN_MATCH {
            let mut values = [7, 8, 9, 10, 7, 8, 9, 10];
            values[4 + mismatch] ^= u32::MAX;
            for length in [0, 3, 4] {
                let mut best = Match {
                    length,
                    distance: 1,
                    symbol: 19,
                };
                let before = match_tuple(best);
                consider_fast_match(&values, 4, 4, 4, ranges, &mut best, || {
                    panic!("prefix mismatch must precede symbol lookup")
                });
                assert_eq!(match_tuple(best), before);
            }
        }
        for limit in 0..MIN_MATCH {
            let values = vec![7; limit + 1];
            let mut best = Match::default();
            consider_fast_match(&values, 1, limit, 1, ranges, &mut best, || {
                panic!("short tail must not resolve a distance symbol")
            });
            assert_eq!(match_tuple(best), match_tuple(Match::default()));
        }
    }
}

#[test]
fn mandatory_prefix_detects_middle_only_hash_collisions() {
    let mut seen = vec![None; 1 << 14];
    let (a, b) = (0..=16384u32)
        .find_map(|middle| {
            let prefix = [17, middle, 23];
            let key = hash(&prefix, 0) as usize & 16383;
            if let Some(previous) = seen[key] {
                Some(([17, previous, 23], prefix))
            } else {
                seen[key] = Some(middle);
                None
            }
        })
        .unwrap();
    assert_ne!(a[1], b[1]);
    assert_eq!(hash(&a, 0) & 16383, hash(&b, 0) & 16383);
    let values = [a, b].concat();
    for ranges in [SourceFastRanges::Disabled, SourceFastRanges::Both] {
        let mut best = Match::default();
        consider_fast_match(&values, 3, 3, 3, ranges, &mut best, || {
            panic!("equal first/third values do not prove the full prefix")
        });
        assert_eq!(match_tuple(best), match_tuple(Match::default()));
        let config = SourceConfig {
            fast_ranges: ranges,
            ..SourceConfig::default()
        };
        let mut matcher = Fast::<1>::new(&values, 0, &config).unwrap();
        matcher.find(0);
        assert_eq!(match_tuple(matcher.find(3)), match_tuple(Match::default()));
    }
}

#[test]
fn deferred_symbols_preserve_length_caps_and_all_symbol_ties() {
    let prefix: Vec<_> = (0..64u32).map(|i| u32::MAX - i * 371).collect();
    for mismatch in 0..=40 {
        let mut values = [prefix.clone(), prefix.clone()].concat();
        values[64 + mismatch] ^= 0x1357;
        for limit in [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 16, 17, 31, 32, 33, 40, 64] {
            for length in [0, 3, 4, 8, 16, 32, 64] {
                for incumbent in [0, 119, 120, 999] {
                    for symbol in [0, 1, 119, 120, 999, WINDOW_SIZE as u32 + 118] {
                        let before = Match {
                            length,
                            distance: 1,
                            symbol: incumbent,
                        };
                        let mut expected = before;
                        reference_consider(&values, 64, limit, 64, symbol, &mut expected);
                        for ranges in [SourceFastRanges::Disabled, SourceFastRanges::Both] {
                            let mut actual = before;
                            consider_fast_match(
                                &values,
                                64,
                                limit,
                                64,
                                ranges,
                                &mut actual,
                                || symbol,
                            );
                            assert_eq!(
                                match_tuple(actual),
                                match_tuple(expected),
                                "mismatch={mismatch} cap={limit} best={length}/{incumbent} symbol={symbol} ranges={ranges:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn mandatory_prefix_extension_preserves_overlap_and_long_matches() {
    for distance in [1, 2, 3, 7, 17] {
        let position = distance + 29;
        let base: Vec<_> = (0..position + 513)
            .map(|i| (i % distance) as u32 * 31337)
            .collect();
        for mismatch in [3, 4, 7, 8, 9, 31, 127, 128, 129, 511, 512] {
            let mut values = base.clone();
            values[position + mismatch] ^= u32::MAX;
            for limit in [3, 4, 8, 128, 129, 512, 513] {
                for symbol in [0, 1, 119, 120] {
                    let mut expected = Match::default();
                    reference_consider(&values, position, limit, distance, symbol, &mut expected);
                    for ranges in [SourceFastRanges::Disabled, SourceFastRanges::Both] {
                        let mut actual = Match::default();
                        let mut lookups = 0;
                        consider_fast_match(
                            &values,
                            position,
                            limit,
                            distance,
                            ranges,
                            &mut actual,
                            || {
                                lookups += 1;
                                symbol
                            },
                        );
                        assert_eq!(lookups, 1);
                        assert_eq!(match_tuple(actual), match_tuple(expected));
                    }
                }
            }
        }
    }
}
