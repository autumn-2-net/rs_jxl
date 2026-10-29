use super::super::search::{fast_match_length, match_length};
use super::*;

fn scalar_length(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn tuple(matched: Match) -> (usize, usize, u32) {
    (matched.length, matched.distance, matched.symbol)
}

#[test]
fn short_head_finds_every_mismatch_bit_without_narrowing_values() {
    for length in 0..=17 {
        let a: Vec<_> = (0..length)
            .map(|i| [0, u32::MAX, 0x8000_0000, 0x0001_0000][i % 4])
            .collect();
        let mut b = a.clone();
        assert_eq!(fast_match_length(&a, &b), length);
        for position in 0..length {
            for bit in 0..32 {
                b[position] ^= 1u32 << bit;
                assert_eq!(fast_match_length(&a, &b), position);
                if position + 1 < length {
                    b[position + 1] ^= u32::MAX;
                    assert_eq!(fast_match_length(&a, &b), position);
                    b[position + 1] ^= u32::MAX;
                }
                b[position] ^= 1u32 << bit;
            }
        }
    }
}

#[test]
fn short_head_preserves_alignment_tails_and_long_block_boundaries() {
    for length in (0..=33).chain([63, 64, 65, 127, 128, 129, 255, 256, 257]) {
        for left_offset in 0..4 {
            for right_offset in 0..4 {
                let mut a = vec![0x89ab_cdef; left_offset + length + 1];
                for (i, value) in a[left_offset..left_offset + length].iter_mut().enumerate() {
                    *value = (i as u32).wrapping_mul(0x9e37_79b9);
                }
                let a = &a[left_offset..left_offset + length];
                let mut b = vec![0x7654_3210; right_offset + length + 1];
                b[right_offset..right_offset + length].copy_from_slice(a);
                assert_eq!(
                    fast_match_length(a, &b[right_offset..right_offset + length]),
                    length
                );
                for mismatch in 0..length {
                    b[right_offset + mismatch] ^= 0x8000_0001;
                    let b_slice = &b[right_offset..right_offset + length];
                    assert_eq!(fast_match_length(a, b_slice), mismatch);
                    assert_eq!(fast_match_length(a, b_slice), match_length(a, b_slice));
                    b[right_offset + mismatch] ^= 0x8000_0001;
                }
            }
        }
    }
}

#[test]
fn short_head_preserves_overlapping_periodic_and_long_constant_matches() {
    for distance in [1, 2, 3, 7, 16, 31] {
        let position = distance + 5;
        for length in [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 127, 128, 129, 4097] {
            let mut values: Vec<_> = (0..position + length)
                .map(|i| u32::MAX.wrapping_sub((i % distance) as u32 * 31337))
                .collect();
            let check = |values: &[u32]| {
                let a = &values[position..position + length];
                let b = &values[position - distance..position - distance + length];
                let expected = scalar_length(a, b);
                assert_eq!(fast_match_length(a, b), expected);
                assert_eq!(match_length(a, b), expected);
            };
            check(&values);
            for mismatch in [0, 1, 6, 7, 8, 9, 15, 16, 17, 127, length.saturating_sub(1)] {
                if mismatch < length {
                    values[position + mismatch] ^= 0x8000_0001;
                    check(&values);
                    values[position + mismatch] ^= 0x8000_0001;
                }
            }
        }
    }
}

#[test]
fn short_extension_preserves_length_caps_and_incumbent_symbol_ties() {
    let prefix: Vec<_> = (0..64u32).map(|i| u32::MAX - i * 371).collect();
    for mismatch in 0..=64 {
        let mut values = [prefix.clone(), prefix.clone()].concat();
        if mismatch < 64 {
            values[64 + mismatch] ^= 0x8000_0001;
        }
        for limit in [
            0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 15, 16, 17, 31, 32, 33, 64,
        ] {
            let length = scalar_length(&values[64..64 + limit], &values[..limit]);
            for incumbent in [0, 3, 4, 7, 10, 11, 12, 64] {
                for symbol in [0, 119, 120] {
                    let before = Match {
                        length: incumbent,
                        distance: 1,
                        symbol: 119,
                    };
                    let expected = if length >= MIN_MATCH
                        && (length > incumbent || (length == incumbent && symbol < before.symbol))
                    {
                        Match {
                            length,
                            distance: 64,
                            symbol,
                        }
                    } else {
                        before
                    };
                    let mut actual = before;
                    consider_fast_match(
                        &values,
                        64,
                        limit,
                        64,
                        SourceFastRanges::Both,
                        &mut actual,
                        || symbol,
                    );
                    assert_eq!(
                        tuple(actual),
                        tuple(expected),
                        "mismatch={mismatch} limit={limit} incumbent={incumbent} symbol={symbol}"
                    );
                }
            }
        }
    }
}

fn check_fast<const N: usize>(values: &[u32]) {
    for width in [0, 1, 2, 37] {
        for maximum in [3, 11, 129] {
            for prefix in [
                SourceFastPrefix::Disabled,
                SourceFastPrefix::Scalar,
                SourceFastPrefix::Simd,
            ] {
                let config = SourceConfig {
                    max_match_length: maximum,
                    fast_ranges: SourceFastRanges::Both,
                    fast_prefix: Some(prefix),
                    ..SourceConfig::default()
                };
                // Keep prefix filtering and repeated insertion identical; only
                // the extension changes from scalar to the short-head route.
                let reference = SourceConfig {
                    fast_ranges: SourceFastRanges::RepeatedInsert,
                    ..config.clone()
                };
                let mut actual = Fast::<N>::new(values, width, &config).unwrap();
                let mut expected = Fast::<N>::new(values, width, &reference).unwrap();
                let mut position = 0;
                while position < values.len() {
                    let matched = actual.find(position);
                    assert_eq!(tuple(matched), tuple(expected.find(position)));
                    let consumed = if position % 5 == 0 {
                        1
                    } else {
                        matched.length.max(1)
                    };
                    let end = position + consumed;
                    actual.update_range(position + 1, end);
                    expected.update_range(position + 1, end);
                    position = end;
                }
                for (a, b) in actual.buckets.iter().zip(&expected.buckets) {
                    assert_eq!(a.positions, b.positions);
                    assert_eq!(a.next, b.next);
                    assert_eq!(a.size, b.size);
                }
            }
        }
    }
}

#[test]
fn short_extension_preserves_fast_candidates_skips_and_physical_rings() {
    let mut values = Vec::new();
    for block in 0..64u32 {
        values.extend((0..3 + block % 15).map(|i| (i + block % 7).wrapping_mul(0x9e37_79b9)));
        values.push(0x8000_0000 | block);
    }
    values.extend([0; 129]);
    values.extend((0..513).map(|i| (i % 7) as u32));
    values.extend([u32::MAX, 0]);
    check_fast::<1>(&values);
    check_fast::<3>(&values);
    check_fast::<7>(&values);
    check_fast::<15>(&values);
    check_fast::<31>(&values);
}

#[test]
fn short_extension_keeps_fast_window_boundary_inclusive() {
    let position = WINDOW_SIZE + 1;
    let mut values = vec![0; position + 17];
    let prefix: Vec<_> = (0..17u32).map(|i| u32::MAX - i * 371).collect();
    values[position..].copy_from_slice(&prefix);
    for distance in [WINDOW_SIZE - 1, WINDOW_SIZE, WINDOW_SIZE + 1] {
        let previous = position - distance;
        values[previous..previous + prefix.len()].copy_from_slice(&prefix);
        let config = SourceConfig {
            fast_ranges: SourceFastRanges::Both,
            fast_prefix: Some(SourceFastPrefix::Disabled),
            ..SourceConfig::default()
        };
        let mut matcher = Fast::<1>::new(&values, 0, &config).unwrap();
        let key = hash(&values, position) as usize & ((1 << 14) - 1);
        matcher.buckets[key].insert(previous);
        let matched = matcher.find(position);
        if distance <= WINDOW_SIZE {
            assert_eq!(tuple(matched), (17, distance, distance as u32 - 1));
        } else {
            assert_eq!(tuple(matched), tuple(Match::default()));
        }
    }
}
