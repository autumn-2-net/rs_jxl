// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

use super::*;

fn generic<'a>(values: &'a [u32], width: usize, config: &SourceConfig) -> Fast<'a, 1> {
    let mut matcher = Fast::<1>::new(values, width, config).unwrap();
    matcher.buckets = FastBuckets {
        single: Vec::new(),
        rings: filled(1 << 14, Ring::<1>::new()).unwrap(),
        generic_one: true,
    };
    matcher
}

#[test]
fn single_storage_sentinel_bounds_and_overwrite_match_generic_ring() {
    let mut actual = FastBuckets::<1>::new(1 << 14).unwrap();
    let mut expected = Ring::<1>::new();
    assert!(actual.rings.is_empty());
    assert_eq!(actual.single.len() * std::mem::size_of::<u32>(), 64 << 10);
    assert_eq!(std::mem::size_of::<Ring<1>>(), 8);
    assert!(actual.single.iter().all(|&p| p == EMPTY_FAST_POSITION));
    assert!(actual.get(0).positions.is_empty());
    assert_eq!(actual.get(0).oldest, None);
    for position in [
        0,
        1,
        WINDOW_SIZE,
        2 * WINDOW_SIZE + 3,
        u32::MAX as usize - 1,
    ] {
        actual.insert(0, position);
        expected.insert(position);
        assert_eq!(actual.get(0).positions, &expected.positions[..1]);
        assert_eq!(actual.get(0).oldest, Some(expected.positions[0]));
    }
    for (begin, end) in [(7, 7), (7, 8), (8, 31), (31, u32::MAX as usize)] {
        actual.repeated(0, begin, end);
        expected.repeated(begin, end);
        assert_eq!(actual.get(0).positions, &expected.positions[..1]);
        assert_eq!(actual.get(0).oldest, Some(expected.positions[0]));
    }
    assert!(validate_ring_input_len(u32::MAX as usize).is_ok());
    if let Some(too_long) = (u32::MAX as usize).checked_add(1) {
        assert!(validate_ring_input_len(too_long).is_err());
    }
    assert!(std::panic::catch_unwind(|| actual.get(1 << 14)).is_err());
    #[cfg(debug_assertions)]
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actual.insert(0, EMPTY_FAST_POSITION as usize);
        }))
        .is_err()
    );
    assert_eq!(actual.get(0).oldest, Some(u32::MAX - 1));
    let fresh = FastBuckets::<1>::new(1).unwrap();
    assert_eq!(fresh.get(0).oldest, None);
    let multiple = FastBuckets::<3>::new(1).unwrap();
    assert!(multiple.single.is_empty());
    assert_eq!(multiple.rings.len(), 1);
}

fn compare(values: &[u32], width: usize, config: &SourceConfig, skip: bool) {
    let mut actual = Fast::<1>::new(values, width, config).unwrap();
    let mut expected = generic(values, width, config);
    // Exercise both activated prefix gates and the cold lazy gate.
    if skip {
        for matcher in [&mut actual, &mut expected] {
            matcher
                .prefix
                .record_candidate_work(matcher.prefix.activation_threshold());
        }
    }
    let mut position = 0;
    while position < values.len() {
        let matched = if skip {
            let matched = actual.find_impl::<true>(position);
            assert_eq!(matched, expected.find_impl::<true>(position));
            matched
        } else {
            let matched = actual.find_impl::<false>(position);
            assert_eq!(matched, expected.find_impl::<false>(position));
            matched
        };
        if position == 0 {
            assert_eq!(matched.length, 0, "current position must not match itself");
        }
        assert_eq!(actual.prefix.activation(), expected.prefix.activation());
        assert_eq!(
            actual.prefix.candidate_work(),
            expected.prefix.candidate_work()
        );
        let consumed = if skip {
            matched.length.max(1 + position % 11)
        } else {
            1
        };
        let end = (position + consumed).min(values.len());
        actual.update_range(position + 1, end);
        expected.update_range(position + 1, end);
        position = end;
    }
    for (a, b) in actual.buckets.iter().zip(expected.buckets.iter()) {
        assert_eq!(a.positions, b.positions);
        assert_eq!((a.next, a.size), (b.next, b.size));
    }
}

#[test]
fn single_matches_generic_ring_across_spatial_ranges_skips_and_tails() {
    let mut values: Vec<_> = (0..193u32).map(|i| (i % 31) * 65537).collect();
    values.extend_from_within(7..104);
    values.extend([0; 97]);
    values.extend([u32::MAX, 0]);
    for width in [0, 1, 37] {
        for range_search in [false, true] {
            for fast_prefix in [
                SourceFastPrefix::Disabled,
                SourceFastPrefix::Scalar,
                SourceFastPrefix::Simd,
            ] {
                for fast_ranges in [SourceFastRanges::Disabled, SourceFastRanges::Both] {
                    let config = SourceConfig {
                        range_search,
                        fast_prefix: Some(fast_prefix),
                        fast_ranges,
                        max_match_length: 65,
                        ..SourceConfig::default()
                    };
                    compare(&values, width, &config, false);
                    compare(&values, width, &config, true);
                }
            }
        }
    }
    for len in 0..=2 {
        compare(&values[..len], 0, &SourceConfig::default(), true);
    }
    // Candidate distance remains inclusive at the physical decoder window.
    let position = WINDOW_SIZE + 2;
    let mut values = vec![0; position + 3];
    values[position..].copy_from_slice(&[17, 19, 23]);
    for distance in [WINDOW_SIZE - 1, WINDOW_SIZE, WINDOW_SIZE + 1] {
        let previous = position - distance;
        values[previous..previous + 3].copy_from_slice(&[17, 19, 23]);
        let config = SourceConfig::default();
        let mut actual = Fast::<1>::new(&values, 0, &config).unwrap();
        let mut expected = generic(&values, 0, &config);
        let key = hash(&values, position) as usize & 16383;
        actual.buckets.insert(key, previous);
        expected.buckets.insert(key, previous);
        let matched = actual.find_impl::<false>(position);
        assert_eq!(matched, expected.find_impl::<false>(position));
        assert_eq!(matched.length, if distance <= WINDOW_SIZE { 3 } else { 0 });
    }
}
