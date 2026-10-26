// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

use super::*;

// Independent FMA expression with the frozen literals rounded to f32 bits.
// Do not call the production polynomial or share its coefficient arrays.
fn fma_log_oracle(value: f32) -> f32 {
    let bits = value.to_bits() as i32;
    let exponent = (bits - 0x3f2aaaab) >> 23;
    let x = f32::from_bits((bits - (exponent << 23)) as u32) - 1.0;
    let numerator = f32::from_bits(0x3f3e11c7)
        .mul_add(x, f32::from_bits(0x3fb6e02b))
        .mul_add(x, f32::from_bits(0xb5f85ab0));
    let denominator = f32::from_bits(0x3e32458c)
        .mul_add(x, f32::from_bits(0x3f813ced))
        .mul_add(x, f32::from_bits(0x3f7d8625));
    numerator / denominator + exponent as f32
}

// Independent scalar expression and lane schedule from frozen enc_ma.cc
// and Highway generic_ops-inl.h; compare each backend to its own profile.
fn default_oracle_term(count: u32, total: u32, fused: bool) -> f32 {
    if count == total {
        return 0.0;
    }
    let p = (count as f32 * (1.0 / total as f32)).max(1.0 / 4096.0);
    if fused {
        return count as f32 * fma_log_oracle(p);
    }
    let bits = p.to_bits() as i32;
    let exponent = (bits - 0x3f2aaaab) >> 23;
    let x = f32::from_bits((bits - (exponent << 23)) as u32) - 1.0;
    let numerator = LOG_P[0] + x * (LOG_P[1] + x * LOG_P[2]);
    let denominator = LOG_Q[0] + x * (LOG_Q[1] + x * LOG_Q[2]);
    count as f32 * (exponent as f32 + numerator / denominator)
}

fn default_oracle(counts: &[u32], total: u32, width: usize, fused: bool) -> f32 {
    let mut sums = [0.0f32; 8];
    for (i, &count) in counts.iter().enumerate() {
        sums[i % width] -= default_oracle_term(count, total, fused);
    }
    oracle_reduce(sums, width)
}

fn oracle_reduce(mut sums: [f32; 8], width: usize) -> f32 {
    if width == 1 {
        return sums[0];
    }
    if width == 8 {
        for i in 0..4 {
            sums[i] += sums[i + 4];
        }
    }
    (sums[0] + sums[3]) + (sums[1] + sums[2])
}

fn default_profiles() -> Vec<(DefaultScorer, usize)> {
    #[allow(unused_mut)]
    let mut profiles = vec![(DefaultScorer::with_backend(DefaultBackend::Scalar), 1)];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            profiles.push((DefaultScorer::with_backend(DefaultBackend::Sse2), 4));
        }
        if std::is_x86_feature_detected!("avx2") {
            profiles.push((DefaultScorer::with_backend(DefaultBackend::Avx2), 8));
            if std::is_x86_feature_detected!("fma") {
                profiles.push((DefaultScorer::with_backend(DefaultBackend::Avx2Fma), 8));
                if DefaultBackend::detect(true) == DefaultBackend::Avx512Fma8 {
                    profiles.push((DefaultScorer::with_backend(DefaultBackend::Avx512Fma8), 8));
                }
            }
        }
    }
    profiles
}

#[test]
fn fma_horner_matches_independent_f32_and_native_lane_oracles() {
    let mut values = Vec::new();
    for exponent in -12..=0 {
        for step in 0..=1024 {
            values.push((1.0 + step as f32 / 1024.0) * 2.0_f32.powi(exponent));
        }
    }
    // Either side of the range-reduction breakpoint, plus exact one.
    values.extend([0x3f2aaaaa, 0x3f2aaaab, 0x3f2aaaac, 0x3f800000].map(f32::from_bits));
    let mut differences = 0;
    for &value in &values {
        let expected = fma_log_oracle(value);
        assert_eq!(fast_log2::<true>(value).to_bits(), expected.to_bits());
        differences += usize::from(expected.to_bits() != fast_log2::<false>(value).to_bits());
        assert!((f64::from(expected) - f64::from(value).log2()).abs() <= 0.000005);
    }
    assert!(differences > 0, "fixture must distinguish FMA from non-FMA");

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
        for chunk in values.chunks(8) {
            let mut input = [1.0; 8];
            input[..chunk.len()].copy_from_slice(chunk);
            let mut fused = [0.0; 8];
            let mut unfused = [0.0; 8];
            // SAFETY: both CPU/OS features detected; all accesses span
            // exactly eight initialized/writable f32 lanes.
            unsafe {
                let lanes = _mm256_loadu_ps(input.as_ptr());
                _mm256_storeu_ps(fused.as_mut_ptr(), x86::avx2_fma::fast_log(lanes));
                _mm256_storeu_ps(unfused.as_mut_ptr(), x86::avx2::fast_log(lanes));
            }
            for ((&value, actual), non_fma) in input.iter().zip(fused).zip(unfused) {
                assert_eq!(actual.to_bits(), fma_log_oracle(value).to_bits());
                assert_eq!(non_fma.to_bits(), fast_log2::<false>(value).to_bits());
            }
        }
    }
}

#[test]
fn default_runtime_dispatch_and_profile_specific_tables() {
    let expected = DefaultBackend::Scalar;
    #[cfg(target_arch = "x86_64")]
    let expected = if std::is_x86_feature_detected!("avx2") {
        if std::is_x86_feature_detected!("fma") {
            if std::env::var("RS_JXL_MA_RCP").as_deref() == Ok("1") {
                DefaultBackend::Avx2Rcp
            } else if std::env::var("RS_JXL_MA_AVX512").as_deref() == Ok("1")
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512dq")
            {
                DefaultBackend::Avx512Fma8
            } else {
                DefaultBackend::Avx2Fma
            }
        } else {
            DefaultBackend::Avx2
        }
    } else if std::is_x86_feature_detected!("sse2") {
        DefaultBackend::Sse2
    } else {
        expected
    };
    assert_eq!(DefaultScorer::new().backend, expected);
    let unfused = DefaultScorer::with_backend(DefaultBackend::Scalar);
    for (profile, _) in default_profiles() {
        let again = DefaultScorer::with_backend(profile.backend);
        assert!(std::ptr::eq(profile.small, again.small));
        if profile.backend.uses_fma() {
            assert!(!std::ptr::eq(profile.small, unfused.small));
            assert!(
                profile
                    .small
                    .iter()
                    .zip(unfused.small)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            );
        } else {
            assert!(std::ptr::eq(profile.small, unfused.small));
        }
    }
}

#[test]
fn fma_scoring_preserves_unfused_entropy_accumulation_and_lane_reduction() {
    let mut seed = 0xb866_b3de_u32;
    let (mut contraction_differs, mut serial_differs) = (false, false);
    for _ in 0..32 {
        let counts: Vec<u32> = (0..257)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                1 + (seed >> 8) % 50000
            })
            .collect();
        let total = counts.iter().sum::<u32>();
        let expected = default_oracle(&counts, total, 8, true);
        let mut contracted = [0.0f32; 8];
        for (i, &count) in counts.iter().enumerate() {
            let probability = (count as f32 * (1.0 / total as f32)).max(1.0 / 4096.0);
            // Deliberately WRONG for this source: catches accidental
            // NegMulAdd contraction of Sub(accumulator, Mul(count, log)).
            contracted[i % 8] =
                (-(count as f32)).mul_add(fma_log_oracle(probability), contracted[i % 8]);
        }
        contraction_differs |= oracle_reduce(contracted, 8).to_bits() != expected.to_bits();
        serial_differs |= default_oracle(&counts, total, 1, true).to_bits() != expected.to_bits();
        for (scorer, width) in default_profiles() {
            assert_eq!(
                scorer.estimate_bits(&counts, total).to_bits(),
                default_oracle(&counts, total, width, scorer.backend.uses_fma()).to_bits()
            );
        }
    }
    assert!(
        contraction_differs,
        "fixture must reject fused accumulation"
    );
    assert!(serial_differs, "fixture must reject serial lane reduction");
}

fn check_default_move(above: &[u32], below: &[u32], delta: &[u32]) {
    let expected_r: Vec<u32> = above.iter().zip(delta).map(|(&r, &d)| r - d).collect();
    let expected_l: Vec<u32> = below.iter().zip(delta).map(|(&l, &d)| l + d).collect();
    let totals = (expected_r.iter().sum(), expected_l.iter().sum());
    for offset in 0..8 {
        let end = offset + delta.len();
        let mut cached = vec![u32::MAX; end + 1];
        cached[offset..end].copy_from_slice(delta);
        let snapshot = cached.clone();
        // All profiles reuse this same immutable, possibly unaligned cache.
        for (scorer, width) in default_profiles() {
            let mut r = vec![u32::MAX; end + 1];
            let mut l = vec![u32::MAX; end + 1];
            r[offset..end].copy_from_slice(above);
            l[offset..end].copy_from_slice(below);
            let actual = scorer.move_and_score(
                &mut r[offset..end],
                &mut l[offset..end],
                &cached[offset..end],
                totals,
            );
            assert_eq!(&r[offset..end], &expected_r);
            assert_eq!(&l[offset..end], &expected_l);
            assert_eq!(cached, snapshot);
            for (bits, counts, total) in [
                (actual.0, expected_r.as_slice(), totals.0),
                (actual.1, expected_l.as_slice(), totals.1),
            ] {
                assert!(bits.is_finite() && bits >= 0.0);
                assert_eq!(
                    bits.to_bits(),
                    default_oracle(counts, total, width, scorer.backend.uses_fma()).to_bits()
                );
                assert_eq!(
                    bits.to_bits(),
                    scorer.estimate_bits(counts, total).to_bits()
                );
                close(bits, default_oracle(counts, total, 1, false));
            }
            for storage in [&r, &l] {
                assert!(storage[..offset].iter().all(|&c| c == u32::MAX));
                assert_eq!(storage[end], u32::MAX);
            }
        }
    }
}

#[test]
fn sparse_blocks_match_dense_native_profiles_and_reject_invalid_addresses() {
    for (scorer, lanes) in default_profiles() {
        for len in [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 255, 256] {
            for scale in [1, 65535] {
                let parent: Vec<u32> = (0..len)
                    .map(|i| {
                        if i % 23 == 0 || i + 1 == len {
                            scale
                        } else {
                            0
                        }
                    })
                    .collect();
                let blocks: Vec<_> = parent
                    .chunks(lanes)
                    .enumerate()
                    .filter_map(|(i, b)| b.iter().any(|&n| n != 0).then_some(i * lanes))
                    .collect();
                let delta: Vec<_> = parent.iter().map(|&n| n / 2).collect();
                let total: u32 = parent.iter().sum();
                let mut sparse_r = parent.clone();
                let mut sparse_l = vec![0; len];
                let mut dense_r = parent.clone();
                let mut dense_l = vec![0; len];
                let mut moved = 0;
                for _ in 0..2 {
                    moved += delta.iter().sum::<u32>();
                    let totals = (total - moved, moved);
                    let a = scorer.move_and_score_sparse(
                        &mut sparse_r,
                        &mut sparse_l,
                        &delta,
                        totals,
                        &blocks,
                    );
                    let b = scorer.move_and_score(&mut dense_r, &mut dense_l, &delta, totals);
                    assert_eq!(
                        (a.0.to_bits(), a.1.to_bits()),
                        (b.0.to_bits(), b.1.to_bits())
                    );
                    assert_eq!(sparse_r, dense_r);
                    assert_eq!(sparse_l, dense_l);
                }
            }
        }
        for blocks in [vec![lanes], vec![0, 0]] {
            let mut above = vec![1; lanes];
            let mut below = vec![0; lanes];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| scorer
                    .move_and_score_sparse(
                        &mut above,
                        &mut below,
                        &vec![0; lanes],
                        (lanes as u32, 0),
                        &blocks
                    )))
                .is_err()
            );
            assert_eq!(above, vec![1; lanes]);
            assert_eq!(below, vec![0; lanes]);
        }
    }
}

#[test]
fn default_split_sweep_matches_each_threshold_all_native_profiles() {
    for (scorer, lanes) in default_profiles() {
        for len in [0, 1, 3, 4, 7, 8, 9, 15, 17, 31, 33, 65, 255, 256] {
            for scale in [1u32, 257, 65535, 4_000_000] {
                for padded in [false, true] {
                    let stride = len.max(1) + if padded { lanes } else { 0 };
                    // Nonadjacent rows and unrelated bin tags exercise packed
                    // layouts. Neighboring storage must stay immutable.
                    let mut storage = vec![0; 1 + 9 * stride + 1];
                    let mut parent = vec![0; len];
                    let mut splits = Vec::new();
                    let mut moved = 0;
                    for row in [0, 2, 5, 8] {
                        let mut weight = 0;
                        for symbol in 0..len {
                            if symbol % 23 == 0 || symbol + 1 == len {
                                let count = scale + ((symbol + row) % 3) as u32;
                                storage[1 + row * stride + symbol] = count;
                                parent[symbol] += count;
                                weight += count;
                            }
                        }
                        moved += weight;
                        splits.push(HistogramSplit {
                            bin: row * 3 + 7,
                            row,
                            totals: (0, moved),
                        });
                    }
                    let total: u32 = parent.iter().sum();
                    for split in &mut splits {
                        split.totals.0 = total - split.totals.1;
                    }
                    let blocks: Vec<_> = parent
                        .chunks(lanes)
                        .enumerate()
                        .filter_map(|(i, b)| b.iter().any(|&n| n != 0).then_some(i * lanes))
                        .collect();
                    let original = storage.clone();
                    for sparse in [None, Some(blocks.as_slice())] {
                        let mut above = vec![123; len + 2];
                        above[1..len + 1].copy_from_slice(&parent);
                        let mut below = vec![123; len + 2];
                        below[1..len + 1].fill(0);
                        let mut expected_above = parent.clone();
                        let mut expected_below = vec![0; len];
                        let mut expected = Vec::new();
                        for &split in &splits {
                            let start = 1 + split.row * stride;
                            let bits = scorer.move_and_score(
                                &mut expected_above,
                                &mut expected_below,
                                &storage[start..start + len],
                                split.totals,
                            );
                            expected.push((split.bin, bits.0.to_bits(), bits.1.to_bits()));
                        }
                        let mut actual = Vec::new();
                        scorer.split_sweep(
                            &mut above[1..len + 1],
                            &mut below[1..len + 1],
                            &storage[1..1 + 9 * stride],
                            stride,
                            total,
                            sparse,
                            splits.iter().copied(),
                            |split, r, l| actual.push((split.bin, r.to_bits(), l.to_bits())),
                        );
                        assert_eq!(
                            actual, expected,
                            "backend={:?}, len={len}, scale={scale}",
                            scorer.backend
                        );
                        assert_eq!(&above[1..len + 1], expected_above);
                        assert_eq!(&below[1..len + 1], expected_below);
                        assert_eq!(
                            (above[0], above[len + 1], below[0], below[len + 1]),
                            (123, 123, 123, 123)
                        );
                        assert_eq!(storage, original);
                    }
                }
            }
        }
    }
}

#[test]
fn default_small_sweep_matches_weighted_moves_and_table_boundaries() {
    for (scorer, lanes) in default_profiles() {
        for len in [1, 3, 4, 7, 8, 9, 15, 16, 17, 33] {
            for total in (0u32..=257).chain([511, 512, 513, 65535, i32::MAX as u32]) {
                let mut parent = vec![0; len];
                let last = if total % 2 == 0 && len > lanes {
                    len / lanes * lanes - 1
                } else {
                    len - 1
                };
                parent[0] += total / 7;
                parent[len / 2] += total / 3;
                parent[last] += total - total / 7 - total / 3;
                let initial_below: Vec<_> = parent.iter().map(|&n| n / 3).collect();
                let initial_above: Vec<_> = parent
                    .iter()
                    .zip(&initial_below)
                    .map(|(&n, &l)| n - l)
                    .collect();
                let stride = len + if total % 2 == 0 { lanes } else { 0 };
                let mut storage = vec![u32::MAX; 1 + 9 * stride + 1];
                let mut remaining = initial_above.clone();
                let mut splits = Vec::new();
                // Packed rows need not follow the stable threshold order.
                for (step, row) in [5, 0, 7, 2, 8].into_iter().enumerate() {
                    let start = 1 + row * stride;
                    for (i, r) in remaining.iter_mut().enumerate() {
                        let d = *r / (5 - step) as u32;
                        storage[start + i] = d;
                        *r -= d;
                    }
                    let right = remaining.iter().sum::<u32>();
                    splits.push(HistogramSplit {
                        bin: step * 17 + 3,
                        row,
                        totals: (right, total - right),
                    });
                }
                let blocks: Vec<_> = parent
                    .chunks(lanes)
                    .enumerate()
                    .filter_map(|(i, b)| b.iter().any(|&n| n != 0).then_some(i * lanes))
                    .collect();
                let snapshot = storage.clone();
                let mut expected_above = initial_above.clone();
                let mut expected_below = initial_below.clone();
                let mut expected = Vec::new();
                for &split in &splits {
                    let start = 1 + split.row * stride;
                    let bits = scorer.move_and_score(
                        &mut expected_above,
                        &mut expected_below,
                        &storage[start..start + len],
                        split.totals,
                    );
                    expected.push((split.bin, split.row, bits.0.to_bits(), bits.1.to_bits()));
                }
                for sparse in [None, Some(blocks.as_slice())] {
                    let offset = 1 + len % lanes;
                    let end = offset + len;
                    let mut above = vec![u32::MAX; end + 1];
                    let mut below = vec![u32::MAX; end + 1];
                    above[offset..end].copy_from_slice(&initial_above);
                    below[offset..end].copy_from_slice(&initial_below);
                    let mut actual = Vec::new();
                    scorer.split_sweep(
                        &mut above[offset..end],
                        &mut below[offset..end],
                        &storage[1..1 + 9 * stride],
                        stride,
                        total,
                        sparse,
                        splits.iter().copied(),
                        |split, r, l| {
                            actual.push((split.bin, split.row, r.to_bits(), l.to_bits()));
                        },
                    );
                    assert_eq!(
                        actual, expected,
                        "backend={:?}, len={len}, total={total}, sparse={sparse:?}",
                        scorer.backend
                    );
                    assert_eq!(&above[offset..end], expected_above);
                    assert_eq!(&below[offset..end], expected_below);
                    assert_eq!(below[offset..end].iter().sum::<u32>(), total);
                    for counts in [&above, &below] {
                        assert!(counts[..offset].iter().all(|&n| n == u32::MAX));
                        assert_eq!(counts[end], u32::MAX);
                    }
                    assert_eq!(storage, snapshot);
                }
            }
        }
    }
}

#[test]
fn default_small_sweep_requires_combined_mass_to_fit_table() {
    for (scorer, _) in default_profiles() {
        let mut above = [128; 4];
        let mut below = [0; 4];
        let mut expected_above = above;
        let mut expected_below = below;
        let splits = [
            HistogramSplit {
                bin: 3,
                row: 0,
                totals: (256, 256),
            },
            HistogramSplit {
                bin: 7,
                row: 1,
                totals: (0, 512),
            },
        ];
        let mut emitted = 0;
        scorer.split_sweep(
            &mut above,
            &mut below,
            &[64; 8],
            4,
            512,
            None,
            splits.into_iter(),
            |split, r, l| {
                let expected = scorer.move_and_score(
                    &mut expected_above,
                    &mut expected_below,
                    &[64; 4],
                    split.totals,
                );
                assert_eq!(r.to_bits(), expected.0.to_bits());
                assert_eq!(l.to_bits(), expected.1.to_bits());
                emitted += 1;
            },
        );
        assert_eq!(emitted, 2);
        assert_eq!(above, expected_above);
        assert_eq!(below, expected_below);
    }
}

#[test]
fn default_small_sweep_empty_and_unwind_preserve_completed_moves() {
    for (scorer, _) in default_profiles() {
        for callback_panics in [false, true] {
            let mut above = [9; 9];
            let mut below = [0; 9];
            scorer.split_sweep(
                &mut above,
                &mut below,
                &[1; 18],
                9,
                81,
                None,
                std::iter::empty(),
                |_, _, _| panic!("empty sweep emitted a split"),
            );
            assert_eq!(above, [9; 9]);
            assert_eq!(below, [0; 9]);
            let splits = [
                HistogramSplit {
                    bin: 3,
                    row: 0,
                    totals: (72, 9),
                },
                HistogramSplit {
                    bin: 7,
                    row: usize::MAX,
                    totals: (63, 18),
                },
            ];
            let mut emitted = 0;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scorer.split_sweep(
                    &mut above,
                    &mut below,
                    &[1; 18],
                    9,
                    81,
                    None,
                    splits.into_iter(),
                    |_, _, _| {
                        emitted += 1;
                        assert!(!callback_panics, "injected callback panic");
                    },
                );
            }));
            assert!(result.is_err());
            assert_eq!(emitted, 1);
            assert_eq!(above, [8; 9]);
            assert_eq!(below, [1; 9]);
        }
    }
}

#[test]
fn default_small_sweep_changed_totals_keep_general_fallback() {
    for (scorer, lanes) in default_profiles() {
        let mut above = [9; 9];
        let mut below = [0; 9];
        let mut expected_above = above;
        let mut expected_below = below;
        let blocks: Vec<_> = (0..above.len()).step_by(lanes).collect();
        let totals = [(72, 9), (257, 18), (54, 257), (256, 256), (36, 45)];
        let mut emitted = 0;
        scorer.split_sweep(
            &mut above,
            &mut below,
            &[1; 45],
            9,
            81,
            Some(&blocks),
            totals
                .into_iter()
                .enumerate()
                .map(|(row, totals)| HistogramSplit {
                    bin: row * 3,
                    row,
                    totals,
                }),
            |split, r, l| {
                let expected = scorer.move_and_score(
                    &mut expected_above,
                    &mut expected_below,
                    &[1; 9],
                    split.totals,
                );
                assert_eq!(r.to_bits(), expected.0.to_bits());
                assert_eq!(l.to_bits(), expected.1.to_bits());
                emitted += 1;
            },
        );
        assert_eq!(emitted, totals.len());
        assert_eq!(above, expected_above);
        assert_eq!(below, expected_below);
    }
}

#[test]
fn default_split_sweep_rejects_invalid_layout_before_mutation() {
    for (scorer, lanes) in default_profiles() {
        let len = lanes * 2;
        for (stride, cells, below_len, row, blocks) in [
            (0, len, len, 0, None),
            (len - 1, len - 1, len, 0, None),
            (len, len + 1, len, 0, None),
            (len, len, len - 1, 0, None),
            (len, len, len, usize::MAX, None),
            (len, len, len, 0, Some(vec![len])),
            (len, len, len, 0, Some(vec![0, 0])),
            (len, len, len, 0, Some(vec![lanes, 0])),
        ] {
            let mut above = vec![1; len];
            let mut below = vec![0; below_len];
            let deltas = vec![0; cells];
            let mut emitted = false;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scorer.split_sweep(
                    &mut above,
                    &mut below,
                    &deltas,
                    stride,
                    len as u32,
                    blocks.as_deref(),
                    std::iter::once(HistogramSplit {
                        bin: 0,
                        row,
                        totals: (len as u32, 0),
                    }),
                    |_, _, _| emitted = true,
                );
            }));
            assert!(result.is_err());
            assert_eq!(above, vec![1; len]);
            assert_eq!(below, vec![0; below_len]);
            assert!(!emitted);
        }
    }
}

#[test]
fn default_small_table_is_shared_source_expression_not_log_cache() {
    let scorer = DefaultScorer::default();
    assert!(std::ptr::eq(scorer.small, DefaultScorer::new().small));
    assert_eq!(std::mem::size_of_val(scorer.small), 132612);
    assert_eq!(scorer.small_row(257), &[]);
    for (profile, width) in default_profiles() {
        assert_eq!(profile.lanes(), width);
        for total in 0..=256 {
            let row = profile.small_row(total);
            assert_eq!(row.len(), total as usize + 1);
            for count in 0..=total {
                assert_eq!(
                    row[count as usize].to_bits(),
                    default_oracle_term(count, total, profile.backend.uses_fma()).to_bits(),
                    "backend={:?}, count={count}, total={total}",
                    profile.backend
                );
            }
        }
        if profile.backend == scorer.backend {
            let counts = [1, 9, 0, 3, 0, 0, 8, 1, 17];
            assert_eq!(
                scorer.estimate_bits(&counts, 39).to_bits(),
                default_oracle(&counts, 39, width, scorer.backend.uses_fma()).to_bits()
            );
        }
    }
}

#[test]
fn default_weighted_moves_immutable_cache_unaligned_tails() {
    let mut state = 0xb866_b3de_u32;
    for len in (0..=33).chain([63, 64, 65, 127, 128, 129]) {
        let mut above = vec![0; len];
        let mut below = vec![0; len];
        let mut delta = vec![0; len];
        for i in 0..len {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            above[i] = state % 300;
            below[i] = (state >> 8) % 90;
            delta[i] = (state >> 16) % (above[i] + 1);
        }
        check_default_move(&above, &below, &delta);
        check_default_move(&above, &vec![0; len], &above);
        check_default_move(&above, &vec![0; len], &vec![0; len]);
    }
}

#[test]
fn default_masked_tails_cover_each_lane_and_independent_table_routes() {
    // Put mass in both ends of the partial vector, not just lane zero.
    // check_default_move also surrounds every unaligned slice with canaries.
    for len in (1..=16).chain(25..=31) {
        for (right, left) in [
            (0, 0),
            (1, 0),
            (256, 256),
            (256, 257),
            (257, 256),
            (4097, 65535),
            ((1 << 24) + 1, 257),
            (MAX_HISTOGRAM_TOTAL - 257, 257),
        ] {
            let tail = len / 8 * 8;
            let first = if tail == len { len - 1 } else { tail };
            let mut r = vec![0; len];
            let mut l = vec![0; len];
            r[first] = right / 3;
            r[len - 1] += right - right / 3;
            l[first] = left / 2;
            l[len - 1] += left - left / 2;
            let delta: Vec<_> = l.iter().map(|&n| n / 2).collect();
            let above: Vec<_> = r.iter().zip(&delta).map(|(&r, &d)| r + d).collect();
            let below: Vec<_> = l.iter().zip(&delta).map(|(&l, &d)| l - d).collect();
            check_default_move(&above, &below, &delta);
        }
    }
}

#[test]
fn default_masked_tails_match_full_rescans_through_weighted_split_sweeps() {
    // Each row is an immutable cached property bin. Include empty rows and
    // saturated sample weights, then reuse the same cache for another sweep.
    for symbols in 1..=33 {
        let rows: Vec<Vec<u32>> = (0..37)
            .map(|bin| {
                (0..symbols)
                    .map(|symbol| {
                        if bin % 7 == 0 || (bin + symbol) % 5 != 0 {
                            0
                        } else if bin == 18 {
                            65535
                        } else {
                            1 + (bin * 13 + symbol * 19) as u32 % 31
                        }
                    })
                    .collect()
            })
            .collect();
        let snapshot = rows.clone();
        let parent: Vec<u32> = (0..symbols)
            .map(|symbol| rows.iter().map(|row| row[symbol]).sum())
            .collect();
        for (scorer, width) in default_profiles() {
            for _ in 0..2 {
                // Exact-sized slices exercise kernels without padding.
                let mut above = parent.clone().into_boxed_slice();
                let mut below = vec![0; symbols].into_boxed_slice();
                for bin in 0..rows.len() {
                    let expected_l: Vec<u32> = (0..symbols)
                        .map(|s| rows[..=bin].iter().map(|row| row[s]).sum())
                        .collect();
                    let expected_r: Vec<u32> = (0..symbols)
                        .map(|s| rows[bin + 1..].iter().map(|row| row[s]).sum())
                        .collect();
                    let totals = (expected_r.iter().sum(), expected_l.iter().sum());
                    let actual = scorer.move_and_score(&mut above, &mut below, &rows[bin], totals);
                    assert_eq!(&*above, expected_r.as_slice());
                    assert_eq!(&*below, expected_l.as_slice());
                    for (bits, counts, total) in [
                        (actual.0, expected_r.as_slice(), totals.0),
                        (actual.1, expected_l.as_slice(), totals.1),
                    ] {
                        assert_eq!(
                            bits.to_bits(),
                            default_oracle(counts, total, width, scorer.backend.uses_fma())
                                .to_bits(),
                            "symbols={symbols}, bin={bin}, backend={:?}",
                            scorer.backend,
                        );
                    }
                }
            }
        }
        assert_eq!(rows, snapshot);
    }
}

#[test]
fn default_masked_sparse_tail_is_optional_and_preserves_neighboring_storage() {
    for (scorer, lanes) in default_profiles() {
        for tail in 1..lanes {
            let len = 2 * lanes + tail;
            for include_tail in [false, true] {
                let mut parent = vec![0; len];
                parent[0] = 65535;
                let mut blocks = vec![0];
                if include_tail {
                    parent[len - 1] = 257;
                    blocks.push(2 * lanes);
                }
                let delta: Vec<_> = parent.iter().map(|&n| n / 2).collect();
                let mut above = parent.clone();
                let mut below = vec![0; len];
                // The following elements are outside the borrowed histogram.
                above.extend([u32::MAX; 8]);
                below.extend([u32::MAX; 8]);
                let moved = delta.iter().sum::<u32>();
                let totals = (parent.iter().sum::<u32>() - moved, moved);
                let actual = scorer.move_and_score_sparse(
                    &mut above[..len],
                    &mut below[..len],
                    &delta,
                    totals,
                    &blocks,
                );
                assert_eq!(&above[len..], &[u32::MAX; 8]);
                assert_eq!(&below[len..], &[u32::MAX; 8]);
                for i in 0..len {
                    assert_eq!(above[i], parent[i] - delta[i]);
                    assert_eq!(below[i], delta[i]);
                }
                for (bits, counts, total) in [
                    (actual.0, &above[..len], totals.0),
                    (actual.1, &below[..len], totals.1),
                ] {
                    assert_eq!(
                        bits.to_bits(),
                        default_oracle(counts, total, lanes, scorer.backend.uses_fma()).to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn default_small_table_boundary_is_independent_per_side() {
    for total_r in [0, 1, 255, 256, 257, 4096, 4097] {
        for total_l in [0, 1, 255, 256, 257, 4096, 4097] {
            let mut after_r = [0; 9];
            let mut after_l = [0; 9];
            after_r[0] = total_r / 2;
            after_r[8] = total_r - after_r[0];
            after_l[0] = total_l / 3;
            after_l[8] = total_l - after_l[0];
            let delta = after_l.map(|c| c / 2);
            let above = std::array::from_fn::<_, 9, _>(|i| after_r[i] + delta[i]);
            let below = std::array::from_fn::<_, 9, _>(|i| after_l[i] - delta[i]);
            check_default_move(&above, &below, &delta);
        }
    }
}

#[test]
fn default_floor_4096_and_integer_mask_near_float_precision_limit() {
    for total in [
        4095,
        4096,
        4097,
        8191,
        8192,
        8193,
        (1 << 24) - 1,
        1 << 24,
        (1 << 24) + 1,
        MAX_HISTOGRAM_TOTAL,
    ] {
        for rare in [1, 2, 7] {
            let mut counts = [0; 17];
            counts[0] = rare;
            counts[16] = total - rare;
            check_default_move(&counts, &[0; 17], &[0; 17]);
            let mut delta = [0; 17];
            delta[0] = rare;
            delta[16] = 251;
            check_default_move(&counts, &[0; 17], &delta);
        }
        for (scorer, _) in default_profiles() {
            let mut single = [0; 17];
            single[16] = total;
            assert_eq!(scorer.estimate_bits(&single, total), 0.0);
        }
    }
    for (scorer, width) in default_profiles() {
        // Adjacent integers can convert to equal f32s, but must not take
        // the exact-integer zero mask. FastLog2f(1) is slightly negative.
        let counts = [1 << 24, 1, 0, 0, 0, 0, 0, 0, 0];
        let total = (1 << 24) + 1;
        assert!(scorer.estimate_bits(&counts, total) > 13.0);
        let total = 1 << 20;
        let mut counts = [1; 257];
        counts[256] = total - 256;
        let actual = scorer.estimate_bits(&counts, total);
        assert_eq!(
            actual.to_bits(),
            default_oracle(&counts, total, width, scorer.backend.uses_fma()).to_bits()
        );
        let shannon: f64 = counts
            .iter()
            .map(|&c| f64::from(c) * (f64::from(total) / f64::from(c)).log2())
            .sum();
        assert!(shannon - f64::from(actual) > 2000.0);
    }
}

#[test]
fn default_length_checks_precede_mutation() {
    for (scorer, _) in default_profiles() {
        for (r_len, l_len, d_len) in [(8, 7, 8), (8, 8, 7), (0, 1, 0), (1, 1, 0)] {
            let mut above = vec![2; r_len];
            let mut below = vec![0; l_len];
            let delta = vec![1; d_len];
            let before = (above.clone(), below.clone());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                scorer.move_and_score(&mut above, &mut below, &delta, (8, 8));
            }));
            assert!(result.is_err());
            assert_eq!((above, below), before);
            assert_eq!(delta, vec![1; d_len]);
        }
    }
}

#[test]
fn count_entropy_cache_and_full_u32_fallback() {
    let table = CountEntropyTable::shared();
    assert!(std::ptr::eq(table, CountEntropyTable::shared()));
    assert_eq!(table.values.len(), MAX_LOG_TABLE_TOTAL as usize + 1);
    for count in (0..=MAX_LOG_TABLE_TOTAL).chain([
        65537,
        1 << 20,
        (1 << 24) - 1,
        1 << 24,
        (1 << 24) + 1,
        i32::MAX as u32,
        i32::MAX as u32 + 1,
        u32::MAX,
    ]) {
        assert_eq!(table.get(count), uncached_count_entropy(count));
    }
}

#[test]
fn count_entropy_native_sums_cover_tails_and_large_unsigned_counts() {
    let table = CountEntropyTable::shared();
    let values = [0, 1, 2, 3, 65535, 65536, 65537, 1 << 24, u32::MAX];
    for len in 0..=137 {
        for offset in 0..4 {
            let mut storage = vec![u32::MAX; offset + len];
            for (i, count) in storage[offset..].iter_mut().enumerate() {
                *count = values[(i + len) % values.len()];
            }
            let counts = &storage[offset..];
            let expected: f64 = counts.iter().map(|&c| uncached_count_entropy(c)).sum();
            let check = |actual: f64| {
                assert!(actual.is_finite());
                assert!((actual - expected).abs() <= 1e-9 + expected * 64.0 * f64::EPSILON);
            };
            check(table.sum(counts));
            check(table.sum_scalar(counts));
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("sse2") {
                    // SAFETY: feature detected; backend accepts all lengths.
                    check(unsafe { count_x86::sum_sse2(table, counts) });
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: feature detected, including OS AVX support.
                    check(unsafe { count_x86::sum_avx2(table, counts) });
                }
            }
        }
    }
}

#[test]
fn count_entropy_native_pair_updates_match_scalar_operations() {
    let table = CountEntropyTable::shared();
    let edges = [0, 1, 4096, 65535, 65536, 65537, i32::MAX as u32, u32::MAX];
    for &r in &edges {
        for &l in &edges {
            for weight in [0, 1, 997, 65536, u32::MAX] {
                let weight = weight.min(r).min(u32::MAX - l);
                let before = [r, l];
                let after = [r - weight, l + weight];
                let sums = [table.get(r) + 19.0, table.get(l) + 23.0];
                let expected = table.replace_pair_scalar(before, after, sums);
                let check = |actual: CountEntropyUpdate| {
                    assert_eq!(actual.before, expected.before);
                    assert_eq!(actual.after, expected.after);
                    assert_eq!(actual.sums, expected.sums);
                };
                check(table.replace_pair(before, after, sums));
                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("sse2") {
                        // SAFETY: feature detected; fixed arrays, full u32 inputs.
                        check(unsafe { count_x86::pair_sse2(table, before, after, sums) });
                    }
                    if std::is_x86_feature_detected!("avx2") {
                        // SAFETY: feature detected; gather bounds checked inside.
                        check(unsafe { count_x86::pair_avx2(table, before, after, sums) });
                    }
                }
            }
        }
    }
}

fn close(actual: f32, expected: f32) {
    assert!(actual.is_finite() && actual >= 0.0);
    assert!(
        (actual - expected).abs() <= 0.00001 * expected.max(1.0),
        "{actual} != {expected}"
    );
}

fn check_estimate(table: &LogTable, counts: &[u32]) {
    let total = table.total(counts).unwrap();
    let expected = table.estimate_bits_scalar(counts).unwrap();
    // Independent f64 mathematical probability-floor oracle.
    let model: f64 = counts
        .iter()
        .filter(|&&c| c != 0)
        .map(|&c| f64::from(c) * (f64::from(total) / f64::from(c)).log2().min(12.0))
        .sum();
    // The f32 log table loses absolute precision near log2(65536).
    // Multiplying that error by a large count is not a relative-cost error.
    let per_sample_error = if total <= table.maximum_total() {
        2.0_f64.powi(-19)
    } else {
        0.000008
    };
    let bound = f64::from(total) * per_sample_error + model.max(1.0) * 0.00001;
    assert!(
        (f64::from(expected) - model).abs() <= bound,
        "{expected} != {model}"
    );
    close(table.estimate_bits(counts).unwrap(), expected);
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: detected CPU and validated counts/table total.
            close(
                unsafe { x86::sse2::estimate(table, counts, total) },
                expected,
            );
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: detected CPU and validated counts/table total.
            close(
                unsafe { x86::avx2::estimate(table, counts, total) },
                expected,
            );
        }
    }
}

fn check_move(table: &LogTable, above: &[u32], below: &[u32], increase: &[u32]) {
    let totals = table.validate_move(above, below, increase).unwrap();
    let expected_r: Vec<_> = above.iter().zip(increase).map(|(&a, &d)| a - d).collect();
    let expected_l: Vec<_> = below.iter().zip(increase).map(|(&a, &d)| a + d).collect();
    let bits = (
        table.estimate_bits_scalar(&expected_r).unwrap(),
        table.estimate_bits_scalar(&expected_l).unwrap(),
    );
    for backend in 0..4 {
        let offset = (above.len() + backend) % 8;
        let end = offset + above.len();
        let mut rs = vec![u32::MAX; end + 1];
        let mut ls = vec![u32::MAX; end + 1];
        let mut ds = vec![u32::MAX; end + 1];
        let r = &mut rs[offset..end];
        let l = &mut ls[offset..end];
        let d = &mut ds[offset..end];
        r.copy_from_slice(above);
        l.copy_from_slice(below);
        d.copy_from_slice(increase);
        let actual = match backend {
            0 => table.move_scalar_validated(r, l, d, totals),
            1 => table.move_counts_and_log_bits(r, l, d).unwrap(),
            #[cfg(target_arch = "x86_64")]
            2 if std::is_x86_feature_detected!("sse2") => {
                // SAFETY: detected CPU, validated matching slices/totals.
                unsafe { x86::sse2::move_and_score(table, r, l, d, totals) }
            }
            #[cfg(target_arch = "x86_64")]
            3 if std::is_x86_feature_detected!("avx2") => {
                // SAFETY: detected CPU, validated matching slices/totals.
                unsafe { x86::avx2::move_and_score(table, r, l, d, totals) }
            }
            _ => continue,
        };
        assert_eq!(r, expected_r.as_slice());
        assert_eq!(l, expected_l.as_slice());
        assert!(d.iter().all(|&c| c == 0));
        close(actual.0, bits.0);
        close(actual.1, bits.1);
        for storage in [&rs, &ls, &ds] {
            assert!(storage[..offset].iter().all(|&v| v == u32::MAX));
            assert_eq!(storage[end], u32::MAX);
        }
    }
}

#[test]
fn lengths_tails_unaligned_counts_and_weighted_transfers() {
    let table = LogTable::new(MAX_LOG_TABLE_TOTAL).unwrap();
    let mut state = 0xb866_b3de_u32;
    for len in (0..=35).chain([63, 64, 65, 127, 128, 129, 255, 256, 257]) {
        for offset in 0..8 {
            let mut storage = vec![u32::MAX; offset + len];
            let above = &mut storage[offset..];
            let mut below = vec![0; len];
            let mut increase = vec![0; len];
            for i in 0..len {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                above[i] = state % 128;
                below[i] = (state >> 8) % 64;
                increase[i] = (state >> 16) % (above[i] + 1);
            }
            check_estimate(&table, above);
            check_move(&table, above, &below, &increase);
            above.fill(0);
            check_estimate(&table, above);
            if len > 0 {
                above[len - 1] = MAX_LOG_TABLE_TOTAL;
                check_estimate(&table, above);
                assert_eq!(table.estimate_bits(above).unwrap(), 0.0);
                below.fill(0);
                increase.copy_from_slice(above);
                check_move(&table, above, &below, &increase);
            }
        }
    }
}

#[test]
fn probability_floor_boundary_and_empty_sides() {
    let table = LogTable::new(MAX_LOG_TABLE_TOTAL).unwrap();
    // Pre-move population may exceed the table when both resulting sides
    // fit. Each moved count remains an exact multiplicity.
    check_move(&table, &[65536, 65536], &[0, 0], &[0, 65536]);
    for total in [4095, 4096, 4097, 8192, 65536] {
        for rare in [1, 2, 15, 16, 17] {
            let mut counts = [0; 17];
            counts[0] = rare;
            counts[8] = total - rare;
            check_estimate(&table, &counts);
            check_move(&table, &counts, &[0; 17], &counts);
            check_move(&table, &counts, &[0; 17], &[0; 17]);
        }
    }
    let zero = LogTable::new(0).unwrap();
    check_estimate(&zero, &[]);
    check_estimate(&zero, &[0; 17]);
    check_move(&zero, &[0; 17], &[0; 17], &[0; 17]);
}

#[test]
fn invalid_limits_and_moves_never_partially_mutate() {
    assert!(matches!(
        LogTable::new(65537),
        Err(MaSimdError::TableLimitTooLarge)
    ));
    assert!(matches!(
        LogTable::new(u32::MAX),
        Err(MaSimdError::TableLimitTooLarge)
    ));
    let table = LogTable::new(65536).unwrap();
    for counts in [
        &[MAX_HISTOGRAM_TOTAL + 1][..],
        &[MAX_HISTOGRAM_TOTAL, 1],
        &[u32::MAX, 1],
    ] {
        assert_eq!(table.estimate_bits(counts), Err(MaSimdError::TotalTooLarge));
        assert_eq!(
            table.estimate_bits_scalar(counts),
            Err(MaSimdError::TotalTooLarge)
        );
    }
    for (a, b, d, error) in [
        (
            vec![1; 17],
            vec![0; 16],
            vec![0; 17],
            MaSimdError::LengthMismatch,
        ),
        (
            vec![1; 17],
            vec![0; 17],
            vec![0; 16],
            MaSimdError::LengthMismatch,
        ),
        (
            vec![1; 17],
            vec![0; 17],
            {
                let mut d = vec![1; 17];
                d[16] = 2;
                d
            },
            MaSimdError::CountUnderflow,
        ),
        (vec![1], vec![u32::MAX], vec![1], MaSimdError::CountOverflow),
        (
            vec![MAX_HISTOGRAM_TOTAL, 1],
            vec![0; 2],
            vec![0; 2],
            MaSimdError::TotalTooLarge,
        ),
        (
            vec![1; 17],
            {
                let mut b = vec![0; 17];
                b[16] = MAX_HISTOGRAM_TOTAL;
                b
            },
            vec![1; 17],
            MaSimdError::TotalTooLarge,
        ),
    ] {
        let (mut above, mut below, mut increase) = (a.clone(), b.clone(), d.clone());
        assert_eq!(
            table.move_counts_and_log_bits(&mut above, &mut below, &mut increase),
            Err(error)
        );
        assert_eq!((above, below, increase), (a, b, d));
    }
}

#[test]
fn fast_log_polynomial_matches_source_accuracy_and_clipped_floor() {
    for exponent in -12..=0 {
        for step in 0..=1024 {
            let value = (1.0 + step as f32 / 1024.0) * 2.0_f32.powi(exponent);
            let error = (f64::from(fast_log2::<false>(value)) - f64::from(value).log2()).abs();
            assert!(error <= 0.000005, "{value}, error={error}");
        }
    }
    let total = 1 << 20;
    let clipped_rare = fast_contribution(1, total, inverse_total(total));
    assert!((clipped_rare - 12.0).abs() < 0.00001);
    assert!(
        20.0 - clipped_rare > 7.99,
        "Shannon rare-symbol cost must not replace the floor"
    );
    // FastLog2f(1) is deliberately not zero. The source masks exact
    // INTEGER equality, but must not mask adjacent counts rounded to 1.0.
    assert_ne!(fast_log2::<false>(1.0), 0.0);
    assert_eq!(fast_contribution(total, total, inverse_total(total)), 0.0);
    assert!(fast_contribution(1 << 24, (1 << 24) + 1, inverse_total((1 << 24) + 1)) > 1.0);
}

#[test]
fn cache_boundary_large_counts_and_integer_equality_mask() {
    for cache in [0, 256, 65536] {
        let table = LogTable::new(cache).unwrap();
        for total in [
            65535,
            65536,
            65537,
            1 << 20,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            MAX_HISTOGRAM_TOTAL,
        ] {
            for rare in [1, 2, 15, 16, 17, 255] {
                let mut counts = [0; 17];
                counts[0] = rare;
                counts[8] = total - rare;
                check_estimate(&table, &counts);
                check_move(&table, &counts, &[0; 17], &counts);
                check_move(&table, &counts, &[0; 17], &[0; 17]);
            }
            for position in 0..17 {
                let mut counts = [0; 17];
                counts[position] = total;
                check_estimate(&table, &counts);
                assert_eq!(table.estimate_bits(&counts).unwrap(), 0.0);
            }
        }
    }
}

#[test]
fn mixed_cache_fallback_splits_and_large_unaligned_tails() {
    let table = LogTable::new(65536).unwrap();
    // Both directions of the cache boundary and both-fallback splits.
    for (a, b, d) in [
        (131072, 0, 65536),
        (131073, 0, 65536),
        (131073, 0, 65537),
        (262144, 0, 131072),
        (MAX_HISTOGRAM_TOTAL, 0, 65536),
    ] {
        let mut above = [0; 17];
        let mut below = [0; 17];
        let mut increase = [0; 17];
        above[8] = a;
        below[8] = b;
        increase[8] = d;
        check_move(&table, &above, &below, &increase);
    }
    let mut state = 0x19371b_u32;
    for len in (0..=35).chain([63, 64, 65, 127, 128, 129, 257]) {
        for offset in 0..8 {
            let mut storage = vec![u32::MAX; offset + len];
            let above = &mut storage[offset..];
            let mut below = vec![0; len];
            let mut increase = vec![0; len];
            for i in 0..len {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                above[i] = state & 0x3fffff;
                below[i] = (state >> 8) & 0x1fffff;
                increase[i] = state % (above[i] + 1);
            }
            check_estimate(&table, above);
            check_move(&table, above, &below, &increase);
        }
    }
}
#[test]
fn count_log_readonly_matches_source_lane_order_and_preserves_cache_deltas() {
    let table = LogTable::shared();
    let width = DefaultScorer::new().lanes();
    for len in [0, 1, 3, 4, 7, 8, 9, 31, 32, 33, 255, 256] {
        let original: Vec<u32> = (0..len).map(|i| (i % 37 + 1) as u32).collect();
        let delta: Vec<_> = original.iter().map(|&n| n / 2).collect();
        let mut above = original.clone();
        let mut below = vec![0; len];
        let totals = (
            original.iter().sum::<u32>() - delta.iter().sum::<u32>(),
            delta.iter().sum(),
        );
        let actual = table.move_known_totals(&mut above, &mut below, &delta, totals);
        for (counts, total, bits) in [
            (above.as_slice(), totals.0, actual.0),
            (below.as_slice(), totals.1, actual.1),
        ] {
            let mut lanes = [0.0; 8];
            for (i, &count) in counts.iter().enumerate() {
                lanes[i % width] += table.contribution(count, total);
            }
            assert_eq!(bits.to_bits(), oracle_reduce(lanes, width).to_bits());
        }
        assert_eq!(below, delta);
        assert_eq!(
            above
                .iter()
                .zip(&below)
                .map(|(&r, &l)| r + l)
                .collect::<Vec<_>>(),
            original
        );
    }
    let mut above = [9];
    let mut below = [0];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            table.move_known_totals(&mut above, &mut below, &[], (9, 0));
        }))
        .is_err()
    );
    assert_eq!(above, [9]);
    assert_eq!(below, [0]);
}
