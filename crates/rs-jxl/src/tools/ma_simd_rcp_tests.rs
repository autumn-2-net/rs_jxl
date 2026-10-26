use super::*;
use std::arch::x86_64::*;

fn available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

#[test]
fn corrected_reciprocal_log_is_finite_and_close_across_probability_range() {
    if !available() {
        return;
    }
    for block in 0..16384 {
        let values: [f32; 8] = std::array::from_fn(|lane| {
            let index = block * 8 + lane;
            MIN_PROBABILITY + (1.0 - MIN_PROBABILITY) * index as f32 / 131071.0
        });
        let mut actual = [0.0; 8];
        let mut expected = [0.0; 8];
        // SAFETY: feature-guarded; exactly eight readable/writable elements.
        unsafe {
            let input = _mm256_loadu_ps(values.as_ptr());
            _mm256_storeu_ps(actual.as_mut_ptr(), x86::avx2_rcp::fast_log(input));
            _mm256_storeu_ps(expected.as_mut_ptr(), x86::avx2_fma::fast_log(input));
        }
        for (a, b) in actual.into_iter().zip(expected) {
            assert!(a.is_finite() && (a - b).abs() <= 2e-6, "{a} vs {b}");
        }
    }
}

#[test]
fn corrected_reciprocal_preserves_counts_tails_and_split_sweep_results() {
    if !available() {
        return;
    }
    let scorer = DefaultScorer::with_backend(DefaultBackend::Avx2Rcp);
    let reference = DefaultScorer::with_backend(DefaultBackend::Avx2Fma);
    for len in 1..=73 {
        for scale in [1, 17, 1024, 65536] {
            let parent: Vec<u32> = (0..len)
                .map(|i| ((i * 17 + 3) % 31) as u32 * scale)
                .collect();
            let total = parent.iter().sum();
            let a = scorer.estimate_bits(&parent, total);
            let b = reference.estimate_bits(&parent, total);
            assert!((a - b).abs() <= 0.01 + b.abs() * 2e-6);
            let mut above = parent.clone();
            let mut below = vec![0; len];
            let mut expected_above = above.clone();
            let mut expected_below = below.clone();
            let mut deltas = Vec::new();
            let mut remaining = parent.clone();
            let mut splits = Vec::new();
            let mut mass = 0;
            for row in 0..7 {
                for value in &mut remaining {
                    let delta = if row == 6 {
                        *value
                    } else {
                        value.div_ceil(7 - row as u32)
                    };
                    *value -= delta;
                    mass += delta;
                    deltas.push(delta);
                }
                splits.push(HistogramSplit {
                    bin: row,
                    row,
                    totals: (total - mass, mass),
                });
            }
            let mut observed = Vec::new();
            scorer.split_sweep(
                &mut above,
                &mut below,
                &deltas,
                len,
                None,
                splits.iter().copied(),
                |_, r, l| observed.push((r, l)),
            );
            for (split, (a, b)) in splits.into_iter().zip(observed) {
                let delta = &deltas[split.row * len..(split.row + 1) * len];
                let expected = reference.move_and_score(
                    &mut expected_above,
                    &mut expected_below,
                    delta,
                    split.totals,
                );
                assert!((a - expected.0).abs() <= 0.01 + expected.0.abs() * 2e-6);
                assert!((b - expected.1).abs() <= 0.01 + expected.1.abs() * 2e-6);
            }
            assert_eq!(above, expected_above);
            assert_eq!(below, expected_below);
            assert_eq!(below, parent);
        }
    }
}

#[test]
fn corrected_reciprocal_sparse_sweep_never_touches_proven_zero_blocks() {
    if !available() {
        return;
    }
    let scorer = DefaultScorer::with_backend(DefaultBackend::Avx2Rcp);
    for len in [9, 17, 25, 33, 65] {
        let mut above = vec![0; len];
        above[0] = 100000;
        above[len - 1] = 7;
        let mut delta = vec![0; len];
        delta[0] = 40000;
        delta[len - 1] = 3;
        let mut below = vec![0; len];
        let mut dense_above = above.clone();
        let mut dense_below = below.clone();
        let blocks = [0, (len - 1) / 8 * 8];
        let sparse =
            scorer.move_and_score_sparse(&mut above, &mut below, &delta, (60004, 40003), &blocks);
        let dense =
            scorer.move_and_score(&mut dense_above, &mut dense_below, &delta, (60004, 40003));
        assert_eq!(sparse, dense);
        assert_eq!(above, dense_above);
        assert_eq!(below, dense_below);
    }
    for total in [0, 1, 256, 65536, i32::MAX as u32] {
        assert_eq!(scorer.estimate_bits(&[0, total, 0], total), 0.0);
    }
    assert_eq!(scorer.estimate_bits(&[], 0), 0.0);
    let reference = DefaultScorer::with_backend(DefaultBackend::Avx2Fma);
    let parent = [i32::MAX as u32 - 65540, 1, 2, 65536, 1];
    let expected = reference.estimate_bits(&parent, i32::MAX as u32);
    let actual = scorer.estimate_bits(&parent, i32::MAX as u32);
    assert!(actual.is_finite() && (actual - expected).abs() <= 0.1 + expected.abs() * 2e-6);
}

#[test]
fn corrected_reciprocal_handles_adjacent_reduction_boundaries() {
    if !available() {
        return;
    }
    for exponent in -12..=0 {
        let center = (0x3f2aaaabu32 as i32 + (exponent * (1 << 23))) as u32;
        for block in [-16i32, -8, 0, 8] {
            let values: [f32; 8] = std::array::from_fn(|lane| {
                f32::from_bits(center.wrapping_add_signed(block + lane as i32))
            });
            let mut actual = [0.0; 8];
            // SAFETY: guarded CPU features and full eight-element storage.
            unsafe {
                _mm256_storeu_ps(
                    actual.as_mut_ptr(),
                    x86::avx2_rcp::fast_log(_mm256_loadu_ps(values.as_ptr())),
                );
            }
            for (input, result) in values.into_iter().zip(actual) {
                assert!(result.is_finite() && (result - fast_log2::<true>(input)).abs() <= 2e-6);
            }
        }
    }
}
