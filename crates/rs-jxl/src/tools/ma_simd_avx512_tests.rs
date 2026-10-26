use super::*;

fn profiles() -> Option<(DefaultScorer, DefaultScorer)> {
    if DefaultBackend::detect(true) != DefaultBackend::Avx512Fma8 {
        return None;
    }
    Some((
        DefaultScorer::with_backend(DefaultBackend::Avx512Fma8),
        DefaultScorer::with_backend(DefaultBackend::Avx2Fma),
    ))
}

fn distribution(total: u32, len: usize) -> Vec<u32> {
    assert!(len != 0 || total == 0);
    if len == 0 {
        return Vec::new();
    }
    let mut counts = vec![total / len as u32; len];
    for count in counts.iter_mut().take(total as usize % len) {
        *count += 1;
    }
    counts
}

#[test]
fn avx512_requires_opt_in_and_all_runtime_features() {
    assert_ne!(DefaultBackend::detect(false), DefaultBackend::Avx512Fma8);
    let supported = std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq");
    assert_eq!(
        DefaultBackend::detect(true) == DefaultBackend::Avx512Fma8,
        supported
    );
    if !supported {
        assert_eq!(DefaultBackend::detect(true), DefaultBackend::detect(false));
    }
}

#[test]
fn avx512_keeps_eight_logical_lanes_and_reuses_the_fma_small_table() {
    let Some((wide, baseline)) = profiles() else {
        return;
    };
    assert_eq!(wide.lanes(), 8);
    assert!(wide.backend.uses_fma());
    assert!(std::ptr::eq(wide.small, baseline.small));
    assert!(std::ptr::eq(
        wide.small,
        DefaultScorer::with_backend(wide.backend).small
    ));
    for total in 0..=SMALL_HISTOGRAM_TOTAL + 1 {
        assert_eq!(wide.small_row(total), baseline.small_row(total));
    }
    for value in [1.0 / 4096.0, 0.5, 1.0, 1.000_001, 256.0, i32::MAX as f32] {
        assert_eq!(wide.log2(value).to_bits(), baseline.log2(value).to_bits());
    }
}

#[test]
fn avx512_fast_log_matches_two_fma_vectors_bitwise() {
    if profiles().is_none() {
        return;
    }
    use std::arch::x86_64::*;
    let mut values = Vec::new();
    for exponent in -12..=0 {
        for step in 0..=256 {
            values.push((1.0 + step as f32 / 256.0) * 2.0_f32.powi(exponent));
        }
    }
    values.extend([0x3f2aaaaa, 0x3f2aaaab, 0x3f2aaaac, 0x3f800000].map(f32::from_bits));
    for chunk in values.chunks(16) {
        let mut input = [1.0; 16];
        input[..chunk.len()].copy_from_slice(chunk);
        let (mut actual, mut expected) = ([0.0f32; 16], [0.0f32; 16]);
        // SAFETY: profiles checked all features; arrays contain 16 valid floats.
        unsafe {
            _mm512_storeu_ps(
                actual.as_mut_ptr(),
                x86::avx512_fma8::fast_log(_mm512_loadu_ps(input.as_ptr())),
            );
            for offset in [0, 8] {
                _mm256_storeu_ps(
                    expected.as_mut_ptr().add(offset),
                    x86::avx2_fma::fast_log(_mm256_loadu_ps(input.as_ptr().add(offset))),
                );
            }
        }
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }
}

#[test]
fn avx512_estimate_preserves_lane_order_tails_and_integer_zero_mask() {
    let Some((wide, baseline)) = profiles() else {
        return;
    };
    let mut seed = 0xb866_b3deu32;
    for len in (0..=65).chain([127, 128, 129, 255, 256, 257]) {
        let random: Vec<_> = (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) % 50001
            })
            .collect();
        let mut cases = vec![vec![0; len], random];
        if len != 0 {
            for total in [
                1,
                255,
                256,
                257,
                (1 << 24) - 1,
                1 << 24,
                (1 << 24) + 1,
                MAX_HISTOGRAM_TOTAL,
            ] {
                let mut single = vec![0; len];
                single[len - 1] = total;
                cases.push(single);
                if len > 1 {
                    let mut rare = vec![0; len];
                    rare[0] = total - 1;
                    rare[len - 1] = 1;
                    cases.push(rare);
                }
            }
        }
        for counts in cases {
            let total = counts.iter().sum();
            for offset in [0, 1, 7, 15] {
                let mut storage = vec![u32::MAX; offset];
                storage.extend_from_slice(&counts);
                assert_eq!(
                    wide.estimate_bits(&storage[offset..], total).to_bits(),
                    baseline.estimate_bits(&counts, total).to_bits(),
                    "len={len} total={total} offset={offset}"
                );
            }
        }
    }
}

#[test]
fn avx512_dense_moves_match_at_cache_boundaries_and_preserve_sentinels() {
    let Some((wide, baseline)) = profiles() else {
        return;
    };
    for len in [
        1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    ] {
        for totals in [
            (0, 0),
            (1, 0),
            (0, 1),
            (255, 256),
            (256, 257),
            (257, 256),
            (257, 257),
            (4096, 65536),
            (65536, 4096),
            (MAX_HISTOGRAM_TOTAL - 257, 257),
        ] {
            let right = distribution(totals.0, len);
            let left = distribution(totals.1, len);
            let delta: Vec<_> = left
                .iter()
                .enumerate()
                .map(|(i, &c)| if i % 3 == 0 { c } else { c / 2 })
                .collect();
            for offset in [0, 1, 15] {
                let end = offset + len;
                let mut above = vec![u32::MAX; end + 1];
                let mut below = above.clone();
                let mut changes = above.clone();
                for i in 0..len {
                    above[offset + i] = right[i] + delta[i];
                    below[offset + i] = left[i] - delta[i];
                    changes[offset + i] = delta[i];
                }
                let snapshot = changes.clone();
                let (mut expected_r, mut expected_l) = (above.clone(), below.clone());
                let expected = baseline.move_and_score(
                    &mut expected_r[offset..end],
                    &mut expected_l[offset..end],
                    &changes[offset..end],
                    totals,
                );
                let actual = wide.move_and_score(
                    &mut above[offset..end],
                    &mut below[offset..end],
                    &changes[offset..end],
                    totals,
                );
                assert_eq!(
                    (actual.0.to_bits(), actual.1.to_bits()),
                    (expected.0.to_bits(), expected.1.to_bits()),
                    "len={len} totals={totals:?}"
                );
                assert_eq!(above, expected_r);
                assert_eq!(below, expected_l);
                assert_eq!(changes, snapshot);
            }
        }
    }
}

#[test]
fn avx512_integer_moves_keep_wrapping_arithmetic_and_bounded_tails() {
    let Some((wide, baseline)) = profiles() else {
        return;
    };
    for len in (0..=65).chain([127, 128, 129, 255, 256, 257]) {
        for offset in [0, 1, 15] {
            let end = offset + len;
            let mut above = vec![12345; end + 1];
            let mut below = vec![54321; end + 1];
            let mut delta = vec![u32::MAX; end + 1];
            for i in 0..len {
                above[offset + i] = [0, u32::MAX, 1, 1 << 31][i % 4];
                below[offset + i] = [u32::MAX, 1 << 31, 0, 17][i % 4];
                delta[offset + i] = [1, 1 << 31, u32::MAX, 0][i % 4];
            }
            let snapshot = delta.clone();
            let (mut expected_r, mut expected_l) = (above.clone(), below.clone());
            baseline.move_counts(
                &mut expected_r[offset..end],
                &mut expected_l[offset..end],
                &delta[offset..end],
            );
            wide.move_counts(
                &mut above[offset..end],
                &mut below[offset..end],
                &delta[offset..end],
            );
            assert_eq!(above, expected_r);
            assert_eq!(below, expected_l);
            assert_eq!(delta, snapshot);
        }
    }
}

#[test]
fn avx512_sweeps_keep_packed_row_order_sparse_blocks_and_cache_transitions() {
    let Some((wide, baseline)) = profiles() else {
        return;
    };
    for len in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257] {
        for sparse in [false, true] {
            let live: Vec<_> = (0..len).filter(|&i| !sparse || i / 8 % 2 == 0).collect();
            let blocks: Vec<_> = (0..len).step_by(8).filter(|&i| i / 8 % 2 == 0).collect();
            let stride = len + 5;
            let mut changes = vec![u32::MAX; 4 * stride];
            let mut above = vec![0; len];
            for (row, total) in [1, 256, 257, 1024].into_iter().enumerate() {
                changes[row * stride..row * stride + len].fill(0);
                for (&i, count) in live.iter().zip(distribution(total, live.len())) {
                    changes[row * stride + i] = count;
                    above[i] += count;
                }
            }
            let mut remaining = above.iter().sum::<u32>();
            let total = remaining;
            let splits: Vec<_> = [2, 0, 3, 1]
                .into_iter()
                .enumerate()
                .map(|(bin, row)| {
                    remaining -= changes[row * stride..row * stride + len]
                        .iter()
                        .sum::<u32>();
                    HistogramSplit {
                        bin: bin * 3,
                        row,
                        totals: (remaining, total - remaining),
                    }
                })
                .collect();
            let snapshot = changes.clone();
            let run = |scorer: DefaultScorer| {
                let mut r = above.clone();
                let mut l = vec![0; len];
                let mut output = Vec::new();
                scorer.split_sweep(
                    &mut r,
                    &mut l,
                    &changes,
                    stride,
                    sparse.then_some(blocks.as_slice()),
                    splits.iter().copied(),
                    |split, rb, lb| {
                        output.push((
                            split.bin,
                            split.row,
                            split.totals,
                            rb.to_bits(),
                            lb.to_bits(),
                        ));
                    },
                );
                (r, l, output)
            };
            assert_eq!(run(wide), run(baseline), "len={len} sparse={sparse}");
            assert_eq!(changes, snapshot);
        }
    }
}

#[test]
fn avx512_wrappers_reject_invalid_slices_and_sparse_blocks_before_mutation() {
    let Some((wide, _)) = profiles() else { return };
    for blocks in [vec![1], vec![16], vec![0, 0], vec![8, 0]] {
        let mut r = vec![1; 16];
        let mut l = vec![0; 16];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            wide.move_and_score_sparse(&mut r, &mut l, &[0; 16], (16, 0), &blocks);
        }));
        assert!(result.is_err());
        assert_eq!(r, [1; 16]);
        assert_eq!(l, [0; 16]);
    }
    for integer_only in [false, true] {
        let mut r = vec![1; 16];
        let mut l = vec![0; 16];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if integer_only {
                wide.move_counts(&mut r, &mut l, &[0; 15]);
            } else {
                wide.move_and_score(&mut r, &mut l, &[0; 15], (16, 0));
            }
        }));
        assert!(result.is_err());
        assert_eq!(r, [1; 16]);
        assert_eq!(l, [0; 16]);
    }
}
