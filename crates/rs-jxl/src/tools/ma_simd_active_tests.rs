// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

use super::*;

fn profiles() -> Vec<DefaultScorer> {
    #[allow(unused_mut)]
    let mut profiles = vec![DefaultScorer::with_backend(DefaultBackend::Scalar)];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            profiles.push(DefaultScorer::with_backend(DefaultBackend::Sse2));
        }
        if std::is_x86_feature_detected!("avx2") {
            profiles.push(DefaultScorer::with_backend(DefaultBackend::Avx2));
            if std::is_x86_feature_detected!("fma") {
                profiles.push(DefaultScorer::with_backend(DefaultBackend::Avx2Fma));
                profiles.push(DefaultScorer::with_backend(DefaultBackend::Avx2Rcp));
                if DefaultBackend::detect(true) == DefaultBackend::Avx512Fma8 {
                    profiles.push(DefaultScorer::with_backend(DefaultBackend::Avx512Fma8));
                }
            }
        }
    }
    profiles
}

fn check_fixture(scorer: DefaultScorer, counts: &[u32], sparse: bool, small_hint: bool) {
    let len = counts.len();
    let stride = len + 3;
    let mut deltas = vec![u32::MAX; 5 * stride];
    let mut expected_r = counts.to_vec();
    let mut expected_l = vec![0; len];
    let mut splits = Vec::new();
    let mut expected = Vec::new();
    let blocks: Vec<_> = (0..len)
        .step_by(scorer.lanes())
        .filter(|&i| {
            counts[i..(i + scorer.lanes()).min(len)]
                .iter()
                .any(|&n| n != 0)
        })
        .collect();
    let sparse = sparse.then_some(blocks.as_slice());
    for step in 0..5 {
        let row = step * 2 % 5;
        let delta = &mut deltas[row * stride..row * stride + len];
        for (d, &n) in delta.iter_mut().zip(&expected_r) {
            *d = match step {
                0 => 0,
                1 => n.min(1),
                2 => n / 2,
                3 => n.saturating_sub(1),
                _ => n,
            };
        }
        let moved: u32 = delta.iter().sum();
        let totals = (
            expected_r.iter().sum::<u32>() - moved,
            expected_l.iter().sum::<u32>() + moved,
        );
        let bits = if let Some(blocks) = sparse {
            scorer.move_and_score_sparse(&mut expected_r, &mut expected_l, delta, totals, blocks)
        } else {
            scorer.move_and_score(&mut expected_r, &mut expected_l, delta, totals)
        };
        let split = HistogramSplit {
            bin: 100 + 3 * step,
            row,
            totals,
        };
        splits.push(split);
        expected.push((
            split.bin,
            split.row,
            split.totals,
            bits.0.to_bits(),
            bits.1.to_bits(),
            (
                expected_r.iter().filter(|&&n| n != 0).count(),
                expected_l.iter().filter(|&&n| n != 0).count(),
            ),
        ));
    }

    // A wrong small-population hint must still use the general bounded kernel
    // for any split whose supplied totals exceed the triangular table.
    let population = if small_hint { 0 } else { counts.iter().sum() };
    let mut baseline_r = counts.to_vec();
    let mut baseline_l = vec![0; len];
    let mut baseline = Vec::new();
    scorer.split_sweep(
        &mut baseline_r,
        &mut baseline_l,
        &deltas,
        stride,
        population,
        sparse,
        splits.iter().copied(),
        |split, r, l| baseline.push((split.bin, split.row, split.totals, r.to_bits(), l.to_bits())),
    );
    assert_eq!(baseline_r, expected_r);
    assert_eq!(baseline_l, expected_l);
    assert_eq!(
        baseline,
        expected
            .iter()
            .map(|&(b, row, t, r, l, _)| (b, row, t, r, l))
            .collect::<Vec<_>>()
    );

    // Offset slices and sentinels cover unaligned access and bounded stores;
    // nonzero delta padding must never affect either scores or active counts.
    let mut above = vec![u32::MAX; len + 2];
    let mut below = vec![u32::MAX; len + 2];
    above[1..len + 1].copy_from_slice(counts);
    below[1..len + 1].fill(0);
    let immutable = deltas.clone();
    let mut actual = Vec::new();
    scorer.split_sweep_with_active(
        &mut above[1..len + 1],
        &mut below[1..len + 1],
        &deltas,
        stride,
        population,
        sparse,
        splits.iter().copied(),
        |split, r, l, active| {
            actual.push((
                split.bin,
                split.row,
                split.totals,
                r.to_bits(),
                l.to_bits(),
                active,
            ))
        },
    );
    assert_eq!(
        actual, expected,
        "profile {:?}, len {len}, sparse {sparse:?}",
        scorer.backend
    );
    assert_eq!(&above[1..len + 1], expected_r);
    assert_eq!(&below[1..len + 1], expected_l);
    assert_eq!(
        [above[0], above[len + 1], below[0], below[len + 1]],
        [u32::MAX; 4]
    );
    assert_eq!(deltas, immutable);
}

#[test]
fn active_sweep_dense_matches_scores_counts_and_tail_bounds() {
    for scorer in profiles() {
        for len in (0..=33).chain([63, 64, 65, 127, 128, 129, 257]) {
            for scale in [0, 1, 33, 8192] {
                let counts: Vec<_> = (0..len)
                    .map(|i| {
                        if i % 5 == 0 {
                            0
                        } else {
                            (i as u32 * 13 % 17 + 1) * scale
                        }
                    })
                    .collect();
                check_fixture(scorer, &counts, false, false);
            }
        }
    }
}

#[test]
fn active_sweep_sparse_handles_omitted_blocks_and_tails() {
    for scorer in profiles() {
        for len in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129] {
            for scale in [0, 1, 2048] {
                for tail_only in [false, true] {
                    let counts: Vec<_> = (0..len)
                        .map(|i| {
                            let active = if tail_only {
                                i >= len / scorer.lanes() * scorer.lanes()
                            } else {
                                i / scorer.lanes() % 3 != 1
                            };
                            if active {
                                (i as u32 % 7 + 1) * scale
                            } else {
                                0
                            }
                        })
                        .collect();
                    check_fixture(scorer, &counts, true, false);
                }
            }
        }
    }
}

#[test]
fn active_sweep_signed_limit_and_small_hint_fallback() {
    for scorer in profiles() {
        for total in [0, 1, 255, 256, 257, 512, 513, 65536, 65537] {
            let mut counts = vec![0; 17];
            counts[1] = total / 2;
            counts[16] = total - counts[1];
            check_fixture(scorer, &counts, false, false);
            check_fixture(scorer, &counts, true, false);
        }
        for len in [1, 3, 4, 7, 8, 9, 15, 16, 17, 33] {
            let mut counts = vec![0; len];
            counts[0] = MAX_HISTOGRAM_TOTAL;
            for i in 1..len {
                let count = i as u32 % 3;
                counts[i] = count;
                counts[0] -= count;
            }
            for sparse in [false, true] {
                check_fixture(scorer, &counts, sparse, false);
                check_fixture(scorer, &counts, sparse, true);
                check_fixture(scorer, &vec![1; len], sparse, true);
            }
        }
    }
}

#[test]
fn active_sweep_rejects_bad_layout_before_mutation() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    for scorer in profiles() {
        for bad in 0..9 {
            let lanes = scorer.lanes();
            let mut above = vec![1; 2 * lanes];
            let mut below = vec![0; above.len()];
            let mut stride = above.len();
            let mut deltas = vec![0; stride];
            let mut blocks = vec![0, lanes];
            let mut split = HistogramSplit {
                bin: 0,
                row: 0,
                totals: (above.len() as u32, 0),
            };
            match bad {
                0 => {
                    below.pop();
                }
                1 => stride = 0,
                2 => stride -= 1,
                3 => deltas.push(0),
                4 => blocks.reverse(),
                5 => blocks.push(above.len()),
                6 => blocks.push(lanes),
                7 => split.row = usize::MAX,
                _ => blocks[1] = if lanes > 1 { 1 } else { above.len() },
            }
            let initial_r = above.clone();
            let initial_l = below.clone();
            let mut emitted = false;
            let result = catch_unwind(AssertUnwindSafe(|| {
                scorer.split_sweep_with_active(
                    &mut above,
                    &mut below,
                    &deltas,
                    stride,
                    2 * lanes as u32,
                    Some(&blocks),
                    std::iter::once(split),
                    |_, _, _, _| emitted = true,
                );
            }));
            assert!(
                result.is_err(),
                "profile {:?}, bad layout {bad}",
                scorer.backend
            );
            assert_eq!(above, initial_r);
            assert_eq!(below, initial_l);
            assert!(!emitted);
        }
    }
}
