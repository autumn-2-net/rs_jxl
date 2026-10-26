//! Node-local predictor dominance for complete, small-population MA sweeps.
//!
//! The caller must gate this helper to weighted total <= 256 and default
//! scoring/search. Both sides then use the default scorer's exhaustive small
//! entropy table, whose costs are nonnegative. No large-population approximate
//! entropy bound, description cost, or partial/refinement search is assumed.

/// Prove that this predictor cannot replace either incumbent at any threshold.
///
/// `splits` must contain every occupied threshold, in ascending bin order, with
/// its histogram row. The terminal occupied bin is not a threshold: its extra
/// bits belong to `parent_extra`, but not to the final below-side prefix. Rows
/// and extra bits describe this predictor on the same live node as the
/// incumbents, which retain their own predictors and penalties. Every prefix
/// must fit in `parent_extra`. Incumbent costs may be uninitialized.
///
/// Substituting zero for nonnegative entropy bounds each side's production f32
/// expression from below. Rounded addition/subtraction are monotone, including
/// addition of a negative predictor penalty. Equality cannot replace an
/// incumbent because update_cost uses strict improvement. Keep the two extra
/// bit conversions separate, especially for the above-side subtraction.
///
/// A true result skips the entire predictor entropy sweep after histogram
/// construction; this function changes neither counts nor incumbents. A false
/// result is inconclusive, including an empty iterator. Never inherit this
/// numerical rejection into a child, whose distribution and penalties differ.
pub(super) fn predictor_cannot_improve(
    splits: impl Iterator<Item = (usize, usize)>,
    extra: &[u64],
    parent_extra: u64,
    penalty: f32,
    below: &[super::Cost],
    above: &[super::Cost],
) -> bool {
    let mut extra_below = 0u64;
    let mut saw_threshold = false;
    for (bin, row) in splits {
        saw_threshold = true;
        extra_below += extra[row];
        debug_assert!(extra_below <= parent_extra);
        let lower_below = 0.0 + extra_below as f32;
        let lower_above = (0.0 + parent_extra as f32) - extra_below as f32;
        if !(lower_below + penalty >= below[bin].bits + below[bin].penalty
            && lower_above + penalty >= above[bin].bits + above[bin].penalty)
        {
            return false;
        }
    }
    saw_threshold
}

#[cfg(test)]
mod tests {
    use super::super::Cost;
    use super::predictor_cannot_improve;
    use crate::tools::ma_simd::{DefaultScorer, HistogramSplit};

    fn cost(bits: f32, penalty: f32) -> Cost {
        Cost {
            bits,
            penalty,
            predictor: 0,
        }
    }

    #[test]
    fn empty_and_uninitialized_incumbents_do_not_prune() {
        assert!(!predictor_cannot_improve(
            std::iter::empty(),
            &[],
            0,
            0.0,
            &[],
            &[],
        ));
        assert!(!predictor_cannot_improve(
            std::iter::once((0, 0)),
            &[3],
            7,
            0.0,
            &[Cost::default()],
            &[Cost::default()],
        ));
    }

    #[test]
    fn penalties_and_strict_ties_match_predictor_selection() {
        for penalty in [-1e-8, 0.0, 1e-8, 4.0] {
            for incumbent_penalty in [-1e-8, 0.0, 1e-8, 4.0] {
                let incumbent = [cost(0.0, incumbent_penalty)];
                assert_eq!(
                    predictor_cannot_improve(
                        std::iter::once((0, 0)),
                        &[0],
                        0,
                        penalty,
                        &incumbent,
                        &incumbent,
                    ),
                    penalty >= incumbent_penalty,
                );
            }
        }
        // At one bit, either epsilon rounds away; these remain strict ties.
        for penalty in [-1e-8, 1e-8] {
            assert!(predictor_cannot_improve(
                std::iter::once((0, 0)),
                &[1],
                2,
                penalty,
                &[cost(1.0, 0.0)],
                &[cost(1.0, 0.0)],
            ));
        }
    }

    #[test]
    fn either_side_can_prevent_rejection() {
        for (below, above, expected) in [
            (3.0, 7.0, true),
            (2.0, 6.0, true),
            (4.0, 7.0, false),
            (3.0, 8.0, false),
        ] {
            assert_eq!(
                predictor_cannot_improve(
                    std::iter::once((0, 0)),
                    &[3],
                    10,
                    0.0,
                    &[cost(below, 0.0)],
                    &[cost(above, 0.0)],
                ),
                expected,
            );
        }
    }

    #[test]
    fn sparse_rows_use_ordered_prefixes_and_check_every_threshold() {
        let splits = [(2, 1), (6, 4), (9, 7)];
        let extra = [0, 2, 0, 0, 3, 0, 0, 5, 7];
        let mut below = [Cost::default(); 10];
        let mut above = [Cost::default(); 10];
        for (bin, prefix) in [(2, 2), (6, 5), (9, 10)] {
            below[bin] = cost(prefix as f32, 0.0);
            above[bin] = cost((17 - prefix) as f32, 0.0);
        }
        assert!(predictor_cannot_improve(
            splits.into_iter(),
            &extra,
            17,
            0.0,
            &below,
            &above,
        ));
        below[9].bits = 10.5;
        assert!(!predictor_cannot_improve(
            splits.into_iter(),
            &extra,
            17,
            0.0,
            &below,
            &above,
        ));
        below[9].bits = 10.0;
        above[9].bits = 7.5;
        assert!(!predictor_cannot_improve(
            splits.into_iter(),
            &extra,
            17,
            0.0,
            &below,
            &above,
        ));
    }

    #[test]
    fn possible_improvement_returns_before_consuming_more_thresholds() {
        let splits = std::iter::once((0, 0)).chain(std::iter::from_fn(|| {
            panic!("the first threshold already prevents pruning")
        }));
        assert!(!predictor_cannot_improve(
            splits,
            &[0],
            0,
            0.0,
            &[cost(1.0, 0.0)],
            &[cost(0.0, 0.0)],
        ));
    }

    #[test]
    fn float_conversion_and_prefix_arithmetic_follow_production_order() {
        // Synthetic large extras stress arithmetic independently of sample limits.
        let boundary = 1u64 << 24;
        for (parent, prefix, incumbent_above, expected) in [
            (boundary + 1, boundary, 0.5, false),
            (boundary + 3, boundary + 1, 3.0, true),
        ] {
            assert_ne!((parent as f32) - (prefix as f32), (parent - prefix) as f32);
            assert_eq!(
                predictor_cannot_improve(
                    std::iter::once((0, 0)),
                    &[prefix],
                    parent,
                    0.0,
                    &[cost(prefix as f32, 0.0)],
                    &[cost(incumbent_above, 0.0)],
                ),
                expected,
            );
        }
        // Prefixes must accumulate in u64: two successive f32 +1 updates vanish.
        let extra = [boundary, 1, 1, 1];
        let below = [
            cost(boundary as f32, 0.0),
            cost((boundary + 1) as f32, 0.0),
            cost((boundary + 2) as f32, 0.0),
        ];
        let above = [cost(4.0, 0.0), cost(4.0, 0.0), cost(2.0, 0.0)];
        assert!(predictor_cannot_improve(
            [(0, 0), (1, 1), (2, 2)].into_iter(),
            &extra,
            boundary + 3,
            0.0,
            &below,
            &above,
        ));
    }

    #[test]
    fn exhaustive_small_table_count_pairs_have_nonnegative_side_scores() {
        let scorer = DefaultScorer::new();
        // Every small-table (count, total) pair occurs in a valid two-symbol
        // histogram. Exercise both sides, vector lanes, zero padding, and tails.
        for total in 0..=256 {
            for count in 0..=total {
                for (len, first, last) in [(2, 0, 1), (9, 0, 8), (17, 7, 16)] {
                    let mut counts = [0u32; 17];
                    counts[first] = count;
                    counts[last] = total - count;
                    for moved in [false, true] {
                        let mut above = counts;
                        let mut below = [0; 17];
                        let delta = if moved { counts } else { [0; 17] };
                        let totals = if moved { (0, total) } else { (total, 0) };
                        let (rbits, lbits) = scorer.move_and_score(
                            &mut above[..len],
                            &mut below[..len],
                            &delta[..len],
                            totals,
                        );
                        assert!(rbits.is_finite() && rbits >= 0.0);
                        assert!(lbits.is_finite() && lbits >= 0.0);
                        if count == 0 || count == total {
                            assert_eq!((rbits, lbits), (0.0, 0.0));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exhaustive_small_three_symbol_moves_have_nonnegative_side_scores() {
        let scorer = DefaultScorer::new();
        for a in 0..=16 {
            for b in 0..=16 - a {
                for c in 0..=16 - a - b {
                    for da in 0..=a {
                        for db in 0..=b {
                            for dc in 0..=c {
                                let mut above = [a, 0, 0, b, 0, 0, 0, 0, c];
                                let mut below = [0; 9];
                                let delta = [da, 0, 0, db, 0, 0, 0, 0, dc];
                                let moved = da + db + dc;
                                let (rbits, lbits) = scorer.move_and_score(
                                    &mut above,
                                    &mut below,
                                    &delta,
                                    (a + b + c - moved, moved),
                                );
                                assert!(rbits.is_finite() && rbits >= 0.0);
                                assert!(lbits.is_finite() && lbits >= 0.0);
                            }
                        }
                    }
                }
            }
        }
    }

    fn full_scores(
        deltas: &[u32],
        extra: &[u64],
        weights: &[u32],
        symbols: usize,
        splits: &[(usize, usize)],
    ) -> Vec<(usize, f32, f32)> {
        let total: u32 = weights.iter().sum();
        assert!(total <= 256);
        let parent_extra: u64 = extra.iter().sum();
        let mut above = vec![0; symbols];
        for row in deltas.chunks_exact(symbols) {
            for (count, &delta) in above.iter_mut().zip(row) {
                *count += delta;
            }
        }
        let mut below = vec![0; symbols];
        let mut weight_below = 0;
        let thresholds = splits.iter().map(|&(bin, row)| {
            weight_below += weights[row];
            HistogramSplit {
                bin,
                row,
                totals: (total - weight_below, weight_below),
            }
        });
        let mut extra_below = 0u64;
        let mut scores = Vec::new();
        DefaultScorer::new().split_sweep(
            &mut above,
            &mut below,
            deltas,
            symbols,
            total,
            None,
            thresholds,
            |split, rbits, lbits| {
                assert!(rbits.is_finite() && rbits >= 0.0);
                assert!(lbits.is_finite() && lbits >= 0.0);
                extra_below += extra[split.row];
                scores.push((
                    split.bin,
                    lbits + extra_below as f32,
                    (rbits + parent_extra as f32) - extra_below as f32,
                ));
            },
        );
        scores
    }

    #[test]
    fn generated_weighted_bins_match_full_scoring_rejections() {
        let mut state = 0xb866_b3de_u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };
        let mut pruned = 0;
        let mut improving = 0;
        for case in 0..256 {
            let symbols = [1, 2, 7, 8, 9, 17, 33][case % 7];
            let occupied = 2 + case % 5;
            let compact = (case / 4) % 2 == 0;
            let bin_rows: Vec<_> = (0..occupied)
                .map(|i| (1 + 3 * i, if compact { i } else { 1 + 3 * i }))
                .collect();
            let rows = bin_rows.last().unwrap().1 + 1;
            let slots = bin_rows.last().unwrap().0 + 1;
            let splits = &bin_rows[..occupied - 1];
            let total = if case % 4 == 0 {
                256
            } else {
                occupied as u32 + next() % (257 - occupied as u32)
            };
            let mut records: Vec<_> = (0..occupied).map(|i| (i, 1u32)).collect();
            let mut remaining = total - occupied as u32;
            while remaining != 0 {
                let weight = 1 + next() % remaining.min(7);
                records.push((next() as usize % occupied, weight));
                remaining -= weight;
            }
            let mut deltas = [vec![0u32; rows * symbols], vec![0; rows * symbols]];
            let mut extras = [vec![0u64; rows], vec![0; rows]];
            let mut weights = vec![0; rows];
            for (ordinal, weight) in records {
                let row = bin_rows[ordinal].1;
                weights[row] += weight;
                for predictor in 0..2 {
                    let token = next() as usize % symbols;
                    let bits = match (case % 4, predictor) {
                        (0, 0) | (1, 1) | (2, _) => 0,
                        (0, 1) | (1, 0) => 31,
                        _ => next() % 32,
                    };
                    deltas[predictor][row * symbols + token] += weight;
                    extras[predictor][row] += u64::from(bits) * u64::from(weight);
                }
            }
            let penalties = [-1e-8, 0.0, 1e-8, 4.0];
            let incumbent_penalty = penalties[(case / 16) % 4];
            let penalty = penalties[(case / 4) % 4];
            let mut below = vec![Cost::default(); slots];
            let mut above = vec![Cost::default(); slots];
            for (bin, lbits, rbits) in
                full_scores(&deltas[0], &extras[0], &weights, symbols, splits)
            {
                below[bin] = cost(lbits, incumbent_penalty);
                above[bin] = cost(rbits, incumbent_penalty);
            }
            let cannot_improve = predictor_cannot_improve(
                splits.iter().copied(),
                &extras[1],
                extras[1].iter().sum(),
                penalty,
                &below,
                &above,
            );
            let scores = full_scores(&deltas[1], &extras[1], &weights, symbols, splits);
            let oracle_cannot_improve = scores.iter().all(|&(bin, lbits, rbits)| {
                lbits + penalty >= below[bin].bits + below[bin].penalty
                    && rbits + penalty >= above[bin].bits + above[bin].penalty
            });
            // The floor is conservative: false need not mean an actual win.
            assert!(!cannot_improve || oracle_cannot_improve, "case {case}");
            if symbols == 1 {
                assert_eq!(cannot_improve, oracle_cannot_improve, "case {case}");
            }
            pruned += usize::from(cannot_improve);
            improving += usize::from(!oracle_cannot_improve);
        }
        assert!(pruned > 0);
        assert!(improving > 0);
    }
}
