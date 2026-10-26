// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! Optional enc_ma.cc scoring dispatch and model description, frozen b866b3d.

use super::TrainOptions;
use crate::Result;
use crate::tools::ma_entropy::SourceClippedEntropy;
use crate::tools::ma_simd::{CountEntropyTable, DefaultScorer, LogTable};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ClippedEntropyMode {
    #[default]
    Disabled,
    /// JXL_MA_CLIPPED_ENTROPY=1.
    Always,
    /// JXL_MA_CLIPPED_ENTROPY=2; distinct records, not weight.
    AtLeast2048,
    /// JXL_MA_CLIPPED_ENTROPY=3.
    AtLeast16384,
    /// JXL_MA_CLIPPED_ENTROPY=4; sliding rare-count cursor.
    Sliding,
    /// JXL_MA_CLIPPED_ENTROPY=5.
    SlidingAtLeast2048,
}

impl ClippedEntropyMode {
    fn enabled(self, records: usize) -> bool {
        match self {
            Self::Disabled => false,
            Self::Always | Self::Sliding => true,
            Self::AtLeast2048 | Self::SlidingAtLeast2048 => records >= 2048,
            Self::AtLeast16384 => records >= 16384,
        }
    }

    fn sliding(self) -> bool {
        matches!(self, Self::Sliding | Self::SlidingAtLeast2048)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DescriptionCost {
    #[default]
    Disabled,
    /// JXL_MA_DESCRIPTION_COST=1.
    Quarter,
    /// JXL_MA_DESCRIPTION_COST=2.
    Half,
    /// JXL_MA_DESCRIPTION_COST=3.
    Full,
}

impl DescriptionCost {
    pub(super) fn cost(self, scorer: &DefaultScorer, counts: &[u32], total: u32) -> f32 {
        if self == Self::Disabled {
            return 0.0;
        }
        self.cost_for_active(
            scorer,
            counts.iter().filter(|&&count| count != 0).count(),
            total,
        )
    }

    pub(super) fn cost_for_active(self, scorer: &DefaultScorer, active: usize, total: u32) -> f32 {
        let weight = match self {
            Self::Disabled => return 0.0,
            Self::Quarter => 0.25,
            Self::Half => 0.5,
            Self::Full => 1.0,
        };
        if active > 1 {
            weight * 0.5 * (active - 1) as f32 * scorer.log2(total as f32 + 1.0)
        } else {
            0.0
        }
    }
}

/// Source scoring experiments, all off by default. FastEntropy takes precedence
/// over clipped entropy and disables token compaction and sparse block scoring.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScoringOptions {
    pub count_log: bool,
    pub fast_entropy: bool,
    pub clipped: ClippedEntropyMode,
    pub description: DescriptionCost,
}

impl ScoringOptions {
    pub fn apply(&self, options: &mut TrainOptions) {
        options.scoring = *self;
    }

    pub(super) fn leaf_bits(self, scorer: &DefaultScorer, counts: &[u32], total: u32) -> f32 {
        if self.fast_entropy {
            let table = CountEntropyTable::shared();
            (table.get(total) - counts.iter().map(|&count| table.get(count)).sum::<f64>()).max(0.0)
                as f32
        } else {
            // Clipped/count-log experiments do not replace the source leaf baseline.
            scorer.estimate_bits(counts, total)
        }
    }
}

#[derive(Default)]
pub(super) struct ScoreScratch {
    clipped: [SourceClippedEntropy; 2],
    incremental: bool,
    sums: [f64; 2],
    #[cfg(test)]
    pub(super) routes: [usize; 5],
}

impl ScoreScratch {
    pub(super) fn reset(
        &mut self,
        options: ScoringOptions,
        counts: &[u32],
        total: u32,
        records: usize,
    ) -> Result<()> {
        self.incremental = !options.fast_entropy && options.clipped.enabled(records);
        self.sums = [0.0; 2];
        if self.incremental {
            self.clipped[0].reset(counts, total, options.clipped.sliding())?;
            self.clipped[1].reset(&[], total, options.clipped.sliding())?;
        } else if options.fast_entropy {
            let table = CountEntropyTable::shared();
            self.sums[0] = counts.iter().map(|&count| table.get(count)).sum();
        }
        #[cfg(test)]
        {
            self.routes = [0; 5];
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn move_and_score(
        &mut self,
        options: ScoringOptions,
        scorer: &DefaultScorer,
        above: &mut [u32],
        below: &mut [u32],
        delta: &[u32],
        totals: (u32, u32),
        sparse: Option<&[usize]>,
    ) -> (f32, f32) {
        assert_eq!(above.len(), below.len());
        assert_eq!(above.len(), delta.len());
        if self.incremental || options.fast_entropy {
            let table = CountEntropyTable::shared();
            for ((r, l), &d) in above.iter_mut().zip(below).zip(delta) {
                if d == 0 {
                    continue;
                }
                if self.incremental {
                    self.clipped[0].replace(*r, *r - d);
                    self.clipped[1].replace(*l, *l + d);
                } else {
                    self.sums = table
                        .replace_pair([*r, *l], [*r - d, *l + d], self.sums)
                        .sums;
                }
                *r -= d;
                *l += d;
            }
            if self.incremental {
                #[cfg(test)]
                {
                    self.routes[0] += 1;
                }
                (
                    self.clipped[0].bits(totals.0),
                    self.clipped[1].bits(totals.1),
                )
            } else {
                #[cfg(test)]
                {
                    self.routes[1] += 1;
                }
                (
                    (table.get(totals.0) - self.sums[0]).max(0.0) as f32,
                    (table.get(totals.1) - self.sums[1]).max(0.0) as f32,
                )
            }
        } else if options.count_log
            && totals.0 <= 65536
            && totals.1 <= 65536
            && u64::from(totals.0) + u64::from(totals.1) > 256
        {
            #[cfg(test)]
            {
                self.routes[2] += 1;
            }
            LogTable::shared().move_known_totals(above, below, delta, totals)
        } else if let Some(blocks) = sparse {
            #[cfg(test)]
            {
                self.routes[3] += 1;
            }
            scorer.move_and_score_sparse(above, below, delta, totals, blocks)
        } else {
            #[cfg(test)]
            {
                self.routes[4] += 1;
            }
            scorer.move_and_score(above, below, delta, totals)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_options_preserve_source_modes_and_forward_only_scoring() {
        let mut train = TrainOptions {
            histogram_shards: true,
            histogram_cache_bytes: 19,
            require_complete_search: true,
            ..TrainOptions::default()
        };
        let options = ScoringOptions {
            count_log: true,
            fast_entropy: true,
            clipped: ClippedEntropyMode::SlidingAtLeast2048,
            description: DescriptionCost::Half,
        };
        options.apply(&mut train);
        assert_eq!(train.scoring, options);
        ScoringOptions::default().apply(&mut train);
        assert_eq!(train.scoring, ScoringOptions::default());
        assert!(train.histogram_shards && train.require_complete_search);
        assert_eq!(train.histogram_cache_bytes, 19);
        for records in [0, 1, 2047, 2048, 16383, 16384] {
            for (mode, limit) in [
                (ClippedEntropyMode::Disabled, usize::MAX),
                (ClippedEntropyMode::Always, 0),
                (ClippedEntropyMode::AtLeast2048, 2048),
                (ClippedEntropyMode::AtLeast16384, 16384),
                (ClippedEntropyMode::Sliding, 0),
                (ClippedEntropyMode::SlidingAtLeast2048, 2048),
            ] {
                assert_eq!(mode.enabled(records), records >= limit);
            }
        }
    }

    #[test]
    fn scoring_source_dispatch_priority_and_count_log_boundaries() {
        let scorer = DefaultScorer::new();
        let mut scratch = ScoreScratch::default();
        for (right, left) in [
            (128, 128),
            (128, 129),
            (65536, 1),
            (65537, 1),
            (65536, 65536),
            (65536, 65537),
        ] {
            let counts = [right + left];
            let total = right + left;
            for (fast, clipped, records, route) in [
                (true, ClippedEntropyMode::Always, 1, 1),
                (false, ClippedEntropyMode::Always, 1, 0),
                (false, ClippedEntropyMode::AtLeast2048, 2048, 0),
                (
                    false,
                    ClippedEntropyMode::AtLeast2048,
                    2047,
                    if right <= 65536 && left <= 65536 && total > 256 {
                        2
                    } else {
                        3
                    },
                ),
            ] {
                let options = ScoringOptions {
                    fast_entropy: fast,
                    count_log: true,
                    clipped,
                    ..ScoringOptions::default()
                };
                scratch.reset(options, &counts, total, records).unwrap();
                let mut r = counts;
                let mut l = [0];
                let delta = [left];
                let actual = scratch.move_and_score(
                    options,
                    &scorer,
                    &mut r,
                    &mut l,
                    &delta,
                    (right, left),
                    Some(&[0]),
                );
                assert_eq!(actual, (0.0, 0.0));
                assert_eq!(r, [right]);
                assert_eq!(l, [left]);
                assert_eq!(delta, [left]);
                assert_eq!(scratch.routes[route], 1);
                assert_eq!(scratch.routes.iter().sum::<usize>(), 1);
            }
            let options = ScoringOptions::default();
            scratch.reset(options, &counts, total, 1).unwrap();
            scratch.move_and_score(
                options,
                &scorer,
                &mut [total],
                &mut [0],
                &[left],
                (right, left),
                None,
            );
            assert_eq!(scratch.routes, [0, 0, 0, 0, 1]);
        }
    }

    #[test]
    fn scoring_incremental_modes_preserve_source_updates_and_leaf_baseline() {
        let scorer = DefaultScorer::new();
        let table = CountEntropyTable::shared();
        let counts = [1, 2, 17, 4096, 65535, 100000];
        let total: u32 = counts.iter().sum();
        let mut scratch = ScoreScratch::default();
        for mode in [ClippedEntropyMode::Always, ClippedEntropyMode::Sliding] {
            for fast in [false, true] {
                let options = ScoringOptions {
                    fast_entropy: fast,
                    clipped: mode,
                    count_log: true,
                    ..ScoringOptions::default()
                };
                scratch.reset(options, &counts, total, 16384).unwrap();
                let mut r = counts;
                let mut l = [0; 6];
                let mut sums = [counts.iter().map(|&n| table.get(n)).sum::<f64>(), 0.0];
                let baseline = if fast {
                    (table.get(total) - sums[0]).max(0.0) as f32
                } else {
                    scorer.estimate_bits(&counts, total)
                };
                assert_eq!(
                    options.leaf_bits(&scorer, &counts, total).to_bits(),
                    baseline.to_bits()
                );
                for step in 0..4 {
                    let delta = r.map(|n| if step == 3 { n } else { n.div_ceil(4) });
                    let next_r: Vec<_> = r.iter().zip(delta).map(|(&n, d)| n - d).collect();
                    let next_l: Vec<_> = l.iter().zip(delta).map(|(&n, d)| n + d).collect();
                    for i in 0..r.len() {
                        if delta[i] != 0 {
                            sums[0] += table.get(next_r[i]) - table.get(r[i]);
                            sums[1] += table.get(next_l[i]) - table.get(l[i]);
                        }
                    }
                    let totals = (next_r.iter().sum(), next_l.iter().sum());
                    let actual = scratch
                        .move_and_score(options, &scorer, &mut r, &mut l, &delta, totals, None);
                    if fast {
                        assert_eq!(
                            actual.0.to_bits(),
                            ((table.get(totals.0) - sums[0]).max(0.0) as f32).to_bits()
                        );
                        assert_eq!(
                            actual.1.to_bits(),
                            ((table.get(totals.1) - sums[1]).max(0.0) as f32).to_bits()
                        );
                    } else {
                        for (histogram, bits) in
                            [(r.as_slice(), actual.0), (l.as_slice(), actual.1)]
                        {
                            let sum = histogram.iter().map(|&n| f64::from(n)).sum::<f64>();
                            let expected: f64 = histogram
                                .iter()
                                .filter(|&&n| n != 0)
                                .map(|&n| f64::from(n) * (sum / f64::from(n)).log2().min(12.0))
                                .sum();
                            assert!((f64::from(bits) - expected).abs() <= 0.25);
                        }
                    }
                    assert_eq!(r.as_slice(), next_r);
                    assert_eq!(l.as_slice(), next_l);
                }
            }
        }
    }

    #[test]
    fn scoring_description_matches_source_f32_order_and_zero_symbol_rule() {
        let scorer = DefaultScorer::new();
        for total in [1, 256, 65536, i32::MAX as u32] {
            for (mode, weight) in [
                (DescriptionCost::Disabled, 0.0),
                (DescriptionCost::Quarter, 0.25),
                (DescriptionCost::Half, 0.5),
                (DescriptionCost::Full, 1.0),
            ] {
                for counts in [vec![total], vec![0, total, 0]] {
                    assert_eq!(mode.cost(&scorer, &counts, total), 0.0);
                }
                if total > 1 {
                    let expected = weight * 0.5 * scorer.log2(total as f32 + 1.0);
                    assert_eq!(
                        mode.cost(&scorer, &[1, total - 1, 0], total).to_bits(),
                        expected.to_bits()
                    );
                }
            }
        }
    }
}
