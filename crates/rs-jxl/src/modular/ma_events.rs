// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! Opt-in sparse split sweeps. The clipped objective is enc_ma_entropy.h's
//! MaClippedEntropy (b866b3de); record bucketing avoids its dense delta scans.

use super::{
    Cost, ParentHistogram, PredictorPlan, PropertyValue, SampleStorage, Samples, SearchOptions,
    SparseHistogramPlan, TrainOptions,
};
use crate::tools::ma_simd::{CountEntropyTable, DefaultScorer};
use crate::{Error, Result};

const MAX_TOTAL: u32 = 65536;
const ANS_SIZE: u32 = 4096;
const RARE_SLOTS: usize = (MAX_TOTAL / ANS_SIZE) as usize;

#[allow(clippy::too_many_arguments)]
pub(super) fn eligible(
    options: &TrainOptions,
    total: u32,
    records: usize,
    used: &[usize],
    parents: &[ParentHistogram],
    plan: &PredictorPlan,
    sparse: &SparseHistogramPlan,
    cached: bool,
) -> bool {
    if !options.histogram_events
        || cached
        || !(2..=MAX_TOTAL).contains(&total)
        || records < 2
        || options.scoring.fast_entropy
        || options.search != SearchOptions::default()
    {
        return false;
    }
    let thresholds = used.iter().filter(|&&n| n != 0).count().saturating_sub(1);
    let lanes = DefaultScorer::new().lanes();
    let symbols = plan.active().iter().fold(0usize, |sum, &p| {
        let width = parents[p].counts.len();
        let work = sparse.blocks(p).map_or(width, |blocks| {
            blocks.iter().map(|&start| lanes.min(width - start)).sum()
        });
        sum.saturating_add(work)
    });
    // At most one event per record/predictor. Require an eightfold reduction
    // against actual scoring cells, including existing compaction/zero blocks,
    // to pay for bucketing, scalar table loads and indirect record traversal.
    let events = records.saturating_mul(plan.active().len());
    events != 0 && events.saturating_mul(8) <= thresholds.saturating_mul(symbols)
}

fn resize<T: Default + Clone>(values: &mut Vec<T>, size: usize) -> Result<()> {
    values
        .try_reserve(size.saturating_sub(values.len()))
        .map_err(|_| Error::Limit("MA event scratch allocation"))?;
    values.resize(size, T::default());
    Ok(())
}

#[derive(Default)]
struct EventEntropy {
    counts: Vec<u32>,
    terms: Vec<f64>,
    sum: f64,
    active: usize,
    frequencies: [u16; RARE_SLOTS],
    cursor: usize,
    tail_mass: u32,
    tail_sum: f64,
}

impl EventEntropy {
    fn reset(&mut self, parent: &[u32], empty: bool, table: &CountEntropyTable) -> Result<()> {
        resize(&mut self.counts, parent.len())?;
        resize(&mut self.terms, parent.len())?;
        self.sum = 0.0;
        self.active = 0;
        self.frequencies.fill(0);
        self.cursor = 0;
        self.tail_mass = 0;
        self.tail_sum = 0.0;
        for (i, &count) in parent.iter().enumerate() {
            let count = if empty { 0 } else { count };
            let term = table.get(count);
            self.counts[i] = count;
            self.terms[i] = term;
            self.sum += term;
            if count != 0 {
                self.active += 1;
                if (count as usize) < RARE_SLOTS {
                    self.frequencies[count as usize] += 1;
                }
            }
        }
        Ok(())
    }

    fn replace(&mut self, symbol: usize, after: u32, table: &CountEntropyTable) {
        let before = self.counts[symbol];
        let old = self.terms[symbol];
        let new = table.get(after);
        self.sum += new - old;
        if before != 0 {
            self.active -= 1;
            if (before as usize) < RARE_SLOTS {
                self.frequencies[before as usize] -= 1;
                if before as usize <= self.cursor {
                    self.tail_mass -= before;
                    self.tail_sum -= old;
                }
            }
        }
        if after != 0 {
            self.active += 1;
            if (after as usize) < RARE_SLOTS {
                self.frequencies[after as usize] += 1;
                if after as usize <= self.cursor {
                    self.tail_mass += after;
                    self.tail_sum += new;
                }
            }
        }
        self.counts[symbol] = after;
        self.terms[symbol] = new;
    }

    fn bits(&mut self, total: u32, table: &CountEntropyTable) -> f32 {
        debug_assert!(total <= MAX_TOTAL);
        if total == 0 {
            return 0.0;
        }
        // The strict probability floor is c*4096 < total. Moving the cursor
        // visits at most 15 buckets over a monotone sweep, not every symbol.
        let end = ((total - 1) / ANS_SIZE) as usize;
        while self.cursor < end {
            self.cursor += 1;
            let count = u32::from(self.frequencies[self.cursor]);
            self.tail_mass += count * self.cursor as u32;
            self.tail_sum += f64::from(count) * table.get(self.cursor as u32);
        }
        while self.cursor > end {
            let count = u32::from(self.frequencies[self.cursor]);
            self.tail_mass -= count * self.cursor as u32;
            self.tail_sum -= f64::from(count) * table.get(self.cursor as u32);
            self.cursor -= 1;
        }
        if self.active <= 1 {
            return 0.0;
        }
        let count_entropy = table.get(total);
        // H = N*log2(N) - sum(c*log2(c)) + Q + M*(12-log2(N)), where
        // Q and M include only counts below the strict probability floor.
        let correction = if self.tail_mass == 0 {
            0.0
        } else {
            self.tail_sum + f64::from(self.tail_mass) * (12.0 - count_entropy / f64::from(total))
        };
        (count_entropy - self.sum + correction).max(0.0) as f32
    }
}

/// Private property-worker storage; no matrix and no allocation when disabled.
#[derive(Default)]
pub(super) struct EventScratch {
    order: Vec<usize>,
    offsets: Vec<usize>,
    cursors: Vec<usize>,
    delta: Vec<u32>,
    touched: Vec<u8>,
    above: EventEntropy,
    below: EventEntropy,
    #[cfg(test)]
    pub(super) thresholds: usize,
    #[cfg(test)]
    pub(super) updates: usize,
}

impl EventScratch {
    #[cfg(test)]
    pub(super) fn reset_trace(&mut self) {
        self.thresholds = 0;
        self.updates = 0;
    }

    fn bucket<T: PropertyValue>(
        &mut self,
        indices: &[usize],
        values: &[T],
        used: &[usize],
    ) -> Result<()> {
        resize(&mut self.order, indices.len())?;
        resize(
            &mut self.offsets,
            used.len()
                .checked_add(1)
                .ok_or(Error::Limit("MA event bins"))?,
        )?;
        resize(&mut self.cursors, used.len())?;
        self.offsets[0] = 0;
        for (bin, &count) in used.iter().enumerate() {
            self.offsets[bin + 1] = self.offsets[bin] + count;
            self.cursors[bin] = self.offsets[bin];
        }
        debug_assert_eq!(self.offsets[used.len()], indices.len());
        for &i in indices {
            let cursor = &mut self.cursors[values[i].index()];
            self.order[*cursor] = i;
            *cursor += 1;
        }
        resize(&mut self.delta, 256)?;
        self.delta.fill(0);
        self.touched.clear();
        self.touched
            .try_reserve(256)
            .map_err(|_| Error::Limit("MA event tokens allocation"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        values: &[T],
        used: &[usize],
        weights: &[u32],
        parents: &[ParentHistogram],
        plan: &PredictorPlan,
        options: &TrainOptions,
        total: u32,
        mapping: Option<&[u8; 256]>,
        below_costs: &mut [Cost],
        above_costs: &mut [Cost],
    ) -> Result<()> {
        debug_assert!((2..=MAX_TOTAL).contains(&total));
        self.bucket(indices, values, used)?;
        let table = CountEntropyTable::shared();
        let scorer = DefaultScorer::new();
        let last = used.iter().rposition(|&n| n != 0).unwrap_or(0);
        for &predictor in plan.active() {
            let parent = &parents[predictor];
            self.above.reset(&parent.counts, false, table)?;
            self.below.reset(&parent.counts, true, table)?;
            let tokens = &samples.residuals[predictor];
            let penalty = plan.penalty[predictor];
            let mut weight_below = 0u32;
            let mut extra_below = 0u64;
            for bin in 0..last {
                if used[bin] == 0 {
                    continue;
                }
                for &i in &self.order[self.offsets[bin]..self.offsets[bin + 1]] {
                    let residual = tokens[i];
                    let token = mapping.map_or(residual.token, |m| m[usize::from(residual.token)]);
                    let delta = &mut self.delta[usize::from(token)];
                    if *delta == 0 {
                        self.touched.push(token);
                    }
                    let weight = u32::from(samples.counts[i]);
                    *delta += weight;
                    extra_below += u64::from(weight) * u64::from(residual.nbits);
                }
                for &token in &self.touched {
                    let s = usize::from(token);
                    let delta = std::mem::take(&mut self.delta[s]);
                    self.above.replace(s, self.above.counts[s] - delta, table);
                    self.below.replace(s, self.below.counts[s] + delta, table);
                }
                #[cfg(test)]
                {
                    self.thresholds += 1;
                    self.updates += self.touched.len();
                }
                self.touched.clear();
                weight_below += weights[bin];
                let weight_above = total - weight_below;
                let mut below = self.below.bits(weight_below, table) + extra_below as f32;
                let mut above = (self.above.bits(weight_above, table) + parent.extra_bits as f32)
                    - extra_below as f32;
                below += options.scoring.description.cost_for_active(
                    &scorer,
                    self.below.active,
                    weight_below,
                );
                above += options.scoring.description.cost_for_active(
                    &scorer,
                    self.above.active,
                    weight_above,
                );
                if below + penalty < below_costs[bin].bits + below_costs[bin].penalty {
                    below_costs[bin] = Cost {
                        bits: below,
                        penalty,
                        predictor,
                    };
                }
                if above + penalty < above_costs[bin].bits + above_costs[bin].penalty {
                    above_costs[bin] = Cost {
                        bits: above,
                        penalty,
                        predictor,
                    };
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "ma_events_tests.rs"]
mod tests;
