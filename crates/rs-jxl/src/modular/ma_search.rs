// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! enc_ma_racing.h and enc_ma.cc optional proposal/refinement, frozen b866b3d.

#[cfg(test)]
use super::TreeSamples;

use super::{
    ClippedEntropyMode, PropertyColumn, PropertyValue, SampleStorage, Samples, ScoringOptions,
    TrainOptions, WP_PROPERTY,
};
use crate::{Error, Result, execution};
use std::{cmp::Ordering, sync::Mutex};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QuantileSplits {
    #[default]
    Disabled,
    /// JXL_MA_QUANTILE_SPLITS=1.
    SixtyFour,
    /// JXL_MA_QUANTILE_SPLITS=2.
    ThirtyTwo,
}

/// Explicit source proposal heuristics. These can change the learned tree,
/// never encoded pixel reconstruction; the complete default sweep is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchOptions {
    pub statistical_race: bool,
    pub quantile_splits: QuantileSplits,
    pub refine_quantiles: bool,
}

impl SearchOptions {
    pub fn apply(&self, options: &mut TrainOptions) {
        options.search = *self;
    }

    fn quantiles(self, scoring: ScoringOptions) -> usize {
        if scoring.fast_entropy || scoring.clipped != ClippedEntropyMode::Disabled {
            return 0;
        }
        if self.refine_quantiles {
            return 32;
        }
        match self.quantile_splits {
            QuantileSplits::Disabled => 0,
            QuantileSplits::SixtyFour => 64,
            QuantileSplits::ThirtyTwo => 32,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_mask(
    options: SearchOptions,
    scoring: ScoringOptions,
    records: usize,
    property_id: usize,
    used: &[usize],
    weights: &[u32],
    total: u32,
    first: usize,
    last: usize,
    histogram_cells: usize,
    predictors: usize,
    mask: &mut Vec<u8>,
) -> Result<bool> {
    mask.clear();
    let quantiles = options.quantiles(scoring);
    if quantiles == 0 || records < 512 || property_id < 2 || last - first <= quantiles {
        return Ok(false);
    }
    let refine = options.refine_quantiles
        && histogram_cells
            .checked_mul(predictors)
            .and_then(|cells| cells.checked_mul(size_of::<u32>()))
            .is_some_and(|bytes| bytes <= 8 << 20);
    // Source refuses an unrefined approximation when refinement storage cannot fit.
    if options.refine_quantiles && !refine {
        return Ok(false);
    }
    mask.try_reserve(last - first)
        .map_err(|_| Error::Limit("MA quantile mask allocation"))?;
    mask.resize(last - first, 0);
    let quantum = (u64::from(total) / quantiles as u64).max(1);
    let mut pending = 0u64;
    let mut last_nonempty = first;
    for bin in first..last {
        if used[bin] == 0 {
            continue;
        }
        last_nonempty = bin;
        pending += u64::from(weights[bin]);
        if bin == first || pending >= quantum {
            mask[bin - first] = 1;
            pending = 0;
        }
    }
    mask[last_nonempty - first] = 1;
    Ok(refine)
}

#[cfg(test)]
pub(super) fn refine_mask(coarse: &mut [u8], costs: &[f32]) {
    if coarse.len() != costs.len() {
        return;
    }
    refine_mask_by(coarse, |i| costs[i]);
}

pub(super) fn refine_mask_by(coarse: &mut [u8], cost: impl Fn(usize) -> f32) {
    // The stable sort only consumes two uncovered minima. Strict minima
    // scans preserve its tie order without points/order/expanded allocations.
    // Both neighborhoods must be found before changing the coarse-point set.
    let Some(first) = refinement_valley(coarse, &cost, 0..0) else {
        return;
    };
    let second = refinement_valley(coarse, &cost, first.clone());
    coarse[first].fill(1);
    if let Some(second) = second {
        coarse[second].fill(1);
    }
}

fn refinement_valley(
    coarse: &[u8],
    cost: &impl Fn(usize) -> f32,
    excluded: std::ops::Range<usize>,
) -> Option<std::ops::Range<usize>> {
    let mut best: Option<std::ops::Range<usize>> = None;
    let mut best_cost = f32::INFINITY;
    let mut previous = 0;
    let mut needs_end = false;
    for (i, &point) in coarse.iter().enumerate() {
        if point == 0 {
            continue;
        }
        let bits = cost(i);
        if !bits.is_finite() {
            continue;
        }
        if needs_end {
            if let Some(range) = &mut best {
                // This is the next original finite coarse point, not the next
                // point outside the first valley's excluded neighborhood.
                range.end = i + 1;
            }
            needs_end = false;
        }
        if !excluded.contains(&i) && bits < best_cost {
            best = Some(previous..coarse.len());
            best_cost = bits;
            needs_end = true;
        }
        previous = i;
    }
    best
}

const PANELS: usize = 4;

#[derive(Default)]
struct RaceState {
    hist: Vec<u32>,
    weight: Vec<u32>,
    extra: Vec<u64>,
    mass: [f64; PANELS],
    mass2: [f64; PANELS],
    costs: [f64; PANELS],
    draws: usize,
}

fn zeroed<T: Default + Clone>(values: &mut Vec<T>, size: usize) -> Result<()> {
    values
        .try_reserve(size.saturating_sub(values.len()))
        .map_err(|_| Error::Limit("MA property race allocation"))?;
    values.resize(size, T::default());
    Ok(())
}

fn sample_index(begin: usize, end: usize, panel: usize, k: usize) -> usize {
    let mut x = (k as u64)
        .wrapping_add((panel as u64).wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add((begin as u64).wrapping_mul(0xbf58476d1ce4e5b9))
        .wrapping_add(end as u64);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    begin + (x % ((end - begin) as u64)) as usize
}

fn count_cost(count: f64) -> f64 {
    if count > 1.0 {
        count * count.log2()
    } else {
        0.0
    }
}

impl RaceState {
    #[allow(clippy::too_many_arguments)]
    fn extend<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        column: &[T],
        values: usize,
        symbols: usize,
        begin: usize,
        target: usize,
    ) -> Result<()> {
        let predictors = samples.residuals.len();
        if self.hist.is_empty() {
            let rows = PANELS
                .checked_mul(predictors)
                .and_then(|n| n.checked_mul(values))
                .ok_or(Error::Limit("MA property race size"))?;
            zeroed(
                &mut self.hist,
                rows.checked_mul(symbols)
                    .ok_or(Error::Limit("MA property race size"))?,
            )?;
            zeroed(&mut self.extra, rows)?;
            zeroed(&mut self.weight, PANELS * values)?;
        }
        let end = begin + indices.len();
        for panel in 0..PANELS {
            for k in self.draws..target {
                let index = indices[sample_index(begin, end, panel, k) - begin];
                let value = column[index].index();
                let weight = u32::from(samples.counts[index]);
                self.weight[panel * values + value] += weight;
                self.mass[panel] += f64::from(weight);
                self.mass2[panel] += f64::from(weight) * f64::from(weight);
                for (predictor, tokens) in samples.residuals.iter().enumerate() {
                    let token = tokens[index];
                    let row = (panel * predictors + predictor) * values + value;
                    self.hist[row * symbols + usize::from(token.token)] += weight;
                    self.extra[row] += u64::from(weight) * u64::from(token.nbits);
                }
            }
            self.costs[panel] = self.score(panel, values, predictors, symbols);
        }
        self.draws = target;
        Ok(())
    }

    fn score(&self, panel: usize, values: usize, predictors: usize, symbols: usize) -> f64 {
        if values < 2 || self.mass[panel] == 0.0 {
            return f64::INFINITY;
        }
        let mut left = vec![f64::INFINITY; values];
        let mut right = left.clone();
        let mut parent_best = f64::INFINITY;
        for predictor in 0..predictors {
            let base = (panel * predictors + predictor) * values;
            let mut above = vec![0u32; symbols];
            let mut below = vec![0u32; symbols];
            let (mut extra_total, mut entropy_above, mut entropy_below) = (0.0, 0.0, 0.0);
            for value in 0..values {
                extra_total += self.extra[base + value] as f64;
                for (s, count) in above.iter_mut().enumerate() {
                    *count += self.hist[(base + value) * symbols + s];
                }
            }
            for &count in &above {
                entropy_above += count_cost(f64::from(count));
            }
            parent_best =
                parent_best.min(count_cost(self.mass[panel]) - entropy_above + extra_total);
            let (mut mass_left, mut extra_left) = (0.0, 0.0);
            for value in 0..values - 1 {
                mass_left += f64::from(self.weight[panel * values + value]);
                extra_left += self.extra[base + value] as f64;
                for s in 0..symbols {
                    let delta = self.hist[(base + value) * symbols + s];
                    if delta == 0 {
                        continue;
                    }
                    entropy_above +=
                        count_cost(f64::from(above[s] - delta)) - count_cost(f64::from(above[s]));
                    entropy_below +=
                        count_cost(f64::from(below[s] + delta)) - count_cost(f64::from(below[s]));
                    above[s] -= delta;
                    below[s] += delta;
                }
                if mass_left == 0.0 || mass_left == self.mass[panel] {
                    continue;
                }
                left[value] = left[value].min(count_cost(mass_left) - entropy_below + extra_left);
                right[value] = right[value].min(
                    count_cost(self.mass[panel] - mass_left) - entropy_above + extra_total
                        - extra_left,
                );
            }
        }
        let mut best = parent_best;
        for value in 0..values - 1 {
            best = best.min(left[value] + right[value]);
        }
        best.max(0.0) / self.mass[panel]
    }

    fn mean(&self) -> f64 {
        self.costs.iter().sum::<f64>() / PANELS as f64
    }
    fn reliable(&self) -> bool {
        (0..PANELS).all(|p| {
            self.mass2[p] != 0.0
                && self.mass[p] * self.mass[p] / self.mass2[p] >= 32.0
                && self.costs[p].is_finite()
        })
    }
}

// enc_lossless_racing.h::PairedLosslessEvidence(., ., .02).ClearlyWorse(.01).
// Scheduler's separate usize projection is not this f64 property evidence API.
fn clearly_worse(candidate: &[f64; PANELS], reference: &[f64; PANELS]) -> bool {
    let (mut mean, mut m2, mut minimum) = (0.0, 0.0, f64::INFINITY);
    for (i, (&candidate, &reference)) in candidate.iter().zip(reference).enumerate() {
        let (candidate, reference) = (candidate + 1.0, reference + 1.0);
        if !candidate.is_finite() || !reference.is_finite() || candidate < 0.0 || reference <= 0.0 {
            return false;
        }
        let delta = candidate / reference - 1.0;
        if !delta.is_finite() {
            return false;
        }
        let old = mean;
        mean += (delta - old) / (i + 1) as f64;
        m2 += (delta - old) * (delta - mean);
        minimum = minimum.min(delta);
    }
    let radius = 0.02 + 2.5 * (m2.max(0.0) / (PANELS * (PANELS - 1)) as f64).sqrt();
    minimum > 0.0 && mean - radius > 0.01
}

fn survivors(states: &[RaceState], ids: &[usize], active: &[u8]) -> Vec<u8> {
    let mut keep = active.to_vec();
    let mut order: Vec<_> = (0..states.len())
        .filter(|&p| ids[p] >= 2 && active[p] != 0)
        .collect();
    order.sort_by(|&a, &b| {
        states[a]
            .mean()
            .partial_cmp(&states[b].mean())
            .unwrap_or(Ordering::Equal)
    });
    let Some(&leader) = order.first() else {
        return keep;
    };
    if !states[leader].reliable() {
        return keep;
    }
    for &property in order.iter().skip(4) {
        if ids[property] != WP_PROPERTY
            && states[property].reliable()
            && clearly_worse(&states[property].costs, &states[leader].costs)
        {
            keep[property] = 0;
        }
    }
    keep
}

fn fits(samples: &Samples<impl SampleStorage>, ids: &[usize], symbols: usize) -> bool {
    let mut values = 0usize;
    for (&id, thresholds) in ids.iter().zip(samples.thresholds.iter()) {
        if id < 2 {
            continue;
        }
        let bins = thresholds.len() + 1;
        if bins > 256 {
            return false;
        }
        let Some(sum) = values.checked_add(bins) else {
            return false;
        };
        values = sum;
    }
    values
        .checked_mul(symbols)
        .and_then(|n| n.checked_mul(samples.residuals.len()))
        .and_then(|n| n.checked_mul(PANELS * size_of::<u32>()))
        .is_some_and(|bytes| bytes <= 32 << 20)
}

#[derive(Default)]
pub(super) struct RaceResult {
    pub(super) keep: Vec<u8>,
    pub(super) draws: usize,
    pub(super) rounds: usize,
}

pub(super) fn race(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    symbols: usize,
    inherited: bool,
) -> Result<RaceResult> {
    if !options.search.statistical_race
        || inherited
        || indices.len() < 262144
        || options.properties.iter().filter(|&&id| id >= 2).count() <= 4
        || !fits(samples, &options.properties, symbols)
    {
        return Ok(RaceResult::default());
    }
    // Production physical partitioning retains contiguous identity indices;
    // the source's absolute range participates in its deterministic sample seed.
    let begin = samples.base() + indices[0];
    debug_assert!(
        indices
            .iter()
            .enumerate()
            .all(|(i, &index)| index == indices[0] + i)
    );
    let mut states: Vec<_> = (0..options.properties.len())
        .map(|_| RaceState::default())
        .collect();
    let mut result = RaceResult {
        keep: vec![1; states.len()],
        ..RaceResult::default()
    };
    let mut previous = 0;
    for draws in [64, 256] {
        let properties: Vec<_> = (0..states.len())
            .filter(|&p| options.properties[p] >= 2 && result.keep[p] != 0)
            .collect();
        if properties.len() <= 4 {
            break;
        }
        let jobs: Vec<_> = states.iter_mut().map(Mutex::new).collect();
        let extend = |&property: &usize| {
            let mut state = jobs[property].lock().expect("MA property race poisoned");
            let values = samples.thresholds[property].len() + 1;
            match &samples.properties[property] {
                PropertyColumn::Packed(column) => {
                    state.extend(samples, indices, column, values, symbols, begin, draws)
                }
                PropertyColumn::Wide(column) => {
                    state.extend(samples, indices, column, values, symbols, begin, draws)
                }
            }
        };
        let results: Vec<Result<()>> = if options.parallel {
            execution::map(&properties, extend)
        } else {
            properties.iter().map(extend).collect()
        };
        for result in results {
            result?;
        }
        drop(jobs);
        result.draws += properties.len() * PANELS * (draws - previous);
        result.rounds += 1;
        previous = draws;
        let keep = survivors(&states, &options.properties, &result.keep);
        if keep == result.keep {
            break;
        }
        result.keep = keep;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::{Predictor, ResidualToken};
    use super::*;

    #[test]
    fn refinement_scans_match_stable_sort_oracle() {
        fn oracle(coarse: &mut [u8], costs: &[f32]) {
            let points: Vec<_> = (0..coarse.len())
                .filter(|&i| coarse[i] != 0 && costs[i].is_finite())
                .collect();
            let mut order: Vec<_> = (0..points.len()).collect();
            order.sort_by(|&a, &b| costs[points[a]].partial_cmp(&costs[points[b]]).unwrap());
            let mut expanded = vec![false; coarse.len()];
            let mut valleys = 0;
            for index in order {
                if expanded[points[index]] {
                    continue;
                }
                let begin = if index == 0 { 0 } else { points[index - 1] };
                let end = points.get(index + 1).map_or(coarse.len(), |&p| p + 1);
                coarse[begin..end].fill(1);
                expanded[begin..end].fill(true);
                valleys += 1;
                if valleys == 2 {
                    break;
                }
            }
        }
        let mut seed = 0x721b4cd3u32;
        for len in [0, 1, 2, 3, 7, 32, 65, 256, 1025] {
            for trial in 0..64 {
                let mut coarse = Vec::new();
                let mut costs = Vec::new();
                for _ in 0..len {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    coarse.push(if trial % 8 == 0 {
                        1
                    } else {
                        ((seed >> 16) % 3) as u8
                    });
                    costs.push(match (seed >> 24) % 9 {
                        0 => f32::NAN,
                        1 => f32::INFINITY,
                        2 => f32::NEG_INFINITY,
                        3 => -0.0,
                        4 => 0.0,
                        n => (n as i32 - 6) as f32,
                    });
                }
                let mut expected = coarse.clone();
                oracle(&mut expected, &costs);
                refine_mask_by(&mut coarse, |i| costs[i]);
                assert_eq!(coarse, expected, "length={len}, trial={trial}");
            }
        }
    }

    #[test]
    fn search_quantile_mask_source_gates_edges_weights_and_refinement_budget() {
        let used = vec![1; 65];
        let mut weights = vec![1; 65];
        weights[20] = 100;
        let mut mask = Vec::new();
        let options = SearchOptions {
            quantile_splits: QuantileSplits::ThirtyTwo,
            ..SearchOptions::default()
        };
        for (records, id, last, expected) in [
            (511, 2, 64, false),
            (512, 1, 64, false),
            (512, 2, 32, false),
            (512, 2, 33, true),
            (512, 2, 64, true),
        ] {
            assert!(
                !prepare_mask(
                    options,
                    ScoringOptions::default(),
                    records,
                    id,
                    &used,
                    &weights,
                    164,
                    0,
                    last,
                    100,
                    2,
                    &mut mask
                )
                .unwrap()
            );
            assert_eq!(!mask.is_empty(), expected);
            if expected {
                assert_eq!(mask[0], 1);
                assert_eq!(mask[last - 1], 1);
                assert_eq!(mask[20], 1);
            }
        }
        let refined = SearchOptions {
            refine_quantiles: true,
            ..options
        };
        for (cells, active) in [(1 << 20, true), ((1 << 20) + 1, false), (usize::MAX, false)] {
            assert_eq!(
                prepare_mask(
                    refined,
                    ScoringOptions::default(),
                    512,
                    2,
                    &used,
                    &weights,
                    164,
                    0,
                    64,
                    cells,
                    2,
                    &mut mask
                )
                .unwrap(),
                active
            );
            assert_eq!(!mask.is_empty(), active);
        }
        for scoring in [
            ScoringOptions {
                fast_entropy: true,
                ..ScoringOptions::default()
            },
            ScoringOptions {
                clipped: ClippedEntropyMode::AtLeast16384,
                ..ScoringOptions::default()
            },
        ] {
            assert!(
                !prepare_mask(
                    refined, scoring, 512, 2, &used, &weights, 164, 0, 64, 1, 1, &mut mask
                )
                .unwrap()
            );
            assert!(mask.is_empty());
        }
        let mut train = TrainOptions::default();
        refined.apply(&mut train);
        assert_eq!(train.search, refined);
        assert_eq!(train.scoring, ScoringOptions::default());
    }

    #[test]
    fn search_refinement_keeps_coarse_points_and_two_separated_finite_valleys() {
        let mut mask = vec![0; 32];
        let mut costs = vec![f32::INFINITY; 32];
        for (point, cost) in [
            (0, 9.0),
            (5, 1.0),
            (10, 8.0),
            (15, 7.0),
            (20, 6.0),
            (25, 2.0),
            (31, 5.0),
        ] {
            mask[point] = 1;
            costs[point] = cost;
        }
        refine_mask(&mut mask, &costs);
        let expected: Vec<u8> = (0..32)
            .map(|i| u8::from(i <= 10 || i >= 20 || i == 15))
            .collect();
        assert_eq!(mask, expected);
        let previous = mask.clone();
        refine_mask(&mut mask, &[]);
        assert_eq!(mask, previous);
        let mut invalid = vec![1, 0, 1];
        refine_mask(&mut invalid, &[f32::NAN, 0.0, f32::INFINITY]);
        assert_eq!(invalid, [1, 0, 1]);
        let mut tied = vec![1, 0, 1, 0, 1, 0, 1];
        refine_mask(&mut tied, &[1.0, 9.0, 1.0, 9.0, 1.0, 9.0, 1.0]);
        assert_eq!(tied, [1; 7]);
    }

    #[test]
    fn search_race_source_hash_evidence_reliability_and_wp_protection() {
        let golden = [
            [234048, 149438, 169933, 23738, 19693],
            [90389, 47337, 185114, 244609, 46463],
            [6618, 31534, 156427, 32447, 43924],
            [164828, 5110, 40929, 209535, 146329],
        ];
        for (panel, expected) in golden.iter().enumerate() {
            for (&k, &index) in [0, 1, 63, 64, 255].iter().zip(expected) {
                assert_eq!(sample_index(137, 262281, panel, k), index);
            }
        }
        assert!(clearly_worse(&[1.0; 4], &[0.0; 4]));
        assert!(!clearly_worse(&[0.02; 4], &[0.0; 4]));
        assert!(!clearly_worse(&[1.0, 1.0, 0.0, 1.0], &[0.0; 4]));
        assert!(!clearly_worse(&[f64::INFINITY; 4], &[0.0; 4]));
        let mut states: Vec<_> = (0..8)
            .map(|i| RaceState {
                mass: [64.0; 4],
                mass2: [64.0; 4],
                costs: [if i < 4 { 0.0 } else { 1.0 }; 4],
                ..RaceState::default()
            })
            .collect();
        let ids = [2, 3, 4, 5, 6, 15, 7, 0];
        assert_eq!(survivors(&states, &ids, &[1; 8]), [1, 1, 1, 1, 0, 1, 0, 1]);
        states[4].mass = [31.0; 4];
        states[4].mass2 = [31.0; 4];
        assert_eq!(survivors(&states, &ids, &[1; 8])[4], 1);
        states[0].mass2 = [65535.0 * 65535.0; 4];
        assert_eq!(survivors(&states, &ids, &[1; 8]), [1; 8]);
    }

    #[test]
    fn race_borrowed_columns_preserve_absolute_hashes_through_nested_ranges() {
        let records = 262144;
        let (mut samples, options) = race_fixture(records + 137);
        let absolute: Vec<_> = (137..records + 137).collect();
        let expected = race(&samples, &absolute, &options, 8, false).unwrap();
        let outer_range = 127..records + 137;
        let mut outer = samples
            .ranges_mut(std::slice::from_ref(&outer_range))
            .unwrap();
        let inner_range = 10..records + 10;
        let inner = outer[0]
            .ranges_mut(std::slice::from_ref(&inner_range))
            .unwrap();
        assert_eq!(inner[0].base(), 137);
        let local: Vec<_> = (0..records).collect();
        let actual = race(&inner[0], &local, &options, 8, false).unwrap();
        assert!(!actual.keep.is_empty());
        assert_eq!(actual.keep, expected.keep);
        assert_eq!(actual.draws, expected.draws);
    }

    fn race_fixture(records: usize) -> (TreeSamples, TrainOptions) {
        let ids = vec![6, 0, 2, 3, 4, 5, 15, 7];
        let signal: Vec<_> = (0..records)
            .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 63) as u8)
            .collect();
        let noise: Vec<_> = (0..records)
            .map(|i| ((i as u64).wrapping_mul(0xbf58476d1ce4e5b9) >> 59) as u8 & 1)
            .collect();
        let properties = ids
            .iter()
            .enumerate()
            .map(|(p, &id)| {
                if id < 2 {
                    PropertyColumn::Wide(vec![0; records])
                } else {
                    PropertyColumn::Packed(if [0, 2, 3, 4].contains(&p) {
                        signal.clone()
                    } else {
                        noise.clone()
                    })
                }
            })
            .collect();
        let tokens: Vec<_> = signal
            .iter()
            .map(|&token| ResidualToken { token, nbits: 0 })
            .collect();
        let samples = TreeSamples {
            properties,
            residuals: vec![tokens.clone(), tokens],
            counts: vec![1; records],
            thresholds: vec![vec![0]; ids.len()],
        };
        let options = TrainOptions {
            properties: ids,
            predictors: vec![Predictor::Weighted, Predictor::Gradient],
            parallel: false,
            split_threshold: 0.0,
            search: SearchOptions {
                statistical_race: true,
                ..SearchOptions::default()
            },
            ..TrainOptions::default()
        };
        (samples, options)
    }

    #[test]
    fn search_race_rounds_gates_parallel_and_inherited_cache_rebuild() {
        use super::super::{
            HistogramBudget, ScratchPool, TrainingStats, find_best_split_with_properties,
        };
        use std::sync::atomic::Ordering;
        let (samples, mut options) = race_fixture(262145);
        let indices: Vec<_> = (0..samples.counts.len()).collect();
        assert!(
            race(&samples, &indices[..262143], &options, 8, false)
                .unwrap()
                .keep
                .is_empty()
        );
        assert!(
            race(&samples, &indices, &options, 8, true)
                .unwrap()
                .keep
                .is_empty()
        );
        assert!(!fits(&samples, &options.properties, usize::MAX));
        let reference = race(&samples, &indices, &options, 8, false).unwrap();
        assert_eq!(reference.keep, [1, 1, 1, 1, 1, 0, 1, 0]);
        assert_eq!(reference.rounds, 2);
        assert_eq!(reference.draws, 7 * 4 * 64 + 5 * 4 * 192);
        options.parallel = true;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let parallel = pool
            .install(|| race(&samples, &indices, &options, 8, false))
            .unwrap();
        assert_eq!(parallel.keep, reference.keep);
        assert_eq!(parallel.draws, reference.draws);
        options.parallel = false;
        let budget = HistogramBudget::new(options.histogram_cache_bytes);
        let mut cache = None;
        let mut masks = Vec::new();
        let stats = TrainingStats::default();
        find_best_split_with_properties(
            &samples,
            &indices,
            0,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &stats,
            &ScratchPool::new(1),
            &[],
            &mut masks,
        )
        .unwrap();
        assert_eq!(stats.race_pruned.load(Ordering::Relaxed), 2);
        assert!(masks.is_empty());
        for p in [5, 7] {
            assert!(!cache.as_ref().unwrap().properties[p].valid);
        }
        let child_stats = TrainingStats::default();
        let actual = find_best_split_with_properties(
            &samples,
            &indices[1..],
            0,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &child_stats,
            &ScratchPool::new(1),
            &indices[..1],
            &mut masks,
        )
        .unwrap();
        assert_eq!(child_stats.race_nodes.load(Ordering::Relaxed), 0);
        assert!(cache.as_ref().unwrap().properties.iter().all(|p| p.valid));
        options.search.statistical_race = false;
        let expected = find_best_split_with_properties(
            &samples,
            &indices[1..],
            0,
            &options,
            &mut None,
            &mut None,
            &HistogramBudget::new(0),
            &TrainingStats::default(),
            &ScratchPool::new(1),
            &[],
            &mut Vec::new(),
        )
        .unwrap();
        let key = |s: Option<super::super::Split>| {
            s.map(|s| {
                (
                    s.property,
                    s.bin,
                    s.below_count,
                    s.below_predictor,
                    s.above_predictor,
                    s.bits.to_bits(),
                )
            })
        };
        assert_eq!(key(actual), key(expected));
    }

    #[test]
    fn search_refinement_reuses_histograms_and_preserves_all_scored_costs() {
        use super::super::{
            HistogramBudget, HistogramCache, PropertyScratch, evaluate_property_cached,
            parent_histograms,
        };
        for packed in [false, true] {
            let records = 1024;
            let values: Vec<_> = (0..records)
                .map(|i| {
                    if packed {
                        ((i % 65) * 3) as u8
                    } else {
                        (i % 256) as u8
                    }
                })
                .collect();
            let tokens: Vec<_> = (0..records)
                .map(|i| ResidualToken {
                    token: ((i * 19 + i / 7) % 32) as u8,
                    nbits: (i % 3) as u8,
                })
                .collect();
            let samples = TreeSamples {
                properties: vec![PropertyColumn::Packed(values)],
                residuals: vec![tokens.clone(), tokens],
                counts: (0..records).map(|i| (i % 9 + 1) as u16).collect(),
                thresholds: vec![(0..255).collect()],
            };
            let mut options = TrainOptions {
                properties: vec![6],
                parallel: false,
                split_threshold: 0.0,
                ..TrainOptions::default()
            };
            let indices: Vec<_> = (0..records).collect();
            let parents = parent_histograms(&samples, &indices);
            let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
            let mut full = PropertyScratch::default();
            evaluate_property_cached(
                &samples, &indices, &options, &parents, 0, total, 0, None, None, &mut full,
            )
            .unwrap();
            let cache = HistogramCache::create(
                &samples,
                &indices,
                &parents,
                &HistogramBudget::new(32 << 20),
                false,
            )
            .unwrap();
            options.property_bins = super::super::PropertyBins::All;
            options.search = SearchOptions {
                refine_quantiles: true,
                ..SearchOptions::default()
            };
            let mut scratch = PropertyScratch::default();
            let mut previous = None;
            for cached in [None, Some(&cache.properties[0]), None] {
                let actual = evaluate_property_cached(
                    &samples,
                    &indices,
                    &options,
                    &parents,
                    0,
                    total,
                    0,
                    cached,
                    None,
                    &mut scratch,
                )
                .unwrap();
                let mask = scratch.score_mask.clone();
                assert!(!mask.is_empty());
                if let Some(previous) = &previous {
                    assert_eq!(&mask, previous);
                } else {
                    previous = Some(mask.clone());
                }
                for (bin, &score) in mask.iter().enumerate() {
                    if score == 0 || full.used[bin] == 0 {
                        continue;
                    }
                    assert_eq!(
                        scratch.below[bin].bits.to_bits(),
                        full.below[bin].bits.to_bits()
                    );
                    assert_eq!(
                        scratch.above[bin].bits.to_bits(),
                        full.above[bin].bits.to_bits()
                    );
                    assert_eq!(scratch.below[bin].predictor, full.below[bin].predictor);
                    assert_eq!(scratch.above[bin].predictor, full.above[bin].predictor);
                }
                for candidate in actual.into_iter().flatten() {
                    assert_eq!(mask[candidate.bin], 1);
                }
                assert_eq!(
                    scratch.saved_histograms.len(),
                    if cached.is_none() { 2 } else { 0 }
                );
                assert_eq!(scratch.scored_predictors.len(), 4);
            }
        }
    }
}
