// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! enc_ma.cc from libjxl PR #5007, pinned at
//! f714d07bb20973ca216709f25bbb190ca53472f3 (out/research/pr5007.diff).
//! Original quantizer boundaries describe syntax; effective occupied bins
//! describe histograms. In particular, never replace a gap boundary by its bin.

use super::super::{ScoringOptions, SearchOptions, WP_PROPERTY, pack_signed};
use super::{Candidates, Cost, DefaultScorer, Split, SymmetricSplit, TrainOptions, keep_better};
use crate::{Error, Result};

const SPLIT_BASE_MULT: f32 = 0.72;
const SPLIT_LOG_RATIO: f32 = 0.052;
const ONE_SIDED_SYM_PENALTY: f32 = 0.10;
const TWO_SIDED_BASE_MULT: f32 = 1.0;
const TWO_SIDED_SYM_PENALTY: f32 = 10.0;
const TWO_SIDED_MAX_D: i32 = 1;
const TWO_SIDED_MAX_DIFF: i64 = 1;
const TWO_SIDED_MIN_SAMPLES: usize = 64;
const TWO_SIDED_SAMPLES_PER_SYM: usize = 2;
const TWO_SIDED_MID_DROP: f32 = 0.0;
pub(super) const TWO_SIDED_EFFICIENCY_RATIO: f32 = 1.40;

pub(in super::super) fn threshold(options: &TrainOptions) -> f32 {
    let threshold = options.split_threshold as f32;
    if options.tree_learning.enabled() {
        threshold * SPLIT_BASE_MULT
    } else {
        threshold
    }
}

pub(in super::super) fn validate_options(options: &TrainOptions) -> Result<()> {
    if options.tree_learning.enabled()
        && (options.search != SearchOptions::default()
            || options.scoring != ScoringOptions::default())
    {
        return Err(Error::InvalidInput(
            "PR5007 tree learning requires default MA search and scoring".into(),
        ));
    }
    Ok(())
}

fn split_cost(value: i32, threshold: f32, scorer: &DefaultScorer) -> f32 {
    let log_value = scorer.log2(1.0 + pack_signed(value) as f32);
    threshold * (1.0 + SPLIT_LOG_RATIO * log_value)
}

fn alphabet_correction(global_max_symbols: usize, multiplier: f32) -> f32 {
    multiplier * global_max_symbols.saturating_sub(1) as f32
}

pub(super) fn one_sided_cost(
    value: i32,
    options: &TrainOptions,
    global_max_symbols: usize,
    scorer: &DefaultScorer,
) -> f32 {
    if !options.tree_learning.enabled() {
        return 0.0;
    }
    split_cost(value, threshold(options), scorer)
        + alphabet_correction(global_max_symbols, ONE_SIDED_SYM_PENALTY)
}

fn symmetric_property(property: usize) -> bool {
    (10..=14).contains(&property) || (property >= 16 && (property - 16) % 4 == 3)
}

struct TwoSidedCandidate {
    low_bin: usize,
    high_bin: usize,
    effective_low: usize,
    effective_high: usize,
    low_count: usize,
    below_count: usize,
    pixels_mid: u32,
    middle: Cost,
}

#[derive(Default)]
pub(crate) struct SymmetricScratch {
    candidates: Vec<TwoSidedCandidate>,
    cumulative: Vec<usize>,
    effective: Vec<usize>,
    middle_counts: Vec<u32>,
}

impl SymmetricScratch {
    pub(super) fn has_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }

    pub(super) fn prepare(
        &mut self,
        options: &TrainOptions,
        property: usize,
        thresholds: &[i32],
        used: &[usize],
        weights: &[u32],
        global_max_symbols: usize,
    ) {
        self.candidates.clear();
        if !options.tree_learning.symmetric() || !symmetric_property(options.properties[property]) {
            return;
        }
        let Some(first) = used.iter().position(|&n| n != 0) else {
            return;
        };
        let last = used.iter().rposition(|&n| n != 0).unwrap();
        self.cumulative.clear();
        self.cumulative.resize(used.len(), 0);
        self.effective.clear();
        self.effective.resize(used.len(), first);
        let mut count = 0;
        let mut effective = first;
        for (bin, &records) in used.iter().enumerate().take(last + 1).skip(first) {
            if records != 0 {
                effective = bin;
            }
            count += records;
            self.cumulative[bin] = count;
            self.effective[bin] = effective;
        }
        // These are DISTINCT prepared records, not weighted pixel counts.
        let min_required =
            TWO_SIDED_MIN_SAMPLES.max(TWO_SIDED_SAMPLES_PER_SYM.saturating_mul(global_max_symbols));
        for high_bin in first..last {
            let d = thresholds[high_bin];
            if d < 0 {
                continue;
            }
            if d > TWO_SIDED_MAX_D {
                break;
            }
            let target_low = -d - 1;
            let mut best_low = None;
            let mut best_diff = i64::MAX;
            for (bin, &value) in thresholds.iter().enumerate().take(high_bin).skip(first) {
                if value >= 0 {
                    break;
                }
                let diff = (i64::from(value) - i64::from(target_low)).abs();
                if diff < best_diff || (diff == best_diff && value <= target_low) {
                    best_low = Some(bin);
                    best_diff = diff;
                }
            }
            let Some(low_bin) = best_low else { continue };
            if best_diff > TWO_SIDED_MAX_DIFF {
                continue;
            }
            let effective_low = self.effective[low_bin];
            let effective_high = self.effective[high_bin];
            if effective_low >= effective_high {
                continue;
            }
            let low_count = self.cumulative[effective_low];
            let below_count = self.cumulative[effective_high];
            if low_count < min_required
                || below_count - low_count < min_required
                || count - below_count < min_required
            {
                continue;
            }
            self.candidates.push(TwoSidedCandidate {
                low_bin,
                high_bin,
                effective_low,
                effective_high,
                low_count,
                below_count,
                pixels_mid: weights[effective_low + 1..=effective_high].iter().sum(),
                middle: Cost::default(),
            });
        }
    }

    /// At most two narrow middle intervals. Reuse collected histogram rows,
    /// including packed/cached/paired/sharded layouts, without rereading samples
    /// or retaining a full prefix-histogram matrix per predictor.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_middle(
        &mut self,
        scorer: &DefaultScorer,
        predictor: usize,
        penalty: f32,
        symbols: usize,
        increase: &[u32],
        extra: &[u64],
        stride: usize,
        used: &[usize],
        packed: bool,
        property_bins: &[usize],
    ) {
        for candidate in &mut self.candidates {
            self.middle_counts.clear();
            self.middle_counts.resize(symbols, 0);
            let mut extra_mid = 0u64;
            for (bin, &count) in used
                .iter()
                .enumerate()
                .take(candidate.effective_high + 1)
                .skip(candidate.effective_low + 1)
            {
                if count == 0 {
                    continue;
                }
                let row = if packed { property_bins[bin] } else { bin };
                let delta = &increase[row * stride..row * stride + symbols];
                for (middle, &delta) in self.middle_counts.iter_mut().zip(delta) {
                    *middle += delta;
                }
                extra_mid += extra[row];
            }
            let bits =
                scorer.estimate_bits(&self.middle_counts, candidate.pixels_mid) + extra_mid as f32;
            super::update_cost(&mut candidate.middle, bits, penalty, predictor);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish(
        &self,
        options: &TrainOptions,
        property: usize,
        thresholds: &[i32],
        global_max_symbols: usize,
        below: &[Cost],
        above: &[Cost],
        base_bits: f32,
        total: u32,
        result: &mut Candidates,
    ) {
        let scorer = DefaultScorer::default();
        for candidate in &self.candidates {
            let low = below[candidate.effective_low];
            let high = above[candidate.effective_high];
            let mid = candidate.middle;
            if candidate.pixels_mid > 0
                && total > 0
                && mid.bits / candidate.pixels_mid as f32
                    > base_bits / total as f32 - TWO_SIDED_MID_DROP
            {
                continue;
            }
            let threshold = threshold(options);
            let node_cost = TWO_SIDED_BASE_MULT
                * (split_cost(thresholds[candidate.high_bin], threshold, &scorer)
                    + split_cost(thresholds[candidate.low_bin], threshold, &scorer))
                + alphabet_correction(global_max_symbols, TWO_SIDED_SYM_PENALTY);
            let uses_wp = options.properties[property] == WP_PROPERTY
                || [low.predictor, high.predictor, mid.predictor]
                    .iter()
                    .any(|&p| options.predictors[p] == super::Predictor::Weighted);
            keep_better(
                &mut result[if uses_wp { 4 } else { 5 }],
                Split {
                    property,
                    bin: candidate.high_bin,
                    below_count: candidate.below_count,
                    below_predictor: low.predictor,
                    above_predictor: high.predictor,
                    bits: low.bits + high.bits + mid.bits,
                    split_cost: node_cost,
                    symmetric: Some(SymmetricSplit {
                        low_bin: candidate.low_bin,
                        low_count: candidate.low_count,
                        middle_predictor: mid.predictor,
                    }),
                },
            );
        }
    }
}

#[cfg(test)]
#[path = "pr5007_tests.rs"]
mod tests;
