// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Node-local predictor proofs, per-property split costs and ordered candidate choice.
//! Scoring keeps the source arithmetic and sweep order; node dispatch lives in `split_search`.

#[cfg(test)]
use super::PropertyColumn;
use super::{
    BinHistogram, DefaultScorer, DescriptionCost, HistogramSplit, OnceLock, ParentHistogram,
    Predictor, PropertyHistogram, PropertyScratch, PropertyValue, SampleStorage, Samples,
    ScoringOptions, SearchOptions, SparseHistogramPlan, TrainOptions, WP_PROPERTY,
    collect_compact_pair, collect_mapped, collect_property_marginals, histogram_stride, ma_binning,
    ma_dominance, ma_events, ma_search, prepare_property_bins,
};
use crate::{Error, Result};

// Per-predictor timers can dominate small sweeps. RS_JXL_PROFILE=1 keeps
// broad/cache timings; these detail scopes additionally require
// RS_JXL_PROFILE_DETAIL=1, read once before the first detail scope.
fn histogram_detail_scope(name: &'static str) -> Option<crate::profile::Scope> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled =
        *ENABLED.get_or_init(|| std::env::var("RS_JXL_PROFILE_DETAIL").as_deref() == Ok("1"));
    (enabled && crate::profile::enabled()).then(|| crate::profile::scope(name))
}

#[derive(Clone, Copy)]
pub(super) struct Cost {
    pub(super) bits: f32,
    pub(super) penalty: f32,
    pub(super) predictor: usize,
}

impl Default for Cost {
    fn default() -> Self {
        Self {
            bits: f32::INFINITY,
            penalty: 0.0,
            predictor: 0,
        }
    }
}

pub(super) fn predictor_penalty(predictor: Predictor, parent: Predictor, threshold: f32) -> f32 {
    let mut penalty = if predictor != parent && parent != Predictor::Weighted {
        800.0 / (100.0 + threshold)
    } else {
        0.0
    };
    if predictor == Predictor::Weighted {
        penalty += 1e-8;
    }
    if predictor == Predictor::Zero {
        penalty -= 1e-8;
    }
    penalty
}

/// enc_ma.cc::FindBestSplitImpl representative/penalty construction. This proof
/// belongs to one node, not to retained worker scratch or an inherited cache.
pub(super) struct PredictorPlan {
    pub(super) representative: [usize; 14],
    pub(super) penalty: [f32; 14],
    pub(super) active: [usize; 14],
    pub(super) active_count: usize,
}

impl PredictorPlan {
    pub(super) fn identity(predictors: usize) -> Self {
        debug_assert!(predictors <= 14);
        Self {
            representative: std::array::from_fn(|i| i),
            penalty: [0.0; 14],
            active: std::array::from_fn(|i| i),
            active_count: predictors,
        }
    }

    pub(super) fn new(
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        options: &TrainOptions,
        parents: &[ParentHistogram],
        parent_predictor: usize,
    ) -> Self {
        let predictors = parents.len();
        let mut plan = Self::identity(predictors);
        plan.active_count = 0;
        for predictor in 0..predictors {
            let penalty = predictor_penalty(
                options.predictors[predictor],
                options.predictors[parent_predictor],
                options.split_threshold as f32,
            );
            plan.penalty[predictor] = penalty;
            if options.predictor_dedup && predictors > 2 {
                for previous in 0..predictor {
                    if plan.representative[previous] != previous
                        || plan.penalty[previous] != penalty
                        || !Self::same_parent(&parents[previous], &parents[predictor])
                    {
                        continue;
                    }
                    if indices
                        .iter()
                        .all(|&i| samples.residuals[previous][i] == samples.residuals[predictor][i])
                    {
                        plan.representative[predictor] = previous;
                        break;
                    }
                }
            }
            if plan.representative[predictor] == predictor {
                plan.active[plan.active_count] = predictor;
                plan.active_count += 1;
            }
        }
        plan
    }

    fn same_parent(a: &ParentHistogram, b: &ParentHistogram) -> bool {
        // Match source memcmp at the common zero-padded stride, including
        // inherited histograms whose trimmed tails differ in length.
        let shared = a.counts.len().min(b.counts.len());
        a.extra_bits == b.extra_bits
            && a.counts[..shared] == b.counts[..shared]
            && a.counts[shared..]
                .iter()
                .chain(&b.counts[shared..])
                .all(|&n| n == 0)
    }

    pub(super) fn active(&self) -> &[usize] {
        &self.active[..self.active_count]
    }

    pub(super) fn cache_pair14_eligible(&self, predictors: usize) -> bool {
        predictors == 14
            && self.active().iter().copied().eq(0..14)
            && self.representative.iter().copied().eq(0..14)
    }

    pub(super) fn pair_available(&self, predictor: usize, predictors: usize) -> bool {
        let start = predictor & !1;
        start + 1 < predictors
            && self.representative[start] == start
            && self.representative[start + 1] == start + 1
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Split {
    pub(super) property: usize,
    pub(super) bin: usize,
    pub(super) below_count: usize,
    pub(super) below_predictor: usize,
    pub(super) above_predictor: usize,
    pub(super) bits: f32,
}

pub(super) type Candidates = [Option<Split>; 4];

pub(super) fn keep_better(best: &mut Option<Split>, candidate: Split) {
    if best.is_none_or(|old| candidate.bits < old.bits) {
        *best = Some(candidate);
    }
}

/// CollectExtraBitsIncrease + the full FindBestSplit per-property sweep.
// Keep the source sweep inputs explicit alongside its reusable cache and scratch.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_property_values<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    parents: &[ParentHistogram],
    plan: &PredictorPlan,
    sparse_plan: &SparseHistogramPlan,
    total: u32,
    property: usize,
    values: &[T],
    cached: Option<&PropertyHistogram>,
    marginals_ready: bool,
    mapping: Option<&[u8; 256]>,
    scratch: &mut PropertyScratch,
) -> Result<Candidates> {
    debug_assert!(cached.is_none() || mapping.is_none());
    debug_assert!(cached.is_none_or(|histogram| histogram.valid));
    scratch.score_mask.clear();
    scratch.dominated_predictors = 0;
    #[cfg(test)]
    {
        scratch.scored_predictors.clear();
        scratch.sharded_predictors.clear();
        scratch.sparse_predictors.clear();
        scratch.packed_bins = 0;
        scratch.events.reset_trace();
    }
    let bins = samples.thresholds[property].len() + 1;
    if cached.is_none() && !marginals_ready {
        collect_property_marginals(
            samples,
            indices,
            values,
            bins,
            &mut scratch.used,
            &mut scratch.weights,
        );
    }
    let used = cached.map_or(scratch.used.as_slice(), |histogram| {
        histogram.used.as_slice()
    });
    let weights = cached.map_or(scratch.weights.as_slice(), |histogram| {
        histogram.weights.as_slice()
    });
    let first = used.iter().position(|&n| n != 0).unwrap_or(0);
    let last = used.iter().rposition(|&n| n != 0).unwrap_or(0);
    if first == last {
        return Ok([None; 4]);
    }
    scratch.below.clear();
    scratch.below.resize(bins, Cost::default());
    scratch.above.clear();
    scratch.above.resize(bins, Cost::default());
    let coarse_rows = if cached.is_none() {
        ma_binning::prepare(
            options.search.histogram_bins,
            options.properties[property],
            used,
            weights,
            total,
            &mut scratch.property_bins,
            &mut scratch.splits,
            &mut scratch.score_mask,
        )?
    } else {
        None
    };
    let coarsened = coarse_rows.is_some();
    let _binning_scope = coarsened.then(|| crate::profile::scope("ma.adaptive_histogram"));
    if !coarsened
        && ma_events::eligible(
            options,
            total,
            indices.len(),
            used,
            parents,
            plan,
            sparse_plan,
            cached.is_some(),
        )
    {
        scratch.events.evaluate(
            samples,
            indices,
            values,
            used,
            weights,
            parents,
            plan,
            options,
            total,
            mapping,
            &mut scratch.below,
            &mut scratch.above,
        )?;
        #[cfg(test)]
        scratch.scored_predictors.extend_from_slice(plan.active());
        return Ok(property_candidates(
            options,
            property,
            used,
            (&scratch.below, &scratch.above),
            first..last,
            &[],
        ));
    }
    let rows = coarse_rows.unwrap_or_else(|| {
        prepare_property_bins(
            used,
            options.property_bins,
            indices.len(),
            mapping.is_some(),
            cached.is_some(),
            &mut scratch.property_bins,
        )
    });
    let packed = rows != bins;
    // Property weights and live thresholds do not depend on the predictor.
    // Small descendants often occupy only a handful of the root's 256 bins.
    // One/two predictors do not amortize materializing a separate split table.
    let reuse_splits = coarsened
        || plan.active_count >= 4
        || options.scoring != ScoringOptions::default()
        || !options.search.default_sweep();
    if !coarsened {
        scratch.splits.clear();
    }
    if reuse_splits && !coarsened {
        let mut weight_below = 0;
        for bin in first..last {
            if used[bin] != 0 {
                weight_below += weights[bin];
                scratch.splits.push(HistogramSplit {
                    bin,
                    row: if packed {
                        scratch.property_bins[bin]
                    } else {
                        bin
                    },
                    totals: (total - weight_below, weight_below),
                });
            }
        }
    }
    #[cfg(test)]
    {
        scratch.packed_bins = if packed { rows } else { 0 };
    }
    let below_costs = &mut scratch.below;
    let above_costs = &mut scratch.above;
    let scorer = DefaultScorer::default();
    let node_stride = histogram_stride(
        parents.iter().map(|p| p.counts.len()).max().unwrap_or(1),
        scorer.lanes(),
        options.padded_histogram && mapping.is_none(),
    );
    let refine = !coarsened
        && ma_search::prepare_mask(
            options.search,
            options.scoring,
            indices.len(),
            options.properties[property],
            used,
            weights,
            total,
            first,
            last,
            rows.checked_mul(node_stride)
                .ok_or(Error::Limit("MA refinement histogram size"))?,
            parents.len(),
            &mut scratch.score_mask,
        )?;
    let saved_count = if refine && cached.is_none() {
        parents.len()
    } else {
        0
    };
    scratch.saved_histograms.truncate(saved_count);
    scratch
        .saved_histograms
        .resize_with(saved_count, || BinHistogram {
            symbols: 0,
            counts: Vec::new(),
            extra: Vec::new(),
        });
    for pass in 0..1 + usize::from(refine) {
        if pass != 0 {
            ma_search::refine_mask_by(&mut scratch.score_mask, |offset| {
                let bin = first + offset;
                (below_costs[bin].bits + below_costs[bin].penalty)
                    + (above_costs[bin].bits + above_costs[bin].penalty)
            });
        }
        for &predictor in plan.active() {
            let parent = &parents[predictor];
            #[cfg(test)]
            scratch.scored_predictors.push(predictor);
            let symbols = parent.counts.len();
            let storage_stride = if options.padded_histogram {
                node_stride
            } else {
                symbols
            };
            let collect_scope = histogram_detail_scope(if cached.is_some() {
                "ma.histogram_cached"
            } else if pass != 0 {
                "ma.histogram_saved"
            } else {
                "ma.histogram_collect"
            });
            let (increase, extra, stride) = if let Some(cached) = cached {
                let histogram = &cached.predictors[predictor];
                (
                    histogram.counts.as_slice(),
                    histogram.extra.as_slice(),
                    histogram.symbols,
                )
            } else if pass != 0 {
                let saved = &scratch.saved_histograms[predictor];
                (
                    saved.counts.as_slice(),
                    saved.extra.as_slice(),
                    saved.symbols,
                )
            } else {
                let paired = !packed
                    && plan.pair_available(predictor, parents.len())
                    && if mapping.is_some() {
                        // Every active adjacent pair can share the compact
                        // property/weight scan, including Variable's 14 tools.
                        indices.len() <= 256
                    } else {
                        indices.len() >= 256
                    };
                if paired && predictor % 2 == 1 {
                    // The preceding predictor populated both exact histograms.
                    // Swapping preserves capacity and handles unequal strides.
                    std::mem::swap(&mut scratch.increase, &mut scratch.paired_increase);
                    std::mem::swap(&mut scratch.extra, &mut scratch.paired_extra);
                } else {
                    scratch.increase.clear();
                    scratch.increase.resize(rows * storage_stride, 0);
                    scratch.extra.clear();
                    scratch.extra.resize(rows, 0);
                    let tokens = &samples.residuals[predictor];
                    if paired {
                        let other_symbols = if options.padded_histogram {
                            node_stride
                        } else {
                            parents[predictor + 1].counts.len()
                        };
                        let other_tokens = &samples.residuals[predictor + 1];
                        scratch.paired_increase.clear();
                        scratch.paired_increase.resize(bins * other_symbols, 0);
                        scratch.paired_extra.clear();
                        scratch.paired_extra.resize(bins, 0);
                        if let Some(mapping) = mapping {
                            collect_compact_pair(
                                samples,
                                indices,
                                values,
                                [tokens, other_tokens],
                                [storage_stride, other_symbols],
                                mapping,
                                [&mut scratch.increase, &mut scratch.paired_increase],
                                [&mut scratch.extra, &mut scratch.paired_extra],
                            );
                        } else {
                            for &i in indices {
                                let bin = values[i].index();
                                let weight = u32::from(samples.counts[i]);
                                let a = tokens[i];
                                let b = other_tokens[i];
                                scratch.increase[bin * storage_stride + usize::from(a.token)] +=
                                    weight;
                                scratch.paired_increase
                                    [bin * other_symbols + usize::from(b.token)] += weight;
                                scratch.extra[bin] += u64::from(weight) * u64::from(a.nbits);
                                scratch.paired_extra[bin] += u64::from(weight) * u64::from(b.nbits);
                            }
                        }
                    } else if options.histogram_shards
                        && !packed
                        && mapping.is_none()
                        && indices.len() >= 4096
                        && indices.len() >= 4 * bins * node_stride
                    {
                        scratch.shards.collect(
                            samples,
                            indices,
                            values,
                            tokens,
                            storage_stride,
                            bins,
                            &mut scratch.increase,
                            &mut scratch.extra,
                        )?;
                        #[cfg(test)]
                        scratch.sharded_predictors.push(predictor);
                    } else {
                        match (packed, mapping.is_some()) {
                            (false, false) => collect_mapped::<T, false, false>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &[],
                                None,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                            (false, true) => collect_mapped::<T, false, true>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &[],
                                mapping,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                            (true, false) => collect_mapped::<T, true, false>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &scratch.property_bins,
                                None,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                            (true, true) => collect_mapped::<T, true, true>(
                                samples,
                                indices,
                                values,
                                tokens,
                                storage_stride,
                                &scratch.property_bins,
                                mapping,
                                &mut scratch.increase,
                                &mut scratch.extra,
                            ),
                        }
                    }
                }
                if refine {
                    let saved = &mut scratch.saved_histograms[predictor];
                    saved.symbols = storage_stride;
                    saved
                        .counts
                        .try_reserve(scratch.increase.len().saturating_sub(saved.counts.len()))
                        .map_err(|_| Error::Limit("MA refinement counts allocation"))?;
                    saved
                        .extra
                        .try_reserve(scratch.extra.len().saturating_sub(saved.extra.len()))
                        .map_err(|_| Error::Limit("MA refinement extra allocation"))?;
                    saved.counts.clear();
                    saved.counts.extend_from_slice(&scratch.increase);
                    saved.extra.clear();
                    saved.extra.extend_from_slice(&scratch.extra);
                }
                (
                    scratch.increase.as_slice(),
                    scratch.extra.as_slice(),
                    storage_stride,
                )
            };
            drop(collect_scope);
            // Both child populations use the nonnegative <=256 contribution
            // table. This lower bound keeps the exact f32 extra-bit order;
            // it is local to this node and must not be inherited by children.
            if options.split_dominance
                && total <= 256
                && options.scoring == ScoringOptions::default()
                && options.search == SearchOptions::default()
                && ma_dominance::predictor_cannot_improve(
                    (first..last).filter(|&bin| used[bin] != 0).map(|bin| {
                        (
                            bin,
                            if packed {
                                scratch.property_bins[bin]
                            } else {
                                bin
                            },
                        )
                    }),
                    extra,
                    parent.extra_bits,
                    plan.penalty[predictor],
                    below_costs,
                    above_costs,
                )
            {
                scratch.dominated_predictors += 1;
                continue;
            }
            let sparse = sparse_plan.blocks(predictor);
            #[cfg(test)]
            if sparse.is_some() {
                scratch.sparse_predictors.push(predictor);
            }
            scratch.counts_above.clone_from(&parent.counts);
            scratch.counts_below.clear();
            scratch.counts_below.resize(symbols, 0);
            scratch
                .scoring
                .reset(options.scoring, &parent.counts, total, indices.len())?;
            let penalty = plan.penalty[predictor];
            let mut extra_below = 0u64;
            let _score_scope = histogram_detail_scope("ma.histogram_score");
            if options.scoring == ScoringOptions::default()
                && (scratch.score_mask.is_empty() || coarsened)
            {
                let mut weight_below = 0;
                let count = if reuse_splits {
                    scratch.splits.len()
                } else {
                    last - first
                };
                let splits = (0..count).filter_map(|offset| {
                    if reuse_splits {
                        return Some(scratch.splits[offset]);
                    }
                    let bin = first + offset;
                    if used[bin] == 0 {
                        return None;
                    }
                    weight_below += weights[bin];
                    Some(HistogramSplit {
                        bin,
                        row: if packed {
                            scratch.property_bins[bin]
                        } else {
                            bin
                        },
                        totals: (total - weight_below, weight_below),
                    })
                });
                scorer.split_sweep(
                    &mut scratch.counts_above,
                    &mut scratch.counts_below,
                    increase,
                    stride,
                    total,
                    sparse,
                    splits,
                    |split, rbits, lbits| {
                        extra_below += extra[split.row];
                        // Keep the same f32 extra-bit and strict tie order as
                        // the configurable per-threshold scoring path below.
                        update_cost(
                            &mut below_costs[split.bin],
                            lbits + extra_below as f32,
                            penalty,
                            predictor,
                        );
                        update_cost(
                            &mut above_costs[split.bin],
                            (rbits + parent.extra_bits as f32) - extra_below as f32,
                            penalty,
                            predictor,
                        );
                    },
                );
                continue;
            }
            for split in &scratch.splits {
                let bin = split.bin;
                let row = split.row;
                extra_below += extra[row];
                // Validated samples and exact cached subsets bound every move and
                // signed SIMD conversion. Keep the cache immutable for child reuse.
                let delta = &increase[row * stride..row * stride + symbols];
                if !scratch.score_mask.is_empty() && scratch.score_mask[bin - first] == 0 {
                    scorer.move_counts(&mut scratch.counts_above, &mut scratch.counts_below, delta);
                    continue;
                }
                let totals = split.totals;
                let (rbits, lbits) = scratch.scoring.move_and_score(
                    options.scoring,
                    &scorer,
                    &mut scratch.counts_above,
                    &mut scratch.counts_below,
                    delta,
                    totals,
                    sparse,
                );
                let mut below_bits = lbits + extra_below as f32;
                // Preserve source f32 addition/subtraction order, not an integer
                // difference followed by one conversion.
                let mut above_bits = (rbits + parent.extra_bits as f32) - extra_below as f32;
                if options.scoring.description != DescriptionCost::Disabled {
                    below_bits +=
                        options
                            .scoring
                            .description
                            .cost(&scorer, &scratch.counts_below, totals.1);
                    above_bits +=
                        options
                            .scoring
                            .description
                            .cost(&scorer, &scratch.counts_above, totals.0);
                }
                update_cost(&mut below_costs[bin], below_bits, penalty, predictor);
                update_cost(&mut above_costs[bin], above_bits, penalty, predictor);
            }
        }
    }
    Ok(property_candidates(
        options,
        property,
        used,
        (below_costs, above_costs),
        first..last,
        &scratch.score_mask,
    ))
}

#[inline]
fn update_cost(best: &mut Cost, bits: f32, penalty: f32, predictor: usize) {
    if bits + penalty < best.bits + best.penalty {
        *best = Cost {
            bits,
            penalty,
            predictor,
        };
    }
}

fn property_candidates(
    options: &TrainOptions,
    property: usize,
    used: &[usize],
    costs: (&[Cost], &[Cost]),
    range: std::ops::Range<usize>,
    score_mask: &[u8],
) -> Candidates {
    let (below_costs, above_costs) = costs;
    let first = range.start;
    let mut result = [None; 4];
    let mut below_count = 0;
    for bin in range {
        if used[bin] == 0 {
            continue;
        }
        below_count += used[bin];
        if !score_mask.is_empty() && score_mask[bin - first] == 0 {
            continue;
        }
        let below = below_costs[bin];
        let above = above_costs[bin];
        let uses_wp = options.properties[property] == WP_PROPERTY
            || options.predictors[below.predictor] == Predictor::Weighted
            || options.predictors[above.predictor] == Predictor::Weighted;
        let class = if options.properties[property] < 2 {
            if below.bits == 0.0 || above.bits == 0.0 {
                0
            } else {
                1
            }
        } else if uses_wp {
            2
        } else {
            3
        };
        // Source penalties select predictors but are NOT part of split cost.
        keep_better(
            &mut result[class],
            Split {
                property,
                bin,
                below_count,
                below_predictor: below.predictor,
                above_predictor: above.predictor,
                bits: below.bits + above.bits,
            },
        );
    }
    result
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_property_cached(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    parents: &[ParentHistogram],
    parent_predictor: usize,
    total: u32,
    property: usize,
    cached: Option<&PropertyHistogram>,
    mapping: Option<&[u8; 256]>,
    scratch: &mut PropertyScratch,
) -> Result<Candidates> {
    let plan = PredictorPlan::new(samples, indices, options, parents, parent_predictor);
    let sparse_plan = SparseHistogramPlan::new(
        parents,
        plan.active(),
        DefaultScorer::new().lanes(),
        options.sparse_entropy && !options.scoring.fast_entropy,
    );
    with_property_values!(&samples.properties[property], |values| {
        evaluate_property_values(
            samples,
            indices,
            options,
            parents,
            &plan,
            &sparse_plan,
            total,
            property,
            values,
            cached,
            false,
            mapping,
            scratch,
        )
    })
}

#[cfg(test)]
pub(super) fn evaluate_property(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    options: &TrainOptions,
    parents: &[ParentHistogram],
    parent_predictor: usize,
    total: u32,
    property: usize,
) -> Result<Candidates> {
    evaluate_property_cached(
        samples,
        indices,
        options,
        parents,
        parent_predictor,
        total,
        property,
        None,
        None,
        &mut PropertyScratch::default(),
    )
}

pub(super) fn choose_split(
    candidates: Candidates,
    base_bits: f32,
    options: &TrainOptions,
) -> Option<Split> {
    let threshold = options.split_threshold as f32;
    let mut best = candidates[2];
    for candidate in [candidates[3], candidates[1]].into_iter().flatten() {
        if candidate.bits + threshold < base_bits
            && candidate.bits
                <= options.fast_decode_multiplier as f32 * best.map_or(f32::INFINITY, |b| b.bits)
        {
            best = Some(candidate);
        }
    }
    if let Some(candidate) = candidates[0]
        && candidate.bits + threshold < base_bits
    {
        best = Some(candidate);
    }
    best.filter(|candidate| candidate.bits + threshold < base_bits)
}
