// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.

use super::super::{
    Candidates, DenseHistogram, DescriptionCost, HistogramBudget, HistogramCache, PropertyColumn,
    PropertyScratch, QuantileSplits, ResidualToken, TreeSamples,
    compact_parent_histograms_with_layout, evaluate_property_cached, parent_histograms,
};
use super::*;

fn clipped_reference(counts: &[u32]) -> f64 {
    let total: u32 = counts.iter().sum();
    counts
        .iter()
        .filter(|&&c| c != 0)
        .map(|&c| f64::from(c) * (f64::from(total) / f64::from(c)).log2().min(12.0))
        .sum()
}

fn close(actual: f32, expected: f64) {
    let tolerance = 4.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0);
    assert!(
        (f64::from(actual) - expected).abs() <= tolerance,
        "{actual} != {expected}"
    );
}

#[test]
fn events_clipped_entropy_matches_every_weighted_move_and_floor_boundary() {
    let table = CountEntropyTable::shared();
    let mut above = EventEntropy::default();
    let mut below = EventEntropy::default();
    for total in [
        0u32, 1, 2, 255, 256, 257, 4095, 4096, 4097, 8192, 8193, 65535, 65536,
    ] {
        let mut parent = vec![0u32; 256];
        let mut remaining = total;
        for (i, count) in parent[1..=15].iter_mut().enumerate() {
            *count = remaining.min(i as u32 + 1);
            remaining -= *count;
        }
        parent[255] = remaining;
        for reverse in [false, true] {
            above.reset(&parent, false, table).unwrap();
            below.reset(&parent, true, table).unwrap();
            let mut right = total;
            close(above.bits(right, table), clipped_reference(&parent));
            close(below.bits(0, table), 0.0);
            let mut symbols: Vec<_> = (0..parent.len()).collect();
            if reverse {
                symbols.reverse();
            }
            for symbol in symbols {
                // Split large counts as well as moving rare counts in/out of
                // the live tail on both sides of the monotone total cursor.
                let count = parent[symbol];
                for delta in [count / 3, count - count / 3] {
                    if delta == 0 {
                        continue;
                    }
                    above.replace(symbol, above.counts[symbol] - delta, table);
                    below.replace(symbol, below.counts[symbol] + delta, table);
                    right -= delta;
                    close(above.bits(right, table), clipped_reference(&above.counts));
                    close(
                        below.bits(total - right, table),
                        clipped_reference(&below.counts),
                    );
                    assert_eq!(
                        above.active,
                        above.counts.iter().filter(|&&c| c != 0).count()
                    );
                    assert_eq!(
                        below.active,
                        below.counts.iter().filter(|&&c| c != 0).count()
                    );
                }
            }
            assert_eq!(right, 0);
            assert_eq!(below.counts, parent);
        }
    }
    let parent = [1, 65535];
    above.reset(&parent, false, table).unwrap();
    let clipped = above.bits(65536, table);
    let shannon = table.get(65536) - table.get(65535);
    assert!(
        shannon - f64::from(clipped) > 3.9,
        "must not silently use Shannon"
    );
}

#[test]
fn events_gate_uses_weighted_total_actual_width_and_conservative_work_savings() {
    let mut options = TrainOptions::variable();
    options.predictors.truncate(3);
    let plan = PredictorPlan::identity(3);
    let parents = vec![
        ParentHistogram {
            counts: vec![1; 64],
            extra_bits: 0
        };
        3
    ];
    let sparse = SparseHistogramPlan::default();
    let mut used = vec![7; 9];
    used[0] = 8;
    assert!(!eligible(
        &options, 65536, 64, &used, &parents, &plan, &sparse, false
    ));
    options.histogram_events = true;
    assert!(eligible(
        &options, 65536, 64, &used, &parents, &plan, &sparse, false
    ));
    assert!(!eligible(
        &options, 65537, 64, &used, &parents, &plan, &sparse, false
    ));
    assert!(!eligible(
        &options, 65536, 64, &used, &parents, &plan, &sparse, true
    ));
    used[0] += 1;
    assert!(!eligible(
        &options, 65, 65, &used, &parents, &plan, &sparse, false
    ));
    used[0] -= 1;
    for search in [
        SearchOptions {
            statistical_race: true,
            ..SearchOptions::default()
        },
        SearchOptions {
            refine_quantiles: true,
            ..SearchOptions::default()
        },
        SearchOptions {
            quantile_splits: QuantileSplits::ThirtyTwo,
            ..SearchOptions::default()
        },
    ] {
        options.search = search;
        assert!(!eligible(
            &options, 64, 64, &used, &parents, &plan, &sparse, false
        ));
    }
    options.search = SearchOptions::default();
    options.scoring.fast_entropy = true;
    assert!(!eligible(
        &options, 64, 64, &used, &parents, &plan, &sparse, false
    ));
    options.scoring.fast_entropy = false;
    let mut holes = parents.clone();
    for p in &mut holes {
        p.counts.fill(0);
        p.counts[0] = 1;
        p.counts[63] = 1;
    }
    let sparse =
        SparseHistogramPlan::new(&holes, plan.active(), DefaultScorer::new().lanes(), true);
    assert!(!eligible(
        &options, 64, 64, &used, &holes, &plan, &sparse, false
    ));
    for p in &mut holes {
        p.counts.truncate(8);
    }
    assert!(!eligible(
        &options,
        64,
        64,
        &used,
        &holes,
        &plan,
        &SparseHistogramPlan::default(),
        false
    ));
    assert!(!eligible(
        &options,
        64,
        64,
        &[0, 64, 0],
        &parents,
        &plan,
        &sparse,
        false
    ));
    let scratch = EventScratch::default();
    assert_eq!(
        scratch.order.capacity() + scratch.delta.capacity() + scratch.above.terms.capacity(),
        0
    );
}

#[test]
fn events_buckets_keep_wide_bins_subsets_holes_and_reused_storage() {
    let values = [1000u32, 0, 4, 1000, 4, 1];
    let mut scratch = EventScratch::default();
    for indices in [vec![4, 3, 2, 0], vec![5, 1], vec![]] {
        let mut used = vec![0; 1002];
        for &i in &indices {
            used[values[i] as usize] += 1;
        }
        scratch.bucket(&indices, &values, &used).unwrap();
        assert_eq!(scratch.order.len(), indices.len());
        for bin in 0..used.len() {
            let expected: Vec<_> = indices
                .iter()
                .copied()
                .filter(|&i| values[i] as usize == bin)
                .collect();
            assert_eq!(
                scratch.order[scratch.offsets[bin]..scratch.offsets[bin + 1]],
                expected
            );
        }
        assert!(scratch.delta.iter().all(|&d| d == 0));
    }
    scratch.bucket(&[1], &[0u8, 1], &[0, 1]).unwrap();
    assert_eq!(scratch.offsets, [0, 0, 1]);
    assert_eq!(scratch.order, [1]);
}

fn fixture(predictors: usize, total: u32) -> (TreeSamples, TrainOptions) {
    let mut options = TrainOptions::variable();
    options.predictors.truncate(predictors);
    options.properties = vec![0, 6, 15, 7];
    options.histogram_events = true;
    let mut counts: Vec<u16> = (0..192).map(|i| 1 + (i % 3) as u16).collect();
    let rest: u32 = counts.iter().skip(1).map(|&n| u32::from(n)).sum();
    counts[0] = u16::try_from(total - rest).unwrap();
    let samples = TreeSamples {
        properties: vec![
            PropertyColumn::Wide((0..192).map(|i| (i / 3 * 11 + 3) as u32).collect()),
            PropertyColumn::Packed((0..192).map(|i| (i / 3 * 3 + 2) as u8).collect()),
            PropertyColumn::Packed((0..192).map(|i| ((i * 37) % 64) as u8).collect()),
            PropertyColumn::Packed(vec![2; 192]),
        ],
        residuals: (0..predictors)
            .map(|p| {
                (0..192)
                    .map(|i| ResidualToken {
                        token: (i / 3 + i / 24 * 16 + p % 4 * 8) as u8,
                        nbits: ((i + 3 * p) % 29) as u8,
                    })
                    .collect()
            })
            .collect(),
        counts,
        thresholds: vec![
            (0..704).collect(),
            (0..200).collect(),
            (0..64).collect(),
            vec![0, 1, 2],
        ],
    };
    (samples, options)
}

// Independent per-threshold full histogram scans; no event state or algebraic
// N*log(N)-sum shortcut. Return every predictor's two costs, not just winners.
fn reference_costs(
    samples: &TreeSamples,
    indices: &[usize],
    options: &TrainOptions,
    property: usize,
    used: &[usize],
) -> Vec<Vec<(f32, f32)>> {
    let bins = used.len();
    let last = used.iter().rposition(|&n| n != 0).unwrap();
    let scorer = DefaultScorer::new();
    (0..options.predictors.len())
        .map(|p| {
            let mut costs = vec![(f32::INFINITY, f32::INFINITY); bins];
            for bin in 0..last {
                if used[bin] == 0 {
                    continue;
                }
                let mut below = [0u32; 256];
                let mut above = [0u32; 256];
                let mut extra_below = 0u64;
                let mut extra_total = 0u64;
                for &i in indices {
                    let token = samples.residuals[p][i];
                    let weight = u32::from(samples.counts[i]);
                    let extra = u64::from(weight) * u64::from(token.nbits);
                    extra_total += extra;
                    if samples.properties[property].value(i) as usize <= bin {
                        below[usize::from(token.token)] += weight;
                        extra_below += extra;
                    } else {
                        above[usize::from(token.token)] += weight;
                    }
                }
                let mut l = clipped_reference(&below) as f32 + extra_below as f32;
                let mut r =
                    (clipped_reference(&above) as f32 + extra_total as f32) - extra_below as f32;
                l += options
                    .scoring
                    .description
                    .cost(&scorer, &below, below.iter().sum());
                r += options
                    .scoring
                    .description
                    .cost(&scorer, &above, above.iter().sum());
                costs[bin] = (l, r);
            }
            costs
        })
        .collect()
}

#[test]
fn events_all_predictors_and_thresholds_match_reference_with_compaction_and_weights() {
    let mut scratch = PropertyScratch::default();
    for (predictors, layout) in [
        (14, None),
        (2, Some(DenseHistogram::Small)),
        (2, Some(DenseHistogram::Disabled)),
    ] {
        for total in [4096, 4097, 65536] {
            let (samples, mut options) = fixture(predictors, total);
            let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
            let parents = parent_histograms(&samples, &indices);
            let original = parents.clone();
            let compact = layout.map(|layout| {
                compact_parent_histograms_with_layout(
                    &parents,
                    DefaultScorer::new().lanes(),
                    layout,
                    false,
                )
                .unwrap()
            });
            let scoring = compact
                .as_ref()
                .map_or(parents.as_slice(), |c| c.parents.as_slice());
            let mapping = compact.as_ref().map(|c| &c.mapping);
            for description in [DescriptionCost::Disabled, DescriptionCost::Full] {
                options.scoring.description = description;
                for property in 0..3 {
                    let actual = evaluate_property_cached(
                        &samples,
                        &indices,
                        &options,
                        scoring,
                        1,
                        total,
                        property,
                        None,
                        mapping,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        scratch.scored_predictors,
                        (0..predictors).collect::<Vec<_>>()
                    );
                    assert_eq!(scratch.events.thresholds, 63 * predictors);
                    assert!(scratch.events.updates >= scratch.events.thresholds);
                    assert!(
                        scratch.increase.is_empty(),
                        "event route must not allocate a matrix"
                    );
                    let expected =
                        reference_costs(&samples, &indices, &options, property, &scratch.used);
                    let plan = PredictorPlan::new(&samples, &indices, &options, scoring, 1);
                    let last = scratch.used.iter().rposition(|&n| n != 0).unwrap();
                    for bin in 0..scratch.used.len() {
                        for (side, costs) in
                            [&scratch.below, &scratch.above].into_iter().enumerate()
                        {
                            let cost = costs[bin];
                            if scratch.used[bin] == 0 || bin >= last {
                                assert!(cost.bits.is_infinite());
                                continue;
                            }
                            let select = |pair: (f32, f32)| if side == 0 { pair.0 } else { pair.1 };
                            close(cost.bits, f64::from(select(expected[cost.predictor][bin])));
                            assert_eq!(cost.penalty, plan.penalty[cost.predictor]);
                            let best = (0..predictors)
                                .map(|p| select(expected[p][bin]) + plan.penalty[p])
                                .fold(f32::INFINITY, f32::min);
                            close(cost.bits + cost.penalty, f64::from(best));
                        }
                    }
                    assert!(actual.iter().any(Option::is_some));
                    for candidate in actual.into_iter().flatten() {
                        assert_eq!(
                            candidate.below_count,
                            scratch.used[..=candidate.bin].iter().sum::<usize>()
                        );
                        assert!(scratch.used[candidate.bin] != 0 && candidate.bin < last);
                    }
                }
            }
            assert_eq!(parents, original);
        }
    }
}

fn same_candidates(a: Candidates, b: Candidates) {
    for (a, b) in a.into_iter().zip(b) {
        match (a, b) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(
                    (
                        a.property,
                        a.bin,
                        a.below_count,
                        a.below_predictor,
                        a.above_predictor,
                        a.bits.to_bits()
                    ),
                    (
                        b.property,
                        b.bin,
                        b.below_count,
                        b.below_predictor,
                        b.above_predictor,
                        b.bits.to_bits()
                    )
                );
            }
            _ => panic!("event fallback changed candidates"),
        }
    }
}

#[test]
fn events_fallback_and_cache_subtraction_preserve_routes_and_reused_scratch() {
    let (mut samples, mut options) = fixture(3, 65536);
    let all: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &all);
    let mut scratch = PropertyScratch::default();
    evaluate_property_cached(
        &samples,
        &all,
        &options,
        &parents,
        0,
        65536,
        1,
        None,
        None,
        &mut scratch,
    )
    .unwrap();
    assert!(scratch.events.thresholds != 0);
    let mut cache = HistogramCache::create(
        &samples,
        &all,
        &parents,
        &HistogramBudget::new(32 * 1024 * 1024),
        false,
    )
    .unwrap();
    for removed in [0, 7] {
        if removed != 0 {
            cache.subtract(&samples, &all[..removed]).unwrap();
        }
        let indices = &all[removed..];
        let parents = parent_histograms(&samples, indices);
        let total = indices.iter().map(|&i| u32::from(samples.counts[i])).sum();
        let before = cache.properties[1].predictors[0].counts.clone();
        let mut reference = PropertyScratch::default();
        let mut disabled = options.clone();
        disabled.histogram_events = false;
        let a = evaluate_property_cached(
            &samples,
            indices,
            &options,
            &parents,
            0,
            total,
            1,
            Some(&cache.properties[1]),
            None,
            &mut scratch,
        )
        .unwrap();
        let b = evaluate_property_cached(
            &samples,
            indices,
            &disabled,
            &parents,
            0,
            total,
            1,
            Some(&cache.properties[1]),
            None,
            &mut reference,
        )
        .unwrap();
        same_candidates(a, b);
        assert_eq!(scratch.events.thresholds, 0);
        assert_eq!(cache.properties[1].predictors[0].counts, before);
    }
    for case in 0..5 {
        options.scoring.fast_entropy = case == 1;
        options.search.refine_quantiles = case == 2;
        samples.counts[1] = if case == 3 { 3 } else { 2 };
        let parents = parent_histograms(&samples, &all);
        let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
        // A constant column takes the normal early exit; a single-token
        // predictor shape exercises the dense-work gate independently.
        let property = if case == 0 { 3 } else { 1 };
        if case == 4 {
            for tokens in &mut samples.residuals {
                for token in tokens {
                    token.token = 0;
                }
            }
        }
        let parents = if case == 4 {
            parent_histograms(&samples, &all)
        } else {
            parents
        };
        let mut disabled = options.clone();
        disabled.histogram_events = false;
        let a = evaluate_property_cached(
            &samples,
            &all,
            &options,
            &parents,
            0,
            total,
            property,
            None,
            None,
            &mut scratch,
        )
        .unwrap();
        let b = evaluate_property_cached(
            &samples,
            &all,
            &disabled,
            &parents,
            0,
            total,
            property,
            None,
            None,
            &mut PropertyScratch::default(),
        )
        .unwrap();
        same_candidates(a, b);
        assert_eq!(scratch.events.thresholds, 0);
    }
}
