// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

use super::super::super::split_search::{duplicate_property, find_best_split_with_scratch};
use super::super::super::{
    HistogramBudget, PropertyBins, PropertyColumn, PropertyHistogram, PropertyScratch,
    ResidualToken, ScratchPool, TrainingStats, TreeLearning, TreeSamples, parent_histograms,
};
use super::super::{choose_split, evaluate_property_cached, predictor_penalty};
use super::*;

fn options(mode: TreeLearning) -> TrainOptions {
    TrainOptions {
        tree_learning: mode,
        properties: vec![10],
        predictors: vec![
            super::super::Predictor::Zero,
            super::super::Predictor::Gradient,
            super::super::Predictor::Top,
        ],
        split_threshold: 10.0,
        parallel: false,
        ..TrainOptions::default()
    }
}

fn split(bits: f32, split_cost: f32, symmetric: bool) -> Split {
    Split {
        property: 0,
        bin: 1,
        below_count: 128,
        below_predictor: 0,
        above_predictor: 2,
        bits,
        split_cost,
        symmetric: symmetric.then_some(SymmetricSplit {
            low_bin: 0,
            low_count: 64,
            middle_predictor: 1,
        }),
    }
}

// Prepared records intentionally remain distinct. Each region has its own
// perfect predictor; the other two predictors have positive binary entropy.
fn fixture() -> TreeSamples {
    let mut samples = TreeSamples {
        properties: vec![PropertyColumn::Packed(Vec::new())],
        residuals: vec![Vec::new(), Vec::new(), Vec::new()],
        counts: vec![1; 192],
        thresholds: vec![vec![-1, 0]],
        global_max_symbols: 3,
    };
    let PropertyColumn::Packed(values) = &mut samples.properties[0] else {
        unreachable!()
    };
    for region in 0..3 {
        for i in 0..64 {
            values.push(region as u8);
            for predictor in 0..3 {
                samples.residuals[predictor].push(ResidualToken {
                    token: if predictor == region {
                        0
                    } else {
                        1 + (i % 2) as u8
                    },
                    nbits: 0,
                });
            }
        }
    }
    samples
}

#[test]
fn source_constants_signed_cost_global_alphabet_and_scaled_penalty() {
    let scorer = DefaultScorer::default();
    let options = options(TreeLearning::Cost);
    assert_eq!(threshold(&options), 10.0 * 0.72);
    for value in [0, -1, 1, -2, 2, i32::MIN, i32::MAX] {
        let expected =
            7.2 * (1.0 + 0.052 * scorer.log2(1.0 + pack_signed(value) as f32)) + 0.10 * 16.0;
        assert!((one_sided_cost(value, &options, 17, &scorer) - expected).abs() < 0.00001);
    }
    assert!(one_sided_cost(-1, &options, 17, &scorer) < one_sided_cost(1, &options, 17, &scorer));
    let predictor = super::super::Predictor::Top;
    let parent = super::super::Predictor::Gradient;
    assert_eq!(
        predictor_penalty(predictor, parent, threshold(&options)),
        800.0 / 107.2
    );
    for symbols in [0, 1] {
        assert_eq!(
            one_sided_cost(0, &options, symbols, &scorer),
            split_cost(0, threshold(&options), &scorer)
        );
    }
    let legacy = options_for_legacy();
    assert_eq!(threshold(&legacy), 10.0);
    assert_eq!(one_sided_cost(i32::MIN, &legacy, 256, &scorer), 0.0);
}

fn options_for_legacy() -> TrainOptions {
    options(TreeLearning::Legacy)
}

#[test]
fn cost_orders_total_but_wp_preference_compares_raw_entropy() {
    let mut best = Some(split(20.0, 30.0, false));
    keep_better(&mut best, split(30.0, 10.0, false));
    assert_eq!(best.unwrap().bits, 30.0);
    // Equal total cost keeps the earlier candidate.
    keep_better(&mut best, split(35.0, 5.0, false));
    assert_eq!(best.unwrap().bits, 30.0);
    let mut candidates = [None; 6];
    candidates[2] = Some(split(20.0, 50.0, false));
    candidates[3] = Some(split(21.0, 1.0, false));
    let options = options(TreeLearning::Cost);
    assert_eq!(
        choose_split(candidates, 100.0, &options).unwrap().bits,
        20.0
    );
    let options = TrainOptions {
        fast_decode_multiplier: 1.1,
        ..options
    };
    assert_eq!(
        choose_split(candidates, 100.0, &options).unwrap().bits,
        21.0
    );
}

#[test]
fn static_constant_priority_and_legacy_strict_threshold_survive() {
    let mut candidates = [None; 6];
    candidates[0] = Some(split(80.0, 1.0, false));
    candidates[2] = Some(split(5.0, 1.0, false));
    assert_eq!(
        choose_split(candidates, 100.0, &options(TreeLearning::Cost))
            .unwrap()
            .bits,
        80.0
    );
    candidates[0] = None;
    candidates[2] = Some(split(90.0, 0.0, false));
    assert!(choose_split(candidates, 100.0, &options(TreeLearning::Legacy)).is_none());
    candidates[2] = Some(split(89.0, 0.0, false));
    assert!(choose_split(candidates, 100.0, &options(TreeLearning::Legacy)).is_some());
}

#[test]
fn two_sided_gain_requires_one_point_four_and_cost_mode_ignores_it() {
    let mut candidates = [None; 6];
    candidates[3] = Some(split(60.0, 10.0, false)); // gain 30
    candidates[5] = Some(split(40.0, 19.0, true)); // gain 41
    let options = options(TreeLearning::Symmetric);
    assert!(
        choose_split(candidates, 100.0, &options)
            .unwrap()
            .symmetric
            .is_none()
    );
    candidates[5] = Some(split(40.0, 18.0, true)); // gain 42: equality accepts
    assert!(
        choose_split(candidates, 100.0, &options)
            .unwrap()
            .symmetric
            .is_some()
    );
    let options = TrainOptions {
        tree_learning: TreeLearning::Cost,
        ..options
    };
    assert!(
        choose_split(candidates, 100.0, &options)
            .unwrap()
            .symmetric
            .is_none()
    );
    candidates[3] = None;
    let options = TrainOptions {
        tree_learning: TreeLearning::Symmetric,
        ..options
    };
    assert!(
        choose_split(candidates, 100.0, &options)
            .unwrap()
            .symmetric
            .is_some()
    );
}

#[test]
fn two_sided_wp_preference_uses_raw_entropy() {
    let mut candidates = [None; 6];
    candidates[4] = Some(split(30.0, 20.0, true));
    candidates[5] = Some(split(32.0, 20.0, true));
    let options = options(TreeLearning::Symmetric);
    assert_eq!(
        choose_split(candidates, 100.0, &options).unwrap().bits,
        30.0
    );
    let options = TrainOptions {
        fast_decode_multiplier: 1.1,
        ..options
    };
    assert_eq!(
        choose_split(candidates, 100.0, &options).unwrap().bits,
        32.0
    );
}

#[test]
fn gap_boundaries_and_negative_tie_are_not_replaced_by_occupied_bins() {
    let mut scratch = SymmetricScratch::default();
    scratch.prepare(
        &options(TreeLearning::Symmetric),
        0,
        &[-3, -1, 0, 1, 2],
        &[64, 0, 64, 0, 0, 64],
        &[64, 0, 640, 0, 0, 64],
        3,
    );
    assert_eq!(scratch.candidates.len(), 2);
    let first = &scratch.candidates[0];
    assert_eq!((first.low_bin, first.high_bin), (1, 2));
    assert_eq!((first.effective_low, first.effective_high), (0, 2));
    let second = &scratch.candidates[1];
    assert_eq!((second.low_bin, second.high_bin), (0, 3));
    assert_eq!((second.effective_low, second.effective_high), (0, 2));
    assert_eq!(
        (second.low_count, second.below_count, second.pixels_mid),
        (64, 128, 640)
    );
}

#[test]
fn minimum_counts_are_records_and_global_alphabet_controls_the_floor() {
    let mut scratch = SymmetricScratch::default();
    let options = options(TreeLearning::Symmetric);
    scratch.prepare(
        &options,
        0,
        &[-1, 0],
        &[63, 64, 64],
        &[63000, 64000, 64000],
        3,
    );
    assert!(!scratch.has_candidates());
    scratch.prepare(&options, 0, &[-1, 0], &[64, 64, 64], &[64, 64, 64], 32);
    assert!(scratch.has_candidates());
    scratch.prepare(&options, 0, &[-1, 0], &[64, 64, 64], &[64, 64, 64], 33);
    assert!(!scratch.has_candidates());
    scratch.prepare(&options, 0, &[-10, 0], &[64, 64, 64], &[64, 64, 64], 3);
    assert!(!scratch.has_candidates());
}

#[test]
fn only_source_symmetric_property_ids_are_eligible() {
    for property in 0..36 {
        let mut options = options(TreeLearning::Symmetric);
        options.properties[0] = property;
        let mut scratch = SymmetricScratch::default();
        scratch.prepare(&options, 0, &[-1, 0], &[64, 64, 64], &[64, 64, 64], 3);
        assert_eq!(
            scratch.has_candidates(),
            (10..=14).contains(&property) || [19, 23, 27, 31, 35].contains(&property),
            "property {property}"
        );
    }
}

#[test]
fn density_gate_uses_weighted_pixels_and_accepts_equality() {
    let options = options(TreeLearning::Symmetric);
    let mut scratch = SymmetricScratch::default();
    scratch.prepare(&options, 0, &[-1, 0], &[64, 64, 64], &[64, 640, 64], 3);
    scratch.candidates[0].middle = Cost {
        bits: 640.0,
        penalty: 900.0,
        predictor: 1,
    };
    let costs = vec![
        Cost {
            bits: 0.0,
            penalty: 0.0,
            predictor: 0
        };
        3
    ];
    let mut result = [None; 6];
    scratch.finish(
        &options,
        0,
        &[-1, 0],
        3,
        &costs,
        &costs,
        768.0,
        768,
        &mut result,
    );
    assert_eq!(result[5].unwrap().bits, 640.0); // Penalties do not leak into raw entropy.
    scratch.candidates[0].middle.bits = 641.0;
    let mut result = [None; 6];
    scratch.finish(
        &options,
        0,
        &[-1, 0],
        3,
        &costs,
        &costs,
        768.0,
        768,
        &mut result,
    );
    assert!(result.iter().all(Option::is_none));
}

#[test]
fn middle_winner_is_scored_with_dominance_enabled_and_cache_matches() {
    let samples = fixture();
    let indices: Vec<_> = (0..192).collect();
    let parents = parent_histograms(&samples, &indices);
    let mut options = options(TreeLearning::Symmetric);
    options.split_dominance = true;
    let cached = PropertyHistogram::collect(&samples, &indices, &parents, 0);
    let mut scratch = PropertyScratch::default();
    for cache in [None, Some(&cached)] {
        let candidates = evaluate_property_cached(
            &samples,
            &indices,
            &options,
            &parents,
            0,
            192,
            0,
            cache,
            None,
            &mut scratch,
        )
        .unwrap();
        let winner = candidates[5].unwrap();
        assert_eq!(scratch.dominated_predictors, 0);
        assert_eq!(winner.bits, 0.0);
        assert_eq!((winner.bin, winner.below_count), (1, 128));
        assert_eq!((winner.below_predictor, winner.above_predictor), (0, 2));
        let symmetric = winner.symmetric.unwrap();
        assert_eq!(
            (
                symmetric.low_bin,
                symmetric.low_count,
                symmetric.middle_predictor
            ),
            (0, 64, 1)
        );
        let base = DefaultScorer::default().estimate_bits(&parents[0].counts, 192);
        assert!(
            choose_split(candidates, base, &options)
                .unwrap()
                .symmetric
                .is_some()
        );
    }
    options.predictors[1] = super::super::Predictor::Weighted;
    let candidates = evaluate_property_cached(
        &samples,
        &indices,
        &options,
        &parents,
        0,
        192,
        0,
        None,
        None,
        &mut scratch,
    )
    .unwrap();
    assert!(candidates[4].is_some());
    assert!(candidates[5].is_none());
}

#[test]
fn global_alphabet_is_not_replaced_by_node_trimmed_histograms() {
    let mut samples = fixture();
    samples.global_max_symbols = 32;
    let indices: Vec<_> = (0..192).collect();
    let parents = parent_histograms(&samples, &indices);
    assert!(parents.iter().all(|parent| parent.counts.len() == 3));
    let options = options(TreeLearning::Symmetric);
    let result = evaluate_property_cached(
        &samples,
        &indices,
        &options,
        &parents,
        0,
        192,
        0,
        None,
        None,
        &mut PropertyScratch::default(),
    )
    .unwrap();
    let expected = split_cost(0, threshold(&options), &DefaultScorer::default())
        + split_cost(-1, threshold(&options), &DefaultScorer::default())
        + 10.0 * 31.0;
    assert_eq!(result[5].unwrap().split_cost, expected);
}

#[test]
fn packed_middle_rows_and_weighted_extra_bits_are_reused() {
    let mut samples = fixture();
    // Occupied bins 0, 10, 20, with the original near-zero syntax at bins 9/10.
    let PropertyColumn::Packed(values) = &mut samples.properties[0] else {
        unreachable!()
    };
    for value in values {
        *value *= 10;
    }
    samples.thresholds[0] = (-10..10).collect();
    for counts in &mut samples.counts {
        *counts = 2;
    }
    for residual in &mut samples.residuals[1][64..128] {
        residual.nbits = 3;
    }
    let indices: Vec<_> = (0..192).collect();
    let parents = parent_histograms(&samples, &indices);
    let mut options = options(TreeLearning::Symmetric);
    options.property_bins = PropertyBins::All;
    let result = evaluate_property_cached(
        &samples,
        &indices,
        &options,
        &parents,
        0,
        384,
        0,
        None,
        None,
        &mut PropertyScratch::default(),
    )
    .unwrap();
    // Gradient's three extra bits/pixel lose to a different middle predictor's
    // one bit/pixel, despite Gradient having a constant token histogram.
    let winner = result[5].unwrap();
    assert_ne!(winner.symmetric.unwrap().middle_predictor, 1);
    assert!(winner.bits >= 128.0 - 0.001);
}

#[test]
fn partition_dedup_does_not_hide_threshold_cost_or_symmetry() {
    let mut samples = fixture();
    samples.properties.push(samples.properties[0].clone());
    samples.thresholds.push(vec![-100, 100]);
    let mut options = options(TreeLearning::Cost);
    options.properties.push(11);
    let indices: Vec<_> = (0..192).collect();
    let mut seen = Vec::new();
    let mut remap = Vec::new();
    let PropertyColumn::Packed(values) = &samples.properties[0] else {
        unreachable!()
    };
    assert!(!duplicate_property(
        &samples, &indices, &options, 0, values, &[64; 3], &mut seen, &mut remap
    ));
    let PropertyColumn::Packed(values) = &samples.properties[1] else {
        unreachable!()
    };
    assert!(!duplicate_property(
        &samples, &indices, &options, 1, values, &[64; 3], &mut seen, &mut remap
    ));
}

#[test]
fn early_bound_allows_gain_between_scaled_and_legacy_thresholds() {
    let mut samples = fixture();
    samples.properties = vec![PropertyColumn::Packed(vec![0, 1])];
    samples.residuals = vec![vec![
        ResidualToken { token: 0, nbits: 0 },
        ResidualToken { token: 1, nbits: 0 },
    ]];
    samples.counts = vec![4, 4];
    samples.thresholds = vec![vec![0]];
    samples.global_max_symbols = 2;
    for mode in [TreeLearning::Legacy, TreeLearning::Cost] {
        let mut options = options(mode);
        options.predictors.truncate(1);
        let result = find_best_split_with_scratch(
            &samples,
            &[0, 1],
            0,
            &options,
            &mut None,
            &mut None,
            &HistogramBudget::new(0),
            &TrainingStats::default(),
            &ScratchPool::new(1),
            &[],
        )
        .unwrap();
        assert_eq!(result.is_some(), mode == TreeLearning::Cost);
    }
}

#[test]
fn experimental_options_are_rejected_without_altering_legacy_controls() {
    let mut options = options(TreeLearning::Cost);
    options.search.histogram_bins = 16;
    assert!(validate_options(&options).is_err());
    options.tree_learning = TreeLearning::Legacy;
    assert!(validate_options(&options).is_ok());
    options.tree_learning = TreeLearning::Symmetric;
    options.search = SearchOptions::default();
    options.scoring.fast_entropy = true;
    assert!(validate_options(&options).is_err());
}
