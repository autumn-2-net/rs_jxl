use super::*;
use crate::modular::ma_train::{Sample, train};
use crate::modular::predictor::Predictor;

fn xor_samples() -> Vec<Sample> {
    (0..4)
        .map(|i| {
            let mut properties = vec![0; 16];
            properties[6] = i & 1;
            properties[7] = (i >> 1) & 1;
            Sample {
                properties,
                residuals: vec![(i & 1) ^ ((i >> 1) & 1)],
                weight: 1000,
            }
        })
        .collect()
}

fn options(receipt: &Arc<LookaheadReceipt>) -> TrainOptions {
    TrainOptions {
        predictors: vec![Predictor::Zero],
        properties: vec![6, 7],
        parallel: false,
        split_threshold: 16.0,
        lookahead: Some(LookaheadOptions {
            width: 2,
            min_records: 4,
            receipt: Some(Arc::clone(receipt)),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn two_ply_escapes_a_zero_immediate_gain_split() {
    for learning in [TreeLearning::Legacy, TreeLearning::Cost] {
        let receipt = Arc::new(LookaheadReceipt::default());
        let mut options = options(&receipt);
        options.tree_learning = learning;
        let samples = xor_samples();
        let greedy = train(
            &samples,
            &TrainOptions {
                lookahead: None,
                ..options.clone()
            },
        )
        .unwrap();
        let tree = train(&samples, &options).unwrap();
        assert_eq!(greedy.leaves(), 1);
        assert_eq!(tree.leaves(), 4);
        let mut contexts = Vec::new();
        for sample in &samples {
            contexts.push(tree.lookup(&sample.properties).unwrap().0);
        }
        contexts.sort_unstable();
        contexts.dedup();
        assert_eq!(contexts.len(), 4);
        let result = receipt.snapshot();
        assert_eq!((result.roots, result.eligible, result.changed), (1, 1, 1));
        assert_eq!((result.probed_candidates, result.child_searches), (2, 4));
        assert!(result.estimated_saved_bits > 3000.0);
        // Training owns its columns; the input samples must remain untouched.
        assert_eq!(samples[3].properties[6..8], [1, 1]);
        assert_eq!(samples[3].residuals, [0]);
    }
}

#[test]
fn existing_leaf_allowance_and_minimum_records_gate_speculation() {
    for (cap, minimum) in [(3, 4), (64, 5)] {
        let receipt = Arc::new(LookaheadReceipt::default());
        let mut options = options(&receipt);
        options.max_leaves = cap;
        options.lookahead.as_mut().unwrap().min_records = minimum;
        assert_eq!(train(&xor_samples(), &options).unwrap().leaves(), 1);
        assert_eq!(receipt.snapshot().eligible, 0);
        assert_eq!(receipt.snapshot().child_searches, 0);
    }
}

#[test]
fn incompatible_plans_and_invalid_configuration_fail_closed() {
    for index in 0..7 {
        let receipt = Arc::new(LookaheadReceipt::default());
        let mut options = options(&receipt);
        match index {
            0 => options.lookahead.as_mut().unwrap().width = 1,
            1 => options.lookahead.as_mut().unwrap().width = 5,
            2 => options.lookahead.as_mut().unwrap().relative_window = f32::NAN,
            3 => options.lookahead.as_mut().unwrap().min_records = 0,
            4 => options.tree_learning = TreeLearning::Symmetric,
            5 => options.seed = Some(Default::default()),
            _ => options.search.histogram_bins = 8,
        }
        assert!(
            train(&xor_samples(), &options).is_err(),
            "configuration {index}"
        );
    }
}

fn proposal(property: usize, bin: usize, bits: f32) -> Split {
    Split {
        property,
        bin,
        bits,
        below_count: 2,
        below_predictor: 0,
        above_predictor: 0,
        split_cost: 0.0,
        symmetric: None,
    }
}

#[test]
fn shortlist_retains_distinct_thresholds_and_stable_ties() {
    let mut candidates = Vec::new();
    for candidate in [
        proposal(0, 4, 10.0),
        proposal(0, 5, 8.0),
        proposal(1, 0, 8.0),
        proposal(0, 5, 8.0),
        proposal(0, 7, 11.0),
    ] {
        retain_best(&mut candidates, candidate, 3);
    }
    assert_eq!(
        candidates
            .iter()
            .map(|s| (s.property, s.bin))
            .collect::<Vec<_>>(),
        [(0, 5), (1, 0), (0, 4)]
    );
}

#[test]
fn disabled_mode_does_not_leave_candidates_in_reused_scratch() {
    let receipt = Arc::new(LookaheadReceipt::default());
    let options = options(&receipt);
    let samples = super::super::TreeSamples::new(&xor_samples(), &options).unwrap();
    let pool = ScratchPool::new(1);
    let stats = TrainingStats::default();
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let budget = HistogramBudget::new(0);
    let mut candidates = Vec::new();
    super::super::split_search::find_best_split_collect(
        &samples,
        &indices,
        0,
        &options,
        &mut None,
        &mut None,
        &budget,
        &stats,
        &pool,
        &[],
        &mut Vec::new(),
        2,
        &mut candidates,
    )
    .unwrap();
    assert_eq!(candidates.len(), 2);
    let greedy = super::super::split_search::find_best_split_with_properties(
        &samples,
        &indices,
        0,
        &options,
        &mut None,
        &mut None,
        &budget,
        &stats,
        &pool,
        &[],
        &mut Vec::new(),
    )
    .unwrap();
    assert!(greedy.is_none());
    let lease = pool.acquire();
    assert_eq!(lease.scratch.lookahead_width, 0);
    assert!(lease.scratch.lookahead_candidates.is_empty());
}
