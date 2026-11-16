use super::*;
use crate::modular::{
    ma_train::{DescriptionCost, LookaheadOptions, MaBudgetMode, Sample, SeedOptions, TreeSamples},
    predictor::Predictor,
    tree::Tree,
};

fn options() -> TrainOptions {
    TrainOptions {
        properties: vec![18],
        parallel: false,
        parallel_nodes: false,
        split_threshold: 0.0,
        budget_mode: Some(MaBudgetMode::Strict),
        ..Default::default()
    }
}

fn sample(value: i32, residuals: Vec<i32>) -> Sample {
    let mut properties = vec![0; 20];
    properties[18] = value;
    Sample {
        properties,
        residuals,
        weight: 100,
    }
}

#[test]
fn e9_initialization_is_cheaper_and_pilots_explicit_seeds_are_excluded() {
    let mut options = options();
    options.tree_learning = TreeLearning::Symmetric;
    options.scoring.description = DescriptionCost::Half;
    options.property_dedup = true;
    let coarse = coarse_options(&options);
    assert_eq!(coarse.tree_learning, TreeLearning::Legacy);
    assert_eq!(coarse.search.quantile_splits, QuantileSplits::Twelve);
    assert_eq!(coarse.scoring, options.scoring);
    assert_eq!(coarse.properties, options.properties);
    assert!(!coarse.property_dedup);
    assert_eq!(options.tree_learning, TreeLearning::Symmetric);
    assert!(eligible(&options, 512));
    assert!(!eligible(&options, 511));
    options.max_leaves = 4;
    assert!(!eligible(&options, 8192));
    options.max_leaves = 64;
    options.seed = Some(SeedOptions::default());
    assert!(!eligible(&options, 8192));
    options.seed = None;
    options.lookahead = Some(LookaheadOptions::default());
    assert!(!eligible(&options, 8192));
}

#[test]
fn wide_property_and_all_predictors_continue_beyond_coarse_depth_without_regather() {
    let mut options = options();
    options.predictors = (0..12).map(|p| Predictor::try_from(p).unwrap()).collect();
    let input: Vec<_> = (0..12)
        .map(|p| {
            let mut residuals = vec![400; 12];
            residuals[p] = 0;
            sample(p as i32, residuals)
        })
        .collect();
    let mut prepared = TreeSamples::new(&input, &options).unwrap();
    let count_storage = prepared.counts.as_ptr();
    let residual_storage = prepared.residuals[0].as_ptr();
    let quantizers = prepared.thresholds.clone();
    let stats = TrainingStats::default();
    let scratch = ScratchPool::new(1);
    let continuation = build(&mut prepared, &options, &stats, &scratch, COARSE_DEPTH).unwrap();
    assert!(continuation.receipt.splits > 0);
    assert!(continuation.nodes.len().div_ceil(2) <= 8);
    assert!(!continuation.pending.is_empty());
    assert!(continuation.pending.iter().all(|s| s.parents.is_some()));
    assert!(matches!(
        continuation.nodes[0],
        Node::Split { property: 18, .. }
    ));
    for state in &continuation.pending {
        let expected = parent_histograms(
            &prepared,
            &continuation.indices[state.job.begin..state.job.end],
        );
        for (actual, expected) in state.parents.as_ref().unwrap().iter().zip(expected) {
            assert_eq!(actual.extra_bits, expected.extra_bits);
            let symbols = actual.counts.len().max(expected.counts.len());
            for i in 0..symbols {
                assert_eq!(
                    actual.counts.get(i).unwrap_or(&0),
                    expected.counts.get(i).unwrap_or(&0)
                );
            }
        }
    }
    let tree = Tree::new(resume(&mut prepared, &options, &stats, &scratch, continuation).unwrap())
        .unwrap();
    assert_eq!(tree.leaves(), 12);
    for (p, sample) in input.iter().enumerate() {
        assert_eq!(
            tree.lookup(&sample.properties).unwrap().1,
            options.predictors[p]
        );
    }
    assert_eq!(prepared.counts.as_ptr(), count_storage);
    assert_eq!(prepared.residuals[0].as_ptr(), residual_storage);
    assert_eq!(prepared.thresholds, quantizers);
    options.max_leaves = 2;
    let mut prepared = TreeSamples::new(&input, &options).unwrap();
    assert!(train(&mut prepared, &options, &stats, &scratch).is_err());
    options.budget_mode = Some(MaBudgetMode::Bounded);
    let mut prepared = TreeSamples::new(&input, &options).unwrap();
    assert_eq!(
        Tree::new(train(&mut prepared, &options, &stats, &scratch).unwrap())
            .unwrap()
            .leaves(),
        2
    );
}

#[test]
fn native_saturation_is_checked_once_and_never_requeued() {
    let mut options = options();
    options.tree_learning = TreeLearning::Symmetric;
    options.scoring.description = DescriptionCost::Half;
    let input = [sample(0, vec![0, 400]), sample(0, vec![400, 0])];
    let mut prepared = TreeSamples::new(&input, &options).unwrap();
    let stats = TrainingStats {
        profile_properties: true,
        ..Default::default()
    };
    let scratch = ScratchPool::new(1);
    let continuation = build(&mut prepared, &options, &stats, &scratch, COARSE_DEPTH).unwrap();
    assert_eq!(continuation.receipt.fine_checks, 1);
    assert_eq!(continuation.receipt.saturated, 1);
    assert!(continuation.pending.is_empty());
    let evaluations: usize = stats
        .property_evals
        .iter()
        .map(|v| v.load(Ordering::Relaxed))
        .sum();
    let nodes = resume(&mut prepared, &options, &stats, &scratch, continuation).unwrap();
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        stats
            .property_evals
            .iter()
            .map(|v| v.load(Ordering::Relaxed))
            .sum::<usize>(),
        evaluations
    );
}

#[test]
fn coarse_rejection_hands_a_cached_native_decision_to_the_ordinary_learner() {
    use crate::tools::ma_simd::DefaultScorer;
    let mut options = options();
    options.properties = vec![10];
    options.tree_learning = TreeLearning::Symmetric;
    options.scoring.description = DescriptionCost::Half;
    let mut input = [sample(0, vec![0, 400]), sample(0, vec![400, 0])];
    input[1].properties[10] = 1;
    let mut prepared = TreeSamples::new(&input, &options).unwrap();
    let parents = parent_histograms(&prepared, &[0, 1]);
    let parent = &parents[0];
    let total = parent.counts.iter().sum();
    let scorer = DefaultScorer::default();
    let cost = options.scoring.leaf_bits(&scorer, &parent.counts, total)
        + parent.extra_bits as f32
        + options
            .scoring
            .description
            .cost(&scorer, &parent.counts, total);
    // Legacy rejects at the leaf baseline; native threshold is 0.72 times this.
    options.split_threshold = f64::from(cost) * 1.1;
    let stats = TrainingStats {
        profile_properties: true,
        ..Default::default()
    };
    let scratch = ScratchPool::new(1);
    let continuation = build(&mut prepared, &options, &stats, &scratch, COARSE_DEPTH).unwrap();
    assert_eq!(continuation.receipt.splits, 0);
    assert_eq!(continuation.receipt.fine_decisions, 1);
    assert_eq!(continuation.pending.len(), 1);
    assert!(continuation.pending[0].decision.flatten().is_some());
    let evaluations = stats.property_evals[10].load(Ordering::Relaxed);
    let tree = Tree::new(resume(&mut prepared, &options, &stats, &scratch, continuation).unwrap())
        .unwrap();
    assert_eq!(tree.leaves(), 2);
    assert_eq!(
        stats.property_evals[10].load(Ordering::Relaxed),
        evaluations
    );
    assert_eq!(
        tree.lookup(&input[0].properties).unwrap().1,
        options.predictors[0]
    );
    assert_eq!(
        tree.lookup(&input[1].properties).unwrap().1,
        options.predictors[1]
    );
}
