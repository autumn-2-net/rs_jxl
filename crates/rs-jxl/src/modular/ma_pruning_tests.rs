use super::*;

fn fixture(records: usize, weight: u16) -> (TreeSamples, TrainOptions) {
    let options = TrainOptions {
        predictors: vec![Predictor::Gradient, Predictor::Weighted, Predictor::Left],
        properties: vec![6, 7, 8],
        parallel: false,
        split_threshold: 1.0,
        ..Default::default()
    };
    let samples = TreeSamples {
        properties: vec![
            PropertyColumn::Packed((0..records).map(|i| (i % 8) as u8).collect()),
            PropertyColumn::Wide((0..records).map(|i| (17 * (i % 8) + 3) as u32).collect()),
            PropertyColumn::Packed(vec![0; records]),
        ],
        thresholds: vec![(0..7).collect(), (0..130).collect(), vec![0]],
        counts: vec![weight; records],
        residuals: (0..3)
            .map(|p| {
                (0..records)
                    .map(|i| ResidualToken {
                        token: if p == 0 { u8::from(i % 8 >= 4) } else { 32 },
                        nbits: [0, 4, 7][p],
                    })
                    .collect()
            })
            .collect(),
    };
    (samples, options)
}

fn decision(
    samples: &TreeSamples,
    indices: &[usize],
    options: &TrainOptions,
    stats: &TrainingStats,
    mask: &mut Vec<u8>,
) -> Option<Split> {
    find_best_split_with_properties(
        samples,
        indices,
        0,
        options,
        &mut None,
        &mut None,
        &HistogramBudget::new(0),
        stats,
        &ScratchPool::new(3),
        &[],
        mask,
    )
    .unwrap()
}

fn same_decision(a: Option<Split>, b: Option<Split>) {
    match (a, b) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(
                (
                    a.property,
                    a.bin,
                    a.below_count,
                    a.above_predictor,
                    a.below_predictor
                ),
                (
                    b.property,
                    b.bin,
                    b.below_count,
                    b.above_predictor,
                    b.below_predictor
                )
            );
            assert_eq!(a.bits.to_bits(), b.bits.to_bits());
        }
        _ => panic!("pruning changed split existence"),
    }
}

#[test]
fn dominance_prunes_scoring_without_changing_split_and_respects_weight_gate() {
    for (records, weight, prunes) in [
        (64, 1, true),
        (128, 2, true),
        (256, 1, true),
        (257, 1, false),
        (64, 5, false),
    ] {
        let (samples, mut options) = fixture(records, weight);
        let indices: Vec<_> = (0..records).collect();
        let expected = decision(
            &samples,
            &indices,
            &options,
            &TrainingStats::default(),
            &mut Vec::new(),
        );
        assert!(expected.is_some());
        options.split_dominance = true;
        let stats = TrainingStats::default();
        let actual = decision(&samples, &indices, &options, &stats, &mut Vec::new());
        same_decision(actual, expected);
        assert_eq!(
            stats.dominated_predictors.load(Ordering::Relaxed) > 0,
            prunes
        );
        options.scoring.fast_entropy = true;
        let stats = TrainingStats::default();
        decision(&samples, &indices, &options, &stats, &mut Vec::new());
        assert_eq!(stats.dominated_predictors.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn monotone_partition_proofs_are_inherited_by_parallel_descendants() {
    let (samples, mut options) = fixture(16384, 1);
    let indices: Vec<_> = (0..16384).collect();
    let expected = decision(
        &samples,
        &indices,
        &options,
        &TrainingStats::default(),
        &mut Vec::new(),
    );
    options.property_dedup = true;
    let mut mask = Vec::new();
    let stats = TrainingStats::default();
    let actual = decision(&samples, &indices, &options, &stats, &mut mask);
    same_decision(actual, expected);
    assert_eq!(mask, [0, 1, 1]);
    assert_eq!(stats.duplicate_properties.load(Ordering::Relaxed), 1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    options.parallel = true;
    for child in [&indices[..8192], &indices[8192..]] {
        let stats = TrainingStats::default();
        let actual =
            pool.install(|| decision(&samples, child, &options, &stats, &mut mask.clone()));
        let mut reference = options.clone();
        reference.property_dedup = false;
        let expected = pool.install(|| {
            decision(
                &samples,
                child,
                &reference,
                &TrainingStats::default(),
                &mut Vec::new(),
            )
        });
        same_decision(actual, expected);
        assert_eq!(stats.inherited_properties.load(Ordering::Relaxed), 2);
    }
}

#[test]
fn modular_options_forward_pruning_without_changing_budget_or_samples() {
    let options = super::super::encode::ModularOptions {
        source_property_dedup: true,
        source_ma_dominance: true,
        max_tree_leaves: 37,
        ..Default::default()
    };
    let train = super::super::encode::train_options(&options);
    assert!(train.property_dedup && train.split_dominance);
    assert_eq!(train.max_leaves, 37);
    assert_eq!(train.budget_mode, None);
}
