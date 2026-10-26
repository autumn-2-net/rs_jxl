use super::*;

fn fixture(channels: usize, records: usize) -> (TreeSamples, TrainOptions) {
    let mut properties = vec![PropertyColumn::Wide(
        (0..records).map(|i| (i % channels) as u32).collect(),
    )];
    for bit in 0..5 {
        properties.push(PropertyColumn::Packed(
            (0..records)
                .map(|i| (((i / channels) >> bit) & 1) as u8)
                .collect(),
        ));
    }
    let residuals: Vec<_> = (0..records)
        .map(|i| {
            let token = ((i / channels) % 32) as u8;
            ResidualToken {
                token,
                nbits: if token < 16 { 0 } else { 1 + (token - 16) / 8 },
            }
        })
        .collect();
    let mut thresholds = vec![(0..channels.saturating_sub(1) as i32).collect()];
    thresholds.extend((0..5).map(|_| vec![0]));
    (
        TreeSamples {
            properties,
            residuals: vec![residuals.clone(), residuals],
            counts: vec![1; records],
            thresholds,
        },
        TrainOptions {
            predictors: vec![Predictor::Zero, Predictor::Left],
            properties: vec![0, 6, 7, 8, 9, 10],
            static_property_range: Some([[0, channels as u32], [0, 1]]),
            split_threshold: 1.0,
            parallel_tools: ParallelOptions {
                subtrees: true,
                small_subtrees: true,
                balanced: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
}

#[test]
fn ma_budget_mode_defaults_and_explicit_overrides() {
    let mut options = TrainOptions::default();
    assert_eq!(options.budget_mode, None);
    assert!(!options.complete_search());
    options.budget_mode = Some(MaBudgetMode::Strict);
    assert!(options.complete_search());
    options.require_complete_search = true;
    options.budget_mode = Some(MaBudgetMode::Bounded);
    assert!(!options.complete_search());
    options.budget_mode = None;
    assert!(options.complete_search());
}

#[test]
fn ma_budget_mode_tiny_caps_cover_serial_ready_children_and_detached_subtrees() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (samples, mut options) = fixture(1, 16_384);
    let mut detached = 0;
    for parallel in [false, true] {
        options.parallel = parallel;
        for cap in [1, 2, 5, 31, 32] {
            // The merged allowance, not the legacy preset, governs descendants.
            options.max_leaves = 1;
            options.max_merged_leaves = Some(cap);
            options.budget_mode = Some(MaBudgetMode::Bounded);
            options.require_complete_search = true;
            let stats = TrainingStats::default();
            let tree = Tree::new(
                pool.install(|| train_nodes_with_stats(&mut samples.clone(), &options, &stats))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(tree.leaves(), cap);
            detached += stats.detached_subtrees.load(Ordering::Relaxed);
            for value in 0..32 {
                let mut properties = [0; 11];
                for bit in 0..5 {
                    properties[6 + bit] = (value >> bit) & 1;
                }
                assert_eq!(tree.lookup(&properties).unwrap().3, 1);
            }
            options.budget_mode = Some(MaBudgetMode::Strict);
            options.require_complete_search = false;
            let strict = pool.install(|| {
                train_nodes_with_stats(&mut samples.clone(), &options, &TrainingStats::default())
            });
            if cap < 32 {
                assert!(matches!(strict, Err(Error::Limit(_))));
            } else {
                assert_eq!(Tree::new(strict.unwrap()).unwrap().leaves(), 32);
            }
        }
    }
    assert!(detached > 0, "bounded detached-subtree path must execute");
}

#[test]
fn ma_budget_mode_channel_roots_reserve_siblings_and_never_skip_required_splits() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (samples, mut options) = fixture(2, 32_768);
    options.channel_tree = ChannelTreeMode::Channels;
    for parallel in [false, true] {
        options.parallel = parallel;
        for mode in [MaBudgetMode::Strict, MaBudgetMode::Bounded] {
            options.budget_mode = Some(mode);
            options.max_merged_leaves = Some(1);
            assert!(matches!(
                pool.install(|| {
                    train_nodes_with_stats(
                        &mut samples.clone(),
                        &options,
                        &TrainingStats::default(),
                    )
                }),
                Err(Error::Limit(
                    "required MA channel partitions exceed leaf budget"
                ))
            ));
        }
        for cap in [2, 3, 7, 63, 64] {
            options.max_merged_leaves = Some(cap);
            options.budget_mode = Some(MaBudgetMode::Bounded);
            let tree = Tree::new(
                pool.install(|| {
                    train_nodes_with_stats(
                        &mut samples.clone(),
                        &options,
                        &TrainingStats::default(),
                    )
                })
                .unwrap(),
            )
            .unwrap();
            assert_eq!(tree.leaves(), cap);
            assert!(matches!(tree.nodes()[0], Node::Split { property: 0, .. }));
            options.budget_mode = Some(MaBudgetMode::Strict);
            let strict = pool.install(|| {
                train_nodes_with_stats(&mut samples.clone(), &options, &TrainingStats::default())
            });
            assert_eq!(strict.is_ok(), cap == 64);
        }
    }
}

#[test]
fn ma_budget_mode_keeps_multiplier_and_constant_constraints_even_without_samples() {
    let mut options = TrainOptions {
        predictors: vec![Predictor::Zero],
        properties: vec![0, 6],
        static_property_range: Some([[0, 3], [0, 1]]),
        constant_channels: vec![ConstantChannel {
            channel: 2,
            value: 9,
        }],
        multiplier_info: vec![
            MultiplierInfo {
                range: [[0, 1], [0, 1]],
                multiplier: 2,
            },
            MultiplierInfo {
                range: [[1, 2], [0, 1]],
                multiplier: 4,
            },
        ],
        split_threshold: 0.0,
        ..Default::default()
    };
    for mode in [MaBudgetMode::Strict, MaBudgetMode::Bounded] {
        options.budget_mode = Some(mode);
        options.max_leaves = 2;
        assert!(matches!(train(&[], &options), Err(Error::Limit(_))));
        options.max_leaves = 3;
        let tree = train(&[], &options).unwrap();
        assert_eq!(tree.leaves(), 3);
        for (channel, offset, multiplier) in [(0, 0, 2), (1, 0, 4), (2, 9, 1)] {
            let (_, predictor, actual_offset, actual_multiplier) =
                tree.lookup(&[channel, 0]).unwrap();
            assert_eq!(
                (predictor, actual_offset, actual_multiplier),
                (Predictor::Zero, offset, multiplier)
            );
        }
    }
    let samples: Vec<_> = (0..4)
        .map(|i| {
            let mut properties = vec![0; 7];
            properties[0] = i / 2;
            properties[6] = i % 2;
            Sample {
                properties,
                residuals: vec![(i % 2) * 64],
                weight: 1024,
            }
        })
        .collect();
    options.budget_mode = Some(MaBudgetMode::Strict);
    assert!(matches!(train(&samples, &options), Err(Error::Limit(_))));
    options.budget_mode = Some(MaBudgetMode::Bounded);
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 3);
    assert_eq!(tree.lookup(&[0, 0]).unwrap().3, 2);
    assert_eq!(tree.lookup(&[1, 0]).unwrap().3, 4);
}

#[test]
fn ma_budget_mode_full_merge_preserves_detached_seed_predictor_offset_and_multiplier() {
    let (mut samples, mut options) = fixture(1, 128);
    options.max_leaves = 2;
    options.budget_mode = Some(MaBudgetMode::Bounded);
    let seed = Node::Leaf {
        predictor: Predictor::Left,
        offset: 9,
        multiplier: 7,
    };
    let mut nodes = vec![
        Node::Split {
            property: 6,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        seed.clone(),
        seed.clone(),
    ];
    let jobs = (0..2)
        .map(|i| ma_parallel::Detached {
            destination: i + 1,
            records: i * 64..(i + 1) * 64,
            range: [[0, 1], [0, 1]],
            redundant: Vec::new(),
            seed: seed.clone(),
            predictor: 1,
            multiplier: 7,
        })
        .collect();
    ma_parallel::finish(
        &mut samples,
        &options,
        &TrainingStats::default(),
        &mut nodes,
        jobs,
        4,
    )
    .unwrap();
    let tree = Tree::new(nodes).unwrap();
    assert_eq!(tree.leaves(), 2);
    for value in [0, 1] {
        let (_, predictor, offset, multiplier) = tree.lookup(&[0, 0, 0, 0, 0, 0, value]).unwrap();
        assert_eq!((predictor, offset, multiplier), (Predictor::Left, 9, 7));
    }
}
