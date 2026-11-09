// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; notice in docs/PORTING_KERNELS.md.

//! Learner integration regressions for PR5007, not an independent scoring port.

use super::split_scoring::SymmetricSplit;
use super::*;

fn fixture(sizes: [usize; 3], scale: i32) -> (Vec<Sample>, TrainOptions) {
    let largest = *sizes.iter().max().unwrap();
    let options = TrainOptions {
        predictors: vec![Predictor::Zero, Predictor::Left, Predictor::Top],
        properties: vec![10, 2, 3],
        property_quantization: Some(PropertyQuantization {
            properties: vec![10, 2, 3],
            // Empty bins retain the original -1/0 syntax boundaries.
            thresholds: vec![
                vec![-2, -1, 0, 1],
                (0..127).collect(),
                (0..(largest.saturating_sub(1) / 128) as i32).collect(),
            ],
        }),
        tree_learning: TreeLearning::Symmetric,
        split_threshold: 1.0,
        max_leaves: 3,
        parallel: false,
        budget_mode: Some(MaBudgetMode::Strict),
        ..Default::default()
    };
    let mut samples = Vec::new();
    for row in 0..largest {
        for (region, &size) in sizes.iter().enumerate() {
            if row >= size {
                continue;
            }
            let mut properties = vec![0; 16];
            properties[10] = [-3, 0, 3][region];
            properties[2] = (row % 128) as i32;
            properties[3] = (row / 128) as i32;
            // A fixed nonzero wrong residual would also have zero entropy.
            let mut residuals = vec![scale * (1 + (row % 2) as i32); 3];
            residuals[region] = 0;
            samples.push(Sample {
                properties,
                residuals,
                weight: 32 + (row % 3) as u32,
            });
        }
    }
    (samples, options)
}

fn signature(tree: &Tree) -> Vec<(usize, u32)> {
    tree.tokens().iter().map(|t| (t.context, t.value)).collect()
}

fn assert_regions(tree: &Tree, channel: i32, multiplier: u32) {
    for (value, predictor) in [
        (-4, Predictor::Zero),
        (-3, Predictor::Zero),
        (0, Predictor::Left),
        (3, Predictor::Top),
        (4, Predictor::Top),
    ] {
        let mut properties = vec![0; 16];
        properties[0] = channel;
        properties[10] = value;
        let (_, actual, offset, actual_multiplier) = tree.lookup(&properties).unwrap();
        assert_eq!(
            (actual, offset, actual_multiplier),
            (predictor, 0, multiplier)
        );
    }
}

fn assert_interval_boundaries(tree: &Tree, channel: i32) {
    for (value, expected) in [
        (-1, Predictor::Zero),
        (0, Predictor::Left),
        (1, Predictor::Top),
    ] {
        let mut properties = vec![0; 16];
        properties[0] = channel;
        properties[10] = value;
        assert_eq!(tree.lookup(&properties).unwrap().1, expected);
    }
}

#[test]
fn symmetric_direct_train_commits_real_interval_and_cost_mode_stays_binary() {
    let (samples, mut options) = fixture([64; 3], 1);
    let prepared = TreeSamples::new(&samples, &options).unwrap();
    assert_eq!(prepared.counts.len(), 192);
    assert_eq!(prepared.global_max_symbols, 5);
    assert!(64 >= 64.max(2 * prepared.global_max_symbols));
    for mode in [TreeLearning::Cost, TreeLearning::Symmetric] {
        options.tree_learning = mode;
        let receipt = Arc::new(TreeLearningReceipt::default());
        options.tree_learning_receipt = Some(Arc::clone(&receipt));
        let tree = train(&samples, &options).unwrap();
        assert_eq!(tree.leaves(), 3);
        assert_regions(&tree, 0, 1);
        let evidence = receipt.snapshot();
        assert_eq!(evidence.learners, 1);
        assert_eq!(evidence.records, 192);
        assert_eq!(
            evidence.symmetric_splits,
            usize::from(mode == TreeLearning::Symmetric)
        );
        assert_eq!(
            evidence.cost_splits,
            if mode == TreeLearning::Cost { 2 } else { 0 }
        );
        if mode == TreeLearning::Symmetric {
            assert_interval_boundaries(&tree, 0);
            assert!(matches!(
                tree.nodes()[0],
                Node::Split {
                    property: 10,
                    threshold: 0,
                    ..
                }
            ));
        }
    }
}

#[test]
fn symmetric_distinct_gate_is_not_replaced_by_sample_weight() {
    for distinct in [false, true] {
        let (mut samples, mut options) = fixture([64; 3], 1);
        if !distinct {
            for sample in &mut samples {
                sample.properties[2] = 0;
                sample.properties[3] = 0;
            }
        }
        let prepared = TreeSamples::new(&samples, &options).unwrap();
        assert_eq!(prepared.counts.len(), if distinct { 192 } else { 6 });
        assert!(prepared.counts.iter().map(|&n| u64::from(n)).sum::<u64>() > 6_000);
        let receipt = Arc::new(TreeLearningReceipt::default());
        options.tree_learning_receipt = Some(Arc::clone(&receipt));
        let tree = train(&samples, &options).unwrap();
        assert_regions(&tree, 0, 1);
        assert_eq!(receipt.snapshot().symmetric_splits, usize::from(distinct));
    }
}

#[test]
fn symmetric_distinct_gate_scales_with_the_original_token_alphabet() {
    for records in [64, 512] {
        let (samples, mut options) = fixture([records; 3], 1 << 20);
        let prepared = TreeSamples::new(&samples, &options).unwrap();
        assert_eq!(prepared.counts.len(), 3 * records);
        assert!(prepared.global_max_symbols > 32);
        let eligible = records >= 64.max(2 * prepared.global_max_symbols);
        assert_eq!(eligible, records == 512);
        let receipt = Arc::new(TreeLearningReceipt::default());
        options.tree_learning_receipt = Some(Arc::clone(&receipt));
        let tree = train(&samples, &options).unwrap();
        assert_regions(&tree, 0, 1);
        assert_eq!(receipt.snapshot().symmetric_splits, usize::from(eligible));
    }
}

#[test]
fn symmetric_two_leaf_growth_is_atomic_in_strict_and_bounded_modes() {
    let (samples, mut options) = fixture([64; 3], 1);
    for mode in [MaBudgetMode::Strict, MaBudgetMode::Bounded] {
        options.budget_mode = Some(mode);
        for cap in 1..=3 {
            options.max_leaves = cap;
            let receipt = Arc::new(TreeLearningReceipt::default());
            options.tree_learning_receipt = Some(Arc::clone(&receipt));
            let result = train(&samples, &options);
            if mode == MaBudgetMode::Strict && cap < 3 {
                assert!(matches!(result, Err(Error::Limit(_))));
            } else {
                let tree = result.unwrap();
                assert_eq!(tree.leaves(), if cap < 3 { 1 } else { 3 });
                if cap == 3 {
                    assert_regions(&tree, 0, 1);
                    assert_interval_boundaries(&tree, 0);
                }
            }
            assert_eq!(receipt.snapshot().symmetric_splits, usize::from(cap == 3));
        }
    }
}

#[test]
fn symmetric_preserves_multipliers_unsampled_ranges_and_constant_suffix() {
    let (samples, mut options) = fixture([64; 3], 2);
    options.properties.insert(0, 0);
    let quantization = options.property_quantization.as_mut().unwrap();
    quantization.properties.insert(0, 0);
    quantization.thresholds.insert(0, vec![0, 1]);
    options.static_property_range = Some([[0, 3], [0, 1]]);
    options.multiplier_info = vec![
        MultiplierInfo {
            range: [[0, 1], [0, 1]],
            multiplier: 2,
        },
        MultiplierInfo {
            range: [[1, 2], [0, 1]],
            multiplier: 4,
        },
    ];
    options.constant_channels = vec![ConstantChannel {
        channel: 2,
        value: 9,
    }];
    for mode in [MaBudgetMode::Strict, MaBudgetMode::Bounded] {
        options.budget_mode = Some(mode);
        for cap in 2..=5 {
            options.max_leaves = cap;
            let receipt = Arc::new(TreeLearningReceipt::default());
            options.tree_learning_receipt = Some(Arc::clone(&receipt));
            let result = train(&samples, &options);
            if cap < 3 || (mode == MaBudgetMode::Strict && cap < 5) {
                assert!(matches!(result, Err(Error::Limit(_))));
                continue;
            }
            let tree = result.unwrap();
            assert_eq!(tree.leaves(), if cap < 5 { 3 } else { 5 });
            assert_eq!(receipt.snapshot().symmetric_splits, usize::from(cap == 5));
            if cap == 5 {
                assert_regions(&tree, 0, 2);
                assert_interval_boundaries(&tree, 0);
            }
            for value in [-511, -1, 0, 1, 511] {
                let mut properties = vec![0; 16];
                properties[10] = value;
                properties[0] = 1;
                let (_, _, offset, multiplier) = tree.lookup(&properties).unwrap();
                assert_eq!((offset, multiplier), (0, 4));
                properties[0] = 2;
                let (_, predictor, offset, multiplier) = tree.lookup(&properties).unwrap();
                assert_eq!((predictor, offset, multiplier), (Predictor::Zero, 9, 1));
            }
        }
    }
}

fn learn_layout<const REORDER: bool>(
    samples: &TreeSamples,
    options: &TrainOptions,
) -> (Tree, TrainingStats) {
    let stats = TrainingStats::default();
    let nodes = train_nodes_with_layout::<REORDER>(
        &mut samples.clone(),
        options,
        &stats,
        &ScratchPool::new(6),
    )
    .unwrap();
    (Tree::new(nodes).unwrap(), stats)
}

#[test]
fn symmetric_and_cost_layouts_match_serial_parallel_and_cached_searches() {
    let (input, mut options) = fixture([4096; 3], 1);
    let samples = TreeSamples::new(&input, &options).unwrap();
    assert_eq!(samples.counts.len(), 12_288);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    options.parallel_tools.inherit_histograms = true;
    for mode in [TreeLearning::Cost, TreeLearning::Symmetric] {
        options.tree_learning = mode;
        options.histogram_cache_bytes = 0;
        options.parallel = false;
        let expected = signature(&learn_layout::<false>(&samples, &options).0);
        for parallel in [false, true] {
            for bytes in [0, 32 << 20] {
                options.parallel = parallel;
                options.histogram_cache_bytes = bytes;
                pool.install(|| {
                    for (tree, stats) in [
                        learn_layout::<false>(&samples, &options),
                        learn_layout::<true>(&samples, &options),
                    ] {
                        assert_eq!(signature(&tree), expected);
                        assert_regions(&tree, 0, 1);
                        if mode == TreeLearning::Symmetric {
                            assert_interval_boundaries(&tree, 0);
                        }
                        assert_eq!(stats.cache_builds.load(Ordering::Relaxed) > 0, bytes != 0);
                        if bytes != 0 && mode == TreeLearning::Cost {
                            assert!(stats.cache_reuses.load(Ordering::Relaxed) > 0);
                        }
                    }
                });
            }
        }
    }
}

fn assert_cache_matches(cache: &HistogramCache, samples: &TreeSamples, indices: &[usize]) {
    let parents = parent_histograms(samples, indices);
    for (property, actual) in cache.properties.iter().enumerate() {
        let expected = PropertyHistogram::collect(samples, indices, &parents, property);
        assert_eq!(actual.valid, expected.valid);
        assert_eq!(actual.used, expected.used);
        assert_eq!(actual.weights, expected.weights);
        if actual.predictors.is_empty() {
            assert!(actual.used.iter().filter(|&&n| n != 0).count() <= 1);
            continue;
        }
        assert_eq!(actual.predictors.len(), samples.residuals.len());
        // Inherited tables retain ancestor strides, even for constant children.
        for (predictor, histogram) in actual.predictors.iter().enumerate() {
            let mut counts = vec![0u32; actual.used.len() * histogram.symbols];
            let mut extra = vec![0u64; actual.used.len()];
            for &i in indices {
                let bin = samples.properties[property].value(i) as usize;
                let residual = samples.residuals[predictor][i];
                let weight = u32::from(samples.counts[i]);
                counts[bin * histogram.symbols + usize::from(residual.token)] += weight;
                extra[bin] += u64::from(weight) * u64::from(residual.nbits);
            }
            assert_eq!(histogram.counts, counts);
            assert_eq!(histogram.extra, extra);
        }
    }
}

fn commit_with_history<const REORDER: bool>(largest: usize) {
    let mut sizes = [66; 3];
    sizes[largest] = 4098;
    let (input, options) = fixture(sizes, 17);
    let mut samples = TreeSamples::new(&input, &options).unwrap();
    assert_eq!(samples.counts.len(), input.len());
    let original = samples.clone();
    let mut indices: Vec<_> = (0..samples.counts.len()).collect();
    let begin = 1;
    let end = indices.len() - 1;
    let low_count = samples.properties[0].count_le(&indices[begin..end], 1);
    let below_count = samples.properties[0].count_le(&indices[begin..end], 2);
    let parents = parent_histograms(&samples, &indices[begin..end]);
    let budget = HistogramBudget::new(32 << 20);
    let mut state = PendingNode::new(TrainingNode {
        node: 0,
        begin,
        end,
        predictor: 0,
        multiplier: 7,
    });
    state.range = [[8, 10], [12, 15]];
    state.redundant_properties = vec![0, 0, 1];
    state.cache = HistogramCache::create(&samples, &indices[begin..end], &parents, &budget, false);
    assert!(state.cache.is_some());
    state.parents = Some(parents);
    let stats = TrainingStats::default();
    let mut nodes = vec![learner::scaled_leaf(Predictor::Zero, 7)];
    let mut children = symmetric::commit::<REORDER>(
        &mut samples,
        &options,
        &stats,
        &mut indices,
        &mut nodes,
        state,
        Split {
            property: 0,
            bin: 2,
            below_count,
            below_predictor: 0,
            above_predictor: 2,
            bits: 0.0,
            split_cost: 0.0,
            symmetric: Some(SymmetricSplit {
                low_bin: 1,
                low_count,
                middle_predictor: 1,
            }),
        },
    )
    .unwrap();
    assert_eq!(nodes.len(), 5);
    let tree = Tree::new(nodes).unwrap();
    assert_regions(&tree, 0, 7);
    assert_interval_boundaries(&tree, 0);
    assert_eq!(stats.parent_reuses.load(Ordering::Relaxed), 1);
    for (region, child) in children.iter_mut().enumerate() {
        assert_eq!(child.range, [[8, 10], [12, 15]]);
        assert_eq!(child.redundant_properties, [0, 0, 1]);
        assert_eq!(child.job.multiplier, 7);
        assert_eq!(child.job.predictor, region);
        let cuts = [begin, begin + low_count, begin + below_count, end];
        assert_eq!(
            (child.job.begin, child.job.end),
            (cuts[region], cuts[region + 1])
        );
        let members = &indices[child.job.begin..child.job.end];
        assert_eq!(
            child.parents.as_ref().unwrap(),
            &parent_histograms(&samples, members)
        );
        for &i in members {
            assert_eq!(samples.properties[0].value(i), [0, 2, 4][region]);
        }
        assert_eq!(child.cache.is_some(), largest != 1 && region == largest);
        if child.cache.is_some() {
            let removed = child.removed.clone().unwrap();
            assert_eq!(
                removed,
                if largest == 0 {
                    begin + low_count..end
                } else {
                    begin..begin + below_count
                }
            );
            // The marked third property is not redundant in this synthetic fixture.
            child.redundant_properties.fill(0);
            ma_frontier::History::take(child, &samples, &indices, &options, &stats)
                .unwrap()
                .restore(child);
            assert!(child.removed.is_none());
            assert_cache_matches(child.cache.as_ref().unwrap(), &samples, members);
        }
    }
    for i in [0, end] {
        assert_eq!(samples.counts[i], original.counts[i]);
        for (actual, expected) in samples.properties.iter().zip(&original.properties) {
            assert_eq!(actual.value(i), expected.value(i));
        }
        for (actual, expected) in samples.residuals.iter().zip(&original.residuals) {
            assert_eq!(actual[i], expected[i]);
        }
    }
    assert_eq!(budget.used.load(Ordering::Relaxed) > 0, largest != 1);
    drop(children);
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
}

#[test]
fn symmetric_commit_inherits_outer_caches_but_drops_middle_cache_in_both_layouts() {
    for largest in 0..3 {
        commit_with_history::<false>(largest);
        commit_with_history::<true>(largest);
    }
}
