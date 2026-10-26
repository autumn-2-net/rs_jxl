// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Regression tests for the MA learner ported from enc_ma.h/cc,
//! frozen at b866b3defec13494bfbaf705db93581cda118e37.

use super::*;

fn sample(property: usize, value: i32, residuals: &[i32], weight: u32) -> Sample {
    let mut properties = vec![0; 16];
    properties[property] = value;
    Sample {
        properties,
        residuals: residuals.to_vec(),
        weight,
    }
}

fn options(property: usize) -> TrainOptions {
    TrainOptions {
        properties: vec![property],
        split_threshold: 0.0,
        parallel: false,
        ..TrainOptions::default()
    }
}

fn signature(tree: &Tree) -> Vec<(usize, u32)> {
    tree.tokens()
        .iter()
        .map(|token| (token.context, token.value))
        .collect()
}

fn estimated_cost(tree: &Tree, samples: &[Sample], options: &TrainOptions) -> f64 {
    let config = HybridUintConfig::new(4, 1, 2).unwrap();
    let mut counts = vec![vec![0u32; 256]; tree.leaves()];
    let mut extra = 0u64;
    for sample in samples {
        let (context, predictor, offset, multiplier) = tree.lookup(&sample.properties).unwrap();
        assert_eq!((offset, multiplier), (0, 1));
        let p = options
            .predictors
            .iter()
            .position(|&p| p == predictor)
            .unwrap();
        let encoded = config.encode(pack_signed(sample.residuals[p]));
        counts[context][encoded.token as usize] += sample.weight;
        extra += u64::from(encoded.nbits) * u64::from(sample.weight);
    }
    extra as f64
        + counts
            .iter()
            .map(|counts| {
                let total: f64 = counts.iter().map(|&n| f64::from(n)).sum();
                counts
                    .iter()
                    .filter(|&&n| n != 0)
                    .map(|&n| {
                        let n = f64::from(n);
                        n * (total / n).log2().min(12.0)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
}

#[test]
fn source_predictor_order_and_defaults() {
    let best = TrainOptions::default();
    assert_eq!(best.predictors, [Predictor::Weighted, Predictor::Gradient]);
    assert!(!best.properties.contains(&1));
    assert_eq!(
        best.properties,
        std::iter::once(0).chain(2..=15).collect::<Vec<_>>()
    );
    let variable = TrainOptions::variable();
    assert_eq!(
        variable
            .predictors
            .iter()
            .map(|&p| p as u32)
            .collect::<Vec<_>>(),
        [6, 5, 2, 3, 4, 1, 0, 7, 8, 9, 10, 11, 12, 13]
    );
}

#[test]
fn adaptive_partition_uses_weighted_and_gradient_and_reduces_residual_cost() {
    let options = options(6);
    let samples = vec![sample(6, -7, &[0, 200], 100), sample(6, 13, &[200, 0], 100)];
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 2);
    assert_eq!(
        tree.lookup(&samples[0].properties).unwrap().1,
        Predictor::Weighted
    );
    assert_eq!(
        tree.lookup(&samples[1].properties).unwrap().1,
        Predictor::Gradient
    );
    let mut boundary = samples[0].properties.clone();
    boundary[6] = -7;
    assert_eq!(tree.lookup(&boundary).unwrap().1, Predictor::Weighted);
    boundary[6] = -6;
    assert_eq!(tree.lookup(&boundary).unwrap().1, Predictor::Gradient);
    let seed = Tree::new(vec![leaf(Predictor::Weighted)]).unwrap();
    assert_eq!(estimated_cost(&tree, &samples, &options), 0.0);
    assert!(estimated_cost(&seed, &samples, &options) > 500.0);
}

#[test]
fn property_zero_is_static_even_without_property_one() {
    let options = options(0);
    let samples = vec![sample(0, 0, &[0, 400], 100), sample(0, 2, &[400, 0], 100)];
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 2);
    assert_eq!(tree.tokens()[0].value, 1);
    assert_eq!(tree.lookup(&[0]).unwrap().1, Predictor::Weighted);
    assert_eq!(tree.lookup(&[1]).unwrap().1, Predictor::Gradient);
}

#[test]
fn supplied_population_quantiles_control_partition_not_training_sites() {
    let mut options = options(6);
    options.property_quantization = Some(PropertyQuantization {
        properties: vec![6],
        thresholds: vec![vec![0]],
    });
    let samples = vec![
        sample(6, -100, &[0, 400], 100),
        sample(6, 100, &[400, 0], 100),
    ];
    let tree = train(&samples, &options).unwrap();
    let mut props = vec![0; 16];
    assert_eq!(tree.lookup(&props).unwrap().1, Predictor::Weighted);
    props[6] = 1;
    assert_eq!(tree.lookup(&props).unwrap().1, Predictor::Gradient);
    let compact = TreeSamples::new(&samples, &options).unwrap();
    assert_eq!(compact.thresholds, [vec![0]]);
    options.property_quantization.as_mut().unwrap().thresholds[0].clear();
    assert_eq!(train(&samples, &options).unwrap().leaves(), 1);
    options.property_quantization.as_mut().unwrap().properties[0] = 7;
    assert!(train(&samples, &options).is_err());
    options.property_quantization.as_mut().unwrap().properties[0] = 6;
    options.property_quantization.as_mut().unwrap().thresholds[0] = vec![1, 1];
    assert!(train(&samples, &options).is_err());
}

#[test]
fn raw_leaf_contexts_above_255_and_static_ids_above_511_are_supported() {
    let mut options = options(1);
    options.max_leaves = 300;
    options.property_quantization = Some(PropertyQuantization {
        properties: vec![1],
        thresholds: vec![(10_000..10_299).collect()],
    });
    let samples: Vec<_> = (0..300)
        .map(|i| {
            let residuals = if i % 2 == 0 { [0, 400] } else { [400, 0] };
            sample(1, 10_000 + i, &residuals, 100)
        })
        .collect();
    let compact = TreeSamples::new(&samples, &options).unwrap();
    assert_eq!(compact.properties[0].value(299), 299);
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 300);
    for (i, sample) in samples.iter().enumerate() {
        let expected = if i % 2 == 0 {
            Predictor::Weighted
        } else {
            Predictor::Gradient
        };
        assert_eq!(tree.lookup(&sample.properties).unwrap().1, expected);
    }
    options.max_leaves = MAX_LEAVES;
    assert!(options.validate().is_ok());
}

#[test]
fn empty_all_zero_and_zero_weight_samples_do_not_split() {
    let options = options(6);
    for samples in [
        vec![],
        vec![sample(6, 0, &[0, 0], 1), sample(6, 9, &[0, 0], 3)],
        vec![sample(6, 0, &[900, 0], 0)],
    ] {
        let tree = train(&samples, &options).unwrap();
        assert_eq!(tree.leaves(), 1);
        assert_eq!(tree.lookup(&[]).unwrap().1, Predictor::Weighted);
    }
}

#[test]
fn exact_split_threshold_and_leaf_budget() {
    let mut options = options(6);
    options.predictors = vec![Predictor::Zero];
    let samples = vec![sample(6, -1, &[0], 50), sample(6, 1, &[1], 50)];
    // Source FastLog2f is approximate even at p=1/2. Test equality with
    // the active estimator, not a mathematically exact log2 cost.
    options.split_threshold = f64::from(DefaultScorer::default().estimate_bits(&[50, 0, 50], 100));
    assert!((options.split_threshold - 100.0).abs() < 0.001);
    assert_eq!(train(&samples, &options).unwrap().leaves(), 1);
    options.split_threshold = 99.0;
    assert_eq!(train(&samples, &options).unwrap().leaves(), 2);
    options.max_leaves = 1;
    assert_eq!(train(&samples, &options).unwrap().leaves(), 1);
}

#[test]
fn complete_search_budget_distinguishes_beneficial_splits_from_finished_leaves() {
    let mut options = options(6);
    options.predictors = vec![Predictor::Zero];
    options.max_leaves = 1;
    let samples = vec![sample(6, -1, &[0], 50), sample(6, 1, &[1], 50)];
    assert!(!options.require_complete_search);
    assert_eq!(train(&samples, &options).unwrap().leaves(), 1);
    options.require_complete_search = true;
    options.split_threshold = 99.0;
    assert!(matches!(train(&samples, &options), Err(Error::Limit(_))));
    options.split_threshold = f64::from(DefaultScorer::default().estimate_bits(&[50, 0, 50], 100));
    assert_eq!(train(&samples, &options).unwrap().leaves(), 1);
    options.split_threshold = 0.0;
    assert_eq!(
        train(&[sample(6, 0, &[400], 100)], &options)
            .unwrap()
            .leaves(),
        1
    );
    options.max_leaves = 2;
    assert_eq!(train(&samples, &options).unwrap().leaves(), 2);
    let unresolved: Vec<_> = (0..4).map(|i| sample(6, i, &[i], 100)).collect();
    assert!(matches!(train(&unresolved, &options), Err(Error::Limit(_))));
    options.max_leaves = 4;
    assert_eq!(train(&unresolved, &options).unwrap().leaves(), 4);
}

fn sample_distribution(samples: &[Sample], options: &TrainOptions) -> HashMap<SampleKey, u64> {
    let quantization = options.property_quantization.as_ref().unwrap();
    let config = HybridUintConfig::new(4, 1, 2).unwrap();
    let mut counts = HashMap::new();
    for sample in samples {
        let key = quantized_key(
            sample,
            &options.properties,
            &quantization.thresholds,
            true,
            &config,
        );
        *counts.entry(key).or_default() += u64::from(sample.weight);
    }
    counts
}

fn prepared_distribution(samples: &TreeSamples) -> HashMap<SampleKey, u64> {
    let mut counts = HashMap::new();
    for (i, &weight) in samples.counts.iter().enumerate() {
        let key = SampleKey {
            properties: samples
                .properties
                .iter()
                .map(|column| column.value(i))
                .collect(),
            residuals: samples.residuals.iter().map(|column| column[i]).collect(),
        };
        *counts.entry(key).or_default() += u64::from(weight);
    }
    counts
}

#[test]
fn prepared_u16_counts_preserve_full_range_weighted_histogram_mass() {
    let mut options = options(6);
    options.properties = vec![6, 1, 0, 15];
    options.property_quantization = Some(PropertyQuantization {
        properties: options.properties.clone(),
        thresholds: vec![
            vec![-511, -1, 0, 510],
            vec![0, 65_535, i32::MAX - 1],
            vec![0, 1000, i32::MAX - 1],
            vec![-255, -1, 0, 255],
        ],
    });
    let extrema = [i32::MIN, i32::MAX, -1, 0, 1, 8, 10];
    let mut input = Vec::new();
    for (i, &residual) in extrema.iter().enumerate() {
        for weight in [0, 1, 65_534, 65_535, 65_536, 131_070, 70_020] {
            let mut sample = sample(
                6,
                residual,
                &[residual, extrema[(i + 1) % extrema.len()]],
                weight,
            );
            sample.properties[0] = [0, 1000, i32::MAX][i % 3];
            sample.properties[1] = [0, 65_535, i32::MAX][i % 3];
            sample.properties[15] = residual;
            input.push(sample);
        }
    }
    options.validate().unwrap();
    let prepared = TreeSamples::new(&input, &options).unwrap();
    assert_eq!(
        prepared_distribution(&prepared),
        sample_distribution(&input, &options)
    );
    assert!(prepared.counts.iter().all(|&n| n != 0));
    assert!(prepared.counts.contains(&u16::MAX));
    assert_eq!(
        std::mem::size_of_val(prepared.counts.as_slice()),
        prepared.counts.len() * 2
    );
    let indices: Vec<_> = (0..prepared.counts.len()).rev().collect();
    let parents = parent_histograms(&prepared, &indices);
    let config = HybridUintConfig::new(4, 1, 2).unwrap();
    for (p, parent) in parents.iter().enumerate() {
        let mut counts = [0u32; 256];
        let mut extra = 0u64;
        for sample in &input {
            let token = config.encode(pack_signed(sample.residuals[p]));
            counts[token.token as usize] += sample.weight;
            extra += u64::from(sample.weight) * u64::from(token.nbits);
        }
        assert_eq!(parent.counts, counts[..parent.counts.len()]);
        assert!(counts[parent.counts.len()..].iter().all(|&n| n == 0));
        assert_eq!(parent.extra_bits, extra);
    }
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let cache = HistogramCache::create(&prepared, &indices, &parents, &budget, false).unwrap();
    assert_cached_population(&cache, &prepared, &indices);
    for (p, histogram) in cache.properties.iter().enumerate() {
        let mut weights = vec![0u32; prepared.thresholds[p].len() + 1];
        for sample in &input {
            let key = quantized_key(
                sample,
                &options.properties,
                &prepared.thresholds,
                true,
                &config,
            );
            weights[key.properties[p] as usize] += sample.weight;
        }
        assert_eq!(histogram.weights, weights);
    }
}

#[test]
fn prepared_u16_limits_preserve_existing_total_ceiling_before_allocation() {
    let options = options(6);
    let largest = sample(6, 0, &[i32::MIN, i32::MAX], MAX_HISTOGRAM_TOTAL);
    let prepared = TreeSamples::new(std::iter::once(&largest), &options).unwrap();
    assert_eq!(
        prepared.counts.len(),
        MAX_HISTOGRAM_TOTAL.div_ceil(65_535) as usize
    );
    assert_eq!(
        prepared.counts.iter().map(|&n| u32::from(n)).sum::<u32>(),
        MAX_HISTOGRAM_TOTAL
    );
    let indices: Vec<_> = (0..prepared.counts.len()).collect();
    for parent in parent_histograms(&prepared, &indices) {
        assert_eq!(parent.counts.iter().sum::<u32>(), MAX_HISTOGRAM_TOTAL);
        assert_eq!(parent.extra_bits, u64::from(MAX_HISTOGRAM_TOTAL) * 28);
    }
    for weight in [MAX_HISTOGRAM_TOTAL + 1, u32::MAX] {
        let rejected = sample(6, 0, &[0, 1], weight);
        assert!(matches!(
            TreeSamples::new(std::iter::once(&rejected), &options),
            Err(Error::Limit(_))
        ));
    }
    let extra = sample(6, 0, &[0, 1], 1);
    assert!(matches!(
        TreeSamples::new([&largest, &extra], &options),
        Err(Error::Limit(_))
    ));
    let zero = sample(6, 0, &[0, 1], 0);
    assert!(
        TreeSamples::new([&zero], &options)
            .unwrap()
            .counts
            .is_empty()
    );
}

#[test]
fn saturated_u16_counts_preserve_cache_subtraction_after_physical_partition() {
    let (mut samples, options) = histogram_fixture(8192, 3);
    for (i, count) in samples.counts.iter_mut().enumerate() {
        *count = if i % 17 == 0 { 1 } else { u16::MAX };
    }
    let mut indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let mut cache = HistogramCache::create(&samples, &indices, &parents, &budget, false).unwrap();
    let below = samples.properties[1].count_le(&indices, 8);
    partition_node::<true>(&mut samples, &mut indices, 1, 8, below);
    assert_eq!(indices, (0..samples.counts.len()).collect::<Vec<_>>());
    cache.subtract(&samples, &indices[..below]).unwrap();
    let live = &indices[below..];
    assert_cached_population(&cache, &samples, live);
    let parents = parent_histograms(&samples, live);
    let total = live.iter().map(|&i| u32::from(samples.counts[i])).sum();
    let mut scratch = PropertyScratch::default();
    for property in 0..options.properties.len() {
        let cached = evaluate_property_cached(
            &samples,
            live,
            &options,
            &parents,
            0,
            total,
            property,
            Some(&cache.properties[property]),
            None,
            &mut scratch,
        )
        .unwrap();
        let fresh =
            evaluate_property(&samples, live, &options, &parents, 0, total, property).unwrap();
        assert_same_candidates(cached, fresh);
    }
    drop(cache);
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
}

#[test]
fn batch_hashes_match_source_wrapping_golden_and_scalar_rehash() {
    let key = SampleKey {
        properties: vec![0, 800, 255],
        residuals: vec![
            ResidualToken { token: 7, nbits: 3 },
            ResidualToken {
                token: 128,
                nbits: 16,
            },
        ],
    };
    let keys: Vec<_> = (0..256)
        .map(|i| SampleKey {
            properties: vec![i, 800 + i, 255 - i],
            ..key.clone()
        })
        .collect();
    let hashes = SampleBatchCollector::hash_batch(&keys);
    assert_eq!(hashes[0], (0x610a_7a6b_7605_8ae6, 0xce04_e503_cb3e_0cee));
    for (key, (first, second)) in keys.iter().zip(hashes) {
        assert_eq!(
            SampleBatchCollector::positions(key, 1023),
            [
                (first >> 16) as usize & 1023,
                (second >> 16) as usize & 1023
            ]
        );
    }
}

#[test]
fn batch_merging_preserves_weighted_token_objective_and_partial_tail() {
    let mut options = options(6);
    options.property_quantization = Some(PropertyQuantization {
        properties: vec![6],
        thresholds: vec![vec![0]],
    });
    let mut input: Vec<_> = (0..1025)
        .map(|i| {
            sample(
                6,
                if i % 3 == 0 { -1 } else { 1 },
                if i % 3 == 0 { &[0, 400] } else { &[400, 0] },
                1,
            )
        })
        .collect();
    input.push(sample(6, 1, &[8, 0], 70_000));
    input.push(sample(6, 1, &[10, 0], 17));
    let mut collector = SampleBatchCollector::new(&options).unwrap();
    for sample in &input {
        collector.push(sample.clone()).unwrap();
    }
    let merged = collector.finish().unwrap();
    assert!(merged.len() < input.len() / 10);
    assert!(merged.iter().all(|s| s.weight <= u16::MAX as u32));
    assert_eq!(
        sample_distribution(&input, &options),
        sample_distribution(&merged, &options)
    );
    let tree = train(&merged, &options).unwrap();
    let reference = train(&input, &options).unwrap();
    assert_eq!(signature(&tree), signature(&reference));
    assert_eq!(
        estimated_cost(&tree, &input, &options),
        estimated_cost(&reference, &input, &options)
    );
}

#[test]
fn full_i32_residual_tokens_preserve_batch_weights_and_adaptive_cost() {
    let mut options = options(0);
    options.require_complete_search = true;
    options.property_quantization = Some(PropertyQuantization {
        properties: vec![0],
        thresholds: vec![vec![0, 1, 2, 3]],
    });
    let residuals = [i32::MIN, i32::MAX, -1, 0, 1];
    let expected = [(239, 28), (238, 28), (1, 0), (0, 0), (2, 0)];
    let config = HybridUintConfig::new(4, 1, 2).unwrap();
    for (&residual, &(token, nbits)) in residuals.iter().zip(&expected) {
        let encoded = config.encode(pack_signed(residual));
        assert_eq!((encoded.token, encoded.nbits), (token, nbits));
    }
    let raw: Vec<_> = (0..1027)
        .map(|i| {
            let index = i % residuals.len();
            sample(0, index as i32, &[residuals[index]; 2], 1 + (i % 11) as u32)
        })
        .collect();
    let mut collector = SampleBatchCollector::new(&options).unwrap();
    for sample in &raw {
        collector.push(sample.clone()).unwrap();
    }
    let merged = collector.finish().unwrap();
    assert_eq!(merged.len(), residuals.len());
    assert_eq!(
        merged.iter().map(|s| s.weight).sum::<u32>(),
        raw.iter().map(|s| s.weight).sum::<u32>()
    );
    let raw_tree = train(&raw, &options).unwrap();
    let merged_tree = train(&merged, &options).unwrap();
    assert_eq!(signature(&raw_tree), signature(&merged_tree));
    assert!(raw_tree.leaves() > 1);
    let cost = estimated_cost(&merged_tree, &merged, &options);
    assert_eq!(cost, estimated_cost(&raw_tree, &raw, &options));
    assert!(
        cost < estimated_cost(
            &Tree::new(vec![leaf(Predictor::Weighted)]).unwrap(),
            &raw,
            &options
        )
    );
}

#[test]
fn weighted_cache_growth_clears_saturated_aliases_and_preserves_collision_misses() {
    let mut options = options(0);
    options.property_quantization = Some(PropertyQuantization {
        properties: vec![0],
        thresholds: vec![(0..1024).collect()],
    });
    let mut collector = SampleBatchCollector::new(&options).unwrap();
    let mut input = vec![sample(0, 0, &[0, 1], 65_534)];
    collector.push(input[0].clone()).unwrap();
    collector.flush().unwrap();
    let initial_slots = collector.table.len();
    for i in 1..=1024 {
        let sample = sample(0, i, &[0, 1], 1);
        input.push(sample.clone());
        collector.push(sample).unwrap();
    }
    collector.flush().unwrap();
    assert!(collector.table.len() > initial_slots);
    for weight in [2, 9] {
        let sample = sample(0, 0, &[0, 1], weight);
        input.push(sample.clone());
        collector.push(sample).unwrap();
        collector.flush().unwrap();
    }
    assert!(
        collector
            .table
            .iter()
            .filter(|&&i| i != SampleBatchCollector::UNUSED)
            .all(|&i| collector.samples[i as usize].weight < u16::MAX as u32)
    );
    assert!(collector.table.len() <= SampleBatchCollector::MAX_SLOTS);
    let merged = collector.finish().unwrap();
    assert_eq!(
        sample_distribution(&input, &options),
        sample_distribution(&merged, &options)
    );
    assert_eq!(
        merged
            .iter()
            .filter(|s| s.properties[0] == 0)
            .map(|s| u64::from(s.weight))
            .sum::<u64>(),
        65_545
    );
}

#[test]
fn batch_collector_rejects_overflow_missing_quantiles_and_raw_constraints() {
    let mut options = options(6);
    assert!(SampleBatchCollector::new(&options).is_err());
    options.property_quantization = Some(PropertyQuantization {
        properties: vec![6],
        thresholds: vec![vec![0]],
    });
    let mut collector = SampleBatchCollector::new(&options).unwrap();
    assert!(collector.push(sample(6, 0, &[0], 1)).is_err());
    collector.push(sample(6, 0, &[0, 0], u32::MAX)).unwrap();
    assert!(matches!(
        collector.push(sample(6, 0, &[0, 0], 1)),
        Err(Error::Limit(_))
    ));
    options.static_property_range = Some([[0, 1], [0, 1]]);
    options.multiplier_info = vec![MultiplierInfo {
        range: [[0, 1], [0, 1]],
        multiplier: 3,
    }];
    assert!(SampleBatchCollector::new(&options).is_err());
}

fn reuse_fixture() -> (Vec<Sample>, TrainOptions) {
    let options = TrainOptions {
        properties: vec![0, 2, 3, 6],
        max_leaves: 8,
        split_threshold: 0.0,
        parallel: false,
        require_complete_search: true,
        ..TrainOptions::default()
    };
    let samples = (0..8192)
        .map(|i| {
            let residuals = if i < 512 {
                [0, 2000]
            } else if i < 4096 {
                [2000, 0]
            } else {
                [2000, 1]
            };
            let mut s = sample(0, i32::from(i < 512), &residuals, 1 + (i % 7) as u32);
            s.properties[2] = i / 128;
            s.properties[3] = i % 128;
            s.properties[6] = i32::from(i >= 4096);
            s
        })
        .collect();
    (samples, options)
}

#[test]
fn active_parent_and_property_cache_reuse_match_uncached_learning() {
    let (input, mut options) = reuse_fixture();
    let samples = TreeSamples::new(&input, &options).unwrap();
    assert_eq!(samples.counts.len(), input.len());
    let stats = TrainingStats::default();
    let cached =
        Tree::new(train_nodes_with_stats(&mut samples.clone(), &options, &stats).unwrap()).unwrap();
    assert!(stats.parent_reuses.load(Ordering::Relaxed) > 0);
    assert!(stats.cache_builds.load(Ordering::Relaxed) > 0);
    assert!(stats.cache_reuses.load(Ordering::Relaxed) > 0);
    options.histogram_cache_bytes = 0;
    options.reuse_parent_histograms = false;
    let reference = train(&input, &options).unwrap();
    assert_eq!(signature(&cached), signature(&reference));
    assert_eq!(
        estimated_cost(&cached, &input, &options),
        estimated_cost(&reference, &input, &options)
    );
    options.histogram_cache_bytes = 1;
    let denied = TrainingStats::default();
    let bounded =
        Tree::new(train_nodes_with_stats(&mut samples.clone(), &options, &denied).unwrap())
            .unwrap();
    assert_eq!(denied.cache_builds.load(Ordering::Relaxed), 0);
    assert_eq!(signature(&bounded), signature(&reference));
}

#[test]
fn cache_subtraction_preserves_record_counts_weights_and_extra_bits_with_trimmed_symbols() {
    let (input, options) = reuse_fixture();
    let samples = TreeSamples::new(&input, &options).unwrap();
    let all: Vec<_> = (0..input.len()).collect();
    let parents = parent_histograms(&samples, &all);
    let small = parent_histograms(&samples, &all[..512]);
    let large = parent_histograms(&samples, &all[512..]);
    assert_eq!(
        subtract_parent_histograms(parents.clone(), &small).unwrap(),
        large
    );
    let budget = HistogramBudget::new(32 * 1024 * 1024);
    let mut cache = HistogramCache::create(&samples, &all, &parents, &budget, false).unwrap();
    let bytes = cache._lease.bytes;
    assert_eq!(budget.used.load(Ordering::Relaxed), bytes);
    cache.subtract(&samples, &all[..512]).unwrap();
    let fresh_budget = HistogramBudget::new(32 * 1024 * 1024);
    let fresh =
        HistogramCache::create(&samples, &all[512..], &large, &fresh_budget, false).unwrap();
    for (cached, fresh) in cache.properties.iter().zip(&fresh.properties) {
        assert_eq!(cached.used, fresh.used);
        assert_eq!(cached.weights, fresh.weights);
        assert_eq!(cached.used.iter().sum::<usize>(), 7680);
        for (a, b) in cached.predictors.iter().zip(&fresh.predictors) {
            assert_eq!(a.extra, b.extra);
            for bin in 0..cached.used.len() {
                assert_eq!(
                    &a.counts[bin * a.symbols..bin * a.symbols + b.symbols],
                    &b.counts[bin * b.symbols..(bin + 1) * b.symbols]
                );
                assert!(
                    a.counts[bin * a.symbols + b.symbols..(bin + 1) * a.symbols]
                        .iter()
                        .all(|&n| n == 0)
                );
            }
        }
    }
    drop(cache);
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    assert!(budget.peak.load(Ordering::Relaxed) <= budget.limit);
    let exact_budget = HistogramBudget::new(bytes);
    let retained = HistogramCache::create(&samples, &all, &parents, &exact_budget, false).unwrap();
    assert!(HistogramCache::create(&samples, &all, &parents, &exact_budget, false).is_none());
    drop(retained);
    assert_eq!(exact_budget.used.load(Ordering::Relaxed), 0);
}

fn histogram_fixture(records: usize, predictors: usize) -> (TreeSamples, TrainOptions) {
    let mut options = TrainOptions::variable();
    options.predictors.truncate(predictors);
    options.properties = vec![0, 6, 15, 7];
    options.split_threshold = 0.0;
    let samples = TreeSamples {
        properties: vec![
            PropertyColumn::Wide((0..records).map(|i| (i % 3) as u32).collect()),
            PropertyColumn::Packed((0..records).map(|i| (i % 17) as u8).collect()),
            PropertyColumn::Packed((0..records).map(|i| (i % 5) as u8).collect()),
            PropertyColumn::Packed(vec![0; records]),
        ],
        residuals: (0..predictors)
            .map(|p| {
                (0..records)
                    .map(|i| ResidualToken {
                        token: match p % 3 {
                            0 => [0, 64, 129, 255][(i / 3 + p) % 4],
                            1 => [0, 8, 17][(i + p) % 3],
                            _ => [0, 33, 193][(i / 5 + p) % 3],
                        },
                        nbits: ((i + p) % 28) as u8,
                    })
                    .collect()
            })
            .collect(),
        counts: (0..records).map(|i| 1 + (i % 13) as u16).collect(),
        thresholds: vec![vec![0, 1], (0..16).collect(), (0..4).collect(), vec![0]],
    };
    (samples, options)
}

fn assert_same_candidates(actual: Candidates, expected: Candidates) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        match (actual, expected) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.property, b.property);
                assert_eq!(a.bin, b.bin);
                assert_eq!(a.below_count, b.below_count);
                assert_eq!(a.below_predictor, b.below_predictor);
                assert_eq!(a.above_predictor, b.above_predictor);
                assert_eq!(a.bits.to_bits(), b.bits.to_bits());
            }
            mismatch => panic!("different split candidates: {mismatch:?}"),
        }
    }
}

#[test]
fn predictor_dedup_requires_exact_node_local_proofs() {
    let (mut samples, mut options) = histogram_fixture(64, 7);
    let representative = samples.residuals[1].clone();
    samples.residuals[2..].fill(representative);
    let indices: Vec<_> = (0..64).rev().collect();
    let parents = parent_histograms(&samples, &indices);
    assert!(!options.predictor_dedup);
    assert_eq!(
        PredictorPlan::new(&samples, &indices, &options, &parents, 0).active(),
        &[0, 1, 2, 3, 4, 5, 6]
    );
    options.predictor_dedup = true;
    let plan = PredictorPlan::new(&samples, &indices, &options, &parents, 0);
    assert_eq!(&plan.representative[..7], &[0, 1, 1, 1, 1, 1, 6]);
    assert_eq!(plan.active(), &[0, 1, 6]);
    // Zero's epsilon and the leaf's change-predictor penalty remain material.
    assert_ne!(plan.penalty[1], plan.penalty[6]);
    let changed_leaf = PredictorPlan::new(&samples, &indices, &options, &parents, 2);
    assert_eq!(changed_leaf.representative[2], 2);
    assert_eq!(changed_leaf.representative[3], 1);

    let mut padded = parents.clone();
    padded[1].counts.extend([0; 9]);
    assert_eq!(
        PredictorPlan::new(&samples, &indices, &options, &padded, 0).representative[2],
        1
    );
    *padded[1].counts.last_mut().unwrap() = 1;
    assert_eq!(
        PredictorPlan::new(&samples, &indices, &options, &padded, 0).representative[2],
        2
    );
    let mut wrong_extra = parents.clone();
    wrong_extra[2].extra_bits += 1;
    assert_eq!(
        PredictorPlan::new(&samples, &indices, &options, &wrong_extra, 0).representative[2],
        2
    );

    // Equal weighted marginals are not a proof of per-record equivalence.
    samples.residuals[2].swap(0, 13);
    assert_eq!(parent_histograms(&samples, &indices), parents);
    assert_eq!(
        PredictorPlan::new(&samples, &indices, &options, &parents, 0).representative[2],
        2
    );
    samples.residuals[2].swap(0, 13);
    // Tokens are equal here; only the individual extra-bit counts differ.
    samples.residuals[2].swap(0, 39);
    assert_eq!(parent_histograms(&samples, &indices), parents);
    assert_eq!(
        PredictorPlan::new(&samples, &indices, &options, &parents, 0).representative[2],
        2
    );
    let live: Vec<_> = indices
        .iter()
        .copied()
        .filter(|&i| i != 0 && i != 39)
        .collect();
    let live_parents = parent_histograms(&samples, &live);
    assert_eq!(
        PredictorPlan::new(&samples, &live, &options, &live_parents, 0).representative[2],
        1
    );

    for predictors in [1, 2] {
        let (samples, mut options) = histogram_fixture(64, predictors);
        options.predictor_dedup = true;
        let parents = parent_histograms(&samples, &indices);
        assert_eq!(
            PredictorPlan::new(&samples, &indices, &options, &parents, 0).active_count,
            predictors
        );
    }
}

#[test]
fn predictor_dedup_matches_all_costs_across_pair_and_compaction_boundaries() {
    let mut actual_scratch = PropertyScratch::default();
    let mut expected_scratch = PropertyScratch::default();
    for records in [63, 64, 255, 256, 257, 8192] {
        for aliases in [&[(2, 1), (5, 4)][..], &[(3, 2)][..], &[][..]] {
            let (mut samples, options) = histogram_fixture(records, 7);
            samples.counts[0] = u16::MAX;
            samples.counts[records - 1] = u16::MAX - 1;
            for &(predictor, previous) in aliases {
                samples.residuals[predictor] = samples.residuals[previous].clone();
            }
            let mut dedup_options = options.clone();
            dedup_options.predictor_dedup = true;
            let indices: Vec<_> = (0..records).rev().collect();
            let parents = parent_histograms(&samples, &indices);
            let plan = PredictorPlan::new(&samples, &indices, &dedup_options, &parents, 0);
            let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
            let compact =
                compact_parent_histograms(&parents, DefaultScorer::new().lanes()).unwrap();
            for property in 0..options.properties.len() {
                let cache = with_property_values!(&samples.properties[property], |values| {
                    PropertyHistogram::collect_values(
                        &samples, &indices, &parents, property, values, &plan,
                    )
                    .unwrap()
                });
                let expected_cache =
                    PropertyHistogram::collect(&samples, &indices, &parents, property);
                assert_eq!(cache.used, expected_cache.used);
                assert_eq!(cache.weights, expected_cache.weights);
                for (actual, expected) in cache.predictors.iter().zip(&expected_cache.predictors) {
                    assert_eq!(actual.symbols, expected.symbols);
                    assert_eq!(actual.counts, expected.counts);
                    assert_eq!(actual.extra, expected.extra);
                }
                for mode in 0..3 {
                    let (scoring_parents, mapping) = if mode == 2 {
                        (compact.parents.as_slice(), Some(&compact.mapping))
                    } else {
                        (parents.as_slice(), None)
                    };
                    let actual = evaluate_property_cached(
                        &samples,
                        &indices,
                        &dedup_options,
                        scoring_parents,
                        0,
                        total,
                        property,
                        (mode == 1).then_some(&cache),
                        mapping,
                        &mut actual_scratch,
                    )
                    .unwrap();
                    let expected = evaluate_property_cached(
                        &samples,
                        &indices,
                        &options,
                        scoring_parents,
                        0,
                        total,
                        property,
                        (mode == 1).then_some(&expected_cache),
                        mapping,
                        &mut expected_scratch,
                    )
                    .unwrap();
                    assert_same_candidates(actual, expected);
                    if property == 3 {
                        assert!(actual_scratch.scored_predictors.is_empty());
                    } else {
                        assert_eq!(actual_scratch.scored_predictors, plan.active());
                        for (a, b) in actual_scratch
                            .below
                            .iter()
                            .chain(&actual_scratch.above)
                            .zip(expected_scratch.below.iter().chain(&expected_scratch.above))
                        {
                            assert_eq!(a.bits.to_bits(), b.bits.to_bits());
                            assert_eq!(a.penalty.to_bits(), b.penalty.to_bits());
                            assert_eq!(a.predictor, b.predictor);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn predictor_dedup_copies_fixed_strides_and_rejects_nonzero_truncation() {
    let source = BinHistogram {
        symbols: 4,
        counts: vec![3, 5, 0, 0, 7, 11, 0, 0],
        extra: vec![13, 17],
    };
    let mut short = BinHistogram {
        symbols: 2,
        counts: vec![99; 4],
        extra: vec![99; 2],
    };
    let allocation = (
        short.counts.as_ptr(),
        short.counts.capacity(),
        short.extra.as_ptr(),
    );
    short.copy_predictor_from(&source).unwrap();
    assert_eq!(short.counts, [3, 5, 7, 11]);
    assert_eq!(short.extra, source.extra);
    assert_eq!(
        allocation,
        (
            short.counts.as_ptr(),
            short.counts.capacity(),
            short.extra.as_ptr()
        )
    );
    let mut wide = BinHistogram {
        symbols: 4,
        counts: vec![99; 8],
        extra: vec![99; 2],
    };
    wide.copy_predictor_from(&short).unwrap();
    assert_eq!(wide.counts, source.counts);
    assert_eq!(wide.extra, source.extra);
    wide.counts[6] = 1;
    assert!(matches!(
        short.copy_predictor_from(&wide),
        Err(Error::Encoding(_))
    ));
}

#[test]
fn predictor_dedup_reproves_inherited_cache_after_subtraction_and_leaf_change() {
    let (mut samples, mut options) = histogram_fixture(8192, 5);
    options.parallel = false;
    options.predictor_dedup = true;
    for residual in &mut samples.residuals[1][..513] {
        residual.token = 255;
    }
    let representative = samples.residuals[1].clone();
    samples.residuals[2..].fill(representative);
    for residual in &mut samples.residuals[3][..513] {
        *residual = ResidualToken { token: 0, nbits: 0 };
    }
    for residual in &mut samples.residuals[4][..513] {
        residual.token = 254;
    }
    let indices: Vec<_> = (0..8192).collect();
    let parents = parent_histograms(&samples, &indices);
    let bytes = expected_cache_bytes(&samples, &parents);
    let budget = HistogramBudget::new(bytes);
    let scratch = ScratchPool::new(1);
    let mut cache = None;
    let mut allocations = None;
    let mut previous = 0;
    for (begin, leaf, equivalents) in [(0, 0, 1), (513, 0, 3), (777, 2, 2)] {
        let stats = TrainingStats::default();
        let actual = find_best_split_with_scratch(
            &samples,
            &indices[begin..],
            leaf,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &stats,
            &scratch,
            &indices[previous..begin],
        )
        .unwrap();
        assert_eq!(
            stats.equivalent_predictors.load(Ordering::Relaxed),
            equivalents
        );
        assert_eq!(budget.used.load(Ordering::Relaxed), bytes);
        let cached = cache.as_ref().unwrap();
        assert_cached_population(cached, &samples, &indices[begin..]);
        let current: Vec<_> = cached
            .properties
            .iter()
            .flat_map(|property| {
                property.predictors.iter().map(|p| {
                    (
                        p.symbols,
                        p.counts.as_ptr(),
                        p.counts.len(),
                        p.counts.capacity(),
                        p.extra.as_ptr(),
                        p.extra.capacity(),
                    )
                })
            })
            .collect();
        if let Some(allocations) = &allocations {
            assert_eq!(&current, allocations);
        } else {
            allocations = Some(current);
        }
        let mut reference_options = options.clone();
        reference_options.predictor_dedup = false;
        let expected = find_best_split(
            &samples,
            &indices[begin..],
            leaf,
            &reference_options,
            &mut None,
            &mut None,
            &HistogramBudget::new(0),
            &TrainingStats::default(),
        )
        .unwrap();
        assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
        previous = begin;
    }
    // A corrupted nonzero source tail must not publish partially copied entries.
    cache.as_mut().unwrap().properties[0].predictors[1].counts[255] = 1;
    assert!(matches!(
        find_best_split_with_scratch(
            &samples,
            &indices[777..],
            0,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &TrainingStats::default(),
            &scratch,
            &[],
        ),
        Err(Error::Encoding(_))
    ));
    assert!(cache.is_none());
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    find_best_split_with_scratch(
        &samples,
        &indices[777..],
        0,
        &options,
        &mut None,
        &mut cache,
        &budget,
        &TrainingStats::default(),
        &scratch,
        &[],
    )
    .unwrap();
    assert!(cache.is_none());
    assert!(scratch.reused.load(Ordering::Relaxed) > 0);
}

#[test]
fn predictor_dedup_preserves_complete_search_serial_parallel_and_null_pool() {
    let (mut input, mut options) = reuse_fixture();
    options.predictors.push(Predictor::Top);
    for sample in &mut input {
        sample.residuals.push(sample.residuals[1]);
    }
    let prepared = TreeSamples::new(&input, &options).unwrap();
    let baseline_stats = TrainingStats::default();
    let baseline = Tree::new(
        train_nodes_with_stats(&mut prepared.clone(), &options, &baseline_stats).unwrap(),
    )
    .unwrap();
    assert_eq!(
        baseline_stats.equivalent_predictors.load(Ordering::Relaxed),
        0
    );
    options.predictor_dedup = true;
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for (parallel, null_pool) in [(false, false), (true, false), (true, true)] {
        options.parallel = parallel;
        for cache_bytes in [0, 32 * 1024 * 1024] {
            options.histogram_cache_bytes = cache_bytes;
            let stats = TrainingStats::default();
            let run = || train_nodes_with_stats(&mut prepared.clone(), &options, &stats);
            let nodes = workers
                .install(|| {
                    if null_pool {
                        execution::serial(run)
                    } else {
                        run()
                    }
                })
                .unwrap();
            let actual = Tree::new(nodes).unwrap();
            assert_eq!(signature(&actual), signature(&baseline));
            assert_eq!(
                estimated_cost(&actual, &input, &options),
                estimated_cost(&baseline, &input, &options)
            );
            assert!(stats.equivalent_predictors.load(Ordering::Relaxed) > 0);
        }
    }
}

#[test]
fn scoring_tools_preserve_cached_costs_and_distinct_record_gates() {
    let mut scratch = PropertyScratch::default();
    let mut reference = PropertyScratch::default();
    for records in [255, 256, 2047, 2048, 16383, 16384] {
        let (samples, mut options) = histogram_fixture(records, 3);
        let indices: Vec<_> = (0..records).rev().collect();
        let parents = parent_histograms(&samples, &indices);
        let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
        let cache = HistogramCache::create(
            &samples,
            &indices,
            &parents,
            &HistogramBudget::new(32 * 1024 * 1024),
            false,
        )
        .unwrap();
        for scoring in [
            ScoringOptions::default(),
            ScoringOptions {
                count_log: true,
                ..ScoringOptions::default()
            },
            ScoringOptions {
                fast_entropy: true,
                ..ScoringOptions::default()
            },
            ScoringOptions {
                clipped: ClippedEntropyMode::AtLeast2048,
                ..ScoringOptions::default()
            },
            ScoringOptions {
                clipped: ClippedEntropyMode::AtLeast16384,
                ..ScoringOptions::default()
            },
            ScoringOptions {
                clipped: ClippedEntropyMode::SlidingAtLeast2048,
                ..ScoringOptions::default()
            },
            ScoringOptions {
                description: DescriptionCost::Half,
                ..ScoringOptions::default()
            },
        ] {
            options.scoring = scoring;
            for property in 0..options.properties.len() {
                let a = evaluate_property_cached(
                    &samples,
                    &indices,
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
                    &indices,
                    &options,
                    &parents,
                    0,
                    total,
                    property,
                    Some(&cache.properties[property]),
                    None,
                    &mut reference,
                )
                .unwrap();
                assert_same_candidates(a, b);
                for (a, b) in scratch
                    .below
                    .iter()
                    .chain(&scratch.above)
                    .zip(reference.below.iter().chain(&reference.above))
                {
                    assert_eq!(a.bits.to_bits(), b.bits.to_bits());
                    assert_eq!(a.penalty.to_bits(), b.penalty.to_bits());
                    assert_eq!(a.predictor, b.predictor);
                }
                if property != 3
                    && matches!(
                        scoring.clipped,
                        ClippedEntropyMode::AtLeast2048 | ClippedEntropyMode::SlidingAtLeast2048
                    )
                {
                    assert_eq!(scratch.scoring.routes[0] > 0, records >= 2048);
                }
            }
        }
    }
}

#[test]
fn scoring_tools_keep_complete_search_and_fast_entropy_disables_compaction() {
    let (input, mut options) = reuse_fixture();
    options.parallel = false;
    let samples = TreeSamples::new(&input, &options).unwrap();
    for scoring in [
        ScoringOptions {
            fast_entropy: true,
            ..ScoringOptions::default()
        },
        ScoringOptions {
            clipped: ClippedEntropyMode::Sliding,
            ..ScoringOptions::default()
        },
        ScoringOptions {
            count_log: true,
            description: DescriptionCost::Full,
            ..ScoringOptions::default()
        },
    ] {
        options.scoring = scoring;
        let mut expected = None;
        for budget in [0, 32 * 1024 * 1024] {
            options.histogram_cache_bytes = budget;
            let stats = TrainingStats::default();
            let tree =
                Tree::new(train_nodes_with_stats(&mut samples.clone(), &options, &stats).unwrap())
                    .unwrap();
            if let Some(reference) = &expected {
                assert_eq!(&signature(&tree), reference);
            } else {
                expected = Some(signature(&tree));
            }
            if scoring.fast_entropy {
                assert_eq!(stats.compact_nodes.load(Ordering::Relaxed), 0);
            }
        }
        let limited = TrainOptions {
            max_leaves: 1,
            require_complete_search: true,
            ..options.clone()
        };
        assert!(matches!(train(&input, &limited), Err(Error::Limit(_))));
    }
}

#[test]
fn sparse_node_plan_is_shared_across_properties_cache_layouts_and_refinement() {
    let lanes = DefaultScorer::new().lanes();
    let (mut samples, mut options) = histogram_fixture(1024, 3);
    for i in 0..samples.counts.len() {
        samples.residuals[0][i].token = (i % lanes) as u8;
        samples.residuals[1][i].token = if i % 2 == 0 { 0 } else { (7 * lanes) as u8 };
        samples.residuals[2][i].token = (i % (2 * lanes + 1)) as u8;
    }
    samples.properties[1] = PropertyColumn::Packed((0..1024).map(|i| (i % 96) as u8).collect());
    samples.thresholds[1] = (0..95).collect();
    options.sparse_entropy = true;
    let indices: Vec<_> = (0..1024).rev().collect();
    let parents = parent_histograms(&samples, &indices);
    let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
    let predictors = PredictorPlan::new(&samples, &indices, &options, &parents, 0);
    let sparse = SparseHistogramPlan::new(&parents, predictors.active(), lanes, true);
    assert_eq!(sparse.blocks(0), None);
    assert_eq!(sparse.blocks(2), None);
    let blocks = sparse.blocks(1).unwrap();
    let snapshot = blocks.to_vec();
    let pointer = blocks.as_ptr();
    let mut actual_scratch = PropertyScratch::default();
    let mut reference_scratch = PropertyScratch::default();
    for padded in [false, true] {
        options.padded_histogram = padded;
        let stride = padded.then(|| {
            histogram_stride(
                parents.iter().map(|p| p.counts.len()).max().unwrap(),
                lanes,
                true,
            )
        });
        for refine in [false, true] {
            options.search.refine_quantiles = refine;
            let mut reference = options.clone();
            reference.sparse_entropy = false;
            for property in 0..options.properties.len() {
                let cached = with_property_values!(&samples.properties[property], |values| {
                    PropertyHistogram::collect_values_with_stride(
                        &samples,
                        &indices,
                        &parents,
                        property,
                        values,
                        &predictors,
                        stride,
                    )
                    .unwrap()
                });
                for cache in [false, true] {
                    let actual = with_property_values!(&samples.properties[property], |values| {
                        evaluate_property_values(
                            &samples,
                            &indices,
                            &options,
                            &parents,
                            &predictors,
                            &sparse,
                            total,
                            property,
                            values,
                            cache.then_some(&cached),
                            false,
                            None,
                            &mut actual_scratch,
                        )
                        .unwrap()
                    });
                    let expected = evaluate_property_cached(
                        &samples,
                        &indices,
                        &reference,
                        &parents,
                        0,
                        total,
                        property,
                        cache.then_some(&cached),
                        None,
                        &mut reference_scratch,
                    )
                    .unwrap();
                    assert_same_candidates(actual, expected);
                    for (a, b) in actual_scratch
                        .below
                        .iter()
                        .chain(&actual_scratch.above)
                        .zip(
                            reference_scratch
                                .below
                                .iter()
                                .chain(&reference_scratch.above),
                        )
                    {
                        assert_eq!(a.bits.to_bits(), b.bits.to_bits());
                        assert_eq!(a.penalty.to_bits(), b.penalty.to_bits());
                        assert_eq!(a.predictor, b.predictor);
                    }
                    if property != 3 {
                        let passes = if refine && property == 1 { 2 } else { 1 };
                        assert_eq!(actual_scratch.sparse_predictors, vec![1; passes]);
                    } else {
                        assert!(actual_scratch.sparse_predictors.is_empty());
                    }
                    assert_eq!(sparse.blocks(1).unwrap(), snapshot);
                    assert_eq!(sparse.blocks(1).unwrap().as_ptr(), pointer);
                }
            }
        }
    }
}

#[test]
fn sparse_node_plans_follow_compaction_and_deferred_child_cache_in_parallel() {
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for (records, dense, cache_bytes) in [
        (128, DenseHistogram::Disabled, 0),
        (128, DenseHistogram::Small, 0),
        (8192, DenseHistogram::Disabled, 0),
        (8192, DenseHistogram::Disabled, 32 << 20),
    ] {
        let (mut samples, mut options) = histogram_fixture(records, 3);
        if records >= 8192 {
            for (i, residual) in samples.residuals[0].iter_mut().enumerate() {
                residual.token = if i < 513 {
                    255
                } else if i % 2 == 0 {
                    0
                } else {
                    64
                };
            }
        }
        options.dense_histogram = dense;
        options.histogram_cache_bytes = cache_bytes;
        let indices: Vec<_> = (0..records).collect();
        for parallel in [false, true] {
            options.parallel = parallel;
            let run = |sparse| {
                let mut options = options.clone();
                options.sparse_entropy = sparse;
                let stats = TrainingStats::default();
                let scratch = ScratchPool::new(3);
                let budget = HistogramBudget::new(cache_bytes);
                let mut parents = None;
                let mut cache = None;
                let root = find_best_split_with_scratch(
                    &samples,
                    &indices,
                    0,
                    &options,
                    &mut parents,
                    &mut cache,
                    &budget,
                    &stats,
                    &scratch,
                    &[],
                )
                .unwrap();
                assert_eq!(
                    stats.compact_nodes.load(Ordering::Relaxed),
                    usize::from(records == 128)
                );
                if records >= 8192 && parallel {
                    assert_eq!(scratch.dispatch_slots.load(Ordering::Relaxed), 3);
                }
                let child = if cache_bytes != 0 {
                    assert_eq!(parents.as_ref().unwrap()[0].counts.len(), 256);
                    assert!(cache.is_some());
                    parents = None;
                    let child = find_best_split_with_scratch(
                        &samples,
                        &indices[513..],
                        0,
                        &options,
                        &mut parents,
                        &mut cache,
                        &budget,
                        &stats,
                        &scratch,
                        &indices[..513],
                    )
                    .unwrap();
                    assert_eq!(parents.as_ref().unwrap()[0].counts.len(), 65);
                    assert_eq!(stats.cache_reuses.load(Ordering::Relaxed), 1);
                    assert_cached_population(cache.as_ref().unwrap(), &samples, &indices[513..]);
                    child
                } else {
                    assert!(cache.is_none());
                    None
                };
                (root, child)
            };
            let expected = workers.install(|| run(false));
            let actual = workers.install(|| run(true));
            assert_same_candidates(
                [actual.0, actual.1, None, None],
                [expected.0, expected.1, None, None],
            );
        }
    }
}

#[test]
fn sparse_node_plan_stays_disabled_for_shannon_scoring() {
    let (samples, mut options) = histogram_fixture(512, 3);
    options.sparse_entropy = true;
    options.scoring.fast_entropy = true;
    let indices: Vec<_> = (0..512).collect();
    let parents = parent_histograms(&samples, &indices);
    let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
    let mut scratch = PropertyScratch::default();
    let actual = evaluate_property_cached(
        &samples,
        &indices,
        &options,
        &parents,
        0,
        total,
        1,
        None,
        None,
        &mut scratch,
    )
    .unwrap();
    assert!(scratch.sparse_predictors.is_empty());
    options.sparse_entropy = false;
    let expected = evaluate_property(&samples, &indices, &options, &parents, 0, total, 1).unwrap();
    assert_same_candidates(actual, expected);
    assert!(!TrainOptions::default().sparse_entropy);
}

#[test]
fn optional_histograms_preserve_all_costs_and_source_dispatch_priority() {
    let mut actual_scratch = PropertyScratch::default();
    let mut reference_scratch = PropertyScratch::default();
    for records in [255, 256, 257, 4096, 4097, 32768] {
        let (mut samples, baseline) = histogram_fixture(records, 3);
        for residuals in &mut samples.residuals {
            for residual in residuals {
                residual.token &= 127;
            }
        }
        samples.properties[1] =
            PropertyColumn::Packed((0..records).map(|i| ((i % 17) * 15) as u8).collect());
        samples.thresholds[1] = (0..255).collect();
        let indices: Vec<_> = (0..records).rev().collect();
        let parents = parent_histograms(&samples, &indices);
        let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
        let mut options = baseline.clone();
        options.histogram_shards = true;
        options.padded_histogram = true;
        options.property_bins = PropertyBins::All;
        options.sparse_entropy = true;
        let stride = histogram_stride(128, DefaultScorer::new().lanes(), true);
        for property in 0..options.properties.len() {
            let cached = with_property_values!(&samples.properties[property], |values| {
                let mut histogram = PropertyHistogram::collect_marginals_values(
                    &samples, &indices, property, values,
                );
                let sharded = histogram
                    .collect_sharded_predictors_values(
                        &samples,
                        &indices,
                        &parents,
                        values,
                        &PredictorPlan::identity(parents.len()),
                        Some(stride),
                        stride,
                        &mut actual_scratch.shards,
                    )
                    .unwrap();
                if property == 0 && records >= 4096 {
                    assert_eq!(sharded, 1);
                }
                histogram
            });
            for cache in [false, true] {
                let actual = evaluate_property_cached(
                    &samples,
                    &indices,
                    &options,
                    &parents,
                    0,
                    total,
                    property,
                    cache.then_some(&cached),
                    None,
                    &mut actual_scratch,
                )
                .unwrap();
                let expected = evaluate_property_cached(
                    &samples,
                    &indices,
                    &baseline,
                    &parents,
                    0,
                    total,
                    property,
                    None,
                    None,
                    &mut reference_scratch,
                )
                .unwrap();
                assert_same_candidates(actual, expected);
                for (a, b) in actual_scratch
                    .below
                    .iter()
                    .chain(&actual_scratch.above)
                    .zip(
                        reference_scratch
                            .below
                            .iter()
                            .chain(&reference_scratch.above),
                    )
                {
                    assert_eq!(a.bits.to_bits(), b.bits.to_bits());
                    assert_eq!(a.penalty.to_bits(), b.penalty.to_bits());
                    assert_eq!(a.predictor, b.predictor);
                }
                assert_eq!(
                    actual_scratch.packed_bins,
                    if !cache && property == 1 { 17 } else { 0 }
                );
                if !cache && property == 0 && records >= 4096 {
                    assert_eq!(actual_scratch.sharded_predictors, [2]);
                } else if cache || property == 1 {
                    assert!(actual_scratch.sharded_predictors.is_empty());
                }
                if property != 3 {
                    assert!(!actual_scratch.sparse_predictors.is_empty());
                }
            }
        }
    }
}

#[test]
fn optional_dense_mapping_preserves_mass_and_exact_node_gates() {
    let mut parent = ParentHistogram {
        counts: vec![0; 64],
        extra_bits: 123,
    };
    for (i, symbol) in [0, 8, 16, 32, 63].into_iter().enumerate() {
        parent.counts[symbol] = (i + 1) as u32;
    }
    for lanes in [1, 4, 8] {
        let dense = compact_parent_histograms_with_layout(
            &[parent.clone()],
            lanes,
            DenseHistogram::Small,
            false,
        )
        .unwrap();
        assert_eq!(dense.parents[0].counts, [1, 2, 3, 4, 5]);
        assert_eq!(dense.parents[0].extra_bits, parent.extra_bits);
        for (expected, symbol) in [0, 8, 16, 32, 63].into_iter().enumerate() {
            assert_eq!(usize::from(dense.mapping[symbol]), expected);
        }
    }
    for records in [256, 257, 4096, 4097] {
        let (samples, mut options) = histogram_fixture(records, 3);
        let indices: Vec<_> = (0..records).collect();
        for mode in [
            DenseHistogram::Disabled,
            DenseHistogram::Small,
            DenseHistogram::Extended,
        ] {
            options.dense_histogram = mode;
            let stats = TrainingStats::default();
            find_best_split(
                &samples,
                &indices,
                0,
                &options,
                &mut None,
                &mut None,
                &HistogramBudget::new(0),
                &stats,
            )
            .unwrap();
            let limit = if mode == DenseHistogram::Extended {
                4096
            } else {
                256
            };
            assert_eq!(
                stats.compact_nodes.load(Ordering::Relaxed),
                usize::from(records <= limit)
            );
        }
    }
}

#[test]
fn optional_padded_cache_budget_paired_subtraction_and_predictor_masks() {
    let (mut samples, mut options) = histogram_fixture(8192, 5);
    for residuals in &mut samples.residuals {
        for residual in residuals {
            residual.token &= 127;
        }
    }
    samples.residuals[2] = samples.residuals[1].clone();
    options.parallel = false;
    options.padded_histogram = true;
    options.paired_cache = true;
    options.histogram_shards = true;
    options.predictor_dedup = true;
    let indices: Vec<_> = (0..8192).collect();
    let parents = parent_histograms(&samples, &indices);
    let stride = histogram_stride(128, DefaultScorer::new().lanes(), true);
    let measuring = HistogramBudget::new(32 * 1024 * 1024);
    let reserved =
        HistogramCache::reserve_with_stride(&samples, &parents, &measuring, Some(stride)).unwrap();
    let bytes = reserved._lease.bytes;
    assert!(bytes > expected_cache_bytes(&samples, &parents));
    drop(reserved);
    assert_eq!(measuring.used.load(Ordering::Relaxed), 0);
    assert!(
        HistogramCache::reserve_with_stride(
            &samples,
            &parents,
            &HistogramBudget::new(bytes - 1),
            Some(stride)
        )
        .is_none()
    );
    for records in [255, 256, 257, 4096, 7680] {
        let budget = HistogramBudget::new(bytes);
        let mut cache = None;
        find_best_split(
            &samples,
            &indices,
            0,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &TrainingStats::default(),
        )
        .unwrap();
        assert_cached_population(cache.as_ref().unwrap(), &samples, &indices);
        let pointers: Vec<_> = cache
            .as_ref()
            .unwrap()
            .properties
            .iter()
            .flat_map(|p| p.predictors.iter().map(|h| (h.counts.as_ptr(), h.symbols)))
            .collect();
        assert!(pointers.iter().all(|&(_, symbols)| symbols == stride));
        let begin = indices.len() - records;
        let stats = TrainingStats::default();
        let actual = find_best_split_with_scratch(
            &samples,
            &indices[begin..],
            0,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &stats,
            &ScratchPool::new(1),
            &indices[..begin],
        )
        .unwrap();
        assert_eq!(
            stats.paired_cache_subtractions.load(Ordering::Relaxed),
            if records >= 256 { 3 } else { 0 }
        );
        assert_cached_population(cache.as_ref().unwrap(), &samples, &indices[begin..]);
        let retained: Vec<_> = cache
            .as_ref()
            .unwrap()
            .properties
            .iter()
            .flat_map(|p| p.predictors.iter().map(|h| (h.counts.as_ptr(), h.symbols)))
            .collect();
        assert_eq!(pointers, retained);
        let expected = find_best_split(
            &samples,
            &indices[begin..],
            0,
            &options,
            &mut None,
            &mut None,
            &HistogramBudget::new(0),
            &TrainingStats::default(),
        )
        .unwrap();
        assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
        drop(cache);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

fn property_dedup_fixture(records: usize, predictors: usize) -> (TreeSamples, TrainOptions) {
    let (mut samples, mut options) = histogram_fixture(records, predictors);
    let values: Vec<_> = (0..records).map(|i| (i % 17) as u8).collect();
    let mut child_equal = values.clone();
    for value in child_equal.iter_mut().take(513) {
        *value = (*value + 1) % 17;
    }
    options.properties = vec![0, 6, 7, 8, 9, 15];
    samples.properties = vec![
        PropertyColumn::Wide((0..records).map(|i| (i % 3) as u32).collect()),
        PropertyColumn::Packed(values.clone()),
        PropertyColumn::Packed(values.clone()),
        PropertyColumn::Packed(child_equal),
        PropertyColumn::Packed(vec![0; records]),
        PropertyColumn::Packed(values),
    ];
    samples.thresholds = vec![
        vec![0, 1],
        (0..16).collect(),
        (0..16).collect(),
        (0..16).collect(),
        vec![0],
        (0..16).collect(),
    ];
    (samples, options)
}

#[test]
fn property_dedup_verifies_partitions_classes_and_wide_columns() {
    assert_eq!(property_signature(&[0, 2, 0, 1]), 0xbe7a597751656fde);
    assert_eq!(
        property_signature(&[0, 2, 0, 1]),
        property_signature(&[0, 2, 0, 1, 0])
    );
    let (mut samples, mut options) = histogram_fixture(96, 3);
    assert!(!options.property_dedup);
    options.properties = vec![0, 6, 15, 7, 1, 8, 9];
    let values: Vec<_> = (0..96).map(|i| (i % 3) as u8).collect();
    let mut different = values.clone();
    different.swap(0, 1);
    samples.properties = vec![
        PropertyColumn::Wide(values.iter().map(|&v| u32::from(v)).collect()),
        PropertyColumn::Packed(values.clone()),
        PropertyColumn::Packed(values.clone()),
        PropertyColumn::Packed(values.clone()),
        PropertyColumn::Wide(values.iter().map(|&v| u32::from(v)).collect()),
        PropertyColumn::Wide(values.iter().map(|&v| u32::from(v)).collect()),
        PropertyColumn::Packed(different),
    ];
    samples.thresholds = vec![vec![0, 1]; 7];
    // Different unused quantizer tails do not change a live partition.
    samples.thresholds[3] = vec![-9, -1, 8, 9];
    samples.thresholds[5] = (0..300).collect();
    let indices: Vec<_> = (0..96).rev().collect();
    let mut seen = Vec::new();
    for (property, expected) in [false, false, false, true, true, true, false]
        .into_iter()
        .enumerate()
    {
        let actual = with_property_values!(&samples.properties[property], |values| {
            let histogram =
                PropertyHistogram::collect_marginals_values(&samples, &indices, property, values);
            duplicate_property(
                &samples,
                &indices,
                &options,
                property,
                values,
                &histogram.used,
                &mut seen,
                &mut Vec::new(),
            )
        });
        assert_eq!(actual, expected, "property {property}");
    }
    assert_eq!(
        seen.iter().map(|p| p.property).collect::<Vec<_>>(),
        [0, 1, 2, 6]
    );
    // A matching signature (including a collision) never bypasses row verification.
    let mut collision = vec![SeenProperty {
        signature: ma_partition_proofs::signature(&[32, 32, 32]),
        property: 6,
    }];
    assert!(!duplicate_property(
        &samples,
        &indices,
        &options,
        1,
        &values,
        &[32, 32, 32],
        &mut collision,
        &mut Vec::new(),
    ));
    let live: Vec<_> = indices.into_iter().filter(|&i| i >= 2).collect();
    let mut seen = Vec::new();
    for (property, expected) in [(1, false), (6, true)] {
        let actual = with_property_values!(&samples.properties[property], |values| {
            let histogram =
                PropertyHistogram::collect_marginals_values(&samples, &live, property, values);
            duplicate_property(
                &samples,
                &live,
                &options,
                property,
                values,
                &histogram.used,
                &mut seen,
                &mut Vec::new(),
            )
        });
        assert_eq!(actual, expected);
    }
}

#[test]
fn property_dedup_obeys_record_and_actual_parallel_gates() {
    for workers in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for records in [63, 64, 1023, 1024] {
            let (samples, mut options) = property_dedup_fixture(records, 14);
            let indices: Vec<_> = (0..records).rev().collect();
            options.parallel = false;
            let expected = find_best_split(
                &samples,
                &indices,
                0,
                &options,
                &mut None,
                &mut None,
                &HistogramBudget::new(0),
                &TrainingStats::default(),
            )
            .unwrap();
            options.property_dedup = true;
            for (parallel, null_pool) in [(false, false), (true, false), (true, true)] {
                options.parallel = parallel;
                let mut mask = Vec::new();
                let stats = TrainingStats::default();
                let actual = pool
                    .install(|| {
                        let mut run = || {
                            find_best_split_with_properties(
                                &samples,
                                &indices,
                                0,
                                &options,
                                &mut None,
                                &mut None,
                                &HistogramBudget::new(0),
                                &stats,
                                &ScratchPool::new(workers),
                                &[],
                                &mut mask,
                            )
                        };
                        if null_pool {
                            execution::serial(run)
                        } else {
                            run()
                        }
                    })
                    .unwrap();
                let actual_parallel = parallel && !null_pool && workers > 1 && records >= 1024;
                let duplicate = usize::from(records >= 64 && !actual_parallel);
                assert_eq!(
                    stats.duplicate_properties.load(Ordering::Relaxed),
                    duplicate
                );
                assert_eq!(stats.constant_properties.load(Ordering::Relaxed), 1);
                assert_eq!(stats.inherited_properties.load(Ordering::Relaxed), 0);
                assert_eq!(mask, [0, 0, duplicate as u8, 0, 1, 0]);
                assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
            }
        }
    }
}

#[test]
fn property_dedup_invalidates_before_token_subtraction_and_inherits_masks() {
    let (mut samples, mut options) = property_dedup_fixture(8192, 3);
    samples.residuals[2] = samples.residuals[1].clone();
    options.predictor_dedup = true;
    options.property_dedup = true;
    options.parallel = false;
    let indices: Vec<_> = (0..8192).collect();
    let parents = parent_histograms(&samples, &indices);
    let bytes = expected_cache_bytes(&samples, &parents);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for child_parallel in [false, true] {
        let budget = HistogramBudget::new(bytes);
        let scratch = ScratchPool::new(3);
        let mut cache = None;
        let mut mask = Vec::new();
        let mut previous = 0;
        let mut retained_tokens = None;
        let mut retained_pointer = None;
        for begin in [0, 513, 777] {
            options.parallel = begin != 0 && child_parallel;
            let stats = TrainingStats::default();
            let actual = pool
                .install(|| {
                    find_best_split_with_properties(
                        &samples,
                        &indices[begin..],
                        0,
                        &options,
                        &mut None,
                        &mut cache,
                        &budget,
                        &stats,
                        &scratch,
                        &indices[previous..begin],
                        &mut mask,
                    )
                })
                .unwrap();
            let new_duplicate = begin != 0 && !child_parallel;
            assert_eq!(mask, [0, 0, 1, u8::from(new_duplicate), 1, 0]);
            assert_eq!(
                stats.inherited_properties.load(Ordering::Relaxed),
                if begin == 0 {
                    0
                } else if begin == 777 && !child_parallel {
                    3
                } else {
                    2
                }
            );
            assert_eq!(budget.used.load(Ordering::Relaxed), bytes);
            let cached = cache.as_ref().unwrap();
            assert!(!cached.properties[2].valid);
            assert!(cached.properties[2].predictors.is_empty());
            assert!(!cached.properties[4].valid);
            let property = &cached.properties[3];
            if begin == 0 {
                retained_tokens = Some(property.predictors[0].counts.clone());
                retained_pointer = Some(property.predictors[0].counts.as_ptr());
            } else if new_duplicate {
                assert!(!property.valid);
                assert_eq!(
                    &property.predictors[0].counts,
                    retained_tokens.as_ref().unwrap()
                );
            }
            assert_eq!(
                property.predictors[0].counts.as_ptr(),
                retained_pointer.unwrap()
            );
            let live_parents = parent_histograms(&samples, &indices[begin..]);
            for (p, &redundant) in mask.iter().enumerate() {
                if redundant != 0 {
                    continue;
                }
                let fresh =
                    PropertyHistogram::collect(&samples, &indices[begin..], &live_parents, p);
                let actual = &cached.properties[p];
                assert!(actual.valid);
                assert_eq!(actual.used, fresh.used);
                assert_eq!(actual.weights, fresh.weights);
                for (a, b) in actual.predictors.iter().zip(&fresh.predictors) {
                    assert_eq!(a.extra, b.extra);
                    for (a, b) in a
                        .counts
                        .chunks_exact(a.symbols)
                        .zip(b.counts.chunks_exact(b.symbols))
                    {
                        assert_eq!(&a[..b.len()], b);
                        assert!(a[b.len()..].iter().all(|&n| n == 0));
                    }
                }
            }
            let mut reference = options.clone();
            reference.property_dedup = false;
            let expected = find_best_split(
                &samples,
                &indices[begin..],
                0,
                &reference,
                &mut None,
                &mut None,
                &HistogramBudget::new(0),
                &TrainingStats::default(),
            )
            .unwrap();
            assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
            previous = begin;
        }
        drop(cache);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn property_dedup_errors_and_unwinds_do_not_publish_partial_proofs() {
    let (samples, mut options) = property_dedup_fixture(8192, 3);
    options.parallel = false;
    options.property_dedup = true;
    options.predictor_dedup = true;
    let indices: Vec<_> = (0..8192).collect();
    let parents = parent_histograms(&samples, &indices);
    let bytes = expected_cache_bytes(&samples, &parents);
    for panic in [false, true] {
        let budget = HistogramBudget::new(bytes);
        let scratch = ScratchPool::new(1);
        let mut cache = None;
        let mut mask = Vec::new();
        let stats = TrainingStats {
            initial_cache_hook: Some(Box::new(move |property, stage| {
                if property == 5 && stage == InitialCacheStage::Scored {
                    assert!(!panic, "injected property-proof unwind");
                    return Err(Error::Encoding("injected property-proof error".into()));
                }
                Ok(())
            })),
            ..TrainingStats::default()
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            find_best_split_with_properties(
                &samples,
                &indices,
                0,
                &options,
                &mut None,
                &mut cache,
                &budget,
                &stats,
                &scratch,
                &[],
                &mut mask,
            )
        }));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        assert!(mask.is_empty());
        assert!(cache.is_none());
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        assert!(!scratch.idle.lock().unwrap().is_empty());
        find_best_split_with_properties(
            &samples,
            &indices,
            0,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &TrainingStats::default(),
            &scratch,
            &[],
            &mut mask,
        )
        .unwrap();
        assert_eq!(mask, [0, 0, 1, 0, 1, 0]);
        let inherited = mask.clone();
        cache.as_mut().unwrap().properties[5].used[0] = 0;
        assert!(
            find_best_split_with_properties(
                &samples,
                &indices[513..],
                0,
                &options,
                &mut None,
                &mut cache,
                &budget,
                &TrainingStats::default(),
                &scratch,
                &indices[..513],
                &mut mask
            )
            .is_err()
        );
        assert_eq!(mask, inherited);
        assert!(cache.is_none());
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn property_dedup_preserves_full_search_with_predictor_reuse_and_column_swaps() {
    let (mut input, mut options) = reuse_fixture();
    options.predictors.push(Predictor::Top);
    options.properties.extend([7, 8]);
    for sample in &mut input {
        sample.residuals.push(sample.residuals[1]);
        sample.properties[7] = sample.properties[6];
        sample.properties[8] = 0;
    }
    let prepared = TreeSamples::new(&input, &options).unwrap();
    let baseline = train(&input, &options).unwrap();
    options.property_dedup = true;
    options.predictor_dedup = true;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for (parallel, null_pool) in [(false, false), (true, false), (true, true)] {
        options.parallel = parallel;
        for bytes in [0, 32 * 1024 * 1024] {
            options.histogram_cache_bytes = bytes;
            for reorder in [false, true] {
                let stats = TrainingStats::default();
                let run = || {
                    let scratch = ScratchPool::new(6);
                    if reorder {
                        train_nodes_with_layout::<true>(
                            &mut prepared.clone(),
                            &options,
                            &stats,
                            &scratch,
                        )
                    } else {
                        train_nodes_with_layout::<false>(
                            &mut prepared.clone(),
                            &options,
                            &stats,
                            &scratch,
                        )
                    }
                };
                let actual = pool
                    .install(|| {
                        if null_pool {
                            execution::serial(run)
                        } else {
                            run()
                        }
                    })
                    .unwrap();
                let actual = Tree::new(actual).unwrap();
                assert_eq!(signature(&actual), signature(&baseline));
                assert_eq!(
                    estimated_cost(&actual, &input, &options),
                    estimated_cost(&baseline, &input, &options)
                );
                assert!(stats.constant_properties.load(Ordering::Relaxed) > 0);
                assert!(stats.inherited_properties.load(Ordering::Relaxed) > 0);
            }
        }
    }
    options.max_leaves = 1;
    assert!(matches!(train(&input, &options), Err(Error::Limit(_))));
    options.property_dedup = false;
    assert!(matches!(train(&input, &options), Err(Error::Limit(_))));
}

fn wide_columns(samples: &TreeSamples) -> TreeSamples {
    let mut wide = samples.clone();
    for column in &mut wide.properties {
        *column = PropertyColumn::Wide(column.to_wide());
    }
    wide
}

#[test]
fn packed_preparation_preserves_strict_boundaries_static_ids_and_dedup_weights() {
    for property in [0, 1, 6, 15] {
        for thresholds in [Vec::new(), (-254..0).collect(), (-255..0).collect()] {
            let mut options = options(property);
            options.property_quantization = Some(PropertyQuantization {
                properties: vec![property],
                thresholds: vec![thresholds.clone()],
            });
            options.validate().unwrap();
            let input: Vec<_> = [
                i32::MIN,
                -512,
                -511,
                -256,
                -255,
                -254,
                -2,
                -1,
                0,
                1,
                511,
                512,
                i32::MAX,
            ]
            .into_iter()
            .map(|value| sample(property, value, &[0, 1], 70_000))
            .collect();
            let prepared = TreeSamples::new(&input, &options).unwrap();
            let mut expected_values = Vec::new();
            let mut expected_counts = Vec::new();
            for sample in &input {
                let value = if property < 2 {
                    sample.properties[property]
                } else {
                    sample.properties[property].clamp(-511, 511)
                };
                let bin = thresholds
                    .iter()
                    .filter(|&&threshold| value > threshold)
                    .count() as u32;
                if let Some(index) = expected_values.iter().position(|&value| value == bin) {
                    expected_counts[index] += sample.weight;
                } else {
                    expected_values.push(bin);
                    expected_counts.push(sample.weight);
                }
            }
            for (&bin, &weight) in expected_values.iter().zip(&expected_counts) {
                assert_eq!(
                    prepared
                        .counts
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| prepared.properties[0].value(i) == bin)
                        .map(|(_, &count)| u32::from(count))
                        .sum::<u32>(),
                    weight
                );
            }
            assert!(
                prepared.properties[0]
                    .to_wide()
                    .iter()
                    .all(|bin| expected_values.contains(bin))
            );
            assert_eq!(prepared.thresholds[0], thresholds);
            assert_eq!(
                matches!(prepared.properties[0], PropertyColumn::Packed(_)),
                property >= 2
            );
        }
    }
    // Storage follows property IDs, not a static-prefix/order assumption.
    let mut options = options(6);
    options.properties = vec![6, 1, 7, 0];
    let prepared = TreeSamples::new(&[sample(6, 1, &[0, 1], 1)], &options).unwrap();
    assert!(matches!(prepared.properties[0], PropertyColumn::Packed(_)));
    assert!(matches!(prepared.properties[1], PropertyColumn::Wide(_)));
    assert!(matches!(prepared.properties[2], PropertyColumn::Packed(_)));
    assert!(matches!(prepared.properties[3], PropertyColumn::Wide(_)));
}

#[test]
fn packed_columns_match_wide_histograms_costs_and_repeated_subtraction() {
    let mut actual_scratch = PropertyScratch::default();
    let mut wide_scratch = PropertyScratch::default();
    for records in [255, 256, 257, 8192] {
        let (samples, options) = histogram_fixture(records, 3);
        let wide = wide_columns(&samples);
        let indices: Vec<_> = (0..records).rev().collect();
        let parents = parent_histograms(&samples, &indices);
        let budget = HistogramBudget::new(64 * 1024 * 1024);
        let mut cache =
            HistogramCache::create(&samples, &indices, &parents, &budget, false).unwrap();
        let mut wide_cache =
            HistogramCache::create(&wide, &indices, &parents, &budget, false).unwrap();
        let mut previous = 0;
        for begin in [0, 17, 33] {
            cache.subtract(&samples, &indices[previous..begin]).unwrap();
            wide_cache
                .subtract(&wide, &indices[previous..begin])
                .unwrap();
            let live = &indices[begin..];
            assert_cached_population(&cache, &samples, live);
            assert_cached_population(&wide_cache, &wide, live);
            let parents = parent_histograms(&samples, live);
            let compact =
                compact_parent_histograms(&parents, DefaultScorer::new().lanes()).unwrap();
            let total = live.iter().map(|&i| u32::from(samples.counts[i])).sum();
            for property in 0..options.properties.len() {
                for mode in 0..3 {
                    let (parents, mapping) = if mode == 2 {
                        (compact.parents.as_slice(), Some(&compact.mapping))
                    } else {
                        (parents.as_slice(), None)
                    };
                    let actual = evaluate_property_cached(
                        &samples,
                        live,
                        &options,
                        parents,
                        0,
                        total,
                        property,
                        (mode == 1).then_some(&cache.properties[property]),
                        mapping,
                        &mut actual_scratch,
                    )
                    .unwrap();
                    let expected = evaluate_property_cached(
                        &wide,
                        live,
                        &options,
                        parents,
                        0,
                        total,
                        property,
                        (mode == 1).then_some(&wide_cache.properties[property]),
                        mapping,
                        &mut wide_scratch,
                    )
                    .unwrap();
                    assert_same_candidates(actual, expected);
                    for (actual, expected) in actual_scratch
                        .below
                        .iter()
                        .zip(&wide_scratch.below)
                        .chain(actual_scratch.above.iter().zip(&wide_scratch.above))
                    {
                        assert_eq!(actual.bits.to_bits(), expected.bits.to_bits());
                        assert_eq!(actual.penalty.to_bits(), expected.penalty.to_bits());
                        assert_eq!(actual.predictor, expected.predictor);
                    }
                }
            }
            previous = begin;
        }
        drop((cache, wide_cache));
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn wide_dynamic_columns_preserve_legacy_quantizers_above_256_bins() {
    let mut options = options(6);
    // Exercise the internal legacy preparation helper without broadening
    // public TrainOptions validation or truncating its resulting quantizer.
    options.max_property_values = 513;
    assert!(options.validate().is_err());
    let input: Vec<_> = (-511..=511)
        .map(|value| sample(6, value, &[if value <= 0 { 0 } else { 2000 }, 1], 1))
        .collect();
    let samples = TreeSamples::new(&input, &options).unwrap();
    assert!(samples.thresholds[0].len() >= 256);
    assert!(matches!(samples.properties[0], PropertyColumn::Wide(_)));
    let bins = samples.properties[0].to_wide();
    assert!(bins.iter().any(|&bin| bin > 255));
    let mut expected = HashMap::<SampleKey, u64>::new();
    let config = HybridUintConfig::new(4, 1, 2).unwrap();
    for sample in &input {
        let bin = samples.thresholds[0]
            .iter()
            .filter(|&&threshold| sample.properties[6] > threshold)
            .count() as u32;
        let key = SampleKey {
            properties: vec![bin],
            residuals: sample
                .residuals
                .iter()
                .map(|&residual| {
                    let token = config.encode(pack_signed(residual));
                    ResidualToken {
                        token: token.token as u8,
                        nbits: token.nbits as u8,
                    }
                })
                .collect(),
        };
        *expected.entry(key).or_default() += u64::from(sample.weight);
    }
    assert_eq!(prepared_distribution(&samples), expected);
    assert_eq!(
        layout_signature::<true>(&samples, &options).unwrap(),
        layout_signature::<false>(&samples, &options).unwrap()
    );
    let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
    let parents = parent_histograms(&samples, &indices);
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let mut cache = HistogramCache::create(&samples, &indices, &parents, &budget, false).unwrap();
    cache.subtract(&samples, &indices[..129]).unwrap();
    assert_cached_population(&cache, &samples, &indices[129..]);
}

#[test]
fn packed_columns_match_wide_complete_search_and_fused_cache_lifetimes() {
    let (input, mut options) = reuse_fixture();
    let samples = TreeSamples::new(&input, &options).unwrap();
    let wide = wide_columns(&samples);
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let cache_bytes = options.histogram_cache_bytes;
    for parallel in [false, true] {
        options.parallel = parallel;
        for bytes in [0, cache_bytes] {
            options.histogram_cache_bytes = bytes;
            workers.install(|| {
                let actual_stats = TrainingStats::default();
                let wide_stats = TrainingStats::default();
                let actual = Tree::new(
                    train_nodes_with_scratch(
                        &mut samples.clone(),
                        &options,
                        &actual_stats,
                        &ScratchPool::new(6),
                    )
                    .unwrap(),
                )
                .unwrap();
                let expected = Tree::new(
                    train_nodes_with_scratch(
                        &mut wide.clone(),
                        &options,
                        &wide_stats,
                        &ScratchPool::new(6),
                    )
                    .unwrap(),
                )
                .unwrap();
                assert_eq!(signature(&actual), signature(&expected));
                for (actual, expected) in [
                    (&actual_stats.cache_builds, &wide_stats.cache_builds),
                    (&actual_stats.cache_reuses, &wide_stats.cache_reuses),
                    (
                        &actual_stats.cache_subtractions,
                        &wide_stats.cache_subtractions,
                    ),
                ] {
                    assert_eq!(
                        actual.load(Ordering::Relaxed),
                        expected.load(Ordering::Relaxed)
                    );
                }
                if bytes != 0 {
                    assert!(actual_stats.cache_builds.load(Ordering::Relaxed) > 0);
                    assert!(actual_stats.cache_subtractions.load(Ordering::Relaxed) > 0);
                }
                assert_eq!(
                    signature(&actual),
                    layout_signature::<false>(&wide, &options).unwrap()
                );
            });
        }
    }
}

#[test]
fn compact_histograms_preserve_lane_order_union_and_source_savings_gate() {
    let (samples, _) = histogram_fixture(127, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    for lanes in [1, 4, 8] {
        let compact = compact_parent_histograms(&parents, lanes).unwrap();
        for (original, packed) in parents.iter().zip(&compact.parents) {
            assert_eq!(original.extra_bits, packed.extra_bits);
            assert_eq!(
                original.counts.iter().sum::<u32>(),
                packed.counts.iter().sum::<u32>()
            );
            for lane in 0..lanes {
                let lane_counts = |histogram: &ParentHistogram| -> Vec<u32> {
                    histogram
                        .counts
                        .iter()
                        .enumerate()
                        .filter(|&(symbol, &count)| symbol % lanes == lane && count != 0)
                        .map(|(_, &count)| count)
                        .collect()
                };
                assert_eq!(lane_counts(original), lane_counts(packed));
            }
            for (symbol, &count) in original.counts.iter().enumerate() {
                if count != 0 {
                    let mapped = usize::from(compact.mapping[symbol]);
                    assert_eq!(symbol % lanes, mapped % lanes);
                    assert_eq!(count, packed.counts[mapped]);
                }
            }
        }
        let mut boundary = ParentHistogram {
            counts: vec![1; 32],
            extra_bits: 9,
        };
        boundary.counts[..8].fill(0);
        assert!(compact_parent_histograms(&[boundary.clone()], lanes).is_some());
        boundary.counts[0] = 1;
        assert!(compact_parent_histograms(&[boundary], lanes).is_none());
        assert!(
            compact_parent_histograms(
                &[ParentHistogram {
                    counts: vec![0; 256],
                    extra_bits: 0,
                }],
                lanes
            )
            .is_none()
        );
    }
    assert!(compact_parent_histograms(&[], 8).is_none());
    assert!(compact_parent_histograms(&parents, 0).is_none());
    assert!(
        compact_parent_histograms(
            &[ParentHistogram {
                counts: vec![1; 257],
                extra_bits: 0,
            }],
            8
        )
        .is_none()
    );
}

#[test]
fn reused_scratch_paired_compact_and_cached_sweeps_match_every_cost() {
    let mut scratch = PropertyScratch::default();
    // Cross the pair/compaction boundaries, odd-predictor tails, shrinking
    // strides and constant properties without replacing worker scratch.
    for (records, predictors) in [
        (8192, 5),
        (255, 3),
        (255, 2),
        (256, 2),
        (17, 2),
        (256, 3),
        (31, 14),
        (256, 14),
        (1024, 14),
        (17, 1),
        (257, 2),
    ] {
        let (mut samples, options) = histogram_fixture(records, predictors);
        if predictors == 2 && records <= 256 {
            for count in samples.counts.iter_mut().step_by(7) {
                *count = u16::MAX;
            }
        }
        let indices: Vec<_> = (0..records).rev().collect();
        let parents = parent_histograms(&samples, &indices);
        let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
        let budget = HistogramBudget::new(32 * 1024 * 1024);
        let cache = HistogramCache::create(&samples, &indices, &parents, &budget, false).unwrap();
        let compact = compact_parent_histograms(&parents, DefaultScorer::new().lanes()).unwrap();
        for property in 0..options.properties.len() {
            let mut reference = PropertyScratch::default();
            let expected = evaluate_property_cached(
                &samples,
                &indices,
                &options,
                &parents,
                0,
                total,
                property,
                Some(&cache.properties[property]),
                None,
                &mut reference,
            )
            .unwrap();
            for packed in [false, true] {
                let (scoring_parents, mapping) = if packed {
                    (compact.parents.as_slice(), Some(&compact.mapping))
                } else {
                    (parents.as_slice(), None)
                };
                let actual = evaluate_property_cached(
                    &samples,
                    &indices,
                    &options,
                    scoring_parents,
                    0,
                    total,
                    property,
                    None,
                    mapping,
                    &mut scratch,
                )
                .unwrap();
                assert_same_candidates(actual, expected);
                if packed && predictors == 2 && records <= 256 && property != 3 {
                    // Predictor one consumes the prepared pair by swapping;
                    // the retained companion now holds predictor zero.
                    let bins = samples.thresholds[property].len() + 1;
                    assert_eq!(
                        scratch.paired_increase.len(),
                        bins * compact.parents[0].counts.len(),
                    );
                    assert_eq!(scratch.paired_extra.len(), bins);
                }
                for (a, b) in scratch
                    .below
                    .iter()
                    .zip(&reference.below)
                    .chain(scratch.above.iter().zip(&reference.above))
                {
                    assert_eq!(a.bits.to_bits(), b.bits.to_bits());
                    assert_eq!(a.penalty.to_bits(), b.penalty.to_bits());
                    assert_eq!(a.predictor, b.predictor);
                }
            }
        }
        assert!(cache.properties[3].predictors.is_empty());
    }
}

#[test]
fn leaf_termination_skips_other_predictors_and_reuses_known_parents() {
    let (mut samples, options) = histogram_fixture(8192, 9);
    samples.residuals[2].fill(ResidualToken { token: 0, nbits: 0 });
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let stats = TrainingStats::default();
    let mut parents = None;
    let mut cache = None;
    assert!(
        find_best_split(
            &samples,
            &indices,
            2,
            &options,
            &mut parents,
            &mut cache,
            &budget,
            &stats,
        )
        .unwrap()
        .is_none()
    );
    assert!(parents.is_none());
    assert!(cache.is_none());
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    let expected = parent_histograms(&samples, &indices);
    parents = Some(expected.clone());
    assert!(
        find_best_split(
            &samples,
            &indices,
            2,
            &options,
            &mut parents,
            &mut cache,
            &budget,
            &stats,
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(parents.as_ref().unwrap(), &expected);
    assert_eq!(parent_histogram(&samples, &[], 0).counts, [0]);
    for parallel in [false, true] {
        let options = TrainOptions {
            parallel,
            ..options.clone()
        };
        parents = None;
        let _ = find_best_split(
            &samples,
            &indices,
            1,
            &options,
            &mut parents,
            &mut cache,
            &budget,
            &stats,
        )
        .unwrap();
        assert_eq!(parents.as_ref().unwrap(), &expected);
    }
}

#[test]
fn small_node_compaction_keeps_decisions_and_raw_parent_histograms() {
    let (samples, mut options) = histogram_fixture(127, 3);
    options.parallel = false;
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let raw = parent_histograms(&samples, &indices);
    let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
    let base = DefaultScorer::new().estimate_bits(&raw[0].counts, total) + raw[0].extra_bits as f32;
    assert!(compact_parent_histograms(&raw, DefaultScorer::new().lanes()).is_some());
    let mut winners = [None; 4];
    for property in 0..options.properties.len() {
        let candidates =
            evaluate_property(&samples, &indices, &options, &raw, 0, total, property).unwrap();
        for (best, candidate) in winners.iter_mut().zip(candidates) {
            if let Some(candidate) = candidate {
                keep_better(best, candidate);
            }
        }
    }
    let expected = choose_split(winners, base, &options);
    let mut parents = None;
    let mut cache = None;
    let actual = find_best_split(
        &samples,
        &indices,
        0,
        &options,
        &mut parents,
        &mut cache,
        &HistogramBudget::new(options.histogram_cache_bytes),
        &TrainingStats::default(),
    )
    .unwrap();
    assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
    assert_eq!(parents.unwrap(), raw);
    assert!(cache.is_none());
}

#[test]
fn parallel_cache_population_and_source_creation_gate_preserve_counts() {
    let (samples, options) = histogram_fixture(8192, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let budget = HistogramBudget::new(32 * 1024 * 1024);
    let serial = HistogramCache::create(&samples, &indices, &parents, &budget, false).unwrap();
    let mut parallel = HistogramCache::create(&samples, &indices, &parents, &budget, true).unwrap();
    for (a, b) in serial.properties.iter().zip(&parallel.properties) {
        assert_eq!(a.used, b.used);
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.predictors.len(), b.predictors.len());
        for (a, b) in a.predictors.iter().zip(&b.predictors) {
            assert_eq!(a.symbols, b.symbols);
            assert_eq!(a.counts, b.counts);
            assert_eq!(a.extra, b.extra);
        }
    }
    parallel.subtract(&samples, &indices[..513]).unwrap();
    assert!(parallel.properties[3].predictors.is_empty());
    assert_eq!(
        parallel.properties[3].used.iter().sum::<usize>(),
        8192 - 513
    );
    assert_eq!(
        parallel.properties[3].weights.iter().sum::<u32>(),
        samples.counts[513..]
            .iter()
            .map(|&n| u32::from(n))
            .sum::<u32>()
    );
    for records in [4096, 8191, 8192] {
        let mut parents = None;
        let mut cache = None;
        let stats = TrainingStats::default();
        let _ = find_best_split(
            &samples,
            &indices[..records],
            0,
            &options,
            &mut parents,
            &mut cache,
            &budget,
            &stats,
        )
        .unwrap();
        assert_eq!(cache.is_some(), records >= 8192);
    }
}

#[test]
fn cache_reservation_precedes_population_and_preserves_exact_budget() {
    let (samples, options) = histogram_fixture(8192, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let bytes = expected_cache_bytes(&samples, &parents);
    let budget = HistogramBudget::new(bytes);
    let reserved = HistogramCache::reserve(&samples, &parents, &budget).unwrap();
    assert_eq!(reserved._lease.bytes, bytes);
    assert_eq!(budget.used.load(Ordering::Relaxed), bytes);
    assert_eq!(budget.peak.load(Ordering::Relaxed), bytes);
    assert_eq!(reserved.properties.len(), options.properties.len());
    assert!(
        reserved
            .properties
            .iter()
            .all(|h| { h.used.is_empty() && h.weights.is_empty() && h.predictors.is_empty() })
    );
    assert!(HistogramCache::reserve(&samples, &parents, &budget).is_none());
    drop(reserved);
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    let mut expected = None;
    for limit in [0, bytes - 1] {
        let denied = HistogramBudget::new(limit);
        let stats = TrainingStats {
            initial_cache_hook: Some(Box::new(|_, _| panic!("denied cache was populated"))),
            ..TrainingStats::default()
        };
        let mut cache = None;
        let actual = find_best_split(
            &samples,
            &indices,
            0,
            &options,
            &mut Some(parents.clone()),
            &mut cache,
            &denied,
            &stats,
        )
        .unwrap();
        if let Some(expected) = expected {
            assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
        } else {
            expected = Some(actual);
        }
        assert!(cache.is_none());
        assert_eq!(stats.cache_builds.load(Ordering::Relaxed), 0);
        assert_eq!(denied.used.load(Ordering::Relaxed), 0);
        assert_eq!(denied.peak.load(Ordering::Relaxed), 0);
    }
}

fn expected_cache_bytes(samples: &TreeSamples, parents: &[ParentHistogram]) -> usize {
    std::mem::size_of::<HistogramCache>()
        + samples
            .thresholds
            .iter()
            .map(|thresholds| {
                let bins = thresholds.len() + 1;
                std::mem::size_of::<PropertyHistogram>()
                    + bins * (std::mem::size_of::<usize>() + std::mem::size_of::<u32>())
                    + parents
                        .iter()
                        .map(|parent| {
                            std::mem::size_of::<BinHistogram>()
                                + bins
                                    * (parent.counts.len() * std::mem::size_of::<u32>()
                                        + std::mem::size_of::<u64>())
                        })
                        .sum::<usize>()
            })
            .sum::<usize>()
}

#[test]
fn initial_cache_fill_and_score_share_workers_and_match_eager_then_deferred_cache() {
    let (samples, mut options) = histogram_fixture(8192, 3);
    options.parallel = false;
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let bytes = expected_cache_bytes(&samples, &parents);
    let eager_budget = HistogramBudget::new(bytes);
    let mut eager = HistogramCache::create(&samples, &indices, &parents, &eager_budget, false);
    let expected = find_best_split(
        &samples,
        &indices,
        0,
        &options,
        &mut Some(parents.clone()),
        &mut eager,
        &eager_budget,
        &TrainingStats::default(),
    )
    .unwrap();
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for (parallel, null_pool) in [(false, false), (true, false), (true, true)] {
        options.parallel = parallel;
        let budget = HistogramBudget::new(bytes);
        let events = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&events);
        let observed_budget = Arc::clone(&budget);
        let stats = TrainingStats {
            initial_cache_hook: Some(Box::new(move |property, stage| {
                assert_eq!(observed_budget.used.load(Ordering::Relaxed), bytes);
                recorded
                    .lock()
                    .unwrap()
                    .push((std::thread::current().id(), property, stage));
                Ok(())
            })),
            ..TrainingStats::default()
        };
        let scratch = ScratchPool::new(3);
        let mut cache = None;
        let mut current_parents = Some(parents.clone());
        let actual = workers
            .install(|| {
                let mut run = || {
                    find_best_split_with_scratch(
                        &samples,
                        &indices,
                        0,
                        &options,
                        &mut current_parents,
                        &mut cache,
                        &budget,
                        &stats,
                        &scratch,
                        &[],
                    )
                };
                if null_pool {
                    execution::serial(run)
                } else {
                    run()
                }
            })
            .unwrap();
        assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
        assert_cached_population(cache.as_ref().unwrap(), &samples, &indices);
        assert!(cache.as_ref().unwrap().properties[3].predictors.is_empty());
        assert_eq!(stats.cache_builds.load(Ordering::Relaxed), 1);
        assert_eq!(stats.cache_reuses.load(Ordering::Relaxed), 0);
        assert_eq!(stats.cache_subtractions.load(Ordering::Relaxed), 0);
        assert_eq!(budget.peak.load(Ordering::Relaxed), bytes);
        let mut by_worker = HashMap::<_, Vec<_>>::new();
        let recorded = events.lock().unwrap();
        assert_eq!(recorded.len(), 2 * options.properties.len());
        for &(thread, property, stage) in recorded.iter() {
            by_worker.entry(thread).or_default().push((property, stage));
        }
        if !parallel || null_pool {
            assert_eq!(by_worker.len(), 1);
            assert_eq!(scratch.created.load(Ordering::Relaxed), 1);
        }
        for events in by_worker.values() {
            assert_eq!(events.len() % 2, 0);
            for pair in events.chunks_exact(2) {
                assert_eq!(pair[0].1, InitialCacheStage::Filled);
                assert_eq!(pair[1], (pair[0].0, InitialCacheStage::Scored));
            }
        }
        drop(recorded);
        let inherited_stats = TrainingStats {
            initial_cache_hook: Some(Box::new(|_, _| panic!("inherited cache was refilled"))),
            ..TrainingStats::default()
        };
        current_parents = None;
        let child = workers
            .install(|| {
                let mut run = || {
                    find_best_split_with_scratch(
                        &samples,
                        &indices[513..],
                        0,
                        &options,
                        &mut current_parents,
                        &mut cache,
                        &budget,
                        &inherited_stats,
                        &scratch,
                        &indices[..513],
                    )
                };
                if null_pool {
                    execution::serial(run)
                } else {
                    run()
                }
            })
            .unwrap();
        let fresh = find_best_split(
            &samples,
            &indices[513..],
            0,
            &options,
            &mut None,
            &mut None,
            &HistogramBudget::new(0),
            &TrainingStats::default(),
        )
        .unwrap();
        assert_same_candidates([child, None, None, None], [fresh, None, None, None]);
        assert_cached_population(cache.as_ref().unwrap(), &samples, &indices[513..]);
        assert_eq!(inherited_stats.cache_builds.load(Ordering::Relaxed), 0);
        assert_eq!(inherited_stats.cache_reuses.load(Ordering::Relaxed), 1);
        assert_eq!(
            inherited_stats.cache_subtractions.load(Ordering::Relaxed),
            options.properties.len()
        );
        drop(cache);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn serial_candidate_reduction_drains_multiple_errors_in_property_order() {
    let (samples, mut options) = histogram_fixture(8192, 3);
    options.parallel = false;
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let budget = HistogramBudget::new(expected_cache_bytes(&samples, &parents));
    let visited = Arc::new(AtomicUsize::new(0));
    let recorded = Arc::clone(&visited);
    let stats = TrainingStats {
        initial_cache_hook: Some(Box::new(move |property, stage| {
            if stage == InitialCacheStage::Filled {
                recorded.fetch_add(1, Ordering::Relaxed);
                if property != 0 {
                    return Err(Error::Encoding(format!("failed property {property}")));
                }
            }
            Ok(())
        })),
        ..TrainingStats::default()
    };
    let scratch = ScratchPool::new(1);
    let mut cache = None;
    let result = find_best_split_with_scratch(
        &samples,
        &indices,
        0,
        &options,
        &mut Some(parents),
        &mut cache,
        &budget,
        &stats,
        &scratch,
        &[],
    );
    assert!(matches!(result, Err(Error::Encoding(message)) if message == "failed property 1"));
    assert_eq!(visited.load(Ordering::Relaxed), options.properties.len());
    assert!(cache.is_none());
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    assert_eq!(scratch.idle.lock().unwrap().len(), 1);
}

#[test]
fn initial_cache_errors_discard_partial_work_and_allow_retry() {
    let (mut samples, mut options) = histogram_fixture(8192, 3);
    options.predictor_dedup = true;
    samples.residuals[2] = samples.residuals[1].clone();
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let bytes = expected_cache_bytes(&samples, &parents);
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for parallel in [false, true] {
        options.parallel = parallel;
        for fail_stage in [InitialCacheStage::Filled, InitialCacheStage::Scored] {
            let budget = HistogramBudget::new(bytes);
            let filled = Arc::new(AtomicUsize::new(0));
            let recorded = Arc::clone(&filled);
            let stats = TrainingStats {
                initial_cache_hook: Some(Box::new(move |property, stage| {
                    if stage == InitialCacheStage::Filled {
                        recorded.fetch_add(1, Ordering::Relaxed);
                    }
                    if property == 1 && stage == fail_stage {
                        return Err(Error::Encoding("injected initial-cache failure".into()));
                    }
                    Ok(())
                })),
                ..TrainingStats::default()
            };
            let scratch = ScratchPool::new(3);
            let mut cache = None;
            let mut current_parents = Some(parents.clone());
            let result = workers.install(|| {
                find_best_split_with_scratch(
                    &samples,
                    &indices,
                    0,
                    &options,
                    &mut current_parents,
                    &mut cache,
                    &budget,
                    &stats,
                    &scratch,
                    &[],
                )
            });
            assert!(matches!(result, Err(Error::Encoding(_))));
            assert_eq!(filled.load(Ordering::Relaxed), options.properties.len());
            assert!(cache.is_none());
            assert_eq!(budget.used.load(Ordering::Relaxed), 0);
            assert_eq!(budget.peak.load(Ordering::Relaxed), bytes);
            assert!(!scratch.idle.lock().unwrap().is_empty());
            let actual = workers
                .install(|| {
                    find_best_split_with_scratch(
                        &samples,
                        &indices,
                        0,
                        &options,
                        &mut current_parents,
                        &mut cache,
                        &budget,
                        &TrainingStats::default(),
                        &scratch,
                        &[],
                    )
                })
                .unwrap();
            assert_cached_population(cache.as_ref().unwrap(), &samples, &indices);
            let expected = find_best_split(
                &samples,
                &indices,
                0,
                &options,
                &mut None,
                &mut None,
                &HistogramBudget::new(0),
                &TrainingStats::default(),
            )
            .unwrap();
            assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
            drop(cache);
            assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn initial_cache_unwind_releases_budget_and_private_scratch() {
    let (mut samples, mut options) = histogram_fixture(8192, 3);
    options.predictor_dedup = true;
    samples.residuals[2] = samples.residuals[1].clone();
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let bytes = expected_cache_bytes(&samples, &parents);
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for parallel in [false, true] {
        options.parallel = parallel;
        let budget = HistogramBudget::new(bytes);
        let stats = TrainingStats {
            initial_cache_hook: Some(Box::new(|property, stage| {
                if property == 1 && stage == InitialCacheStage::Filled {
                    panic!("injected initial-cache unwind");
                }
                Ok(())
            })),
            ..TrainingStats::default()
        };
        let scratch = ScratchPool::new(3);
        let mut cache = None;
        let mut current_parents = Some(parents.clone());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            workers.install(|| {
                find_best_split_with_scratch(
                    &samples,
                    &indices,
                    0,
                    &options,
                    &mut current_parents,
                    &mut cache,
                    &budget,
                    &stats,
                    &scratch,
                    &[],
                )
            })
        }));
        assert!(result.is_err());
        assert!(cache.is_none());
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        assert_eq!(budget.peak.load(Ordering::Relaxed), bytes);
        assert!(!scratch.idle.lock().unwrap().is_empty());
        assert!(scratch.idle.lock().unwrap().len() <= scratch.retained_limit);
        workers
            .install(|| {
                find_best_split_with_scratch(
                    &samples,
                    &indices,
                    0,
                    &options,
                    &mut current_parents,
                    &mut cache,
                    &budget,
                    &TrainingStats::default(),
                    &scratch,
                    &[],
                )
            })
            .unwrap();
        assert_cached_population(cache.as_ref().unwrap(), &samples, &indices);
        assert!(scratch.reused.load(Ordering::Relaxed) > 0);
        drop(cache);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn scratch_leases_retain_buffers_without_aliasing_or_global_state() {
    let pool = ScratchPool::new(2);
    let mut first = pool.acquire();
    first.scratch.increase.resize(257, 11);
    let first_pointer = first.scratch.increase.as_ptr();
    let mut second = pool.acquire();
    second.scratch.increase.resize(257, 23);
    assert_ne!(first_pointer, second.scratch.increase.as_ptr());
    drop(second);
    drop(first);
    let retained = pool.acquire();
    assert_eq!(retained.scratch.increase.as_ptr(), first_pointer);
    assert_eq!(retained.scratch.increase, vec![11; 257]);
    drop(retained);
    assert_eq!(pool.created.load(Ordering::Relaxed), 2);
    assert_eq!(pool.reused.load(Ordering::Relaxed), 1);
    // A suspended lease does not hold the pool lock. Unwinding returns a
    // partially used buffer, which the next property sweep must reset.
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut outer = pool.acquire();
            outer.scratch.extra.resize(5, 99);
            let mut inner = pool.acquire();
            inner.scratch.below.resize(7, Cost::default());
            panic!("scratch lease unwind");
        }))
        .is_err()
    );
    assert_eq!(pool.idle.lock().unwrap().len(), 2);
    let independent = ScratchPool::new(1);
    assert!(independent.acquire().scratch.increase.is_empty());
    assert_eq!(independent.created.load(Ordering::Relaxed), 1);
    let leases: Vec<_> = (0..4).map(|_| pool.acquire()).collect();
    drop(leases);
    assert_eq!(pool.idle.lock().unwrap().len(), pool.retained_limit);
}

#[test]
fn cross_node_scratch_reuse_matches_fresh_serial_and_ready_child_searches() {
    let (input, mut options) = reuse_fixture();
    let samples = TreeSamples::new(&input, &options).unwrap();
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for parallel in [false, true] {
        options.parallel = parallel;
        let retained = ScratchPool::new(if parallel { 6 } else { 1 });
        let fresh = ScratchPool::new(0);
        let stats = TrainingStats::default();
        let actual = workers.install(|| {
            Tree::new(
                train_nodes_with_scratch(&mut samples.clone(), &options, &stats, &retained)
                    .unwrap(),
            )
            .unwrap()
        });
        let expected = workers.install(|| {
            Tree::new(
                train_nodes_with_scratch(
                    &mut samples.clone(),
                    &options,
                    &TrainingStats::default(),
                    &fresh,
                )
                .unwrap(),
            )
            .unwrap()
        });
        assert_eq!(signature(&actual), signature(&expected));
        assert!(retained.reused.load(Ordering::Relaxed) > 0);
        assert!(retained.created.load(Ordering::Relaxed) < fresh.created.load(Ordering::Relaxed));
        assert!(stats.cache_subtractions.load(Ordering::Relaxed) > 0);
        assert!(retained.idle.lock().unwrap().len() <= retained.retained_limit);
        assert!(fresh.idle.lock().unwrap().is_empty());
        if !parallel {
            assert_eq!(retained.created.load(Ordering::Relaxed), 1);
        }
    }
    options.parallel = true;
    let retained = ScratchPool::new(2);
    let null_stats = TrainingStats::default();
    workers.install(|| {
        execution::serial(|| {
            train_nodes_with_scratch(&mut samples.clone(), &options, &null_stats, &retained)
                .unwrap();
        })
    });
    assert_eq!(retained.created.load(Ordering::Relaxed), 1);
    assert_eq!(null_stats.ready_nodes.load(Ordering::Relaxed), 0);
    assert!(retained.reused.load(Ordering::Relaxed) > 0);
}

fn assert_cached_population(cache: &HistogramCache, samples: &TreeSamples, indices: &[usize]) {
    for (property, histogram) in cache.properties.iter().enumerate() {
        let bins = samples.thresholds[property].len() + 1;
        let mut used = vec![0; bins];
        let mut weights = vec![0; bins];
        for &i in indices {
            let bin = samples.properties[property].value(i) as usize;
            used[bin] += 1;
            weights[bin] += u32::from(samples.counts[i]);
        }
        assert_eq!(histogram.used, used);
        assert_eq!(histogram.weights, weights);
        if histogram.predictors.is_empty() {
            assert!(used.iter().filter(|&&n| n != 0).count() <= 1);
            continue;
        }
        assert_eq!(histogram.predictors.len(), samples.residuals.len());
        for (predictor, histogram) in histogram.predictors.iter().enumerate() {
            let mut counts = vec![0; bins * histogram.symbols];
            let mut extra = vec![0u64; bins];
            for &i in indices {
                let bin = samples.properties[property].value(i) as usize;
                let token = samples.residuals[predictor][i];
                counts[bin * histogram.symbols + usize::from(token.token)] +=
                    u32::from(samples.counts[i]);
                extra[bin] += u64::from(token.nbits) * u64::from(samples.counts[i]);
            }
            assert_eq!(histogram.counts, counts);
            assert_eq!(histogram.extra, extra);
        }
    }
}

#[test]
fn deferred_property_subtraction_matches_fresh_repeated_inheritance() {
    let (mut samples, mut options) = histogram_fixture(8192, 3);
    samples.properties[0] = PropertyColumn::Wide((0..8192).map(|i| u32::from(i >= 513)).collect());
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for parallel in [false, true] {
        options.parallel = parallel;
        let budget = HistogramBudget::new(options.histogram_cache_bytes);
        let root_parents = parent_histograms(&samples, &indices);
        let mut inherited =
            HistogramCache::create(&samples, &indices, &root_parents, &budget, false);
        let scratch = ScratchPool::new(6);
        let stats = TrainingStats::default();
        let mut begin = 0;
        for end_removed in [513, 1025] {
            let remaining = &indices[end_removed..];
            let mut parents = None;
            let actual = workers
                .install(|| {
                    find_best_split_with_scratch(
                        &samples,
                        remaining,
                        0,
                        &options,
                        &mut parents,
                        &mut inherited,
                        &budget,
                        &stats,
                        &scratch,
                        &indices[begin..end_removed],
                    )
                })
                .unwrap();
            assert_cached_population(inherited.as_ref().unwrap(), &samples, remaining);
            let fresh_parents = parent_histograms(&samples, remaining);
            let mut fresh =
                HistogramCache::create(&samples, remaining, &fresh_parents, &budget, false);
            let expected = workers
                .install(|| {
                    find_best_split(
                        &samples,
                        remaining,
                        0,
                        &options,
                        &mut Some(fresh_parents),
                        &mut fresh,
                        &budget,
                        &TrainingStats::default(),
                    )
                })
                .unwrap();
            assert_same_candidates([actual, None, None, None], [expected, None, None, None]);
            begin = end_removed;
        }
        assert_eq!(
            stats.cache_subtractions.load(Ordering::Relaxed),
            2 * options.properties.len()
        );
    }
}

#[test]
fn pending_cache_subtraction_uses_membership_once_and_drops_terminal_caches() {
    let (samples, mut options) = histogram_fixture(8192, 3);
    options.parallel = false;
    let mut indices: Vec<_> = (0..samples.counts.len()).collect();
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let root_parents = parent_histograms(&samples, &indices);
    let scratch = ScratchPool::new(1);
    let stats = TrainingStats::default();
    let job = TrainingNode {
        node: 0,
        begin: 513,
        end: 8192,
        predictor: 0,
        multiplier: 1,
    };
    let mut pending = PendingNode::new(job);
    pending.cache = HistogramCache::create(&samples, &indices, &root_parents, &budget, false);
    pending.removed = Some(0..513);
    // DFS may already have repartitioned the smaller sibling's interval.
    indices[..513].reverse();
    pending
        .evaluate(&samples, &indices, &options, &budget, &stats, &scratch)
        .unwrap();
    assert!(pending.removed.is_none());
    assert_cached_population(pending.cache.as_ref().unwrap(), &samples, &indices[513..]);
    assert_eq!(
        stats.cache_subtractions.load(Ordering::Relaxed),
        options.properties.len()
    );
    pending
        .evaluate(&samples, &indices, &options, &budget, &stats, &scratch)
        .unwrap();
    assert_eq!(
        stats.cache_subtractions.load(Ordering::Relaxed),
        options.properties.len()
    );
    drop(pending);
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    for begin in [513, 8191, 8192] {
        let mut pending = PendingNode::new(TrainingNode { begin, ..job });
        pending.cache = HistogramCache::create(&samples, &indices, &root_parents, &budget, false);
        pending.removed = Some(0..begin);
        let stop = TrainOptions {
            split_threshold: 1e30,
            ..options.clone()
        };
        let created = scratch.created.load(Ordering::Relaxed);
        pending
            .evaluate(&samples, &indices, &stop, &budget, &stats, &scratch)
            .unwrap();
        assert!(pending.decision.unwrap().is_none());
        assert!(pending.cache.is_none());
        assert!(pending.removed.is_none());
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        assert_eq!(scratch.created.load(Ordering::Relaxed), created);
    }
}

#[test]
fn deferred_subtraction_failure_invalidates_partial_cache_and_releases_scratch() {
    let (samples, options) = histogram_fixture(8192, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let parents = parent_histograms(&samples, &indices);
    let mut cache = HistogramCache::create(&samples, &indices, &parents, &budget, false);
    cache.as_mut().unwrap().properties[0].used[0] = 0;
    let scratch = ScratchPool::new(4);
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let result = workers.install(|| {
        find_best_split_with_scratch(
            &samples,
            &indices[1024..],
            0,
            &options,
            &mut None,
            &mut cache,
            &budget,
            &TrainingStats::default(),
            &scratch,
            &indices[..1024],
        )
    });
    assert!(matches!(result, Err(Error::Encoding(_))));
    assert!(cache.is_none());
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    assert!(!scratch.idle.lock().unwrap().is_empty());
    assert!(scratch.idle.lock().unwrap().len() <= scratch.retained_limit);
}

#[test]
fn ready_child_parallel_search_preserves_dfs_budget_and_complete_search_contract() {
    let (input, mut options) = reuse_fixture();
    options.parallel = true;
    let samples = TreeSamples::new(&input, &options).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let stats = TrainingStats::default();
    let parallel = pool.install(|| {
        Tree::new(train_nodes_with_stats(&mut samples.clone(), &options, &stats).unwrap()).unwrap()
    });
    assert!(stats.ready_nodes.load(Ordering::Relaxed) >= 2);
    let null_stats = TrainingStats::default();
    let null_tree = pool.install(|| {
        execution::serial(|| {
            assert_eq!(execution::worker_count(), 1);
            Tree::new(train_nodes_with_stats(&mut samples.clone(), &options, &null_stats).unwrap())
                .unwrap()
        })
    });
    assert_eq!(null_stats.ready_nodes.load(Ordering::Relaxed), 0);
    assert_eq!(signature(&null_tree), signature(&parallel));
    assert_eq!(
        estimated_cost(&null_tree, &input, &options),
        estimated_cost(&parallel, &input, &options)
    );
    options.parallel_nodes = false;
    let serial = train(&input, &options).unwrap();
    assert_eq!(signature(&parallel), signature(&serial));
    options.max_leaves = 2;
    options.parallel_nodes = true;
    assert!(pool.install(|| matches!(train(&input, &options), Err(Error::Limit(_)))));
    assert!(pool.install(|| execution::serial(|| {
        matches!(train(&input, &options), Err(Error::Limit(_)))
    })));
    options.require_complete_search = false;
    let truncated = pool.install(|| train(&input, &options).unwrap());
    let null_truncated = pool.install(|| execution::serial(|| train(&input, &options).unwrap()));
    assert_eq!(signature(&null_truncated), signature(&truncated));
    options.parallel = false;
    assert_eq!(
        signature(&truncated),
        signature(&train(&input, &options).unwrap())
    );
    assert_eq!(truncated.leaves(), 2);
}

#[test]
fn recursive_learning_respects_budget_and_preserves_reachable_ranges() {
    let mut options = options(6);
    options.predictors = vec![Predictor::Zero];
    options.max_leaves = 5;
    let samples: Vec<_> = (0..16).map(|i| sample(6, i - 8, &[i], 100)).collect();
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 5);
    let mut populated = vec![false; tree.leaves()];
    for sample in &samples {
        let (context, predictor, offset, multiplier) = tree.lookup(&sample.properties).unwrap();
        populated[context] = true;
        assert_eq!((predictor, offset, multiplier), (Predictor::Zero, 0, 1));
    }
    assert!(populated.iter().all(|&used| used));
    let seed = Tree::new(vec![leaf(Predictor::Zero)]).unwrap();
    assert!(estimated_cost(&tree, &samples, &options) < estimated_cost(&seed, &samples, &options));
}

#[test]
fn multiplicity_matches_expansion_and_changes_predictor_choice() {
    let options = options(6);
    let weighted = vec![
        sample(6, 0, &[0, 200], 80),
        sample(6, 0, &[200, 0], 1),
        sample(6, 1, &[200, 0], 80),
    ];
    let expanded: Vec<_> = weighted
        .iter()
        .flat_map(|s| {
            (0..s.weight).map(move |_| Sample {
                weight: 1,
                ..s.clone()
            })
        })
        .collect();
    let tree = train(&weighted, &options).unwrap();
    assert_eq!(
        signature(&tree),
        signature(&train(&expanded, &options).unwrap())
    );
    assert_eq!(
        tree.lookup(&weighted[0].properties).unwrap().1,
        Predictor::Weighted
    );
    let reversed = vec![
        sample(6, 0, &[0, 200], 1),
        sample(6, 0, &[200, 0], 80),
        sample(6, 1, &[0, 200], 80),
    ];
    let tree = train(&reversed, &options).unwrap();
    assert_eq!(
        tree.lookup(&reversed[0].properties).unwrap().1,
        Predictor::Gradient
    );
}

#[test]
fn dedup_uses_tokens_and_extra_bit_counts_not_payloads() {
    let mut options = options(6);
    options.predictors = vec![Predictor::Zero];
    // PackSigned(8)=16 and PackSigned(10)=20 share a (4,1,2) token.
    let samples = vec![sample(6, 0, &[8], 70_000), sample(6, 0, &[10], 20)];
    let prepared = TreeSamples::new(&samples, &options).unwrap();
    assert_eq!(prepared.counts, [65_535, 4_485]);
    assert_eq!(
        prepared.residuals[0][0],
        ResidualToken {
            token: 16,
            nbits: 1
        }
    );
}

#[test]
fn quantization_retains_source_boundaries_and_weighted_quantiles() {
    assert!(quantize_histogram(&[1, 0, 1, 8], 4, 0).is_empty());
    assert_eq!(quantize_histogram(&[1, 1, 1, 1], 4, -2), [-2, -1, 0]);
    assert_eq!(property_thresholds(&[], 3, 4), [63, 127, 191]);
    assert_eq!(property_thresholds(&[], 15, 16).len(), 15);
    assert_eq!(property_thresholds(&[], 15, 32).len(), 29);
    assert_eq!(property_thresholds(&[], 15, 64).len(), 55);
    let options = options(6);
    let samples = vec![
        sample(6, i32::MIN, &[0, 1], 1),
        sample(6, 0, &[0, 1], 1),
        sample(6, i32::MAX, &[0, 1], 1),
    ];
    let prepared = TreeSamples::new(&samples, &options).unwrap();
    assert_eq!(prepared.thresholds[0], [-512, 0]);
    // The first two values share a quantized bin AND residual tokens.
    assert_eq!(prepared.properties[0].to_wide(), [1, 2]);
    assert_eq!(prepared.counts, [2, 1]);
}

#[test]
fn partition_count_is_not_sample_weight_and_columns_stay_aligned() {
    let values = [3u32, 0, 2, 1, 3, 0];
    let mut indices = vec![0, 1, 2, 3, 4, 5];
    partition_samples(&mut indices, &values, 1, 3);
    assert!(indices[..3].iter().all(|&i| values[i] <= 1));
    assert!(indices[3..].iter().all(|&i| values[i] > 1));
    indices.sort_unstable();
    assert_eq!(indices, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn batched_column_partition_matches_scalar_permutation_and_retains_buffers() {
    for pairs in [0, 1, 2, 127, 128, 129, 255, 256, 257] {
        let records = 2 * pairs + 17;
        let range = 7..7 + 2 * pairs;
        for below_count in [0, pairs, 2 * pairs] {
            for (property, bin, packed) in [
                (0, 0, false),
                (0, 513, false),
                (1, 0, true),
                (1, 254, true),
                (1, 513, false),
            ] {
                let (mut original, _) = histogram_fixture(records, 3);
                let mut values = original.properties[property].to_wide();
                for (i, value) in values[range.clone()].iter_mut().enumerate() {
                    *value = bin as u32 + u32::from(i < range.len() - below_count);
                }
                let mut expected: Vec<_> = (0..records).collect();
                partition_samples(&mut expected[range.clone()], &values, bin, below_count);
                original.properties[property] = if packed {
                    PropertyColumn::Packed(
                        values
                            .into_iter()
                            .map(|v| u8::try_from(v).unwrap())
                            .collect(),
                    )
                } else {
                    PropertyColumn::Wide(values)
                };
                let mut actual = original.clone();
                let property_ptrs: Vec<_> = actual.properties.iter().map(|p| p.as_ptr()).collect();
                let residual_ptrs: Vec<_> = actual.residuals.iter().map(|p| p.as_ptr()).collect();
                let count_ptr = actual.counts.as_ptr();
                let mut positions: Vec<_> = (0..records).collect();
                partition_node::<true>(
                    &mut actual,
                    &mut positions[range.clone()],
                    property,
                    bin,
                    below_count,
                );
                assert_eq!(positions, (0..records).collect::<Vec<_>>());
                assert_eq!(actual.thresholds, original.thresholds);
                assert_eq!(count_ptr, actual.counts.as_ptr());
                for (p, column) in actual.properties.iter().enumerate() {
                    assert_eq!(property_ptrs[p], column.as_ptr());
                    for (i, &source) in expected.iter().enumerate() {
                        assert_eq!(column.value(i), original.properties[p].value(source));
                    }
                }
                for (p, column) in actual.residuals.iter().enumerate() {
                    assert_eq!(residual_ptrs[p], column.as_ptr());
                    for (i, &source) in expected.iter().enumerate() {
                        assert_eq!(column[i], original.residuals[p][source]);
                    }
                }
                for (i, &source) in expected.iter().enumerate() {
                    assert_eq!(actual.counts[i], original.counts[source]);
                }
            }
        }
    }
}

fn layout_signature<const REORDER: bool>(
    samples: &TreeSamples,
    options: &TrainOptions,
) -> Result<Vec<(usize, u32)>> {
    let nodes = train_nodes_with_layout::<REORDER>(
        &mut samples.clone(),
        options,
        &TrainingStats::default(),
        &ScratchPool::new(6),
    )?;
    Ok(signature(&Tree::new(nodes)?))
}

#[test]
fn column_locality_matches_indexed_search_cache_parallelism_and_leaf_limits() {
    let (input, mut options) = reuse_fixture();
    let samples = TreeSamples::new(&input, &options).unwrap();
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let cache_bytes = options.histogram_cache_bytes;
    for parallel in [false, true] {
        options.parallel = parallel;
        for bytes in [0, cache_bytes] {
            options.histogram_cache_bytes = bytes;
            for (limit, complete) in [(8, true), (2, false), (2, true)] {
                options.max_leaves = limit;
                options.require_complete_search = complete;
                workers.install(|| {
                    let actual = layout_signature::<true>(&samples, &options);
                    let expected = layout_signature::<false>(&samples, &options);
                    if limit == 2 && complete {
                        assert!(matches!(&actual, Err(Error::Limit(_))));
                    } else {
                        assert!(actual.is_ok());
                    }
                    assert_eq!(
                        actual.map_err(|e| e.to_string()),
                        expected.map_err(|e| e.to_string()),
                    );
                });
            }
        }
    }
}

#[test]
fn column_moves_preserve_pending_cache_sibling_membership() {
    let (mut samples, mut options) = histogram_fixture(8192, 3);
    options.parallel = true;
    let mut indices: Vec<_> = (0..samples.counts.len()).collect();
    let budget = HistogramBudget::new(options.histogram_cache_bytes);
    let parents = parent_histograms(&samples, &indices);
    let mut pending = PendingNode::new(TrainingNode {
        node: 0,
        begin: 513,
        end: 8192,
        predictor: 0,
        multiplier: 1,
    });
    pending.cache = HistogramCache::create(&samples, &indices, &parents, &budget, false);
    pending.removed = Some(0..513);
    // Repartition both the excluded sibling and the live population after
    // capturing the parent cache, without changing either membership.
    for range in [0..513, 513..8192] {
        let below_count = indices[range.clone()]
            .iter()
            .filter(|&&i| samples.properties[1].value(i) <= 8)
            .count();
        partition_node::<true>(&mut samples, &mut indices[range], 1, 8, below_count);
    }
    assert_eq!(indices, (0..8192).collect::<Vec<_>>());
    let scratch = ScratchPool::new(3);
    let stats = TrainingStats::default();
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    workers.install(|| {
        pending
            .evaluate(&samples, &indices, &options, &budget, &stats, &scratch)
            .unwrap();
    });
    assert_cached_population(pending.cache.as_ref().unwrap(), &samples, &indices[513..]);
    assert_eq!(
        stats.cache_subtractions.load(Ordering::Relaxed),
        options.properties.len()
    );
    let mut fresh_cache = None;
    let mut fresh_parents = None;
    let expected = find_best_split(
        &samples,
        &indices[513..],
        0,
        &options,
        &mut fresh_parents,
        &mut fresh_cache,
        &budget,
        &TrainingStats::default(),
    )
    .unwrap();
    assert_same_candidates(
        [pending.decision.flatten(), None, None, None],
        [expected, None, None, None],
    );
    assert!(pending.removed.is_none());
    drop(pending);
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
}

#[test]
fn all_candidates_and_weighted_property_are_searched() {
    let mut options = TrainOptions::variable();
    options.properties = vec![0, 6, 15];
    options.split_threshold = 0.0;
    let mut samples = Vec::new();
    for (value, winner) in [(-3, Predictor::Average4), (3, Predictor::TopRight)] {
        let mut residuals = vec![1000; 14];
        let index = options
            .predictors
            .iter()
            .position(|&p| p == winner)
            .unwrap();
        residuals[index] = 0;
        samples.push(sample(15, value, &residuals, 100));
    }
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.tokens()[0].value, 16);
    assert_eq!(
        tree.lookup(&samples[0].properties).unwrap().1,
        Predictor::Average4
    );
    assert_eq!(
        tree.lookup(&samples[1].properties).unwrap().1,
        Predictor::TopRight
    );
    assert!(tree.uses_weighted());
}

#[test]
fn predictor_penalties_and_static_constant_preference_follow_source() {
    assert_eq!(
        predictor_penalty(Predictor::Top, Predictor::Gradient, 100.0),
        4.0
    );
    assert_eq!(
        predictor_penalty(Predictor::Top, Predictor::Weighted, 100.0),
        0.0
    );
    assert_eq!(
        predictor_penalty(Predictor::Weighted, Predictor::Weighted, 0.0),
        1e-8
    );
    assert_eq!(
        predictor_penalty(Predictor::Zero, Predictor::Zero, 0.0),
        -1e-8
    );
    let split = |property, bits| {
        Some(Split {
            property,
            bits,
            bin: 0,
            below_count: 1,
            below_predictor: 0,
            above_predictor: 0,
        })
    };
    let options = TrainOptions {
        split_threshold: 10.0,
        ..TrainOptions::default()
    };
    let chosen = choose_split(
        [split(0, 80.0), None, split(15, 5.0), split(6, 6.0)],
        100.0,
        &options,
    )
    .unwrap();
    assert_eq!(chosen.property, 0);
}

#[test]
fn huge_weights_and_signed_extremes_have_bounded_histograms() {
    let mut options = options(6);
    options.predictors = vec![Predictor::Zero];
    let samples = vec![
        sample(6, -1, &[i32::MIN], 2_000_000),
        sample(6, 1, &[i32::MAX], 2_000_000),
    ];
    let prepared = TreeSamples::new(&samples, &options).unwrap();
    assert!(
        prepared.residuals[0]
            .iter()
            .all(|r| r.token < 240 && r.nbits <= 28)
    );
    assert_eq!(train(&samples, &options).unwrap().leaves(), 2);
    let scorer = DefaultScorer::default();
    let mut above = [1, 2_000_000];
    let mut below = [0, 0];
    assert!(scorer.estimate_bits(&above, 2_000_001).is_finite());
    let (rbits, lbits) = scorer.move_and_score(&mut above, &mut below, &[1, 0], (2_000_000, 1));
    assert_eq!(rbits + lbits, 0.0);
    assert_eq!(above, [0, 2_000_000]);
    assert_eq!(below, [1, 0]);
}

#[test]
fn source_signed_histogram_limit_is_checked_before_scoring() {
    let mut options = options(6);
    options.predictors = vec![Predictor::Zero];
    let mut samples = vec![sample(6, 0, &[0], MAX_HISTOGRAM_TOTAL)];
    assert!(train(&samples, &options).is_ok());
    samples.push(sample(6, 1, &[1], 1));
    assert!(matches!(train(&samples, &options), Err(Error::Limit(_))));
}

#[test]
fn split_threshold_uses_f32_rounding_and_strict_improvement() {
    let mut options = TrainOptions {
        split_threshold: 1.0,
        ..TrainOptions::default()
    };
    let split = Split {
        property: 0,
        bin: 0,
        below_count: 1,
        below_predictor: 0,
        above_predictor: 0,
        bits: 16_777_216.0,
    };
    // 2^24 + 1 rounds back to 2^24. Adding in f64 would reject this split.
    assert!(choose_split([None, None, Some(split), None], 16_777_218.0, &options).is_some());
    assert!(choose_split([None, None, Some(split), None], split.bits, &options).is_none());
    options.split_threshold = f64::MAX;
    assert!(options.validate().is_err());
}

#[test]
fn serial_and_parallel_sweeps_have_identical_stable_ties() {
    let mut options = TrainOptions {
        properties: vec![6, 7, 9, 10, 11, 12, 13, 14, 15],
        predictors: vec![
            Predictor::Weighted,
            Predictor::Gradient,
            Predictor::Zero,
            Predictor::Left,
            Predictor::Top,
        ],
        max_leaves: 8,
        ..TrainOptions::default()
    };
    let samples: Vec<_> = (0..4096)
        .map(|i| {
            let value = i % 511 - 255;
            let mut s = sample(
                6,
                value,
                &[
                    if value <= 0 { i % 4 } else { 500 + i % 17 },
                    if value > 0 { i % 4 } else { 500 + i % 17 },
                    1000,
                    1000,
                    1000,
                ],
                1,
            );
            for &p in &options.properties {
                s.properties[p] = value;
            }
            s
        })
        .collect();
    let prepared = TreeSamples::new(&samples, &options).unwrap();
    assert!(prepared.counts.len() >= 1024);
    assert!(prepared.counts.len() * options.predictors.len() * options.properties.len() >= 65536);
    let parallel = train(&samples, &options).unwrap();
    options.parallel = false;
    let serial = train(&samples, &options).unwrap();
    assert_eq!(signature(&parallel), signature(&serial));
    assert!(parallel.leaves() > 1 && parallel.leaves() <= 8);
}

// Independent full rescans: no histogram transfers or incremental entropy.
fn brute_side_cost(
    samples: &TreeSamples,
    indices: &[usize],
    property: usize,
    bin: usize,
    above: bool,
    predictor: usize,
) -> f32 {
    let mut counts = [0u32; 256];
    let mut extra = 0u64;
    for &i in indices {
        if (samples.properties[property].value(i) as usize > bin) != above {
            continue;
        }
        let r = samples.residuals[predictor][i];
        counts[usize::from(r.token)] += u32::from(samples.counts[i]);
        extra += u64::from(r.nbits) * u64::from(samples.counts[i]);
    }
    let total: f64 = counts.iter().map(|&n| f64::from(n)).sum();
    (extra as f64
        + counts
            .iter()
            .filter(|&&n| n != 0)
            .map(|&n| {
                let n = f64::from(n);
                n * (total / n).log2().min(12.0)
            })
            .sum::<f64>()) as f32
}

#[test]
fn fused_split_costs_match_all_thresholds_full_histogram_oracle() {
    let options = TrainOptions {
        properties: vec![0, 6, 7, 15],
        predictors: vec![Predictor::Weighted, Predictor::Gradient, Predictor::Zero],
        split_threshold: 7.0,
        ..TrainOptions::default()
    };
    let raw: Vec<_> = (0..60)
        .map(|i| {
            let mut s = sample(
                6,
                i % 7 - 3,
                &[
                    i * 17 % 511 - 255,
                    (i % 3) * 4,
                    if i < 30 { 0 } else { i * 13 % 101 },
                ],
                1 + (i % 7) as u32 * 1000,
            );
            s.properties[0] = i % 2;
            s.properties[7] = (i % 5) * 13;
            s.properties[15] = i % 9 - 4;
            s
        })
        .collect();
    let samples = TreeSamples::new(&raw, &options).unwrap();
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let total = samples.counts.iter().map(|&n| u32::from(n)).sum();
    for property in 0..options.properties.len() {
        let actual =
            evaluate_property(&samples, &indices, &options, &parents, 0, total, property).unwrap();
        let mut expected = [None; 4];
        for bin in 0..samples.thresholds[property].len() {
            if !indices
                .iter()
                .any(|&i| samples.properties[property].value(i) as usize == bin)
            {
                continue;
            }
            let below_count = indices
                .iter()
                .filter(|&&i| samples.properties[property].value(i) as usize <= bin)
                .count();
            if below_count == 0 || below_count == indices.len() {
                continue;
            }
            let mut sides = [Cost::default(); 2];
            for (side, best) in sides.iter_mut().enumerate() {
                for predictor in 0..options.predictors.len() {
                    let bits =
                        brute_side_cost(&samples, &indices, property, bin, side == 1, predictor);
                    let penalty = predictor_penalty(
                        options.predictors[predictor],
                        options.predictors[0],
                        options.split_threshold as f32,
                    );
                    if bits + penalty < best.bits + best.penalty {
                        *best = Cost {
                            bits,
                            penalty,
                            predictor,
                        };
                    }
                }
            }
            let [below, above] = sides;
            let uses_wp = options.properties[property] == 15
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
            keep_better(
                &mut expected[class],
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
        for (actual, expected) in actual.into_iter().zip(expected) {
            match (actual, expected) {
                (None, None) => {}
                (Some(actual), Some(expected)) => {
                    let tolerance = 0.01 + expected.bits * 2e-6;
                    assert!(
                        (actual.bits - expected.bits).abs() <= tolerance,
                        "property {property}: {actual:?} versus {expected:?}"
                    );
                    let rescored = brute_side_cost(
                        &samples,
                        &indices,
                        property,
                        actual.bin,
                        false,
                        actual.below_predictor,
                    ) + brute_side_cost(
                        &samples,
                        &indices,
                        property,
                        actual.bin,
                        true,
                        actual.above_predictor,
                    );
                    assert!((actual.bits - rescored).abs() <= tolerance);
                }
                mismatch => panic!("candidate classes differ: {mismatch:?}"),
            }
        }
    }
}

fn constrained_options(range: StaticPropRange) -> TrainOptions {
    TrainOptions {
        predictors: vec![Predictor::Zero],
        properties: vec![0, 1, 6],
        static_property_range: Some(range),
        split_threshold: 0.0,
        parallel: false,
        ..TrainOptions::default()
    }
}

#[test]
fn multiplier_box_intersection_matches_source_axis_order() {
    let range = [[0, 4], [0, 8]];
    assert_eq!(box_intersects(range, range), Intersection::Inside);
    assert_eq!(box_intersects(range, [[4, 5], [0, 8]]), Intersection::None);
    assert_eq!(
        box_intersects(range, [[1, 3], [2, 6]]),
        Intersection::Partial {
            axis: 1,
            threshold: 1
        }
    );
    assert_eq!(
        box_intersects([[0, 4], [2, 6]], [[1, 3], [2, 6]]),
        Intersection::Partial {
            axis: 0,
            threshold: 0
        }
    );
    assert_eq!(
        box_intersects([[1, 4], [2, 6]], [[1, 3], [2, 6]]),
        Intersection::Partial {
            axis: 0,
            threshold: 2
        }
    );
}

#[test]
fn forced_multiplier_splits_cover_zero_entropy_and_unsampled_regions() {
    let mut options = constrained_options([[0, 3], [0, 2]]);
    options.max_leaves = 3;
    options.split_threshold = 100_000.0;
    options.multiplier_info = vec![
        MultiplierInfo {
            range: [[0, 1], [0, 2]],
            multiplier: 2,
        },
        MultiplierInfo {
            range: [[1, 3], [0, 1]],
            multiplier: 3,
        },
        MultiplierInfo {
            range: [[1, 3], [1, 2]],
            multiplier: 5,
        },
    ];
    for samples in [vec![], vec![sample(0, 0, &[0], 1)]] {
        let tree = train(&samples, &options).unwrap();
        assert_eq!(tree.leaves(), 3);
        for channel in 0..3 {
            for group in 0..2 {
                let expected = if channel == 0 {
                    2
                } else if group == 0 {
                    3
                } else {
                    5
                };
                let (_, predictor, offset, multiplier) = tree.lookup(&[channel, group]).unwrap();
                assert_eq!(
                    (predictor, offset, multiplier),
                    (Predictor::Zero, 0, expected)
                );
            }
        }
    }
    options.max_leaves = 2;
    assert!(matches!(train(&[], &options), Err(Error::Limit(_))));
}

#[test]
fn exact_static_multiplier_boundaries_are_not_clipped_to_511() {
    let mut options = constrained_options([[600, 603], [900, 902]]);
    options.multiplier_info = vec![MultiplierInfo {
        range: [[601, 603], [901, 902]],
        multiplier: 7,
    }];
    let mut samples = Vec::new();
    for channel in 600..603 {
        for group in 900..902 {
            let mut s = sample(0, channel, &[0], 1);
            s.properties[1] = group;
            samples.push(s);
        }
    }
    let prepared = TreeSamples::new(&samples, &options).unwrap();
    assert_eq!(prepared.thresholds[0], [600]);
    assert_eq!(prepared.thresholds[1], [900]);
    let tree = train(&samples, &options).unwrap();
    for s in &samples {
        let expected = if s.properties[0] >= 601 && s.properties[1] == 901 {
            7
        } else {
            1
        };
        assert_eq!(tree.lookup(&s.properties).unwrap().3, expected);
    }
}

#[test]
fn static_constraint_bins_can_exceed_byte_range() {
    let mut options = constrained_options([[0, 301], [0, 1]]);
    // The first rectangle covers the domain. Remaining equal-multiplier
    // rectangles add quantization boundaries without demanding more leaves.
    options.multiplier_info.push(MultiplierInfo {
        range: [[0, 301], [0, 1]],
        multiplier: 3,
    });
    for channel in 0..300 {
        options.multiplier_info.push(MultiplierInfo {
            range: [[channel, channel + 1], [0, 1]],
            multiplier: 3,
        });
    }
    let samples = [sample(0, 300, &[6], 1)];
    let prepared = TreeSamples::new(&samples, &options).unwrap();
    assert_eq!(prepared.properties[0].to_wide(), [300]);
    assert_eq!(
        train(&samples, &options)
            .unwrap()
            .lookup(&[300, 0])
            .unwrap()
            .3,
        3
    );
}

#[test]
fn forced_static_splits_match_indexed_layout_with_empty_and_sparse_domains() {
    let mut options = constrained_options([[600, 603], [900, 902]]);
    options.multiplier_info = vec![MultiplierInfo {
        range: [[601, 603], [901, 902]],
        multiplier: 7,
    }];
    options.max_leaves = 16;
    let input: Vec<_> = (0..1024)
        .map(|i| {
            let channel = 600 + i % 3;
            let group = 900 + i % 2;
            let multiplier = if channel >= 601 && group == 901 { 7 } else { 1 };
            let mut s = sample(6, i % 2, &[(i % 2) * 100 * multiplier], 1 + i as u32 % 7);
            s.properties[0] = channel;
            s.properties[1] = group;
            s
        })
        .collect();
    for raw in [&[][..], &input[..1], input.as_slice()] {
        let samples = TreeSamples::new(raw, &options).unwrap();
        let actual = layout_signature::<true>(&samples, &options).unwrap();
        assert_eq!(
            actual,
            layout_signature::<false>(&samples, &options).unwrap()
        );
        assert_eq!(actual, signature(&train(raw, &options).unwrap()));
    }
    options.max_leaves = 2;
    let empty = TreeSamples::new(&[], &options).unwrap();
    assert!(matches!(
        layout_signature::<true>(&empty, &options),
        Err(Error::Limit(_))
    ));
    assert_eq!(
        layout_signature::<true>(&empty, &options)
            .unwrap_err()
            .to_string(),
        layout_signature::<false>(&empty, &options)
            .unwrap_err()
            .to_string(),
    );
}

#[test]
fn adaptive_children_inherit_multiplier_at_leaf_budget() {
    let mut options = constrained_options([[0, 2], [0, 1]]);
    options.max_leaves = 3;
    options.multiplier_info = vec![
        MultiplierInfo {
            range: [[0, 1], [0, 1]],
            multiplier: 3,
        },
        MultiplierInfo {
            range: [[1, 2], [0, 1]],
            multiplier: 5,
        },
    ];
    let mut samples = Vec::new();
    for channel in 0..2 {
        let multiplier = if channel == 0 { 3 } else { 5 };
        for value in 0..4 {
            let mut s = sample(6, value, &[value * multiplier], 100);
            s.properties[0] = channel;
            samples.push(s);
        }
    }
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 3);
    for s in &samples {
        let (_, predictor, offset, multiplier) = tree.lookup(&s.properties).unwrap();
        let expected = if s.properties[0] == 0 { 3 } else { 5 };
        assert_eq!(
            (predictor, offset, multiplier),
            (Predictor::Zero, 0, expected)
        );
        let coded = (i64::from(s.residuals[0]) - i64::from(offset)) / i64::from(multiplier);
        assert_eq!(
            coded * i64::from(multiplier) + i64::from(offset),
            i64::from(s.residuals[0])
        );
    }
}

#[test]
fn multiplier_validation_checks_raw_payloads_after_training_dedup() {
    let mut options = constrained_options([[0, 1], [0, 1]]);
    options.multiplier_info = vec![MultiplierInfo {
        range: [[0, 1], [0, 1]],
        multiplier: 4,
    }];
    // 8 and 10 share the source learning token, but 10 is not divisible by 4.
    let samples = [sample(0, 0, &[8], 1), sample(0, 0, &[10], 1)];
    assert_eq!(TreeSamples::new(&samples, &options).unwrap().counts, [2]);
    assert!(train(&samples, &options).is_err());
    let samples = [sample(0, 0, &[-12], 1), sample(0, 0, &[8], 1)];
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.lookup(&[0, 0]).unwrap().3, 4);
    // Source multiplier serialization: multiplier = (bits + 1) << log.
    let tokens = tree.tokens();
    assert_eq!(tokens[3].value, 2);
    assert_eq!(tokens[4].value, 0);
}

#[test]
fn narrowed_wide_residuals_do_not_reject_valid_odd_multiplier_factors() {
    let pairs = [(i32::MAX, i32::MIN + 3), (i32::MIN + 3, i32::MAX)];
    for multiplier in [3, 6] {
        let mut options = constrained_options([[0, 1], [0, 1]]);
        options.predictors = vec![Predictor::Gradient];
        options.multiplier_info = vec![MultiplierInfo {
            range: [[0, 1], [0, 1]],
            multiplier,
        }];
        let samples: Vec<_> = pairs
            .iter()
            .map(|&(pixel, prediction)| {
                let wide = i64::from(pixel) - i64::from(prediction);
                assert_eq!(wide.abs(), 4_294_967_292);
                assert_eq!(wide % i64::from(multiplier), 0);
                assert_ne!(i64::from(wide as i32) % i64::from(multiplier), 0);
                sample(0, 0, &[wide as i32], 1)
            })
            .collect();
        let tree = train(&samples, &options).unwrap();
        let (_, predictor, offset, fitted) = tree.lookup(&samples[0].properties).unwrap();
        assert_eq!(
            (predictor, offset, fitted),
            (Predictor::Gradient, 0, multiplier)
        );
        for (pixel, prediction) in pairs {
            let wide = i64::from(pixel) - i64::from(prediction) - i64::from(offset);
            let quotient = (wide / i64::from(fitted)) as i32;
            assert_eq!(
                (i64::from(quotient) * i64::from(fitted)
                    + i64::from(prediction)
                    + i64::from(offset)) as i32,
                pixel
            );
        }
        // The narrowed sample cannot establish the odd factor. Even when
        // it passes this check, full-plane wide divisibility is mandatory.
        let ambiguous = if multiplier == 3 { 1 } else { 2 };
        assert_ne!(ambiguous % multiplier as i32, 0);
        assert!(train(&[sample(0, 0, &[ambiguous], 1)], &options).is_ok());
        if multiplier == 6 {
            assert!(train(&[sample(0, 0, &[1], 1)], &options).is_err());
        }
    }
}

#[test]
fn validates_multiplier_domain_overlap_axes_and_endpoints() {
    let mut options = constrained_options([[0, 2], [0, 2]]);
    options.multiplier_info = vec![MultiplierInfo {
        range: [[0, 2], [0, 1]],
        multiplier: 3,
    }];
    let mut invalid = options.clone();
    invalid.properties.retain(|&p| p != 1);
    assert!(train(&[], &invalid).is_err());
    invalid = options.clone();
    invalid.static_property_range = None;
    assert!(train(&[], &invalid).is_err());
    invalid = options.clone();
    invalid.multiplier_info.push(MultiplierInfo {
        range: [[1, 2], [0, 2]],
        multiplier: 5,
    });
    assert!(train(&[], &invalid).is_err());
    for multiplier in [0, i32::MAX as u32 + 1] {
        invalid = options.clone();
        invalid.multiplier_info[0].multiplier = multiplier;
        assert!(train(&[], &invalid).is_err());
    }
    for range in [[[0, 0], [0, 1]], [[0, 1], [2, 1]], [[0, 1], [0, u32::MAX]]] {
        invalid = options.clone();
        invalid.static_property_range = Some(range);
        assert!(train(&[], &invalid).is_err());
    }
    let range = [[i32::MAX as u32 - 1, i32::MAX as u32 + 1], [0, 1]];
    options = constrained_options(range);
    options.multiplier_info = vec![MultiplierInfo {
        range: [[i32::MAX as u32, i32::MAX as u32 + 1], [0, 1]],
        multiplier: 7,
    }];
    let tree = train(&[], &options).unwrap();
    assert_eq!(tree.lookup(&[i32::MAX - 1, 0]).unwrap().3, 1);
    assert_eq!(tree.lookup(&[i32::MAX, 0]).unwrap().3, 7);
}

fn view(channel: u32, group: u32, pixels: &[i32]) -> ChannelView<'_> {
    ChannelView {
        channel,
        group,
        width: pixels.len(),
        height: 1,
        stride: pixels.len(),
        pixels,
        is_meta: false,
    }
}

#[test]
fn constant_scan_checks_all_groups_variable_dimensions_and_visible_rows() {
    let varying = [1, 2];
    let padded = [-7, -7, 900, -7, -7];
    let narrow = [-7];
    let constant = [31, 31, 31];
    let channels = [
        view(0, 0, &varying),
        ChannelView {
            width: 2,
            height: 2,
            stride: 3,
            ..view(1, 0, &padded)
        },
        view(1, 1, &narrow),
        view(2, 0, &constant),
        view(2, 1, &[]),
    ];
    assert_eq!(
        find_constant_channels(&channels, [[0, 3], [0, 2]], 32).unwrap(),
        [
            ConstantChannel {
                channel: 1,
                value: -7
            },
            ConstantChannel {
                channel: 2,
                value: 31
            }
        ]
    );
    let mut changed = channels;
    changed[2] = view(1, 1, &[8]);
    assert_eq!(
        find_constant_channels(&changed, [[0, 3], [0, 2]], 32).unwrap(),
        [ConstantChannel {
            channel: 2,
            value: 31
        }]
    );
}

#[test]
fn constant_scan_rejects_metadata_oversize_missing_and_late_outliers() {
    let pixels = [9, 9, 9, 9, 8];
    assert!(
        find_constant_channels(&[view(0, 0, &pixels)], [[0, 1], [0, 1]], 32)
            .unwrap()
            .is_empty()
    );
    let pixels = [9, 9];
    let channel = view(0, 0, &pixels);
    assert!(
        find_constant_channels(&[channel], [[0, 1], [0, 1]], 1)
            .unwrap()
            .is_empty()
    );
    assert!(
        find_constant_channels(
            &[ChannelView {
                is_meta: true,
                ..channel
            }],
            [[0, 1], [0, 1]],
            32
        )
        .unwrap()
        .is_empty()
    );
    assert!(
        find_constant_channels(&[channel], [[0, 2], [0, 1]], 32)
            .unwrap()
            .is_empty()
    );
    assert!(
        find_constant_channels(&[channel, view(0, 1, &[10])], [[0, 1], [0, 2]], 32)
            .unwrap()
            .is_empty()
    );
    assert!(
        find_constant_channels(
            &[ChannelView {
                stride: 1,
                ..channel
            }],
            [[0, 1], [0, 1]],
            32
        )
        .is_err()
    );
    assert!(
        find_constant_channels(
            &[ChannelView {
                height: 2,
                ..channel
            }],
            [[0, 1], [0, 1]],
            32
        )
        .is_err()
    );
    assert!(find_constant_channels(&[channel, channel], [[0, 1], [0, 1]], 32).is_err());
}

#[test]
fn borrowed_variable_samples_match_owned_subset_and_constant_splice() {
    let mut input = vec![
        sample(0, 2, &[37, 0], 11),
        sample(0, 1, &[5, 1], 3),
        sample(0, 0, &[0, 4], 5),
        sample(0, 1, &[5, 1], 7),
        sample(0, 0, &[123, 456], 0),
        sample(0, 2, &[37, 0], 13),
    ];
    for sample in &mut input {
        sample.properties[1] = 21;
        sample.properties[6] = if sample.properties[0] == 1 { 7 } else { -3 };
    }
    let variable_only: Vec<_> = input
        .iter()
        .filter(|s| s.weight != 0 && s.properties[0] < 2)
        .cloned()
        .collect();
    for prequantized in [false, true] {
        let mut options = constrained_options([[0, 3], [21, 23]]);
        options.predictors = vec![Predictor::Zero, Predictor::Gradient];
        options.max_leaves = 8;
        options.max_property_values = 4;
        options.constant_channels = vec![ConstantChannel {
            channel: 2,
            value: 37,
        }];
        if prequantized {
            options.property_quantization = Some(PropertyQuantization {
                properties: options.properties.clone(),
                thresholds: vec![vec![0], vec![21], vec![0]],
            });
        }
        let learning = TrainOptions {
            max_leaves: options.max_leaves - 1,
            static_property_range: Some([[0, 2], [21, 23]]),
            constant_channels: Vec::new(),
            ..options.clone()
        };
        for supplied in [input.as_slice(), variable_only.as_slice()] {
            let borrowed = supplied
                .iter()
                .filter(|s| s.weight != 0 && s.properties[0] < 2);
            let owned: Vec<_> = borrowed.clone().cloned().collect();
            for sample in borrowed.clone() {
                assert!(
                    supplied
                        .iter()
                        .any(|original| std::ptr::eq(original, sample))
                );
            }
            let actual = TreeSamples::new(borrowed, &learning).unwrap();
            let reference = TreeSamples::new(&owned, &learning).unwrap();
            assert_eq!(actual.thresholds, reference.thresholds);
            assert_eq!(actual.properties, reference.properties);
            assert_eq!(actual.residuals, reference.residuals);
            assert_eq!(actual.counts, reference.counts);
            assert_eq!(actual.counts.iter().map(|&n| u32::from(n)).sum::<u32>(), 15);
            let mut nodes = train_nodes(&mut reference.clone(), &learning).unwrap();
            append_constants(&mut nodes, &options.constant_channels);
            let reference_tree = Tree::new(nodes).unwrap();
            let actual_tree = train(supplied, &options).unwrap();
            assert_eq!(signature(&actual_tree), signature(&reference_tree));
            assert_eq!(actual_tree.lookup(&[2, 21]).unwrap().2, 37);
        }
    }
}

#[test]
fn borrowed_sample_preparation_preserves_empty_and_validation_behavior() {
    let options = options(6);
    let empty = [sample(0, 2, &[0, 0], 1)];
    let prepared =
        TreeSamples::new(empty.iter().filter(|s| s.properties[0] < 2), &options).unwrap();
    let reference = TreeSamples::new(&[], &options).unwrap();
    assert_eq!(prepared.thresholds, reference.thresholds);
    assert!(prepared.counts.is_empty());
    let invalid = [sample(6, 0, &[0], 0)];
    assert!(TreeSamples::new(invalid.iter(), &options).is_err());
    let overweight = [sample(6, 0, &[0, 0], u32::MAX), sample(6, 0, &[0, 0], 1)];
    assert!(matches!(
        TreeSamples::new(overweight.iter(), &options),
        Err(Error::Limit(_))
    ));
}

#[test]
fn verified_constant_suffix_produces_nonzero_offsets_and_reserves_leaves() {
    let channels = [
        view(0, 0, &[1, 2]),
        view(1, 0, &[-7, -7]),
        view(2, 0, &[31]),
    ];
    let range = [[0, 3], [0, 1]];
    let options = TrainOptions {
        static_property_range: Some(range),
        constant_channels: find_constant_channels(&channels, range, 32).unwrap(),
        max_leaves: 3,
        ..TrainOptions::default()
    };
    let samples = [sample(0, 0, &[3, 4], 1), sample(0, 1, &[0, 0], 1)];
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 3);
    assert_eq!(tree.lookup(&[0, 0]).unwrap().1, Predictor::Weighted);
    let (_, predictor, offset, multiplier) = tree.lookup(&[1, 0]).unwrap();
    assert_eq!((predictor, offset, multiplier), (Predictor::Zero, -7, 1));
    assert_eq!(tree.lookup(&[2, 0]).unwrap().2, 31);
    assert!(
        tree.tokens()
            .iter()
            .any(|t| t.context == 3 && t.value == pack_signed(-7))
    );
    let bad = TrainOptions {
        max_leaves: 2,
        ..options
    };
    assert!(matches!(train(&samples, &bad), Err(Error::Limit(_))));
}

#[test]
fn all_constant_offsets_support_signed_extremes_without_training_samples() {
    let range = [[0, 2], [0, 1]];
    let channels = [view(0, 0, &[i32::MIN]), view(1, 0, &[i32::MAX])];
    let options = TrainOptions {
        static_property_range: Some(range),
        constant_channels: find_constant_channels(&channels, range, 32).unwrap(),
        max_leaves: 2,
        ..TrainOptions::default()
    };
    let tree = train(&[], &options).unwrap();
    assert_eq!(tree.leaves(), 2);
    assert_eq!(tree.lookup(&[0, 0]).unwrap().2, i32::MIN);
    assert_eq!(tree.lookup(&[1, 0]).unwrap().2, i32::MAX);
    for pixel in [i32::MIN, i32::MAX] {
        let channel = i32::from(pixel == i32::MAX);
        let (_, _, offset, multiplier) = tree.lookup(&[channel, 0]).unwrap();
        let residual = (i64::from(pixel) - i64::from(offset)) / i64::from(multiplier);
        assert_eq!(residual, 0);
        assert_eq!(
            residual * i64::from(multiplier) + i64::from(offset),
            i64::from(pixel)
        );
    }
}

#[test]
fn constants_and_multipliers_coexist_without_budget_or_range_leakage() {
    let mut options = constrained_options([[0, 3], [0, 1]]);
    options.max_leaves = 3;
    options.constant_channels = vec![ConstantChannel {
        channel: 2,
        value: -11,
    }];
    options.multiplier_info = vec![
        MultiplierInfo {
            range: [[0, 1], [0, 1]],
            multiplier: 3,
        },
        MultiplierInfo {
            range: [[1, 3], [0, 1]],
            multiplier: 5,
        },
    ];
    let samples = [
        sample(0, 0, &[6], 1),
        sample(0, 1, &[10], 1),
        sample(0, 2, &[-11], 1),
    ];
    let tree = train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 3);
    assert_eq!(tree.lookup(&[0, 0]).unwrap().3, 3);
    assert_eq!(tree.lookup(&[1, 0]).unwrap().3, 5);
    let (_, _, offset, multiplier) = tree.lookup(&[2, 0]).unwrap();
    assert_eq!((offset, multiplier), (-11, 1));
    assert!(train(&[sample(0, 2, &[-10], 1)], &options).is_err());
    options.constant_channels[0].channel = 1;
    assert!(train(&[], &options).is_err());
}

#[test]
fn validates_inputs_without_panicking() {
    let options = options(6);
    assert!(train(&[sample(6, 0, &[0], 1)], &options).is_err());
    assert!(
        train(
            &[Sample {
                properties: vec![],
                residuals: vec![0, 0],
                weight: 1
            }],
            &options
        )
        .is_err()
    );
    let overflow = [sample(6, 0, &[0, 0], u32::MAX), sample(6, 0, &[0, 0], 1)];
    assert!(train(&overflow, &options).is_err());
    for limit in [0, MAX_LEAVES + 1, usize::MAX] {
        assert!(
            train(
                &[],
                &TrainOptions {
                    max_leaves: limit,
                    ..options.clone()
                }
            )
            .is_err()
        );
    }
    for threshold in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            train(
                &[],
                &TrainOptions {
                    split_threshold: threshold,
                    ..options.clone()
                }
            )
            .is_err()
        );
    }
    assert!(
        train(
            &[],
            &TrainOptions {
                predictors: vec![Predictor::Best],
                ..options.clone()
            }
        )
        .is_err()
    );
    assert!(
        train(
            &[],
            &TrainOptions {
                predictors: vec![Predictor::Zero; 2],
                ..options.clone()
            }
        )
        .is_err()
    );
    assert!(
        train(
            &[],
            &TrainOptions {
                properties: vec![256],
                ..options.clone()
            }
        )
        .is_err()
    );
    assert!(
        train(
            &[],
            &TrainOptions {
                properties: vec![6, 6],
                ..options
            }
        )
        .is_err()
    );
}
