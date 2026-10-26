use super::*;
use crate::modular::{
    encode::{self, ModularOptions},
    sample::{PropertySamples, SamplingPolicy, SamplingStats, SourcePolicy},
};

fn fixture(records: usize, predictors: usize) -> (TreeSamples, TrainOptions) {
    let mut options = TrainOptions::variable();
    options.predictors.truncate(predictors);
    options.properties = vec![6];
    options.parallel = false;
    options.split_threshold = 0.0;
    let samples = TreeSamples {
        properties: vec![PropertyColumn::Packed(
            (0..records).map(|i| (i % 8) as u8).collect(),
        )],
        residuals: (0..predictors)
            .map(|p| {
                (0..records)
                    .map(|i| ResidualToken {
                        token: ((i / 8 + 3 * p) % 17) as u8,
                        nbits: (i % 3) as u8,
                    })
                    .collect()
            })
            .collect(),
        counts: (0..records)
            .map(|i| {
                if i % 17 == 0 {
                    u16::MAX
                } else {
                    1 + (i % 7) as u16
                }
            })
            .collect(),
        thresholds: vec![(0..7).collect()],
    };
    (samples, options)
}

fn assert_same_split(actual: Option<Split>, expected: Option<Split>) {
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
        other => panic!("cache admission changed split: {other:?}"),
    }
}

#[test]
fn early_cache_14_defaults_and_candidate_gate() {
    assert!(!HistogramOptions::default().early_cache_14);
    assert!(!TrainOptions::default().early_cache_14);
    assert!(!TrainOptions::variable().early_cache_14);
    assert_eq!(TrainOptions::default().histogram_cache_bytes, 32 << 20);
    for predictors in 1..=14 {
        let mut options = TrainOptions::variable();
        options.predictors.truncate(predictors);
        for enabled in [false, true] {
            HistogramOptions {
                early_cache_14: enabled,
                ..HistogramOptions::default()
            }
            .apply(&mut options);
            assert_eq!(options.early_cache_14, enabled);
            let expected = if enabled && predictors == 14 {
                (2048, 1024)
            } else {
                (8192, 4096)
            };
            assert_eq!(options.histogram_cache_thresholds(), expected);
            options.predictor_dedup = true;
            assert_eq!(options.histogram_cache_thresholds(), expected);
            assert_eq!(options.histogram_cache_bytes, 32 << 20);
        }
    }
}

#[test]
fn early_cache_14_survives_source_option_and_sampling_projection() {
    for enabled in [false, true] {
        let source = SourcePolicy {
            predictor: Predictor::Variable,
            ..SourcePolicy::default()
        };
        let policy = SamplingPolicy::from_source(&source).unwrap();
        let modular = ModularOptions {
            predictor: Predictor::Variable,
            source_ma_histograms: HistogramOptions {
                early_cache_14: enabled,
                ..HistogramOptions::default()
            },
            ..ModularOptions::default()
        };
        let mut train = encode::train_options(&modular);
        assert_eq!(train.histogram_cache_bytes, 32 << 20);
        train.histogram_cache_bytes = 12345;
        let predictors = train.predictors.clone();
        policy
            .apply_to_train(
                &PropertySamples::default(),
                SamplingStats {
                    visited_pixels: 1024,
                    selected_samples: 1024,
                    source_total_pixels: 2048,
                },
                &mut train,
            )
            .unwrap();
        assert_eq!(train.early_cache_14, enabled);
        assert_eq!(train.predictors, predictors);
        assert_eq!(train.predictors.len(), 14);
        assert_eq!(train.histogram_cache_bytes, 12345);
        assert_eq!(train.properties, policy.properties);
        assert!(train.property_quantization.is_some());
        train.validate().unwrap();
    }
}

#[test]
fn early_cache_14_creation_boundaries_preserve_weighted_split() {
    for (records, predictors, enabled, admitted) in [
        (2047, 14, true, false),
        (2048, 14, true, true),
        (2049, 14, true, true),
        (8191, 14, false, false),
        (8192, 14, false, true),
        (2048, 2, true, false),
        (8192, 2, true, true),
        (2048, 13, true, false),
        (8192, 13, true, true),
    ] {
        let (samples, mut options) = fixture(records, predictors);
        options.early_cache_14 = enabled;
        let indices: Vec<_> = (0..records).rev().collect();
        let raw_parents = parent_histograms(&samples, &indices);
        let budget = HistogramBudget::new(options.histogram_cache_bytes);
        let stats = TrainingStats::default();
        let mut parents = None;
        let mut cache = None;
        let actual = find_best_split(
            &samples,
            &indices,
            0,
            &options,
            &mut parents,
            &mut cache,
            &budget,
            &stats,
        )
        .unwrap();
        assert_eq!(parents.as_ref().unwrap(), &raw_parents);
        assert_eq!(cache.is_some(), admitted);
        assert_eq!(
            stats.cache_builds.load(Ordering::Relaxed),
            usize::from(admitted)
        );
        if let Some(cache) = &cache {
            let histogram = &cache.properties[0];
            let total = samples.counts.iter().map(|&n| u32::from(n)).sum::<u32>();
            assert!(histogram.valid);
            assert_eq!(histogram.used.iter().sum::<usize>(), records);
            assert_eq!(histogram.weights.iter().sum::<u32>(), total);
            for (cached, parent) in histogram.predictors.iter().zip(&raw_parents) {
                let mut counts = vec![0; cached.symbols];
                for row in cached.counts.chunks_exact(cached.symbols) {
                    for (count, &n) in counts.iter_mut().zip(row) {
                        *count += n;
                    }
                }
                assert_eq!(counts, parent.counts);
                assert_eq!(cached.extra.iter().sum::<u64>(), parent.extra_bits);
            }
        }
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
        assert_same_split(actual, expected);
        assert!(budget.peak.load(Ordering::Relaxed) <= options.histogram_cache_bytes);
        drop(cache);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn early_cache_14_respects_exact_and_denied_byte_budgets() {
    let (samples, mut options) = fixture(2048, 14);
    options.early_cache_14 = true;
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let measuring = HistogramBudget::new(32 << 20);
    let reserved = HistogramCache::reserve(&samples, &parents, &measuring).unwrap();
    let bytes = reserved._lease.bytes;
    drop(reserved);
    assert!(bytes > 1 && bytes <= 32 << 20);
    assert_eq!(measuring.used.load(Ordering::Relaxed), 0);
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
    for limit in [0, bytes - 1, bytes] {
        options.histogram_cache_bytes = limit;
        let budget = HistogramBudget::new(limit);
        let stats = TrainingStats::default();
        let mut cache = None;
        let actual = find_best_split(
            &samples, &indices, 0, &options, &mut None, &mut cache, &budget, &stats,
        )
        .unwrap();
        assert_same_split(actual, expected);
        assert_eq!(cache.is_some(), limit == bytes);
        let expected_bytes = if limit == bytes { bytes } else { 0 };
        assert_eq!(budget.used.load(Ordering::Relaxed), expected_bytes);
        assert_eq!(budget.peak.load(Ordering::Relaxed), expected_bytes);
        drop(cache);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn early_cache_14_inheritance_keeps_unbalanced_and_record_boundaries() {
    for (predictors, enabled, low, high, inherited) in [
        (14, true, 16, 1023, false),
        (14, true, 16, 1024, true),
        (14, false, 16, 4095, false),
        (14, false, 16, 4096, true),
        (2, true, 16, 1024, false),
        (2, true, 16, 4096, true),
        (14, true, 1024, 1024, false),
    ] {
        let records = low + high;
        let (mut samples, mut options) = fixture(records, predictors);
        options.early_cache_14 = enabled;
        samples.properties[0] =
            PropertyColumn::Packed((0..records).map(|i| u8::from(i >= low)).collect());
        samples.thresholds[0] = vec![0];
        let mut indices: Vec<_> = (0..records).collect();
        let parents = parent_histograms(&samples, &indices);
        let budget = HistogramBudget::new(options.histogram_cache_bytes);
        // Model an already evaluated ancestor; child admission is exercised by
        // the real DFS, including cache transfer, deferred subtraction and drop.
        let mut pending = PendingNode::new(TrainingNode {
            node: 0,
            begin: 0,
            end: records,
            predictor: 0,
            multiplier: 1,
        });
        pending.cache = HistogramCache::create(&samples, &indices, &parents, &budget, false);
        assert!(pending.cache.is_some());
        pending.parents = Some(parents);
        pending.decision = Some(Some(Split {
            property: 0,
            bin: 0,
            below_count: low,
            below_predictor: 0,
            above_predictor: 0,
            bits: 0.0,
        }));
        let stats = TrainingStats::default();
        let nodes = train_pending_with_layout::<false>(
            &mut samples,
            &options,
            &stats,
            &ScratchPool::new(1),
            &mut indices,
            vec![leaf(options.predictors[0])],
            vec![pending],
        )
        .unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(
            stats.cache_reuses.load(Ordering::Relaxed),
            usize::from(inherited)
        );
        assert_eq!(
            stats.cache_subtractions.load(Ordering::Relaxed),
            usize::from(inherited)
        );
        assert_eq!(stats.cache_builds.load(Ordering::Relaxed), 0);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        assert!(budget.peak.load(Ordering::Relaxed) <= options.histogram_cache_bytes);
    }
}
