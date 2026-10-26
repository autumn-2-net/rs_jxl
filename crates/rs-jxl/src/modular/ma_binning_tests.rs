use super::*;

fn fixture() -> (TreeSamples, TrainOptions) {
    let mut options = TrainOptions::variable();
    options.properties = vec![0, 6, 15];
    let records = 384;
    let samples = TreeSamples {
        properties: vec![
            PropertyColumn::Packed((0..records).map(|i| (i % 3) as u8).collect()),
            PropertyColumn::Wide((0..records).map(|i| (i * 73 % 257 * 3) as u32).collect()),
            PropertyColumn::Packed((0..records).map(|i| (i * 29 % 251) as u8).collect()),
        ],
        residuals: (0..14)
            .map(|p| {
                (0..records)
                    .map(|i| ResidualToken {
                        token: ((i * 17 + p * 11) % 89) as u8,
                        nbits: ((i + p) % 13) as u8,
                    })
                    .collect()
            })
            .collect(),
        counts: (0..records).map(|i| (1 + i % 13) as u16).collect(),
        thresholds: vec![(0..2).collect(), (0..771).collect(), (0..251).collect()],
    };
    (samples, options)
}

#[test]
fn adaptive_histograms_match_full_scoring_at_retained_thresholds() {
    let (samples, options) = fixture();
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let total = samples.counts.iter().map(|&w| u32::from(w)).sum();
    let mut dense = PropertyScratch::default();
    let mut coarse = PropertyScratch::default();
    for property in [1, 2] {
        evaluate_property_cached(
            &samples, &indices, &options, &parents, 0, total, property, None, None, &mut dense,
        )
        .unwrap();
        for limit in [2, 8, 16, 32, 64, 256] {
            let mut trial = options.clone();
            trial.search.histogram_bins = limit;
            evaluate_property_cached(
                &samples,
                &indices,
                &trial,
                &parents,
                0,
                total,
                property,
                None,
                None,
                &mut coarse,
            )
            .unwrap();
            assert_eq!(coarse.scored_predictors.len(), 14);
            let first = coarse.used.iter().position(|&n| n != 0).unwrap();
            let last = coarse.used.iter().rposition(|&n| n != 0).unwrap();
            let mut checked = 0;
            for bin in first..last {
                if coarse.used[bin] == 0
                    || (!coarse.score_mask.is_empty() && coarse.score_mask[bin - first] == 0)
                {
                    continue;
                }
                checked += 1;
                for (a, b) in [
                    (coarse.below[bin], dense.below[bin]),
                    (coarse.above[bin], dense.above[bin]),
                ] {
                    assert!(
                        (a.bits - b.bits).abs() <= 0.01,
                        "limit={limit} bin={bin}: {} != {}",
                        a.bits,
                        b.bits
                    );
                    assert_eq!(a.predictor, b.predictor);
                    assert_eq!(a.penalty, b.penalty);
                }
            }
            assert!(checked > 0 && checked < usize::from(limit));
        }
    }
}

#[test]
fn adaptive_histograms_keep_static_splits_and_reject_conflicting_modes() {
    let (samples, options) = fixture();
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let total = samples.counts.iter().map(|&w| u32::from(w)).sum();
    let mut a = PropertyScratch::default();
    let mut b = PropertyScratch::default();
    let mut trial = options.clone();
    trial.search.histogram_bins = 2;
    let original = evaluate_property_cached(
        &samples, &indices, &options, &parents, 0, total, 0, None, None, &mut a,
    )
    .unwrap();
    let result = evaluate_property_cached(
        &samples, &indices, &trial, &parents, 0, total, 0, None, None, &mut b,
    )
    .unwrap();
    assert_eq!(format!("{original:?}"), format!("{result:?}"));
    assert!(b.score_mask.is_empty());
    assert!(trial.validate().is_ok());
    for bins in [1, 257, u16::MAX] {
        trial.search.histogram_bins = bins;
        assert!(trial.validate().is_err());
    }
    trial.search.histogram_bins = 32;
    trial.search.refine_quantiles = true;
    assert!(trial.validate().is_err());
    trial.search.refine_quantiles = false;
    trial.search.statistical_race = true;
    assert!(trial.validate().is_err());
}

#[test]
fn adaptive_histograms_build_valid_trees_without_fine_property_cache() {
    let (samples, mut options) = fixture();
    options.search.histogram_bins = 16;
    for limit in [8, 64, 256] {
        options.max_leaves = limit;
        for parallel in [false, true] {
            options.parallel = parallel;
            let mut input = samples.clone();
            let stats = TrainingStats::default();
            let nodes = train_nodes_with_stats(&mut input, &options, &stats).unwrap();
            let tree = Tree::new(nodes).unwrap();
            assert!(tree.leaves() <= limit);
            assert_eq!(stats.cache_builds.load(Ordering::Relaxed), 0);
            assert_eq!(stats.cache_reuses.load(Ordering::Relaxed), 0);
        }
    }
}
