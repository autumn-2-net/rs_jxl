use super::ma_frontier::History;
use super::*;

fn fixture(records: usize, predictors: usize) -> TreeSamples {
    TreeSamples {
        properties: vec![
            PropertyColumn::Packed((0..records).map(|i| (i % 7) as u8).collect()),
            PropertyColumn::Wide((0..records).map(|i| ((i * 31) % 259) as u32).collect()),
            PropertyColumn::Packed((0..records).map(|i| u8::from(i < records / 4)).collect()),
            PropertyColumn::Packed(vec![0; records]),
        ],
        residuals: (0..predictors)
            .map(|p| {
                (0..records)
                    .map(|i| ResidualToken {
                        token: match p % 3 {
                            0 => [0, 64, 129, 255][i % 4],
                            1 => [0, 8, 17][i % 3],
                            _ => [1, 65, 201][i % 3],
                        },
                        nbits: (1 + (i + p) % 28) as u8,
                    })
                    .collect()
            })
            .collect(),
        counts: (0..records)
            .map(|i| [1, 2, 17, 257, u16::MAX][i % 5])
            .collect(),
        thresholds: vec![(0..6).collect(), (0..258).collect(), vec![0], Vec::new()],
    }
}

fn options(predictors: usize) -> TrainOptions {
    let mut options = TrainOptions::variable();
    options.predictors.truncate(predictors);
    options.properties = vec![6, 7, 8, 9];
    options.parallel = false;
    options
}

fn cache_for(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    budget: &Arc<HistogramBudget>,
    plan: &PredictorPlan,
    stride: Option<usize>,
) -> HistogramCache {
    let parents = parent_histograms(samples, indices);
    let mut cache = HistogramCache::reserve_with_stride(samples, &parents, budget, stride).unwrap();
    for (property, histogram) in cache.properties.iter_mut().enumerate() {
        *histogram = match &samples.properties[property] {
            PropertyColumn::Packed(values) => PropertyHistogram::collect_values_with_stride(
                samples, indices, &parents, property, values, plan, stride,
            )
            .unwrap(),
            PropertyColumn::Wide(values) => PropertyHistogram::collect_values_with_stride(
                samples, indices, &parents, property, values, plan, stride,
            )
            .unwrap(),
        };
    }
    cache
}

#[derive(Debug, PartialEq)]
struct PropertySnapshot {
    used: Vec<usize>,
    weights: Vec<u32>,
    predictors: Vec<(usize, Vec<u32>, Vec<u64>)>,
    valid: bool,
}

fn snapshot(histogram: &PropertyHistogram) -> PropertySnapshot {
    PropertySnapshot {
        used: histogram.used.clone(),
        weights: histogram.weights.clone(),
        predictors: histogram
            .predictors
            .iter()
            .map(|p| (p.symbols, p.counts.clone(), p.extra.clone()))
            .collect(),
        valid: histogram.valid,
    }
}

fn allocations(cache: &HistogramCache) -> Vec<usize> {
    let mut pointers = vec![cache.properties.as_ptr() as usize];
    for property in &cache.properties {
        pointers.extend([
            property.used.as_ptr() as usize,
            property.weights.as_ptr() as usize,
            property.predictors.as_ptr() as usize,
        ]);
        for predictor in &property.predictors {
            pointers.extend([
                predictor.counts.as_ptr() as usize,
                predictor.extra.as_ptr() as usize,
            ]);
        }
    }
    pointers
}

fn property_bin(samples: &Samples<impl SampleStorage>, property: usize, i: usize) -> usize {
    match &samples.properties[property] {
        PropertyColumn::Packed(values) => usize::from(values[i]),
        PropertyColumn::Wide(values) => values[i] as usize,
    }
}

fn assert_population(
    histogram: &PropertyHistogram,
    samples: &Samples<impl SampleStorage>,
    property: usize,
    indices: &[usize],
) {
    assert!(histogram.valid);
    let bins = samples.thresholds[property].len() + 1;
    let mut used = vec![0; bins];
    let mut weights = vec![0u32; bins];
    for &i in indices {
        let bin = property_bin(samples, property, i);
        used[bin] += 1;
        weights[bin] += u32::from(samples.counts[i]);
    }
    assert_eq!(histogram.used, used);
    assert_eq!(histogram.weights, weights);
    if histogram.predictors.is_empty() {
        assert!(used.iter().filter(|&&n| n != 0).count() <= 1);
        return;
    }
    assert_eq!(histogram.predictors.len(), samples.residuals.len());
    for (p, predictor) in histogram.predictors.iter().enumerate() {
        let mut counts = vec![0u32; bins * predictor.symbols];
        let mut extra = vec![0u64; bins];
        for &i in indices {
            let bin = property_bin(samples, property, i);
            let residual = samples.residuals[p][i];
            let weight = u32::from(samples.counts[i]);
            counts[bin * predictor.symbols + usize::from(residual.token)] += weight;
            extra[bin] += u64::from(weight) * u64::from(residual.nbits);
        }
        assert_eq!(predictor.counts, counts);
        assert_eq!(predictor.extra, extra);
    }
}

fn assert_cache_population(
    cache: &HistogramCache,
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
) {
    for (property, histogram) in cache.properties.iter().enumerate() {
        assert_population(histogram, samples, property, indices);
    }
}

fn pending(begin: usize, end: usize) -> PendingNode {
    PendingNode::new(TrainingNode {
        node: 9,
        begin,
        end,
        predictor: 1,
        multiplier: 7,
    })
}

#[test]
fn handoff_normalization_guards_redundant_invalid_and_constant_entries() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for parallel in [false, true] {
        let samples = fixture(2305, 3);
        let indices: Vec<_> = (0..samples.counts.len()).collect();
        let first = samples.counts.len() / 4;
        let end_removed = first + 1024;
        let mut options = options(3);
        options.parallel = parallel;
        options.property_dedup = true;
        let budget = HistogramBudget::new(32 << 20);
        let mut cache = cache_for(
            &samples,
            &indices,
            &budget,
            &PredictorPlan::identity(3),
            None,
        );

        // Reproduce a cached property becoming constant after only its marginals
        // were updated: its valid token tables still describe the ancestor.
        for property in [0, 1, 3] {
            cache.properties[property]
                .subtract(&samples, property, &indices[..first])
                .unwrap();
        }
        let PropertyColumn::Packed(values) = &samples.properties[2] else {
            unreachable!();
        };
        cache.properties[2]
            .subtract_marginals_values(&samples, &indices[..first], values)
            .unwrap();
        assert!(cache.properties[2].valid);
        assert_eq!(cache.properties[2].used, [indices.len() - first, 0]);
        assert_eq!(cache.properties[2].predictors.len(), 3);
        let mut redundant = snapshot(&cache.properties[2]);
        redundant.valid = false;

        // An invalid raced-out entry need not contain usable marginal tables.
        let invalid = &mut cache.properties[1];
        invalid.valid = false;
        invalid.used.clear();
        invalid.weights.clear();
        invalid.predictors.clear();
        let invalid = snapshot(invalid);
        assert!(cache.properties[3].valid);
        assert!(cache.properties[3].predictors.is_empty());
        let pointers = allocations(&cache);
        let bytes = budget.used.load(Ordering::Relaxed);

        pool.install(|| {
            cache.normalize_for_handoff(
                &samples,
                &indices[first..end_removed],
                &options,
                &[0, 0, 1, 0],
            )
        })
        .unwrap();
        assert_eq!(snapshot(&cache.properties[1]), invalid);
        assert_eq!(snapshot(&cache.properties[2]), redundant);
        for property in [0, 3] {
            assert_population(
                &cache.properties[property],
                &samples,
                property,
                &indices[end_removed..],
            );
        }
        assert_eq!(allocations(&cache), pointers);
        assert_eq!(budget.used.load(Ordering::Relaxed), bytes);
        drop(cache);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn handoff_normalization_subtracts_all_raw_predictors_after_representative_copy() {
    for predictors in [2, 3, 14] {
        for stride in [None, Some(264)] {
            let mut samples = fixture(257, predictors);
            samples.residuals[1] = samples.residuals[0].clone();
            let indices: Vec<_> = (0..samples.counts.len()).collect();
            assert!(
                indices
                    .iter()
                    .any(|&i| samples.properties[1].value(i) > 255)
            );
            let mut plan = PredictorPlan::identity(predictors);
            plan.representative[1] = 0;
            plan.active_count = 0;
            for predictor in 0..predictors {
                if predictor != 1 {
                    plan.active[plan.active_count] = predictor;
                    plan.active_count += 1;
                }
            }
            let budget = HistogramBudget::new(32 << 20);
            let mut cache = cache_for(&samples, &indices, &budget, &plan, stride);
            let pointers = allocations(&cache);
            let bytes = budget.used.load(Ordering::Relaxed);
            let options = options(predictors);
            let mut previous = 0;
            for end in [0, 37, 89, indices.len()] {
                cache
                    .normalize_for_handoff(&samples, &indices[previous..end], &options, &[])
                    .unwrap();
                assert_cache_population(&cache, &samples, &indices[end..]);
                assert_eq!(cache.stride, stride);
                assert_eq!(allocations(&cache), pointers);
                assert_eq!(budget.used.load(Ordering::Relaxed), bytes);
                previous = end;
            }
            drop(cache);
            assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn handoff_history_consumes_nonzero_sibling_before_nested_disjoint_rebasing() {
    let mut samples = fixture(384, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let options = options(3);
    let stats = TrainingStats::default();
    let budget = HistogramBudget::new(32 << 20);
    let mut source = pending(70, 355);
    source.range = [[3, 7], [11, 17]];
    source.redundant_properties = vec![0; 4];
    source.parents = Some(parent_histograms(&samples, &indices[70..355]));
    source.cache = Some(cache_for(
        &samples,
        &indices[17..355],
        &budget,
        &PredictorPlan::identity(3),
        None,
    ));
    source.removed = Some(17..70);
    let pointers = allocations(source.cache.as_ref().unwrap());
    let parent_pointer = source.parents.as_ref().unwrap().as_ptr();
    let bytes = budget.used.load(Ordering::Relaxed);
    let history = History::take(&mut source, &samples, &indices, &options, &stats).unwrap();
    assert!(source.parents.is_none());
    assert!(source.cache.is_none());
    assert!(source.decision.is_none());
    assert!(source.removed.is_none());
    assert_eq!(budget.used.load(Ordering::Relaxed), bytes);

    {
        let mut views = samples.ranges_mut(&[17..70, 70..355]).unwrap();
        let mut view = views.pop().unwrap();
        // The excluded sibling is no longer needed by the transferred cache.
        views[0].counts.fill(u16::MAX);
        let local: Vec<_> = (0..view.counts.len()).collect();
        let mut child = pending(0, local.len());
        child.job.node = 0;
        child.range = source.range;
        child.redundant_properties = std::mem::take(&mut source.redundant_properties);
        history.restore(&mut child);
        drop(source);
        assert_eq!(view.base(), 70);
        assert_eq!(child.parents.as_ref().unwrap().as_ptr(), parent_pointer);
        assert_eq!(
            child.parents.as_ref().unwrap(),
            &parent_histograms(&view, &local)
        );
        assert_eq!(allocations(child.cache.as_ref().unwrap()), pointers);
        assert_cache_population(child.cache.as_ref().unwrap(), &view, &local);
        assert!(child.removed.is_none());
        assert!(child.decision.is_none());
        assert_eq!(
            (child.job.node, child.job.predictor, child.job.multiplier),
            (0, 1, 7)
        );
        assert_eq!(child.range, [[3, 7], [11, 17]]);
        assert_eq!(child.redundant_properties, [0; 4]);

        child.job.begin = 33;
        child.parents = Some(parent_histograms(&view, &local[33..]));
        child.removed = Some(0..33);
        let history = History::take(&mut child, &view, &local, &options, &stats).unwrap();
        let mut nested = view.ranges_mut(&[0..33, 33..local.len()]).unwrap();
        let grandchild_view = nested.pop().unwrap();
        let nested_indices: Vec<_> = (0..grandchild_view.counts.len()).collect();
        let mut grandchild = pending(0, nested_indices.len());
        grandchild.job.node = 0;
        grandchild.range = child.range;
        grandchild.redundant_properties = std::mem::take(&mut child.redundant_properties);
        history.restore(&mut grandchild);
        drop(child);
        assert_eq!(grandchild_view.base(), 103);
        assert_eq!(
            grandchild.parents.as_ref().unwrap(),
            &parent_histograms(&grandchild_view, &nested_indices)
        );
        assert_cache_population(
            grandchild.cache.as_ref().unwrap(),
            &grandchild_view,
            &nested_indices,
        );
        assert_eq!(allocations(grandchild.cache.as_ref().unwrap()), pointers);
        assert!(Arc::ptr_eq(
            &grandchild.cache.as_ref().unwrap()._lease.budget,
            &budget
        ));
        assert!(grandchild.removed.is_none());
        assert_eq!(budget.used.load(Ordering::Relaxed), bytes);
        drop(grandchild);
    }
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    assert_eq!(stats.histogram_handoffs.load(Ordering::Relaxed), 2);
    assert_eq!(stats.cache_builds.load(Ordering::Relaxed), 0);
}

#[test]
fn handoff_history_preserves_unevaluated_terminal_and_ready_decisions() {
    let samples = fixture(32, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let options = options(3);
    let stats = TrainingStats::default();
    let split = Split {
        property: 0,
        bin: 2,
        below_count: indices
            .iter()
            .filter(|&&i| samples.properties[0].value(i) <= 2)
            .count(),
        below_predictor: 1,
        above_predictor: 2,
        bits: 123.25,
    };
    for decision in [None, Some(None), Some(Some(split))] {
        let mut source = pending(0, indices.len());
        source.parents = Some(parent_histograms(&samples, &indices));
        source.decision = decision;
        let history = History::take(&mut source, &samples, &indices, &options, &stats).unwrap();
        let mut destination = pending(0, indices.len());
        destination.job.node = 0;
        history.restore(&mut destination);
        assert!(source.parents.is_none());
        assert!(source.decision.is_none());
        assert_eq!(
            destination.parents.as_ref().unwrap(),
            &parent_histograms(&samples, &indices)
        );
        match (destination.decision, decision) {
            (None, None) | (Some(None), Some(None)) => {}
            (Some(Some(actual)), Some(Some(expected))) => {
                assert_eq!(actual.property, expected.property);
                assert_eq!(actual.bin, expected.bin);
                assert_eq!(actual.below_count, expected.below_count);
                assert_eq!(actual.below_predictor, expected.below_predictor);
                assert_eq!(actual.above_predictor, expected.above_predictor);
                assert_eq!(actual.bits.to_bits(), expected.bits.to_bits());
            }
            _ => panic!("handoff changed the pending decision state"),
        }
    }
    assert_eq!(stats.histogram_handoffs.load(Ordering::Relaxed), 0);
}

#[test]
fn handoff_histories_keep_one_shared_budget_and_release_each_moved_lease_once() {
    let samples = fixture(64, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let options = options(3);
    let stats = TrainingStats::default();
    let measuring = HistogramBudget::new(32 << 20);
    let probe = cache_for(
        &samples,
        &indices,
        &measuring,
        &PredictorPlan::identity(3),
        None,
    );
    let bytes = probe._lease.bytes;
    drop(probe);
    assert_eq!(measuring.used.load(Ordering::Relaxed), 0);
    let budget = HistogramBudget::new(2 * bytes);
    let weak = Arc::downgrade(&budget);
    let mut root = pending(0, indices.len());
    root.cache = Some(cache_for(
        &samples,
        &indices,
        &budget,
        &PredictorPlan::identity(3),
        None,
    ));
    let mut child = pending(32, indices.len());
    child.cache = Some(cache_for(
        &samples,
        &indices[32..],
        &budget,
        &PredictorPlan::identity(3),
        None,
    ));
    assert_eq!(child.cache.as_ref().unwrap()._lease.bytes, bytes);
    let child_pointers = allocations(child.cache.as_ref().unwrap());
    let root_history = History::take(&mut root, &samples, &indices, &options, &stats).unwrap();
    let child_history = History::take(&mut child, &samples, &indices, &options, &stats).unwrap();
    drop((root, child));
    assert_eq!(budget.used.load(Ordering::Relaxed), 2 * bytes);
    assert!(budget.reserve(1).is_none());
    drop(budget);
    drop(root_history);

    let observer = weak.upgrade().unwrap();
    assert_eq!(observer.used.load(Ordering::Relaxed), bytes);
    let spare = observer.reserve(bytes).unwrap();
    assert!(observer.reserve(1).is_none());
    drop(spare);
    let child_history =
        std::thread::scope(|scope| scope.spawn(move || child_history).join().unwrap());
    let mut restored = pending(32, indices.len());
    child_history.restore(&mut restored);
    assert_eq!(
        allocations(restored.cache.as_ref().unwrap()),
        child_pointers
    );
    assert!(Arc::ptr_eq(
        &restored.cache.as_ref().unwrap()._lease.budget,
        &observer
    ));
    let history = History::take(&mut restored, &samples, &indices, &options, &stats).unwrap();
    drop(restored);
    assert_eq!(observer.used.load(Ordering::Relaxed), bytes);
    std::thread::scope(|scope| scope.spawn(move || drop(history)).join().unwrap());
    assert_eq!(observer.used.load(Ordering::Relaxed), 0);
    assert_eq!(observer.peak.load(Ordering::Relaxed), 2 * bytes);
    drop(observer);
    assert!(weak.upgrade().is_none());
}

#[test]
fn handoff_history_underflow_discards_partial_cache_before_returning() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let samples = fixture(2305, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    for parallel in [false, true] {
        for failure in 0..4 {
            let mut options = options(3);
            options.parallel = parallel;
            let stats = TrainingStats::default();
            let budget = HistogramBudget::new(32 << 20);
            let mut state = pending(1031, indices.len());
            let child_parents = parent_histograms(&samples, &indices[1031..]);
            state.parents = Some(child_parents.clone());
            state.cache = Some(cache_for(
                &samples,
                &indices[7..],
                &budget,
                &PredictorPlan::identity(3),
                None,
            ));
            state.removed = Some(7..1031);
            let bin = samples.properties[1].value(7) as usize;
            let property = &mut state.cache.as_mut().unwrap().properties[1];
            match failure {
                0 => property.used[bin] = 0,
                1 => property.weights[bin] = 0,
                2 => {
                    let predictor = &mut property.predictors[2];
                    let token = usize::from(samples.residuals[2][7].token);
                    predictor.counts[bin * predictor.symbols + token] = 0;
                }
                3 => property.predictors[2].extra[bin] = 0,
                _ => unreachable!(),
            }
            // Keep the state alive: cleanup must not depend on dropping its owner.
            let result = pool.install(|| {
                History::take(&mut state, &samples, &indices, &options, &stats).map(drop)
            });
            let error = result.unwrap_err();
            assert!(matches!(&error, Error::Encoding(_)));
            let kind = ["record", "weight", "token", "extra-bit"][failure];
            assert!(
                error
                    .to_string()
                    .contains(&format!("MA cached {kind} subtraction underflow"))
            );
            assert!(state.cache.is_none());
            assert!(state.removed.is_none());
            assert!(state.decision.is_none());
            assert_eq!(state.parents.as_ref().unwrap(), &child_parents);
            assert_eq!(budget.used.load(Ordering::Relaxed), 0);
            assert_eq!(Arc::strong_count(&budget), 1);
            assert_eq!(stats.histogram_handoffs.load(Ordering::Relaxed), 0);

            // Rebuild only the live child; the failed ancestor cache and its
            // deferred sibling removal must not participate in the next handoff.
            state.cache = Some(cache_for(
                &samples,
                &indices[1031..],
                &budget,
                &PredictorPlan::identity(3),
                None,
            ));
            let history = pool
                .install(|| History::take(&mut state, &samples, &indices, &options, &stats))
                .unwrap();
            history.restore(&mut state);
            assert_eq!(state.parents.as_ref().unwrap(), &child_parents);
            assert!(state.removed.is_none());
            assert_cache_population(state.cache.as_ref().unwrap(), &samples, &indices[1031..]);
            drop(state);
            assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn handoff_history_normalization_unwind_discards_cache_before_owner_drop() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let samples = fixture(2305, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    for parallel in [false, true] {
        let mut options = options(3);
        options.parallel = parallel;
        let stats = TrainingStats::default();
        let budget = HistogramBudget::new(32 << 20);
        let mut state = pending(1031, indices.len());
        let child_parents = parent_histograms(&samples, &indices[1031..]);
        state.parents = Some(child_parents.clone());
        state.cache = Some(cache_for(
            &samples,
            &indices[7..],
            &budget,
            &PredictorPlan::identity(3),
            None,
        ));
        state.removed = Some(7..1031);
        // Inject a checked-index panic in an unmasked valid property after the
        // cache has moved out of state, rather than after a successful handoff.
        let histogram = &mut state.cache.as_mut().unwrap().properties[1];
        assert!(histogram.valid);
        histogram.used.clear();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.install(|| {
                History::take(&mut state, &samples, &indices, &options, &stats).map(drop)
            })
        }));
        assert!(result.is_err());
        assert!(state.cache.is_none());
        assert!(state.removed.is_none());
        assert!(state.decision.is_none());
        assert_eq!(state.parents.as_ref().unwrap(), &child_parents);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
        assert_eq!(Arc::strong_count(&budget), 1);
        assert_eq!(stats.histogram_handoffs.load(Ordering::Relaxed), 0);
        drop(state);
        assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn handoff_history_unwind_drops_transferred_cache_and_releases_budget() {
    let samples = fixture(64, 3);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let options = options(3);
    let stats = TrainingStats::default();
    let budget = HistogramBudget::new(32 << 20);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut state = pending(17, indices.len());
        state.cache = Some(cache_for(
            &samples,
            &indices,
            &budget,
            &PredictorPlan::identity(3),
            None,
        ));
        state.removed = Some(0..17);
        let _history = History::take(&mut state, &samples, &indices, &options, &stats).unwrap();
        assert!(state.cache.is_none());
        assert!(budget.used.load(Ordering::Relaxed) > 0);
        panic!("injected handoff owner unwind");
    }));
    assert!(result.is_err());
    assert_eq!(budget.used.load(Ordering::Relaxed), 0);
    assert_eq!(Arc::strong_count(&budget), 1);
}
