use super::*;

fn fixture(records: usize, predictors: usize, wide: bool) -> TreeSamples {
    let bins = if wide { 259 } else { 17 };
    let values: Vec<u32> = (0..records).map(|i| ((i * 37) % bins) as u32).collect();
    TreeSamples {
        properties: vec![if wide {
            PropertyColumn::Wide(values)
        } else {
            PropertyColumn::Packed(values.into_iter().map(|v| v as u8).collect())
        }],
        residuals: (0..predictors)
            .map(|p| {
                (0..records)
                    .map(|i| ResidualToken {
                        token: match p % 3 {
                            0 => [0, 64, 129, 255][(i + p) % 4],
                            1 => [0, 3, 9, 17][(i / 3 + p) % 4],
                            _ => [0, 31, 65][(i + p) % 3],
                        },
                        nbits: ((i + p) % 29) as u8,
                    })
                    .collect()
            })
            .collect(),
        counts: (0..records)
            .map(|i| [1, u16::MAX, 2, 17, 257][i % 5])
            .collect(),
        thresholds: vec![(0..bins - 1).map(|i| i as i32).collect()],
    }
}

fn collect<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    plan: &PredictorPlan,
    stride: Option<usize>,
    pair14: bool,
) -> PropertyHistogram {
    let parents = parent_histograms(samples, indices);
    let mut histogram = PropertyHistogram::collect_marginals_values(samples, indices, 0, values);
    histogram
        .collect_predictors_values_with_pair14(
            samples, indices, &parents, values, plan, stride, pair14,
        )
        .unwrap();
    histogram
}

fn assert_same(actual: &PropertyHistogram, expected: &PropertyHistogram) {
    assert_eq!(actual.used, expected.used);
    assert_eq!(actual.weights, expected.weights);
    assert_eq!(actual.valid, expected.valid);
    assert_eq!(actual.predictors.len(), expected.predictors.len());
    for (actual, expected) in actual.predictors.iter().zip(&expected.predictors) {
        assert_eq!(actual.symbols, expected.symbols);
        assert_eq!(actual.counts, expected.counts);
        assert_eq!(actual.extra, expected.extra);
    }
}

#[test]
fn pair14_exact_bins_match_old_record_major_kernel() {
    let plan = PredictorPlan::identity(14);
    assert!(plan.cache_pair14_eligible(14));
    for wide in [false, true] {
        for records in [0, 1, 2, 7, 255, 257, 513] {
            let samples = fixture(records, 14, wide);
            let indices: Vec<_> = (0..records).rev().filter(|i| i % 5 != 2).collect();
            with_property_values!(&samples.properties[0], |values| {
                for stride in [None, Some(264)] {
                    let actual = collect(&samples, &indices, values, &plan, stride, true);
                    // Explicit dispatch avoids mutating process environment or
                    // depending on the runtime OnceLock in parallel tests.
                    let expected = collect(&samples, &indices, values, &plan, stride, false);
                    assert_same(&actual, &expected);
                    assert_eq!(actual.used.iter().sum::<usize>(), indices.len());
                    assert_eq!(
                        actual.weights.iter().sum::<u32>(),
                        indices
                            .iter()
                            .map(|&i| u32::from(samples.counts[i]))
                            .sum::<u32>()
                    );
                }
            });
        }
    }
}

#[test]
fn pair14_accepts_borrowed_columns_with_nonzero_range_base() {
    let mut samples = fixture(513, 14, true);
    let plan = PredictorPlan::identity(14);
    let views = samples.ranges_mut(&[3..42, 44..513]).unwrap();
    for samples in views {
        assert_ne!(samples.base(), 0);
        let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
        with_property_values!(&samples.properties[0], |values| {
            let actual = collect(&samples, &indices, values, &plan, None, true);
            let expected = collect(&samples, &indices, values, &plan, None, false);
            assert_same(&actual, &expected);
        });
    }
}

#[test]
fn pair14_aliases_and_nonidentity_active_order_use_fallback() {
    for wide in [false, true] {
        for aliases in [1, 7, 13] {
            let mut samples = fixture(65, 14, wide);
            let mut plan = PredictorPlan::identity(14);
            for p in 1..14 {
                let representative = match aliases {
                    1 if p == 13 => 0,
                    7 if p % 2 == 1 => p - 1,
                    13 => 0,
                    _ => p,
                };
                plan.representative[p] = representative;
                if representative != p {
                    samples.residuals[p] = samples.residuals[representative].clone();
                }
            }
            plan.active_count = 0;
            for p in 0..14 {
                if plan.representative[p] == p {
                    plan.active[plan.active_count] = p;
                    plan.active_count += 1;
                }
            }
            assert!(!plan.cache_pair14_eligible(14));
            let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
            with_property_values!(&samples.properties[0], |values| {
                let actual = collect(&samples, &indices, values, &plan, Some(264), true);
                let expected = collect(&samples, &indices, values, &plan, Some(264), false);
                assert_same(&actual, &expected);
                for (p, histogram) in actual.predictors.iter().enumerate() {
                    let representative = &actual.predictors[plan.representative[p]];
                    assert_eq!(histogram.counts, representative.counts);
                    assert_eq!(histogram.extra, representative.extra);
                }
            });
        }
    }
    let samples = fixture(65, 14, false);
    let mut plan = PredictorPlan::identity(14);
    plan.active.rotate_left(1);
    assert!(!plan.cache_pair14_eligible(14));
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    with_property_values!(&samples.properties[0], |values| {
        let actual = collect(&samples, &indices, values, &plan, None, true);
        let expected = collect(&samples, &indices, values, &plan, None, false);
        assert_same(&actual, &expected);
    });
    let mut plan = PredictorPlan::identity(14);
    plan.representative[13] = 0;
    assert!(!plan.cache_pair14_eligible(14));
}

#[test]
fn pair14_other_predictor_widths_keep_existing_dispatch() {
    for predictors in [1, 2, 3, 13] {
        let samples = fixture(65, predictors, false);
        let plan = PredictorPlan::identity(predictors);
        assert!(!plan.cache_pair14_eligible(predictors));
        let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
        with_property_values!(&samples.properties[0], |values| {
            let actual = collect(&samples, &indices, values, &plan, None, true);
            let expected = collect(&samples, &indices, values, &plan, None, false);
            assert_same(&actual, &expected);
        });
    }
}

#[test]
fn pair14_constant_or_empty_population_does_not_allocate_predictors() {
    let plan = PredictorPlan::identity(14);
    for wide in [false, true] {
        for records in [0, 1, 33] {
            let mut samples = fixture(records, 14, wide);
            samples.properties[0] = if wide {
                PropertyColumn::Wide(vec![258; records])
            } else {
                PropertyColumn::Packed(vec![16; records])
            };
            let indices: Vec<_> = (0..records).collect();
            with_property_values!(&samples.properties[0], |values| {
                let actual = collect(&samples, &indices, values, &plan, None, true);
                let expected = collect(&samples, &indices, values, &plan, None, false);
                assert_same(&actual, &expected);
                assert!(actual.valid);
                assert!(actual.predictors.is_empty());
            });
        }
    }
}
