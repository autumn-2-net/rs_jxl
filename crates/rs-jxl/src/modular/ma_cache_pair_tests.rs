use super::*;

fn fixture(records: usize, predictors: usize, wide: bool) -> TreeSamples {
    let bins = if wide { 513 } else { 256 };
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
                        token: if p % 2 == 0 {
                            [0, 64, 129, 255][i % 4]
                        } else {
                            [0, 8, 17][i % 3]
                        },
                        nbits: (1 + (i + p) % 28) as u8,
                    })
                    .collect()
            })
            .collect(),
        counts: (0..records)
            .map(|i| [1, 2, u16::MAX, 17, 257][i % 5])
            .collect(),
        thresholds: vec![(0..bins - 1).map(|i| i as i32).collect()],
    }
}

// Keep the original record-major loops as an independent dispatch oracle.
fn reference_collect<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parents: &[ParentHistogram],
    values: &[T],
    plan: &PredictorPlan,
    stride: Option<usize>,
) -> PropertyHistogram {
    let mut result = PropertyHistogram::collect_marginals_values(samples, indices, 0, values);
    if result.allocate_predictors(parents, stride) {
        for &i in indices {
            let bin = values[i].index();
            let weight = u32::from(samples.counts[i]);
            for &predictor in plan.active() {
                let histogram = &mut result.predictors[predictor];
                let residual = samples.residuals[predictor][i];
                histogram.counts[bin * histogram.symbols + usize::from(residual.token)] += weight;
                histogram.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
            }
        }
        result.copy_predictors(plan).unwrap();
        result.valid = true;
    }
    result
}

fn reference_subtract<T: PropertyValue>(
    histogram: &mut PropertyHistogram,
    samples: &Samples<impl SampleStorage>,
    removed: &[usize],
    values: &[T],
    plan: &PredictorPlan,
) -> Result<()> {
    for &i in removed {
        let bin = values[i].index();
        let weight = u32::from(samples.counts[i]);
        histogram.subtract_marginal(bin, weight)?;
        histogram.subtract_predictors(samples, i, bin, weight, plan)?;
    }
    histogram.copy_predictors(plan)
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
fn paired_cache_collection_matches_record_major_reference() {
    for wide in [false, true] {
        for records in [0, 1, 2, 31, 255, 257, 529] {
            let samples = fixture(records, 2, wide);
            let indices: Vec<_> = (0..records).rev().filter(|i| i % 7 != 3).collect();
            let parents = parent_histograms(&samples, &indices);
            let plan = PredictorPlan::identity(2);
            with_property_values!(&samples.properties[0], |values| {
                for stride in [None, Some(256)] {
                    let actual = PropertyHistogram::collect_values_with_stride(
                        &samples, &indices, &parents, 0, values, &plan, stride,
                    )
                    .unwrap();
                    let expected =
                        reference_collect(&samples, &indices, &parents, values, &plan, stride);
                    assert_same(&actual, &expected);
                }
            });
        }
    }
}

#[test]
fn paired_cache_subtraction_matches_reference_through_empty_cache() {
    for wide in [false, true] {
        let samples = fixture(529, 2, wide);
        let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
        let parents = parent_histograms(&samples, &indices);
        let plan = PredictorPlan::identity(2);
        with_property_values!(&samples.properties[0], |values| {
            for stride in [None, Some(256)] {
                let mut actual = PropertyHistogram::collect_values_with_stride(
                    &samples, &indices, &parents, 0, values, &plan, stride,
                )
                .unwrap();
                let mut expected =
                    reference_collect(&samples, &indices, &parents, values, &plan, stride);
                actual
                    .subtract_values(&samples, &[], values, &plan)
                    .unwrap();
                assert_same(&actual, &expected);
                for removed in indices.chunks(37) {
                    actual
                        .subtract_values(&samples, removed, values, &plan)
                        .unwrap();
                    reference_subtract(&mut expected, &samples, removed, values, &plan).unwrap();
                    assert_same(&actual, &expected);
                }
                assert!(actual.used.iter().all(|&n| n == 0));
                assert!(actual.weights.iter().all(|&n| n == 0));
                for histogram in &actual.predictors {
                    assert!(histogram.counts.iter().all(|&n| n == 0));
                    assert!(histogram.extra.iter().all(|&n| n == 0));
                }
            }
        });
    }
}

#[test]
fn paired_cache_underflow_preserves_error_and_partial_state_order() {
    let samples = fixture(32, 2, false);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let parents = parent_histograms(&samples, &indices);
    let plan = PredictorPlan::identity(2);
    with_property_values!(&samples.properties[0], |values| {
        for failure in 0..6 {
            let mut actual = reference_collect(&samples, &indices, &parents, values, &plan, None);
            let mut expected = reference_collect(&samples, &indices, &parents, values, &plan, None);
            let i = 2;
            let bin = values[i].index();
            for histogram in [&mut actual, &mut expected] {
                match failure {
                    0 => histogram.used[bin] = 0,
                    1 => histogram.weights[bin] = 0,
                    2 | 4 => {
                        let p = (failure - 2) / 2;
                        let token = usize::from(samples.residuals[p][i].token);
                        let histogram = &mut histogram.predictors[p];
                        histogram.counts[bin * histogram.symbols + token] = 0;
                    }
                    3 | 5 => histogram.predictors[(failure - 3) / 2].extra[bin] = 0,
                    _ => unreachable!(),
                }
            }
            let actual_error = actual
                .subtract_values(&samples, &[1, i], values, &plan)
                .unwrap_err()
                .to_string();
            let expected_error =
                reference_subtract(&mut expected, &samples, &[1, i], values, &plan)
                    .unwrap_err()
                    .to_string();
            assert_eq!(actual_error, expected_error);
            let kind = [
                "record",
                "weight",
                "token",
                "extra-bit",
                "token",
                "extra-bit",
            ][failure];
            assert!(actual_error.contains(&format!("MA cached {kind} subtraction underflow")));
            assert_same(&actual, &expected);
        }
    });
}

#[test]
fn paired_cache_falls_back_for_other_predictor_counts_and_representatives() {
    for predictors in [1, 2, 3, 14] {
        for dedup in [false, true] {
            let mut samples = fixture(97, predictors, false);
            let mut plan = PredictorPlan::identity(predictors);
            if dedup && predictors >= 2 {
                samples.residuals[1] = samples.residuals[0].clone();
                plan.representative[1] = 0;
                plan.active_count = 1;
                for predictor in 2..predictors {
                    plan.active[plan.active_count] = predictor;
                    plan.active_count += 1;
                }
            }
            let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
            let parents = parent_histograms(&samples, &indices);
            with_property_values!(&samples.properties[0], |values| {
                let mut actual = PropertyHistogram::collect_values_with_stride(
                    &samples, &indices, &parents, 0, values, &plan, None,
                )
                .unwrap();
                let mut expected =
                    reference_collect(&samples, &indices, &parents, values, &plan, None);
                assert_same(&actual, &expected);
                for removed in indices.chunks(13) {
                    actual
                        .subtract_values(&samples, removed, values, &plan)
                        .unwrap();
                    reference_subtract(&mut expected, &samples, removed, values, &plan).unwrap();
                    assert_same(&actual, &expected);
                }
            });
        }
    }
}

#[test]
fn paired_cache_constant_property_retains_only_marginals() {
    for records in [0, 37] {
        let mut samples = fixture(records, 2, false);
        samples.properties[0] = PropertyColumn::Packed(vec![0; records]);
        let indices: Vec<_> = (0..records).collect();
        let parents = parent_histograms(&samples, &indices);
        let plan = PredictorPlan::identity(2);
        with_property_values!(&samples.properties[0], |values| {
            let mut actual = PropertyHistogram::collect_values_with_stride(
                &samples, &indices, &parents, 0, values, &plan, None,
            )
            .unwrap();
            let mut expected = reference_collect(&samples, &indices, &parents, values, &plan, None);
            assert!(actual.predictors.is_empty());
            actual
                .subtract_values(&samples, &indices, values, &plan)
                .unwrap();
            reference_subtract(&mut expected, &samples, &indices, values, &plan).unwrap();
            assert_same(&actual, &expected);
        });
    }
}

#[test]
fn paired_cache_accepts_disjoint_borrowed_subtree_columns() {
    let mut samples = fixture(529, 2, true);
    let views = samples.ranges_mut(&[3..42, 44..529]).unwrap();
    for samples in views {
        let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
        let parents = parent_histograms(&samples, &indices);
        let plan = PredictorPlan::identity(2);
        with_property_values!(&samples.properties[0], |values| {
            let mut actual = PropertyHistogram::collect_values_with_stride(
                &samples, &indices, &parents, 0, values, &plan, None,
            )
            .unwrap();
            let mut expected = reference_collect(&samples, &indices, &parents, values, &plan, None);
            assert_same(&actual, &expected);
            actual
                .subtract_values(&samples, &indices, values, &plan)
                .unwrap();
            reference_subtract(&mut expected, &samples, &indices, values, &plan).unwrap();
            assert_same(&actual, &expected);
        });
    }
}
