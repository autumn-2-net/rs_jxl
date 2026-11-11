use super::super::{TreeSamples, parent_histograms};
use super::*;

fn fixture(records: usize, predictors: usize, wide: bool) -> TreeSamples {
    let bins = if wide { 259 } else { 17 };
    let values: Vec<u32> = (0..records).map(|i| ((i * 37) % bins) as u32).collect();
    TreeSamples {
        global_max_symbols: 0,
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
                            1 => [0, 3, 9, 17][(i + p) % 4],
                            _ => [0, 31, 65][(i + p) % 3],
                        },
                        nbits: (1 + (i + p) % 31) as u8,
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

fn collect<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    plan: &PredictorPlan,
    stride: Option<usize>,
) -> PropertyHistogram {
    let parents = parent_histograms(samples, indices);
    PropertyHistogram::collect_values_with_stride(
        samples, indices, &parents, 0, values, plan, stride,
    )
    .unwrap()
}

// Original dispatch-independent record-major subtraction is the oracle.
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

fn check_success<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    plan: &PredictorPlan,
    stride: Option<usize>,
) {
    let mut actual = collect(samples, indices, values, plan, stride);
    let mut expected = collect(samples, indices, values, plan, stride);
    let allocations: Vec<_> = actual
        .predictors
        .iter()
        .map(|h| (h.counts.as_ptr(), h.extra.as_ptr()))
        .collect();
    actual.subtract_values(samples, &[], values, plan).unwrap();
    assert_same(&actual, &expected);
    for removed in indices.chunks(97) {
        actual
            .subtract_values(samples, removed, values, plan)
            .unwrap();
        reference_subtract(&mut expected, samples, removed, values, plan).unwrap();
        assert_same(&actual, &expected);
        for (histogram, &(counts, extra)) in actual.predictors.iter().zip(&allocations) {
            assert_eq!(histogram.counts.as_ptr(), counts);
            assert_eq!(histogram.extra.as_ptr(), extra);
        }
    }
    assert!(actual.used.iter().all(|&n| n == 0));
    assert!(actual.weights.iter().all(|&n| n == 0));
    for histogram in &actual.predictors {
        assert!(histogram.counts.iter().all(|&n| n == 0));
        assert!(histogram.extra.iter().all(|&n| n == 0));
    }
}

#[test]
fn identity14_subtraction_matches_reference_with_weighted_strides_and_empty_removals() {
    let plan = PredictorPlan::identity(14);
    assert!(plan.cache_pair14_eligible(14));
    for wide in [false, true] {
        for records in [0, 1, 2, 31, 257] {
            let samples = fixture(records, 14, wide);
            let indices: Vec<_> = (0..records).rev().collect();
            with_property_values!(&samples.properties[0], |values| {
                for stride in [None, Some(264)] {
                    check_success(&samples, &indices, values, &plan, stride);
                }
            });
        }
    }
}

#[test]
fn identity14_subtraction_accepts_borrowed_columns_and_preserves_zero_tails() {
    let plan = PredictorPlan::identity(14);
    for wide in [false, true] {
        let mut samples = fixture(529, 14, wide);
        for view in samples.ranges_mut(&[3..42, 44..529]).unwrap() {
            assert_ne!(view.base(), 0);
            let indices: Vec<_> = (0..view.counts.len()).rev().collect();
            with_property_values!(&view.properties[0], |values| {
                for stride in [None, Some(264)] {
                    check_success(&view, &indices, values, &plan, stride);
                }
            });
        }
    }
}

#[test]
fn identity14_subtraction_preserves_each_underflow_error_and_partial_state() {
    let samples = fixture(8, 14, false);
    let plan = PredictorPlan::identity(14);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    with_property_values!(&samples.properties[0], |values| {
        for stride in [None, Some(264)] {
            // Marginals, then token/extra-bit failures for every predictor.
            for failure in 0..30 {
                let mut actual = collect(&samples, &indices, values, &plan, stride);
                let mut expected = collect(&samples, &indices, values, &plan, stride);
                let i = 2;
                let bin = values[i].index();
                for histogram in [&mut actual, &mut expected] {
                    match failure {
                        0 => histogram.used[bin] = 0,
                        1 => histogram.weights[bin] = 0,
                        _ => {
                            let predictor = (failure - 2) / 2;
                            let residual = samples.residuals[predictor][i];
                            let histogram = &mut histogram.predictors[predictor];
                            if failure % 2 == 0 {
                                histogram.counts
                                    [bin * histogram.symbols + usize::from(residual.token)] = 0;
                            } else {
                                histogram.extra[bin] = 0;
                            }
                        }
                    }
                }
                let actual_error = actual
                    .subtract_values(&samples, &[1, i, 3], values, &plan)
                    .unwrap_err()
                    .to_string();
                let expected_error =
                    reference_subtract(&mut expected, &samples, &[1, i, 3], values, &plan)
                        .unwrap_err()
                        .to_string();
                let kind = match failure {
                    0 => "record",
                    1 => "weight",
                    n if n % 2 == 0 => "token",
                    _ => "extra-bit",
                };
                assert!(actual_error.contains(&format!("MA cached {kind} subtraction underflow")));
                assert_eq!(actual_error, expected_error);
                assert_same(&actual, &expected);
            }
        }
    });
}

#[test]
fn identity14_subtraction_falls_back_for_other_widths_aliases_and_active_orders() {
    for predictors in [1, 3, 13, 14] {
        for aliases in [false, true] {
            for reordered in [false, true] {
                let mut samples = fixture(97, predictors, false);
                let mut plan = PredictorPlan::identity(predictors);
                if aliases && predictors > 1 {
                    samples.residuals[predictors - 1] = samples.residuals[0].clone();
                    plan.representative[predictors - 1] = 0;
                    plan.active_count -= 1;
                }
                if reordered {
                    plan.active[..plan.active_count].reverse();
                }
                if predictors == 14 && !aliases && !reordered {
                    continue;
                }
                assert!(!plan.cache_pair14_eligible(predictors));
                let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
                with_property_values!(&samples.properties[0], |values| {
                    check_success(&samples, &indices, values, &plan, None);
                });
            }
        }
    }
}

#[test]
fn identity14_subtraction_fallback_preserves_reordered_failure_state() {
    let samples = fixture(8, 14, false);
    let indices: Vec<_> = (0..samples.counts.len()).collect();
    let mut plan = PredictorPlan::identity(14);
    plan.active.reverse();
    assert!(!plan.cache_pair14_eligible(14));
    with_property_values!(&samples.properties[0], |values| {
        let mut actual = collect(&samples, &indices, values, &plan, None);
        let mut expected = collect(&samples, &indices, values, &plan, None);
        let bin = values[2].index();
        for histogram in [&mut actual, &mut expected] {
            let first = &mut histogram.predictors[0];
            first.counts[bin * first.symbols + usize::from(samples.residuals[0][2].token)] = 0;
            histogram.predictors[13].extra[bin] = 0;
        }
        let error = actual
            .subtract_values(&samples, &[1, 2, 3], values, &plan)
            .unwrap_err()
            .to_string();
        let reference = reference_subtract(&mut expected, &samples, &[1, 2, 3], values, &plan)
            .unwrap_err()
            .to_string();
        assert!(error.contains("MA cached extra-bit subtraction underflow"));
        assert_eq!(error, reference);
        assert_same(&actual, &expected);
    });
}
