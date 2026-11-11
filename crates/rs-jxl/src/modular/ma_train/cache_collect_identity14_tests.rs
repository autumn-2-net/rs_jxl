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
                            1 => [0, 3, 9, 17][(i / 3 + p) % 4],
                            _ => [0, 31, 65][(i + p) % 3],
                        },
                        nbits: ((i + p) % 33) as u8,
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

fn reference_collect<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    parents: &[ParentHistogram],
    values: &[T],
    plan: &PredictorPlan,
    stride: Option<usize>,
) -> PropertyHistogram {
    let mut histogram = PropertyHistogram::collect_marginals_values(samples, indices, 0, values);
    if !histogram.allocate_predictors(parents, stride) {
        return histogram;
    }
    // Original record-major collection, independent of optimized dispatch.
    for &i in indices {
        let bin = values[i].index();
        let weight = u32::from(samples.counts[i]);
        for &predictor in plan.active() {
            let h = &mut histogram.predictors[predictor];
            let residual = samples.residuals[predictor][i];
            h.counts[bin * h.symbols + usize::from(residual.token)] += weight;
            h.extra[bin] += u64::from(weight) * u64::from(residual.nbits);
        }
    }
    histogram.copy_predictors(plan).unwrap();
    histogram.valid = true;
    histogram
}

fn check<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    plan: &PredictorPlan,
) {
    let parents = parent_histograms(samples, indices);
    for stride in [None, Some(264)] {
        let expected = reference_collect(samples, indices, &parents, values, plan, stride);
        let mut actual = PropertyHistogram::collect_marginals_values(samples, indices, 0, values);
        // Explicitly disable the process-wide seven-pass experiment.
        actual
            .collect_predictors_values_with_pair14(
                samples, indices, &parents, values, plan, stride, false,
            )
            .unwrap();
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
}

#[test]
fn identity14_collection_matches_scalar_with_weighted_strides_and_repeated_indices() {
    let plan = PredictorPlan::identity(14);
    for wide in [false, true] {
        for records in [0, 1, 2, 31, 257, 513] {
            let samples = fixture(records, 14, wide);
            let indices: Vec<_> = (0..records)
                .rev()
                .filter(|i| i % 5 != 2)
                .flat_map(|i| [i, i])
                .collect();
            with_property_values!(&samples.properties[0], |values| {
                check(&samples, &indices, values, &plan);
            });
        }
    }
}

#[test]
fn identity14_collection_accepts_borrowed_columns_with_nonzero_range_base() {
    let plan = PredictorPlan::identity(14);
    for wide in [false, true] {
        let mut samples = fixture(529, 14, wide);
        for view in samples.ranges_mut(&[3..42, 44..529]).unwrap() {
            assert_ne!(view.base(), 0);
            let indices: Vec<_> = (0..view.counts.len()).rev().collect();
            with_property_values!(&view.properties[0], |values| {
                check(&view, &indices, values, &plan);
            });
        }
    }
}

#[test]
fn identity14_collection_fallbacks_match_scalar_for_aliases_order_and_widths() {
    for wide in [false, true] {
        for predictors in [1, 2, 3, 13, 14] {
            for alias in [false, true] {
                let mut samples = fixture(65, predictors, wide);
                let mut plan = PredictorPlan::identity(predictors);
                if alias && predictors > 1 {
                    let last = predictors - 1;
                    samples.residuals[last] = samples.residuals[0].clone();
                    plan.representative[last] = 0;
                    plan.active_count -= 1;
                }
                let indices: Vec<_> = (0..samples.counts.len()).rev().collect();
                for reverse in [false, true] {
                    if reverse {
                        plan.active[..plan.active_count].reverse();
                    }
                    with_property_values!(&samples.properties[0], |values| {
                        check(&samples, &indices, values, &plan);
                    });
                }
            }
        }
    }
}

#[test]
fn identity14_collection_constant_population_retains_only_marginals() {
    let plan = PredictorPlan::identity(14);
    for wide in [false, true] {
        let mut samples = fixture(33, 14, wide);
        samples.properties[0] = if wide {
            PropertyColumn::Wide(vec![258; 33])
        } else {
            PropertyColumn::Packed(vec![16; 33])
        };
        let indices: Vec<_> = (0..33).collect();
        with_property_values!(&samples.properties[0], |values| {
            check(&samples, &indices, values, &plan);
            let parents = parent_histograms(&samples, &indices);
            let actual = PropertyHistogram::collect_values_with_stride(
                &samples, &indices, &parents, 0, values, &plan, None,
            )
            .unwrap();
            assert!(actual.valid);
            assert!(actual.predictors.is_empty());
        });
    }
}
