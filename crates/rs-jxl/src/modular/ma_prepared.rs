// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! Ownership handoff for enc_ma.h::TreeSamples/AllSamplesDone, frozen b866b3d.

use super::super::training::gather_samples::{GatherColumn, PreparedSamples};
use super::{
    MAX_HISTOGRAM_TOTAL, Predictor, PropertyColumn, PropertyQuantization, ResidualToken,
    TrainOptions, Tree, TreeSamples, train_nodes, train_with_constants,
};
use crate::{Error, Result};

// Separate the documented column contract from gather's temporary adapter fields.
pub(super) struct PreparedInput {
    predictors: Vec<Predictor>,
    quantization: PropertyQuantization,
    properties: Vec<GatherColumn>,
    residuals: Vec<Vec<[u8; 2]>>,
    counts: Vec<u16>,
    total: u64,
}

impl From<PreparedSamples> for PreparedInput {
    fn from(samples: PreparedSamples) -> Self {
        Self {
            predictors: samples.predictors,
            quantization: samples.quantization,
            properties: samples.properties,
            residuals: samples.residuals,
            counts: samples.counts,
            total: samples.total,
        }
    }
}

impl PreparedInput {
    fn into_tree_samples(self, options: &TrainOptions) -> Result<TreeSamples> {
        options.validate()?;
        if !options.multiplier_info.is_empty() {
            return Err(Error::InvalidInput(
                "MA prepared samples cannot validate raw multiplier constraints".into(),
            ));
        }
        if self.predictors != options.predictors
            || options.property_quantization.as_ref() != Some(&self.quantization)
            || self.properties.len() != options.properties.len()
            || self.residuals.len() != options.predictors.len()
        {
            return Err(Error::InvalidInput("MA prepared schema mismatch".into()));
        }
        if self.total > u64::from(MAX_HISTOGRAM_TOTAL) {
            return Err(Error::Limit("MA sample weight exceeds source i32::MAX"));
        }
        let rows = self.counts.len();
        let mut total = 0u32;
        for &count in &self.counts {
            if count == 0 {
                return Err(Error::InvalidInput("zero MA prepared multiplicity".into()));
            }
            total = total
                .checked_add(u32::from(count))
                .filter(|&sum| sum <= MAX_HISTOGRAM_TOTAL)
                .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
        }
        if u64::from(total) != self.total {
            return Err(Error::InvalidInput("MA prepared total mismatch".into()));
        }
        let variable_range = options.static_property_range.map(|mut range| {
            if let Some(first) = options.constant_channels.first() {
                range[0][1] = first.channel;
            }
            range
        });
        if rows != 0 && variable_range.is_some_and(|range| range[0][0] == range[0][1]) {
            return Err(Error::InvalidInput(
                "MA prepared records supplied for an all-constant domain".into(),
            ));
        }
        for ((&id, thresholds), column) in options
            .properties
            .iter()
            .zip(&self.quantization.thresholds)
            .zip(&self.properties)
        {
            let valid = match column {
                GatherColumn::Packed(values) => {
                    id >= 2
                        && values.len() == rows
                        && values
                            .iter()
                            .all(|&bin| usize::from(bin) <= thresholds.len())
                }
                GatherColumn::Static(values) => {
                    id < 2
                        && values.len() == rows
                        && values.iter().all(|&bin| (bin as usize) <= thresholds.len())
                        && variable_range.is_none_or(|range| {
                            // Bins can straddle range edges. Reject disjoint bins,
                            // never invent a representative raw ID as proof.
                            let [begin, end] = range[id];
                            let first =
                                thresholds.partition_point(|&t| i64::from(t) < i64::from(begin));
                            let last =
                                thresholds.partition_point(|&t| i64::from(t) < i64::from(end) - 1);
                            values
                                .iter()
                                .all(|&bin| (first..=last).contains(&(bin as usize)))
                        })
                }
            };
            if !valid {
                return Err(Error::InvalidInput(
                    "invalid MA prepared property column".into(),
                ));
            }
        }
        for column in &self.residuals {
            // HybridUint(4, 1, 2) represents packed u32 with tokens 0..=239.
            if column.len() != rows
                || column.iter().any(|&[token, nbits]| {
                    token > 239 || nbits != if token < 16 { 0 } else { 1 + (token - 16) / 8 }
                })
            {
                return Err(Error::InvalidInput(
                    "invalid MA prepared residual column".into(),
                ));
            }
        }
        let mut properties = Vec::new();
        properties
            .try_reserve_exact(self.properties.len())
            .map_err(|_| Error::Limit("MA prepared property allocation"))?;
        for column in self.properties {
            properties.push(match column {
                GatherColumn::Packed(values) => PropertyColumn::Packed(values),
                GatherColumn::Static(values) => PropertyColumn::Wide(values),
            });
        }
        let mut residuals = Vec::new();
        residuals
            .try_reserve_exact(self.residuals.len())
            .map_err(|_| Error::Limit("MA prepared residual allocation"))?;
        for column in self.residuals {
            let mut converted = Vec::new();
            converted
                .try_reserve_exact(rows)
                .map_err(|_| Error::Limit("MA prepared residual allocation"))?;
            converted.extend(
                column
                    .into_iter()
                    .map(|[token, nbits]| ResidualToken { token, nbits }),
            );
            residuals.push(converted);
        }
        Ok(TreeSamples {
            properties,
            residuals,
            counts: self.counts,
            thresholds: self.quantization.thresholds,
        })
    }
}

pub(super) fn train(input: PreparedInput, options: &TrainOptions) -> Result<Tree> {
    let _scope = crate::profile::scope("ma.train_prepared");
    let mut samples = input.into_tree_samples(options)?;
    train_with_constants(options, |learning| train_nodes(&mut samples, learning))
}

#[cfg(test)]
mod tests {
    use super::super::{ConstantChannel, HybridUintConfig, MultiplierInfo, Sample, pack_signed};
    use super::*;

    fn options() -> TrainOptions {
        let properties = vec![6, 1, 0, 2];
        TrainOptions {
            predictors: vec![Predictor::Weighted, Predictor::Gradient, Predictor::Zero],
            property_quantization: Some(PropertyQuantization {
                properties: properties.clone(),
                thresholds: vec![vec![-3, 0, 7], vec![10, 20], vec![0, 1], vec![-1, 3]],
            }),
            properties,
            static_property_range: Some([[0, 3], [10, 22]]),
            split_threshold: 0.0,
            parallel: false,
            ..TrainOptions::default()
        }
    }

    fn raw_samples() -> Vec<Sample> {
        (0..32)
            .map(|i| {
                let mut properties = vec![0; 7];
                properties[0] = i % 2;
                properties[1] = 10 + i % 12;
                properties[2] = i % 7 - 2;
                properties[6] = [-3, -2, 0, 1, 7, 8][i as usize % 6];
                Sample {
                    properties,
                    residuals: if i % 2 == 0 {
                        vec![0, 100 + i, -17]
                    } else {
                        vec![100 + i, 0, 19]
                    },
                    weight: 1 + i as u32 % 17,
                }
            })
            .collect()
    }

    // No deduplication: source collision records are a valid handoff population.
    fn columns(raw: &[Sample], options: &TrainOptions) -> PreparedInput {
        let quantization = options.property_quantization.clone().unwrap();
        let config = HybridUintConfig::new(4, 1, 2).unwrap();
        PreparedInput {
            predictors: options.predictors.clone(),
            properties: quantization
                .properties
                .iter()
                .zip(&quantization.thresholds)
                .map(|(&id, thresholds)| {
                    let bins = raw.iter().map(|s| {
                        let value = if id < 2 {
                            s.properties[id]
                        } else {
                            s.properties[id].clamp(-511, 511)
                        };
                        thresholds.partition_point(|&t| value > t) as u32
                    });
                    if id < 2 {
                        GatherColumn::Static(bins.collect())
                    } else {
                        GatherColumn::Packed(bins.map(|bin| bin as u8).collect())
                    }
                })
                .collect(),
            quantization,
            residuals: (0..options.predictors.len())
                .map(|p| {
                    raw.iter()
                        .map(|s| {
                            let value = config.encode(pack_signed(s.residuals[p]));
                            [value.token as u8, value.nbits as u8]
                        })
                        .collect()
                })
                .collect(),
            counts: raw
                .iter()
                .map(|s| u16::try_from(s.weight).unwrap())
                .collect(),
            total: raw.iter().map(|s| u64::from(s.weight)).sum(),
        }
    }

    fn assert_same_tree(actual: &Tree, expected: &Tree) {
        let tokens = |tree: &Tree| {
            tree.tokens()
                .iter()
                .map(|token| (token.context, token.value))
                .collect::<Vec<_>>()
        };
        assert_eq!(tokens(actual), tokens(expected));
    }

    #[test]
    fn prepared_columns_move_buffers_preserve_order_and_collision_records() {
        let options = options();
        let mut raw = raw_samples();
        raw[0].weight = u32::from(u16::MAX);
        raw.push(Sample {
            weight: 1,
            ..raw[0].clone()
        });
        let input = columns(&raw, &options);
        let counts = input.counts.as_ptr();
        let properties: Vec<_> = input
            .properties
            .iter()
            .map(|column| match column {
                GatherColumn::Packed(values) => values.as_ptr().cast::<()>(),
                GatherColumn::Static(values) => values.as_ptr().cast::<()>(),
            })
            .collect();
        let thresholds: Vec<_> = input
            .quantization
            .thresholds
            .iter()
            .map(Vec::as_ptr)
            .collect();
        let samples = input.into_tree_samples(&options).unwrap();
        assert_eq!(samples.counts.as_ptr(), counts);
        assert_eq!(samples.counts.len(), raw.len());
        assert_eq!(samples.counts[0], u16::MAX);
        assert_eq!(samples.counts.last(), Some(&1));
        assert_eq!(
            samples
                .properties
                .iter()
                .map(PropertyColumn::as_ptr)
                .collect::<Vec<_>>(),
            properties
        );
        assert_eq!(
            samples
                .thresholds
                .iter()
                .map(Vec::as_ptr)
                .collect::<Vec<_>>(),
            thresholds
        );
        let reference = TreeSamples::new(&raw, &options).unwrap();
        let actual_parents =
            super::super::parent_histograms(&samples, &(0..raw.len()).collect::<Vec<_>>());
        let expected_parents = super::super::parent_histograms(
            &reference,
            &(0..reference.counts.len()).collect::<Vec<_>>(),
        );
        for (actual, expected) in actual_parents.iter().zip(&expected_parents) {
            assert_eq!(actual.counts, expected.counts);
            assert_eq!(actual.extra_bits, expected.extra_bits);
        }
    }

    #[test]
    fn prepared_train_preserves_constant_suffix_empty_seed_and_leaf_errors() {
        let mut options = options();
        let raw = raw_samples();
        for constants in [false, true] {
            options.constant_channels = if constants {
                vec![ConstantChannel {
                    channel: 2,
                    value: i32::MIN,
                }]
            } else {
                Vec::new()
            };
            for population in [raw.as_slice(), &[]] {
                let actual = train(columns(population, &options), &options).unwrap();
                let expected = super::super::train(population, &options).unwrap();
                assert_same_tree(&actual, &expected);
                if constants {
                    assert_eq!(actual.lookup(&[2, 10]).unwrap().2, i32::MIN);
                }
            }
            options.max_leaves = 1 + usize::from(constants);
            options.require_complete_search = true;
            assert!(matches!(
                train(columns(&raw, &options), &options),
                Err(Error::Limit(_))
            ));
            assert!(matches!(
                super::super::train(&raw, &options),
                Err(Error::Limit(_))
            ));
            options.require_complete_search = false;
            assert_same_tree(
                &train(columns(&raw, &options), &options).unwrap(),
                &super::super::train(&raw, &options).unwrap(),
            );
            options.max_leaves = 64;
        }
        options.constant_channels = (0..3)
            .map(|channel| ConstantChannel {
                channel,
                value: [i32::MIN, 0, i32::MAX][channel as usize],
            })
            .collect();
        let actual = train(columns(&[], &options), &options).unwrap();
        assert_same_tree(&actual, &super::super::train(&[], &options).unwrap());
        assert!(train(columns(&raw, &options), &options).is_err());
    }

    #[test]
    fn prepared_validation_rejects_schema_counts_and_shapes_without_raw_proofs() {
        let options = options();
        let raw = raw_samples();
        for case in 0..16 {
            let mut input = columns(&raw, &options);
            match case {
                0 => input.predictors.swap(0, 1),
                1 => input.quantization.properties.swap(0, 1),
                2 => input.quantization.thresholds[0][0] -= 1,
                3 => {
                    input.properties.pop();
                }
                4 => {
                    input.residuals.pop();
                }
                5 => input.counts[0] = 0,
                6 => input.total += 1,
                7 => input.residuals[0][0] = [240, 29],
                8 => input.residuals[0][0] = [0, 1],
                9 => {
                    input.residuals[0].pop();
                }
                10 => input.properties[0] = GatherColumn::Static(vec![0; raw.len()]),
                11 => input.properties[1] = GatherColumn::Packed(vec![0; raw.len()]),
                12 => input.properties[0] = GatherColumn::Packed(vec![4; raw.len()]),
                13 => input.properties[0] = GatherColumn::Packed(vec![0; raw.len() - 1]),
                14 => input.properties[1] = GatherColumn::Static(vec![3; raw.len()]),
                15 => input.properties[1] = GatherColumn::Static(vec![0; raw.len() + 1]),
                _ => unreachable!(),
            }
            assert!(
                matches!(
                    input.into_tree_samples(&options),
                    Err(Error::InvalidInput(_))
                ),
                "case {case}"
            );
        }
        let mut changed = options.clone();
        changed.property_quantization = None;
        assert!(columns(&raw, &options).into_tree_samples(&changed).is_err());
        // Even multiplier 1 requires the raw-domain API, never a reconstructed token.
        for multiplier in [1, 2, 3, i32::MAX as u32] {
            changed = options.clone();
            changed.multiplier_info.push(MultiplierInfo {
                range: options.static_property_range.unwrap(),
                multiplier,
            });
            assert!(columns(&raw, &options).into_tree_samples(&changed).is_err());
        }
    }

    #[test]
    fn prepared_static_bins_check_intersection_without_claiming_exact_ids() {
        let mut options = options();
        options.static_property_range = Some([[1, 3], [11, 21]]);
        options.constant_channels = vec![ConstantChannel {
            channel: 2,
            value: 0,
        }];
        let mut raw = raw_samples();
        for sample in &mut raw {
            sample.properties[0] = 1;
            sample.properties[1] = 11;
        }
        assert!(columns(&raw, &options).into_tree_samples(&options).is_ok());
        for (axis, id) in [(0, 0), (0, 2), (1, 10), (1, 21)] {
            raw[0].properties[axis] = id;
            assert!(columns(&raw, &options).into_tree_samples(&options).is_err());
            raw[0].properties[0] = 1;
            raw[0].properties[1] = 11;
        }
        // A shared bin spanning a valid group and an excluded group is not proof
        // of either raw ID. The caller guarantees that gather visited the domain.
        options.static_property_range.as_mut().unwrap()[1] = [12, 13];
        assert!(columns(&raw, &options).into_tree_samples(&options).is_ok());
        // Absent static split properties carry no per-row raw-ID evidence either.
        options.properties = vec![6];
        options.property_quantization = Some(PropertyQuantization {
            properties: vec![6],
            thresholds: vec![vec![0]],
        });
        assert!(columns(&raw, &options).into_tree_samples(&options).is_ok());
    }

    #[test]
    fn prepared_residual_shapes_cover_full_u32_and_weight_limit() {
        let mut options = options();
        options.properties = vec![6];
        options.predictors = vec![Predictor::Zero];
        options.property_quantization = Some(PropertyQuantization {
            properties: vec![6],
            thresholds: vec![vec![0]],
        });
        let config = HybridUintConfig::new(4, 1, 2).unwrap();
        let mut raw = Vec::new();
        for exponent in 0..32 {
            for packed in [
                (1u32 << exponent).wrapping_sub(1),
                1u32 << exponent,
                u32::MAX >> exponent,
            ] {
                let mut sample = raw_samples().remove(0);
                sample.residuals = vec![((packed >> 1) as i32) ^ -((packed & 1) as i32)];
                raw.push(sample);
            }
        }
        let samples = columns(&raw, &options).into_tree_samples(&options).unwrap();
        for (sample, token) in raw.iter().zip(&samples.residuals[0]) {
            let expected = config.encode(pack_signed(sample.residuals[0]));
            assert_eq!(
                (u32::from(token.token), u32::from(token.nbits)),
                (expected.token, expected.nbits)
            );
        }
        let mut input = columns(&[], &options);
        let full = MAX_HISTOGRAM_TOTAL as usize / usize::from(u16::MAX);
        input.counts = vec![u16::MAX; full];
        input
            .counts
            .push((MAX_HISTOGRAM_TOTAL % u32::from(u16::MAX)) as u16);
        input.total = u64::from(MAX_HISTOGRAM_TOTAL);
        input.properties = vec![GatherColumn::Packed(vec![0; input.counts.len()])];
        input.residuals = vec![vec![[0, 0]; input.counts.len()]];
        let samples = input.into_tree_samples(&options).unwrap();
        assert_eq!(
            samples.counts.iter().map(|&n| u64::from(n)).sum::<u64>(),
            u64::from(MAX_HISTOGRAM_TOTAL)
        );
        let mut overweight = columns(&[], &options);
        overweight.counts = samples.counts;
        *overweight.counts.last_mut().unwrap() += 1;
        overweight.total = u64::from(MAX_HISTOGRAM_TOTAL) + 1;
        assert!(matches!(
            overweight.into_tree_samples(&options),
            Err(Error::Limit(_))
        ));
    }
}
