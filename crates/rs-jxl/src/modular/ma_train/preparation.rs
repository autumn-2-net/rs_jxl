// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Legacy row validation, quantization and source sample deduplication.
//! Prepared-column handoff stays in the existing `prepared_storage` module.

use super::{
    HybridUintConfig, MAX_HISTOGRAM_TOTAL, PROPERTY_RANGE, PropertyColumn, QUANTILE_RANGE, Sample,
    TrainOptions, TreeSamples, WP_PROPERTY, constraint_thresholds, pack_signed, range_contains,
    sample_storage,
};
use crate::{Error, Result};

pub(super) fn validate_samples<'a>(
    samples: impl IntoIterator<Item = &'a Sample>,
    options: &TrainOptions,
) -> Result<()> {
    let mut total = 0u32;
    for sample in samples {
        if sample.residuals.len() != options.predictors.len()
            || options
                .properties
                .iter()
                .any(|&p| p >= sample.properties.len())
        {
            return Err(Error::InvalidInput(
                "MA sample does not match predictors/properties".into(),
            ));
        }
        if let Some(range) = options.static_property_range
            && sample.weight != 0
            && (sample.properties.len() < 2
                || sample.properties[0] < 0
                || sample.properties[1] < 0
                || !range_contains(
                    range,
                    [sample.properties[0] as u32, sample.properties[1] as u32],
                ))
        {
            return Err(Error::InvalidInput(
                "MA sample is outside the static property range".into(),
            ));
        }
        total = total
            .checked_add(sample.weight)
            .filter(|&total| total <= MAX_HISTOGRAM_TOTAL)
            .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct ResidualToken {
    pub(super) token: u8,
    pub(super) nbits: u8,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct SampleKey {
    pub(super) properties: Vec<u32>,
    pub(super) residuals: Vec<ResidualToken>,
}

#[cfg(test)]
pub(super) fn quantized_key(
    sample: &Sample,
    properties: &[usize],
    thresholds: &[Vec<i32>],
    exact_static: bool,
    config: &HybridUintConfig,
) -> SampleKey {
    let mut key = SampleKey {
        properties: Vec::new(),
        residuals: Vec::new(),
    };
    quantized_key_into(
        &mut key,
        sample,
        properties,
        thresholds,
        exact_static,
        config,
    );
    key
}

fn quantized_key_into(
    key: &mut SampleKey,
    sample: &Sample,
    properties: &[usize],
    thresholds: &[Vec<i32>],
    exact_static: bool,
    config: &HybridUintConfig,
) {
    key.properties.clear();
    key.properties
        .extend(properties.iter().enumerate().map(|(p, &id)| {
            let value = if id < 2 && exact_static {
                sample.properties[id]
            } else {
                sample.properties[id].clamp(-PROPERTY_RANGE, PROPERTY_RANGE)
            };
            thresholds[p].partition_point(|&threshold| value > threshold) as u32
        }));
    key.residuals.clear();
    key.residuals
        .extend(sample.residuals.iter().map(|&residual| {
            let encoded = config.encode(pack_signed(residual));
            ResidualToken {
                token: encoded.token as u8,
                nbits: encoded.nbits as u8,
            }
        }));
}

/// TreeSamples::FlushSampleBatch/HashBatch/AddWeightedToTableAndMerge. The
/// cache is two-choice and bounded: a collision retains the sample, never loses
/// it. Quantization is shared across all waves/streams. Source 16-bit saturation
/// is retained here and by the prepared TreeSamples count columns.
/// This collector is for unconstrained gather only: it must not hide raw
/// residual divisibility checks required by caller-supplied multipliers.
#[cfg(test)]
pub(super) struct SampleBatchCollector {
    properties: Vec<usize>,
    thresholds: Vec<Vec<i32>>,
    predictors: usize,
    config: HybridUintConfig,
    pending: Vec<Sample>,
    pub(super) samples: Vec<Sample>,
    keys: Vec<SampleKey>,
    pub(super) table: Vec<u32>,
    total: u32,
}

#[cfg(test)]
impl SampleBatchCollector {
    const BATCH: usize = 256;
    pub(super) const MAX_SLOTS: usize = 1 << 20;
    pub(super) const UNUSED: u32 = u32::MAX;
    const MAX_WEIGHT: u32 = u16::MAX as u32;

    pub(super) fn new(options: &TrainOptions) -> Result<Self> {
        options.validate()?;
        if !options.multiplier_info.is_empty() || !options.constant_channels.is_empty() {
            return Err(Error::InvalidInput(
                "MA batch gather requires unconstrained variable samples".into(),
            ));
        }
        let quantization = options.property_quantization.as_ref().ok_or_else(|| {
            Error::InvalidInput("MA batch gather requires shared population quantiles".into())
        })?;
        Ok(Self {
            properties: options.properties.clone(),
            thresholds: quantization.thresholds.clone(),
            predictors: options.predictors.len(),
            config: HybridUintConfig::new(4, 1, 2).map_err(|e| Error::Encoding(e.to_string()))?,
            pending: Vec::with_capacity(Self::BATCH),
            samples: Vec::new(),
            keys: Vec::new(),
            table: Vec::new(),
            total: 0,
        })
    }

    pub(super) fn push(&mut self, sample: Sample) -> Result<()> {
        if sample.residuals.len() != self.predictors
            || self
                .properties
                .iter()
                .any(|&p| p >= sample.properties.len())
        {
            return Err(Error::InvalidInput(
                "invalid MA batch sample dimensions".into(),
            ));
        }
        self.total = self
            .total
            .checked_add(sample.weight)
            .ok_or(Error::Limit("MA sample weight overflow"))?;
        if sample.weight == 0 {
            return Ok(());
        }
        self.pending.push(sample);
        if self.pending.len() == Self::BATCH {
            self.flush()?;
        }
        Ok(())
    }

    pub(super) fn positions(key: &SampleKey, mask: usize) -> [usize; 2] {
        const A: u64 = 0x1e35_a7bd;
        const B: u64 = 0x1e35_a7bd_1e35_a7bd;
        let (mut first, mut second) = (A, B);
        for r in &key.residuals {
            first = first
                .wrapping_mul(A)
                .wrapping_add(u64::from(r.token))
                .wrapping_mul(A)
                .wrapping_add(u64::from(r.nbits));
        }
        for &property in &key.properties {
            first = first.wrapping_mul(A).wrapping_add(u64::from(property));
            second = second.wrapping_mul(B) ^ u64::from(property);
        }
        for r in &key.residuals {
            second =
                (second.wrapping_mul(B) ^ u64::from(r.token)).wrapping_mul(B) ^ u64::from(r.nbits);
        }
        [
            (first >> 16) as usize & mask,
            (second >> 16) as usize & mask,
        ]
    }

    pub(super) fn hash_batch(keys: &[SampleKey]) -> Vec<(u64, u64)> {
        const A: u64 = 0x1e35_a7bd;
        const B: u64 = 0x1e35_a7bd_1e35_a7bd;
        let mut hashes = vec![(A, B); keys.len()];
        let Some(first) = keys.first() else {
            return hashes;
        };
        for predictor in 0..first.residuals.len() {
            for (key, hash) in keys.iter().zip(&mut hashes) {
                let r = key.residuals[predictor];
                hash.0 = hash
                    .0
                    .wrapping_mul(A)
                    .wrapping_add(u64::from(r.token))
                    .wrapping_mul(A)
                    .wrapping_add(u64::from(r.nbits));
            }
        }
        for property in 0..first.properties.len() {
            for (key, hash) in keys.iter().zip(&mut hashes) {
                let value = u64::from(key.properties[property]);
                hash.0 = hash.0.wrapping_mul(A).wrapping_add(value);
                hash.1 = hash.1.wrapping_mul(B) ^ value;
            }
        }
        for predictor in 0..first.residuals.len() {
            for (key, hash) in keys.iter().zip(&mut hashes) {
                let r = key.residuals[predictor];
                hash.1 = (hash.1.wrapping_mul(B) ^ u64::from(r.token)).wrapping_mul(B)
                    ^ u64::from(r.nbits);
            }
        }
        hashes
    }

    fn grow_table(&mut self, additional: usize) {
        let target = self
            .samples
            .len()
            .saturating_add(additional)
            .saturating_mul(3)
            .div_ceil(2)
            .clamp(512, Self::MAX_SLOTS)
            .next_power_of_two();
        if target <= self.table.len() {
            return;
        }
        // Rebuild from empty slots so saturation cannot leave a stale alias.
        self.table = vec![Self::UNUSED; target];
        for index in 0..self.samples.len() {
            if self.samples[index].weight == Self::MAX_WEIGHT {
                continue;
            }
            for position in Self::positions(&self.keys[index], target - 1) {
                if self.table[position] == Self::UNUSED {
                    self.table[position] = index as u32;
                    break;
                }
            }
        }
    }

    pub(super) fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.grow_table(self.pending.len());
        let keys: Vec<_> = self
            .pending
            .iter()
            .map(|sample| {
                quantized_key(
                    sample,
                    &self.properties,
                    &self.thresholds,
                    true,
                    &self.config,
                )
            })
            .collect();
        let hashes = Self::hash_batch(&keys);
        let pending = std::mem::replace(&mut self.pending, Vec::with_capacity(Self::BATCH));
        let mask = self.table.len() - 1;
        for ((mut sample, key), (first, second)) in pending.into_iter().zip(keys).zip(hashes) {
            let positions = [
                (first >> 16) as usize & mask,
                (second >> 16) as usize & mask,
            ];
            for &position in &positions {
                let previous = self.table[position];
                if previous == Self::UNUSED || self.keys[previous as usize] != key {
                    continue;
                }
                let previous = previous as usize;
                let take = sample
                    .weight
                    .min(Self::MAX_WEIGHT - self.samples[previous].weight);
                self.samples[previous].weight += take;
                sample.weight -= take;
                if self.samples[previous].weight == Self::MAX_WEIGHT {
                    for &slot in &positions {
                        if self.table[slot] == previous as u32 {
                            self.table[slot] = Self::UNUSED;
                        }
                    }
                }
                if sample.weight == 0 {
                    break;
                }
            }
            while sample.weight != 0 {
                let weight = sample.weight.min(Self::MAX_WEIGHT);
                sample.weight -= weight;
                let index = self.samples.len();
                self.samples
                    .try_reserve(1)
                    .map_err(|_| Error::Limit("MA batch samples allocation"))?;
                self.keys
                    .try_reserve(1)
                    .map_err(|_| Error::Limit("MA batch keys allocation"))?;
                if weight != Self::MAX_WEIGHT {
                    for &position in &positions {
                        if self.table[position] == Self::UNUSED {
                            self.table[position] = index as u32;
                            break;
                        }
                    }
                }
                if sample.weight == 0 {
                    sample.weight = weight;
                    self.samples.push(sample);
                    self.keys.push(key);
                    break;
                }
                self.samples.push(Sample {
                    weight,
                    ..sample.clone()
                });
                self.keys.push(key.clone());
            }
        }
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Vec<Sample>> {
        self.flush()?;
        debug_assert_eq!(
            self.samples
                .iter()
                .map(|s| u64::from(s.weight))
                .sum::<u64>(),
            u64::from(self.total)
        );
        Ok(self.samples)
    }
}

impl TreeSamples {
    pub(super) fn new<'a, I>(samples: I, options: &TrainOptions) -> Result<Self>
    where
        I: IntoIterator<Item = &'a Sample>,
        I::IntoIter: Clone,
    {
        let _scope = crate::profile::scope("ma.prepare_samples");
        let samples = samples.into_iter();
        validate_samples(samples.clone(), options)?;
        let thresholds: Vec<_> = options
            .properties
            .iter()
            .enumerate()
            .map(|(index, &property)| {
                constraint_thresholds(options, property)
                    .or_else(|| {
                        options
                            .property_quantization
                            .as_ref()
                            .map(|q| q.thresholds[index].clone())
                    })
                    .unwrap_or_else(|| {
                        property_thresholds(samples.clone(), property, options.max_property_values)
                    })
            })
            .collect();
        let mut result = Self {
            properties: options
                .properties
                .iter()
                .zip(&thresholds)
                .map(|(&property, thresholds)| PropertyColumn::new(property, thresholds.len()))
                .collect(),
            residuals: vec![Vec::new(); options.predictors.len()],
            counts: Vec::new(),
            thresholds,
        };
        let config = HybridUintConfig::new(4, 1, 2).map_err(|e| Error::Encoding(e.to_string()))?;
        let capacity = sample_storage::split_record_capacity(samples.clone())?;
        let mut dedup = sample_storage::DedupTable::new(&options.properties);
        dedup.prepare(&mut result, capacity)?;
        let mut key = SampleKey {
            properties: Vec::with_capacity(options.properties.len()),
            residuals: Vec::with_capacity(options.predictors.len()),
        };
        let mut input_records = 0usize;
        for sample in samples {
            input_records += 1;
            if sample.weight == 0 {
                continue;
            }
            quantized_key_into(
                &mut key,
                sample,
                &options.properties,
                &result.thresholds,
                !options.multiplier_info.is_empty() || options.property_quantization.is_some(),
                &config,
            );
            dedup.insert(&mut result, &key, sample.weight)?;
        }
        crate::profile::note(
            "ma.sample_counts",
            format_args!(
                "input_records={} distinct={} predictors={} properties={}",
                input_records,
                result.counts.len(),
                result.residuals.len(),
                result.properties.len(),
            ),
        );
        Ok(result)
    }
}

/// QuantizeHistogram, including skipping repeated quantiles and removing the
/// terminal threshold. u64 arithmetic retains exact sample multiplicities.
pub(super) fn quantize_histogram(histogram: &[u64], num_chunks: usize, bias: i32) -> Vec<i32> {
    let sum: u64 = histogram.iter().sum();
    if sum == 0 {
        return Vec::new();
    }
    let mut thresholds = Vec::new();
    let mut cumsum = 0u64;
    let mut threshold = 1u64;
    for (i, &count) in histogram.iter().enumerate() {
        cumsum += count;
        if cumsum * num_chunks as u64 >= threshold * sum {
            thresholds.push(i as i32 + bias);
            while cumsum * num_chunks as u64 >= threshold * sum {
                threshold += 1;
            }
        }
    }
    thresholds.pop();
    thresholds
}

pub(super) fn property_thresholds<'a>(
    samples: impl IntoIterator<Item = &'a Sample>,
    property: usize,
    max_values: usize,
) -> Vec<i32> {
    if property == 2 || property == 3 {
        return (1..max_values)
            .map(|i| (i * 256 / max_values) as i32 - 1)
            .collect();
    }
    if property == WP_PROPERTY {
        return if max_values < 32 {
            vec![-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127]
        } else if max_values < 64 {
            vec![
                -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -5, -3, -1, 0, 1, 3, 5, 7,
                11, 15, 23, 31, 47, 63, 95, 127, 191, 255,
            ]
        } else {
            vec![
                -255, -223, -191, -159, -127, -111, -95, -79, -63, -55, -47, -39, -31, -27, -23,
                -19, -15, -13, -11, -9, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11,
                13, 15, 19, 23, 27, 31, 39, 47, 55, 63, 79, 95, 111, 127, 159, 191, 223, 255,
            ]
        };
    }
    let range = if property < 2 {
        PROPERTY_RANGE
    } else {
        QUANTILE_RANGE
    };
    let mut histogram = vec![0u64; (2 * range + 1) as usize];
    for sample in samples {
        let value = sample.properties[property].clamp(-range, range);
        histogram[(value + range) as usize] += u64::from(sample.weight);
    }
    quantize_histogram(&histogram, max_values, -range)
}
