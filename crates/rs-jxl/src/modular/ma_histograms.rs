// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! Optional enc_ma.cc / enc_ma_histograms.h histogram layouts, frozen b866b3d.

#[cfg(test)]
use super::TreeSamples;

use super::{PropertyValue, ResidualToken, SampleStorage, Samples};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DenseHistogram {
    #[default]
    Disabled,
    /// JXL_MA_DENSE_HISTOGRAM=1: dense remapping through 256 records.
    Small,
    /// JXL_MA_DENSE_HISTOGRAM=2: dense remapping through 4096 records.
    Extended,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PropertyBins {
    #[default]
    Disabled,
    /// JXL_MA_PROPERTY_BINS=1: below 256 records or compacted tokens.
    SmallOrCompacted,
    /// JXL_MA_PROPERTY_BINS=2: any uncached node meeting the savings test.
    All,
}

pub(super) fn histogram_stride(symbols: usize, lanes: usize, padded: bool) -> usize {
    let stride = symbols.div_ceil(lanes) * lanes;
    if padded && (32..256).contains(&stride) && stride.is_power_of_two() {
        stride + lanes
    } else {
        stride
    }
}

pub(super) fn prepare_property_bins(
    used: &[usize],
    mode: PropertyBins,
    records: usize,
    compact: bool,
    cached: bool,
    mapping: &mut Vec<usize>,
) -> usize {
    if mode == PropertyBins::Disabled
        || cached
        || (mode != PropertyBins::All && records >= 256 && !compact)
    {
        return used.len();
    }
    let bins = used.iter().filter(|&&n| n != 0).count();
    if bins * 2 >= used.len() {
        return used.len();
    }
    mapping.clear();
    mapping.resize(used.len(), 0);
    let mut next = 0;
    for (i, &n) in used.iter().enumerate() {
        if n != 0 {
            mapping[i] = next;
            next += 1;
        }
    }
    bins
}

pub(super) fn active_blocks(counts: &[u32], lanes: usize, blocks: &mut Vec<usize>) {
    blocks.clear();
    blocks.extend(
        counts
            .chunks(lanes)
            .enumerate()
            .filter_map(|(i, block)| block.iter().any(|&n| n != 0).then_some(i * lanes)),
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_mapped<T: PropertyValue, const PACKED: bool, const COMPACT: bool>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    tokens: &[ResidualToken],
    stride: usize,
    property_mapping: &[usize],
    token_mapping: Option<&[u8; 256]>,
    counts: &mut [u32],
    extra: &mut [u64],
) {
    for &i in indices {
        let bin = values[i].index();
        let bin = if PACKED { property_mapping[bin] } else { bin };
        let residual = tokens[i];
        let symbol = if COMPACT {
            token_mapping.expect("compact mapping")[usize::from(residual.token)]
        } else {
            residual.token
        };
        let weight = u32::from(samples.counts[i]);
        counts[bin * stride + usize::from(symbol)] += weight;
        extra[bin] += u64::from(weight) * u64::from(residual.nbits);
    }
}

#[derive(Default)]
pub(super) struct HistogramShards {
    counts: Vec<u32>,
    extra: Vec<u64>,
}

fn zeroed<T: Default + Clone>(storage: &mut Vec<T>, size: usize) -> Result<()> {
    storage
        .try_reserve(size.saturating_sub(storage.len()))
        .map_err(|_| Error::Limit("MA histogram shard allocation"))?;
    storage.clear();
    storage.resize(size, T::default());
    Ok(())
}

impl HistogramShards {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn collect<T: PropertyValue>(
        &mut self,
        samples: &Samples<impl SampleStorage>,
        indices: &[usize],
        values: &[T],
        tokens: &[ResidualToken],
        stride: usize,
        bins: usize,
        counts: &mut [u32],
        extra: &mut [u64],
    ) -> Result<()> {
        let cells = bins
            .checked_mul(stride)
            .ok_or(Error::Limit("MA histogram shard size"))?;
        let scratch_cells = cells
            .checked_mul(3)
            .ok_or(Error::Limit("MA histogram shard size"))?;
        let scratch_bins = bins
            .checked_mul(3)
            .ok_or(Error::Limit("MA histogram shard size"))?;
        zeroed(&mut self.counts, scratch_cells)?;
        zeroed(&mut self.extra, scratch_bins)?;
        let mut add = |i: usize, shard: usize| {
            let residual = tokens[i];
            let bin = values[i].index();
            let weight = u32::from(samples.counts[i]);
            let cell = bin * stride + usize::from(residual.token);
            let bits = u64::from(weight) * u64::from(residual.nbits);
            if shard == 0 {
                counts[cell] += weight;
                extra[bin] += bits;
            } else {
                self.counts[(shard - 1) * cells + cell] += weight;
                self.extra[(shard - 1) * bins + bin] += bits;
            }
        };
        let mut chunks = indices.chunks_exact(4);
        for group in &mut chunks {
            for (shard, &i) in group.iter().enumerate() {
                add(i, shard);
            }
        }
        for &i in chunks.remainder() {
            add(i, 0);
        }
        for (cell, count) in counts.iter_mut().enumerate() {
            *count += self.counts[cell] + self.counts[cells + cell] + self.counts[2 * cells + cell];
        }
        for (bin, bits) in extra.iter_mut().enumerate() {
            *bits += self.extra[bin] + self.extra[bins + bin] + self.extra[2 * bins + bin];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_options_forward_only_explicit_tools() {
        use super::super::{HistogramOptions, TrainOptions};
        let mut options = TrainOptions {
            histogram_cache_bytes: 17,
            require_complete_search: true,
            predictor_dedup: true,
            property_dedup: true,
            ..TrainOptions::default()
        };
        let tools = HistogramOptions {
            shards: true,
            padded: true,
            dense: DenseHistogram::Extended,
            property_bins: PropertyBins::All,
            sparse: true,
            paired: true,
        };
        tools.apply(&mut options);
        assert!(options.histogram_shards && options.padded_histogram);
        assert!(options.sparse_entropy && options.paired_cache);
        assert_eq!(options.dense_histogram, DenseHistogram::Extended);
        assert_eq!(options.property_bins, PropertyBins::All);
        HistogramOptions::default().apply(&mut options);
        assert!(!options.histogram_shards && !options.padded_histogram);
        assert!(!options.sparse_entropy && !options.paired_cache);
        assert_eq!(options.dense_histogram, DenseHistogram::Disabled);
        assert_eq!(options.property_bins, PropertyBins::Disabled);
        assert_eq!(options.histogram_cache_bytes, 17);
        assert!(
            options.require_complete_search && options.predictor_dedup && options.property_dedup
        );
    }

    #[test]
    fn optional_layout_source_stride_and_property_bin_gates() {
        for lanes in [1usize, 4, 8] {
            for symbols in [1usize, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256] {
                let base = symbols.div_ceil(lanes) * lanes;
                assert_eq!(histogram_stride(symbols, lanes, false), base);
                let extra = usize::from([32, 64, 128].contains(&base)) * lanes;
                assert_eq!(histogram_stride(symbols, lanes, true), base + extra);
            }
        }
        let mut used = vec![0; 256];
        used[3] = 1;
        used[255] = 65535;
        let mut mapping = vec![99; 300];
        for mode in [
            PropertyBins::Disabled,
            PropertyBins::SmallOrCompacted,
            PropertyBins::All,
        ] {
            for records in [255, 256, 4096] {
                for compact in [false, true] {
                    for cached in [false, true] {
                        let enabled = mode != PropertyBins::Disabled
                            && !cached
                            && (mode == PropertyBins::All || records < 256 || compact);
                        assert_eq!(
                            prepare_property_bins(
                                &used,
                                mode,
                                records,
                                compact,
                                cached,
                                &mut mapping
                            ),
                            if enabled { 2 } else { 256 }
                        );
                        if enabled {
                            assert_eq!((mapping[3], mapping[255]), (0, 1));
                        }
                    }
                }
            }
        }
        assert_eq!(
            prepare_property_bins(
                &[1, 0, 1, 0],
                PropertyBins::All,
                64,
                false,
                false,
                &mut mapping
            ),
            4
        );
    }

    #[test]
    fn optional_shards_preserve_weighted_counts_tails_and_reused_storage() {
        let mut scratch = HistogramShards::default();
        for records in [0, 1, 3, 4, 5, 255, 256, 257, 4097] {
            let values: Vec<_> = (0..records).map(|i| (i % 3) as u8).collect();
            let tokens: Vec<_> = (0..records)
                .map(|i| ResidualToken {
                    token: [0, 31, 255][i % 3],
                    nbits: (i % 32) as u8,
                })
                .collect();
            let samples = TreeSamples {
                properties: Vec::new(),
                thresholds: Vec::new(),
                residuals: Vec::new(),
                counts: (0..records)
                    .map(|i| if i % 7 == 0 { u16::MAX } else { 1 })
                    .collect(),
            };
            let indices: Vec<_> = (0..records).rev().collect();
            let (mut expected, mut expected_extra) = (vec![0; 3 * 256], vec![0; 3]);
            collect_mapped::<_, false, false>(
                &samples,
                &indices,
                &values,
                &tokens,
                256,
                &[],
                None,
                &mut expected,
                &mut expected_extra,
            );
            let wide: Vec<_> = values.iter().map(|&v| u32::from(v)).collect();
            for packed in [false, true] {
                let (mut counts, mut extra) = (vec![0; 3 * 256], vec![0; 3]);
                if packed {
                    scratch.collect(
                        &samples,
                        &indices,
                        &values,
                        &tokens,
                        256,
                        3,
                        &mut counts,
                        &mut extra,
                    )
                } else {
                    scratch.collect(
                        &samples,
                        &indices,
                        &wide,
                        &tokens,
                        256,
                        3,
                        &mut counts,
                        &mut extra,
                    )
                }
                .unwrap();
                assert_eq!(counts, expected);
                assert_eq!(extra, expected_extra);
            }
            assert!(
                scratch
                    .collect(
                        &samples,
                        &[],
                        &values,
                        &tokens,
                        2,
                        usize::MAX,
                        &mut [],
                        &mut []
                    )
                    .is_err()
            );
        }
    }
}
