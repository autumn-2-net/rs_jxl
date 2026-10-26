// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! Optional enc_ma.cc / enc_ma_histograms.h histogram layouts, frozen b866b3d.

#[cfg(test)]
use super::TreeSamples;

use super::{ParentHistogram, PropertyValue, ResidualToken, SampleStorage, Samples};
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

/// Immutable node-local zero-block proofs in the scoring parents' token layout.
/// Build after compaction and share across properties/passes, never descendants.
#[derive(Default)]
pub(super) struct SparseHistogramPlan {
    blocks: Vec<usize>,
    ranges: Vec<Option<std::ops::Range<usize>>>,
}

impl SparseHistogramPlan {
    pub(super) fn new(
        parents: &[ParentHistogram],
        predictors: &[usize],
        lanes: usize,
        enabled: bool,
    ) -> Self {
        if !enabled {
            return Self::default();
        }
        let mut plan = Self {
            blocks: Vec::new(),
            ranges: vec![None; parents.len()],
        };
        for &predictor in predictors {
            let counts = &parents[predictor].counts;
            let dense_blocks = counts.len().div_ceil(lanes);
            let begin = plan.blocks.len();
            plan.blocks.extend(
                counts
                    .chunks(lanes)
                    .enumerate()
                    .filter_map(|(i, block)| block.iter().any(|&n| n != 0).then_some(i * lanes)),
            );
            let active = plan.blocks.len() - begin;
            // Common/padded storage strides are not scoring work. Require a
            // real 25% block reduction to amortize sparse validation/indexing.
            if active < dense_blocks && active * 4 <= dense_blocks * 3 {
                plan.ranges[predictor] = Some(begin..plan.blocks.len());
            } else {
                plan.blocks.truncate(begin);
            }
        }
        plan
    }

    pub(super) fn blocks(&self, predictor: usize) -> Option<&[usize]> {
        self.ranges
            .get(predictor)?
            .as_ref()
            .map(|range| &self.blocks[range.clone()])
    }
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

/// Two compacted predictors share the property/weight scan, retaining their
/// independent trimmed strides and exact extra-bit accumulators.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_compact_pair<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    tokens: [&[ResidualToken]; 2],
    strides: [usize; 2],
    mapping: &[u8; 256],
    counts: [&mut [u32]; 2],
    extra: [&mut [u64]; 2],
) {
    let [counts_a, counts_b] = counts;
    let [extra_a, extra_b] = extra;
    let [tokens_a, tokens_b] = tokens;
    let [stride_a, stride_b] = strides;
    for &i in indices {
        let bin = values[i].index();
        let weight = u32::from(samples.counts[i]);
        let a = tokens_a[i];
        let b = tokens_b[i];
        counts_a[bin * stride_a + usize::from(mapping[usize::from(a.token)])] += weight;
        counts_b[bin * stride_b + usize::from(mapping[usize::from(b.token)])] += weight;
        extra_a[bin] += u64::from(weight) * u64::from(a.nbits);
        extra_b[bin] += u64::from(weight) * u64::from(b.nbits);
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
    fn compact_pair_matches_separate_weighted_collectors() {
        fn check<T: PropertyValue>(
            samples: &TreeSamples,
            indices: &[usize],
            values: &[T],
            tokens: [&[ResidualToken]; 2],
            mapping: &[u8; 256],
            strides: [usize; 2],
        ) {
            let [sa, sb] = strides;
            let mut a = vec![0; 5 * sa + 2];
            let mut b = vec![0; 5 * sb + 2];
            a[0] = u32::MAX;
            a[5 * sa + 1] = u32::MAX;
            b[0] = u32::MAX;
            b[5 * sb + 1] = u32::MAX;
            let mut ea = [0; 7];
            let mut eb = [0; 7];
            ea[0] = u64::MAX;
            ea[6] = u64::MAX;
            eb[0] = u64::MAX;
            eb[6] = u64::MAX;
            let (mut expected_a, mut expected_b) = (a.clone(), b.clone());
            let (mut expected_ea, mut expected_eb) = (ea, eb);
            // Also preserve additive (not implicitly cleared) collection.
            for _ in 0..2 {
                collect_compact_pair(
                    samples,
                    indices,
                    values,
                    tokens,
                    strides,
                    mapping,
                    [&mut a[1..5 * sa + 1], &mut b[1..5 * sb + 1]],
                    [&mut ea[1..6], &mut eb[1..6]],
                );
                for (token, stride, counts, extra) in [
                    (tokens[0], sa, &mut expected_a, &mut expected_ea),
                    (tokens[1], sb, &mut expected_b, &mut expected_eb),
                ] {
                    collect_mapped::<_, false, true>(
                        samples,
                        indices,
                        values,
                        token,
                        stride,
                        &[],
                        Some(mapping),
                        &mut counts[1..5 * stride + 1],
                        &mut extra[1..6],
                    );
                }
                assert_eq!(a, expected_a);
                assert_eq!(b, expected_b);
                assert_eq!(ea, expected_ea);
                assert_eq!(eb, expected_eb);
            }
        }
        let mut mapping = [0; 256];
        for (raw, compact) in [(0, 0), (7, 8), (255, 15), (1, 3), (31, 4)] {
            mapping[raw] = compact;
        }
        for records in [0, 1, 2, 17, 255, 256] {
            let samples = TreeSamples {
                properties: Vec::new(),
                thresholds: Vec::new(),
                residuals: Vec::new(),
                counts: (0..records)
                    .map(|i| if i % 7 == 0 { u16::MAX } else { 1 })
                    .collect(),
            };
            let values: Vec<_> = (0..records).map(|i| (i % 5) as u8).collect();
            let wide: Vec<_> = values.iter().map(|&v| u32::from(v)).collect();
            let a: Vec<_> = (0..records)
                .map(|i| ResidualToken {
                    token: [0, 7, 255][i % 3],
                    nbits: (i % 32) as u8,
                })
                .collect();
            let b: Vec<_> = (0..records)
                .map(|i| ResidualToken {
                    token: [1, 31][i % 2],
                    nbits: ((i + 13) % 32) as u8,
                })
                .collect();
            let indices: Vec<_> = (0..records).rev().collect();
            for padding in [0, 8] {
                for (tokens, strides) in [
                    ([a.as_slice(), b.as_slice()], [16 + padding, 5 + padding]),
                    ([b.as_slice(), a.as_slice()], [5 + padding, 16 + padding]),
                ] {
                    check(&samples, &indices, &values, tokens, &mapping, strides);
                    check(&samples, &indices, &wide, tokens, &mapping, strides);
                }
            }
        }
    }

    #[test]
    fn sparse_plan_requires_predictor_local_quarter_block_savings() {
        for lanes in [1, 4, 8] {
            let mut quarter = vec![1; 4 * lanes];
            quarter[lanes..2 * lanes].fill(0);
            let mut fifth = vec![1; 5 * lanes];
            fifth[lanes..2 * lanes].fill(0);
            let mut sparse_tail = vec![1; 2 * lanes + 1];
            sparse_tail[lanes..2 * lanes].fill(0);
            let mut wide = vec![0; 8 * lanes];
            wide[0] = 65535;
            wide[8 * lanes - 1] = 1;
            let parents: Vec<_> = [
                vec![1; lanes],
                vec![1; 2 * lanes + 1],
                quarter,
                fifth,
                sparse_tail,
                wide,
            ]
            .into_iter()
            .map(|counts| ParentHistogram {
                counts,
                extra_bits: 0,
            })
            .collect();
            let plan = SparseHistogramPlan::new(&parents, &[0, 1, 2, 3, 4, 5], lanes, true);
            // Other predictors' longer strides cannot make these dense ones sparse.
            assert_eq!(plan.blocks(0), None);
            assert_eq!(plan.blocks(1), None);
            assert_eq!(plan.blocks(2), Some([0, 2 * lanes, 3 * lanes].as_slice()));
            assert_eq!(plan.blocks(3), None);
            assert_eq!(plan.blocks(4), Some([0, 2 * lanes].as_slice()));
            assert_eq!(plan.blocks(5), Some([0, 7 * lanes].as_slice()));
            assert_eq!(plan.blocks(6), None);

            let selected = SparseHistogramPlan::new(&parents, &[5], lanes, true);
            assert_eq!(selected.blocks(2), None);
            assert_eq!(selected.blocks(5), plan.blocks(5));
            let disabled = SparseHistogramPlan::new(&parents, &[5], lanes, false);
            assert_eq!(disabled.blocks(5), None);
            assert_eq!(disabled.blocks.capacity(), 0);
            assert_eq!(disabled.ranges.capacity(), 0);
        }
    }

    #[test]
    fn sparse_plan_is_immutable_and_rebuilt_for_changed_parent_layouts() {
        let mut parents = vec![ParentHistogram {
            counts: vec![0; 33],
            extra_bits: 0,
        }];
        parents[0].counts[0] = 17;
        parents[0].counts[32] = 65535;
        let raw = SparseHistogramPlan::new(&parents, &[0], 8, true);
        let pointer = raw.blocks(0).unwrap().as_ptr();
        assert_eq!(raw.blocks(0), Some([0, 32].as_slice()));

        // A compacted scoring histogram or trimmed child has different addresses.
        parents[0].counts = vec![17, 65535];
        let compact = SparseHistogramPlan::new(&parents, &[0], 8, true);
        assert_eq!(compact.blocks(0), None);
        parents[0].counts = vec![0; 17];
        parents[0].counts[16] = 65535;
        let child = SparseHistogramPlan::new(&parents, &[0], 8, true);
        assert_eq!(child.blocks(0), Some([16].as_slice()));
        assert_eq!(raw.blocks(0), Some([0, 32].as_slice()));
        assert_eq!(raw.blocks(0).unwrap().as_ptr(), pointer);
    }

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
            early_cache_14: false,
            shards: true,
            padded: true,
            dense: DenseHistogram::Extended,
            property_bins: PropertyBins::All,
            sparse: true,
            paired: true,
            events: true,
        };
        tools.apply(&mut options);
        assert!(options.histogram_shards && options.padded_histogram);
        assert!(options.sparse_entropy && options.paired_cache);
        assert!(options.histogram_events);
        assert_eq!(options.dense_histogram, DenseHistogram::Extended);
        assert_eq!(options.property_bins, PropertyBins::All);
        HistogramOptions::default().apply(&mut options);
        assert!(!options.histogram_shards && !options.padded_histogram);
        assert!(!options.sparse_entropy && !options.paired_cache);
        assert!(!options.histogram_events);
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
