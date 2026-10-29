// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Node parent collection, lane-preserving compaction and sibling subtraction.
//! These raw parent counts stay independent of retained property-cache storage.

use super::{DenseHistogram, SampleStorage, Samples, histogram_stride};
use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ParentHistogram {
    pub(super) counts: Vec<u32>,
    pub(super) extra_bits: u64,
}

pub(super) fn parent_histograms(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
) -> Vec<ParentHistogram> {
    (0..samples.residuals.len())
        .map(|predictor| parent_histogram(samples, indices, predictor))
        .collect()
}

/// enc_ma_histograms.h::CollectMaParentHistogram: tokens fit a fixed 256-bin
/// accumulator, so finding the highest symbol needs no separate sample scan.
pub(super) fn parent_histogram(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    predictor: usize,
) -> ParentHistogram {
    let mut counts = [0u32; 256];
    let mut extra_bits = 0;
    let tokens = &samples.residuals[predictor];
    for &i in indices {
        let token = tokens[i];
        let weight = u32::from(samples.counts[i]);
        counts[usize::from(token.token)] += weight;
        extra_bits += u64::from(token.nbits) * u64::from(weight);
    }
    let symbols = counts
        .iter()
        .rposition(|&count| count != 0)
        .map_or(1, |s| s + 1);
    ParentHistogram {
        counts: counts[..symbols].to_vec(),
        extra_bits,
    }
}

pub(super) struct CompactHistograms {
    pub(super) parents: Vec<ParentHistogram>,
    pub(super) mapping: [u8; 256],
}

/// Source CompactMaHistograms's default, not its opt-in dense remapping.
/// The union of parent-active tokens preserves every possible child token.
#[cfg(test)]
pub(super) fn compact_parent_histograms(
    parents: &[ParentHistogram],
    lanes: usize,
) -> Option<CompactHistograms> {
    compact_parent_histograms_with_layout(parents, lanes, DenseHistogram::Disabled, false)
}

pub(super) fn compact_parent_histograms_with_layout(
    parents: &[ParentHistogram],
    lanes: usize,
    dense: DenseHistogram,
    padded: bool,
) -> Option<CompactHistograms> {
    if parents.is_empty() || !matches!(lanes, 1 | 4 | 8) {
        return None;
    }
    let symbols = parents.iter().map(|p| p.counts.len()).max()?;
    let stride = histogram_stride(symbols, lanes, padded);
    if !(32..=256).contains(&stride) {
        return None;
    }
    let mut used = [false; 256];
    for parent in parents {
        for (symbol, &count) in parent.counts.iter().enumerate() {
            used[symbol] |= count != 0;
        }
    }
    let mut next = [0usize; 8];
    let mut mapping = [0u8; 256];
    let mut compact_stride = 0;
    let mut dense_symbols = 0usize;
    for symbol in 0..stride {
        if used[symbol] {
            let lane = symbol % lanes;
            if dense != DenseHistogram::Disabled {
                mapping[symbol] = dense_symbols as u8;
                dense_symbols += 1;
                compact_stride = dense_symbols.div_ceil(lanes) * lanes;
            } else {
                mapping[symbol] = (next[lane] * lanes + lane) as u8;
                next[lane] += 1;
                compact_stride = compact_stride.max(next[lane] * lanes);
            }
        }
    }
    if compact_stride == 0 || compact_stride * 4 > stride * 3 {
        return None;
    }
    let parents = parents
        .iter()
        .map(|parent| {
            let mut counts = vec![0; compact_stride];
            for (symbol, &count) in parent.counts.iter().enumerate() {
                if used[symbol] {
                    counts[usize::from(mapping[symbol])] = count;
                }
            }
            // The default trim-entropy path is independent of storage stride.
            while counts.len() > 1 && counts.last() == Some(&0) {
                counts.pop();
            }
            ParentHistogram {
                counts,
                extra_bits: parent.extra_bits,
            }
        })
        .collect();
    Some(CompactHistograms { parents, mapping })
}

pub(super) fn subtract_parent_histograms(
    mut parent: Vec<ParentHistogram>,
    smaller: &[ParentHistogram],
) -> Result<Vec<ParentHistogram>> {
    for (parent, smaller) in parent.iter_mut().zip(smaller) {
        for (symbol, &count) in smaller.counts.iter().enumerate() {
            parent.counts[symbol] = parent.counts[symbol].checked_sub(count).ok_or_else(|| {
                Error::Encoding("MA parent histogram subtraction underflow".into())
            })?;
        }
        parent.extra_bits = parent
            .extra_bits
            .checked_sub(smaller.extra_bits)
            .ok_or_else(|| Error::Encoding("MA parent extra-bit subtraction underflow".into()))?;
        while parent.counts.len() > 1 && parent.counts.last() == Some(&0) {
            parent.counts.pop();
        }
    }
    Ok(parent)
}
