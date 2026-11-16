//! Exact paired accumulation for fresh two-predictor property caches.

use super::{BinHistogram, ParentHistogram, PropertyValue, SampleStorage, Samples};

/// Pack two independent nonnegative counters into each u64. The parent's
/// totals bound every bin, so neither low half can carry into the high half.
/// Fresh buffers are borrowed as temporary packed storage and unpacked before
/// publication; no extra allocation or change to the cached layout is needed.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_pair<T: PropertyValue>(
    samples: &Samples<impl SampleStorage>,
    indices: &[usize],
    values: &[T],
    parents: &[ParentHistogram],
    used: &mut [usize],
    weights: &mut [u32],
    first: &mut BinHistogram,
    second: &mut BinHistogram,
    contiguous: Option<std::ops::Range<usize>>,
) -> bool {
    if usize::BITS < 64
        || indices.len() < 512
        || indices.len() > u32::MAX as usize
        || parents.len() != 2
        || parents.iter().any(|p| p.extra_bits > u64::from(u32::MAX))
        || parents[0]
            .counts
            .iter()
            .try_fold(0u32, |n, &w| n.checked_add(w))
            .is_none()
    {
        return false;
    }
    debug_assert!(used.iter().all(|&v| v == 0));
    debug_assert!(weights.iter().all(|&v| v == 0));
    debug_assert!(first.extra.iter().chain(&second.extra).all(|&v| v == 0));
    let (a, b) = (&samples.residuals[0][..], &samples.residuals[1][..]);
    if let Some(range) = contiguous {
        debug_assert!(indices.iter().copied().eq(range.clone()));
        let rows = values[range.clone()]
            .iter()
            .zip(&samples.counts[range.clone()])
            .zip(&a[range.clone()])
            .zip(&b[range]);
        for (((&value, &weight), &a), &b) in rows {
            let bin = value.index();
            let weight = u64::from(weight);
            used[bin] += ((weight << 32) | 1) as usize;
            first.counts[bin * first.symbols + usize::from(a.token)] += weight as u32;
            second.counts[bin * second.symbols + usize::from(b.token)] += weight as u32;
            first.extra[bin] += weight * ((u64::from(b.nbits) << 32) | u64::from(a.nbits));
        }
    } else {
        for &i in indices {
            let bin = values[i].index();
            let weight = u64::from(samples.counts[i]);
            used[bin] += ((weight << 32) | 1) as usize;
            let (a, b) = (a[i], b[i]);
            first.counts[bin * first.symbols + usize::from(a.token)] += weight as u32;
            second.counts[bin * second.symbols + usize::from(b.token)] += weight as u32;
            first.extra[bin] += weight * ((u64::from(b.nbits) << 32) | u64::from(a.nbits));
        }
    }
    for (records, weight) in used.iter_mut().zip(weights) {
        *weight = ((*records as u64) >> 32) as u32;
        *records = *records as u32 as usize;
    }
    for (a, b) in first.extra.iter_mut().zip(&mut second.extra) {
        *b = *a >> 32;
        *a = u64::from(*a as u32);
    }
    true
}

#[cfg(test)]
#[path = "cache_packed_tests.rs"]
mod tests;
