//! Opt-in approximate MA split search using node-local weighted quantiles.
//!
//! This is not an exact libjxl search-equivalence optimization. Whole original
//! bins are retained, so histogram collection remains exact at selected splits.

use super::HistogramSplit;
use crate::{Error, Result};

/// Prepare contiguous coarse rows without splitting any occupied original bin.
///
/// Live weights must be positive, sum to `total`, and share `used`'s indexing.
/// Quantiles select the nearest live prefix (ties select the earlier prefix).
/// Repeated boundaries collapse, including when one bin spans many quantiles.
/// `None` leaves all output buffers unchanged. On `Some`, `mask` addresses
/// `[first_live, last_live)` and mapping covers the complete original bin range.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    limit: u16,
    property_id: usize,
    used: &[usize],
    weights: &[u32],
    total: u32,
    mapping: &mut Vec<usize>,
    splits: &mut Vec<HistogramSplit>,
    mask: &mut Vec<u8>,
) -> Result<Option<usize>> {
    if limit == 0 {
        return Ok(None);
    }
    if !(2..=256).contains(&limit) {
        return Err(Error::Limit("MA histogram bin limit must be 0 or 2..=256"));
    }
    if property_id < 2 {
        return Ok(None);
    }
    let live = used.iter().filter(|&&count| count != 0).count();
    if live <= usize::from(limit) || total == 0 {
        return Ok(None);
    }
    debug_assert_eq!(used.len(), weights.len());
    debug_assert!(used.iter().zip(weights).all(|(&n, &w)| n == 0 || w > 0));
    debug_assert_eq!(
        used.iter()
            .zip(weights)
            .filter(|(n, _)| **n != 0)
            .map(|(_, &w)| u64::from(w))
            .sum::<u64>(),
        u64::from(total)
    );

    let first = used.iter().position(|&n| n != 0).unwrap();
    let last = used.iter().rposition(|&n| n != 0).unwrap();
    let max_splits = usize::from(limit) - 1;
    mapping
        .try_reserve(used.len().saturating_sub(mapping.len()))
        .map_err(|_| Error::Limit("MA histogram bin mapping allocation"))?;
    splits
        .try_reserve(max_splits.saturating_sub(splits.len()))
        .map_err(|_| Error::Limit("MA histogram bin splits allocation"))?;
    mask.try_reserve((last - first).saturating_sub(mask.len()))
        .map_err(|_| Error::Limit("MA histogram bin mask allocation"))?;
    splits.clear();
    mask.clear();
    mask.resize(last - first, 0);
    mapping.resize(used.len(), 0);

    let scale = u64::from(limit);
    let mut boundaries = (first + 1..last).filter(|&bin| used[bin] != 0);
    let mut current = (first, weights[first]);
    let mut previous = current;
    for quantile in 1..limit {
        let target = u64::from(total) * u64::from(quantile);
        while u64::from(current.1) * scale < target {
            let Some(bin) = boundaries.next() else {
                break;
            };
            previous = current;
            current = (bin, current.1 + weights[bin]);
        }
        // Compare scaled integer distances, not rounded quantile populations.
        // The first/last legal prefixes clamp targets inside giant edge bins.
        let previous_distance = (u64::from(previous.1) * scale).abs_diff(target);
        let current_distance = (u64::from(current.1) * scale).abs_diff(target);
        let (bin, cumulative) = if previous_distance <= current_distance {
            previous
        } else {
            current
        };
        if splits.last().is_none_or(|split| split.bin != bin) {
            splits.push(HistogramSplit {
                bin,
                row: splits.len(),
                totals: (total - cumulative, cumulative),
            });
        }
    }

    let rows = splits.len() + 1;
    debug_assert!(rows >= 2 && rows <= usize::from(limit) && rows < live);
    let mut row = 0;
    for (bin, mapped) in mapping.iter_mut().enumerate() {
        *mapped = row;
        if row < splits.len() && splits[row].bin == bin {
            mask[bin - first] = 1;
            row += 1;
        }
    }
    Ok(Some(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(used: &[usize], weights: &[u32]) -> u32 {
        u32::try_from(
            used.iter()
                .zip(weights)
                .filter(|(n, _)| **n != 0)
                .map(|(_, &w)| u64::from(w))
                .sum::<u64>(),
        )
        .unwrap()
    }

    fn split_values(splits: &[HistogramSplit]) -> Vec<(usize, usize, (u32, u32))> {
        splits.iter().map(|s| (s.bin, s.row, s.totals)).collect()
    }

    // Independently enumerate every legal prefix for every target quantile.
    fn expected_bins(limit: u16, used: &[usize], weights: &[u32]) -> Vec<usize> {
        let population = total(used, weights);
        let last = used.iter().rposition(|&n| n != 0).unwrap();
        let mut cumulative = 0u64;
        let mut prefixes = Vec::new();
        for bin in 0..last {
            if used[bin] != 0 {
                cumulative += u64::from(weights[bin]);
                prefixes.push((bin, cumulative));
            }
        }
        let mut result = Vec::new();
        for quantile in 1..limit {
            let target = u64::from(population) * u64::from(quantile);
            let &(bin, _) = prefixes
                .iter()
                .min_by_key(|&&(bin, prefix)| ((prefix * u64::from(limit)).abs_diff(target), bin))
                .unwrap();
            if result.last() != Some(&bin) {
                result.push(bin);
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_layout(
        limit: u16,
        used: &[usize],
        weights: &[u32],
        rows: usize,
        mapping: &[usize],
        splits: &[HistogramSplit],
        mask: &[u8],
    ) {
        let live = used.iter().filter(|&&n| n != 0).count();
        let first = used.iter().position(|&n| n != 0).unwrap();
        let last = used.iter().rposition(|&n| n != 0).unwrap();
        assert!(2 <= rows && rows <= usize::from(limit) && rows < live);
        assert_eq!(mapping.len(), used.len());
        assert_eq!(splits.len(), rows - 1);
        assert_eq!(mask.len(), last - first);
        assert_eq!(
            splits.iter().map(|s| s.bin).collect::<Vec<_>>(),
            expected_bins(limit, used, weights)
        );

        let mut row_records = vec![0; rows];
        let mut row_weights = vec![0u64; rows];
        for bin in 0..used.len() {
            let row = mapping[bin];
            assert!(row < rows);
            assert_eq!(row, splits.partition_point(|s| s.bin < bin));
            if used[bin] != 0 {
                row_records[row] += used[bin];
                row_weights[row] += u64::from(weights[bin]);
            }
        }
        assert!(row_records.iter().all(|&n| n != 0));
        assert!(row_weights.iter().all(|&w| w != 0));
        assert_eq!(
            row_records.iter().sum::<usize>(),
            used.iter().sum::<usize>()
        );
        assert_eq!(
            row_weights.iter().sum::<u64>(),
            u64::from(total(used, weights))
        );
        assert!(mapping.windows(2).all(|pair| pair[0] <= pair[1]));
        for (row, split) in splits.iter().enumerate() {
            assert_eq!(split.row, row);
            assert_eq!(
                Some(split.bin),
                (first..=last)
                    .rev()
                    .find(|&bin| used[bin] != 0 && mapping[bin] == row)
            );
            let original: Vec<_> = (0..used.len())
                .filter(|&bin| used[bin] != 0 && bin <= split.bin)
                .collect();
            let coarse: Vec<_> = (0..used.len())
                .filter(|&bin| used[bin] != 0 && mapping[bin] <= row)
                .collect();
            assert_eq!(coarse, original);
            let below: u64 = original.iter().map(|&bin| u64::from(weights[bin])).sum();
            let above: u64 = (split.bin + 1..used.len())
                .filter(|&bin| used[bin] != 0)
                .map(|bin| u64::from(weights[bin]))
                .sum();
            assert_eq!(
                (above, below),
                (u64::from(split.totals.0), u64::from(split.totals.1))
            );
            assert_eq!(row_weights[..=row].iter().sum::<u64>(), below);
            assert_eq!(row_weights[row + 1..].iter().sum::<u64>(), above);
        }
        for (offset, &marked) in mask.iter().enumerate() {
            assert_eq!(
                marked,
                u8::from(splits.iter().any(|s| s.bin == first + offset))
            );
        }
    }

    fn check(limit: u16, used: &[usize], weights: &[u32]) -> Vec<usize> {
        let (mut mapping, mut splits, mut mask) = (Vec::new(), Vec::new(), Vec::new());
        let result = prepare(
            limit,
            2,
            used,
            weights,
            total(used, weights),
            &mut mapping,
            &mut splits,
            &mut mask,
        )
        .unwrap();
        if used.iter().filter(|&&n| n != 0).count() <= usize::from(limit) {
            assert_eq!(result, None);
            assert_eq!(
                (mapping.capacity(), splits.capacity(), mask.capacity()),
                (0, 0, 0)
            );
        } else {
            assert_layout(
                limit,
                used,
                weights,
                result.unwrap(),
                &mapping,
                &splits,
                &mask,
            );
        }
        splits.iter().map(|s| s.bin).collect()
    }

    #[test]
    fn exhaustive_small_weighted_partitions_and_holes() {
        for width in 0..=7 {
            for mut pattern in 0..4usize.pow(width as u32) {
                let mut used = vec![0; width];
                let mut weights = vec![0; width];
                for bin in 0..width {
                    let state = pattern % 4;
                    pattern /= 4;
                    if state != 0 {
                        used[bin] = 1 + bin % 3;
                        weights[bin] = [0, 3, 7, 23][state];
                    }
                }
                for limit in 2..=6 {
                    check(limit, &used, &weights);
                }
            }
        }
    }

    #[test]
    fn quantiles_use_weight_not_record_count_and_ties_choose_earlier() {
        assert_eq!(check(3, &[1; 6], &[1, 1, 1, 9, 9, 9]), [3, 4]);
        assert_eq!(check(3, &[1, 1, 1, 9, 9, 9], &[1, 1, 1, 9, 9, 9]), [3, 4]);
        assert_eq!(check(2, &[1; 3], &[1; 3]), [0]);
        assert_eq!(check(3, &[1; 8], &[1; 8]), [2, 4]);
    }

    #[test]
    fn giant_bins_collapse_quantiles_without_empty_rows() {
        assert_eq!(check(4, &[1; 5], &[1000, 1, 1, 1, 1]), [0]);
        assert_eq!(check(4, &[1; 5], &[1, 1, 1000, 1, 1]), [1, 2]);
        assert_eq!(check(4, &[1; 5], &[1, 1, 1, 1, 1000]), [3]);
        assert_eq!(check(4, &[1; 5], &[1, 1, u32::MAX - 4, 1, 1]), [1, 2]);
    }

    #[test]
    fn limit_endpoints_and_full_u32_weight_sums() {
        check(
            2,
            &[1, 0, 1, 0, 1],
            &[2_000_000_000, 0, 2_000_000_000, 0, 294_967_295],
        );
        check(2, &[1; 3], &[1, 1, u32::MAX - 2]);
        for limit in [2, 255, 256] {
            let width = usize::from(limit) + 1;
            assert_eq!(
                check(limit, &vec![1; width], &vec![1; width]).len(),
                width - 2
            );
        }
        let mut weights = vec![16_000_000; 257];
        weights[256] = u32::MAX - 16_000_000 * 256;
        check(256, &vec![1; 257], &weights);
        assert!(check(256, &vec![1; 256], &vec![1; 256]).is_empty());
    }

    #[test]
    fn skipped_nodes_preserve_empty_and_reused_buffers_without_reserving() {
        for (limit, property, used, weights, population) in [
            (0, 2, &[1usize; 5][..], &[][..], 0),
            (2, 0, &[1; 5][..], &[][..], 0),
            (2, 1, &[1; 5][..], &[][..], 0),
            (2, 2, &[][..], &[][..], 0),
            (2, 2, &[0, 1, 0][..], &[0, 5, 0][..], 5),
            (2, 2, &[1, 0, 1][..], &[1, 0, 3][..], 4),
            (256, 2, &[1; 5][..], &[1; 5][..], 5),
            (2, 2, &[1; 5][..], &[0; 5][..], 0),
        ] {
            for reused in [false, true] {
                let (mut mapping, mut splits, mut mask) = (Vec::new(), Vec::new(), Vec::new());
                if reused {
                    mapping.extend_from_slice(&[88, 99, 77]);
                    splits.push(HistogramSplit {
                        bin: 44,
                        row: 55,
                        totals: (66, 77),
                    });
                    mask.extend_from_slice(&[4, 5, 6, 7]);
                }
                let before = (mapping.clone(), split_values(&splits), mask.clone());
                let capacities = (mapping.capacity(), splits.capacity(), mask.capacity());
                let pointers = (mapping.as_ptr(), splits.as_ptr(), mask.as_ptr());
                assert_eq!(
                    prepare(
                        limit,
                        property,
                        used,
                        weights,
                        population,
                        &mut mapping,
                        &mut splits,
                        &mut mask
                    )
                    .unwrap(),
                    None
                );
                assert_eq!(
                    (&mapping, split_values(&splits), &mask),
                    (&before.0, before.1, &before.2)
                );
                assert_eq!(
                    (mapping.capacity(), splits.capacity(), mask.capacity()),
                    capacities
                );
                assert_eq!((mapping.as_ptr(), splits.as_ptr(), mask.as_ptr()), pointers);
                if !reused {
                    assert_eq!(capacities, (0, 0, 0));
                }
            }
        }
    }

    #[test]
    fn invalid_limits_do_not_mutate_or_allocate_outputs() {
        for limit in [1, 257, u16::MAX] {
            let (mut mapping, mut splits, mut mask) = (Vec::new(), Vec::new(), Vec::new());
            assert!(matches!(
                prepare(limit, 2, &[], &[], 0, &mut mapping, &mut splits, &mut mask),
                Err(Error::Limit(_))
            ));
            assert_eq!(
                (mapping.capacity(), splits.capacity(), mask.capacity()),
                (0, 0, 0)
            );
        }
    }

    #[test]
    fn wide_original_bins_and_reused_buffers_clear_old_rows_and_masks() {
        let width = 70_003;
        let mut mapping = vec![usize::MAX; width + 10];
        let mut splits = vec![
            HistogramSplit {
                bin: usize::MAX,
                row: usize::MAX,
                totals: (0, 0)
            };
            300
        ];
        let mut mask = vec![255; width + 10];
        let pointers = (mapping.as_ptr(), splits.as_ptr(), mask.as_ptr());
        for bins in [
            vec![1, 257, 4096, 65_537, 70_001],
            vec![3, 1000, 65_536, 70_000, 70_002],
            vec![0, 1, 2, 3, 4],
        ] {
            let mut used = vec![0; width];
            let mut weights = vec![0; width];
            for (i, &bin) in bins.iter().enumerate() {
                used[bin] = i + 1;
                weights[bin] = 7 * (i as u32 + 1);
            }
            let rows = prepare(
                3,
                16,
                &used,
                &weights,
                total(&used, &weights),
                &mut mapping,
                &mut splits,
                &mut mask,
            )
            .unwrap()
            .unwrap();
            assert_layout(3, &used, &weights, rows, &mapping, &splits, &mask);
            assert_eq!((mapping.as_ptr(), splits.as_ptr(), mask.as_ptr()), pointers);
        }
        let used = [1; 4];
        let weights = [1; 4];
        let rows = prepare(
            2,
            2,
            &used,
            &weights,
            4,
            &mut mapping,
            &mut splits,
            &mut mask,
        )
        .unwrap()
        .unwrap();
        assert_layout(2, &used, &weights, rows, &mapping, &splits, &mask);
        assert_eq!((mapping.as_ptr(), splits.as_ptr(), mask.as_ptr()), pointers);
    }
}
