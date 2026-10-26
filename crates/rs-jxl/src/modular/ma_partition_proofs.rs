//! Exact node-local proofs for monotone property partition reuse.

use super::PropertyValue;

/// Hash the ordered nonzero occupancy counts, ignoring empty bin labels.
/// Equal fingerprints only select candidates; they never prove equivalence.
pub(super) fn signature(used: &[usize]) -> u64 {
    let mut hash = 14695981039346656037u64;
    for &count in used {
        if count != 0 {
            hash = (hash ^ count as u64).wrapping_mul(1099511628211);
        }
    }
    hash
}

/// Prove identical ordered threshold partitions on exactly `indices`.
///
/// Every live current bin must map to one previous bin, and those mappings
/// must be strictly increasing. Thus neither splitting, merging nor reversal
/// is allowed, but arbitrary gaps and different quantizer tails are harmless.
///
/// Callers provide valid row indices in both columns and current bins below
/// `bins`. Previous bins need not fit `bins`. Scratch contents are unspecified
/// after either outcome; capacity is retained. Work is O(indices.len() + bins)
/// with O(bins) scratch, independent of the largest previous-bin label.
pub(super) fn same_partition<T: PropertyValue, U: PropertyValue>(
    indices: &[usize],
    current: &[T],
    previous: &[U],
    bins: usize,
    remap: &mut Vec<usize>,
) -> bool {
    remap.clear();
    if indices.is_empty() {
        return true;
    }
    remap.resize(bins, usize::MAX);

    // Store representative rows, not bin labels: usize::MAX cannot be a
    // valid row index, even when a wide previous-bin label equals it.
    for &row in indices {
        let representative = &mut remap[current[row].index()];
        if *representative == usize::MAX {
            *representative = row;
        } else if previous[*representative].index() != previous[row].index() {
            return false;
        }
    }

    let mut last = None;
    for &representative in remap.iter() {
        if representative == usize::MAX {
            continue;
        }
        let mapped = previous[representative].index();
        if last.is_some_and(|last| mapped <= last) {
            return false;
        }
        last = Some(mapped);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{PropertyValue, same_partition, signature};

    #[test]
    fn signature_ignores_empty_labels_but_preserves_count_order() {
        assert_eq!(signature(&[]), signature(&[0; 32]));
        assert_eq!(signature(&[2, 1, 3]), signature(&[0, 2, 0, 1, 0, 3, 0]));
        assert_ne!(signature(&[2, 1, 3]), signature(&[1, 2, 3]));
        assert_ne!(signature(&[2, 1, 3]), signature(&[2, 1, 4]));
        assert_ne!(signature(&[2, 1]), signature(&[3]));
        assert_eq!(signature(&[usize::MAX]), signature(&[0, usize::MAX, 0]));
    }

    #[test]
    fn equal_fingerprints_do_not_prove_row_membership() {
        let current = [0u8, 0, 2, 2];
        let previous = [5u32, 7, 5, 7];
        assert_eq!(signature(&[2, 0, 2]), signature(&[0, 0, 0, 0, 0, 2, 0, 2]));
        assert!(!same_partition(
            &[3, 1, 2, 0],
            &current,
            &previous,
            3,
            &mut Vec::new(),
        ));
    }

    #[test]
    fn rejects_reversed_merged_and_split_mappings() {
        let current = [1u8, 1, 3, 3, 7, 7];
        let indices = [5, 0, 2, 4, 1, 3];
        let mut remap = Vec::new();
        for previous in [
            [9u32, 9, 5, 5, 2, 2],
            [2, 2, 2, 2, 8, 8],
            [2, 2, 5, 5, 8, 9],
            [2, 2, 9, 9, 5, 5],
        ] {
            assert!(!same_partition(
                &indices, &current, &previous, 8, &mut remap,
            ));
            assert!(!same_partition(
                &indices, &previous, &current, 10, &mut remap,
            ));
        }
        assert!(same_partition(
            &indices,
            &current,
            &[2u32, 2, 5, 5, 9, 9],
            8,
            &mut remap,
        ));
    }

    #[test]
    fn empty_and_single_live_bin_partitions() {
        let mut remap = vec![123; 16];
        assert!(same_partition::<u8, u32>(&[], &[], &[], 0, &mut remap));
        assert!(remap.is_empty());
        assert!(same_partition(
            &[],
            &[255u8],
            &[0u32, 1],
            usize::MAX,
            &mut remap,
        ));
        assert!(same_partition(
            &[2, 0, 1, 2],
            &[4u8, 4, 4],
            &[u32::MAX; 3],
            8,
            &mut remap,
        ));
        assert!(!same_partition(
            &[2, 0, 1],
            &[4u8, 4, 4],
            &[u32::MAX, 0, u32::MAX],
            8,
            &mut remap,
        ));
        assert!(same_partition(
            &[1],
            &[0u8, 255, 1],
            &[9u32, u32::MAX, 0],
            256,
            &mut remap,
        ));
    }

    #[test]
    fn sparse_labels_and_different_widths_are_equivalent() {
        let packed = [2u8, 17, 255, 17];
        let wide = [0u32, 4096, 65535, 4096];
        let indices = [2, 0, 3, 1];
        let mut remap = Vec::new();
        assert!(same_partition(&indices, &wide, &packed, 65536, &mut remap));
        assert!(same_partition(&indices, &packed, &wide, 256, &mut remap));
        assert!(same_partition(
            &indices,
            &packed,
            &[1u32, 1_000_000, u32::MAX, 1_000_000],
            256,
            &mut remap,
        ));
    }

    #[test]
    fn successful_proofs_are_inherited_by_every_subset() {
        let current = [0u8, 2, 2, 7, 12, 12, 15];
        let previous = [9u32, 500, 500, 900, 1000, 1000, 60000];
        let mut remap = Vec::new();
        for mask in 0usize..(1 << current.len()) {
            let indices: Vec<_> = (0..current.len())
                .rev()
                .filter(|&row| mask & (1 << row) != 0)
                .collect();
            assert!(same_partition(
                &indices, &current, &previous, 16, &mut remap,
            ));
        }

        // A failed parent proof does not rule out equivalent descendants.
        assert!(!same_partition(
            &[0, 1, 2],
            &[0u8, 1, 2],
            &[0u32, 2, 1],
            3,
            &mut remap,
        ));
        assert!(same_partition(
            &[2, 0],
            &[0u8, 1, 2],
            &[0u32, 2, 1],
            3,
            &mut remap,
        ));
    }

    #[test]
    fn scratch_is_reset_after_failures_and_retains_capacity() {
        let mut remap = vec![17; 8192];
        let pointer = remap.as_ptr();
        let capacity = remap.capacity();
        assert!(!same_partition(
            &[0, 1],
            &[0u8, 0],
            &[2u32, 3],
            4096,
            &mut remap,
        ));
        assert!(same_partition(
            &[1, 0],
            &[1u8, 3],
            &[20u32, 30],
            4,
            &mut remap,
        ));
        assert!(!same_partition(
            &[0, 1],
            &[0u32, 4095],
            &[8u8, 4],
            4096,
            &mut remap,
        ));
        assert!(same_partition(
            &[0, 1],
            &[0u32, 4095],
            &[4u8, 8],
            4096,
            &mut remap,
        ));
        assert!(same_partition::<u8, u32>(&[], &[], &[], 0, &mut remap));
        assert_eq!(remap.as_ptr(), pointer);
        assert_eq!(remap.capacity(), capacity);
    }

    // Enumerate actual nontrivial scalar threshold cuts, independently of
    // the production mapping proof. Empty gaps produce no additional cuts.
    fn threshold_partitions<T: PropertyValue>(indices: &[usize], values: &[T]) -> Vec<Vec<bool>> {
        let mut thresholds: Vec<_> = indices.iter().map(|&row| values[row].index()).collect();
        thresholds.sort_unstable();
        thresholds.dedup();
        thresholds.pop();
        thresholds
            .into_iter()
            .map(|threshold| {
                indices
                    .iter()
                    .map(|&row| values[row].index() <= threshold)
                    .collect()
            })
            .collect()
    }

    fn assert_matches_oracle<T: PropertyValue, U: PropertyValue>(
        indices: &[usize],
        current: &[T],
        previous: &[U],
        bins: usize,
        remap: &mut Vec<usize>,
    ) {
        let current_cuts = threshold_partitions(indices, current);
        let previous_cuts = threshold_partitions(indices, previous);
        assert_eq!(
            same_partition(indices, current, previous, bins, remap),
            current_cuts == previous_cuts,
            "indices={indices:?}, current cuts={current_cuts:?}, previous cuts={previous_cuts:?}",
        );
    }

    #[test]
    fn generated_columns_match_scalar_threshold_partition_oracle() {
        let mut remap = Vec::new();
        // Exhaust every pair of three-row, three-bin columns and every
        // live subset, with independent labels in all four width pairings.
        for current_code in 0..27 {
            let digits = [current_code % 3, current_code / 3 % 3, current_code / 9];
            let current_packed = digits.map(|digit| [0u8, 3, 9][digit]);
            let current_wide = digits.map(|digit| [0u32, 257, 1024][digit]);
            for previous_code in 0..27 {
                let digits = [previous_code % 3, previous_code / 3 % 3, previous_code / 9];
                let previous_packed = digits.map(|digit| [2u8, 7, 255][digit]);
                let previous_wide = digits.map(|digit| [17u32, 4096, u32::MAX][digit]);
                for mask in 0..8 {
                    let indices: Vec<_> = [2, 0, 1]
                        .into_iter()
                        .filter(|&row| mask & (1 << row) != 0)
                        .collect();
                    assert_matches_oracle(
                        &indices,
                        &current_packed,
                        &previous_packed,
                        10,
                        &mut remap,
                    );
                    assert_matches_oracle(
                        &indices,
                        &current_packed,
                        &previous_wide,
                        10,
                        &mut remap,
                    );
                    assert_matches_oracle(
                        &indices,
                        &current_wide,
                        &previous_packed,
                        1025,
                        &mut remap,
                    );
                    assert_matches_oracle(
                        &indices,
                        &current_wide,
                        &previous_wide,
                        1025,
                        &mut remap,
                    );

                    if threshold_partitions(&indices, &current_packed)
                        == threshold_partitions(&indices, &previous_packed)
                    {
                        let mut current_used = [0; 10];
                        let mut previous_used = [0; 256];
                        for &row in &indices {
                            current_used[current_packed[row].index()] += 1;
                            previous_used[previous_packed[row].index()] += 1;
                        }
                        assert_eq!(signature(&current_used), signature(&previous_used));
                    }
                }
            }
        }
    }
}
