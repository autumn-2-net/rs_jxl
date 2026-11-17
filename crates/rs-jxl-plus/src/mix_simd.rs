// Mixer arithmetic translated from PJA's pas/pjav7c.c, pjav7c_loop.h,
// pjaj2kmc.c, pjaj2khm.c and pjaj2kpm.c at revision
// 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd (MIT).
// See ../LICENSE-PJA and ../NOTICE-PJA.
//! Portable and runtime-dispatched x86_64 AVX2 mixer primitives.
//!
//! Every product, reduction and weight addition wraps modulo 2^32. The
//! wrapped signed product is divided by 2^shift with truncation toward zero
//! before weight addition and signed clamping. No widened multiply or
//! saturating addition may replace these operations.

const MAX_INPUTS: usize = 16;

/// Fixed-width v7 mixer. Unlike the general helpers, the dimensions and
/// arithmetic parameters are compile-time constants. Runtime detection still
/// protects every AVX2 instruction; other targets use the same scalar rules.
#[inline]
pub(crate) fn v7_dot(inputs: &[i32; 8], weights: &[i32; 8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected; arrays contain all eight accessed lanes.
        return unsafe { v7_dot_avx2(inputs, weights) };
    }
    dot_scalar(inputs, weights)
}

/// v7 uses zero input/weight padding up to eight lanes. Those lanes stay zero,
/// so this full-width update equals updating only the active prefix.
#[inline]
pub(crate) fn v7_update(weights: &mut [i32; 8], inputs: &[i32; 8], error: i32) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected; complete arrays, exclusive weight borrow.
        unsafe { v7_update_avx2(weights, inputs, error) };
        return;
    }
    update_scalar(weights, inputs, error, 14, 131072);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn v7_dot_avx2(inputs: &[i32; 8], weights: &[i32; 8]) -> i32 {
    use std::arch::x86_64::*;
    // SAFETY: each unaligned load reads exactly one valid eight-element array.
    let input = unsafe { _mm256_loadu_si256(inputs.as_ptr().cast()) };
    let weight = unsafe { _mm256_loadu_si256(weights.as_ptr().cast()) };
    let product = _mm256_mullo_epi32(input, weight);
    let sum = _mm_add_epi32(
        _mm256_castsi256_si128(product),
        _mm256_extracti128_si256::<1>(product),
    );
    let sum = _mm_add_epi32(sum, _mm_srli_si128::<8>(sum));
    _mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_srli_si128::<4>(sum)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn v7_update_avx2(weights: &mut [i32; 8], inputs: &[i32; 8], error: i32) {
    use std::arch::x86_64::*;
    // SAFETY: complete, valid arrays; unaligned load/store has no alignment
    // precondition, and the exclusive borrow permits the eight-lane store.
    let input = unsafe { _mm256_loadu_si256(inputs.as_ptr().cast()) };
    let weight = unsafe { _mm256_loadu_si256(weights.as_ptr().cast()) };
    let product = _mm256_mullo_epi32(input, _mm256_set1_epi32(error));
    let negative = _mm256_srai_epi32::<31>(product);
    let rounded = _mm256_add_epi32(
        product,
        _mm256_and_si256(negative, _mm256_set1_epi32(16383)),
    );
    let delta = _mm256_srai_epi32::<14>(rounded);
    let updated = _mm256_add_epi32(weight, delta);
    let updated = _mm256_min_epi32(
        _mm256_max_epi32(updated, _mm256_set1_epi32(-131072)),
        _mm256_set1_epi32(131072),
    );
    unsafe { _mm256_storeu_si256(weights.as_mut_ptr().cast(), updated) };
}

/// Wrapping signed dot product of equally sized slices, with at most 16 lanes.
pub(crate) fn dot(inputs: &[i32], weights: &[i32]) -> i32 {
    assert_eq!(inputs.len(), weights.len());
    assert!(inputs.len() <= MAX_INPUTS);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is detected, and equal lengths have been checked. The
        // implementation accesses only complete eight-lane chunks in bounds.
        return unsafe { dot_avx2(inputs, weights) };
    }
    dot_scalar(inputs, weights)
}

/// Update at most 16 weights; slices must have equal length, shift < 32 and
/// max >= 0. Products and additions wrap before the signed clamp is applied.
pub(crate) fn update(weights: &mut [i32], inputs: &[i32], error: i32, shift: u32, max: i32) {
    assert_eq!(weights.len(), inputs.len());
    assert!(inputs.len() <= MAX_INPUTS);
    assert!(shift < 32 && max >= 0);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: feature detection and argument checks satisfy the target
        // feature and bounded-access preconditions of update_avx2.
        unsafe { update_avx2(weights, inputs, error, shift, max) };
        return;
    }
    update_scalar(weights, inputs, error, shift, max);
}

fn dot_scalar(inputs: &[i32], weights: &[i32]) -> i32 {
    inputs
        .iter()
        .zip(weights)
        .fold(0i32, |sum, (&input, &weight)| {
            sum.wrapping_add(input.wrapping_mul(weight))
        })
}

fn trunc_shift(value: i32, shift: u32) -> i32 {
    let mask = ((1u32 << shift) - 1) as i32;
    value.wrapping_add((value >> 31) & mask) >> shift
}

fn update_scalar(weights: &mut [i32], inputs: &[i32], error: i32, shift: u32, max: i32) {
    for (weight, &input) in weights.iter_mut().zip(inputs) {
        let delta = trunc_shift(input.wrapping_mul(error), shift);
        *weight = weight.wrapping_add(delta).clamp(-max, max);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(inputs: &[i32], weights: &[i32]) -> i32 {
    use std::arch::x86_64::*;
    let n = inputs.len() / 8 * 8;
    let mut acc = _mm256_setzero_si256();
    for offset in (0..n).step_by(8) {
        // SAFETY: offset+8 <= n <= both slice lengths. Unaligned loads
        // require no additional alignment. The caller has detected AVX2.
        let input = unsafe { _mm256_loadu_si256(inputs.as_ptr().add(offset).cast()) };
        let weight = unsafe { _mm256_loadu_si256(weights.as_ptr().add(offset).cast()) };
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(input, weight));
    }
    let mut lanes = [0i32; 8];
    // SAFETY: this store initializes exactly the eight i32 elements of lanes;
    // storeu imposes no alignment requirement.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
    lanes
        .into_iter()
        .fold(dot_scalar(&inputs[n..], &weights[n..]), i32::wrapping_add)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_avx2(weights: &mut [i32], inputs: &[i32], error: i32, shift: u32, max: i32) {
    use std::arch::x86_64::*;
    let n = inputs.len() / 8 * 8;
    let error = _mm256_set1_epi32(error);
    let rounding = _mm256_set1_epi32(((1u32 << shift) - 1) as i32);
    let shifts = _mm256_set1_epi32(shift as i32);
    let low = _mm256_set1_epi32(-max);
    let high = _mm256_set1_epi32(max);
    for offset in (0..n).step_by(8) {
        // SAFETY: complete chunks are in bounds of both equally sized
        // slices; loadu/storeu accept their natural i32 alignment. Rust's
        // mutable borrow excludes conflicting references to weights, and
        // the caller has detected AVX2 and checked shift/max.
        let input = unsafe { _mm256_loadu_si256(inputs.as_ptr().add(offset).cast()) };
        let weight = unsafe { _mm256_loadu_si256(weights.as_ptr().add(offset).cast()) };
        let product = _mm256_mullo_epi32(input, error);
        let negative = _mm256_srai_epi32::<31>(product);
        let rounded = _mm256_add_epi32(product, _mm256_and_si256(negative, rounding));
        let delta = _mm256_srav_epi32(rounded, shifts);
        let weight = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(weight, delta), low), high);
        unsafe { _mm256_storeu_si256(weights.as_mut_ptr().add(offset).cast(), weight) };
    }
    // The vector error is broadcast; recover its scalar value for the tail.
    let scalar_error = _mm256_extract_epi32::<0>(error);
    update_scalar(&mut weights[n..], &inputs[n..], scalar_error, shift, max);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn next(state: &mut u32) -> i32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state as i32
    }

    #[test]
    fn fixed_v7_matches_active_scalar_prefix() {
        let mut seed = 0x64ab_eb9c;
        for active in 4..=8 {
            for _ in 0..1024 {
                let mut inputs = [0; 8];
                let mut weights = [0; 8];
                for i in 0..active {
                    inputs[i] = next(&mut seed);
                    weights[i] = next(&mut seed);
                }
                assert_eq!(v7_dot(&inputs, &weights), dot_scalar(&inputs, &weights));
                let error = next(&mut seed);
                let mut expected = weights;
                update_scalar(
                    &mut expected[..active],
                    &inputs[..active],
                    error,
                    14,
                    131072,
                );
                v7_update(&mut weights, &inputs, error);
                assert_eq!(weights, expected);
            }
        }
    }

    #[test]
    fn scalar_truncation_boundaries() {
        for value in [i32::MIN, i32::MIN + 1, -65537, -1, 0, 1, 65537, i32::MAX] {
            for shift in [0, 1, 14, 16, 31] {
                assert_eq!(
                    trunc_shift(value, shift),
                    (i64::from(value) / (1i64 << shift)) as i32
                );
            }
        }
        assert_eq!(dot(&[], &[]), 0);
        let mut empty = [];
        update(&mut empty, &[], i32::MIN, 31, 0);
        assert_eq!(dot(&[i32::MIN, i32::MAX], &[i32::MIN, i32::MAX]), 1);
    }

    #[test]
    fn runtime_dispatch_matches_scalar() {
        let mut state = 0x91aa_047bu32;
        for count in 0..=MAX_INPUTS {
            for _ in 0..64 {
                let inputs: Vec<_> = (0..count).map(|_| next(&mut state)).collect();
                let weights: Vec<_> = (0..count).map(|_| next(&mut state)).collect();
                assert_eq!(dot(&inputs, &weights), dot_scalar(&inputs, &weights));
                let error = next(&mut state);
                for shift in [0, 1, 14, 16, 31] {
                    for max in [0, 131072, 1 << 22, i32::MAX] {
                        let mut expected = weights.clone();
                        let mut actual = weights.clone();
                        update_scalar(&mut expected, &inputs, error, shift, max);
                        update(&mut actual, &inputs, error, shift, max);
                        assert_eq!(actual, expected, "count{count} shift{shift} max{max}");
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn direct_avx2_extremes_and_random_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let extreme = [i32::MIN, i32::MAX, -1, 0, 1, -131072, 131072, 1 << 22];
        let mut state = 0x3914_c0efu32;
        for count in 0..=MAX_INPUTS {
            for iteration in 0..128 {
                let inputs: Vec<_> = (0..count)
                    .map(|i| {
                        if iteration == 0 {
                            extreme[i % 8]
                        } else {
                            next(&mut state)
                        }
                    })
                    .collect();
                let weights: Vec<_> = (0..count)
                    .map(|i| {
                        if iteration == 0 {
                            extreme[(i + 3) % 8]
                        } else {
                            next(&mut state)
                        }
                    })
                    .collect();
                // SAFETY: feature detection above, equal bounded slice lengths.
                assert_eq!(
                    unsafe { dot_avx2(&inputs, &weights) },
                    dot_scalar(&inputs, &weights)
                );
                for error in [i32::MIN, i32::MAX, -1, 0, 1, next(&mut state)] {
                    for shift in [0, 1, 14, 16, 31] {
                        let mut expected = weights.clone();
                        let mut actual = weights.clone();
                        update_scalar(&mut expected, &inputs, error, shift, 1 << 22);
                        // SAFETY: AVX2 detected; shift/max and lengths satisfy contract.
                        unsafe { update_avx2(&mut actual, &inputs, error, shift, 1 << 22) };
                        assert_eq!(actual, expected, "count{count} error{error} shift{shift}");
                    }
                }
            }
        }
    }
}
