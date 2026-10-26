// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Bit-plane classification for the JPEG refinement writer. Masks use zigzag
//! indices; no entropy state or coefficient values are changed here.

use super::coefficients::ScanBand;
use super::data::{BLOCK_SIZE, NATURAL_ORDER};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Masks {
    pub nonzero: u64,
    pub newly_nonzero: u64,
    pub correction: u64,
    pub positive: u64,
}

pub(super) fn classify(coefficients: &[i16; BLOCK_SIZE], band: ScanBand) -> Option<Masks> {
    debug_assert!(band.start <= band.end && band.end < 64 && band.low_bit <= 10);
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime dispatch checks AVX2; the kernel only loads complete
        // chunks of its local 64-element array.
        return unsafe { classify_avx2(coefficients, band) };
    }
    classify_scalar(coefficients, band)
}

fn classify_scalar(coefficients: &[i16; BLOCK_SIZE], band: ScanBand) -> Option<Masks> {
    let mut masks = Masks::default();
    for k in usize::from(band.start.max(1))..=usize::from(band.end) {
        let value = i32::from(coefficients[NATURAL_ORDER[k]]);
        let magnitude = value.abs();
        if magnitude >= 2048 {
            return None;
        }
        let shifted = magnitude >> band.low_bit;
        masks.nonzero |= u64::from(shifted != 0) << k;
        masks.newly_nonzero |= u64::from(shifted == 1) << k;
        masks.correction |= ((shifted & 1) as u64) << k;
        masks.positive |= u64::from(value > 0) << k;
    }
    Some(masks)
}

/// Pack selected correction bits in scan order, first coefficient at the MSB.
pub(super) fn correction_bits(correction: u64, selected: u64) -> (u64, u8) {
    let count = selected.count_ones() as u8;
    if count == 0 {
        return (0, 0);
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("bmi2") {
        // SAFETY: BMI2 was checked; this intrinsic only operates on registers.
        let bits = unsafe { extract_bits(correction, selected) };
        return (bits.reverse_bits() >> (64 - count), count);
    }
    let mut pending = selected;
    let mut bits = 0;
    while pending != 0 {
        let k = pending.trailing_zeros();
        bits = (bits << 1) | ((correction >> k) & 1);
        pending &= pending - 1;
    }
    (bits, count)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn extract_bits(value: u64, mask: u64) -> u64 {
    std::arch::x86_64::_pext_u64(value, mask)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
fn mask_pair(a: std::arch::x86_64::__m256i, b: std::arch::x86_64::__m256i) -> u32 {
    use std::arch::x86_64::*;
    let packed = _mm256_packs_epi16(a, b);
    let ordered = _mm256_permute4x64_epi64::<0xd8>(packed);
    _mm256_movemask_epi8(ordered) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn classify_avx2(coefficients: &[i16; BLOCK_SIZE], band: ScanBand) -> Option<Masks> {
    use std::arch::x86_64::*;
    // Constant indices allow LLVM to combine the gather into fixed shuffles.
    // Unlike a 32-bit gather on i16 input, this never reads past coefficient 63.
    let ordered: [i16; BLOCK_SIZE] = std::array::from_fn(|k| coefficients[NATURAL_ORDER[k]]);
    let zero = _mm256_setzero_si256();
    let one = _mm256_set1_epi16(1);
    let high = _mm256_set1_epi16(-2048);
    let shift = _mm_cvtsi32_si128(i32::from(band.low_bit));
    let mut masks = Masks::default();
    let mut valid = 0u64;
    for offset in (0..BLOCK_SIZE).step_by(16) {
        // SAFETY: offsets 0,16,32,48 each address 16 initialized i16 elements.
        let value = unsafe { _mm256_loadu_si256(ordered.as_ptr().add(offset).cast()) };
        let magnitude = _mm256_abs_epi16(value);
        let shifted = _mm256_srl_epi16(magnitude, shift);
        let kinds = mask_pair(
            _mm256_cmpgt_epi16(shifted, zero),
            _mm256_cmpeq_epi16(shifted, one),
        );
        let bits = mask_pair(
            _mm256_cmpeq_epi16(_mm256_and_si256(shifted, one), one),
            _mm256_cmpgt_epi16(value, zero),
        );
        // The high-bit test also rejects abs(i16::MIN), which remains 0x8000.
        let in_range = mask_pair(
            _mm256_cmpeq_epi16(_mm256_and_si256(magnitude, high), zero),
            zero,
        );
        masks.nonzero |= u64::from(kinds & 0xffff) << offset;
        masks.newly_nonzero |= u64::from(kinds >> 16) << offset;
        masks.correction |= u64::from(bits & 0xffff) << offset;
        masks.positive |= u64::from(bits >> 16) << offset;
        valid |= u64::from(in_range & 0xffff) << offset;
    }
    let selected = (u64::MAX << band.start.max(1)) & (u64::MAX >> (63 - band.end));
    if valid & selected != selected {
        return None;
    }
    masks.nonzero &= selected;
    masks.newly_nonzero &= selected;
    masks.correction &= selected;
    masks.positive &= selected;
    Some(masks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_matches_scalar_for_all_planes_bands_and_invalid_values() {
        let mut seed = 0x391d_87fau32;
        for low_bit in 0..=10 {
            for start in 1..64 {
                for end in start..64 {
                    let band = ScanBand {
                        start,
                        end,
                        low_bit,
                    };
                    let mut coefficients = [i16::MIN; 64];
                    for &index in &NATURAL_ORDER[start as usize..=end as usize] {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        coefficients[index] = (seed % 4095) as i16 - 2047;
                    }
                    assert_eq!(
                        classify(&coefficients, band),
                        classify_scalar(&coefficients, band)
                    );
                    for invalid in [i16::MIN, -2048, 2048, i16::MAX] {
                        coefficients[NATURAL_ORDER[end as usize]] = invalid;
                        assert_eq!(classify(&coefficients, band), None);
                    }
                }
            }
        }
    }

    #[test]
    fn packed_corrections_match_scan_order() {
        let mut seed = 0xa13e_874c_019d_6fb2u64;
        for count in 0..64 {
            for _ in 0..100 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let selected = seed & (u64::MAX >> (63 - count));
                let correction = seed.rotate_left(23);
                let mut expected = 0;
                for k in 0..64 {
                    if selected & (1 << k) != 0 {
                        expected = (expected << 1) | ((correction >> k) & 1);
                    }
                }
                assert_eq!(
                    correction_bits(correction, selected),
                    (expected, selected.count_ones() as u8)
                );
            }
        }
    }
}
