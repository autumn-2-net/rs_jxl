// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Checked nonzero positions for sequential JPEG reconstruction, in scan order.

use super::data::{BLOCK_SIZE, NATURAL_ORDER};

pub(super) fn nonzero_ac(coefficients: &[i16; BLOCK_SIZE]) -> Option<u64> {
    #[cfg(all(target_arch = "x86_64", feature = "decoder-avx512"))]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        // SAFETY: both required features are available; input is a complete block.
        return unsafe { nonzero_ac_avx512(coefficients) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is available; every load covers a complete local array chunk.
        return unsafe { nonzero_ac_avx2(coefficients) };
    }
    nonzero_ac_scalar(coefficients)
}

fn nonzero_ac_scalar(coefficients: &[i16; BLOCK_SIZE]) -> Option<u64> {
    let mut mask = 0;
    for k in 1..BLOCK_SIZE {
        let value = coefficients[NATURAL_ORDER[k]];
        if !(-2047..=2047).contains(&value) {
            return None;
        }
        mask |= u64::from(value != 0) << k;
    }
    Some(mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nonzero_ac_avx2(coefficients: &[i16; BLOCK_SIZE]) -> Option<u64> {
    use std::arch::x86_64::*;
    let mut ordered: [i16; BLOCK_SIZE] = std::array::from_fn(|k| coefficients[NATURAL_ORDER[k]]);
    ordered[0] = 0; // DC has its separate difference/range validation.
    // SAFETY: four nonoverlapping complete 16-element chunks of initialized i16s.
    let (a, b, c, d) = unsafe {
        (
            _mm256_loadu_si256(ordered.as_ptr().cast()),
            _mm256_loadu_si256(ordered.as_ptr().add(16).cast()),
            _mm256_loadu_si256(ordered.as_ptr().add(32).cast()),
            _mm256_loadu_si256(ordered.as_ptr().add(48).cast()),
        )
    };
    let magnitude = _mm256_or_si256(
        _mm256_or_si256(_mm256_abs_epi16(a), _mm256_abs_epi16(b)),
        _mm256_or_si256(_mm256_abs_epi16(c), _mm256_abs_epi16(d)),
    );
    // Also rejects abs(i16::MIN), whose high bit stays set.
    if _mm256_testz_si256(magnitude, _mm256_set1_epi16(-2048)) == 0 {
        return None;
    }
    let zero = _mm256_setzero_si256();
    let ab = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(
        _mm256_cmpeq_epi16(a, zero),
        _mm256_cmpeq_epi16(b, zero),
    ));
    let cd = _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi16(
        _mm256_cmpeq_epi16(c, zero),
        _mm256_cmpeq_epi16(d, zero),
    ));
    let low = !_mm256_movemask_epi8(ab) as u32;
    let high = !_mm256_movemask_epi8(cd) as u32;
    Some(u64::from(low) | (u64::from(high) << 32))
}

#[cfg(all(target_arch = "x86_64", feature = "decoder-avx512"))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nonzero_ac_avx512(coefficients: &[i16; BLOCK_SIZE]) -> Option<u64> {
    use std::arch::x86_64::*;
    const ORDER: [i16; BLOCK_SIZE] = {
        let mut order = [0; BLOCK_SIZE];
        let mut k = 0;
        while k < BLOCK_SIZE {
            order[k] = NATURAL_ORDER[k] as i16;
            k += 1;
        }
        order
    };
    // SAFETY: each input/index array has 64 initialized i16 elements. The two
    // unaligned loads cover disjoint 32-element halves without reading padding.
    let (a, b, low_order, high_order) = unsafe {
        (
            _mm512_loadu_si512(coefficients.as_ptr().cast()),
            _mm512_loadu_si512(coefficients.as_ptr().add(32).cast()),
            _mm512_loadu_si512(ORDER.as_ptr().cast()),
            _mm512_loadu_si512(ORDER.as_ptr().add(32).cast()),
        )
    };
    // Exclude DC before validation and permutation, as in the AVX2 kernel.
    let a = _mm512_maskz_mov_epi16(0xffff_fffe, a);
    let magnitude = _mm512_or_si512(_mm512_abs_epi16(a), _mm512_abs_epi16(b));
    if _mm512_test_epi16_mask(magnitude, _mm512_set1_epi16(-2048)) != 0 {
        return None;
    }
    let low = _mm512_permutex2var_epi16(a, low_order, b);
    let high = _mm512_permutex2var_epi16(a, high_order, b);
    let zero = _mm512_setzero_si512();
    Some(
        u64::from(_mm512_cmpneq_epi16_mask(low, zero))
            | (u64::from(_mm512_cmpneq_epi16_mask(high, zero)) << 32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_kernels(block: &[i16; BLOCK_SIZE], expected: Option<u64>) {
        assert_eq!(nonzero_ac_scalar(block), expected);
        assert_eq!(nonzero_ac(block), expected);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: feature checked here; block is a complete input array.
            assert_eq!(unsafe { nonzero_ac_avx2(block) }, expected);
        }
        #[cfg(all(target_arch = "x86_64", feature = "decoder-avx512"))]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            // SAFETY: both features checked here; block is a complete input array.
            assert_eq!(unsafe { nonzero_ac_avx512(block) }, expected);
        }
    }

    #[test]
    fn all_signed_values_positions_and_dc_are_checked() {
        let mut block = [0; BLOCK_SIZE];
        for position in [1, 8, 16, 31, 32, 48, 63] {
            for value in i16::MIN..=i16::MAX {
                block[0] = value;
                block[NATURAL_ORDER[position]] = value;
                let expected = if (-2047..=2047).contains(&value) {
                    Some(u64::from(value != 0) << position)
                } else {
                    None
                };
                check_kernels(&block, expected);
            }
            block[NATURAL_ORDER[position]] = 0;
        }
        for value in [i16::MIN, -2048, 2048, i16::MAX] {
            block[0] = value;
            check_kernels(&block, Some(0));
        }
    }

    #[test]
    fn mixed_masks_and_invalid_lanes_match_scalar() {
        let mut state = 0xa8c1_6df3u32;
        for density in 0..=64 {
            for _ in 0..64 {
                let mut block = [0; BLOCK_SIZE];
                for value in &mut block {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    if state % 64 < density {
                        *value = ((state >> 8) % 4095) as i16 - 2047;
                    }
                }
                check_kernels(&block, nonzero_ac_scalar(&block));
                for position in 1..64 {
                    let old = block[position];
                    block[position] = if position & 1 == 0 { i16::MIN } else { 2048 };
                    check_kernels(&block, None);
                    block[position] = old;
                }
            }
        }
    }

    #[test]
    fn unaligned_blocks_preserve_input_and_scan_order() {
        #[repr(align(64))]
        struct Storage([i16; 128]);
        for offset in 0..32 {
            let mut storage = Storage([1234; 128]);
            for position in 0..64 {
                let values = &mut storage.0[offset..offset + 64];
                values.fill(0);
                values[NATURAL_ORDER[position]] = -2047;
                values[0] = i16::MIN;
                let saved = storage.0;
                let block = (&storage.0[offset..offset + 64]).try_into().unwrap();
                check_kernels(block, Some(if position == 0 { 0 } else { 1 << position }));
                assert_eq!(storage.0, saved);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "decoder-avx512"))]
    #[test]
    #[ignore = "manual Release-only kernel benchmark; run without concurrent jobs"]
    fn benchmark_nonzero_ac_widths() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};
        if !(std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw"))
        {
            eprintln!("AVX2/AVX512 unavailable");
            return;
        }
        let blocks: Vec<[i16; 64]> = (0..1024)
            .map(|i| {
                std::array::from_fn(|k| {
                    if (k * 13 + i) % 64 < i % 64 {
                        ((i * 97 + k * 7) % 4095) as i16 - 2047
                    } else {
                        0
                    }
                })
            })
            .collect();
        #[target_feature(enable = "avx2")]
        unsafe fn avx2(blocks: &[[i16; 64]]) -> Duration {
            let start = Instant::now();
            for _ in 0..2048 {
                for block in black_box(blocks) {
                    // SAFETY: this gateway requires AVX2; each block is complete.
                    black_box(unsafe { nonzero_ac_avx2(black_box(block)) });
                }
            }
            start.elapsed()
        }
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn avx512(blocks: &[[i16; 64]]) -> Duration {
            let start = Instant::now();
            for _ in 0..2048 {
                for block in black_box(blocks) {
                    // SAFETY: this gateway requires both AVX512 features.
                    black_box(unsafe { nonzero_ac_avx512(black_box(block)) });
                }
            }
            start.elapsed()
        }
        for round in 0..9 {
            for slot in 0..2 {
                let wide = (round + slot) % 2 != 0;
                // SAFETY: all required features were checked before timing.
                let elapsed = unsafe { if wide { avx512(&blocks) } else { avx2(&blocks) } };
                if round != 0 {
                    eprintln!(
                        "round={round} width={} ns/block={:.4}",
                        if wide { 512 } else { 256 },
                        elapsed.as_secs_f64() * 1e9 / (2048 * blocks.len()) as f64,
                    );
                }
            }
        }
    }
}
