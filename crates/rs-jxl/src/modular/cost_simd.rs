// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the complete upstream notice is retained
// in docs/PORTING_KERNELS.md.

//! Fused gradient-cost rows from libjxl b866b3d, enc_modular_simd.cc::EstimateCost.
//! This is NOT a narrow replacement for generic Modular prediction. Cost uses
//! wrapping i32 residuals before PackSigned, including for full-range samples.
//! The caller retains histogram updates, entropy profiles and accumulation order.
//!
//! Source SIMD caps unsigned neighbor spread at 500 and uses ContextMap; the
//! scalar profile reverses context numbering. HybridUint(4,2,0) classification
//! uses the source float-exponent/mantissa sequence, including its large-value
//! shift. All converted integers are <= 2^22-1, so conversion is exact.
//! Unlike source padded planes, these slices need no readable row padding:
//! column zero and incomplete tail vectors use the scalar source counterpart.

use super::{Backend, GRADIENT_CUTOFFS};
use crate::entropy::HybridUintConfig;
use crate::{Error, Result};

const MAX_DIFF: u32 = 500; // enc_modular_simd.h::kLastThreshold - 1
const CONTEXT_MAP: [u8; MAX_DIFF as usize + 1] = {
    let mut map = [0; MAX_DIFF as usize + 1];
    let mut difference = 0;
    while difference < map.len() {
        let mut context = 0;
        while context < GRADIENT_CUTOFFS.len() && GRADIENT_CUTOFFS[context] <= difference as u32 {
            context += 1;
        }
        map[difference] = context as u8;
        difference += 1;
    }
    map
};

pub(super) fn context(capped_difference: u32, mode: Backend) -> usize {
    let ascending = usize::from(CONTEXT_MAP[capped_difference as usize]);
    if mode == Backend::Scalar {
        16 - ascending
    } else {
        ascending
    }
}

fn scalar_pixel(value: i32, left: i32, top: i32, top_left: i32) -> (u32, u32, usize) {
    let low = left.min(top);
    let high = left.max(top);
    let difference = high.max(top_left).wrapping_sub(low.min(top_left)) as u32;
    let gradient = left.wrapping_add(top).wrapping_sub(top_left);
    let prediction = if top_left > high {
        low
    } else if top_left < low {
        high
    } else {
        gradient
    };
    let residual = value.wrapping_sub(prediction);
    let packed = ((residual as u32) << 1) ^ ((residual >> 31) as u32);
    let encoded = HybridUintConfig::default().encode(packed);
    (
        difference.min(MAX_DIFF),
        encoded.token,
        encoded.nbits as usize,
    )
}

fn scalar_at(current: &[i32], top: Option<&[i32]>, x: usize) -> (u32, u32, usize) {
    let left = if x != 0 {
        current[x - 1]
    } else {
        top.map_or(0, |row| row[0])
    };
    let north = top.map_or(left, |row| row[x]);
    let north_west = if x != 0 {
        top.map_or(left, |row| row[x - 1])
    } else {
        left
    };
    scalar_pixel(current[x], left, north, north_west)
}

/// Write capped unsigned neighbor spreads and HybridUint tokens, and return
/// the exact extra-bit sum. Output suffixes are untouched. Validation errors
/// precede writes; the source per-channel i32 population limit bounds sums on
/// x86_64, while the scalar fallback also checks usize arithmetic.
pub(super) fn row(
    current: &[i32],
    top: Option<&[i32]>,
    mode: Backend,
    differences: &mut [u32],
    tokens: &mut [u32],
) -> Result<usize> {
    let width = current.len();
    if width > i32::MAX as usize {
        return Err(Error::Limit("cost row population exceeds i32::MAX"));
    }
    if top.is_some_and(|row| row.len() != width)
        || differences.len() < width
        || tokens.len() < width
    {
        return Err(Error::InvalidInput(
            "invalid gradient cost row buffers".into(),
        ));
    }
    if width == 0 {
        return Ok(0);
    }
    let (difference, token, mut extra) = scalar_at(current, top, 0);
    differences[0] = difference;
    tokens[0] = token;
    #[cfg(target_arch = "x86_64")]
    let (next, bits) = {
        // Backend is private to cost.rs and originates in runtime detection;
        // tests likewise select only supported modes. Fma needs only AVX2 here.
        // SAFETY: lengths above are checked. Backends load complete vectors
        // starting at x >= 1; no load crosses a visible row or output boundary.
        unsafe {
            match mode {
                Backend::Sse2 => x86::sse2::row(current, top, differences, tokens),
                Backend::Avx2 | Backend::Fma => x86::avx2::row(current, top, differences, tokens),
                Backend::Scalar => (1, 0),
            }
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (next, bits) = {
        let _ = mode;
        (1, 0)
    };
    extra = extra
        .checked_add(bits)
        .ok_or(Error::Limit("cost extra-bit total"))?;
    for x in next..width {
        let (difference, token, bits) = scalar_at(current, top, x);
        differences[x] = difference;
        tokens[x] = token;
        extra = extra
            .checked_add(bits)
            .ok_or(Error::Limit("cost extra-bit total"))?;
    }
    Ok(extra)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    // Only the native instruction spelling and lane count differ. Integer
    // masks/arithmetic follow source Highway's i32/u32 lanes, not i64 predicts.
    macro_rules! kernel {
        ($feature:literal, $lanes:literal, $vector:ty) => {
            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn choose(mask: $vector, yes: $vector, no: $vector) -> $vector {
                or(and(mask, yes), andnot(mask, no))
            }

            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn unsigned_gt(a: $vector, b: $vector) -> $vector {
                let bias = splat(i32::MIN);
                gt(xor(a, bias), xor(b, bias))
            }

            #[inline]
            #[target_feature(enable = $feature)]
            unsafe fn encode(residual: $vector) -> ($vector, $vector) {
                // SAFETY: all helpers have the same target feature.
                unsafe {
                    let packed = xor(shl::<1>(residual), sar::<31>(residual));
                    let large = unsigned_gt(packed, splat((1 << 22) - 1));
                    let fixed = choose(large, shr::<10>(packed), packed);
                    let float_bits = float_bits(fixed);
                    let raw_bits = sub(shr::<23>(float_bits), splat(129));
                    let bits = add(raw_bits, and(large, splat(10)));
                    let token = add(
                        add(splat(8), shl::<2>(bits)),
                        and(shr::<21>(float_bits), splat(3)),
                    );
                    let nonliteral = unsigned_gt(packed, splat(15));
                    (choose(nonliteral, token, packed), and(nonliteral, bits))
                }
            }

            #[target_feature(enable = $feature)]
            unsafe fn run<const HAS_TOP: bool>(
                current: &[i32],
                top: &[i32],
                differences: &mut [u32],
                tokens: &mut [u32],
            ) -> (usize, usize) {
                let mut x = 1;
                let mut extra = 0usize;
                // SAFETY: row() validated all lengths and CPU features. Each
                // complete block reads current[x-1..x+LANES] and the same top
                // positions when HAS_TOP; stores cover x..x+LANES only.
                unsafe {
                    while current.len() - x >= $lanes {
                        // Bound each u32 lane sum, then widen for the parent's
                        // existing explicit u32-total overflow check. This is
                        // exact integer regrouping, not a float-cost reduction.
                        let blocks = ((current.len() - x) / $lanes).min(4096);
                        let end = x + blocks * $lanes;
                        let mut bits_sum = splat(0);
                        while x < end {
                            let left = load(current.as_ptr().add(x - 1));
                            let central = load(current.as_ptr().add(x));
                            let (prediction, difference) = if HAS_TOP {
                                let north = load(top.as_ptr().add(x));
                                let north_west = load(top.as_ptr().add(x - 1));
                                let left_gt_top = gt(left, north);
                                let low = choose(left_gt_top, north, left);
                                let high = choose(left_gt_top, left, north);
                                let maximum = choose(gt(north_west, high), north_west, high);
                                let minimum = choose(gt(low, north_west), north_west, low);
                                let difference = sub(maximum, minimum);
                                let cap = splat(super::super::MAX_DIFF as i32);
                                let difference =
                                    choose(unsigned_gt(difference, cap), cap, difference);
                                let gradient = sub(add(north, left), north_west);
                                let prediction = choose(
                                    gt(north_west, high),
                                    low,
                                    choose(gt(low, north_west), high, gradient),
                                );
                                (prediction, difference)
                            } else {
                                (left, splat(0))
                            };
                            let (token, bits) = encode(sub(central, prediction));
                            store(differences.as_mut_ptr().add(x), difference);
                            store(tokens.as_mut_ptr().add(x), token);
                            bits_sum = add(bits_sum, bits);
                            x += $lanes;
                        }
                        let mut lane_bits = [0u32; $lanes];
                        store(lane_bits.as_mut_ptr(), bits_sum);
                        extra += lane_bits.iter().map(|&n| n as usize).sum::<usize>();
                    }
                }
                (x, extra)
            }

            #[target_feature(enable = $feature)]
            pub(in super::super) unsafe fn row(
                current: &[i32],
                top: Option<&[i32]>,
                differences: &mut [u32],
                tokens: &mut [u32],
            ) -> (usize, usize) {
                // SAFETY: propagate the checked row/CPU contract; HAS_TOP is
                // selected once per row, keeping the first-row loop specialized.
                unsafe {
                    if let Some(top) = top {
                        run::<true>(current, top, differences, tokens)
                    } else {
                        run::<false>(current, &[], differences, tokens)
                    }
                }
            }
        };
    }

    pub(super) mod sse2 {
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm_add_epi32 as add, _mm_and_si128 as and, _mm_andnot_si128 as andnot,
            _mm_cmpgt_epi32 as gt, _mm_or_si128 as or, _mm_set1_epi32 as splat,
            _mm_slli_epi32 as shl, _mm_srai_epi32 as sar, _mm_srli_epi32 as shr,
            _mm_sub_epi32 as sub, _mm_xor_si128 as xor,
        };

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn load(ptr: *const i32) -> __m128i {
            // SAFETY: the shared kernel supplies four readable samples.
            unsafe { _mm_loadu_si128(ptr.cast()) }
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn store(ptr: *mut u32, value: __m128i) {
            // SAFETY: the shared kernel supplies four writable elements.
            unsafe { _mm_storeu_si128(ptr.cast(), value) };
        }

        #[inline]
        #[target_feature(enable = "sse2")]
        unsafe fn float_bits(value: __m128i) -> __m128i {
            _mm_castps_si128(_mm_cvtepi32_ps(value))
        }

        kernel!("sse2", 4, __m128i);
    }

    pub(super) mod avx2 {
        use std::arch::x86_64::*;
        use std::arch::x86_64::{
            _mm256_add_epi32 as add, _mm256_and_si256 as and, _mm256_andnot_si256 as andnot,
            _mm256_cmpgt_epi32 as gt, _mm256_or_si256 as or, _mm256_set1_epi32 as splat,
            _mm256_slli_epi32 as shl, _mm256_srai_epi32 as sar, _mm256_srli_epi32 as shr,
            _mm256_sub_epi32 as sub, _mm256_xor_si256 as xor,
        };

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn load(ptr: *const i32) -> __m256i {
            // SAFETY: the shared kernel supplies eight readable samples.
            unsafe { _mm256_loadu_si256(ptr.cast()) }
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn store(ptr: *mut u32, value: __m256i) {
            // SAFETY: the shared kernel supplies eight writable elements.
            unsafe { _mm256_storeu_si256(ptr.cast(), value) };
        }

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn float_bits(value: __m256i) -> __m256i {
            _mm256_castps_si256(_mm256_cvtepi32_ps(value))
        }

        kernel!("avx2", 8, __m256i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported() -> Vec<Backend> {
        super::super::tests::supported()
    }

    // Wide clamp and integer HybridUint formulas are independent of both the
    // wrapping gradient and the SIMD float-exponent/mantissa implementation.
    fn reference(current: &[i32], top: Option<&[i32]>, x: usize) -> (u32, u32, usize) {
        let left = if x > 0 {
            current[x - 1]
        } else {
            top.map_or(0, |p| p[0])
        };
        let north = top.map_or(left, |p| p[x]);
        let nw = if x > 0 {
            top.map_or(left, |p| p[x - 1])
        } else {
            left
        };
        let spread = i64::from(left.max(north).max(nw)) - i64::from(left.min(north).min(nw));
        let prediction = (i64::from(left) + i64::from(north) - i64::from(nw))
            .clamp(i64::from(left.min(north)), i64::from(left.max(north)));
        let residual = (i64::from(current[x]) - prediction) as i32;
        let packed = if residual < 0 {
            (-2 * i64::from(residual) - 1) as u32
        } else {
            residual as u32 * 2
        };
        let (token, bits) = if packed < 16 {
            (packed, 0)
        } else {
            let bits = 31 - packed.leading_zeros() - 2;
            (8 + 4 * bits + ((packed >> bits) & 3), bits as usize)
        };
        ((spread as u32).min(MAX_DIFF), token, bits)
    }

    fn check(current: &[i32], top: Option<&[i32]>) {
        let expected: Vec<_> = (0..current.len())
            .map(|x| reference(current, top, x))
            .collect();
        let expected_bits: usize = expected.iter().map(|&(_, _, bits)| bits).sum();
        for mode in supported() {
            // Output subslices exercise unaligned stores and guard both ends.
            let mut differences = vec![u32::MAX; current.len() + 4];
            let mut tokens = vec![u32::MAX; current.len() + 4];
            let bits = row(current, top, mode, &mut differences[1..], &mut tokens[1..]).unwrap();
            assert_eq!(bits, expected_bits, "{mode:?}, width {}", current.len());
            for (x, &(spread, token, _)) in expected.iter().enumerate() {
                assert_eq!(differences[x + 1], spread, "{mode:?}, x {x}");
                assert_eq!(tokens[x + 1], token, "{mode:?}, x {x}");
            }
            assert_eq!(differences[0], u32::MAX);
            assert_eq!(tokens[0], u32::MAX);
            assert!(
                differences[current.len() + 1..]
                    .iter()
                    .all(|&n| n == u32::MAX)
            );
            assert!(tokens[current.len() + 1..].iter().all(|&n| n == u32::MAX));
        }
    }

    #[test]
    fn source_context_map_and_scalar_numbering() {
        for spread in 0..=MAX_DIFF {
            let ascending = GRADIENT_CUTOFFS.iter().filter(|&&n| n <= spread).count();
            assert_eq!(context(spread, Backend::Sse2), ascending);
            assert_eq!(context(spread, Backend::Scalar), 16 - ascending);
        }
        assert_eq!(scalar_pixel(0, i32::MIN, i32::MAX, 0).0, 500);
        assert_eq!(
            scalar_pixel(i32::MIN, i32::MAX, i32::MAX, i32::MAX),
            (0, 2, 0)
        );
    }

    #[test]
    fn all_extreme_neighbor_combinations_in_native_lanes() {
        let values = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX];
        for value in values {
            for left in values {
                for north in values {
                    for nw in values {
                        let mut current = [value; 17];
                        let mut top = [north; 17];
                        current[4] = left;
                        top[4] = nw;
                        check(&current, Some(&top));
                    }
                }
            }
        }
    }

    #[test]
    fn layouts_first_rows_unaligned_inputs_and_all_vector_tails() {
        let mut random = 0xb866_b3d0u32;
        for width in 0..=65 {
            for padding in [0, 1, 7] {
                let stride = width + padding;
                let offset = padding + 1;
                // The last row ends exactly at the allocation boundary.
                let mut storage = vec![i32::MIN; offset + stride * 3 + width];
                for y in 0..4 {
                    for x in 0..width {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        storage[offset + y * stride + x] = random as i32;
                    }
                }
                for y in 0..4 {
                    let start = offset + y * stride;
                    let top = (y != 0).then(|| &storage[start - stride..start - stride + width]);
                    check(&storage[start..start + width], top);
                }
            }
        }
    }

    #[test]
    fn hybrid_token_boundaries_and_large_float_conversion_fix() {
        let mut packed: Vec<u32> = (0..=20).collect();
        for exponent in 4..=31 {
            for mantissa in 4..=8u64 {
                let boundary = mantissa << (exponent - 2);
                for candidate in boundary.saturating_sub(3)..=boundary + 3 {
                    if let Ok(candidate) = u32::try_from(candidate) {
                        packed.push(candidate);
                    }
                }
            }
        }
        packed.extend([
            (1 << 22) - 2,
            (1 << 22) - 1,
            1 << 22,
            (1 << 22) + 1,
            u32::MAX,
        ]);
        let mut previous = 0i32;
        let current: Vec<_> = packed
            .iter()
            .map(|&value| {
                let residual = ((value >> 1) as i32) ^ -((value & 1) as i32);
                previous = previous.wrapping_add(residual);
                previous
            })
            .collect();
        check(&current, None);
    }

    #[test]
    fn extra_bits_survive_chunk_reduction_and_tail() {
        for width in [4096 * 4 + 3, 4096 * 8 + 3] {
            let current: Vec<_> = (0..width)
                .map(|x| if x % 2 == 0 { i32::MIN } else { 0 })
                .collect();
            check(&current, None);
            for mode in supported() {
                let bits = row(
                    &current,
                    None,
                    mode,
                    &mut vec![0; width],
                    &mut vec![0; width],
                )
                .unwrap();
                assert_eq!(bits, 29 * width);
            }
        }
    }

    #[test]
    fn invalid_row_shapes_do_not_write_outputs() {
        for (top, differences_len, tokens_len) in
            [(None, 1, 2), (None, 2, 1), (Some(&[0][..]), 2, 2)]
        {
            let mut differences = [17; 2];
            let mut tokens = [19; 2];
            assert!(
                row(
                    &[1, 2],
                    top,
                    Backend::Scalar,
                    &mut differences[..differences_len],
                    &mut tokens[..tokens_len],
                )
                .is_err()
            );
            assert_eq!(differences, [17; 2]);
            assert_eq!(tokens, [19; 2]);
        }
    }
}
