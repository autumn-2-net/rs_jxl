// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.
//! Checked JPEG AC recovery, using native Rust x86 intrinsics. Failures leave
//! output untouched so the scalar caller can retain ordered error/partial state.

#[inline]
/// Recover natural-order ACs if SIMD and the checked fast arithmetic apply.
/// Returns false without modifying output otherwise. DC is never changed.
/// `output` must contain exactly 64 coefficients.
pub fn capture(raw: &[i32; 64], cfl: Option<(&[i32; 64], &[i32; 64])>, output: &mut [i16]) -> bool {
    assert_eq!(output.len(), 64);
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        // SAFETY: runtime detection covers the target features. Fixed array
        // sizes and the assertion cover all accesses; stores follow validation.
        return unsafe {
            match cfl {
                Some((luma, scales)) => x86::capture_avx512::<true>(raw, luma, scales, output),
                None => x86::capture_avx512::<false>(raw, raw, raw, output),
            }
        };
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime dispatch checks AVX2; fixed-size inputs and the output
        // assertion cover every unaligned load/store. The kernel guards all
        // arithmetic ranges before committing and preserves the original DC.
        return unsafe {
            match cfl {
                Some((luma, scales)) => x86::capture::<true>(raw, luma, scales, output),
                None => x86::capture::<false>(raw, raw, raw, output),
            }
        };
    }
    let _ = (raw, cfl);
    false
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod x86 {
    use std::arch::x86_64::*;

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn transpose_avx512(raw: &[i32; 64]) -> [__m512i; 4] {
        // Each register holds two rows. Unpack within 128-bit lanes, then
        // interleave even/odd source rows with one two-source permute per pair.
        // SAFETY: four unaligned loads cover exactly the 64 input coefficients.
        let r = std::array::from_fn::<_, 4, _>(|i| unsafe {
            _mm512_loadu_si512(raw.as_ptr().add(i * 16).cast())
        });
        let t = [
            _mm512_unpacklo_epi32(r[0], r[1]),
            _mm512_unpackhi_epi32(r[0], r[1]),
            _mm512_unpacklo_epi32(r[2], r[3]),
            _mm512_unpackhi_epi32(r[2], r[3]),
        ];
        let s = [
            _mm512_unpacklo_epi64(t[0], t[2]),
            _mm512_unpackhi_epi64(t[0], t[2]),
            _mm512_unpacklo_epi64(t[1], t[3]),
            _mm512_unpackhi_epi64(t[1], t[3]),
        ];
        let low = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 16, 24, 17, 25, 18, 26, 19, 27);
        let high = _mm512_setr_epi32(4, 12, 5, 13, 6, 14, 7, 15, 20, 28, 21, 29, 22, 30, 23, 31);
        [
            _mm512_permutex2var_epi32(s[0], low, s[1]),
            _mm512_permutex2var_epi32(s[2], low, s[3]),
            _mm512_permutex2var_epi32(s[0], high, s[1]),
            _mm512_permutex2var_epi32(s[2], high, s[3]),
        ]
    }

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f,avx512bw")]
    fn outside_avx512(value: __m512i, bound: i32) -> __mmask16 {
        // Unsigned wrapping translation maps exactly [-bound,bound] to
        // [0,2*bound], including rejection of i32::MIN/MAX without abs overflow.
        _mm512_cmpgt_epu32_mask(
            _mm512_add_epi32(value, _mm512_set1_epi32(bound)),
            _mm512_set1_epi32(2 * bound),
        )
    }

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn capture_avx512<const CFL: bool>(
        raw: &[i32; 64],
        luma: &[i32; 64],
        scales: &[i32; 64],
        output: &mut [i16],
    ) -> bool {
        // SAFETY: caller guarantees target features and exactly 64 output
        // elements. Loads/stores cover fixed in-bounds blocks, never input-
        // selected addresses. No store occurs before every range check passes.
        unsafe {
            let mut values = transpose_avx512(raw);
            values[0] = _mm512_maskz_mov_epi32(0xfffe, values[0]);
            let mut invalid = 0;
            if CFL {
                let mut y = transpose_avx512(luma);
                y[0] = _mm512_maskz_mov_epi32(0xfffe, y[0]);
                for i in 0..4 {
                    let mut scale = _mm512_loadu_si512(scales.as_ptr().add(i * 16).cast());
                    if i == 0 {
                        scale = _mm512_maskz_mov_epi32(0xfffe, scale);
                    }
                    invalid |= outside_avx512(values[i], 65535);
                    invalid |= outside_avx512(y[i], 4095);
                    invalid |= outside_avx512(scale, 524288);
                    // Same bounded integer arithmetic as AVX2. Invalid inputs
                    // may wrap, but their masks prevent any output commitment.
                    let product = _mm512_mullo_epi32(y[i], scale);
                    let prediction =
                        _mm512_srai_epi32::<11>(_mm512_add_epi32(product, _mm512_set1_epi32(1024)));
                    values[i] = _mm512_add_epi32(values[i], prediction);
                }
            }
            for value in values {
                invalid |= outside_avx512(value, 4095);
            }
            if invalid != 0 {
                return false;
            }
            let dc = output[0];
            for (i, value) in values.into_iter().enumerate() {
                let packed = _mm512_cvtepi32_epi16(value);
                _mm256_storeu_si256(output.as_mut_ptr().add(i * 16).cast(), packed);
            }
            output[0] = dc;
            true
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn transpose(raw: &[i32; 64]) -> [__m256i; 8] {
        // SAFETY: eight loads of eight i32 values cover exactly the input array.
        let r = std::array::from_fn::<_, 8, _>(|i| unsafe {
            _mm256_loadu_si256(raw.as_ptr().add(i * 8).cast())
        });
        let t = [
            _mm256_unpacklo_epi32(r[0], r[1]),
            _mm256_unpackhi_epi32(r[0], r[1]),
            _mm256_unpacklo_epi32(r[2], r[3]),
            _mm256_unpackhi_epi32(r[2], r[3]),
            _mm256_unpacklo_epi32(r[4], r[5]),
            _mm256_unpackhi_epi32(r[4], r[5]),
            _mm256_unpacklo_epi32(r[6], r[7]),
            _mm256_unpackhi_epi32(r[6], r[7]),
        ];
        let s = [
            _mm256_unpacklo_epi64(t[0], t[2]),
            _mm256_unpackhi_epi64(t[0], t[2]),
            _mm256_unpacklo_epi64(t[1], t[3]),
            _mm256_unpackhi_epi64(t[1], t[3]),
            _mm256_unpacklo_epi64(t[4], t[6]),
            _mm256_unpackhi_epi64(t[4], t[6]),
            _mm256_unpacklo_epi64(t[5], t[7]),
            _mm256_unpackhi_epi64(t[5], t[7]),
        ];
        [
            _mm256_permute2x128_si256::<0x20>(s[0], s[4]),
            _mm256_permute2x128_si256::<0x20>(s[1], s[5]),
            _mm256_permute2x128_si256::<0x20>(s[2], s[6]),
            _mm256_permute2x128_si256::<0x20>(s[3], s[7]),
            _mm256_permute2x128_si256::<0x31>(s[0], s[4]),
            _mm256_permute2x128_si256::<0x31>(s[1], s[5]),
            _mm256_permute2x128_si256::<0x31>(s[2], s[6]),
            _mm256_permute2x128_si256::<0x31>(s[3], s[7]),
        ]
    }

    #[target_feature(enable = "avx2")]
    fn outside(value: __m256i, bound: i32) -> __m256i {
        // Translate [-bound,bound] onto the lowest signed interval. Everything
        // outside wraps above its upper endpoint, so one comparison suffices.
        _mm256_cmpgt_epi32(
            _mm256_add_epi32(value, _mm256_set1_epi32(bound.wrapping_add(i32::MIN))),
            _mm256_set1_epi32((2 * bound).wrapping_add(i32::MIN)),
        )
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn capture<const CFL: bool>(
        raw: &[i32; 64],
        luma: &[i32; 64],
        scales: &[i32; 64],
        output: &mut [i16],
    ) -> bool {
        // SAFETY: caller checks CPU support and all array lengths. All pointers
        // below stay inside their arrays, independent of coefficient contents.
        unsafe {
            let mut values = transpose(raw);
            let zero = _mm256_setzero_si256();
            values[0] = _mm256_blend_epi32::<1>(values[0], zero);
            let mut invalid = zero;
            if CFL {
                let mut y = transpose(luma);
                y[0] = _mm256_blend_epi32::<1>(y[0], zero);
                for i in 0..8 {
                    let mut scale = _mm256_loadu_si256(scales.as_ptr().add(i * 8).cast());
                    if i == 0 {
                        scale = _mm256_blend_epi32::<1>(scale, zero);
                    }
                    invalid = _mm256_or_si256(invalid, outside(values[i], 65535));
                    invalid = _mm256_or_si256(invalid, outside(y[i], 4095));
                    invalid = _mm256_or_si256(invalid, outside(scale, 524288));
                    // 4095 * 524288 + 1024 fits i32. After >> 11, adding
                    // a residual in [-65535,65535] also fits i32. Invalid input
                    // may wrap in SIMD, but is rejected before any output store.
                    let product = _mm256_mullo_epi32(y[i], scale);
                    let prediction =
                        _mm256_srai_epi32::<11>(_mm256_add_epi32(product, _mm256_set1_epi32(1024)));
                    values[i] = _mm256_add_epi32(values[i], prediction);
                }
            }
            for value in values {
                invalid = _mm256_or_si256(invalid, outside(value, 4095));
            }
            if _mm256_movemask_epi8(invalid) != 0 {
                return false;
            }
            let dc = output[0];
            for (i, value) in values.into_iter().enumerate() {
                let packed = _mm_packs_epi32(
                    _mm256_castsi256_si128(value),
                    _mm256_extracti128_si256::<1>(value),
                );
                _mm_storeu_si128(output.as_mut_ptr().add(i * 8).cast(), packed);
            }
            output[0] = dc;
            true
        }
    }
}

#[cfg(all(test, target_arch = "x86_64", feature = "avx512"))]
#[path = "jpeg_capture_tests.rs"]
mod avx512_tests;

#[cfg(all(test, target_arch = "x86_64", feature = "avx"))]
mod tests {
    use super::*;

    #[test]
    fn vector_capture_matches_scalar_with_transposition_and_signed_rounding() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for factor in -128..=127 {
            for seed in 0..8 {
                let mut luma =
                    std::array::from_fn(|k| ((k as i32 * 193 + seed * 113) % 8191) - 4095);
                let scales = std::array::from_fn(|k| {
                    if k == 0 {
                        i32::MAX
                    } else {
                        (factor * 2048 / 84) + k as i32 * seed
                    }
                });
                let mut raw = [0; 64];
                let mut expected = [0; 64];
                expected[0] = 1234;
                for k in 1..64 {
                    expected[k] = ((k as i32 * 41 + seed * 73) % 8191 - 4095) as i16;
                    let pos = (k % 8) * 8 + k / 8;
                    raw[pos] = i32::from(expected[k]) - ((luma[pos] * scales[k] + 1024) >> 11);
                }
                raw[0] = i32::MIN;
                luma[0] = i32::MAX;
                let mut actual = [1234; 64];
                assert!(capture(&raw, Some((&luma, &scales)), &mut actual));
                assert_eq!(actual, expected, "factor={factor}, seed={seed}");
            }
        }
        for v in [-4095, -1, 0, 1, 4095] {
            let mut raw = [v; 64];
            raw[0] = i32::MIN;
            let mut actual = [257; 64];
            assert!(capture(&raw, None, &mut actual));
            assert_eq!(actual[0], 257);
            assert!(actual[1..].iter().all(|&x| i32::from(x) == v));
        }
    }

    #[test]
    fn rejected_vector_inputs_never_commit_and_cover_every_lane() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for k in 1..64 {
            for v in [i32::MIN, -65536, -4096, 4096, 65536, i32::MAX] {
                let mut raw = [0; 64];
                raw[k] = v;
                let mut output = [29; 64];
                assert!(!capture(&raw, None, &mut output));
                assert_eq!(output, [29; 64]);
                assert!(!capture(&raw, Some((&[0; 64], &[0; 64])), &mut output));
                assert_eq!(output, [29; 64]);
            }
            for (y, s, residual) in [
                (4096, 0, 0),
                (0, 524289, 0),
                (i32::MAX, i32::MAX, 0),
                (-4095, -524288, 0),
                (0, 0, i32::MIN),
                (4095, 524288, i32::MAX),
            ] {
                let mut raw = [0; 64];
                let mut luma = [0; 64];
                let mut scales = [0; 64];
                let pos = k % 8 * 8 + k / 8;
                raw[pos] = residual;
                luma[pos] = y;
                scales[k] = s;
                let mut output = [17; 64];
                assert!(!capture(&raw, Some((&luma, &scales)), &mut output));
                assert_eq!(output, [17; 64]);
            }
        }
    }
}
