// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Native port of enc_ans_simd.cc::EstimateTokenCostImpl, enc_cluster.cc's
//! conditioning/Entropy/Distance/KL kernels and base/fast_math-inl.h at
//! b866b3defec13494bfbaf705db93581cda118e37. No padded caller buffers required.
//!
//! AVX2 uses the source's eight histogram lanes. FMA is used ONLY for the
//! rational polynomial and KL NegMulAdd. Entropy's multiply/subtract and
//! accumulation stay separate. AArch64 NEON uses four lanes and native FMA,
//! with the source's adjacent-pair reduction (see neon.rs). Other targets use
//! the non-fused scalar source. This crate uses std; all architecture-specific
//! imports/detection are target-cfg guarded, with no optional std feature.
//! Public histogram storage remains u64; SIMD narrows only validated weights.

use super::hybrid_uint::HybridUintConfig;

#[cfg(target_arch = "aarch64")]
#[path = "neon.rs"]
mod neon;

/// Bound temporary token storage independently of the input volume. Callers
/// scatter these tokens into raw or weighted histograms, never into bitstreams.
pub(super) const TOKEN_BATCH: usize = 256;

pub(super) fn tokens(
    values: &[u32],
    config: HybridUintConfig,
    output: &mut [u32],
    widths: &mut [u32],
) -> u64 {
    assert!(output.len() >= values.len() && widths.len() >= values.len());
    // The source's ten-bit large-value shift preserves at most twelve leading
    // mantissa bits at exponent 22. Its candidate list needs only two; wider
    // public configurations retain the exact scalar implementation.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if config.msb_in_token() <= 12 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature check; the callee checks full-vector bounds.
        return unsafe { x86::tokens(values, config, output, widths) };
    }
    #[cfg(target_arch = "aarch64")]
    if config.msb_in_token() <= 12 && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: NEON detected (normally baseline on std AArch64 targets);
        // full vectors stay in bounds, and the scalar kernel handles the tail.
        return unsafe { neon::tokens(values, config, output, widths) };
    }
    scalar_tokens(values, config, output, widths)
}

fn scalar_tokens(
    values: &[u32],
    config: HybridUintConfig,
    output: &mut [u32],
    widths: &mut [u32],
) -> u64 {
    let mut extra = 0;
    for ((&value, token), width) in values.iter().zip(output).zip(widths) {
        let encoded = config.encode(value);
        *token = encoded.token;
        *width = encoded.nbits;
        extra += u64::from(encoded.nbits);
    }
    extra
}

pub(super) fn add_tokens(values: &[u32], config: HybridUintConfig, counts: &mut [u64]) -> u64 {
    let mut output = [0; TOKEN_BATCH];
    let mut widths = [0; TOKEN_BATCH];
    let mut extra = 0;
    for batch in values.chunks(TOKEN_BATCH) {
        extra += tokens(batch, config, &mut output, &mut widths);
        for &token in &output[..batch.len()] {
            counts[token as usize] += 1;
        }
    }
    extra
}

/// Validate the signed source domain before narrowing; return exact nonzero
/// extent rather than exposing Highway's target-dependent trailing padding.
pub(super) fn condition(counts: &[u64]) -> Option<(u64, usize)> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature check; unaligned loads stay inside the slice.
        return unsafe { x86::condition(counts) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: feature detected; the callee validates before narrowing.
        return unsafe { neon::condition(counts) };
    }
    scalar_condition(counts)
}

fn scalar_condition(counts: &[u64]) -> Option<(u64, usize)> {
    let mut total = 0u64;
    let mut length = 0;
    for (i, &count) in counts.iter().enumerate() {
        total = total.checked_add(count)?;
        if total > i32::MAX as u64 {
            return None;
        }
        if count != 0 {
            length = i + 1;
        }
    }
    Some((total, length))
}

// Keep the frozen decimal literals and their f32 rounding, not f64 substitutes.
#[allow(clippy::excessive_precision)]
const P: [f32; 3] = [
    -0.0000018503833400518310,
    1.4287160470083755,
    0.74245873327820566,
];
#[allow(clippy::excessive_precision)]
const Q: [f32; 3] = [0.99032814277590719, 1.0096718572241148, 0.17409343003366853];

pub(super) fn fast_log2(value: f32) -> f32 {
    debug_assert!(value > 0.0 && value.is_finite());
    scalar_log2(value, false)
}

fn scalar_log2(value: f32, fused: bool) -> f32 {
    let bits = value.to_bits() as i32;
    let exponent = bits.wrapping_sub(0x3f2a_aaab) >> 23;
    let mantissa = f32::from_bits(bits.wrapping_sub(exponent << 23) as u32) - 1.0;
    let madd = |a: f32, b: f32, c: f32| if fused { a.mul_add(b, c) } else { a * b + c };
    let numerator = madd(madd(P[2], mantissa, P[1]), mantissa, P[0]);
    let denominator = madd(madd(Q[2], mantissa, Q[1]), mantissa, Q[0]);
    numerator / denominator + exponent as f32
}

pub(super) fn entropy_term(count: u64, inverse: f32, total: u64) -> f32 {
    if count == 0 || count as f32 == total as f32 {
        0.0
    } else {
        0.0 - count as f32 * fast_log2(inverse * count as f32)
    }
}

pub(super) fn entropy(counts: &[u64], total: u64) -> f32 {
    combined_entropy(counts, &[], total)
}

/// HistogramDistance uses this same kernel with integer counts added BEFORE
/// conversion. Large Rust-only combined totals retain the overflow-safe scalar
/// extension instead of reproducing source signed-int overflow.
pub(super) fn combined_entropy(a: &[u64], b: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if total <= i32::MAX as u64 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: histogram callers validated weights; sum bounds each lane.
        return unsafe { x86::entropy(a, b, total, std::arch::is_x86_feature_detected!("fma")) };
    }
    #[cfg(target_arch = "aarch64")]
    if total <= i32::MAX as u64 && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: validated histograms; the combined total bounds each lane.
        return unsafe { neon::entropy(a, b, total) };
    }
    let inverse = 1.0 / total as f32;
    (0..a.len().max(b.len()))
        .map(|i| {
            entropy_term(
                a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0),
                inverse,
                total,
            )
        })
        .sum()
}

pub(super) fn coding_cost(actual: &[u64], coding: &[u64], coding_total: u64) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: feature detected; callers validated both histograms.
        return unsafe { neon::coding_cost(actual, coding, coding_total) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: callers validated both histograms; loads handle short tails.
        return unsafe {
            x86::coding_cost(
                actual,
                coding,
                coding_total,
                std::arch::is_x86_feature_detected!("fma"),
            )
        };
    }
    let inverse = 1.0 / coding_total as f32;
    let mut cost = 0.0;
    for (i, &count) in actual.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let coding_count = coding.get(i).copied().unwrap_or(0);
        if coding_count == 0 {
            return f32::INFINITY;
        }
        cost -= count as f32 * fast_log2(coding_count as f32 * inverse);
    }
    cost
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn tokens(
        values: &[u32],
        cfg: HybridUintConfig,
        output: &mut [u32],
        widths: &mut [u32],
    ) -> u64 {
        // SAFETY: dispatched only with AVX2; every load/store covers eight
        // elements within the caller-checked slices. Scalar handles the tail.
        unsafe {
            let sign = _mm256_set1_epi32(i32::MIN);
            let split = _mm256_set1_epi32(cfg.split_token() as i32 - 1);
            let large_threshold = _mm256_set1_epi32(((1 << 22) - 1) ^ i32::MIN);
            let ml = cfg.msb_in_token() + cfg.lsb_in_token();
            let shift_ml = _mm_cvtsi32_si128(ml as i32);
            let shift_m = _mm_cvtsi32_si128((23 - ml) as i32);
            let base = _mm256_set1_epi32(
                cfg.split_token().wrapping_sub(cfg.split_exponent() << ml) as i32,
            );
            let mask_l = _mm256_set1_epi32(cfg.lsb_mask() as i32);
            let mask_m = _mm256_set1_epi32(((1 << cfg.msb_in_token()) - 1) << cfg.lsb_in_token());
            let mut extra = _mm256_setzero_si256();
            let end = values.len() / 8 * 8;
            for i in (0..end).step_by(8) {
                let value = _mm256_loadu_si256(values.as_ptr().add(i).cast());
                let large = _mm256_cmpgt_epi32(_mm256_xor_si256(value, sign), large_threshold);
                let nonliteral = _mm256_cmpgt_epi32(
                    _mm256_xor_si256(value, sign),
                    _mm256_xor_si256(split, sign),
                );
                let fixed = _mm256_blendv_epi8(value, _mm256_srli_epi32::<10>(value), large);
                // Values above 2^22-1 were shifted by ten, so signed conversion
                // is exact and cannot round an exponent upward at u32::MAX.
                let bits = _mm256_castps_si256(_mm256_cvtepi32_ps(fixed));
                let exp = _mm256_add_epi32(
                    _mm256_srli_epi32::<23>(bits),
                    _mm256_and_si256(large, _mm256_set1_epi32(10)),
                );
                let n = _mm256_sub_epi32(exp, _mm256_set1_epi32(127));
                let eb = _mm256_and_si256(
                    nonliteral,
                    _mm256_sub_epi32(exp, _mm256_set1_epi32((127 + ml) as i32)),
                );
                let token = _mm256_or_si256(
                    _mm256_add_epi32(base, _mm256_sll_epi32(n, shift_ml)),
                    _mm256_or_si256(
                        _mm256_and_si256(value, mask_l),
                        _mm256_and_si256(_mm256_srl_epi32(bits, shift_m), mask_m),
                    ),
                );
                _mm256_storeu_si256(
                    output.as_mut_ptr().add(i).cast(),
                    _mm256_blendv_epi8(value, token, nonliteral),
                );
                _mm256_storeu_si256(widths.as_mut_ptr().add(i).cast(), eb);
                // Widen before accumulation: preserve Rust's existing >2^32
                // extra-bit contract rather than the source uint32 wraparound.
                extra = _mm256_add_epi64(extra, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(eb)));
                extra = _mm256_add_epi64(
                    extra,
                    _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(eb)),
                );
            }
            let mut lanes = [0u64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), extra);
            lanes.into_iter().sum::<u64>()
                + scalar_tokens(&values[end..], cfg, &mut output[end..], &mut widths[end..])
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn condition(counts: &[u64]) -> Option<(u64, usize)> {
        // SAFETY: only complete four-u64 vectors are loaded; the caller may
        // supply arbitrary u64 values, so reject both high-bit and >i32 weights.
        unsafe {
            let zero = _mm256_setzero_si256();
            let limit = _mm256_set1_epi64x(i32::MAX as i64);
            let mut sum = zero;
            let mut length = 0;
            let end = counts.len() / 4 * 4;
            for i in (0..end).step_by(4) {
                let v = _mm256_loadu_si256(counts.as_ptr().add(i).cast());
                let invalid =
                    _mm256_or_si256(_mm256_cmpgt_epi64(v, limit), _mm256_cmpgt_epi64(zero, v));
                if _mm256_movemask_epi8(invalid) != 0 {
                    return None;
                }
                sum = _mm256_add_epi64(sum, v);
                let nonzero =
                    (!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v, zero)))) & 15;
                if nonzero != 0 {
                    length = i + (32 - nonzero.leading_zeros()) as usize;
                }
            }
            let mut lanes = [0u64; 4];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum);
            let mut total = lanes.into_iter().sum::<u64>();
            for (i, &v) in counts[end..].iter().enumerate() {
                total = total.checked_add(v)?;
                if v != 0 {
                    length = end + i + 1;
                }
            }
            (total <= i32::MAX as u64).then_some((total, length))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn counts8(counts: &[u64], i: usize) -> __m256i {
        // SAFETY: complete loads checked locally. We pack validated u64 counts
        // to source i32 lanes without changing the public histogram layout.
        unsafe {
            if i + 8 <= counts.len() {
                let indices = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
                let lo = _mm256_permutevar8x32_epi32(
                    _mm256_loadu_si256(counts.as_ptr().add(i).cast()),
                    indices,
                );
                let hi = _mm256_permutevar8x32_epi32(
                    _mm256_loadu_si256(counts.as_ptr().add(i + 4).cast()),
                    indices,
                );
                _mm256_inserti128_si256::<1>(
                    _mm256_castsi128_si256(_mm256_castsi256_si128(lo)),
                    _mm256_castsi256_si128(hi),
                )
            } else {
                let mut tail = [0i32; 8];
                for (lane, value) in tail.iter_mut().enumerate() {
                    *value = counts.get(i + lane).copied().unwrap_or(0) as i32;
                }
                _mm256_loadu_si256(tail.as_ptr().cast())
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fused_madd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fused_nmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fnmadd_ps(a, b, c)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn madd<const FMA: bool>(a: __m256, b: __m256, c: __m256) -> __m256 {
        if FMA {
            // SAFETY: FMA specialization selected only after its feature check.
            unsafe { fused_madd(a, b, c) }
        } else {
            _mm256_add_ps(_mm256_mul_ps(a, b), c)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn log2<const FMA: bool>(value: __m256) -> __m256 {
        // SAFETY: AVX2 is active; madd's fused specialization is feature checked.
        unsafe {
            let bits = _mm256_castps_si256(value);
            let exponent =
                _mm256_srai_epi32::<23>(_mm256_sub_epi32(bits, _mm256_set1_epi32(0x3f2a_aaab)));
            let mantissa = _mm256_sub_ps(
                _mm256_castsi256_ps(_mm256_sub_epi32(bits, _mm256_slli_epi32::<23>(exponent))),
                _mm256_set1_ps(1.0),
            );
            let p = madd::<FMA>(
                madd::<FMA>(_mm256_set1_ps(P[2]), mantissa, _mm256_set1_ps(P[1])),
                mantissa,
                _mm256_set1_ps(P[0]),
            );
            let q = madd::<FMA>(
                madd::<FMA>(_mm256_set1_ps(Q[2]), mantissa, _mm256_set1_ps(Q[1])),
                mantissa,
                _mm256_set1_ps(Q[0]),
            );
            _mm256_add_ps(_mm256_div_ps(p, q), _mm256_cvtepi32_ps(exponent))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sum8(value: __m256) -> f32 {
        // Highway generic_ops-inl.h reduces across blocks, then Reverse4 and
        // Reverse2: ((v0+v4)+(v3+v7)) + ((v1+v5)+(v2+v6)).
        let half = _mm_add_ps(
            _mm256_castps256_ps128(value),
            _mm256_extractf128_ps::<1>(value),
        );
        let pairs = _mm_add_ps(half, _mm_shuffle_ps::<0x1b>(half, half));
        _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps::<0xb1>(pairs, pairs)))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn entropy(a: &[u64], b: &[u64], total: u64, fused: bool) -> f32 {
        // SAFETY: the dispatch checked AVX2/FMA and the total's signed range.
        unsafe {
            if fused {
                entropy_fused(a, b, total)
            } else {
                entropy_impl::<false>(a, b, total)
            }
        }
    }

    // Feature-enabled entry points allow the complete fused loop to inline,
    // without a cross-feature function call for every polynomial coefficient.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn entropy_fused(a: &[u64], b: &[u64], total: u64) -> f32 {
        // SAFETY: both features are active in this function.
        unsafe { entropy_impl::<true>(a, b, total) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn entropy_impl<const FMA: bool>(a: &[u64], b: &[u64], total: u64) -> f32 {
        // SAFETY: counts8 handles unpadded tails and log2's zero input is finite
        // under source range reduction. Zero terms are masked explicitly.
        unsafe {
            let inv = _mm256_set1_ps(1.0 / total as f32);
            let total = _mm256_set1_ps(total as f32);
            let zero = _mm256_setzero_ps();
            let mut sum = zero;
            for i in (0..a.len().max(b.len())).step_by(8) {
                let count = _mm256_cvtepi32_ps(_mm256_add_epi32(counts8(a, i), counts8(b, i)));
                let skip = _mm256_or_ps(
                    _mm256_cmp_ps::<_CMP_EQ_OQ>(count, total),
                    _mm256_cmp_ps::<_CMP_EQ_OQ>(count, zero),
                );
                let term = _mm256_sub_ps(
                    zero,
                    _mm256_mul_ps(count, log2::<FMA>(_mm256_mul_ps(inv, count))),
                );
                sum = _mm256_add_ps(sum, _mm256_andnot_ps(skip, term));
            }
            sum8(sum)
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn coding_cost(a: &[u64], b: &[u64], total: u64, fused: bool) -> f32 {
        // SAFETY: dispatch checked AVX2/FMA and callers validate counts.
        unsafe {
            if fused {
                coding_cost_fused(a, b, total)
            } else {
                coding_cost_impl::<false>(a, b, total)
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn coding_cost_fused(a: &[u64], b: &[u64], total: u64) -> f32 {
        // SAFETY: both features are active in this function.
        unsafe { coding_cost_impl::<true>(a, b, total) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn coding_cost_impl<const FMA: bool>(a: &[u64], b: &[u64], total: u64) -> f32 {
        // SAFETY: counts8 masks tails. Select zero for unused actual symbols
        // BEFORE NegMulAdd so absent coding symbols cannot cause 0 * infinity.
        unsafe {
            let inverse = _mm256_set1_ps(1.0 / total as f32);
            let zero = _mm256_setzero_si256();
            let mut sum = _mm256_setzero_ps();
            for i in (0..a.len()).step_by(8) {
                let actual = counts8(a, i);
                let coding = counts8(b, i);
                let unused = _mm256_castsi256_ps(_mm256_cmpeq_epi32(actual, zero));
                let absent = _mm256_castsi256_ps(_mm256_cmpeq_epi32(coding, zero));
                let log = log2::<FMA>(_mm256_mul_ps(_mm256_cvtepi32_ps(coding), inverse));
                let neg_cost = _mm256_andnot_ps(
                    unused,
                    _mm256_blendv_ps(log, _mm256_set1_ps(f32::NEG_INFINITY), absent),
                );
                let count = _mm256_cvtepi32_ps(actual);
                sum = if FMA {
                    fused_nmadd(count, neg_cost, sum)
                } else {
                    _mm256_sub_ps(sum, _mm256_mul_ps(count, neg_cost))
                };
            }
            sum8(sum)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_dispatch_matches_scalar_all_configs_boundaries_and_tails() {
        let mut values = vec![0, 1, u32::MAX, i32::MAX as u32, 1 << 31];
        for bit in 1..32 {
            for delta in 0..=3 {
                values.push((1u32 << bit).wrapping_sub(delta));
                values.push((1u32 << bit).saturating_add(delta));
            }
        }
        let mut random = 0x9e37_79b9u32;
        for _ in 0..513 {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            values.push(random);
        }
        for split in 0..=15 {
            for msb in 0..=split {
                for lsb in 0..=split - msb {
                    let cfg = HybridUintConfig::new(split, msb, lsb).unwrap();
                    for tail in 0..8 {
                        let values = &values[tail..];
                        let mut actual = vec![u32::MAX; values.len() + 1];
                        let mut widths = actual.clone();
                        let mut expected = actual.clone();
                        let mut expected_widths = actual.clone();
                        assert_eq!(
                            tokens(values, cfg, &mut actual, &mut widths),
                            scalar_tokens(values, cfg, &mut expected, &mut expected_widths)
                        );
                        assert_eq!(actual, expected, "{cfg:?} tail={tail}");
                        assert_eq!(widths, expected_widths);
                    }
                }
            }
        }
    }

    #[test]
    fn conditioning_checks_wide_weights_and_exact_extent() {
        for length in 0..=263 {
            let mut counts = vec![0; length];
            for i in (0..length).step_by(11) {
                counts[i] = (i + 1) as u64;
            }
            assert_eq!(condition(&counts), scalar_condition(&counts));
            for value in [i32::MAX as u64, i32::MAX as u64 + 1, u64::MAX, 1 << 63] {
                if length != 0 {
                    counts[length - 1] = value;
                    assert_eq!(condition(&counts), scalar_condition(&counts));
                }
            }
        }
        assert_eq!(condition(&[i32::MAX as u64, 1]), None);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn reference_sum(lanes: [f32; 8]) -> f32 {
        let a = lanes[0] + lanes[4];
        let b = lanes[1] + lanes[5];
        let c = lanes[2] + lanes[6];
        let d = lanes[3] + lanes[7];
        (a + d) + (b + c)
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn fused_and_nonfused_histogram_lanes_match_source_order() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for fused in [false, true] {
            if fused && !std::arch::is_x86_feature_detected!("fma") {
                continue;
            }
            for length in [1, 7, 8, 9, 31, 255, 256, 269] {
                let a: Vec<_> = (0..length)
                    .map(|i| if i % 7 == 0 { 0 } else { (i * 173 + 1) as u64 })
                    .collect();
                let b: Vec<_> = (0..length + 3).map(|i| (i * 19 + 3) as u64).collect();
                let total = a.iter().chain(&b).sum::<u64>();
                let coding_total = b.iter().sum::<u64>();
                let mut entropy_lanes = [0.0f32; 8];
                let mut cost_lanes = [0.0f32; 8];
                for i in 0..b.len() {
                    let count = a.get(i).copied().unwrap_or(0) + b[i];
                    let term = if count as f32 == total as f32 {
                        0.0
                    } else {
                        0.0 - count as f32 * scalar_log2((1.0 / total as f32) * count as f32, fused)
                    };
                    entropy_lanes[i % 8] += term;
                    if i < a.len() {
                        let log = if a[i] == 0 {
                            0.0
                        } else {
                            scalar_log2(b[i] as f32 * (1.0 / coding_total as f32), fused)
                        };
                        cost_lanes[i % 8] = if fused {
                            (-(a[i] as f32)).mul_add(log, cost_lanes[i % 8])
                        } else {
                            cost_lanes[i % 8] - a[i] as f32 * log
                        };
                    }
                }
                // SAFETY: feature checks above cover each requested variant.
                unsafe {
                    assert_eq!(
                        x86::entropy(&a, &b, total, fused).to_bits(),
                        reference_sum(entropy_lanes).to_bits()
                    );
                    assert_eq!(
                        x86::coding_cost(&a, &b, coding_total, fused).to_bits(),
                        reference_sum(cost_lanes).to_bits()
                    );
                    assert!(x86::coding_cost(&[0, 1], &[1], 1, fused).is_infinite());
                    assert!(x86::coding_cost(&[0, 1], &[0, 1], 1, fused).is_finite());
                }
            }
        }
    }

    #[test]
    fn entropy_empty_singleton_and_large_combined_total() {
        assert_eq!(entropy(&[], 0), 0.0);
        assert_eq!(entropy(&[0, 100], 100), 0.0);
        assert_eq!(
            entropy_term((1 << 25) - 1, 1.0 / (1 << 25) as f32, 1 << 25),
            0.0
        );
        let a = [i32::MAX as u64, 0];
        let b = [0, i32::MAX as u64];
        let total = 2 * i32::MAX as u64;
        let expected = 2.0 * entropy_term(i32::MAX as u64, 1.0 / total as f32, total);
        assert_eq!(combined_entropy(&a, &b, total), expected);
    }
}
