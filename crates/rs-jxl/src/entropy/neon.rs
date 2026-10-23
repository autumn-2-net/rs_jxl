// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Native AArch64 translation of libjxl at
//! b866b3defec13494bfbaf705db93581cda118e37:
//! enc_ans_simd.cc::EstimateTokenCostImpl; enc_cluster.cc::{HistogramCondition,
//! Entropy, HistogramEntropy, HistogramDistance, HistogramKLDivergence};
//! base/{fast_math-inl.h, rational_polynomial-inl.h}.
//!
//! Highway's arm_neon-inl.h uses four f32 lanes, native vfma/vfms, actual vdiv,
//! and vaddvq_f32 reduction: (lane0 + lane1) + (lane2 + lane3). Only polynomial
//! MulAdd and KL NegMulAdd fuse; Entropy's multiply/subtract/add stay separate.
//! Scalar tails for tokens are exact; histogram tails are zero-filled locally
//! to retain lane accumulation order without requiring caller padding.
//! Histogram inputs to entropy/coding_cost have been validated by the caller.

use std::arch::aarch64::*;

use super::{HybridUintConfig, P, Q, scalar_tokens};

/// The source float-exponent trick is exact for the encoder's configurations.
/// The parent keeps public configs with msb_in_token > 12 on the scalar path.
#[target_feature(enable = "neon")]
pub(super) unsafe fn tokens(
    values: &[u32],
    cfg: HybridUintConfig,
    output: &mut [u32],
    widths: &mut [u32],
) -> u64 {
    debug_assert!(cfg.msb_in_token() <= 12);
    debug_assert!(output.len() >= values.len() && widths.len() >= values.len());
    // SAFETY: the parent checks NEON and the slice lengths. Every vector
    // load/store covers four in-bounds values, without 16-byte alignment needs.
    unsafe {
        let split = vdupq_n_u32(cfg.split_token());
        let large_threshold = vdupq_n_u32((1 << 22) - 1);
        let ml = cfg.msb_in_token() + cfg.lsb_in_token();
        let shift_ml = vdupq_n_s32(ml as i32);
        let shift_m = vdupq_n_s32(-((23 - ml) as i32));
        let base = vdupq_n_u32(cfg.split_token().wrapping_sub(cfg.split_exponent() << ml));
        let mask_l = vdupq_n_u32(cfg.lsb_mask());
        let mask_m = vdupq_n_u32(((1 << cfg.msb_in_token()) - 1) << cfg.lsb_in_token());
        let mut extra = vdupq_n_u64(0);
        let end = values.len() / 4 * 4;
        for i in (0..end).step_by(4) {
            let value = vld1q_u32(values.as_ptr().add(i));
            let large = vcgtq_u32(value, large_threshold);
            let nonliteral = vcgeq_u32(value, split);
            let fixed = vbslq_u32(large, vshrq_n_u32::<10>(value), value);
            // Both ranges fit exactly in f32; u32::MAX cannot round to 2^32.
            let bits = vreinterpretq_u32_f32(vcvtq_f32_u32(fixed));
            let exp = vaddq_u32(vshrq_n_u32::<23>(bits), vandq_u32(large, vdupq_n_u32(10)));
            let n = vsubq_u32(exp, vdupq_n_u32(127));
            let eb = vandq_u32(nonliteral, vsubq_u32(exp, vdupq_n_u32(127 + ml)));
            let token = vorrq_u32(
                vaddq_u32(base, vshlq_u32(n, shift_ml)),
                vorrq_u32(
                    vandq_u32(value, mask_l),
                    vandq_u32(vshlq_u32(bits, shift_m), mask_m),
                ),
            );
            vst1q_u32(
                output.as_mut_ptr().add(i),
                vbslq_u32(nonliteral, token, value),
            );
            vst1q_u32(widths.as_mut_ptr().add(i), eb);
            // Preserve the Rust u64 bit-count contract, not source u32 wrap.
            extra = vpadalq_u32(extra, eb);
        }
        vaddvq_u64(extra)
            + scalar_tokens(&values[end..], cfg, &mut output[end..], &mut widths[end..])
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn condition(counts: &[u64]) -> Option<(u64, usize)> {
    // SAFETY: NEON detected. Load two complete pairs, reject wide unsigned
    // counts before narrowing, and check each block's total before continuing.
    unsafe {
        let limit = vdupq_n_u64(i32::MAX as u64);
        let positions = [1u32, 2, 3, 4];
        let positions = vld1q_u32(positions.as_ptr());
        let mut total = 0u64;
        let mut length = 0;
        let end = counts.len() / 4 * 4;
        for i in (0..end).step_by(4) {
            let lo = vld1q_u64(counts.as_ptr().add(i));
            let hi = vld1q_u64(counts.as_ptr().add(i + 2));
            let invalid = vorrq_u64(vcgtq_u64(lo, limit), vcgtq_u64(hi, limit));
            if vmaxvq_u32(vreinterpretq_u32_u64(invalid)) != 0 {
                return None;
            }
            let count = vcombine_u32(vmovn_u64(lo), vmovn_u64(hi));
            total += vaddlvq_u32(count);
            if total > i32::MAX as u64 {
                return None;
            }
            let last = vmaxvq_u32(vandq_u32(vcgtq_u32(count, vdupq_n_u32(0)), positions));
            if last != 0 {
                length = i + last as usize;
            }
        }
        for (i, &count) in counts[end..].iter().enumerate() {
            total = total.checked_add(count)?;
            if total > i32::MAX as u64 {
                return None;
            }
            if count != 0 {
                length = end + i + 1;
            }
        }
        Some((total, length))
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn counts4(counts: &[u64], i: usize) -> int32x4_t {
    // SAFETY: callers validated source i32 weights. Full loads are checked
    // locally; short/missing slices use a zero-filled four-lane stack buffer.
    unsafe {
        if i <= counts.len() && counts.len() - i >= 4 {
            let lo = vmovn_u64(vld1q_u64(counts.as_ptr().add(i)));
            let hi = vmovn_u64(vld1q_u64(counts.as_ptr().add(i + 2)));
            vreinterpretq_s32_u32(vcombine_u32(lo, hi))
        } else {
            let mut tail = [0i32; 4];
            for (lane, count) in tail.iter_mut().enumerate() {
                *count = counts.get(i + lane).copied().unwrap_or(0) as i32;
            }
            vld1q_s32(tail.as_ptr())
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn log2(value: float32x4_t) -> float32x4_t {
    let bits = vreinterpretq_s32_f32(value);
    let exponent = vshrq_n_s32::<23>(vsubq_s32(bits, vdupq_n_s32(0x3f2a_aaab)));
    let mantissa = vsubq_f32(
        vreinterpretq_f32_s32(vsubq_s32(bits, vshlq_n_s32::<23>(exponent))),
        vdupq_n_f32(1.0),
    );
    let p = vfmaq_f32(
        vdupq_n_f32(P[0]),
        vfmaq_f32(vdupq_n_f32(P[1]), vdupq_n_f32(P[2]), mantissa),
        mantissa,
    );
    let q = vfmaq_f32(
        vdupq_n_f32(Q[0]),
        vfmaq_f32(vdupq_n_f32(Q[1]), vdupq_n_f32(Q[2]), mantissa),
        mantissa,
    );
    vaddq_f32(vdivq_f32(p, q), vcvtq_f32_s32(exponent))
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn entropy(a: &[u64], b: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    debug_assert!(total <= i32::MAX as u64);
    // SAFETY: dispatch checks NEON and the combined signed total. counts4
    // handles every tail, and log2 follows the source even for masked zeros.
    unsafe {
        let inverse = vdupq_n_f32(1.0 / total as f32);
        let total = vdupq_n_f32(total as f32);
        let zero = vdupq_n_f32(0.0);
        let mut sum = zero;
        for i in (0..a.len().max(b.len())).step_by(4) {
            let count = vcvtq_f32_s32(vaddq_s32(counts4(a, i), counts4(b, i)));
            let skip = vorrq_u32(vceqq_f32(count, total), vceqq_f32(count, zero));
            let term = vsubq_f32(zero, vmulq_f32(count, log2(vmulq_f32(inverse, count))));
            sum = vaddq_f32(sum, vbslq_f32(skip, zero, term));
        }
        vaddvq_f32(sum)
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn coding_cost(actual: &[u64], coding: &[u64], total: u64) -> f32 {
    // SAFETY: dispatch checks NEON; the caller validated both histograms.
    // Mask unused actual symbols BEFORE vfms, including missing coding lanes,
    // so 0 * infinity cannot contaminate the accumulated cost with NaN.
    unsafe {
        let inverse = vdupq_n_f32(1.0 / total as f32);
        let zero = vdupq_n_s32(0);
        let mut sum = vdupq_n_f32(0.0);
        for i in (0..actual.len()).step_by(4) {
            let count = counts4(actual, i);
            let coding = counts4(coding, i);
            let unused = vceqq_s32(count, zero);
            let absent = vceqq_s32(coding, zero);
            let log = log2(vmulq_f32(vcvtq_f32_s32(coding), inverse));
            let neg_cost = vbslq_f32(
                unused,
                vdupq_n_f32(0.0),
                vbslq_f32(absent, vdupq_n_f32(f32::NEG_INFINITY), log),
            );
            sum = vfmsq_f32(sum, vcvtq_f32_s32(count), neg_cost);
        }
        vaddvq_f32(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{scalar_condition, scalar_log2};
    use super::*;

    fn source_sum(lanes: [f32; 4]) -> f32 {
        (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    }

    // Scalar arithmetic oracles preserve NEON's four independent accumulators,
    // fused polynomial/KL operations, and adjacent-pair final reduction.
    fn entropy_oracle(a: &[u64], b: &[u64], total: u64) -> f32 {
        if total == 0 {
            return 0.0;
        }
        let inverse = 1.0 / total as f32;
        let mut lanes = [0.0f32; 4];
        for i in 0..a.len().max(b.len()) {
            let count = a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0);
            let count = count as f32;
            let term = if count == 0.0 || count == total as f32 {
                0.0
            } else {
                0.0 - count * scalar_log2(inverse * count, true)
            };
            lanes[i % 4] += term;
        }
        source_sum(lanes)
    }

    fn coding_oracle(actual: &[u64], coding: &[u64], total: u64) -> f32 {
        let inverse = 1.0 / total as f32;
        let mut lanes = [0.0f32; 4];
        for (i, &count) in actual.iter().enumerate() {
            let coding = coding.get(i).copied().unwrap_or(0);
            let log = if count == 0 {
                0.0
            } else if coding == 0 {
                f32::NEG_INFINITY
            } else {
                scalar_log2(coding as f32 * inverse, true)
            };
            lanes[i % 4] = (-(count as f32)).mul_add(log, lanes[i % 4]);
        }
        source_sum(lanes)
    }

    #[test]
    fn log_polynomial_and_reduction_match_source_operations() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for bits in (0x3000_0000u32..=0x3f80_0000).step_by(7919) {
            let values = [bits - 1, bits, bits + 1, 0x3f2a_aaab].map(f32::from_bits);
            let mut actual = [0.0; 4];
            // SAFETY: NEON checked; each array contains a complete vector.
            unsafe { vst1q_f32(actual.as_mut_ptr(), log2(vld1q_f32(values.as_ptr()))) };
            assert_eq!(
                actual.map(f32::to_bits),
                values.map(|v| scalar_log2(v, true).to_bits()),
                "input bits {bits:x}"
            );
        }
        let lanes = [16_777_216.0, -16_777_216.0, 1.0, 1.0];
        // SAFETY: NEON checked; full vector. This distinguishes adjacent-pair
        // reduction from the x86 Reverse4/Reverse2 four-lane reduction.
        let sum = unsafe { vaddvq_f32(vld1q_f32(lanes.as_ptr())) };
        assert_eq!(sum, 2.0);
        assert_eq!(sum.to_bits(), source_sum(lanes).to_bits());
    }

    #[test]
    fn tokens_match_integer_oracle_all_native_configs_and_unaligned_tails() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let mut input = vec![0, 1, u32::MAX, i32::MAX as u32, 1 << 31];
        for bit in 1..32 {
            for delta in 0..=3 {
                input.push((1u32 << bit).wrapping_sub(delta));
                input.push((1u32 << bit).saturating_add(delta));
            }
        }
        let mut random = 0xb866_b3deu32;
        for _ in 0..257 {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            input.push(random);
        }
        for split in 0u32..=15 {
            for msb in 0..=split.min(12) {
                for lsb in 0..=split - msb {
                    let cfg = HybridUintConfig::new(split, msb, lsb).unwrap();
                    for offset in 0..4 {
                        let values = &input[offset..];
                        let mut actual = vec![u32::MAX; values.len() + 8];
                        let mut widths = actual.clone();
                        let mut expected = actual.clone();
                        let mut expected_widths = actual.clone();
                        let out_offset = (offset + 1) % 4;
                        let width_offset = (offset + 3) % 4;
                        let out_range = out_offset..out_offset + values.len();
                        let width_range = width_offset..width_offset + values.len();
                        let expected_extra = scalar_tokens(
                            values,
                            cfg,
                            &mut expected[out_range.clone()],
                            &mut expected_widths[width_range.clone()],
                        );
                        // SAFETY: NEON checked; supported config and exact
                        // slice extents. Sentinels surround both outputs.
                        let extra = unsafe {
                            tokens(
                                values,
                                cfg,
                                &mut actual[out_range],
                                &mut widths[width_range],
                            )
                        };
                        assert_eq!(extra, expected_extra, "{cfg:?}, offset {offset}");
                        assert_eq!(actual, expected);
                        assert_eq!(widths, expected_widths);
                    }
                }
            }
        }
        for len in 0..8 {
            let values = &input[..len];
            let cfg = HybridUintConfig::default();
            let mut output = vec![0; len];
            let mut widths = vec![0; len];
            // SAFETY: NEON checked; output extents equal the input extent.
            let extra = unsafe { tokens(values, cfg, &mut output, &mut widths) };
            let expected: Vec<_> = values.iter().map(|&v| cfg.encode(v)).collect();
            assert_eq!(
                extra,
                expected.iter().map(|v| u64::from(v.nbits)).sum::<u64>()
            );
            assert_eq!(output, expected.iter().map(|v| v.token).collect::<Vec<_>>());
            assert_eq!(widths, expected.iter().map(|v| v.nbits).collect::<Vec<_>>());
        }
    }

    #[test]
    fn condition_matches_checked_oracle_before_narrowing() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for len in 0..=263 {
            for offset in 0..2 {
                let mut storage = vec![0; offset + len];
                let counts = &mut storage[offset..];
                for i in (0..len).step_by(7) {
                    counts[i] = (i * 17 + 1) as u64;
                }
                // SAFETY: NEON checked; condition accepts arbitrary u64s.
                assert_eq!(unsafe { condition(counts) }, scalar_condition(counts));
                // Exercise every vector lane and the scalar tail, including
                // invalid high bits whose low 32 bits look valid after narrowing.
                for i in 0..len.min(8) {
                    let i = len - 1 - i;
                    let old = counts[i];
                    for value in [i32::MAX as u64, 1 << 31, 1 << 32, 1 << 63, u64::MAX] {
                        counts[i] = value;
                        // SAFETY: same checked feature and input contract.
                        assert_eq!(unsafe { condition(counts) }, scalar_condition(counts));
                    }
                    counts[i] = old;
                }
            }
        }
        for len in 1..=9 {
            for pos in 0..len {
                let mut counts = vec![0; len];
                counts[pos] = i32::MAX as u64;
                // SAFETY: NEON checked; singleton sits in each possible lane.
                assert_eq!(
                    unsafe { condition(&counts) },
                    Some((i32::MAX as u64, pos + 1))
                );
            }
        }
        // SAFETY: NEON checked; total overflow spans separate vector blocks.
        assert_eq!(
            unsafe { condition(&[i32::MAX as u64, 0, 0, 0, 1, 0, 0, 0]) },
            None
        );
    }

    #[test]
    fn histogram_kernels_match_fused_four_lane_source_order() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for len in [0, 1, 2, 3, 4, 5, 7, 8, 9, 31, 255, 256, 269] {
            for offset in 0..2 {
                let a: Vec<_> = (0..len + offset)
                    .map(|i| if i % 7 == 0 { 0 } else { (i * 173 + 1) as u64 })
                    .collect();
                let b: Vec<_> = (0..len + 3 + offset)
                    .map(|i| if i % 11 == 0 { 0 } else { (i * 19 + 3) as u64 })
                    .collect();
                let a = &a[offset..];
                let b = &b[offset..];
                for (a, b) in [(a, b), (b, a), (a, a), (a, &[][..])] {
                    let total = a.iter().chain(b).sum();
                    let coding_total = b.iter().sum();
                    let entropy_expected = entropy_oracle(a, b, total).to_bits();
                    let coding_expected = coding_oracle(a, b, coding_total).to_bits();
                    // SAFETY: NEON checked; generated totals fit signed i32.
                    unsafe {
                        assert_eq!(entropy(a, b, total).to_bits(), entropy_expected);
                        assert_eq!(coding_cost(a, b, coding_total).to_bits(), coding_expected);
                    }
                    assert_eq!(
                        super::super::combined_entropy(a, b, total).to_bits(),
                        entropy_expected
                    );
                    assert_eq!(
                        super::super::coding_cost(a, b, coding_total).to_bits(),
                        coding_expected
                    );
                }
                let total = a.iter().sum();
                assert_eq!(
                    super::super::entropy(a, total).to_bits(),
                    entropy_oracle(a, &[], total).to_bits()
                );
            }
        }
    }

    #[test]
    fn histograms_cover_float_rounding_zero_and_missing_support() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        for (a, b) in [
            (vec![], vec![]),
            (vec![0; 9], vec![]),
            (vec![0, 1], vec![1]),
            (vec![0, 1], vec![0, 1]),
            (vec![0, i32::MAX as u64], vec![]),
            (vec![(1 << 25) - 1, 1], vec![]),
            (vec![1 << 24, 1, 3, 5, 7], vec![1, 2, 3]),
            (vec![1 << 24, 1, 3, 5, 7], vec![3, 2, 7, 1, 13]),
        ] {
            let total = a.iter().chain(&b).sum();
            let coding_total = b.iter().sum();
            // SAFETY: NEON checked; the listed histograms fit source limits.
            unsafe {
                assert_eq!(
                    entropy(&a, &b, total).to_bits(),
                    entropy_oracle(&a, &b, total).to_bits()
                );
                assert_eq!(
                    coding_cost(&a, &b, coding_total).to_bits(),
                    coding_oracle(&a, &b, coding_total).to_bits()
                );
            }
        }
    }
}
