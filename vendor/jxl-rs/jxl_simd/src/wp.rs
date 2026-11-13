// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Four-lane helpers for the weighted predictor's error history.
//!
//! The weighted predictor keeps one i32 error per sub-predictor (four lanes).
//! These helpers apply the same wrapping arithmetic to all four lanes at once.
//! The lane count is fixed at four, so the x86-64 baseline SSE2 path needs no
//! runtime dispatch; non-x86 targets keep the scalar spelling. Wrapping is
//! preserved exactly: `_mm_add_epi32` is two's-complement addition mod 2^32,
//! which matches `i32::wrapping_add`.

/// `n[k] + ne[k] + nw[k]` (wrapping) for all four lanes.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn sum3(n: &[i32; 4], ne: &[i32; 4], nw: &[i32; 4]) -> [i32; 4] {
    // SAFETY: x86-64 guarantees SSE2. Each load/store touches exactly 16 bytes
    // of a `[i32; 4]`; unaligned access is explicitly allowed.
    unsafe {
        use std::arch::x86_64::*;
        let a = _mm_loadu_si128(n.as_ptr().cast());
        let b = _mm_loadu_si128(ne.as_ptr().cast());
        let c = _mm_loadu_si128(nw.as_ptr().cast());
        let sum = _mm_add_epi32(_mm_add_epi32(a, b), c);
        let mut out = [0i32; 4];
        _mm_storeu_si128(out.as_mut_ptr().cast(), sum);
        out
    }
}

/// `n[k] + ne[k] + nw[k]` (wrapping) for all four lanes (scalar fallback).
#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn sum3(n: &[i32; 4], ne: &[i32; 4], nw: &[i32; 4]) -> [i32; 4] {
    [
        n[0].wrapping_add(ne[0]).wrapping_add(nw[0]),
        n[1].wrapping_add(ne[1]).wrapping_add(nw[1]),
        n[2].wrapping_add(ne[2]).wrapping_add(nw[2]),
        n[3].wrapping_add(ne[3]).wrapping_add(nw[3]),
    ]
}

/// `dst[k] = dst[k] + err[k]` (wrapping) for all four lanes.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn accumulate(dst: &mut [i32; 4], err: &[i32; 4]) {
    // SAFETY: x86-64 guarantees SSE2. Each load/store touches exactly 16 bytes
    // of a `[i32; 4]`; unaligned access is explicitly allowed.
    unsafe {
        use std::arch::x86_64::*;
        let a = _mm_loadu_si128(dst.as_ptr().cast());
        let b = _mm_loadu_si128(err.as_ptr().cast());
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_add_epi32(a, b));
    }
}

/// `dst[k] = dst[k] + err[k]` (wrapping) for all four lanes (scalar fallback).
#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn accumulate(dst: &mut [i32; 4], err: &[i32; 4]) {
    dst[0] = dst[0].wrapping_add(err[0]);
    dst[1] = dst[1].wrapping_add(err[1]);
    dst[2] = dst[2].wrapping_add(err[2]);
    dst[3] = dst[3].wrapping_add(err[3]);
}

/// `((p[k] - value).abs() + round) >> shift` for all four lanes.
///
/// Only bit-exact for `0 <= round`, `shift < 32` and intermediates fitting
/// nonnegative `i32` after rounding. The modular cost walk checks the actual
/// transformed samples and predictions before selecting this kernel.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn err4(p: &[i32; 4], value: i32, round: i32, shift: u32) -> [i32; 4] {
    // SAFETY: x86-64 guarantees SSE2. Each access touches exactly 16 bytes of a
    // `[i32; 4]`; unaligned access is explicitly allowed. `_mm_srl_epi32` takes
    // the count in an xmm register, so the shift stays a runtime value.
    unsafe {
        use std::arch::x86_64::*;
        let a = _mm_loadu_si128(p.as_ptr().cast());
        let d = _mm_sub_epi32(a, _mm_set1_epi32(value));
        let sign = _mm_srai_epi32(d, 31);
        let abs = _mm_sub_epi32(_mm_xor_si128(d, sign), sign);
        let rounded = _mm_add_epi32(abs, _mm_set1_epi32(round));
        let shifted = _mm_srl_epi32(rounded, _mm_cvtsi32_si128(shift as i32));
        let mut out = [0i32; 4];
        _mm_storeu_si128(out.as_mut_ptr().cast(), shifted);
        out
    }
}

/// Scalar twin of [`err4`] for non-x86 targets.
#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn err4(p: &[i32; 4], value: i32, round: i32, shift: u32) -> [i32; 4] {
    p.map(|v| ((v - value).abs() + round) >> shift)
}

/// `n[k] + ne[k] + nw[k]` (wrapping) over `u32` lanes.
///
/// Bit-identical to [`sum3`]: wrapping addition and the `i32`/`u32` pair
/// reinterpretation agree on every bit pattern, so the same instruction serves
/// both spellings of the error history.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn sum3_u32(n: &[u32; 4], ne: &[u32; 4], nw: &[u32; 4]) -> [u32; 4] {
    // SAFETY: x86-64 guarantees SSE2. Each access touches exactly 16 bytes of a
    // `[u32; 4]`; unaligned access is explicitly allowed.
    unsafe {
        use std::arch::x86_64::*;
        let a = _mm_loadu_si128(n.as_ptr().cast());
        let b = _mm_loadu_si128(ne.as_ptr().cast());
        let c = _mm_loadu_si128(nw.as_ptr().cast());
        let sum = _mm_add_epi32(_mm_add_epi32(a, b), c);
        let mut out = [0u32; 4];
        _mm_storeu_si128(out.as_mut_ptr().cast(), sum);
        out
    }
}

/// Scalar twin of [`sum3_u32`] for non-x86 targets.
#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn sum3_u32(n: &[u32; 4], ne: &[u32; 4], nw: &[u32; 4]) -> [u32; 4] {
    [
        n[0].wrapping_add(ne[0]).wrapping_add(nw[0]),
        n[1].wrapping_add(ne[1]).wrapping_add(nw[1]),
        n[2].wrapping_add(ne[2]).wrapping_add(nw[2]),
        n[3].wrapping_add(ne[3]).wrapping_add(nw[3]),
    ]
}

/// `dst[k] = dst[k] + err[k]` (wrapping) over `u32` lanes.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn accumulate_u32(dst: &mut [u32; 4], err: &[u32; 4]) {
    // SAFETY: x86-64 guarantees SSE2. Each access touches exactly 16 bytes of a
    // `[u32; 4]`; unaligned access is explicitly allowed.
    unsafe {
        use std::arch::x86_64::*;
        let a = _mm_loadu_si128(dst.as_ptr().cast());
        let b = _mm_loadu_si128(err.as_ptr().cast());
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_add_epi32(a, b));
    }
}

/// Scalar twin of [`accumulate_u32`] for non-x86 targets.
#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
pub fn accumulate_u32(dst: &mut [u32; 4], err: &[u32; 4]) {
    dst[0] = dst[0].wrapping_add(err[0]);
    dst[1] = dst[1].wrapping_add(err[1]);
    dst[2] = dst[2].wrapping_add(err[2]);
    dst[3] = dst[3].wrapping_add(err[3]);
}
