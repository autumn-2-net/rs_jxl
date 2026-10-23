// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Native AVX2 input/YCoCg/gradient kernels, fused prediction/prefix
//! counting, scalar fallback, and source chunk widths.

use super::{Backend, ImageView, entropy::PrefixCode, pixels};
use crate::{Result, bit_writer::BitWriter};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[path = "input_simd.rs"]
mod input_simd;

#[derive(Clone, Copy, Debug)]
pub(super) struct Kernel {
    avx2: bool,
}

impl Kernel {
    pub fn choose(backend: Backend) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let avx2 = backend == Backend::Auto && std::is_x86_feature_detected!("avx2");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let avx2 = {
            let _ = backend;
            false
        };
        Self { avx2 }
    }

    pub fn chunk_size(self) -> usize {
        if self.avx2 { 16 } else { 8 }
    }

    /// Load a group row from the caller's validated image/region view.
    /// Preserve the left border, unused channels and samples outside the area.
    pub fn load_row(
        self,
        image: ImageView<'_>,
        x: usize,
        y: usize,
        width: usize,
        rows: &mut [[i32; 257]; 4],
    ) -> Result<()> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 {
            // SAFETY: runtime AVX2 gate; the helper checks format and all
            // address bounds before mutation, and never loads past the tail.
            return unsafe { input_simd::load_row(image, x, y, width, rows) };
        }
        for (c, row) in rows
            .iter_mut()
            .enumerate()
            .take(usize::from(image.channels))
        {
            for (dx, value) in row[1..1 + width].iter_mut().enumerate() {
                *value = i32::from(image.sample(x + dx, y, c));
            }
        }
        Ok(())
    }

    /// Source FillRow: deinterleave and transform RGB/RGBA before row stores.
    pub fn load_transformed_row(
        self,
        image: ImageView<'_>,
        x: usize,
        y: usize,
        width: usize,
        rows: &mut [[i32; 257]; 4],
    ) -> Result<()> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 {
            // SAFETY: runtime AVX2 gate; helper checks format/address bounds
            // before stores and preserves source depth/endian conversion order.
            return unsafe { input_simd::load_transformed_row(image, x, y, width, rows) };
        }
        self.load_row(image, x, y, width, rows)?;
        if image.channels >= 3 {
            self.ycocg(rows, width);
        }
        Ok(())
    }

    pub fn literals(
        self,
        code: &PrefixCode,
        residuals: &[u32],
        writer: &mut BitWriter,
    ) -> Result<()> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 {
            // SAFETY: this kernel exists only after runtime AVX2 detection.
            // The callee checks the slice length and residual range before loads.
            let packed = unsafe { super::entropy_simd::pack(residuals, &code.lookup) }?;
            return packed.write(writer);
        }
        for &value in residuals {
            code.literal(value, writer)?;
        }
        Ok(())
    }

    pub fn ycocg(self, rows: &mut [[i32; 257]; 4], width: usize) {
        #[allow(unused_mut)] // Non-x86 targets have no native branch.
        let mut start = 0;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 {
            // SAFETY: runtime AVX2 gate; all three rows have width <= 256.
            start = unsafe { x86::ycocg(rows, width) };
        }
        let [red, green, blue, _] = rows;
        for ((r, g), b) in red[start + 1..width + 1]
            .iter_mut()
            .zip(&mut green[start + 1..width + 1])
            .zip(&mut blue[start + 1..width + 1])
        {
            let co = *r - *b;
            let tmp = *b + (co >> 1);
            let cg = *g - tmp;
            *r = tmp + (cg >> 1);
            *g = co;
            *b = cg;
        }
    }

    #[cfg(test)] // Previous widened kernel remains an independent regression baseline.
    pub fn predict(
        self,
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
        narrow: bool,
        out: &mut [u32],
    ) {
        debug_assert_eq!(row.len(), out.len());
        debug_assert!(
            left.len() >= out.len() && top.len() >= out.len() && topleft.len() >= out.len()
        );
        #[allow(unused_mut)] // Non-x86 targets have no native branch.
        let mut start = 0;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 {
            // SAFETY: runtime AVX2 gate, matching slice lengths, unaligned loads.
            start = unsafe { x86::predict(row, left, top, topleft, narrow, out) };
        }
        for i in start..out.len() {
            out[i] = pixels::predict(row[i], left[i], top[i], topleft[i], narrow);
        }
    }

    /// Source PredictPixels<SIMDVec16> and Mask16::CountPrefix. Inputs retain
    /// the existing i32 storage but must represent signed i16 sample lanes.
    /// The scalar fallback/tail emulates packed arithmetic, including residual
    /// subtraction; source scalar's promoted residual agrees on valid images.
    pub fn predict_chunk_narrow(
        self,
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
        out: &mut [u32],
    ) -> usize {
        assert!(out.len() <= 16);
        assert!(row.len() >= out.len() && left.len() >= out.len());
        assert!(top.len() >= out.len() && topleft.len() >= out.len());
        debug_assert!([row, left, top, topleft].iter().all(|values| {
            values[..out.len()]
                .iter()
                .all(|&v| i16::try_from(v).is_ok())
        }));
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 && out.len() == 16 {
            // SAFETY: runtime AVX2 gate and sixteen readable/writable elements.
            // Packing signed i16 values from i32 storage cannot saturate.
            return unsafe { x86::predict_chunk_narrow(row, left, top, topleft, out) };
        }
        let mut prefix = 0;
        for i in 0..out.len() {
            let (px, a, b, c) = (
                row[i] as i16,
                left[i] as i16,
                top[i] as i16,
                topleft[i] as i16,
            );
            let ac = a.wrapping_sub(c);
            let ab = a.wrapping_sub(b);
            let bc = b.wrapping_sub(c);
            let clamp = if ab ^ bc < 0 { b } else { a };
            let pred = if ac ^ bc < 0 {
                ac.wrapping_add(b)
            } else {
                clamp
            };
            let residual = px.wrapping_sub(pred);
            out[i] = u32::from(((residual as u16) << 1) ^ ((residual >> 15) as u16));
            if prefix == i && out[i] == 0 {
                prefix += 1;
            }
        }
        prefix
    }

    /// Source PredictPixels<SIMDVec32> plus ProcessChunk's prefix accumulation.
    /// Wide profiles use two eight-lane vectors per sixteen-sample AVX2 chunk.
    pub fn predict_chunk_wide(
        self,
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
        out: &mut [u32],
    ) -> usize {
        assert!(out.len() <= 16);
        assert!(row.len() >= out.len() && left.len() >= out.len());
        assert!(top.len() >= out.len() && topleft.len() >= out.len());
        #[allow(unused_mut)] // Non-x86 targets use the same scalar fallback.
        let mut start = 0;
        let mut prefix = 0;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.avx2 {
            // SAFETY: runtime AVX2 gate and checked input lengths. The callee
            // only accesses complete vectors; the tail below needs no padding.
            prefix = unsafe { x86::predict_chunk_wide(row, left, top, topleft, out) };
            start = out.len() / 8 * 8;
        }
        for i in start..out.len() {
            out[i] = pixels::predict(row[i], left[i], top[i], topleft[i], false);
            if prefix == i && out[i] == 0 {
                prefix += 1;
            }
        }
        prefix
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ycocg(rows: &mut [[i32; 257]; 4], width: usize) -> usize {
        let end = width / 8 * 8;
        // SAFETY: caller guarantees AVX2; each load/store covers eight elements
        // within initialized rows. Rows are disjoint and no pointers escape.
        unsafe {
            for x in (1..=end).step_by(8) {
                let r = _mm256_loadu_si256(rows[0].as_ptr().add(x).cast());
                let g = _mm256_loadu_si256(rows[1].as_ptr().add(x).cast());
                let b = _mm256_loadu_si256(rows[2].as_ptr().add(x).cast());
                let co = _mm256_sub_epi32(r, b);
                let tmp = _mm256_add_epi32(b, _mm256_srai_epi32::<1>(co));
                let cg = _mm256_sub_epi32(g, tmp);
                let y = _mm256_add_epi32(tmp, _mm256_srai_epi32::<1>(cg));
                _mm256_storeu_si256(rows[0].as_mut_ptr().add(x).cast(), y);
                _mm256_storeu_si256(rows[1].as_mut_ptr().add(x).cast(), co);
                _mm256_storeu_si256(rows[2].as_mut_ptr().add(x).cast(), cg);
            }
        }
        end
    }

    #[cfg(test)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn predict(
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
        narrow: bool,
        out: &mut [u32],
    ) -> usize {
        let end = out.len() / 8 * 8;
        // SAFETY: caller guarantees AVX2 and all input lengths >= out.len().
        // Full eight-lane blocks only; scalar handles tails without overreads.
        unsafe {
            for x in (0..end).step_by(8) {
                let px = _mm256_loadu_si256(row.as_ptr().add(x).cast());
                let a = _mm256_loadu_si256(left.as_ptr().add(x).cast());
                let b = _mm256_loadu_si256(top.as_ptr().add(x).cast());
                let c = _mm256_loadu_si256(topleft.as_ptr().add(x).cast());
                let mut ac = _mm256_sub_epi32(a, c);
                let mut ab = _mm256_sub_epi32(a, b);
                let mut bc = _mm256_sub_epi32(b, c);
                if narrow {
                    ac = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(ac));
                    ab = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(ab));
                    bc = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(bc));
                }
                let mut grad = _mm256_add_epi32(ac, b);
                if narrow {
                    grad = _mm256_srai_epi32::<16>(_mm256_slli_epi32::<16>(grad));
                }
                let d = _mm256_srai_epi32::<31>(_mm256_xor_si256(ab, bc));
                let s = _mm256_srai_epi32::<31>(_mm256_xor_si256(ac, bc));
                let clamp = _mm256_blendv_epi8(a, b, d);
                let pred = _mm256_blendv_epi8(clamp, grad, s);
                let diff = _mm256_sub_epi32(px, pred);
                let mut packed =
                    _mm256_xor_si256(_mm256_slli_epi32::<1>(diff), _mm256_srai_epi32::<31>(diff));
                if narrow {
                    packed = _mm256_and_si256(packed, _mm256_set1_epi32(65535));
                }
                _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), packed);
            }
        }
        end
    }

    #[target_feature(enable = "avx2")]
    unsafe fn load_narrow(values: &[i32]) -> __m256i {
        // SAFETY: caller supplies sixteen signed-i16 values in i32 storage.
        // Signed packing is exact; 0xd8 restores order after lane-local packs.
        unsafe {
            let low = _mm256_loadu_si256(values.as_ptr().cast());
            let high = _mm256_loadu_si256(values.as_ptr().add(8).cast());
            _mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(low, high))
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn predict_chunk_narrow(
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
        out: &mut [u32],
    ) -> usize {
        // SAFETY: caller checks AVX2 and sixteen input/output cells. Arithmetic
        // is the source packed-i16 sequence; residuals widen as UNSIGNED words.
        unsafe {
            let px = load_narrow(row);
            let left = load_narrow(left);
            let top = load_narrow(top);
            let topleft = load_narrow(topleft);
            let zero = _mm256_setzero_si256();
            let ac = _mm256_sub_epi16(left, topleft);
            let ab = _mm256_sub_epi16(left, top);
            let bc = _mm256_sub_epi16(top, topleft);
            let grad = _mm256_add_epi16(ac, top);
            let d = _mm256_cmpgt_epi16(zero, _mm256_xor_si256(ab, bc));
            let clamp = _mm256_blendv_epi8(left, top, d);
            let s = _mm256_cmpgt_epi16(zero, _mm256_xor_si256(ac, bc));
            let pred = _mm256_blendv_epi8(clamp, grad, s);
            let residual = _mm256_sub_epi16(px, pred);
            let twice = _mm256_add_epi16(residual, residual);
            let packed = _mm256_blendv_epi8(
                twice,
                _mm256_sub_epi16(_mm256_set1_epi16(-1), twice),
                _mm256_cmpgt_epi16(zero, residual),
            );
            let low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(packed));
            let high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(packed));
            _mm256_storeu_si256(out.as_mut_ptr().cast(), low);
            _mm256_storeu_si256(out.as_mut_ptr().add(8).cast(), high);
            // Zero-extend the 32 byte-sign bits BEFORE inversion: the sentinel
            // bit at position 32 makes an all-zero vector's prefix sixteen.
            let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi16(packed, zero)) as u32;
            ((!u64::from(mask)).trailing_zeros() / 2) as usize
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn predict_chunk_wide(
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
        out: &mut [u32],
    ) -> usize {
        let end = out.len() / 8 * 8;
        let mut prefix = 0;
        // SAFETY: caller checked AVX2 and all slice lengths. Only complete
        // eight-element vectors are loaded/stored, without source spill padding.
        unsafe {
            let zero = _mm256_setzero_si256();
            for x in (0..end).step_by(8) {
                let px = _mm256_loadu_si256(row.as_ptr().add(x).cast());
                let left = _mm256_loadu_si256(left.as_ptr().add(x).cast());
                let top = _mm256_loadu_si256(top.as_ptr().add(x).cast());
                let topleft = _mm256_loadu_si256(topleft.as_ptr().add(x).cast());
                let ac = _mm256_sub_epi32(left, topleft);
                let ab = _mm256_sub_epi32(left, top);
                let bc = _mm256_sub_epi32(top, topleft);
                let grad = _mm256_add_epi32(ac, top);
                let d = _mm256_cmpgt_epi32(zero, _mm256_xor_si256(ab, bc));
                let clamp = _mm256_blendv_epi8(left, top, d);
                let s = _mm256_cmpgt_epi32(zero, _mm256_xor_si256(ac, bc));
                let pred = _mm256_blendv_epi8(clamp, grad, s);
                let residual = _mm256_sub_epi32(px, pred);
                let twice = _mm256_add_epi32(residual, residual);
                let packed = _mm256_blendv_epi8(
                    twice,
                    _mm256_sub_epi32(_mm256_set1_epi32(-1), twice),
                    _mm256_cmpgt_epi32(zero, residual),
                );
                _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), packed);
                // Source Mask32::CountPrefix widens the eight-bit mask before
                // inversion, so an all-zero vector has prefix eight, not 64.
                let mask =
                    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(packed, zero))) as u8;
                if prefix == x {
                    prefix += (!u64::from(mask)).trailing_zeros() as usize;
                }
            }
        }
        prefix
    }
}
