// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Native AVX2 input deinterleaving from frozen libjxl
//! b866b3defec13494bfbaf705db93581cda118e37, enc_fast_lossless.cc:
//! SIMDVec16::LoadG/GA/RGB/RGBA8/16 (1666..1830), SwapEndian (1831),
//! Upcast (1655), FillRow (3193..3413), and StoreYCoCg (3272..3310).
//!
//! Kernel's detected AVX2 branch can call load_transformed_row for source
//! FillRow output: G/GA unchanged, RGB/RGBA converted to Y/Co/Cg[/A] before
//! storing. Do not apply another YCoCg pass to that output. load_row retains
//! its raw unsigned output for callers that need a separate color transform.
//! Palette input bypasses both entries; dispatch belongs to the parent module.
//!
//! Source lanes contain unsigned sample words. Upcast zero-extends, including
//! 0x8000..0xffff; the packus inputs are masked/shifted into 0..65535, so their
//! saturation does not change bits. RGB16 separates low/high bytes before
//! applying the RGB8 shuffle independently in each 128-bit lane. Permute 0xd8
//! restores pixel order after lane-local packs. Endian swap is after separation
//! and before color conversion/widening. At depths <=14, StoreYCoCg uses packed
//! wrapping i16 add/subtract and arithmetic shifts; Co/Cg are SIGN-extended to
//! the Rust i32 rows. At depths 15/16, unsigned input words are widened FIRST,
//! then transformed in i32 lanes. Y and alpha remain unsigned sample values.
//! Prediction remains a separate source stage. There is no FP or FMA here.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::ImageView;
use crate::{Error, Result};

/// Decode one bounded interleaved row segment into unsigned i32 sample words.
/// Only `rows[..channels][1..1+width]` change; border cells, unused channels and
/// the remainder of each row are untouched. Width zero is accepted.
///
/// All input/output address bounds and the channel/depth format are checked
/// before any destination mutation. The caller's existing per-region sample
/// validation still checks values against a narrow declared bit depth: this
/// loader preserves the stored word and does not clamp or mask sample values.
/// No alignment or accessible padding beyond the selected segment is required.
///
/// # Safety
/// The executing CPU must support AVX2, as established by Kernel dispatch.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn load_row(
    image: ImageView<'_>,
    x: usize,
    y: usize,
    width: usize,
    rows: &mut [[i32; 257]; 4],
) -> Result<()> {
    let input = checked_row(image, x, y, width)?;
    // SAFETY: the caller establishes AVX2; checked_row proves the entire
    // segment is readable and width <= 256. load_chunks uses full blocks only.
    unsafe {
        match (image.channels, image.bit_depth > 8) {
            (1, false) => load_chunks::<1, false>(input, width, image.big_endian, rows),
            (1, true) => load_chunks::<1, true>(input, width, image.big_endian, rows),
            (2, false) => load_chunks::<2, false>(input, width, image.big_endian, rows),
            (2, true) => load_chunks::<2, true>(input, width, image.big_endian, rows),
            (3, false) => load_chunks::<3, false>(input, width, image.big_endian, rows),
            (3, true) => load_chunks::<3, true>(input, width, image.big_endian, rows),
            (4, false) => load_chunks::<4, false>(input, width, image.big_endian, rows),
            (4, true) => load_chunks::<4, true>(input, width, image.big_endian, rows),
            _ => unreachable!("checked channel count"),
        }
    }
    Ok(())
}

/// Load source FillRow output directly, with RGB/RGBA transformed in registers.
/// Channels are G, G/A, Y/Co/Cg, or Y/Co/Cg/A; alpha is not transformed.
/// Only `rows[..channels][1..1+width]` change. Width zero is accepted.
///
/// Like load_row, checks format and all address bounds before mutation, needs
/// no padding/alignment, and leaves sample-range validation to the caller's
/// existing region validator. Source narrow arithmetic requires samples to fit
/// the declared bit depth; this helper does not clamp invalid stored values.
/// Do not call the separate YCoCg stage on this output.
///
/// # Safety
/// The executing CPU must support AVX2, as established by Kernel dispatch.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn load_transformed_row(
    image: ImageView<'_>,
    x: usize,
    y: usize,
    width: usize,
    rows: &mut [[i32; 257]; 4],
) -> Result<()> {
    let input = checked_row(image, x, y, width)?;
    // SAFETY: AVX2 is a caller precondition. checked_row proves width <= 256
    // and the readable segment; each specialization uses complete blocks only.
    unsafe {
        match (image.channels, image.bit_depth) {
            (1, 1..=8) => load_chunks::<1, false>(input, width, image.big_endian, rows),
            (1, 9..=16) => load_chunks::<1, true>(input, width, image.big_endian, rows),
            (2, 1..=8) => load_chunks::<2, false>(input, width, image.big_endian, rows),
            (2, 9..=16) => load_chunks::<2, true>(input, width, image.big_endian, rows),
            (3, 1..=8) => {
                load_transformed_chunks::<3, false, true>(input, width, image.big_endian, rows)
            }
            (3, 9..=14) => {
                load_transformed_chunks::<3, true, true>(input, width, image.big_endian, rows)
            }
            (3, 15..=16) => {
                load_transformed_chunks::<3, true, false>(input, width, image.big_endian, rows)
            }
            (4, 1..=8) => {
                load_transformed_chunks::<4, false, true>(input, width, image.big_endian, rows)
            }
            (4, 9..=14) => {
                load_transformed_chunks::<4, true, true>(input, width, image.big_endian, rows)
            }
            (4, 15..=16) => {
                load_transformed_chunks::<4, true, false>(input, width, image.big_endian, rows)
            }
            _ => unreachable!("checked channel count and bit depth"),
        }
    }
    Ok(())
}

fn checked_row(image: ImageView<'_>, x: usize, y: usize, width: usize) -> Result<&[u8]> {
    if !(1..=4).contains(&image.channels)
        || !(1..=16).contains(&image.bit_depth)
        || width > 256
        || y >= image.height
        || x.checked_add(width).is_none_or(|end| end > image.width)
    {
        return Err(Error::InvalidInput(
            "FJXL input row geometry or format".into(),
        ));
    }
    let pixel_size = usize::from(image.channels) * if image.bit_depth <= 8 { 1 } else { 2 };
    let full_row = image
        .width
        .checked_mul(pixel_size)
        .ok_or(Error::Limit("FJXL input row size"))?;
    if image.row_stride < full_row {
        return Err(Error::InvalidInput(
            "FJXL input row stride is too short".into(),
        ));
    }
    let start = y
        .checked_mul(image.row_stride)
        .and_then(|base| {
            x.checked_mul(pixel_size)
                .and_then(|offset| base.checked_add(offset))
        })
        .ok_or(Error::Limit("FJXL input row offset"))?;
    let end = start
        .checked_add(width * pixel_size)
        .ok_or(Error::Limit("FJXL input row end"))?;
    image
        .pixels
        .get(start..end)
        .ok_or_else(|| Error::InvalidInput("FJXL input row buffer is too short".into()))
}

#[target_feature(enable = "avx2")]
unsafe fn load_chunks<const CHANNELS: usize, const WIDE: bool>(
    input: &[u8],
    width: usize,
    big_endian: bool,
    rows: &mut [[i32; 257]; 4],
) {
    let sample_size = if WIDE { 2 } else { 1 };
    let pixel_size = CHANNELS * sample_size;
    let end = width / 16 * 16;
    for x in (0..end).step_by(16) {
        // SAFETY: these private instantiations have CHANNELS 1..=4; this block
        // reads exactly 16 * pixel_size bytes and stores 16 values per channel.
        // Checked input and fixed destination lengths cover both in full.
        unsafe {
            let mut channels = load_block::<CHANNELS, WIDE>(input.as_ptr().add(x * pixel_size));
            for (channel, row) in channels.iter_mut().zip(rows.iter_mut()).take(CHANNELS) {
                if WIDE && big_endian {
                    *channel = swap_endian(*channel);
                }
                store_widened(*channel, row.as_mut_ptr().add(1 + x));
            }
        }
    }
    // Source FillRow tails: read only each remaining sample, never a padded
    // vector or the next row. U8 ignores the endian flag, as in the source.
    for x in end..width {
        for (c, row) in rows.iter_mut().enumerate().take(CHANNELS) {
            let at = x * pixel_size + c * sample_size;
            row[x + 1] = if WIDE {
                let pair = [input[at], input[at + 1]];
                i32::from(if big_endian {
                    u16::from_be_bytes(pair)
                } else {
                    u16::from_le_bytes(pair)
                })
            } else {
                i32::from(input[at])
            };
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn load_transformed_chunks<const CHANNELS: usize, const WIDE: bool, const NARROW: bool>(
    input: &[u8],
    width: usize,
    big_endian: bool,
    rows: &mut [[i32; 257]; 4],
) {
    let sample_size = if WIDE { 2 } else { 1 };
    let pixel_size = CHANNELS * sample_size;
    let end = width / 16 * 16;
    for x in (0..end).step_by(16) {
        // SAFETY: only RGB/RGBA specializations reach here. Each load consumes
        // exactly sixteen pixels; all stores lie within rows[..][1..=width].
        unsafe {
            let mut channels = load_block::<CHANNELS, WIDE>(input.as_ptr().add(x * pixel_size));
            if WIDE && big_endian {
                for channel in channels.iter_mut().take(CHANNELS) {
                    *channel = swap_endian(*channel);
                }
            }
            let [r, g, b, a] = channels;
            if NARROW {
                store_ycocg_narrow(r, g, b, rows, x + 1);
            } else {
                store_ycocg_wide(r, g, b, rows, x + 1);
            }
            if CHANNELS == 4 {
                store_widened(a, rows[3].as_mut_ptr().add(x + 1));
            }
        }
    }
    // Source FillRow scalar tail. All valid <=14-bit intermediates fit i16;
    // using i32 here also covers the source's 15/16-bit pixel_t specialization.
    for x in end..width {
        let mut channels = [0i32; 4];
        for (c, value) in channels.iter_mut().enumerate().take(CHANNELS) {
            let at = x * pixel_size + c * sample_size;
            *value = if WIDE {
                let pair = [input[at], input[at + 1]];
                i32::from(if big_endian {
                    u16::from_be_bytes(pair)
                } else {
                    u16::from_le_bytes(pair)
                })
            } else {
                i32::from(input[at])
            };
        }
        let [r, g, b, a] = channels;
        let co = r - b;
        let tmp = b + (co >> 1);
        let cg = g - tmp;
        rows[0][x + 1] = tmp + (cg >> 1);
        rows[1][x + 1] = co;
        rows[2][x + 1] = cg;
        if CHANNELS == 4 {
            rows[3][x + 1] = a;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_ycocg_narrow(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    rows: &mut [[i32; 257]; 4],
    x: usize,
) {
    let co = _mm256_sub_epi16(r, b);
    let tmp = _mm256_add_epi16(b, _mm256_srai_epi16::<1>(co));
    let cg = _mm256_sub_epi16(g, tmp);
    let y = _mm256_add_epi16(tmp, _mm256_srai_epi16::<1>(cg));
    // SAFETY: caller provides sixteen writable cells per row. Source stores
    // i16 words; this adapter must retain negative chroma in Rust's i32 rows.
    unsafe {
        store_widened(y, rows[0].as_mut_ptr().add(x));
        store_signed_widened(co, rows[1].as_mut_ptr().add(x));
        store_signed_widened(cg, rows[2].as_mut_ptr().add(x));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_ycocg_wide(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    rows: &mut [[i32; 257]; 4],
    x: usize,
) {
    // SAFETY: caller establishes AVX2 and sixteen writable cells per row.
    // Upcast is unsigned, including 0x8000..0xffff, BEFORE subtraction.
    unsafe {
        let [r_low, r_high] = upcast(r);
        let [g_low, g_high] = upcast(g);
        let [b_low, b_high] = upcast(b);
        for (offset, (r, g, b)) in [(r_low, g_low, b_low), (r_high, g_high, b_high)]
            .into_iter()
            .enumerate()
        {
            let co = _mm256_sub_epi32(r, b);
            let tmp = _mm256_add_epi32(b, _mm256_srai_epi32::<1>(co));
            let cg = _mm256_sub_epi32(g, tmp);
            let y = _mm256_add_epi32(tmp, _mm256_srai_epi32::<1>(cg));
            let at = x + 8 * offset;
            _mm256_storeu_si256(rows[0].as_mut_ptr().add(at).cast(), y);
            _mm256_storeu_si256(rows[1].as_mut_ptr().add(at).cast(), co);
            _mm256_storeu_si256(rows[2].as_mut_ptr().add(at).cast(), cg);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_block<const CHANNELS: usize, const WIDE: bool>(data: *const u8) -> [__m256i; 4] {
    let zero = _mm256_setzero_si256();
    // SAFETY: the chunk callers prove a complete 16-pixel block; the largest load
    // ends at data + 16 * CHANNELS * sample_size. All loads are unaligned.
    unsafe {
        match (CHANNELS, WIDE) {
            (1, false) => [load_g8(data), zero, zero, zero],
            (1, true) => [load_g16(data), zero, zero, zero],
            (2, false) => {
                let [g, a] = load_ga8(data);
                [g, a, zero, zero]
            }
            (2, true) => {
                let [g, a] = load_ga16(data);
                [g, a, zero, zero]
            }
            (3, false) => {
                let [r, g, b] = load_rgb8(data);
                [r, g, b, zero]
            }
            (3, true) => {
                let [r, g, b] = load_rgb16(data);
                [r, g, b, zero]
            }
            (4, false) => load_rgba8(data),
            (4, true) => load_rgba16(data),
            _ => unreachable!("private channel specialization"),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_g8(data: *const u8) -> __m256i {
    // SAFETY: the complete G8 block contains exactly 16 bytes.
    let bytes = unsafe { _mm_loadu_si128(data.cast()) };
    _mm256_cvtepu8_epi16(bytes)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_g16(data: *const u8) -> __m256i {
    // SAFETY: the complete G16 block contains exactly 32 bytes.
    unsafe { _mm256_loadu_si256(data.cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_ga8(data: *const u8) -> [__m256i; 2] {
    // SAFETY: the complete GA8 block contains exactly 32 bytes.
    let bytes = unsafe { _mm256_loadu_si256(data.cast()) };
    [
        _mm256_and_si256(bytes, _mm256_set1_epi16(0xff)),
        _mm256_srli_epi16::<8>(bytes),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_ga16(data: *const u8) -> [__m256i; 2] {
    // SAFETY: two 32-byte loads fit exactly in the 64-byte GA16 block.
    let (first, second) = unsafe {
        (
            _mm256_loadu_si256(data.cast()),
            _mm256_loadu_si256(data.add(32).cast()),
        )
    };
    let mask = _mm256_set1_epi32(0xffff);
    let gray = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(
        _mm256_and_si256(first, mask),
        _mm256_and_si256(second, mask),
    ));
    let alpha = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(
        _mm256_srli_epi32::<16>(first),
        _mm256_srli_epi32::<16>(second),
    ));
    [gray, alpha]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_rgb8(data: *const u8) -> [__m256i; 3] {
    // SAFETY: three 16-byte loads fit exactly in the 48-byte RGB8 block.
    let (bytes0, bytes1, bytes2) = unsafe {
        (
            _mm_loadu_si128(data.cast()),
            _mm_loadu_si128(data.add(16).cast()),
            _mm_loadu_si128(data.add(32).cast()),
        )
    };
    let idx = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
    let r6b5g5_0 = _mm_shuffle_epi8(bytes0, idx);
    let g6r5b5_1 = _mm_shuffle_epi8(bytes1, idx);
    let b6g5r5_2 = _mm_shuffle_epi8(bytes2, idx);
    let mask010 = _mm_setr_epi8(0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0);
    let mask001 = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1);
    let b2g2b1 = _mm_blendv_epi8(b6g5r5_2, g6r5b5_1, mask001);
    let b2b0b1 = _mm_blendv_epi8(b2g2b1, r6b5g5_0, mask010);
    let r0r1b1 = _mm_blendv_epi8(r6b5g5_0, g6r5b5_1, mask010);
    let r0r1r2 = _mm_blendv_epi8(r0r1b1, b6g5r5_2, mask001);
    let g1r1g0 = _mm_blendv_epi8(g6r5b5_1, r6b5g5_0, mask001);
    let g1g2g0 = _mm_blendv_epi8(g1r1g0, b6g5r5_2, mask010);
    let g0g1g2 = _mm_alignr_epi8::<11>(g1g2g0, g1g2g0);
    let b0b1b2 = _mm_alignr_epi8::<6>(b2b0b1, b2b0b1);
    [
        _mm256_cvtepu8_epi16(r0r1r2),
        _mm256_cvtepu8_epi16(g0g1g2),
        _mm256_cvtepu8_epi16(b0b1b2),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn split_low_high(data: *const u8) -> __m256i {
    // SAFETY: load_rgb16 provides a complete 32-byte portion of its block.
    let bytes = unsafe { _mm256_loadu_si256(data.cast()) };
    let low = _mm256_and_si256(bytes, _mm256_set1_epi16(0xff));
    let high = _mm256_srli_epi16::<8>(bytes);
    _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(low, high))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn combine_low_high(value: __m256i) -> __m256i {
    let low = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<0>(value));
    let high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(value));
    _mm256_or_si256(_mm256_slli_epi16::<8>(high), low)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_rgb16(data: *const u8) -> [__m256i; 3] {
    // SAFETY: the three split loads cover exactly 96 bytes; AVX2 is enabled.
    let (bytes0, bytes1, bytes2) = unsafe {
        (
            split_low_high(data),
            split_low_high(data.add(32)),
            split_low_high(data.add(64)),
        )
    };
    let idx = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
    ));
    let r6b5g5_0 = _mm256_shuffle_epi8(bytes0, idx);
    let g6r5b5_1 = _mm256_shuffle_epi8(bytes1, idx);
    let b6g5r5_2 = _mm256_shuffle_epi8(bytes2, idx);
    let mask010 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0,
    ));
    let mask001 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1,
    ));
    let b2g2b1 = _mm256_blendv_epi8(b6g5r5_2, g6r5b5_1, mask001);
    let b2b0b1 = _mm256_blendv_epi8(b2g2b1, r6b5g5_0, mask010);
    let r0r1b1 = _mm256_blendv_epi8(r6b5g5_0, g6r5b5_1, mask010);
    let r0r1r2 = _mm256_blendv_epi8(r0r1b1, b6g5r5_2, mask001);
    let g1r1g0 = _mm256_blendv_epi8(g6r5b5_1, r6b5g5_0, mask001);
    let g1g2g0 = _mm256_blendv_epi8(g1r1g0, b6g5r5_2, mask010);
    let g0g1g2 = _mm256_alignr_epi8::<11>(g1g2g0, g1g2g0);
    let b0b1b2 = _mm256_alignr_epi8::<6>(b2b0b1, b2b0b1);
    // SAFETY: these helpers only rearrange registers under the same AVX2 gate.
    unsafe {
        [
            combine_low_high(r0r1r2),
            combine_low_high(g0g1g2),
            combine_low_high(b0b1b2),
        ]
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_rgba8(data: *const u8) -> [__m256i; 4] {
    // SAFETY: two 32-byte loads fit exactly in the 64-byte RGBA8 block.
    let (first, second) = unsafe {
        (
            _mm256_loadu_si256(data.cast()),
            _mm256_loadu_si256(data.add(32).cast()),
        )
    };
    let mask = _mm256_set1_epi32(0xffff);
    let rg = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(
        _mm256_and_si256(first, mask),
        _mm256_and_si256(second, mask),
    ));
    let ba = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(
        _mm256_srli_epi32::<16>(first),
        _mm256_srli_epi32::<16>(second),
    ));
    let mask = _mm256_set1_epi16(0xff);
    [
        _mm256_and_si256(rg, mask),
        _mm256_srli_epi16::<8>(rg),
        _mm256_and_si256(ba, mask),
        _mm256_srli_epi16::<8>(ba),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pack_low32(a: __m256i, b: __m256i) -> __m256i {
    let mask = _mm256_set1_epi32(0xffff);
    _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(
        _mm256_and_si256(a, mask),
        _mm256_and_si256(b, mask),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pack_high32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi32(
        _mm256_srli_epi32::<16>(a),
        _mm256_srli_epi32::<16>(b),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_rgba16(data: *const u8) -> [__m256i; 4] {
    // SAFETY: four 32-byte loads fit exactly in the 128-byte RGBA16 block;
    // all packing helpers require only the AVX2 feature already enabled here.
    unsafe {
        let bytes0 = _mm256_loadu_si256(data.cast());
        let bytes1 = _mm256_loadu_si256(data.add(32).cast());
        let bytes2 = _mm256_loadu_si256(data.add(64).cast());
        let bytes3 = _mm256_loadu_si256(data.add(96).cast());
        let rb0 = pack_low32(bytes0, bytes1);
        let rb1 = pack_low32(bytes2, bytes3);
        let ga0 = pack_high32(bytes0, bytes1);
        let ga1 = pack_high32(bytes2, bytes3);
        [
            pack_low32(rb0, rb1),
            pack_low32(ga0, ga1),
            pack_high32(rb0, rb1),
            pack_high32(ga0, ga1),
        ]
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn swap_endian(value: __m256i) -> __m256i {
    let indices = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    ));
    _mm256_shuffle_epi8(value, indices)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn upcast(value: __m256i) -> [__m256i; 2] {
    let zero = _mm256_setzero_si256();
    let v02 = _mm256_unpacklo_epi16(value, zero);
    let v13 = _mm256_unpackhi_epi16(value, zero);
    [
        _mm256_permute2x128_si256::<0x20>(v02, v13),
        _mm256_permute2x128_si256::<0x31>(v02, v13),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_widened(value: __m256i, output: *mut i32) {
    // SAFETY: caller provides sixteen writable i32 cells and enables AVX2.
    // No alignment requirement; low/high are consecutive eight-pixel groups.
    unsafe {
        let [low, high] = upcast(value);
        _mm256_storeu_si256(output.cast(), low);
        _mm256_storeu_si256(output.add(8).cast(), high);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_signed_widened(value: __m256i, output: *mut i32) {
    let low = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(value));
    let high = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(value));
    // SAFETY: caller provides sixteen writable i32 cells; stores are unaligned.
    unsafe {
        _mm256_storeu_si256(output.cast(), low);
        _mm256_storeu_si256(output.add(8).cast(), high);
    }
}

#[cfg(test)]
#[path = "input_simd_tests.rs"]
mod tests;
