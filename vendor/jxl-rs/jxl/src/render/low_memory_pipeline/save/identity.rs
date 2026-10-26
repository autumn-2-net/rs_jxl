// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::Range;

use jxl_simd::{
    F32SimdVec, I16SimdVec, I32SimdVec, SimdDescriptor, SimdMask, SimdMask16, U8SimdVec,
    U16SimdVec, simd_function,
};

use crate::api::{Endianness, JxlDataFormat, JxlOutputBuffer};
use crate::image::ImageDataType;
use crate::render::low_memory_pipeline::row_buffers::RowBuffer;
use crate::render::save::{SaveChannelType, SaveStage};
use crate::util::DitherTable;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn f32_to_u8_simd<D: SimdDescriptor>(
    d: D,
    input: &[f32],
    output: &mut [u8],
    max: f32,
    position: (usize, usize),
    channel: usize,
    xsize: usize,
    dither_table: &DitherTable,
) {
    let (x0, y0) = position;
    let simd_width = D::F32Vec::LEN;
    let zero = D::F32Vec::splat(d, 0.0);
    let scale = D::F32Vec::splat(d, max);
    let dither_y = (y0 + channel * 13) % 32;

    for (block, (input_chunk, output_chunk)) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
        .enumerate()
    {
        let x = block * simd_width;
        let val = D::F32Vec::load(d, input_chunk);
        let dither_x = (x0 + x + channel * 23) % 32;
        let dither = D::F32Vec::load(d, &dither_table[dither_y][dither_x..]);
        let scaled = val * scale;
        let dithered = scaled + dither;
        let clamped = dithered.max(zero).min(scale);
        clamped.round_store_u8(output_chunk);
    }
}

simd_function!(
    f32_to_u8,
    d: D,
    #[allow(clippy::too_many_arguments)]
    pub(super) fn f32_to_u8_impl(
        input: &[f32],
        output: &mut [u8],
        max: f32,
        position: (usize, usize),
        channel: usize,
        xsize: usize,
        dither_table: &DitherTable,
    ) {
        f32_to_u8_simd(d, input, output, max, position, channel, xsize, dither_table);
    }
);

#[inline(always)]
fn f32_to_u16_simd<D: SimdDescriptor>(
    d: D,
    input: &[f32],
    output: &mut [u16],
    max: f32,
    xsize: usize,
) {
    let simd_width = D::F32Vec::LEN;
    let zero = D::F32Vec::splat(d, 0.0);
    let one = D::F32Vec::splat(d, 1.0);
    let scale = D::F32Vec::splat(d, max);

    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::F32Vec::load(d, input_chunk);
        let clamped = val.max(zero).min(one);
        let scaled = clamped * scale;
        scaled.round_store_u16(output_chunk);
    }
}

simd_function!(
    f32_to_u16,
    d: D,
    pub(super) fn f32_to_u16_impl(input: &[f32], output: &mut [u16], max: f32, xsize: usize) {
        f32_to_u16_simd(d, input, output, max, xsize);
    }
);

#[inline(always)]
fn f32_to_f16_simd<D: SimdDescriptor>(d: D, input: &[f32], output: &mut [u16], xsize: usize) {
    let simd_width = D::F32Vec::LEN;
    let exponent_mask = D::I32Vec::splat(d, 0x7f80_0000);
    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::F32Vec::load(d, input_chunk);
        val.store_f16_bits(output_chunk);
        if !exponent_mask.gt(val.bitcast_to_i32() & exponent_mask).all() {
            // Preserve the sign, payload and quiet bit of expanded F16 NaNs.
            // Ordinary half conversion may quiet or canonicalize them.
            for (&sample, output) in input_chunk.iter().zip(output_chunk) {
                let bits = sample.to_bits();
                if bits & 0x7f80_0000 == 0x7f80_0000 {
                    let mantissa = bits & 0x007f_ffff;
                    let mut payload = (mantissa >> 13) as u16;
                    if mantissa != 0 && payload == 0 {
                        // A payload too small for F16 must not become infinity.
                        payload = 1;
                    }
                    *output = ((bits >> 16) as u16 & 0x8000) | 0x7c00 | payload;
                }
            }
        }
    }
}

simd_function!(
    f32_to_f16,
    d: D,
    pub(super) fn f32_to_f16_impl(input: &[f32], output: &mut [u16], xsize: usize) {
        f32_to_f16_simd(d, input, output, xsize);
    }
);

#[inline(always)]
fn i16_to_u8_simd<D: SimdDescriptor>(
    d: D,
    input: &[i16],
    output: &mut [u8],
    scale: i32,
    max: i32,
    xsize: usize,
) {
    let simd_width = D::I16Vec::LEN;
    let scale = D::I16Vec::splat(d, scale as i16);
    let max = D::I16Vec::splat(d, max as i16);
    let zero = D::I16Vec::splat(d, 0);

    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::I16Vec::load(d, input_chunk);
        let scaled = val * scale;
        let zeroclip = scaled.lt_zero().if_then_else_i16(zero, scaled);
        let clip = scaled.gt(max).if_then_else_i16(max, zeroclip);
        clip.store_u8(output_chunk);
    }
}

simd_function!(
    i16_to_u8,
    d: D,
    pub(super) fn i16_to_u8_impl(
        input: &[i16],
        output: &mut [u8],
        scale: i32,
        max: i32,
        xsize: usize,
    ) {
        i16_to_u8_simd(d, input, output, scale, max, xsize);
    }
);

#[inline(always)]
fn i16_to_f16_simd<D: SimdDescriptor>(
    d: D,
    input: &[i16],
    output: &mut [u16],
    scale: f32,
    xsize: usize,
) {
    let simd_width = D::F32Vec::LEN;
    let scale = D::F32Vec::splat(d, scale);
    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::I32Vec::load_from_i16(d, input_chunk).as_f32() * scale;
        val.store_f16_bits(output_chunk);
    }
}

simd_function!(
    i16_to_f16,
    d: D,
    pub(super) fn i16_to_f16_impl(input: &[i16], output: &mut [u16], scale: f32, xsize: usize) {
        i16_to_f16_simd(d, input, output, scale, xsize);
    }
);

#[inline(always)]
fn i32_to_u8_simd<D: SimdDescriptor>(
    d: D,
    input: &[i32],
    output: &mut [u8],
    scale: i32,
    max: i32,
    xsize: usize,
) {
    let simd_width = D::F32Vec::LEN;
    let scale = D::I32Vec::splat(d, scale);
    let max = D::I32Vec::splat(d, max);
    let zero = D::I32Vec::splat(d, 0);

    for (input_chunk, output_chunk) in input
        .chunks_exact(simd_width)
        .zip(output.chunks_exact_mut(simd_width))
        .take(xsize.div_ceil(simd_width))
    {
        let val = D::I32Vec::load(d, input_chunk);
        let scaled = val * scale;
        let zeroclip = scaled.lt_zero().if_then_else_i32(zero, scaled);
        let clip = scaled.gt(max).if_then_else_i32(max, zeroclip);
        clip.store_u8(output_chunk);
    }
}

simd_function!(
    i32_to_u8,
    d: D,
    pub(super) fn i32_to_u8_impl(
        input: &[i32],
        output: &mut [u8],
        scale: i32,
        max: i32,
        xsize: usize,
    ) {
        i32_to_u8_simd(d, input, output, scale, max, xsize);
    }
);

#[inline(always)]
fn load_f32_as_u8<D: SimdDescriptor>(
    d: D,
    src: &[f32],
    max: f32,
    pos: (usize, usize),
    c: usize,
    dither_table: &DitherTable,
) -> D::U8Vec {
    let len = D::U8Vec::LEN;
    let mut buf = [0u8; 64];
    f32_to_u8_simd(d, src, &mut buf[..len], max, pos, c, len, dither_table);
    D::U8Vec::load(d, &buf[..len])
}

#[inline(always)]
fn load_i16_as_u8<D: SimdDescriptor>(d: D, src: &[i16], scale: i32, max: i32) -> D::U8Vec {
    let len = D::U8Vec::LEN;
    let mut buf = [0u8; 64];
    i16_to_u8_simd(d, src, &mut buf[..len], scale, max, len);
    D::U8Vec::load(d, &buf[..len])
}

#[inline(always)]
fn load_f32_as_f16<D: SimdDescriptor>(d: D, src: &[f32]) -> D::U16Vec {
    let len = D::U16Vec::LEN;
    let mut buf = [0u16; 32];
    f32_to_f16_simd(d, src, &mut buf[..len], len);
    D::U16Vec::load(d, &buf[..len])
}

#[inline(always)]
fn load_i16_as_f16<D: SimdDescriptor>(d: D, src: &[i16], scale: f32) -> D::U16Vec {
    let len = D::U16Vec::LEN;
    let mut buf = [0u16; 32];
    i16_to_f16_simd(d, src, &mut buf[..len], scale, len);
    D::U16Vec::load(d, &buf[..len])
}

macro_rules! run_simd_chunks {
    ($out:expr, $ty:ty, $vec_trait:ident, $cnt:expr, |$n_var:ident, $out_chunk:ident| $body:block) => {{
        let len = D::$vec_trait::LEN;
        let step = {
            #[inline(always)]
            |$n_var: usize, $out_chunk: &mut [$ty]| $body
        };
        let mut n = 0;

        let mut chunks = $out.chunks_exact_mut(len * $cnt);
        for chunk in &mut chunks {
            step(n, chunk);
            n += len;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let mut tail = [<$ty>::default(); 64 / std::mem::size_of::<$ty>() * $cnt];
            step(n, &mut tail[..len * $cnt]);
            rem.copy_from_slice(&tail[..rem.len()]);
        }
    }};
}

macro_rules! define_store_interleaved {
    ($fn_name:ident, $impl_name:ident, $ty:ty, $vec_trait:ident, $simd_trait:ident) => {
        simd_function!(
            $fn_name,
            d: D,
            fn $impl_name(inputs: &[&[$ty]], out: &mut [$ty]) {
                match inputs.len() {
                    2 => run_simd_chunks!(out, $ty, $vec_trait, 2, |n, out_chunk| {
                        let a = D::$vec_trait::load(d, &inputs[0][n..]);
                        let b = D::$vec_trait::load(d, &inputs[1][n..]);
                        $simd_trait::store_interleaved_2(a, b, out_chunk);
                    }),
                    3 => run_simd_chunks!(out, $ty, $vec_trait, 3, |n, out_chunk| {
                        let a = D::$vec_trait::load(d, &inputs[0][n..]);
                        let b = D::$vec_trait::load(d, &inputs[1][n..]);
                        let c = D::$vec_trait::load(d, &inputs[2][n..]);
                        $simd_trait::store_interleaved_3(a, b, c, out_chunk);
                    }),
                    4 => run_simd_chunks!(out, $ty, $vec_trait, 4, |n, out_chunk| {
                        let a = D::$vec_trait::load(d, &inputs[0][n..]);
                        let b = D::$vec_trait::load(d, &inputs[1][n..]);
                        let c = D::$vec_trait::load(d, &inputs[2][n..]);
                        let e = D::$vec_trait::load(d, &inputs[3][n..]);
                        $simd_trait::store_interleaved_4(a, b, c, e, out_chunk);
                    }),
                    _ => unreachable!(),
                }
            }
        );
    };
}

define_store_interleaved!(
    store_interleaved_f32,
    store_interleaved_impl_f32,
    f32,
    F32Vec,
    F32SimdVec
);
define_store_interleaved!(
    store_interleaved_u8,
    store_interleaved_impl_u8,
    u8,
    U8Vec,
    U8SimdVec
);
define_store_interleaved!(
    store_interleaved_u16,
    store_interleaved_impl_u16,
    u16,
    U16Vec,
    U16SimdVec
);

#[derive(Clone, Copy)]
enum AlphaU8<'a> {
    None,
    Opaque,
    I16(&'a [i16], i32),
}

#[derive(Clone, Copy)]
enum AlphaF16<'a> {
    None,
    Opaque,
    I16(&'a [i16], f32),
}

simd_function!(
    store_f32_to_u8_interleaved,
    d: D,
    #[allow(clippy::too_many_arguments)]
    fn store_f32_to_u8_interleaved_impl(
        r: &[f32],
        g: &[f32],
        b: &[f32],
        channels: [usize; 3],
        pos: (usize, usize),
        max: u8,
        alpha: AlphaU8<'_>,
        out: &mut [u8],
        dither_table: &DitherTable,
    ) {
        let max_f32 = max as f32;
        let max_i32 = max as i32;
        match alpha {
            AlphaU8::None => run_simd_chunks!(out, u8, U8Vec, 3, |n, out_chunk| {
                let p = (pos.0 + n, pos.1);
                let vr = load_f32_as_u8(d, &r[n..], max_f32, p, channels[0], dither_table);
                let vg = load_f32_as_u8(d, &g[n..], max_f32, p, channels[1], dither_table);
                let vb = load_f32_as_u8(d, &b[n..], max_f32, p, channels[2], dither_table);
                U8SimdVec::store_interleaved_3(vr, vg, vb, out_chunk);
            }),
            AlphaU8::Opaque => {
                let va = D::U8Vec::splat(d, max);
                run_simd_chunks!(out, u8, U8Vec, 4, |n, out_chunk| {
                    let p = (pos.0 + n, pos.1);
                    let vr = load_f32_as_u8(d, &r[n..], max_f32, p, channels[0], dither_table);
                    let vg = load_f32_as_u8(d, &g[n..], max_f32, p, channels[1], dither_table);
                    let vb = load_f32_as_u8(d, &b[n..], max_f32, p, channels[2], dither_table);
                    U8SimdVec::store_interleaved_4(vr, vg, vb, va, out_chunk);
                });
            }
            AlphaU8::I16(a, a_scale) => {
                run_simd_chunks!(out, u8, U8Vec, 4, |n, out_chunk| {
                    let p = (pos.0 + n, pos.1);
                    let vr = load_f32_as_u8(d, &r[n..], max_f32, p, channels[0], dither_table);
                    let vg = load_f32_as_u8(d, &g[n..], max_f32, p, channels[1], dither_table);
                    let vb = load_f32_as_u8(d, &b[n..], max_f32, p, channels[2], dither_table);
                    let va = load_i16_as_u8(d, &a[n..], a_scale, max_i32);
                    U8SimdVec::store_interleaved_4(vr, vg, vb, va, out_chunk);
                });
            }
        }
    }
);

simd_function!(
    store_f32_to_f16_interleaved,
    d: D,
    fn store_f32_to_f16_interleaved_impl(
        r: &[f32],
        g: &[f32],
        b: &[f32],
        alpha: AlphaF16<'_>,
        out: &mut [u16],
    ) {
        match alpha {
            AlphaF16::None => run_simd_chunks!(out, u16, U16Vec, 3, |n, out_chunk| {
                let vr = load_f32_as_f16(d, &r[n..]);
                let vg = load_f32_as_f16(d, &g[n..]);
                let vb = load_f32_as_f16(d, &b[n..]);
                U16SimdVec::store_interleaved_3(vr, vg, vb, out_chunk);
            }),
            AlphaF16::Opaque => {
                let va = D::U16Vec::splat(d, 0x3c00);
                run_simd_chunks!(out, u16, U16Vec, 4, |n, out_chunk| {
                    let vr = load_f32_as_f16(d, &r[n..]);
                    let vg = load_f32_as_f16(d, &g[n..]);
                    let vb = load_f32_as_f16(d, &b[n..]);
                    U16SimdVec::store_interleaved_4(vr, vg, vb, va, out_chunk);
                });
            }
            AlphaF16::I16(a, a_scale) => {
                run_simd_chunks!(out, u16, U16Vec, 4, |n, out_chunk| {
                    let vr = load_f32_as_f16(d, &r[n..]);
                    let vg = load_f32_as_f16(d, &g[n..]);
                    let vb = load_f32_as_f16(d, &b[n..]);
                    let va = load_i16_as_f16(d, &a[n..], a_scale);
                    U16SimdVec::store_interleaved_4(vr, vg, vb, va, out_chunk);
                });
            }
        }
    }
);

simd_function!(
    store_i16_to_u8_interleaved,
    d: D,
    fn store_i16_to_u8_interleaved_impl(
        r: (&[i16], i32),
        g: (&[i16], i32),
        b: (&[i16], i32),
        max: u8,
        alpha: AlphaU8<'_>,
        out: &mut [u8],
    ) {
        let max_i32 = max as i32;
        match alpha {
            AlphaU8::None => run_simd_chunks!(out, u8, U8Vec, 3, |n, out_chunk| {
                let vr = load_i16_as_u8(d, &r.0[n..], r.1, max_i32);
                let vg = load_i16_as_u8(d, &g.0[n..], g.1, max_i32);
                let vb = load_i16_as_u8(d, &b.0[n..], b.1, max_i32);
                U8SimdVec::store_interleaved_3(vr, vg, vb, out_chunk);
            }),
            AlphaU8::Opaque => {
                let va = D::U8Vec::splat(d, max);
                run_simd_chunks!(out, u8, U8Vec, 4, |n, out_chunk| {
                    let vr = load_i16_as_u8(d, &r.0[n..], r.1, max_i32);
                    let vg = load_i16_as_u8(d, &g.0[n..], g.1, max_i32);
                    let vb = load_i16_as_u8(d, &b.0[n..], b.1, max_i32);
                    U8SimdVec::store_interleaved_4(vr, vg, vb, va, out_chunk);
                });
            }
            AlphaU8::I16(a, a_scale) => {
                run_simd_chunks!(out, u8, U8Vec, 4, |n, out_chunk| {
                    let vr = load_i16_as_u8(d, &r.0[n..], r.1, max_i32);
                    let vg = load_i16_as_u8(d, &g.0[n..], g.1, max_i32);
                    let vb = load_i16_as_u8(d, &b.0[n..], b.1, max_i32);
                    let va = load_i16_as_u8(d, &a[n..], a_scale, max_i32);
                    U8SimdVec::store_interleaved_4(vr, vg, vb, va, out_chunk);
                });
            }
        }
    }
);

pub(super) fn store_fused(
    stage: &SaveStage,
    input_buf: &[&RowBuffer],
    input_y: usize,
    group_x0: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
) -> bool {
    let len = xrange.len();
    let out_channels = stage.output_channels();
    let f32_row = |i: usize| {
        let off = RowBuffer::x0_offset::<f32>() + xrange.start;
        &input_buf[i].get_row::<f32>(input_y)[off..]
    };
    let i16_row = |i: usize| {
        let off = RowBuffer::x0_offset::<i16>() + xrange.start;
        &input_buf[i].get_row::<i16>(input_y)[off..]
    };
    match stage.data_format {
        JxlDataFormat::U8 { bit_depth } => {
            let max = ((1u16 << bit_depth) - 1) as u8;
            let scale = |bd: u8| (max as i32) / ((1i32 << bd) - 1);
            let pos = (group_x0 + xrange.start, input_y);
            let output_slice = &mut output_buf.row_mut(output_y)[0..len * out_channels];
            match stage.channel_types.as_slice() {
                [
                    SaveChannelType::F32,
                    SaveChannelType::F32,
                    SaveChannelType::F32,
                    rest @ ..,
                ] => {
                    let alpha = match (rest, stage.fill_opaque_alpha) {
                        ([], false) => AlphaU8::None,
                        ([], true) => AlphaU8::Opaque,
                        ([SaveChannelType::I16 { bit_depth: a_bd }], false) => {
                            AlphaU8::I16(i16_row(3), scale(*a_bd))
                        }
                        _ => return false,
                    };
                    let chs = [stage.channels[0], stage.channels[1], stage.channels[2]];
                    store_f32_to_u8_interleaved(
                        f32_row(0),
                        f32_row(1),
                        f32_row(2),
                        chs,
                        pos,
                        max,
                        alpha,
                        output_slice,
                        stage.dither_table,
                    );
                    true
                }
                [
                    SaveChannelType::I16 { bit_depth: bd0 },
                    SaveChannelType::I16 { bit_depth: bd1 },
                    SaveChannelType::I16 { bit_depth: bd2 },
                    rest @ ..,
                ] => {
                    let alpha = match (rest, stage.fill_opaque_alpha) {
                        ([], false) => AlphaU8::None,
                        ([], true) => AlphaU8::Opaque,
                        ([SaveChannelType::I16 { bit_depth: bd3 }], false) => {
                            AlphaU8::I16(i16_row(3), scale(*bd3))
                        }
                        _ => return false,
                    };
                    store_i16_to_u8_interleaved(
                        (i16_row(0), scale(*bd0)),
                        (i16_row(1), scale(*bd1)),
                        (i16_row(2), scale(*bd2)),
                        max,
                        alpha,
                        output_slice,
                    );
                    true
                }
                _ => false,
            }
        }
        JxlDataFormat::F16 { endianness } if endianness == Endianness::native() => {
            let output_slice = &mut output_buf.row_mut(output_y)[0..len * out_channels * 2];
            if output_slice
                .as_mut_ptr()
                .align_offset(std::mem::align_of::<u16>())
                != 0
            {
                return false;
            }
            let output_u16 = u16::cast_slice_mut(output_slice);
            let [
                SaveChannelType::F32,
                SaveChannelType::F32,
                SaveChannelType::F32,
                rest @ ..,
            ] = stage.channel_types.as_slice()
            else {
                return false;
            };
            let alpha = match (rest, stage.fill_opaque_alpha) {
                ([], false) => AlphaF16::None,
                ([], true) => AlphaF16::Opaque,
                ([SaveChannelType::I16 { bit_depth: a_bd }], false) => {
                    AlphaF16::I16(i16_row(3), 1.0 / ((1u64 << a_bd) - 1) as f32)
                }
                _ => return false,
            };
            store_f32_to_f16_interleaved(f32_row(0), f32_row(1), f32_row(2), alpha, output_u16);
            true
        }
        _ => false,
    }
}

pub(super) fn store(
    input_buf: &[&RowBuffer],
    input_y: usize,
    xrange: Range<usize>,
    output_buf: &mut JxlOutputBuffer,
    output_y: usize,
    data_format: JxlDataFormat,
) -> bool {
    let byte_start = xrange.start * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let byte_end = xrange.end * data_format.bytes_per_sample() + RowBuffer::x0_byte_offset();
    let is_native_endian = match data_format {
        JxlDataFormat::U8 { .. } => true,
        JxlDataFormat::F16 { endianness, .. }
        | JxlDataFormat::U16 { endianness, .. }
        | JxlDataFormat::F32 { endianness, .. } => endianness == Endianness::native(),
    };
    let output_buf = output_buf.row_mut(output_y);
    let output_buf = &mut output_buf[0..(byte_end - byte_start) * input_buf.len()];
    match (
        input_buf.len(),
        data_format.bytes_per_sample(),
        is_native_endian,
    ) {
        (1, _, true) => {
            // We can just do a memcpy.
            let input_buf = &input_buf[0].get_row::<u8>(input_y)[byte_start..byte_end];
            output_buf.copy_from_slice(input_buf);
            true
        }
        (channels, 1, true) if (2..=4).contains(&channels) => {
            let start_u8 = byte_start;
            let mut slices = [&[] as &[u8]; 4];
            for (i, buf) in input_buf.iter().enumerate() {
                slices[i] = &buf.get_row::<u8>(input_y)[start_u8..];
            }
            store_interleaved_u8(&slices[..channels], output_buf);
            true
        }
        (channels, 2, true) if (2..=4).contains(&channels) => {
            let ptr = output_buf.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<u16>()) == 0 {
                let output_u16 = u16::cast_slice_mut(output_buf);
                let start_u16 = byte_start / 2;
                let mut slices = [&[] as &[u16]; 4];
                for (i, buf) in input_buf.iter().enumerate() {
                    slices[i] = &buf.get_row::<u16>(input_y)[start_u16..];
                }
                store_interleaved_u16(&slices[..channels], output_u16);
                true
            } else {
                false
            }
        }
        (channels, 4, true) if (2..=4).contains(&channels) => {
            let ptr = output_buf.as_mut_ptr();
            if ptr.align_offset(std::mem::align_of::<f32>()) == 0 {
                let output_f32 = f32::cast_slice_mut(output_buf);
                let start_f32 = byte_start / 4;
                let mut slices = [&[] as &[f32]; 4];
                for (i, buf) in input_buf.iter().enumerate() {
                    slices[i] = &buf.get_row::<f32>(input_y)[start_f32..];
                }
                store_interleaved_f32(&slices[..channels], output_f32);
                true
            } else {
                false
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod test {
    use jxl_simd::{f16, test_all_instruction_sets};

    use super::*;
    use crate::api::DitherMode;
    use crate::util::dither_table;

    #[test]
    fn u8_dither_modes_dispatch_interleaving_and_tails() {
        let channels = [2, 0, 3];
        let pos = (29, 31);
        for mode in [DitherMode::JxlRs, DitherMode::Libjxl] {
            let table = dither_table(mode);
            for max in [15u8, 255] {
                for xsize in 1..=65 {
                    let inputs: [Vec<f32>; 3] = std::array::from_fn(|c| {
                        (0..xsize + 64)
                            .map(|i| ((i + c) % 13) as f32 / 16.0 + 0.001)
                            .collect()
                    });
                    let mut planar = vec![0u8; xsize + 64];
                    f32_to_u8(
                        &inputs[0],
                        &mut planar,
                        max as f32,
                        pos,
                        channels[0],
                        xsize,
                        table,
                    );
                    let integer_alpha = vec![7i16; xsize + 64];
                    for alpha in [
                        AlphaU8::None,
                        AlphaU8::Opaque,
                        AlphaU8::I16(&integer_alpha, 1),
                    ] {
                        let nc = if matches!(alpha, AlphaU8::None) { 3 } else { 4 };
                        let mut output = vec![0xa5; xsize * nc + 1];
                        store_f32_to_u8_interleaved(
                            &inputs[0],
                            &inputs[1],
                            &inputs[2],
                            channels,
                            pos,
                            max,
                            alpha,
                            &mut output[..xsize * nc],
                            table,
                        );
                        for i in 0..xsize {
                            assert_eq!(planar[i], output[i * nc]);
                            for c in 0..3 {
                                let channel = channels[c];
                                let noise = table[(pos.1 + channel * 13) % 32]
                                    [(pos.0 + i + channel * 23) % 32];
                                let expected = (inputs[c][i] * max as f32 + noise)
                                    .clamp(0.0, max as f32)
                                    .round_ties_even()
                                    as u8;
                                assert_eq!(
                                    output[i * nc + c],
                                    expected,
                                    "{mode:?}, {xsize}, {i}, {c}"
                                );
                            }
                            if nc == 4 {
                                assert_eq!(
                                    output[i * nc + 3],
                                    if matches!(alpha, AlphaU8::Opaque) {
                                        max
                                    } else {
                                        7
                                    }
                                );
                            }
                        }
                        assert_eq!(output[xsize * nc], 0xa5);
                    }
                }
            }
        }
    }

    fn expand_f16(bits: u16) -> f32 {
        if bits & 0x7c00 == 0x7c00 {
            f32::from_bits(
                (u32::from(bits & 0x8000) << 16) | 0x7f80_0000 | (u32::from(bits & 0x03ff) << 13),
            )
        } else {
            f16::from_bits(bits).to_f32()
        }
    }

    fn f16_output_all_patterns<D: SimdDescriptor>(d: D) {
        let patterns: Vec<u16> = (0..=u16::MAX).map(|i| i.wrapping_mul(40503)).collect();
        let input: Vec<f32> = patterns.iter().copied().map(expand_f16).collect();
        let mut output = vec![0u16; patterns.len()];
        d.call(|d| f32_to_f16_simd(d, &input, &mut output, patterns.len()));
        for (i, (&actual, &expected)) in output.iter().zip(&patterns).enumerate() {
            assert_eq!(actual, expected, "F16 roundtrip at {i}, {d:?}");
        }
    }

    test_all_instruction_sets!(f16_output_all_patterns);

    fn f16_output_rounding_and_small_nan_payloads<D: SimdDescriptor>(d: D) {
        let cases = [
            (0x0000_0000, 0x0000),
            (0x8000_0000, 0x8000),
            (0x3f80_1000, 0x3c00), // Halfway, ties to even.
            (0x3f80_3000, 0x3c02),
            (0x3300_0000, 0x0000), // Half the smallest F16 subnormal.
            (0x3300_0001, 0x0001),
            (0xb300_0000, 0x8000),
            (0x477f_e000, 0x7bff),
            (0x477f_f000, 0x7c00),
            (0x7f80_0000, 0x7c00),
            (0xff80_0000, 0xfc00),
            (0x7f80_0001, 0x7c01), // A truncated NaN must not become infinity.
            (0xff80_1fff, 0xfc01),
            (0x7fc1_2345, 0x7e09),
            (0xffff_ffff, 0xffff),
        ];
        let len = cases.len().next_multiple_of(D::F32Vec::LEN);
        let input: Vec<f32> = (0..len)
            .map(|i| f32::from_bits(cases[i % cases.len()].0))
            .collect();
        let mut output = vec![0u16; len];
        d.call(|d| f32_to_f16_simd(d, &input, &mut output, cases.len()));
        for (i, &actual) in output.iter().enumerate() {
            assert_eq!(actual, cases[i % cases.len()].1, "index {i}, {d:?}");
        }
    }

    test_all_instruction_sets!(f16_output_rounding_and_small_nan_payloads);

    #[test]
    fn f16_output_dispatch_and_interleaved_tails() {
        let patterns = [
            0x7c01, 0x0000, 0x8000, 0x0001, 0x83ff, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7dff, 0xfc01,
            0xfdff, 0x7e55, 0xffff,
        ];
        for xsize in 1..=33 {
            // The fused conversion loads padded SIMD input even for a short tail.
            let inputs: [Vec<f32>; 3] = std::array::from_fn(|c| {
                (0..xsize + 32)
                    .map(|i| expand_f16(patterns[(i + c) % patterns.len()]))
                    .collect()
            });
            let mut planar = vec![0u16; xsize + 32];
            f32_to_f16(&inputs[0], &mut planar, xsize);
            for i in 0..xsize {
                assert_eq!(planar[i], patterns[i % patterns.len()]);
            }

            for channels in [3, 4] {
                let mut output = vec![0xdead; xsize * channels + 1];
                let alpha = if channels == 3 {
                    AlphaF16::None
                } else {
                    AlphaF16::Opaque
                };
                store_f32_to_f16_interleaved(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    alpha,
                    &mut output[..xsize * channels],
                );
                for i in 0..xsize {
                    for c in 0..channels {
                        let expected = if c == 3 {
                            0x3c00
                        } else {
                            patterns[(i + c) % patterns.len()]
                        };
                        assert_eq!(
                            output[i * channels + c],
                            expected,
                            "width {xsize}, channels {channels}, pixel {i}, channel {c}"
                        );
                    }
                }
                assert_eq!(output[xsize * channels], 0xdead);
            }
        }
    }
}
