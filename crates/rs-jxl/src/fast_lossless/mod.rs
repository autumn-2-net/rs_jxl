// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Pure Rust port of b866b3d enc_fast_lossless.{cc,h}.
//!
//! This is the separate FJXL encoder, not the general Modular effort policy.
//! Standalone output has integer sRGB/gray metadata and optional straight alpha.
//! Frame output requires matching outer metadata and a non-animation, full-canvas
//! Replace frame. It has no duration, crop, name, extra channels or reference ID.

mod entropy;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod entropy_simd;
mod header;
mod palette;
mod pixels;
mod regions;
mod simd;

use crate::bit_writer::BitWriter;
use crate::{Error, Result};
use entropy::{LZ, PrefixCode, RAW};
use palette::Palette;
use pixels::{Area, Encoder, Histogram, Processor};
pub use regions::{
    BorrowedRegion, EncodedFrame, FrameInfo, OutputProcessor, PreparedFrame, ProcessedFrame,
    Region, RegionBuffer, RegionSource, prepare_frame,
};
use simd::Kernel;

/// Interleaved integer samples: one byte for depths 1..=8, two for 9..=16.
/// Values must fit the declared depth; padding at the end of each row is ignored.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
    pub channels: u8,
    pub bit_depth: u8,
    pub big_endian: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// Runtime AVX2 when available, otherwise scalar.
    #[default]
    Auto,
    /// Source scalar eight-sample chunks, independent of host SIMD support.
    Scalar,
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_pixels: u64,
    /// Complete returned codestream/frame, not merely its pixel payload.
    pub max_output_bytes: usize,
    /// Total TOC entries, including empty DC and AC-global sections.
    pub max_groups: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_pixels: u64::MAX,
            max_output_bytes: usize::MAX,
            max_groups: usize::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// FJXL sampling effort 0..=127; independent of general encoder effort.
    pub effort: u8,
    /// Source oneshot sampling/palette policy, independent of input storage.
    pub oneshot: bool,
    pub backend: Backend,
    pub limits: Limits,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            effort: 2,
            oneshot: true,
            backend: Backend::Auto,
            limits: Limits::default(),
        }
    }
}

impl ImageView<'_> {
    fn sample(self, x: usize, y: usize, c: usize) -> u16 {
        let bytes = if self.bit_depth <= 8 { 1 } else { 2 };
        let index = y * self.row_stride + (x * usize::from(self.channels) + c) * bytes;
        if bytes == 1 {
            u16::from(self.pixels[index])
        } else {
            let pair = [self.pixels[index], self.pixels[index + 1]];
            if self.big_endian {
                u16::from_be_bytes(pair)
            } else {
                u16::from_le_bytes(pair)
            }
        }
    }

    fn pixel8(self, x: usize, y: usize) -> u32 {
        let index = y * self.row_stride + x * usize::from(self.channels);
        self.pixels[index..index + usize::from(self.channels)]
            .iter()
            .enumerate()
            .fold(0, |p, (c, &v)| p | (u32::from(v) << (8 * c)))
    }

    fn validate(self, options: &Options) -> Result<()> {
        FrameInfo::from(self).validate(options)?;
        let row = self
            .width
            .checked_mul(usize::from(self.channels))
            .and_then(|n| n.checked_mul(if self.bit_depth <= 8 { 1 } else { 2 }))
            .ok_or(Error::Limit("FJXL row size"))?;
        let required = (self.height - 1)
            .checked_mul(self.row_stride)
            .and_then(|n| n.checked_add(row))
            .ok_or(Error::Limit("FJXL input size"))?;
        if self.row_stride < row || self.pixels.len() < required {
            return Err(Error::InvalidInput(
                "FJXL input stride or buffer is too short".into(),
            ));
        }
        if self.bit_depth != 8 && self.bit_depth != 16 {
            let max = (1u32 << self.bit_depth) - 1;
            for y in 0..self.height {
                for x in 0..self.width {
                    for c in 0..usize::from(self.channels) {
                        if u32::from(self.sample(x, y, c)) > max {
                            return Err(Error::InvalidInput(
                                "FJXL sample exceeds declared bit depth".into(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Encode a complete raw JXL codestream on the calling thread, without Rayon.
pub fn encode(image: ImageView<'_>, options: &Options) -> Result<Vec<u8>> {
    crate::execution::serial(|| encode_impl(image, options, true, true))
}

/// Encode a full-canvas frame only. Caller supplies compatible image metadata.
pub fn encode_frame(image: ImageView<'_>, options: &Options, is_last: bool) -> Result<Vec<u8>> {
    crate::execution::serial(|| encode_impl(image, options, false, is_last))
}

/// Encode using only the caller's explicit pool for independent AC groups.
pub fn encode_with_pool(
    image: ImageView<'_>,
    options: &Options,
    pool: &rayon::ThreadPool,
) -> Result<Vec<u8>> {
    crate::execution::with_pool(pool, || encode_impl(image, options, true, true))
}

pub fn encode_frame_with_pool(
    image: ImageView<'_>,
    options: &Options,
    is_last: bool,
    pool: &rayon::ThreadPool,
) -> Result<Vec<u8>> {
    crate::execution::with_pool(pool, || encode_impl(image, options, false, is_last))
}

fn histograms<S: RegionSource + ?Sized>(
    source: &S,
    info: FrameInfo,
    options: &Options,
    palette: Option<&Palette>,
    kernel: Kernel,
    groups_x: usize,
    groups_y: usize,
) -> Result<[Histogram; 4]> {
    let one = groups_x == 1 && groups_y == 1;
    let mut totals: [Histogram; 4] = std::array::from_fn(|_| Histogram::default());
    let mut sample = |xg: usize, yg: usize, rows: usize| -> Result<()> {
        let ys = (info.height - yg * 256).min(256);
        let xs = (info.width - xg * 256).min(256);
        let begin = ys.saturating_sub(rows) / 2;
        let count = rows.min(ys - begin - 1);
        let area = Area {
            x: 0,
            y: begin,
            width: xs / kernel.chunk_size() * kernel.chunk_size(),
            height: count + 1,
            skip: 1,
        };
        let channels = if palette.is_some() {
            1
        } else {
            usize::from(info.channels)
        };
        let mut processors: Vec<_> = (0..channels)
            .map(|_| Processor::new(Histogram::default()))
            .collect();
        let region = Region {
            x: xg * 256,
            y: yg * 256,
            width: xs,
            height: ys,
        };
        info.with_region(source, region, options, |image| {
            pixels::process(image, area, palette, kernel, &mut processors)
        })?;
        for (c, p) in processors.into_iter().enumerate() {
            let target = &mut totals[if palette.is_some() {
                usize::from(one)
            } else {
                c
            }];
            for (a, b) in target.raw.iter_mut().zip(p.sink.raw) {
                *a += b;
            }
            for (a, b) in target.lz.iter_mut().zip(p.sink.lz) {
                *a += b;
            }
        }
        Ok(())
    };
    if options.oneshot || options.effort >= 64 {
        for yg in 0..groups_y {
            let ys = (info.height - yg * 256).min(256);
            for xg in 0..groups_x {
                sample(xg, yg, 2 * usize::from(options.effort) * ys / 256)?;
            }
        }
    } else {
        let rows = groups_x
            .checked_mul(groups_y)
            .and_then(|n| n.checked_mul(2 * usize::from(options.effort)))
            .ok_or(Error::Limit("FJXL sampling rows"))?;
        sample((groups_x - 1) / 2, (groups_y - 1) / 2, rows)?;
    }
    let mut base_raw: [u64; RAW] = [
        3843, 852, 1270, 1214, 1014, 727, 481, 300, 159, 51, 5, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    let symbols = match info.bit_depth {
        1..=13 => {
            usize::from(info.bit_depth)
                + if info.channels > 2 || palette.is_some() {
                    3
                } else {
                    2
                }
        }
        14 => 17,
        _ => 19,
    };
    base_raw[symbols..].fill(0);
    let base_lz: [u64; LZ] = [
        29, 27, 25, 23, 21, 21, 19, 18, 21, 17, 16, 15, 15, 14, 13, 13, 137, 98, 61, 34, 1, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    ];
    for h in &mut totals {
        for (n, base) in h.raw.iter_mut().zip(base_raw) {
            *n = n
                .checked_mul(256)
                .and_then(|n| n.checked_add(base))
                .ok_or(Error::Limit("FJXL raw frequencies"))?;
        }
        for (n, base) in h.lz.iter_mut().zip(base_lz) {
            *n = n
                .checked_mul(256)
                .and_then(|n| n.checked_add(base))
                .ok_or(Error::Limit("FJXL LZ frequencies"))?;
        }
    }
    if let Some(palette) = palette {
        let token = entropy::uint000(pixels::pack_signed(palette.colors.len() as i32 - 1)).0;
        for n in &mut totals[0].raw[..=token] {
            *n = (*n).max(1);
        }
        if token + 1 < 10 {
            totals[0].raw[token + 1..10].fill(1);
        }
    }
    Ok(totals)
}

#[allow(clippy::too_many_arguments)] // Source group inputs plus bounded destination.
fn group(
    image: ImageView<'_>,
    area: Area,
    palette: Option<&Palette>,
    kernel: Kernel,
    codes: &[PrefixCode; 4],
    one: bool,
    mut writer: BitWriter,
    limit: usize,
) -> Result<BitWriter> {
    if !one {
        writer.write(4, 3)?;
    }
    let channels = if palette.is_some() {
        1
    } else {
        usize::from(image.channels)
    };
    let mut processors: Vec<_> = (0..channels)
        .map(|c| {
            let code = &codes[if palette.is_some() {
                usize::from(one)
            } else {
                c
            }];
            Processor::new(Encoder {
                code,
                writer: BitWriter::new(limit),
                kernel,
            })
        })
        .collect();
    pixels::process(image, area, palette, kernel, &mut processors)?;
    for processor in processors {
        writer.append_unaligned(&processor.sink.writer)?;
    }
    writer.zero_pad_to_byte()?;
    Ok(writer)
}

fn encode_impl(
    image: ImageView<'_>,
    options: &Options,
    image_header: bool,
    is_last: bool,
) -> Result<Vec<u8>> {
    image.validate(options)?;
    regions::prepare_impl(&image, options)?
        .process_impl()?
        .prepare_header(image_header, is_last)?
        .into_bytes()
}

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;
