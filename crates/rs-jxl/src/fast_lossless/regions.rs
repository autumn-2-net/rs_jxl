// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! b866b3d enc_fast_lossless.{cc,h}: region leases, prepared states and output.

use super::{
    ImageView, Options, Palette, PrefixCode, group, header, histograms, pixels, simd::Kernel,
};
use crate::{Error, Result, bit_writer::BitWriter};

/// Fixed format negotiated before any region is acquired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameInfo {
    pub width: usize,
    pub height: usize,
    pub channels: u8,
    pub bit_depth: u8,
    pub big_endian: bool,
}

impl From<ImageView<'_>> for FrameInfo {
    fn from(image: ImageView<'_>) -> Self {
        Self {
            width: image.width,
            height: image.height,
            channels: image.channels,
            bit_depth: image.bit_depth,
            big_endian: image.big_endian,
        }
    }
}

impl FrameInfo {
    pub(super) fn validate(self, options: &Options) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.width > 1 << 30
            || self.height > 1 << 30
            || !(1..=4).contains(&self.channels)
            || !(1..=16).contains(&self.bit_depth)
        {
            return Err(Error::InvalidInput(
                "FJXL requires nonzero dimensions <= 2^30, 1..4 channels and depth 1..16".into(),
            ));
        }
        if options.effort > 127 {
            return Err(Error::InvalidInput("FJXL effort must be 0..127".into()));
        }
        let pixels = self
            .width
            .checked_mul(self.height)
            .ok_or(Error::Limit("FJXL pixels"))?;
        if pixels as u64 > options.limits.max_pixels {
            return Err(Error::Limit("FJXL pixels"));
        }
        if options.limits.max_output_bytes == 0 || options.limits.max_groups == 0 {
            return Err(Error::Limit("FJXL output/groups"));
        }
        Ok(())
    }

    pub(super) fn with_region<S: RegionSource + ?Sized, R>(
        self,
        source: &S,
        region: Region,
        options: &Options,
        work: impl FnOnce(ImageView<'_>) -> Result<R>,
    ) -> Result<R> {
        // The user's lease is dropped on success, validation failure, work error
        // and unwind. No returned ImageView can outlive this call.
        let lease = source.get_region(region)?;
        let image = ImageView {
            pixels: lease.bytes(),
            width: region.width,
            height: region.height,
            row_stride: lease.row_stride(),
            channels: self.channels,
            bit_depth: self.bit_depth,
            big_endian: self.big_endian,
        };
        image.validate(options)?;
        work(image)
    }
}

/// FJXL requests entire AC groups: origins are multiples of 256, dimensions
/// are 1..=256, and partial sizes occur only at right/bottom image edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// A successful acquisition owns one lease. Implement Drop to release custom
/// storage/resources exactly once; bytes remain valid until that drop.
pub trait RegionBuffer {
    fn bytes(&self) -> &[u8];
    fn row_stride(&self) -> usize;
}

/// Re-readable immutable pixel source. `info` is queried once during prepare.
/// Sampling/palette detection may request a region again during processing.
/// Pool processing may hold multiple leases concurrently, up to sixteen.
pub trait RegionSource: Sync {
    type Buffer<'a>: RegionBuffer
    where
        Self: 'a;
    fn info(&self) -> FrameInfo;
    fn get_region(&self, region: Region) -> Result<Self::Buffer<'_>>;
}

pub struct BorrowedRegion<'a> {
    pub pixels: &'a [u8],
    pub row_stride: usize,
}

impl RegionBuffer for BorrowedRegion<'_> {
    fn bytes(&self) -> &[u8] {
        self.pixels
    }
    fn row_stride(&self) -> usize {
        self.row_stride
    }
}

impl RegionSource for ImageView<'_> {
    type Buffer<'a>
        = BorrowedRegion<'a>
    where
        Self: 'a;
    fn info(&self) -> FrameInfo {
        (*self).into()
    }
    fn get_region(&self, region: Region) -> Result<Self::Buffer<'_>> {
        if region.width == 0
            || region.height == 0
            || region
                .x
                .checked_add(region.width)
                .is_none_or(|x| x > self.width)
            || region
                .y
                .checked_add(region.height)
                .is_none_or(|y| y > self.height)
        {
            return Err(Error::InvalidInput("FJXL region outside input".into()));
        }
        let pixel_size = usize::from(self.channels) * if self.bit_depth <= 8 { 1 } else { 2 };
        let full_row = self
            .width
            .checked_mul(pixel_size)
            .ok_or(Error::Limit("FJXL row size"))?;
        if self.row_stride < full_row {
            return Err(Error::InvalidInput("FJXL input stride is too short".into()));
        }
        let start = region
            .y
            .checked_mul(self.row_stride)
            .and_then(|y| {
                region
                    .x
                    .checked_mul(pixel_size)
                    .and_then(|x| y.checked_add(x))
            })
            .ok_or(Error::Limit("FJXL region offset"))?;
        let length = (region.height - 1)
            .checked_mul(self.row_stride)
            .and_then(|n| {
                region
                    .width
                    .checked_mul(pixel_size)
                    .and_then(|row| n.checked_add(row))
            })
            .ok_or(Error::Limit("FJXL region length"))?;
        let end = start
            .checked_add(length)
            .ok_or(Error::Limit("FJXL region end"))?;
        let pixels = self
            .pixels
            .get(start..end)
            .ok_or_else(|| Error::InvalidInput("FJXL input buffer is too short".into()))?;
        Ok(BorrowedRegion {
            pixels,
            row_stride: self.row_stride,
        })
    }
}

/// Output callbacks at an absolute byte position. Only one buffer is acquired
/// at a time. Successful get_buffer calls receive exactly one release_buffer,
/// including release(0) on empty buffers or unwind. Returning an empty buffer
/// requests a terminal stop, not a retry. Errors leave partial output invalid.
///
/// `process_to` needs forward/backward seek for multiple groups; write_output
/// and EncodedFrame::write_to do not. Seek must not precede finalized bytes.
/// The caller finalizes explicitly after any outer container backpatching;
/// FJXL itself never calls set_finalized_position (matching source LLProcess).
pub trait OutputProcessor {
    fn position(&self) -> u64;
    fn get_buffer(&mut self, suggested_size: usize) -> Result<&mut [u8]>;
    fn release_buffer(&mut self, written_bytes: usize) -> Result<()>;
    fn seek(&mut self, position: u64) -> Result<()>;
    fn set_finalized_position(&mut self, position: u64) -> Result<()>;
}

struct OutputLease<'a, O: OutputProcessor + ?Sized> {
    output: &'a mut O,
    active: bool,
}

impl<O: OutputProcessor + ?Sized> Drop for OutputLease<'_, O> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.output.release_buffer(0);
        }
    }
}

fn write_some<O: OutputProcessor + ?Sized>(output: &mut O, bytes: &[u8]) -> Result<usize> {
    debug_assert!(!bytes.is_empty());
    let start = output.position();
    let mut lease = OutputLease {
        output,
        active: false,
    };
    let buffer = lease.output.get_buffer(bytes.len())?;
    lease.active = true;
    if buffer.is_empty() {
        return Err(Error::Encoding(
            "FJXL output processor requested stop".into(),
        ));
    }
    let count = bytes.len().min(buffer.len());
    let end = start
        .checked_add(count as u64)
        .ok_or(Error::Limit("FJXL output position"))?;
    buffer[..count].copy_from_slice(&bytes[..count]);
    // A failing release is still one release, never repeated by Drop.
    lease.active = false;
    lease.output.release_buffer(count)?;
    if lease.output.position() != end {
        return Err(Error::Encoding(
            "FJXL output processor did not advance by released bytes".into(),
        ));
    }
    Ok(count)
}

fn write_all<O: OutputProcessor + ?Sized>(output: &mut O, mut bytes: &[u8]) -> Result<()> {
    while !bytes.is_empty() {
        let n = write_some(output, bytes)?;
        bytes = &bytes[n..];
    }
    Ok(())
}

fn seek<O: OutputProcessor + ?Sized>(output: &mut O, position: u64) -> Result<()> {
    output.seek(position)?;
    if output.position() != position {
        return Err(Error::Encoding(
            "FJXL output processor did not seek to requested position".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub gx: usize,
    pub gy: usize,
    pub ac: usize,
    pub dc: usize,
    pub entries: usize,
}

impl Geometry {
    fn new(info: FrameInfo, options: &Options) -> Result<Self> {
        info.validate(options)?;
        let gx = info.width.div_ceil(256);
        let gy = info.height.div_ceil(256);
        let ac = gx.checked_mul(gy).ok_or(Error::Limit("FJXL groups"))?;
        let dc = info
            .width
            .div_ceil(2048)
            .checked_mul(info.height.div_ceil(2048))
            .ok_or(Error::Limit("FJXL DC groups"))?;
        let entries = if ac == 1 {
            1
        } else {
            ac.checked_add(dc)
                .and_then(|n| n.checked_add(2))
                .ok_or(Error::Limit("FJXL TOC"))?
        };
        if entries > options.limits.max_groups {
            return Err(Error::Limit("FJXL groups"));
        }
        Ok(Self {
            gx,
            gy,
            ac,
            dc,
            entries,
        })
    }

    pub fn region(self, info: FrameInfo, id: usize) -> Region {
        let x = id % self.gx * 256;
        let y = id / self.gx * 256;
        Region {
            x,
            y,
            width: (info.width - x).min(256),
            height: (info.height - y).min(256),
        }
    }
}

/// Prepared global palette/tree/histograms. Holds the source, but no input
/// region lease. Processing consumes this state, including on failure.
pub struct PreparedFrame<'a, S: RegionSource + ?Sized> {
    source: &'a S,
    info: FrameInfo,
    options: Options,
    geometry: Geometry,
    kernel: Kernel,
    palette: Option<Palette>,
    codes: [PrefixCode; 4],
    global: BitWriter,
}

/// Prepare serially, like source LLPrepare. No global Rayon pool is entered.
pub fn prepare_frame<'a, S: RegionSource + ?Sized>(
    source: &'a S,
    options: &Options,
) -> Result<PreparedFrame<'a, S>> {
    crate::execution::serial(|| prepare_impl(source, options))
}

pub(super) fn prepare_impl<'a, S: RegionSource + ?Sized>(
    source: &'a S,
    options: &Options,
) -> Result<PreparedFrame<'a, S>> {
    let info = source.info();
    let geometry = Geometry::new(info, options)?;
    let kernel = Kernel::choose(options.backend);
    let palette = if options.effort >= 2 && info.bit_depth == 8 && options.oneshot {
        let mut table = vec![0u32; 65536];
        let mut collided = false;
        for id in 0..geometry.ac {
            collided = info.with_region(source, geometry.region(info, id), options, |image| {
                Ok(!Palette::add_region(&mut table, image))
            })?;
            if collided {
                break;
            }
        }
        if collided {
            None
        } else {
            Palette::from_table(table, info.channels)
        }
    } else {
        None
    };
    let [a, b, c, d] = histograms(
        source,
        info,
        options,
        palette.as_ref(),
        kernel,
        geometry.gx,
        geometry.gy,
    )?;
    let codes = [
        PrefixCode::new(&a.raw, &a.lz, info.bit_depth)?,
        PrefixCode::new(&b.raw, &b.lz, info.bit_depth)?,
        PrefixCode::new(&c.raw, &c.lz, info.bit_depth)?,
        PrefixCode::new(&d.raw, &d.lz, info.bit_depth)?,
    ];
    let mut global = BitWriter::new(options.limits.max_output_bytes);
    header::global_common(&mut global, &codes)?;
    header::transforms(
        &mut global,
        info.channels,
        palette.as_ref().map(|p| p.colors.len()),
    )?;
    if let Some(palette) = &palette {
        global = pixels::palette_data(palette, info.channels, &codes[0], kernel, global)?;
    }
    Ok(PreparedFrame {
        source,
        info,
        options: *options,
        geometry,
        kernel,
        palette,
        codes,
        global,
    })
}

impl<S: RegionSource + ?Sized> PreparedFrame<'_, S> {
    pub fn info(&self) -> FrameInfo {
        self.info
    }

    pub fn process(self) -> Result<ProcessedFrame> {
        crate::execution::serial(|| self.process_impl())
    }

    pub fn process_with_pool(self, pool: &rayon::ThreadPool) -> Result<ProcessedFrame> {
        crate::execution::with_pool(pool, || self.process_impl())
    }

    /// Stream AC sections in batches of at most sixteen, then backpatch the
    /// reserved header/global prefix. Requires seek for a multi-group frame.
    /// Leaves position at frame end; caller retains responsibility for finalization.
    pub fn process_to<O: OutputProcessor + ?Sized>(
        self,
        output: &mut O,
        add_image_header: bool,
        is_last: bool,
    ) -> Result<usize> {
        crate::execution::serial(|| self.process_to_impl(output, add_image_header, is_last))
    }

    pub fn process_to_with_pool<O: OutputProcessor + Send + ?Sized>(
        self,
        output: &mut O,
        add_image_header: bool,
        is_last: bool,
        pool: &rayon::ThreadPool,
    ) -> Result<usize> {
        crate::execution::with_pool(pool, || {
            self.process_to_impl(output, add_image_header, is_last)
        })
    }

    fn encode_group(&self, id: usize, writer: BitWriter) -> Result<Vec<u8>> {
        let region = self.geometry.region(self.info, id);
        self.info
            .with_region(self.source, region, &self.options, |image| {
                if let Some(palette) = &self.palette {
                    for y in 0..image.height {
                        for x in 0..image.width {
                            let pixel = image.pixel8(x, y);
                            if palette.colors[usize::from(palette.index(pixel))] != pixel {
                                return Err(Error::InvalidInput(
                                    "FJXL region source changed after palette preparation".into(),
                                ));
                            }
                        }
                    }
                }
                group(
                    image,
                    pixels::Area {
                        x: 0,
                        y: 0,
                        width: image.width,
                        height: image.height,
                        skip: 0,
                    },
                    self.palette.as_ref(),
                    self.kernel,
                    &self.codes,
                    self.geometry.ac == 1,
                    writer,
                    self.options.limits.max_output_bytes,
                )?
                .into_bytes()
                .map_err(Error::from)
            })
    }

    fn batch(&self, start: usize) -> Result<Vec<Vec<u8>>> {
        let ids: Vec<_> = (start..self.geometry.ac.min(start + 16)).collect();
        crate::execution::try_map(&ids, |_, &id| {
            self.encode_group(id, BitWriter::new(self.options.limits.max_output_bytes))
        })
    }

    pub(super) fn process_impl(mut self) -> Result<ProcessedFrame> {
        let limit = self.options.limits.max_output_bytes;
        let mut global = std::mem::replace(&mut self.global, BitWriter::new(limit));
        let mut sections = Vec::new();
        if self.geometry.ac == 1 {
            sections.push(self.encode_group(0, global)?);
        } else {
            global.zero_pad_to_byte()?;
            sections.push(global.into_bytes()?);
            let mut total = sections[0].len();
            sections.resize_with(self.geometry.dc + 2, Vec::new);
            for start in (0..self.geometry.ac).step_by(16) {
                for bytes in self.batch(start)? {
                    total = total
                        .checked_add(bytes.len())
                        .ok_or(Error::Limit("FJXL output"))?;
                    if total > limit {
                        return Err(Error::Limit("FJXL output"));
                    }
                    sections.push(bytes);
                }
            }
        }
        Ok(ProcessedFrame {
            info: self.info,
            sections,
            limit,
        })
    }

    fn process_to_impl<O: OutputProcessor + ?Sized>(
        mut self,
        output: &mut O,
        add_image_header: bool,
        is_last: bool,
    ) -> Result<usize> {
        if self.geometry.ac == 1 {
            let mut frame = self
                .process_impl()?
                .prepare_header(add_image_header, is_last)?;
            let size = frame.output_size();
            frame.write_to(output)?;
            return Ok(size);
        }
        let limit = self.options.limits.max_output_bytes;
        let mut global = std::mem::replace(&mut self.global, BitWriter::new(limit));
        global.zero_pad_to_byte()?;
        let global = global.into_bytes()?;
        let (minimum, offset) = reservation(global.len(), self.geometry.dc, self.geometry.ac)?;
        let mut prefix = BitWriter::new(limit);
        if add_image_header {
            header::image(&mut prefix, self.info)?;
        }
        let prefix = prefix.into_bytes()?;
        let reserved = prefix
            .len()
            .checked_add(offset)
            .ok_or(Error::Limit("FJXL output reservation"))?;
        if reserved > limit {
            return Err(Error::Limit("FJXL output reservation"));
        }
        let start = output.position();
        let ac_start = start
            .checked_add(reserved as u64)
            .ok_or(Error::Limit("FJXL output position"))?;
        seek(output, ac_start)?;
        let mut sizes = vec![0; self.geometry.entries];
        sizes[0] = minimum;
        let mut total = reserved;
        for first in (0..self.geometry.ac).step_by(16) {
            for (i, bytes) in self.batch(first)?.into_iter().enumerate() {
                // Source reservation allocates at most 24 TOC bits per AC group.
                if toc_bits(bytes.len())? > 24 {
                    return Err(Error::Limit("FJXL AC group reservation"));
                }
                total = total
                    .checked_add(bytes.len())
                    .ok_or(Error::Limit("FJXL output"))?;
                if total > limit {
                    return Err(Error::Limit("FJXL output"));
                }
                sizes[2 + self.geometry.dc + first + i] = bytes.len();
                write_all(output, &bytes)?;
            }
        }
        let end = start
            .checked_add(total as u64)
            .ok_or(Error::Limit("FJXL output position"))?;
        let mut frame_header = BitWriter::new(limit);
        header::frame(&mut frame_header, self.info.channels, is_last, &sizes)?;
        let header_size = frame_header.bits_written() / 8;
        let padded_global = offset
            .checked_sub(header_size)
            .ok_or(Error::Encoding("FJXL reservation underflow".into()))?;
        let padding = padded_global
            .checked_sub(global.len())
            .ok_or(Error::Encoding("FJXL global padding underflow".into()))?;
        sizes[0] = padded_global;
        let mut frame_header = BitWriter::new(limit);
        header::frame(&mut frame_header, self.info.channels, is_last, &sizes)?;
        if frame_header.bits_written() / 8 != header_size {
            return Err(Error::Encoding(
                "FJXL global padding changed TOC bucket".into(),
            ));
        }
        seek(output, start)?;
        write_all(output, &prefix)?;
        write_all(output, frame_header.bytes()?)?;
        write_all(output, &global)?;
        let zeroes = [0u8; 4096];
        let mut left = padding;
        while left != 0 {
            let n = left.min(zeroes.len());
            write_all(output, &zeroes[..n])?;
            left -= n;
        }
        if output.position() != ac_start {
            return Err(Error::Encoding(
                "FJXL backpatch length differs from reservation".into(),
            ));
        }
        seek(output, end)?;
        Ok(total)
    }
}

fn toc_bits(size: usize) -> Result<usize> {
    match size {
        0..1024 => Ok(12),
        1024..17408 => Ok(16),
        17408..4211712 => Ok(24),
        _ if (size as u64) < 4211712 + (1u64 << 30) => Ok(32),
        _ => Err(Error::Limit("FJXL group size")),
    }
}

// Source ComputeAcGroupDataOffset. Keep the padded global size within one
// TOC bucket so backpatching never changes the reserved prefix length.
fn reservation(global: usize, dc: usize, ac: usize) -> Result<(usize, usize)> {
    let ac_max = ac
        .checked_mul(24)
        .ok_or(Error::Limit("FJXL TOC reservation"))?;
    let ac_min = ac
        .checked_mul(12)
        .ok_or(Error::Limit("FJXL TOC reservation"))?;
    let padding = (ac_max - ac_min)
        .div_ceil(8)
        .checked_add(1)
        .ok_or(Error::Limit("FJXL TOC padding"))?;
    let mut minimum = global;
    loop {
        let upper = minimum
            .checked_add(padding)
            .ok_or(Error::Limit("FJXL global padding"))?;
        let bucket = toc_bits(upper)?;
        if bucket == toc_bits(minimum)? {
            break;
        }
        minimum = match bucket {
            16 => 1024,
            24 => 17408,
            32 => 4211712,
            _ => unreachable!(),
        };
    }
    let dc_toc = dc
        .checked_add(1)
        .and_then(|n| n.checked_mul(12))
        .ok_or(Error::Limit("FJXL DC TOC"))?;
    let toc = toc_bits(minimum)?
        .checked_add(dc_toc)
        .and_then(|n| n.checked_add(ac_max))
        .ok_or(Error::Limit("FJXL TOC"))?;
    let offset = 5usize
        .checked_add(toc.div_ceil(8))
        .and_then(|n| n.checked_add(minimum))
        .ok_or(Error::Limit("FJXL group offset"))?;
    Ok((minimum, offset))
}

/// Compressed sections only; no source reference or input pixels are retained.
pub struct ProcessedFrame {
    info: FrameInfo,
    sections: Vec<Vec<u8>>,
    limit: usize,
}

impl ProcessedFrame {
    pub fn prepare_header(self, add_image_header: bool, is_last: bool) -> Result<EncodedFrame> {
        let sizes: Vec<_> = self.sections.iter().map(Vec::len).collect();
        let mut header = BitWriter::new(self.limit);
        if add_image_header {
            header::image(&mut header, self.info)?;
        }
        header::frame(&mut header, self.info.channels, is_last, &sizes)?;
        let header = header.into_bytes()?;
        let total = sizes
            .iter()
            .try_fold(header.len(), |sum, &n| sum.checked_add(n))
            .ok_or(Error::Limit("FJXL output"))?;
        if total > self.limit {
            return Err(Error::Limit("FJXL output"));
        }
        let mut sections = self.sections;
        sections.insert(0, header);
        Ok(EncodedFrame {
            info: self.info,
            limit: self.limit,
            sections,
            total,
            written: 0,
            section: 0,
            offset: 0,
            failed: false,
        })
    }
}

/// Header-complete output cursor. Safe Rust writes need no source 32-byte
/// spill padding: any nonempty destination makes progress until EOF.
pub struct EncodedFrame {
    info: FrameInfo,
    limit: usize,
    sections: Vec<Vec<u8>>,
    total: usize,
    written: usize,
    section: usize,
    offset: usize,
    failed: bool,
}

impl EncodedFrame {
    pub fn output_size(&self) -> usize {
        self.total
    }
    pub fn remaining_size(&self) -> usize {
        self.total - self.written
    }

    /// Source PrepareHeader/process_done reuse: rebuild only the header and
    /// reset the output cursor, without acquiring regions or encoding again.
    /// Use a fresh destination (or explicitly rewind/discard the old output).
    pub fn reset_header(&mut self, add_image_header: bool, is_last: bool) -> Result<()> {
        let sizes: Vec<_> = self.sections[1..].iter().map(Vec::len).collect();
        let mut header = BitWriter::new(self.limit);
        if add_image_header {
            header::image(&mut header, self.info)?;
        }
        header::frame(&mut header, self.info.channels, is_last, &sizes)?;
        let header = header.into_bytes()?;
        let total = (self.total - self.sections[0].len())
            .checked_add(header.len())
            .ok_or(Error::Limit("FJXL output"))?;
        if total > self.limit {
            return Err(Error::Limit("FJXL output"));
        }
        self.sections[0] = header;
        self.total = total;
        self.written = 0;
        self.section = 0;
        self.offset = 0;
        self.failed = false;
        Ok(())
    }

    pub fn write_output(&mut self, output: &mut [u8]) -> Result<usize> {
        if self.failed {
            return Err(Error::Encoding(
                "FJXL output cursor failed previously".into(),
            ));
        }
        if self.remaining_size() == 0 {
            return Ok(0);
        }
        if output.is_empty() {
            return Err(Error::InvalidInput(
                "FJXL output buffer must not be empty before EOF".into(),
            ));
        }
        let mut used = 0;
        while used < output.len() && self.section < self.sections.len() {
            let bytes = &self.sections[self.section];
            let n = (output.len() - used).min(bytes.len() - self.offset);
            output[used..used + n].copy_from_slice(&bytes[self.offset..self.offset + n]);
            self.offset += n;
            used += n;
            if self.offset == bytes.len() {
                self.section += 1;
                self.offset = 0;
            }
        }
        self.written += used;
        Ok(used)
    }

    /// Writes remaining bytes without seeking or finalizing. The caller owns
    /// finalization after outer framing. A callback error poisons this cursor;
    /// discard partial output, or collect the original complete frame with into_bytes.
    pub fn write_to<O: OutputProcessor + ?Sized>(&mut self, output: &mut O) -> Result<()> {
        if self.failed {
            return Err(Error::Encoding(
                "FJXL output cursor failed previously".into(),
            ));
        }
        // Set before callbacks so unwinding also prevents an unsafe replay.
        self.failed = true;
        while self.section < self.sections.len() {
            let bytes = &self.sections[self.section];
            if self.offset == bytes.len() {
                self.section += 1;
                self.offset = 0;
                continue;
            }
            let n = write_some(output, &bytes[self.offset..])?;
            self.offset += n;
            self.written += n;
        }
        self.failed = false;
        Ok(())
    }

    /// Collect the full encoded output, regardless of the incremental cursor.
    pub fn into_bytes(self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.total)
            .map_err(|_| Error::Limit("FJXL output allocation"))?;
        for section in self.sections {
            bytes.extend(section);
        }
        Ok(bytes)
    }
}

#[cfg(test)]
#[path = "regions_tests.rs"]
pub(super) mod tests;
