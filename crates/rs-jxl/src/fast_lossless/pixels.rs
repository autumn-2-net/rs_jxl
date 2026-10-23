// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! b866b3d enc_fast_lossless.cc: row conversion, prediction and chunk RLE.

use super::{
    ImageView,
    entropy::{LZ, PrefixCode, RAW, uint000, uint400},
    palette::Palette,
    simd::Kernel,
};
use crate::Result;
use crate::bit_writer::BitWriter;

pub(super) fn pack_signed(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

pub(super) fn predict(px: i32, left: i32, top: i32, topleft: i32, narrow: bool) -> u32 {
    let cast = |n: i32| if narrow { i32::from(n as i16) } else { n };
    let ac = cast(left.wrapping_sub(topleft));
    let ab = cast(left.wrapping_sub(top));
    let bc = cast(top.wrapping_sub(topleft));
    let grad = cast(ac.wrapping_add(top));
    let clamp = if ab ^ bc < 0 { top } else { left };
    let pred = if ac ^ bc < 0 { grad } else { clamp };
    let residual = pack_signed(px.wrapping_sub(pred));
    if narrow {
        u32::from(residual as u16)
    } else {
        residual
    }
}

pub(super) trait Sink {
    fn chunk(&mut self, run: usize, residuals: &[u32]) -> Result<()>;
    fn finish(&mut self, run: usize) -> Result<()>;
}

pub(super) struct Encoder<'a> {
    pub code: &'a PrefixCode,
    pub writer: BitWriter,
    pub kernel: Kernel,
}

impl Sink for Encoder<'_> {
    fn chunk(&mut self, run: usize, residuals: &[u32]) -> Result<()> {
        if run != 0 {
            self.code.run(run, &mut self.writer)?;
        }
        self.kernel.literals(self.code, residuals, &mut self.writer)
    }
    fn finish(&mut self, run: usize) -> Result<()> {
        if run != 0 {
            self.code.run(run, &mut self.writer)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct Histogram {
    pub raw: [u64; RAW],
    pub lz: [u64; LZ],
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            raw: [0; RAW],
            lz: [0; LZ],
        }
    }
}

impl Sink for Histogram {
    fn chunk(&mut self, run: usize, residuals: &[u32]) -> Result<()> {
        if run != 0 {
            self.raw[0] += 1;
            self.lz[uint400((run - 8) as u32).0] += 1;
        }
        for &v in residuals {
            self.raw[uint000(v).0] += 1;
        }
        Ok(())
    }
    fn finish(&mut self, _run: usize) -> Result<()> {
        Ok(())
    }
}

pub(super) struct Processor<S> {
    pub sink: S,
    run: usize,
}

impl<S: Sink> Processor<S> {
    pub fn new(sink: S) -> Self {
        Self { sink, run: 0 }
    }

    #[cfg(test)] // Fixed-residual baseline for the fused prediction paths.
    pub fn row(&mut self, residuals: &[u32], chunk_size: usize) -> Result<()> {
        for chunk in residuals.chunks(chunk_size) {
            let prefix = chunk.iter().take_while(|&&v| v == 0).count();
            self.chunk(chunk, prefix)?;
        }
        Ok(())
    }

    fn chunk(&mut self, residuals: &[u32], prefix: usize) -> Result<()> {
        debug_assert!(prefix <= residuals.len());
        if prefix == residuals.len() && (self.run != 0 || prefix > 7) {
            self.run += prefix;
        } else if prefix + self.run > 7 {
            self.sink.chunk(self.run + prefix, &residuals[prefix..])?;
            self.run = 0;
        } else {
            self.sink.chunk(0, residuals)?;
        }
        Ok(())
    }

    pub fn predicted_wide_row(
        &mut self,
        kernel: Kernel,
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
    ) -> Result<()> {
        self.predicted_row::<false>(kernel, row, left, top, topleft)
    }

    pub fn predicted_narrow_row(
        &mut self,
        kernel: Kernel,
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
    ) -> Result<()> {
        self.predicted_row::<true>(kernel, row, left, top, topleft)
    }

    fn predicted_row<const NARROW: bool>(
        &mut self,
        kernel: Kernel,
        row: &[i32],
        left: &[i32],
        top: &[i32],
        topleft: &[i32],
    ) -> Result<()> {
        assert!(left.len() >= row.len() && top.len() >= row.len() && topleft.len() >= row.len());
        let mut residuals = [0; 16];
        for x in (0..row.len()).step_by(kernel.chunk_size()) {
            let end = row.len().min(x + kernel.chunk_size());
            let chunk = &mut residuals[..end - x];
            let prefix = if NARROW {
                kernel.predict_chunk_narrow(
                    &row[x..end],
                    &left[x..end],
                    &top[x..end],
                    &topleft[x..end],
                    chunk,
                )
            } else {
                kernel.predict_chunk_wide(
                    &row[x..end],
                    &left[x..end],
                    &top[x..end],
                    &topleft[x..end],
                    chunk,
                )
            };
            self.chunk(chunk, prefix)?;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.sink.finish(self.run)?;
        self.run = 0;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Area {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub skip: usize,
}

pub(super) fn process<S: Sink>(
    image: ImageView<'_>,
    area: Area,
    palette: Option<&Palette>,
    kernel: Kernel,
    processors: &mut [Processor<S>],
) -> Result<()> {
    let count = if palette.is_some() {
        1
    } else {
        usize::from(image.channels)
    };
    let mut previous = [[0i32; 257]; 4];
    let mut current = [[0i32; 257]; 4];
    for y in 0..area.height {
        if let Some(palette) = palette {
            for x in 0..area.width {
                current[0][x + 1] = i32::from(palette.index(image.pixel8(area.x + x, area.y + y)));
            }
        } else {
            kernel.load_transformed_row(image, area.x, area.y + y, area.width, &mut current)?;
        }
        for c in 0..count {
            current[c][0] = if y == 0 { 0 } else { previous[c][1] };
            previous[c][0] = current[c][0];
        }
        // Source palette sampling intentionally does not skip its warm-up row.
        if y >= area.skip || palette.is_some() {
            for c in 0..count {
                let row = &current[c];
                let top = if y == 0 {
                    &row[..area.width]
                } else {
                    &previous[c][1..1 + area.width]
                };
                let topleft = if y == 0 {
                    &row[..area.width]
                } else {
                    &previous[c][..area.width]
                };
                if image.bit_depth > 14 {
                    processors[c].predicted_wide_row(
                        kernel,
                        &row[1..1 + area.width],
                        &row[..area.width],
                        top,
                        topleft,
                    )?;
                } else {
                    processors[c].predicted_narrow_row(
                        kernel,
                        &row[1..1 + area.width],
                        &row[..area.width],
                        top,
                        topleft,
                    )?;
                }
            }
        }
        std::mem::swap(&mut current, &mut previous);
    }
    for processor in processors {
        processor.finish()?;
    }
    Ok(())
}

pub(super) fn palette_data(
    palette: &Palette,
    channels: u8,
    code: &PrefixCode,
    kernel: Kernel,
    writer: BitWriter,
) -> Result<BitWriter> {
    let width = palette.colors.len();
    let mut previous = vec![0; width + 1];
    let mut current = vec![0; width + 1];
    let mut processor = Processor::new(Encoder {
        code,
        writer,
        kernel,
    });
    for c in 0..channels {
        for (i, &color) in palette.colors.iter().enumerate() {
            current[i + 1] = ((color >> (8 * c)) & 255) as i32;
        }
        current[0] = previous[1];
        previous[0] = previous[1];
        let top = if c == 0 {
            &current[..width]
        } else {
            &previous[1..]
        };
        let topleft = if c == 0 {
            &current[..width]
        } else {
            &previous[..width]
        };
        processor.predicted_narrow_row(kernel, &current[1..], &current[..width], top, topleft)?;
        std::mem::swap(&mut current, &mut previous);
    }
    processor.finish()?;
    Ok(processor.sink.writer)
}
