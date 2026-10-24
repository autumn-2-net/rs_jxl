// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! JPEG integer capture, ported from libjxl dec_frame.cc/dec_group.cc/dec_modular.cc at
//! b866b3defec13494bfbaf705db93581cda118e37, before dequantization and IDCT.
//! Unlike the C++ fallback conversions, fractional DC, clamping and arithmetic
//! overflow are rejected rather than used to manufacture JPEG coefficients.

use crate::api::{
    JxlJpegCoefficientCapture, JxlJpegCoefficients, JxlJpegComponent, JxlParallelRunner,
};
use crate::bit_reader::BitReader;
use crate::error::{Error, Result};
use crate::headers::bit_depth::BitDepth;
use crate::headers::color_encoding::ColorSpace;
use crate::headers::frame_header::{Encoding, FrameHeader, FrameType};
use crate::image::{ImageDataType, ImageRect, RawImageRect};
use crate::util::sync::{Arc, Mutex};

use super::color_correlation_map::DEFAULT_COLOR_FACTOR;
use super::group::decode_vardct_group;
use super::modular::{ModularStorage, ModularStreamId};
use super::quant_weights::QuantEncoding;
use super::{ColorCorrelationParams, DataStatus, Frame};

const FIXED_ONE: i32 = 1 << 11;
const ROUND: i32 = FIXED_ONE / 2;

pub(crate) fn unsupported(reason: &'static str) -> Error {
    Error::ApiUsageError(reason)
}

#[derive(Debug)]
struct CaptureState {
    output: Option<JxlJpegCoefficients>,
    quantization: Option<[[u16; 64]; 3]>,
    lf_done: Vec<bool>,
    hf_done: Vec<bool>,
    hf_passes: Vec<usize>,
    complete: bool,
}

#[derive(Debug)]
pub(crate) struct JpegCapture {
    state: Mutex<CaptureState>,
}

fn zeroed_coefficients(blocks: usize) -> Result<Vec<i16>> {
    let count = blocks.checked_mul(64).ok_or(Error::SizeOverflow)?;
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    result.resize(count, 0);
    Ok(result)
}

impl Frame {
    pub fn enable_jpeg_coefficient_capture(&mut self) -> Result<JxlJpegCoefficientCapture> {
        if self.lf_global.is_some() || self.decoder_state.jpeg_capture.is_some() {
            return Err(unsupported(
                "JPEG capture must be enabled once, before frame sections",
            ));
        }
        let header = &self.header;
        let file = &self.decoder_state.file_header;
        let metadata = &file.image_metadata;
        if header.encoding != Encoding::VarDCT || metadata.xyb_encoded {
            return Err(unsupported(
                "JPEG capture requires original-color VarDCT, not Modular/XYB",
            ));
        }
        if header.frame_type != FrameType::RegularFrame
            || !header.is_last
            || header.lf_level != 0
            || header.has_lf_frame()
            || metadata.animation.is_some()
            || self.frame_counters != (1, 0)
        {
            return Err(unsupported(
                "JPEG capture supports only one final frame, without LF frames",
            ));
        }
        if header.needs_blending()
            || header.upsampling != 1
            || header.num_extra_channels != 0
            || header.size() != (file.size.xsize() as usize, file.size.ysize() as usize)
            || metadata.bit_depth != BitDepth::integer_samples(8)
            || header.width > 65535
            || header.height > 65535
        {
            return Err(unsupported(
                "JPEG capture requires a full-canvas 8-bit frame without extra channels",
            ));
        }
        if header.has_noise() || header.has_patches() || header.has_splines() {
            return Err(unsupported(
                "JPEG capture does not support noise, patches or splines",
            ));
        }
        let grayscale = metadata.color_encoding.color_space == ColorSpace::Gray;
        if !header.do_ycbcr && !header.is444() {
            return Err(unsupported("JPEG capture does not support subsampled RGB"));
        }
        if grayscale && !header.is444() {
            return Err(unsupported(
                "JPEG capture does not support subsampled grayscale",
            ));
        }
        let inner = Arc::new(JpegCapture::new(header, grayscale)?);
        self.decoder_state.jpeg_capture = Some(inner.clone());
        Ok(JxlJpegCoefficientCapture { inner })
    }

    /// Decode coefficient sections without allocating or running a pixel pipeline.
    pub fn decode_jpeg_groups(
        &mut self,
        groups: &mut [(usize, Vec<(usize, BitReader)>)],
        parallel_runner: &mut dyn JxlParallelRunner,
    ) -> Result<()> {
        if !self.is_jpeg_reconstruction() {
            return Err(unsupported("JPEG capture is not enabled"));
        }
        // A group owns its accumulator; reject duplicate work before parallel dispatch.
        let mut seen = std::collections::HashSet::new();
        for (group, _) in groups.iter() {
            if *group >= self.header.num_groups() || !seen.insert(*group) {
                return Err(unsupported("Invalid or duplicate JPEG coefficient group"));
            }
        }
        parallel_runner.run_ordered(groups.len(), Some(groups.len().max(1)), &|i| {
            let (group, passes) = &groups[i];
            let mut passes = passes.clone();
            self.decode_jpeg_group(*group, &mut passes)
        })?;
        for (group, passes) in groups.iter() {
            if let Some((pass, _)) = passes.last() {
                let status = if *pass + 1 == self.header.passes.num_passes as usize {
                    DataStatus::Final
                } else {
                    DataStatus::Partial
                };
                for c in 0..3 {
                    self.group_status.update_status(*group, c, status);
                }
            }
        }
        Ok(())
    }

    pub(super) fn decode_jpeg_group(
        &self,
        group: usize,
        passes: &mut [(usize, BitReader)],
    ) -> Result<()> {
        if passes.is_empty() {
            return Ok(());
        }
        let lf_global = self
            .lf_global
            .as_ref()
            .ok_or(unsupported("Missing JPEG LF global"))?;
        let hf_global = self
            .hf_global
            .as_ref()
            .ok_or(unsupported("Missing JPEG HF global"))?;
        let mut buffers = self.buffer_recycler.vardct_buffers.get();
        // No ensure_allocated(): the pixel transform scratch is not needed here.
        decode_vardct_group(
            group,
            passes,
            &self.header,
            lf_global,
            hf_global,
            self.hf_meta
                .as_ref()
                .ok_or(unsupported("Missing JPEG HF metadata"))?,
            self.lf_image
                .as_ref()
                .ok_or(unsupported("Missing JPEG LF image"))?,
            &self
                .decoder_state
                .file_header
                .transform_data
                .opsin_inverse_matrix
                .quant_biases,
            &mut None,
            &mut buffers,
            self.decoder_state.jpeg_capture.as_deref(),
        )?;
        for (pass, br) in passes.iter_mut() {
            lf_global.modular_global.read_stream(
                ModularStreamId::ModularHF { group, pass: *pass },
                &self.header,
                &lf_global.tree,
                br,
                None,
            )?;
        }
        Ok(())
    }
}

impl JpegCapture {
    fn new(header: &FrameHeader, grayscale: bool) -> Result<Self> {
        let channels: &[usize] = if grayscale {
            &[1]
        } else if header.do_ycbcr {
            &[1, 0, 2]
        } else {
            &[0, 1, 2]
        };
        let mut components = Vec::new();
        for &c in channels {
            let width_in_blocks = header.size_blocks().0 >> header.hshift(c);
            let height_in_blocks = header.size_blocks().1 >> header.vshift(c);
            components.push(JxlJpegComponent {
                jxl_channel: c,
                horizontal_sampling_factor: 1 << (header.maxhs as usize - header.hshift(c)),
                vertical_sampling_factor: 1 << (header.maxvs as usize - header.vshift(c)),
                width_in_blocks,
                height_in_blocks,
                quantization: [0; 64],
                coefficients: zeroed_coefficients(
                    width_in_blocks
                        .checked_mul(height_in_blocks)
                        .ok_or(Error::SizeOverflow)?,
                )?,
            });
        }
        Ok(Self {
            state: Mutex::new(CaptureState {
                output: Some(JxlJpegCoefficients {
                    width: header.width,
                    height: header.height,
                    is_ycbcr: header.do_ycbcr,
                    components,
                }),
                quantization: None,
                lf_done: vec![false; header.num_lf_groups()],
                hf_done: vec![false; header.num_groups()],
                hf_passes: vec![0; header.num_groups()],
                complete: false,
            }),
        })
    }
    pub(crate) fn validate_lf_params(&self, params: &ColorCorrelationParams) -> Result<()> {
        if params.color_factor != DEFAULT_COLOR_FACTOR
            || params.base_correlation_x != 0.0
            || params.base_correlation_b != 0.0
            || params.ytox_lf != 0
            || params.ytob_lf != 0
        {
            return Err(unsupported(
                "JPEG capture requires JPEG-compatible color correlation (no DC CfL)",
            ));
        }
        // dec_frame.cc clears DC multipliers for JPEG output; the global and
        // per-block quantizers affect entropy contexts, not recovered integers.
        Ok(())
    }

    pub(crate) fn validate_passes(
        &self,
        header: &FrameHeader,
        group: usize,
        passes: impl Iterator<Item = usize>,
    ) -> Result<bool> {
        let state = self.state.lock().unwrap();
        let mut next = *state
            .hf_passes
            .get(group)
            .ok_or(unsupported("Invalid JPEG group"))?;
        if state.complete || state.output.is_none() || state.hf_done[group] {
            return Err(unsupported("JPEG coefficient group is no longer active"));
        }
        for pass in passes {
            if pass != next || pass >= header.passes.num_passes as usize {
                return Err(unsupported(
                    "JPEG coefficient passes must be contiguous and unique",
                ));
            }
            next += 1;
        }
        Ok(next == header.passes.num_passes as usize)
    }

    pub(crate) fn record_passes(&self, group: usize, count: usize) {
        self.state.lock().unwrap().hf_passes[group] += count;
    }

    pub(crate) fn set_quantization(&self, encoding: &QuantEncoding) -> Result<()> {
        let QuantEncoding::Raw { qtable, qtable_den } = encoding else {
            return Err(unsupported(
                "JPEG capture requires a RAW DCT8 JPEG quantization table",
            ));
        };
        if !qtable_den.is_finite()
            || (qtable_den - 1.0 / 2040.0).abs() > 1e-8
            || qtable.len() != 192
        {
            return Err(unsupported(
                "JPEG capture requires the JPEG RAW quantization denominator 1/2040",
            ));
        }
        let mut tables = [[0; 64]; 3];
        for c in 0..3 {
            for k in 0..64 {
                let value = qtable[c * 64 + (k % 8) * 8 + k / 8];
                if !(1..=65535).contains(&value) {
                    return Err(unsupported(
                        "JPEG capture found an invalid JPEG quantization entry",
                    ));
                }
                tables[c][k] = value as u16;
            }
        }
        let mut state = self.state.lock().unwrap();
        if let Some(output) = &mut state.output {
            for component in &mut output.components {
                component.quantization = tables[component.jxl_channel];
            }
        }
        state.quantization = Some(tables);
        Ok(())
    }

    pub(crate) fn capture_lf(
        &self,
        header: &FrameHeader,
        group: usize,
        input: [RawImageRect<'_>; 3],
        storage: ModularStorage,
        extra_precision: u64,
    ) -> Result<()> {
        if extra_precision > 3 {
            return Err(unsupported("Invalid JPEG extra DC precision"));
        }
        match storage {
            ModularStorage::I16 => self.capture_lf_impl(
                header,
                group,
                input.map(ImageRect::<i16>::from_raw),
                extra_precision,
            ),
            ModularStorage::I32 => self.capture_lf_impl(
                header,
                group,
                input.map(ImageRect::<i32>::from_raw),
                extra_precision,
            ),
        }
    }

    fn capture_lf_impl<T: ImageDataType + Into<i32> + Copy>(
        &self,
        header: &FrameHeader,
        group: usize,
        input: [ImageRect<'_, T>; 3],
        extra_precision: u64,
    ) -> Result<()> {
        let rect = header.lf_group_rect(group);
        let mut state = self.state.lock().unwrap();
        let output = state
            .output
            .as_mut()
            .ok_or(unsupported("JPEG capture is no longer active"))?;
        for component in &mut output.components {
            let c = component.jxl_channel;
            let source = &input[if c < 2 { c ^ 1 } else { c }];
            let x0 = rect.origin.0 >> header.hshift(c);
            let y0 = rect.origin.1 >> header.vshift(c);
            let width = rect.size.0 >> header.hshift(c);
            let height = rect.size.1 >> header.vshift(c);
            for y in 0..height {
                for (x, &value) in source.row(y)[..width].iter().enumerate() {
                    // Retain the RGB DC offset until the RAW quantization tables arrive.
                    let value = value.into();
                    let divisor = 1i32 << extra_precision;
                    if value % divisor != 0 {
                        return Err(unsupported(
                            "JPEG capture DC precision would require truncation",
                        ));
                    }
                    let dc = i16::try_from(value / divisor)
                        .map_err(|_| unsupported("JPEG capture DC coefficient is out of range"))?;
                    component.coefficients[((y0 + y) * component.width_in_blocks + x0 + x) * 64] =
                        dc;
                }
            }
        }
        state.lf_done[group] = true;
        Ok(())
    }

    pub(crate) fn new_group(&self, header: &FrameHeader, group: usize) -> Result<JpegGroupCapture> {
        let state = self.state.lock().unwrap();
        let quantization = state
            .quantization
            .ok_or(unsupported("JPEG capture has no quantization tables"))?;
        let output = state
            .output
            .as_ref()
            .ok_or(unsupported("JPEG capture is no longer active"))?;
        let mut active = [false; 3];
        for component in &output.components {
            active[component.jxl_channel] = true;
        }
        let rect = header.block_group_rect(group);
        let mut coefficients = [Vec::new(), Vec::new(), Vec::new()];
        for c in 0..3 {
            if active[c] {
                coefficients[c] = zeroed_coefficients(
                    (rect.size.0 >> header.hshift(c)) * (rect.size.1 >> header.vshift(c)),
                )?;
            }
        }
        Ok(JpegGroupCapture {
            quantization,
            active,
            coefficients,
        })
    }

    pub(crate) fn store_group(
        &self,
        header: &FrameHeader,
        group: usize,
        data: JpegGroupCapture,
    ) -> Result<()> {
        let rect = header.block_group_rect(group);
        let mut state = self.state.lock().unwrap();
        let output = state
            .output
            .as_mut()
            .ok_or(unsupported("JPEG capture is no longer active"))?;
        for component in &mut output.components {
            let c = component.jxl_channel;
            let x0 = rect.origin.0 >> header.hshift(c);
            let y0 = rect.origin.1 >> header.vshift(c);
            let width = rect.size.0 >> header.hshift(c);
            let height = rect.size.1 >> header.vshift(c);
            for y in 0..height {
                for x in 0..width {
                    let to = ((y0 + y) * component.width_in_blocks + x0 + x) * 64;
                    let from = (y * width + x) * 64;
                    component.coefficients[to + 1..to + 64]
                        .copy_from_slice(&data.coefficients[c][from + 1..from + 64]);
                }
            }
        }
        state.hf_done[group] = true;
        Ok(())
    }

    pub(crate) fn finish(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.complete {
            return Ok(());
        }
        if state.quantization.is_none()
            || state.lf_done.contains(&false)
            || state.hf_done.contains(&false)
        {
            return Err(unsupported(
                "JPEG capture did not decode every coefficient group",
            ));
        }
        let output = state
            .output
            .as_mut()
            .ok_or(unsupported("JPEG capture is no longer active"))?;
        for component in &output.components {
            let offset = if output.is_ycbcr {
                0
            } else {
                1024 / i32::from(component.quantization[0])
            };
            // Validate all DCs before modifying any of them.
            for block in component.coefficients.chunks_exact(64) {
                if !(-2047..=2047).contains(&(i32::from(block[0]) - offset)) {
                    return Err(unsupported(
                        "JPEG capture DC reconstruction would require clamping",
                    ));
                }
            }
        }
        for component in &mut output.components {
            let offset = if output.is_ycbcr {
                0
            } else {
                1024 / i32::from(component.quantization[0])
            };
            for block in component.coefficients.chunks_exact_mut(64) {
                block[0] = (i32::from(block[0]) - offset) as i16;
            }
        }
        state.complete = true;
        Ok(())
    }

    pub(crate) fn abort(&self) {
        let mut state = self.state.lock().unwrap();
        if !state.complete {
            state.output = None;
        }
    }

    pub(crate) fn take(&self) -> Option<JxlJpegCoefficients> {
        let mut state = self.state.lock().unwrap();
        if state.complete {
            state.output.take()
        } else {
            None
        }
    }
}

pub(crate) struct JpegGroupCapture {
    quantization: [[u16; 64]; 3],
    active: [bool; 3],
    coefficients: [Vec<i16>; 3],
}

fn restore_ac(residual: i32, luma: i32, q_y: u16, q_c: u16, factor: i8) -> Result<i16> {
    let ratio = FIXED_ONE * i32::from(q_y) / i32::from(q_c);
    let scale = i32::from(factor) * FIXED_ONE / DEFAULT_COLOR_FACTOR as i32;
    let scaled = ratio
        .checked_mul(scale)
        .and_then(|x| x.checked_add(ROUND))
        .ok_or(unsupported("JPEG capture CfL scale overflow"))?
        >> 11;
    let prediction = luma
        .checked_mul(scaled)
        .and_then(|x| x.checked_add(ROUND))
        .ok_or(unsupported("JPEG capture CfL prediction overflow"))?
        >> 11;
    let value = residual
        .checked_add(prediction)
        .ok_or(unsupported("JPEG capture AC coefficient overflow"))?;
    if !(-4095..=4095).contains(&value) {
        return Err(unsupported(
            "JPEG capture AC coefficient is outside the 8-bit JPEG range",
        ));
    }
    Ok(value as i16)
}

impl JpegGroupCapture {
    pub(crate) fn capture_block(
        &mut self,
        header: &FrameHeader,
        group: usize,
        bx: usize,
        by: usize,
        raw: &[[i32; 64]; 3],
        factors: [i8; 3],
    ) -> Result<()> {
        let rect = header.block_group_rect(group);
        for c in [1, 0, 2] {
            let hs = header.hshift(c);
            let vs = header.vshift(c);
            if !self.active[c] || ((bx >> hs) << hs) != bx || ((by >> vs) << vs) != by {
                continue;
            }
            let offset = ((by >> vs) * (rect.size.0 >> hs) + (bx >> hs)) * 64;
            for k in 1..64 {
                let transposed = (k % 8) * 8 + k / 8;
                let factor = if header.is444() && c != 1 {
                    factors[c]
                } else {
                    0
                };
                self.coefficients[c][offset + k] = restore_ac(
                    raw[c][transposed],
                    raw[1][transposed],
                    self.quantization[1][k],
                    self.quantization[c][k],
                    factor,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "jpeg_tests.rs"]
mod tests;
