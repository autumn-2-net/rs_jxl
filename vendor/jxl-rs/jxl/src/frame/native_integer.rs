// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Native samples are intercepted after Modular inverse transforms, before
//! stage_convert's normalization. Reference/patch semantics follow libjxl
//! dec_modular.cc, stage_blending.cc and dec_patch_dictionary.cc. Arithmetic
//! here is checked integer arithmetic; fractional render operations are rejected.

use crate::api::{JxlNativeIntegerFrame, JxlNativeIntegerPlane, JxlParallelRunner};
use crate::bit_reader::BitReader;
use crate::error::{Error, Result};
use crate::features::patches::{PatchBlendMode, PatchBlending, PatchesDictionary};
use crate::headers::color_encoding::ColorSpace;
use crate::headers::extra_channels::{ExtraChannel, ExtraChannelInfo};
use crate::headers::frame_header::{Encoding, FrameHeader, FrameType};
use crate::headers::{FileHeader, Orientation};
use crate::image::{Image, ImageDataType, OwnedRawImage};
use crate::util::sync::{Arc, Mutex};

use super::Frame;
use super::modular::{ModularStorage, ModularStreamId};

type ReferenceInfo = Option<((usize, usize), bool)>;

fn unsupported(reason: &'static str) -> Error {
    Error::NativeIntegerUnsupported(reason)
}

fn invalid(reason: &'static str) -> Error {
    Error::NativeIntegerInvalid(reason)
}

fn zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len)?;
    values.resize(len, T::default());
    Ok(values)
}

#[derive(Debug)]
struct IntegerImage {
    size: (usize, usize),
    // Three internal color planes even for gray, then extra channels.
    // Signed intermediate samples are needed for additive patch residuals.
    planes: Vec<Vec<i64>>,
}

impl IntegerImage {
    fn new(size: (usize, usize), channels: usize) -> Result<Self> {
        let area = size.0.checked_mul(size.1).ok_or(Error::SizeOverflow)?;
        let mut planes = Vec::new();
        planes.try_reserve_exact(channels)?;
        for _ in 0..channels {
            planes.push(zeroed(area)?);
        }
        Ok(Self { size, planes })
    }

    fn output(&self, file: &FileHeader, orientation: Orientation) -> Result<JxlNativeIntegerFrame> {
        let metadata = &file.image_metadata;
        let num_color_channels = if metadata.color_encoding.color_space == ColorSpace::Gray {
            1
        } else {
            3
        };
        let (width, height) = orientation.map_size(self.size);
        let mut planes = Vec::new();
        planes.try_reserve_exact(num_color_channels + metadata.extra_channel_info.len())?;
        for c in (0..num_color_channels).chain(3..self.planes.len()) {
            let bits_per_sample = if c < 3 {
                metadata.bit_depth
            } else {
                metadata.extra_channel_info[c - 3].bit_depth()
            }
            .bits_per_sample();
            let max = (1i64 << bits_per_sample) - 1;
            let mut samples = zeroed(self.planes[c].len())?;
            for (index, &sample) in self.planes[c].iter().enumerate() {
                if !(0..=max).contains(&sample) {
                    return Err(unsupported(
                        "output sample is outside its declared unsigned bit depth",
                    ));
                }
                let (x, y) = orientation
                    .display_pixel((index % self.size.0, index / self.size.0), self.size);
                samples[y * width + x] = sample as u32;
            }
            planes.push(JxlNativeIntegerPlane {
                bits_per_sample,
                samples,
            });
        }
        Ok(JxlNativeIntegerFrame {
            width,
            height,
            num_color_channels,
            planes,
        })
    }
}

#[derive(Debug)]
struct IntegerReference {
    image: Arc<IntegerImage>,
    before_color_transform: bool,
}

#[derive(Debug, Default)]
struct CaptureState {
    output: Option<JxlNativeIntegerFrame>,
    references: [Option<IntegerReference>; 4],
}

#[derive(Debug, Default)]
pub(crate) struct NativeIntegerCapture {
    state: Mutex<CaptureState>,
}

impl NativeIntegerCapture {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn take(&self) -> Option<JxlNativeIntegerFrame> {
        self.state.lock().unwrap().output.take()
    }

    pub(crate) fn abort(&self) {
        *self.state.lock().unwrap() = CaptureState::default();
    }

    pub(crate) fn begin_frame(&self) {
        self.state.lock().unwrap().output = None;
    }

    pub(crate) fn keep_slots(&self, keep: [bool; 8]) {
        let mut state = self.state.lock().unwrap();
        for (reference, &retain) in state.references.iter_mut().zip(&keep) {
            if !retain {
                *reference = None;
            }
        }
    }

    pub(crate) fn reference_info(&self) -> [ReferenceInfo; 4] {
        let state = self.state.lock().unwrap();
        std::array::from_fn(|i| {
            state.references[i]
                .as_ref()
                .map(|r| (r.image.size, r.before_color_transform))
        })
    }
}

struct FrameBufferState {
    image: Option<IntegerImage>,
    complete: Vec<Vec<bool>>,
}

pub(crate) struct NativeIntegerFrameBuffer {
    state: Mutex<FrameBufferState>,
}

impl NativeIntegerFrameBuffer {
    pub(crate) fn new(header: &FrameHeader, file: &FileHeader, render_spots: bool) -> Result<Self> {
        let metadata = &file.image_metadata;
        if header.encoding != Encoding::Modular || metadata.xyb_encoded || header.do_ycbcr {
            return Err(unsupported(
                "native output requires original-color Modular, not VarDCT/XYB/YCbCr",
            ));
        }
        if header.lf_level != 0 || header.has_lf_frame() || header.frame_type == FrameType::LFFrame
        {
            return Err(unsupported("LF preview frames"));
        }
        if header.upsampling != 1
            || !header.is444()
            || header.ec_upsampling.iter().any(|&u| u != 1)
            || metadata
                .extra_channel_info
                .iter()
                .any(|e| e.dim_shift() != 0)
        {
            return Err(unsupported("color or extra-channel resampling"));
        }
        if header.has_noise()
            || header.has_splines()
            || header.restoration_filter.gab
            || header.restoration_filter.epf_iters != 0
        {
            return Err(unsupported("noise, splines or restoration filters"));
        }
        for depth in std::iter::once(metadata.bit_depth)
            .chain(metadata.extra_channel_info.iter().map(|e| e.bit_depth()))
        {
            if depth.floating_point_sample() || !(1..=31).contains(&depth.bits_per_sample()) {
                return Err(unsupported(
                    "native capture requires integer 1..31-bit planes",
                ));
            }
        }
        if render_spots
            && metadata
                .extra_channel_info
                .iter()
                .any(|e| e.ec_type == ExtraChannel::SpotColor)
        {
            return Err(unsupported(
                "disable spot-color rendering for original integer samples",
            ));
        }
        let channels = 3 + metadata.extra_channel_info.len();
        let image = IntegerImage::new(header.size(), channels)?;
        let mut complete = Vec::new();
        complete.try_reserve_exact(channels)?;
        for _ in 0..channels {
            complete.push(zeroed(header.num_groups())?);
        }
        Ok(Self {
            state: Mutex::new(FrameBufferState {
                image: Some(image),
                complete,
            }),
        })
    }

    fn write_group<T: ImageDataType + Copy + Into<i64>>(
        &self,
        header: &FrameHeader,
        channel: usize,
        group: usize,
        image: Image<T>,
    ) -> Result<()> {
        let rect = header.group_rect(group);
        if image.size() != rect.size {
            return Err(invalid("unexpected Modular group dimensions"));
        }
        let mut state = self.state.lock().unwrap();
        let target = state
            .image
            .as_mut()
            .ok_or(invalid("frame already completed"))?;
        let plane = target
            .planes
            .get_mut(channel)
            .ok_or(invalid("invalid Modular output channel"))?;
        for y in 0..rect.size.1 {
            let start = (rect.origin.1 + y) * target.size.0 + rect.origin.0;
            for (dst, &src) in plane[start..start + rect.size.0]
                .iter_mut()
                .zip(image.row(y))
            {
                *dst = src.into();
            }
        }
        state.complete[channel][group] = true;
        Ok(())
    }

    fn consume(&self) -> Result<IntegerImage> {
        let mut state = self.state.lock().unwrap();
        if state.complete.iter().flatten().any(|&done| !done) {
            return Err(invalid("incomplete Modular plane at frame completion"));
        }
        state.image.take().ok_or(invalid("frame already completed"))
    }
}

impl Frame {
    pub(super) fn decode_native_integer_groups(
        &mut self,
        groups: &mut [(usize, Vec<(usize, BitReader)>)],
        runner: &mut dyn JxlParallelRunner,
    ) -> Result<()> {
        let Some(lf_global) = self.lf_global.as_mut() else {
            return Ok(());
        };
        let modular = &mut lf_global.modular_global;
        modular
            .set_pipeline_used_channels(&vec![true; 3 + self.header.num_extra_channels as usize]);
        for (group, passes) in groups.iter() {
            for (pass, _) in passes {
                modular.mark_final(2 + *pass, *group);
            }
        }
        modular.prepare_render(&self.header, |_, _, _| {});
        let transform_tasks = modular.num_ready_steps().div_ceil(3);
        let modular = &self.lf_global.as_ref().unwrap().modular_global;
        let capture = self.native_integer_frame.as_ref().unwrap();
        let write = |channel: usize, group: usize, complete: bool, raw: OwnedRawImage| {
            if !complete {
                return Err(invalid("non-final Modular output in native capture"));
            }
            match modular.storage() {
                ModularStorage::I16 => {
                    capture.write_group(&self.header, channel, group, Image::<i16>::from_raw(raw))
                }
                ModularStorage::I32 => {
                    capture.write_group(&self.header, channel, group, Image::<i32>::from_raw(raw))
                }
            }
        };
        // Ready LF/global transforms and HF decoding use the same dependency
        // scheduler as the renderer. Nothing passes through an f32 image.
        runner.run_ordered(groups.len() + transform_tasks, None, &|index| {
            if index < groups.len() {
                let (group, passes) = &groups[index];
                for (pass, br) in passes {
                    modular.read_stream(
                        ModularStreamId::ModularHF {
                            group: *group,
                            pass: *pass,
                        },
                        &self.header,
                        &self.lf_global.as_ref().unwrap().tree,
                        &mut br.clone(),
                        Some(&write),
                    )?;
                }
            } else {
                modular.run_transforms(&self.header, &write)?;
            }
            Ok(())
        })?;
        modular.validate_state_after_transforms();
        Ok(())
    }

    pub(crate) fn finish_native_integer_frame(
        &mut self,
        coalescing: bool,
        publish: bool,
    ) -> Result<()> {
        let Some(buffer) = &self.native_integer_frame else {
            return Ok(());
        };
        let capture = self.decoder_state.native_integer_capture.as_ref().unwrap();
        let mut layer = buffer.consume()?;
        let mut state = capture.state.lock().unwrap();
        let file = &self.decoder_state.file_header;
        let extras = &file.image_metadata.extra_channel_info;
        if self.header.has_patches() {
            apply_patches(
                &mut layer,
                &self.patches.read().unwrap(),
                &state.references,
                extras,
            )?;
        }
        let layer = Arc::new(layer);
        let needs_canvas = self.header.needs_blending()
            && (coalescing || (self.header.can_be_referenced && !self.header.save_before_ct));
        let composed = if needs_canvas {
            Arc::new(coalesce(&layer, &self.header, file, &state.references)?)
        } else {
            layer.clone()
        };
        // Construct and validate the output before changing references.
        let output = if publish {
            let orientation = if self.decoder_state.adjust_orientation {
                file.image_metadata.orientation
            } else {
                Orientation::Identity
            };
            Some(if coalescing { &composed } else { &layer }.output(file, orientation)?)
        } else {
            None
        };
        if self.header.can_be_referenced {
            state.references[self.header.save_as_reference as usize] = Some(IntegerReference {
                image: if self.header.save_before_ct {
                    layer
                } else {
                    composed
                },
                before_color_transform: self.header.save_before_ct,
            });
        }
        if publish {
            state.output = output;
        }
        Ok(())
    }
}

fn checked_add(a: i64, b: i64) -> Result<i64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("integer blend overflow"))
}

/// Returns exact integer results only. Binary opacity still requires the
/// alpha=0/alpha=0 unassociated case to produce zero, not hidden background RGB.
fn combine(
    current: &[i64],
    reference: &[i64],
    channel: usize,
    blend: &PatchBlending,
    extras: &[ExtraChannelInfo],
) -> Result<i64> {
    let old = current[channel];
    let other = reference[channel];
    match blend.mode {
        PatchBlendMode::None => Ok(old),
        PatchBlendMode::Replace => Ok(other),
        PatchBlendMode::Add => checked_add(old, other),
        PatchBlendMode::BlendAbove | PatchBlendMode::BlendBelow => {
            if channel < 3 && !extras.iter().any(|e| e.ec_type == ExtraChannel::Alpha) {
                return Ok(if blend.mode == PatchBlendMode::BlendAbove {
                    other
                } else {
                    old
                });
            }
            let alpha = extras
                .get(blend.alpha_channel)
                .ok_or_else(|| invalid("invalid alpha channel"))?;
            if alpha.ec_type != ExtraChannel::Alpha {
                return Err(unsupported(
                    "blending controlled by a non-alpha extra channel",
                ));
            }
            let a = 3 + blend.alpha_channel;
            let maximum = (1i64 << alpha.bit_depth().bits_per_sample()) - 1;
            let (top, bottom) = if blend.mode == PatchBlendMode::BlendAbove {
                (reference, current)
            } else {
                (current, reference)
            };
            let opacity = if blend.clamp {
                top[a].clamp(0, maximum)
            } else {
                top[a]
            };
            if opacity != 0 && opacity != maximum {
                return Err(unsupported(
                    "fractional alpha blending has no exact integer result contract",
                ));
            }
            if channel == a {
                return Ok(if opacity == maximum {
                    maximum
                } else {
                    bottom[a]
                });
            }
            if alpha.alpha_associated() {
                return if opacity == maximum {
                    Ok(top[channel])
                } else {
                    checked_add(top[channel], bottom[channel])
                };
            }
            if opacity == maximum {
                Ok(top[channel])
            } else if bottom[a] > 0 {
                Ok(bottom[channel])
            } else {
                Ok(0)
            }
        }
        PatchBlendMode::Mul
        | PatchBlendMode::AlphaWeightedAddAbove
        | PatchBlendMode::AlphaWeightedAddBelow => {
            Err(unsupported("multiply or alpha-weighted-add blending"))
        }
    }
}

fn blend_pixel(
    current: &[i64],
    reference: &[i64],
    blends: &[PatchBlending],
    extras: &[ExtraChannelInfo],
    result: &mut [i64],
) -> Result<()> {
    for (c, out) in result.iter_mut().enumerate() {
        *out = combine(
            current,
            reference,
            c,
            &blends[if c < 3 { 0 } else { c - 2 }],
            extras,
        )?;
    }
    // Color alpha blending writes its selected alpha after the EC operations,
    // even when that alpha's explicit EC blend mode is Replace/None.
    if matches!(
        blends[0].mode,
        PatchBlendMode::BlendAbove | PatchBlendMode::BlendBelow
    ) && extras.iter().any(|e| e.ec_type == ExtraChannel::Alpha)
    {
        let channel = 3 + blends[0].alpha_channel;
        let value = combine(current, reference, channel, &blends[0], extras)?;
        *result
            .get_mut(channel)
            .ok_or_else(|| invalid("invalid alpha channel"))? = value;
    }
    Ok(())
}

fn apply_patches(
    image: &mut IntegerImage,
    patches: &PatchesDictionary,
    references: &[Option<IntegerReference>; 4],
    extras: &[ExtraChannelInfo],
) -> Result<()> {
    let count = image.planes.len();
    let mut current = zeroed(count)?;
    let mut reference = zeroed(count)?;
    let mut result = zeroed(count)?;
    for (index, position) in patches.positions.iter().enumerate() {
        let region = &patches.ref_positions[position.ref_pos_idx];
        let source = references[region.reference]
            .as_ref()
            .ok_or(Error::PatchesInvalidReference(region.reference))?;
        if !source.before_color_transform {
            return Err(Error::PatchesPostColorTransform());
        }
        let blends =
            &patches.blendings[index * patches.blendings_stride..][..patches.blendings_stride];
        for y in 0..region.ysize.min(image.size.1.saturating_sub(position.y)) {
            for x in 0..region.xsize.min(image.size.0.saturating_sub(position.x)) {
                let dst = (position.y + y) * image.size.0 + position.x + x;
                let src = (region.y0 + y) * source.image.size.0 + region.x0 + x;
                for c in 0..count {
                    current[c] = image.planes[c][dst];
                    reference[c] = source.image.planes[c][src];
                }
                blend_pixel(&current, &reference, blends, extras, &mut result)?;
                for (plane, &value) in image.planes.iter_mut().zip(&result) {
                    plane[dst] = value;
                }
            }
        }
    }
    Ok(())
}

fn coalesce(
    layer: &IntegerImage,
    header: &FrameHeader,
    file: &FileHeader,
    references: &[Option<IntegerReference>; 4],
) -> Result<IntegerImage> {
    let size = (file.size.xsize() as usize, file.size.ysize() as usize);
    let mut output = IntegerImage::new(size, layer.planes.len())?;
    let blends: Vec<PatchBlending> = std::iter::once(&header.blending_info)
        .chain(&header.ec_blending_info)
        .map(PatchBlending::from)
        .collect();
    // Each extra plane may choose a different background reference.
    for (c, plane) in output.planes.iter_mut().enumerate() {
        let info = if c < 3 {
            &header.blending_info
        } else {
            &header.ec_blending_info[c - 3]
        };
        if let Some(reference) = &references[info.source as usize] {
            if reference.before_color_transform {
                return Err(Error::BlendingPreColorTransform(info.source as usize));
            }
            if reference.image.size != size {
                return Err(invalid("reference canvas size mismatch"));
            }
            plane.copy_from_slice(&reference.image.planes[c]);
        }
    }
    let count = layer.planes.len();
    let mut current = zeroed(count)?;
    let mut reference = zeroed(count)?;
    let mut result = zeroed(count)?;
    for y in 0..layer.size.1 {
        let target_y = i64::from(header.y0) + y as i64;
        if !(0..size.1 as i64).contains(&target_y) {
            continue;
        }
        for x in 0..layer.size.0 {
            let target_x = i64::from(header.x0) + x as i64;
            if !(0..size.0 as i64).contains(&target_x) {
                continue;
            }
            let src = y * layer.size.0 + x;
            let dst = target_y as usize * size.0 + target_x as usize;
            for c in 0..count {
                current[c] = layer.planes[c][src];
                reference[c] = output.planes[c][dst];
            }
            blend_pixel(
                &current,
                &reference,
                &blends,
                &file.image_metadata.extra_channel_info,
                &mut result,
            )?;
            for (plane, &value) in output.planes.iter_mut().zip(&result) {
                plane[dst] = value;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
#[path = "native_integer_tests.rs"]
mod tests;
