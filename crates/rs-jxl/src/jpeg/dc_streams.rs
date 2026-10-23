// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG DC/control-plane preparation from enc_modular.cc::{AddVarDCTDC,
//! AddACMetadata}, compressed_dc.cc::DequantDC, enc_frame.cc and the JPEG output
//! checks in dec_frame.cc/dec_group.cc, frozen at
//! b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Only nl_dc=false, jpeg_transcode=true, DCT8 and unit raw quantization are
//! prepared. DC CfL factors and extra DC precision are zero. No stream encoding,
//! entropy model, frame output, pixel decoding or coefficient mutation occurs.

use std::fmt;

use super::frame_header::{BlockRect, FrameGeometry};
use super::quantization::DcQuantization;
use super::transcode_layout::{DcContexts, TranscodeLayout};
use crate::modular::group::Channel;

/// Source Image::Create bit depth is nominal, not a signed sample range limit.
pub const JPEG_STREAM_BITS: u8 = 8;

#[derive(Clone, Copy, Debug)]
pub struct PrepareLimits {
    /// Sum of all returned Channel samples and qdc entries, per call.
    pub max_samples: usize,
    /// Logical owned sample bytes: i32 channels plus u8 qdc, per call.
    pub max_output_bytes: usize,
}

impl Default for PrepareLimits {
    fn default() -> Self {
        Self {
            max_samples: 1_048_576,
            max_output_bytes: 4 * 1024 * 1024,
        }
    }
}

#[derive(Debug)]
pub enum DcStreamError {
    Invalid(&'static str),
    Limit(&'static str),
    Quantization(crate::Error),
    DcOutOfRange {
        channel: usize,
        x: usize,
        y: usize,
        coefficient: i32,
    },
    DcNotReversible {
        channel: usize,
        x: usize,
        y: usize,
        adjusted: i32,
        encoded: i32,
    },
}

impl fmt::Display for DcStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid JPEG DC/control preparation: {reason}"),
            Self::Limit(resource) => write!(f, "JPEG DC/control preparation limit: {resource}"),
            Self::Quantization(error) => error.fmt(f),
            Self::DcOutOfRange {
                channel,
                x,
                y,
                coefficient,
            } => write!(
                f,
                "JPEG DC {coefficient} at JXL channel {channel}, block ({x}, {y}) would be clamped by source JPEG reconstruction"
            ),
            Self::DcNotReversible {
                channel,
                x,
                y,
                adjusted,
                encoded,
            } => write!(
                f,
                "JPEG DC at JXL channel {channel}, block ({x}, {y}) changes from adjusted {adjusted} to encoded {encoded} through source float scaling"
            ),
        }
    }
}

impl std::error::Error for DcStreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Quantization(error) => Some(error),
            _ => None,
        }
    }
}

type Result<T> = std::result::Result<T, DcStreamError>;

#[derive(Debug)]
pub struct DcGroup {
    /// Full-resolution frame block coordinates; padded MCU edges participate.
    pub rect: BlockRect,
    /// Modular order Y, X, B (internal JXL channels 1, 0, 2).
    /// Dimensions are subsampled, but source channel shifts stay zero.
    pub channels: [Channel; 3],
    /// Full-resolution, rect-local row-major DC context buckets.
    pub quant_dc: Vec<u8>,
    pub extra_dc_precision: u8,
}

/// Full-frame row-major maps, one i8 sample per 8x8 block tile. AC correlation
/// selection/eligibility belongs to the caller, not this preparation module.
#[derive(Clone, Copy, Debug)]
pub struct CflMaps<'a> {
    pub width: usize,
    pub height: usize,
    pub ytox: &'a [i8],
    pub ytob: &'a [i8],
}

#[derive(Debug)]
pub struct AcMetadataGroup {
    pub rect: BlockRect,
    /// YToX tiles, YToB tiles, flattened [strategy; raw_quant-1] rows, EPF.
    pub channels: [Channel; 4],
    /// DCT8 makes every block a first block: this equals rect.width*rect.height.
    pub ac_metadata_size: usize,
}

fn area(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .ok_or(DcStreamError::Limit("channel sample count overflow"))
}

fn check_budget(channel_samples: usize, qdc_samples: usize, limits: &PrepareLimits) -> Result<()> {
    let samples = channel_samples
        .checked_add(qdc_samples)
        .ok_or(DcStreamError::Limit("output sample count overflow"))?;
    let bytes = channel_samples
        .checked_mul(std::mem::size_of::<i32>())
        .and_then(|n| n.checked_add(qdc_samples))
        .ok_or(DcStreamError::Limit("output byte count overflow"))?;
    if samples > limits.max_samples {
        return Err(DcStreamError::Limit("output samples"));
    }
    if bytes > limits.max_output_bytes {
        return Err(DcStreamError::Limit("output sample bytes"));
    }
    Ok(())
}

fn allocate_channel(width: usize, height: usize, shift: i32) -> Result<Channel> {
    let count = area(width, height)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(count)
        .map_err(|_| DcStreamError::Limit("channel allocation"))?;
    samples.resize(count, 0);
    Ok(Channel {
        width,
        height,
        hshift: shift,
        vshift: shift,
        samples,
    })
}

fn dc_rect(geometry: FrameGeometry, group: usize) -> Result<BlockRect> {
    geometry
        .dc_group_rect(group)
        .ok_or(DcStreamError::Invalid("DC group index"))
}

fn channel_rect(rect: BlockRect, hshift: u8, vshift: u8) -> Result<BlockRect> {
    let h = 1usize << hshift;
    let v = 1usize << vshift;
    if !rect.x.is_multiple_of(h)
        || !rect.width.is_multiple_of(h)
        || !rect.y.is_multiple_of(v)
        || !rect.height.is_multiple_of(v)
    {
        return Err(DcStreamError::Invalid(
            "DC group is not aligned to chroma sampling",
        ));
    }
    Ok(BlockRect {
        x: rect.x >> hshift,
        y: rect.y >> vshift,
        width: rect.width >> hshift,
        height: rect.height >> vshift,
    })
}

// std::round in AddVarDCTDC is ties-away-from-zero, not nearest-even. Keep the
// original DC scaling and F16-rounded inverse multiply as separate f32 steps.
fn encoded_dc(adjusted: i32, original_scale: f32, rounded_inverse: f32) -> Result<i32> {
    let original = adjusted as f32 * original_scale;
    let encoded = (original * rounded_inverse).round();
    if !encoded.is_finite() || encoded < i32::MIN as f32 || encoded >= 2_147_483_648.0f32 {
        return Err(DcStreamError::Invalid(
            "source DC scaling is outside the finite i32 range",
        ));
    }
    Ok(encoded as i32)
}

fn luma_buckets(
    channel: &Channel,
    rect: BlockRect,
    shifts: (u8, u8),
    thresholds: &[i32],
    out: &mut [u8],
) {
    for y in 0..rect.height {
        for x in 0..rect.width {
            let encoded = channel.samples[(y >> shifts.1) * channel.width + (x >> shifts.0)];
            // JPEG contexts have empty X/B thresholds. DequantDC's general
            // X/B/Y mixed-radix bucket consequently reduces to this strict cut.
            out[y * rect.width + x] = thresholds.iter().filter(|&&t| encoded > t).count() as u8;
        }
    }
}

/// Prepare one source DC group with nominal depth 8 and extra precision zero.
/// `quantization` must be the F16-roundtripped factors for this exact layout;
/// `contexts` supplies the caller's frame-global source JPEG DC thresholds.
/// Bounds precede all output allocation. Any error discards the local group.
/// Values that source JPEG output would change/clamp are rejected, never fixed
/// by bypassing the specified float path or silently truncating input i16 DCs.
pub fn prepare_dc_group(
    layout: &TranscodeLayout<'_>,
    quantization: &DcQuantization,
    contexts: &DcContexts,
    geometry: FrameGeometry,
    dc_group: usize,
    limits: &PrepareLimits,
) -> Result<DcGroup> {
    if geometry.subsampling() != layout.subsampling() {
        return Err(DcStreamError::Invalid(
            "frame subsampling disagrees with layout",
        ));
    }
    let (width, height) = layout.dimensions_in_blocks();
    if geometry.dimensions_in_blocks() != (width as usize, height as usize) {
        return Err(DcStreamError::Invalid(
            "frame geometry disagrees with layout blocks",
        ));
    }
    let rect = dc_rect(geometry, dc_group)?;
    let hshift = layout.subsampling().horizontal_shifts();
    let vshift = layout.subsampling().vertical_shifts();
    let mut rects = [rect; 3];
    let mut samples = 0usize;
    for c in 0..3 {
        rects[c] = channel_rect(rect, hshift[c], vshift[c])?;
        let expected = (width >> hshift[c], height >> vshift[c]);
        if layout.planes()[c].dimensions_in_blocks() != expected {
            return Err(DcStreamError::Invalid(
                "plane dimensions disagree with source sampling",
            ));
        }
        samples = samples
            .checked_add(area(rects[c].width, rects[c].height)?)
            .ok_or(DcStreamError::Limit("DC sample count overflow"))?;
    }
    let qdc_samples = area(rect.width, rect.height)?;
    check_budget(samples, qdc_samples, limits)?;
    let expected =
        DcQuantization::from_layout(layout.planes().each_ref().map(|p| p.quantization()))
            .map_err(DcStreamError::Quantization)?;
    if *quantization != expected {
        return Err(DcStreamError::Invalid(
            "DC quantization does not match the layout's F16 roundtrip",
        ));
    }
    let thresholds = contexts.luma_thresholds();
    if thresholds.len() > 7 || thresholds.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(DcStreamError::Invalid(
            "JPEG luma DC thresholds must be sorted, unique and at most seven",
        ));
    }
    let mut channels = [
        allocate_channel(rects[1].width, rects[1].height, 0)?,
        allocate_channel(rects[0].width, rects[0].height, 0)?,
        allocate_channel(rects[2].width, rects[2].height, 0)?,
    ];
    let mut quant_dc = Vec::new();
    quant_dc
        .try_reserve_exact(qdc_samples)
        .map_err(|_| DcStreamError::Limit("qdc allocation"))?;
    quant_dc.resize(qdc_samples, 0);
    for (modular, c) in [1, 0, 2].into_iter().enumerate() {
        let plane = &layout.planes()[c];
        let r = rects[c];
        for y in 0..r.height {
            for x in 0..r.width {
                let absolute_x = r.x + x;
                let absolute_y = r.y + y;
                let adjusted = plane
                    .adjusted_dc(absolute_x as u32, absolute_y as u32)
                    .ok_or(DcStreamError::Invalid(
                        "DC block lies outside the source plane",
                    ))?;
                let original = if plane.is_synthetic_zero() {
                    0
                } else {
                    adjusted - plane.quantization().dc_offset()
                };
                if !(-2047..=2047).contains(&original) {
                    return Err(DcStreamError::DcOutOfRange {
                        channel: c,
                        x: absolute_x,
                        y: absolute_y,
                        coefficient: original,
                    });
                }
                let encoded = encoded_dc(
                    adjusted,
                    plane.quantization().inverse_dc_quantization(),
                    quantization.inverse_dc_quants()[c],
                )?;
                if encoded != adjusted {
                    return Err(DcStreamError::DcNotReversible {
                        channel: c,
                        x: absolute_x,
                        y: absolute_y,
                        adjusted,
                        encoded,
                    });
                }
                channels[modular].samples[y * r.width + x] = encoded;
            }
        }
    }
    luma_buckets(
        &channels[0],
        rect,
        (hshift[1], vshift[1]),
        thresholds,
        &mut quant_dc,
    );
    Ok(DcGroup {
        rect,
        channels,
        quant_dc,
        extra_dc_precision: 0,
    })
}

/// Prepare source AddACMetadata's four channels for one DC group. CfL maps
/// describe the entire frame's ceil(blocks/8) tile grid, including padded edges.
/// The caller must select zero maps when AC CfL is disabled. DC CfL is unrelated
/// and remains zero. Strategies are DCT8=0, raw quant=1 (wire value 0), EPF=0.
pub fn prepare_ac_metadata_group(
    geometry: FrameGeometry,
    dc_group: usize,
    cfl_maps: CflMaps<'_>,
    limits: &PrepareLimits,
) -> Result<AcMetadataGroup> {
    let rect = dc_rect(geometry, dc_group)?;
    let (width, height) = geometry.dimensions_in_blocks();
    if (cfl_maps.width, cfl_maps.height) != (width.div_ceil(8), height.div_ceil(8)) {
        return Err(DcStreamError::Invalid(
            "CfL map dimensions disagree with frame tiles",
        ));
    }
    let count = area(cfl_maps.width, cfl_maps.height)?;
    if cfl_maps.ytox.len() != count || cfl_maps.ytob.len() != count {
        return Err(DcStreamError::Invalid("CfL map storage length"));
    }
    let tile_width = rect.width.div_ceil(8);
    let tile_height = rect.height.div_ceil(8);
    let tiles = area(tile_width, tile_height)?;
    let blocks = area(rect.width, rect.height)?;
    let samples = blocks
        .checked_mul(3)
        .and_then(|n| tiles.checked_mul(2).and_then(|t| n.checked_add(t)))
        .ok_or(DcStreamError::Limit("AC metadata sample count overflow"))?;
    check_budget(samples, 0, limits)?;
    let mut channels = [
        allocate_channel(tile_width, tile_height, 3)?,
        allocate_channel(tile_width, tile_height, 3)?,
        allocate_channel(blocks, 2, 0)?,
        allocate_channel(rect.width, rect.height, 0)?,
    ];
    for (channel, map) in channels[..2].iter_mut().zip([cfl_maps.ytox, cfl_maps.ytob]) {
        for y in 0..tile_height {
            let source = ((rect.y >> 3) + y) * cfl_maps.width + (rect.x >> 3);
            for (out, &value) in channel.samples[y * tile_width..(y + 1) * tile_width]
                .iter_mut()
                .zip(&map[source..source + tile_width])
            {
                *out = i32::from(value);
            }
        }
    }
    Ok(AcMetadataGroup {
        rect,
        channels,
        ac_metadata_size: blocks,
    })
}

#[cfg(test)]
#[path = "dc_streams_tests.rs"]
mod tests;
