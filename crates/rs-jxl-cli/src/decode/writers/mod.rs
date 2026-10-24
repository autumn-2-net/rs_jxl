//! Bounded serializers; file publication belongs to outputs, never a codec.
mod buffer;
#[path = "exr.rs"]
mod exr_writer;
mod npy;
#[path = "png.rs"]
mod png_writer;
mod portable;

use crate::decode::{
    DecodedBox, DecodedFrame, DecodedImage, ExtraChannel, JxlBasicInfo, JxlBitDepth,
    MAX_OUTPUT_BYTES, OutputFormat, PixelData, PixelLayout, Result, SampleFormat,
};
pub(in crate::decode) use buffer::Output;
pub(in crate::decode) use npy::write_npy;
#[cfg(test)]
pub(in crate::decode) use png_writer::png_depth;
#[cfg(test)]
pub(in crate::decode) use portable::write_portable;
#[cfg(test)]
pub(in crate::decode) use portable::write_portable_depth;
use rs_jxl::decode::{AlphaMode, JxlColorProfile};
use std::io::Write;

#[derive(Clone, Copy)]
pub(in crate::decode) struct Raster<'a> {
    pub info: &'a JxlBasicInfo,
    pub frames: &'a [DecodedFrame],
    pub output_profile: Option<&'a JxlColorProfile>,
    pub boxes: &'a [DecodedBox],
    pub alpha_mode: AlphaMode,
    pub output_intensity_target: f32,
}

impl<'a> From<&'a DecodedImage> for Raster<'a> {
    fn from(image: &'a DecodedImage) -> Self {
        Self {
            info: &image.info,
            frames: &image.frames,
            output_profile: image.output_profile.as_ref(),
            boxes: &image.boxes,
            alpha_mode: AlphaMode::AsIs,
            output_intensity_target: image.output_intensity_target,
        }
    }
}

impl Raster<'_> {
    pub(in crate::decode) fn alpha_associated(&self) -> bool {
        match self.alpha_mode {
            AlphaMode::Premultiplied => true,
            AlphaMode::Unpremultiplied => false,
            AlphaMode::AsIs => self
                .info
                .extra_channels
                .iter()
                .find(|ec| ec.ec_type == ExtraChannel::Alpha)
                .is_some_and(|ec| ec.alpha_associated),
        }
    }
}

pub(in crate::decode) fn validate_frames(frames: &[DecodedFrame]) -> Result<&DecodedFrame> {
    let first = frames.first().ok_or("decoder returned no frames")?;
    for frame in frames {
        let pixels = frame
            .width
            .checked_mul(frame.height)
            .filter(|&n| n > 0)
            .ok_or("invalid frame dimensions")?;
        let samples = pixels
            .checked_mul(frame.layout.channels())
            .ok_or("sample count overflow")?;
        if frame.pixels.len() != samples {
            return Err("decoded frame sample count does not match its dimensions/layout".into());
        }
        if frame.width != first.width
            || frame.height != first.height
            || frame.layout != first.layout
            || dtype(&frame.pixels) != dtype(&first.pixels)
            || frame.extra_planes.len() != first.extra_planes.len()
        {
            return Err("output requires matching frames; use --output-frames or --frame".into());
        }
        if frame.extra_planes.iter().any(|p| p.len() != pixels) {
            return Err("extra planes must have full frame resolution".into());
        }
    }
    Ok(first)
}

pub(in crate::decode) fn dtype(pixels: &PixelData) -> &'static str {
    match pixels {
        PixelData::U8(_) => "|u1",
        PixelData::U16(_) => "<u2",
        PixelData::U32(_) => "<u4",
        PixelData::F16(_) => "<f2",
        PixelData::F32(_) => "<f4",
    }
}

fn write_sample(output: &mut impl Write, pixels: &PixelData, index: usize) -> Result<()> {
    match pixels {
        PixelData::U8(values) => output.write_all(&[values[index]])?,
        PixelData::U16(values) | PixelData::F16(values) => {
            output.write_all(&values[index].to_le_bytes())?
        }
        PixelData::U32(values) => output.write_all(&values[index].to_le_bytes())?,
        PixelData::F32(values) => output.write_all(&values[index].to_le_bytes())?,
    }
    Ok(())
}

pub(in crate::decode) fn integer_depth(depth: &JxlBitDepth, format: SampleFormat) -> Result<u8> {
    match format {
        SampleFormat::U8 => Ok(8),
        SampleFormat::U16 => Ok(16),
        SampleFormat::U8Bits(n) if (1..=8).contains(&n) => Ok(n),
        SampleFormat::U16Bits(n) if (1..=16).contains(&n) => Ok(n),
        SampleFormat::Native => match depth {
            JxlBitDepth::Int { bits_per_sample } if (1..=16).contains(bits_per_sample) => {
                Ok(*bits_per_sample as u8)
            }
            _ => Err("integer output needs explicit --bits 8/16 for float or wide sources".into()),
        },
        _ => Err("integer output requires integer samples".into()),
    }
}

pub(in crate::decode) fn serialize_raster(
    image: Raster<'_>,
    format: OutputFormat,
    selected: bool,
    keep_orientation: bool,
    samples: SampleFormat,
    extras_separate: bool,
    limit: usize,
) -> Result<Vec<u8>> {
    let first = validate_frames(image.frames)?;
    if image
        .frames
        .iter()
        .any(|f| f.extra_planes.len() != image.info.extra_channels.len())
    {
        return Err("decoded extra planes do not match channel metadata".into());
    }
    let mut output = Output::with_limit(limit);
    if format == OutputFormat::Png && samples == SampleFormat::Native {
        let color_depth = integer_depth(&image.info.bit_depth, samples)?;
        if color_depth != first.pixels.bits()
            || image.info.extra_channels.iter().any(|ec| {
                ec.ec_type == ExtraChannel::Alpha
                    && integer_depth(&ec.bit_depth, samples).ok() != Some(first.pixels.bits())
            })
        {
            return Err("PNG needs full-range 8/16-bit samples; use --bits 8/16 when exporting a PNG preview alongside native output".into());
        }
    }
    match format {
        OutputFormat::Png => png_writer::write_png(
            &mut output,
            &image,
            selected,
            keep_orientation,
            extras_separate,
        )?,
        OutputFormat::Npy => write_npy(&mut output, image.frames)?,
        OutputFormat::Exr if image.frames.len() == 1 => exr_writer::write_exr(&mut output, &image)?,
        OutputFormat::Pam if image.frames.len() == 1 => {
            portable::write_pam(&mut output, &image, samples)?
        }
        OutputFormat::Ppm
        | OutputFormat::Pgm
        | OutputFormat::Pnm
        | OutputFormat::Pfm
        | OutputFormat::Pgx
            if image.frames.len() == 1 =>
        {
            let depth = if format == OutputFormat::Pfm {
                32
            } else {
                integer_depth(&image.info.bit_depth, samples)?
            };
            portable::write_portable_depth(&mut output, first, format, depth, extras_separate)?;
        }
        OutputFormat::Jpg => {
            return Err("JPEG reconstruction must not use pixel serialization".into());
        }
        _ if image.frames.len() > 1 => {
            return Err(
                "this format cannot store multiple frames; use --output-frames or --frame".into(),
            );
        }
        _ => return Err("metadata output must not use pixel serialization".into()),
    }
    Ok(output.into_bytes())
}

pub(in crate::decode) fn serialize_plane(
    frame: &DecodedFrame,
    index: usize,
    format: OutputFormat,
    depth: u8,
    limit: usize,
) -> Result<Vec<u8>> {
    validate_frames(std::slice::from_ref(frame))?;
    let plane = frame.extra_planes.get(index).ok_or("missing extra plane")?;
    let mut output = Output::with_limit(limit);
    if format == OutputFormat::Png {
        png_writer::write_plane_png(&mut output, frame.width, frame.height, plane, depth)?;
    } else {
        portable::write_plane(&mut output, frame.width, frame.height, plane, format, depth)?;
    }
    Ok(output.into_bytes())
}
