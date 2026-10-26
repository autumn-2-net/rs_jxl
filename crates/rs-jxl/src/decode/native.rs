//! Exact integer output is captured before the floating-point render pipeline.
use super::{
    AlphaMode, DecodeOptions, DecodedFrame, ExtraChannel, JxlBasicInfo, JxlBitDepth,
    JxlFrameHeader, SampleFormat, failure,
};
use crate::{Error, PixelData, PixelLayout, Result};
use jxl_decoder::api::{JxlNativeIntegerFrame, JxlNativeIntegerPlane};

pub(super) fn needed(info: &JxlBasicInfo, options: &DecodeOptions) -> Result<bool> {
    let wide = std::iter::once(&info.bit_depth)
        .chain(info.extra_channels.iter().map(|c| &c.bit_depth))
        .any(|d| matches!(d, JxlBitDepth::Int { bits_per_sample } if *bits_per_sample > 16));
    if options.format != SampleFormat::Native || !wide {
        return Ok(false);
    }
    if !options.color.is_noop() || options.alpha_mode != AlphaMode::AsIs {
        return Err(Error::InvalidInput("native wide integers require original color and alpha association; request F32 explicitly for rendering transforms".into()));
    }
    Ok(true)
}

fn bits(depth: &JxlBitDepth) -> Result<u32> {
    match depth {
        JxlBitDepth::Int { bits_per_sample } if (1..=31).contains(bits_per_sample) => {
            Ok(*bits_per_sample)
        }
        _ => Err(failure(
            "exact native integer output requires integer color and extra channels",
        )),
    }
}

fn sample_bytes(bits: u32) -> usize {
    if bits <= 8 {
        1
    } else if bits <= 16 {
        2
    } else {
        4
    }
}

pub(super) fn byte_count(
    info: &JxlBasicInfo,
    color: usize,
    size: (usize, usize),
    options: &DecodeOptions,
) -> Result<usize> {
    let pixels = size
        .0
        .checked_mul(size.1)
        .filter(|n| *n <= options.limits.max_pixels)
        .ok_or(Error::Limit("native integer pixel count"))?;
    let alpha = info
        .extra_channels
        .iter()
        .any(|c| c.ec_type == ExtraChannel::Alpha);
    let mut bytes_per_pixel = (color + usize::from(alpha)) * sample_bytes(bits(&info.bit_depth)?);
    if options.include_extra_channels {
        for ec in &info.extra_channels {
            bytes_per_pixel = bytes_per_pixel
                .checked_add(sample_bytes(bits(&ec.bit_depth)?))
                .ok_or(Error::Limit("native output channel count"))?;
        }
    }
    pixels
        .checked_mul(bytes_per_pixel)
        .filter(|n| *n <= options.limits.max_output_bytes)
        .ok_or(Error::Limit("native integer output bytes"))
}

fn pack(count: usize, depth: u32, mut value: impl FnMut(usize) -> u32) -> Result<PixelData> {
    macro_rules! output {
        ($ty:ty, $variant:ident) => {{
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| Error::Limit("native sample output allocation"))?;
            values.extend((0..count).map(|i| value(i) as $ty));
            PixelData::$variant(values)
        }};
    }
    Ok(if depth <= 8 {
        output!(u8, U8)
    } else if depth <= 16 {
        output!(u16, U16)
    } else {
        output!(u32, U32)
    })
}

fn plane(plane: JxlNativeIntegerPlane) -> Result<PixelData> {
    if plane.bits_per_sample > 16 {
        return Ok(PixelData::U32(plane.samples));
    }
    pack(plane.samples.len(), plane.bits_per_sample, |i| {
        plane.samples[i]
    })
}

pub(super) fn frame(
    captured: JxlNativeIntegerFrame,
    info: &JxlBasicInfo,
    header: &JxlFrameHeader,
    index: usize,
    options: &DecodeOptions,
) -> Result<DecodedFrame> {
    byte_count(
        info,
        captured.num_color_channels,
        (captured.width, captured.height),
        options,
    )?;
    let color = captured.num_color_channels;
    let count = captured
        .width
        .checked_mul(captured.height)
        .ok_or(Error::Limit("native dimensions"))?;
    if !matches!(color, 1 | 3)
        || captured.planes.len() != color + info.extra_channels.len()
        || captured.planes.iter().any(|p| p.samples.len() != count)
        || (captured.width, captured.height) != header.size
    {
        return Err(failure(
            "native capture geometry/plane count differs from frame header",
        ));
    }
    let alpha = info
        .extra_channels
        .iter()
        .position(|c| c.ec_type == ExtraChannel::Alpha);
    let layout = match (color, alpha.is_some()) {
        (1, false) => PixelLayout::Gray,
        (1, true) => PixelLayout::GrayAlpha,
        (3, false) => PixelLayout::Rgb,
        _ => PixelLayout::Rgba,
    };
    let depth = bits(&info.bit_depth)?;
    let max = (1u64 << depth) - 1;
    let channels = layout.channels();
    let samples = count
        .checked_mul(channels)
        .ok_or(Error::Limit("native interleaved samples"))?;
    let pixels = pack(samples, depth, |i| {
        let channel = i % channels;
        let pixel = i / channels;
        if channel < color {
            return captured.planes[channel].samples[pixel];
        }
        let alpha = &captured.planes[color + alpha.unwrap()];
        let alpha_max = (1u64 << alpha.bits_per_sample) - 1;
        // Interleaved alpha shares the color range; its separate plane remains exact.
        ((u64::from(alpha.samples[pixel]) * max + alpha_max / 2) / alpha_max) as u32
    })?;
    let extra_planes = if options.include_extra_channels {
        captured
            .planes
            .into_iter()
            .skip(color)
            .map(plane)
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    Ok(DecodedFrame {
        index,
        width: captured.width,
        height: captured.height,
        layout,
        pixels,
        extra_planes,
        duration_ticks: header.duration_ticks,
        name: header.name.clone(),
        header: header.clone(),
        complete: true,
    })
}
