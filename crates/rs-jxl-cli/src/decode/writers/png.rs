//! PNG/APNG serialization with exact timing and output-profile tagging.
use super::*;
use crate::decode::metadata::{orientation_exif, png_exif};
use std::borrow::Cow;

pub(in crate::decode) fn check_png_planes(image: &Raster<'_>, extras_separate: bool) -> Result<()> {
    let first = image
        .frames
        .first()
        .ok_or("decoder returned no displayed frames")?;
    let primary_alpha = image
        .info
        .extra_channels
        .iter()
        .position(|ec| ec.ec_type == ExtraChannel::Alpha);
    for frame in image.frames {
        if frame.extra_planes.len() != image.info.extra_channels.len() {
            return Err("decoded extra planes do not match channel metadata".into());
        }
        if !extras_separate
            && frame
                .extra_planes
                .iter()
                .enumerate()
                .any(|(index, _)| !frame.layout.has_alpha() || Some(index) != primary_alpha)
        {
            return Err(
                "this output cannot preserve non-primary extra planes; use NPY or the decoder API"
                    .into(),
            );
        }
        if let Some(alpha) = primary_alpha
            .and_then(|index| frame.extra_planes.get(index))
            .filter(|_| !extras_separate)
        {
            let stride = frame.layout.channels();
            let matches = match (&frame.pixels, alpha) {
                (PixelData::U8(pixels), PixelData::U8(alpha)) => pixels
                    .chunks_exact(stride)
                    .map(|pixel| pixel[stride - 1])
                    .eq(alpha.iter().copied()),
                (PixelData::U16(pixels), PixelData::U16(alpha)) => pixels
                    .chunks_exact(stride)
                    .map(|pixel| pixel[stride - 1])
                    .eq(alpha.iter().copied()),
                _ => false,
            };
            if !matches {
                return Err("the separate primary alpha differs from interleaved alpha; PNG cannot preserve both, use NPY".into());
            }
        }
    }
    if first.layout.has_alpha() && image.alpha_associated() {
        return Err("PNG requires straight alpha but the decoder returned associated alpha; use NPY to preserve samples".into());
    }
    Ok(())
}

pub(in crate::decode) fn png_depth(pixels: &PixelData) -> Result<png::BitDepth> {
    match pixels {
        PixelData::U8(_) => Ok(png::BitDepth::Eight),
        PixelData::U16(_) => Ok(png::BitDepth::Sixteen),
        _ => Err("PNG supports only U8/U16 samples; explicitly choose --bits 8 or 16".into()),
    }
}

pub(in crate::decode) fn png_delay(
    ticks: u32,
    tps_numerator: u32,
    tps_denominator: u32,
) -> Result<(u16, u16)> {
    if tps_numerator == 0 || tps_denominator == 0 {
        return Err("invalid animation timebase".into());
    }
    let numerator = u64::from(ticks) * u64::from(tps_denominator);
    let denominator = u64::from(tps_numerator);
    let (mut a, mut b) = (numerator, denominator);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let numerator = u16::try_from(numerator / a);
    let denominator = u16::try_from(denominator / a);
    match (numerator, denominator) {
        (Ok(n), Ok(d)) => Ok((n, d)),
        _ => Err("frame timing cannot be represented exactly in APNG; select --frame or use NPY (without timing)".into()),
    }
}

pub(in crate::decode) fn write_png(
    output: &mut impl Write,
    image: &Raster<'_>,
    selected: bool,
    keep_orientation: bool,
    extras_separate: bool,
) -> Result<()> {
    let first = validate_frames(image.frames)?;
    check_png_planes(image, extras_separate)?;
    let mut info = png::Info::with_size(u32::try_from(first.width)?, u32::try_from(first.height)?);
    info.bit_depth = png_depth(&first.pixels)?;
    info.color_type = match first.layout {
        PixelLayout::Gray => png::ColorType::Grayscale,
        PixelLayout::GrayAlpha => png::ColorType::GrayscaleAlpha,
        PixelLayout::Rgb => png::ColorType::Rgb,
        PixelLayout::Rgba => png::ColorType::Rgba,
    };
    // An embedded profile can describe a different color space than output pixels.
    let profile = image
        .output_profile
        .as_ref()
        .ok_or("PNG output requires the decoder's output color profile; refusing to assume sRGB")?;
    let icc = profile
        .try_as_icc()
        .ok_or("output color profile cannot be represented as ICC; refusing untagged PNG output")?;
    let color_space = rs_jxl::icc::validate_profile(icc.as_slice())?;
    let expected = if first.layout.channels() < 3 {
        rs_jxl::icc::IccColorSpace::Gray
    } else {
        rs_jxl::icc::IccColorSpace::Rgb
    };
    if color_space != expected {
        return Err("output ICC color space does not match the PNG pixel layout".into());
    }
    info.icc_profile = Some(Cow::Borrowed(icc.as_slice()));
    let orientation = if keep_orientation {
        image.info.orientation as u16
    } else {
        1
    };
    let mut exif = None;
    for metadata in image.boxes {
        match &metadata.kind {
            b"Exif" => {
                if exif.is_some() {
                    return Err("multiple EXIF boxes cannot be preserved in one PNG".into());
                }
                exif = Some(png_exif(&metadata.data, orientation)?);
            }
            b"xml " => {
                let text = std::str::from_utf8(&metadata.data)
                    .map_err(|_| "XMP is not UTF-8 and cannot be preserved as PNG iTXt")?;
                info.utf8_text.push(png::text_metadata::ITXtChunk::new(
                    "XML:com.adobe.xmp",
                    text,
                ));
            }
            _ => (),
        }
    }
    if exif.is_none() && orientation != 1 {
        exif = Some(orientation_exif(orientation));
    }
    info.exif_metadata = exif.map(Cow::Owned);
    let animation = if selected {
        None
    } else {
        image.info.animation.as_ref()
    };
    if image.frames.len() > 1 && animation.is_none() {
        return Err("multiple frames without an animation timebase cannot be written as APNG; use NPY or --frame".into());
    }
    let delays = animation
        .map(|animation| {
            image
                .frames
                .iter()
                .map(|frame| {
                    png_delay(
                        frame.duration_ticks,
                        animation.tps_numerator,
                        animation.tps_denominator,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut encoder = png::Encoder::with_info(output, info)?;
    encoder.validate_sequence(true);
    if let Some(animation) = animation {
        encoder.set_animated(u32::try_from(image.frames.len())?, animation.num_loops)?;
        encoder.set_blend_op(png::BlendOp::Source)?;
        encoder.set_dispose_op(png::DisposeOp::None)?;
    }
    let mut writer = encoder.write_header()?;
    for (index, frame) in image.frames.iter().enumerate() {
        if let Some(delays) = &delays {
            writer.set_frame_delay(delays[index].0, delays[index].1)?;
        }
        match &frame.pixels {
            PixelData::U8(values) => writer.write_image_data(values)?,
            PixelData::U16(values) => {
                let length = values
                    .len()
                    .checked_mul(2)
                    .ok_or("PNG frame size overflow")?;
                let mut bytes = Vec::new();
                bytes.try_reserve_exact(length)?;
                for value in values {
                    bytes.extend_from_slice(&value.to_be_bytes());
                }
                writer.write_image_data(&bytes)?;
            }
            _ => return Err("PNG requires integer samples".into()),
        }
    }
    writer.finish()?;
    Ok(())
}

pub(in crate::decode) fn write_plane_png(
    output: &mut impl Write,
    width: usize,
    height: usize,
    plane: &PixelData,
    depth: u8,
) -> Result<()> {
    if !(1..=16).contains(&depth) {
        return Err("PNG extra planes require integer depth <=16".into());
    }
    let storage = if depth <= 8 { 8 } else { 16 };
    let mut encoder = png::Encoder::new(output, u32::try_from(width)?, u32::try_from(height)?);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(if storage == 8 {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    });
    let count = width.checked_mul(height).ok_or("plane size overflow")?;
    if plane.len() != count {
        return Err("PNG extra plane size mismatch".into());
    }
    let max = (1u32 << depth) - 1;
    let target = (1u32 << storage) - 1;
    let mut bytes = Output::default();
    for index in 0..count {
        let value = match plane {
            PixelData::U8(values) => u32::from(values[index]),
            PixelData::U16(values) => u32::from(values[index]),
            _ => return Err("PNG extra planes require U8/U16".into()),
        };
        if value > max {
            return Err("PNG extra sample exceeds declared range".into());
        }
        let value =
            ((u64::from(value) * u64::from(target) + u64::from(max / 2)) / u64::from(max)) as u16;
        if storage == 8 {
            bytes.write_all(&[value as u8])?;
        } else {
            bytes.write_all(&value.to_be_bytes())?;
        }
    }
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&bytes.into_bytes())?;
    writer.finish()?;
    Ok(())
}
