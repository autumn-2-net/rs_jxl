use crate::{Result, message, pixels};
use rs_jxl::{
    PixelData, PixelLayout,
    decode::{DecodeOptions, DecodedImage, ExtraChannel, SampleFormat},
};
use std::{
    borrow::Cow,
    io::{self, Write},
};

struct Output {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for Output {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|n| *n <= self.limit && *n <= isize::MAX as usize)
            .ok_or_else(|| io::Error::other("PNG output exceeds max_output_bytes"))?;
        self.bytes
            .try_reserve(end - self.bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn encode(image: &DecodedImage, options: &DecodeOptions) -> Result<Vec<u8>> {
    if image.frames.len() != 1 {
        return Err("PNG export requires exactly one selected frame".into());
    }
    let frame = &image.frames[0];
    if !frame.complete {
        return Err("PNG export requires a complete selected frame".into());
    }
    let (depth, storage_bits) = match &frame.pixels {
        PixelData::U8(_) => (::png::BitDepth::Eight, 8),
        PixelData::U16(_) => (::png::BitDepth::Sixteen, 16),
        _ => {
            return Err("PNG cannot preserve native floating-point or wide integer samples".into());
        }
    };
    if options.format == SampleFormat::Native
        && super::metadata::depth(&image.info.bit_depth) != (storage_bits, 0)
    {
        return Err("PNG native export requires an exact 8- or 16-bit integer depth".into());
    }
    if image.info.extra_channels.len() > usize::from(frame.layout.has_alpha())
        || image
            .info
            .extra_channels
            .iter()
            .any(|e| e.ec_type != ExtraChannel::Alpha)
    {
        return Err("PNG cannot preserve non-primary extra channels".into());
    }
    if let Some(alpha) = image.info.extra_channels.first() {
        if alpha.alpha_associated {
            return Err("PNG cannot preserve premultiplied alpha samples".into());
        }
        if options.format == SampleFormat::Native
            && super::metadata::depth(&alpha.bit_depth) != (storage_bits, 0)
        {
            return Err(
                "PNG cannot preserve an alpha channel with a different native depth".into(),
            );
        }
        if let Some(plane) = frame.extra_planes.first()
            && *plane != pixels::alpha_plane(&frame.pixels, frame.layout.channels())
        {
            return Err(
                "PNG cannot preserve separate alpha differing from interleaved alpha".into(),
            );
        }
    }
    let width = u32::try_from(frame.width).map_err(message)?;
    let height = u32::try_from(frame.height).map_err(message)?;
    let mut info = ::png::Info::with_size(width, height);
    info.bit_depth = depth;
    info.color_type = match frame.layout {
        PixelLayout::Gray => ::png::ColorType::Grayscale,
        PixelLayout::GrayAlpha => ::png::ColorType::GrayscaleAlpha,
        PixelLayout::Rgb => ::png::ColorType::Rgb,
        PixelLayout::Rgba => ::png::ColorType::Rgba,
    };
    let profile = image
        .output_profile
        .as_ref()
        .ok_or("PNG output color profile is unavailable")?;
    let icc = profile
        .try_as_icc()
        .ok_or("PNG output color profile cannot be represented as ICC")?;
    let space = rs_jxl::icc::validate_profile(icc.as_slice()).map_err(message)?;
    let expected = if frame.layout.channels() < 3 {
        rs_jxl::icc::IccColorSpace::Gray
    } else {
        rs_jxl::icc::IccColorSpace::Rgb
    };
    if space != expected {
        return Err("PNG output ICC color space disagrees with pixels".into());
    }
    info.icc_profile = Some(Cow::Borrowed(icc.as_slice()));
    // Ancillary metadata cannot be silently discarded or copied with a stale
    // EXIF orientation. The packed decode API preserves all boxes in JSON.
    if !image.boxes.is_empty() {
        return Err("PNG export cannot preserve ancillary JXL boxes; use decode to retrieve metadata separately".into());
    }
    if !options.adjust_orientation && image.info.orientation as u16 != 1 {
        let orientation = image.info.orientation as u16;
        let mut exif = vec![
            b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 18, 1, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        ];
        exif[18..20].copy_from_slice(&orientation.to_le_bytes());
        info.exif_metadata = Some(Cow::Owned(exif));
    }
    let samples = match &frame.pixels {
        PixelData::U8(values) => Cow::Borrowed(values.as_slice()),
        PixelData::U16(values) => {
            let length = values
                .len()
                .checked_mul(2)
                .ok_or("PNG sample byte count overflow")?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(length).map_err(message)?;
            // PNG stores 16-bit samples big-endian, unlike the ABI sample protocol.
            for value in values {
                bytes.extend_from_slice(&value.to_be_bytes());
            }
            Cow::Owned(bytes)
        }
        _ => return Err("PNG requires U8/U16".into()),
    };
    let mut output = Output {
        bytes: Vec::new(),
        limit: options.limits.max_output_bytes,
    };
    {
        let encoder = ::png::Encoder::with_info(&mut output, info).map_err(message)?;
        let mut writer = encoder.write_header().map_err(message)?;
        writer.write_image_data(samples.as_ref()).map_err(message)?;
        writer.finish().map_err(message)?;
    }
    Ok(output.bytes)
}
