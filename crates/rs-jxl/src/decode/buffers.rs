use super::{
    DecodeOptions, DecodedFrame, ExtraChannel, JxlBasicInfo, JxlBitDepth, SampleFormat, failure,
};
use crate::{Error, PixelData, PixelLayout, Result};
use jxl_decoder::api::{Endianness, JxlColorType, JxlDataFormat, JxlPixelFormat, VisibleFrameInfo};
use jxl_decoder::image::OwnedRawImage;

pub(super) struct Buffers {
    pub images: Vec<OwnedRawImage>,
    pub byte_count: usize,
    formats: Vec<JxlDataFormat>,
    width: usize,
    height: usize,
    layout: PixelLayout,
}
fn format(depth: &JxlBitDepth, requested: SampleFormat) -> JxlDataFormat {
    match (requested, depth) {
        (SampleFormat::U8, _) => JxlDataFormat::U8 { bit_depth: 8 },
        (SampleFormat::U16, _) => JxlDataFormat::U16 {
            bit_depth: 16,
            endianness: Endianness::LittleEndian,
        },
        (
            SampleFormat::Native,
            JxlBitDepth::Int {
                bits_per_sample: bits,
            },
        ) if *bits <= 8 => JxlDataFormat::U8 {
            bit_depth: *bits as u8,
        },
        (
            SampleFormat::Native,
            JxlBitDepth::Int {
                bits_per_sample: bits,
            },
        ) if *bits <= 16 => JxlDataFormat::U16 {
            bit_depth: *bits as u8,
            endianness: Endianness::LittleEndian,
        },
        _ => JxlDataFormat::F32 {
            endianness: Endianness::LittleEndian,
        },
    }
}
impl Buffers {
    pub(super) fn new(
        info: &JxlBasicInfo,
        color: JxlColorType,
        options: &DecodeOptions,
    ) -> Result<(Self, JxlPixelFormat)> {
        let (width, height) = info.size;
        let count = width
            .checked_mul(height)
            .ok_or(Error::Limit("pixel count overflow"))?;
        if count > options.limits.max_pixels {
            return Err(Error::Limit("decoded pixel count"));
        }
        let alpha = info
            .extra_channels
            .iter()
            .any(|ec| ec.ec_type == ExtraChannel::Alpha);
        let (layout, color_type) = match (color.is_grayscale(), alpha) {
            (true, false) => (PixelLayout::Gray, JxlColorType::Grayscale),
            (true, true) => (PixelLayout::GrayAlpha, JxlColorType::GrayscaleAlpha),
            (false, false) => (PixelLayout::Rgb, JxlColorType::Rgb),
            (false, true) => (PixelLayout::Rgba, JxlColorType::Rgba),
        };
        let main = format(&info.bit_depth, options.format);
        let extras: Vec<_> = info
            .extra_channels
            .iter()
            .map(|ec| {
                options
                    .include_extra_channels
                    .then(|| format(&ec.bit_depth, options.format))
            })
            .collect();
        let formats: Vec<_> = std::iter::once(main)
            .chain(extras.iter().flatten().copied())
            .collect();
        let mut byte_count = 0usize;
        let mut sizes = Vec::new();
        for (i, fmt) in formats.iter().enumerate() {
            let channels = if i == 0 { layout.channels() } else { 1 };
            let row = width
                .checked_mul(channels)
                .and_then(|v| v.checked_mul(fmt.bytes_per_sample()))
                .ok_or(Error::Limit("output row overflow"))?;
            byte_count = row
                .checked_mul(height)
                .and_then(|n| n.checked_add(byte_count))
                .ok_or(Error::Limit("output frame overflow"))?;
            sizes.push((row, height));
        }
        if byte_count > options.limits.max_output_bytes {
            return Err(Error::Limit("decoded frame allocation"));
        }
        let images = sizes
            .into_iter()
            .map(|size| OwnedRawImage::new(size).map_err(failure))
            .collect::<Result<_>>()?;
        Ok((
            Self {
                images,
                byte_count,
                formats,
                width,
                height,
                layout,
            },
            JxlPixelFormat {
                color_type,
                color_data_format: Some(main),
                extra_channel_format: extras,
            },
        ))
    }
    pub(super) fn copy_frame(
        &self,
        index: usize,
        header: &VisibleFrameInfo,
    ) -> Result<DecodedFrame> {
        let mut planes = self
            .images
            .iter()
            .zip(&self.formats)
            .map(|(image, format)| copy_samples(image, *format));
        let pixels = planes.next().ok_or_else(|| failure("no color output"))??;
        Ok(DecodedFrame {
            index,
            width: self.width,
            height: self.height,
            layout: self.layout,
            pixels,
            extra_planes: planes.collect::<Result<_>>()?,
            name: header.name.clone(),
            duration_ticks: header.duration_ticks,
        })
    }
}
fn copy_samples(image: &OwnedRawImage, format: JxlDataFormat) -> Result<PixelData> {
    let (row_bytes, height) = image.byte_size();
    let count = row_bytes
        .checked_mul(height)
        .ok_or(Error::Limit("sample count overflow"))?
        / format.bytes_per_sample();
    macro_rules! samples {
        ($size:literal, $convert:expr, $variant:ident) => {{
            let mut output = Vec::new();
            output
                .try_reserve_exact(count)
                .map_err(|_| Error::Limit("decoded sample allocation"))?;
            for y in 0..height {
                output.extend(
                    image
                        .row(y)
                        .chunks_exact($size)
                        .map(|sample| $convert(sample.try_into().unwrap())),
                );
            }
            PixelData::$variant(output)
        }};
    }
    Ok(match format {
        JxlDataFormat::U8 { .. } => samples!(1, |v: [u8; 1]| v[0], U8),
        JxlDataFormat::U16 { .. } => samples!(2, u16::from_le_bytes, U16),
        JxlDataFormat::F16 { .. } => samples!(2, u16::from_le_bytes, F16),
        JxlDataFormat::F32 { .. } => samples!(4, f32::from_le_bytes, F32),
    })
}
