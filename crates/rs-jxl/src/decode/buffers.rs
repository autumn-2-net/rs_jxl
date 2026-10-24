use super::{
    DecodeOptions, DecodedFrame, ExtraChannel, JxlBasicInfo, JxlBitDepth, JxlFrameHeader,
    SampleFormat, failure,
};
use crate::{Error, PixelData, PixelLayout, Result};
use jxl_decoder::api::{Endianness, JxlColorType, JxlDataFormat, JxlPixelFormat};
use jxl_decoder::image::OwnedRawImage;

pub(super) struct Buffers {
    pub images: Vec<OwnedRawImage>,
    pub byte_count: usize,
    formats: Vec<JxlDataFormat>,
    width: usize,
    height: usize,
    layout: PixelLayout,
}
pub(super) fn format(depth: &JxlBitDepth, requested: SampleFormat) -> JxlDataFormat {
    match (requested, depth) {
        (SampleFormat::U8, _) => JxlDataFormat::U8 { bit_depth: 8 },
        (SampleFormat::U8Bits(bit_depth), _) => JxlDataFormat::U8 { bit_depth },
        (SampleFormat::U16Bits(bit_depth), _) => JxlDataFormat::U16 {
            bit_depth,
            endianness: Endianness::LittleEndian,
        },
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
    pub(super) fn has_size(&self, size: (usize, usize)) -> bool {
        (self.width, self.height) == size
    }
    pub(super) fn pixel_format(
        info: &JxlBasicInfo,
        color: JxlColorType,
        options: &DecodeOptions,
    ) -> JxlPixelFormat {
        let alpha = info
            .extra_channels
            .iter()
            .any(|ec| ec.ec_type == ExtraChannel::Alpha);
        let color_type = match (color.is_grayscale(), alpha) {
            (true, false) => JxlColorType::Grayscale,
            (true, true) => JxlColorType::GrayscaleAlpha,
            (false, false) => JxlColorType::Rgb,
            (false, true) => JxlColorType::Rgba,
        };
        let processing = options.color.needs_float();
        let requested_main = if processing {
            SampleFormat::F32
        } else {
            options.format
        };
        let main = format(&info.bit_depth, requested_main);
        // CMYK conversion needs K even if the caller does not retain extras.
        // Keep complete header order internally; finish() discards these on request.
        let include_extras = options.include_extra_channels
            || (processing
                && info
                    .extra_channels
                    .iter()
                    .any(|ec| ec.ec_type == ExtraChannel::Black));
        let extras: Vec<_> = info
            .extra_channels
            .iter()
            .map(|ec| {
                include_extras.then(|| {
                    // Alpha must use the same final quantizer as interleaved alpha,
                    // not backend dithering for one copy and rounding for the other.
                    let requested = if processing
                        && matches!(ec.ec_type, ExtraChannel::Alpha | ExtraChannel::Black)
                    {
                        SampleFormat::F32
                    } else {
                        options.format
                    };
                    format(&ec.bit_depth, requested)
                })
            })
            .collect();
        JxlPixelFormat {
            color_type,
            color_data_format: Some(main),
            extra_channel_format: extras,
        }
    }
    pub(super) fn new(
        size: (usize, usize),
        pixel_format: &JxlPixelFormat,
        options: &DecodeOptions,
    ) -> Result<Self> {
        let (width, height) = size;
        let count = width
            .checked_mul(height)
            .ok_or(Error::Limit("pixel count overflow"))?;
        if count > options.limits.max_pixels {
            return Err(Error::Limit("decoded pixel count"));
        }
        let layout = match pixel_format.color_type {
            JxlColorType::Grayscale => PixelLayout::Gray,
            JxlColorType::GrayscaleAlpha => PixelLayout::GrayAlpha,
            JxlColorType::Rgb => PixelLayout::Rgb,
            JxlColorType::Rgba => PixelLayout::Rgba,
            _ => return Err(failure("unsupported decoder color layout")),
        };
        let main = pixel_format
            .color_data_format
            .ok_or_else(|| failure("missing color format"))?;
        let extras = &pixel_format.extra_channel_format;
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
        Ok(Self {
            images,
            byte_count,
            formats,
            width,
            height,
            layout,
        })
    }
    pub(super) fn copy_frame(
        &self,
        index: usize,
        header: &JxlFrameHeader,
        complete: bool,
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
            header: header.clone(),
            complete,
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
