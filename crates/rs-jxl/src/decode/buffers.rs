use super::{
    DecodeOptions, DecodedFrame, ExtraChannel, JxlBasicInfo, JxlBitDepth, JxlFrameHeader,
    SampleFormat, failure,
};
use crate::{Error, PixelData, PixelLayout, Result};
use jxl_decoder::api::{Endianness, JxlColorType, JxlDataFormat, JxlOutputBuffer, JxlPixelFormat};

pub(super) struct Buffers {
    planes: Vec<PixelData>,
    pub byte_count: usize,
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
            endianness: Endianness::native(),
        },
        (SampleFormat::U16, _) => JxlDataFormat::U16 {
            bit_depth: 16,
            endianness: Endianness::native(),
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
            endianness: Endianness::native(),
        },
        _ => JxlDataFormat::F32 {
            endianness: Endianness::native(),
        },
    }
}
impl Buffers {
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
        if width == 0 || height == 0 {
            return Err(failure("empty decoder output"));
        }
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
        let formats = std::iter::once(main).chain(extras.iter().flatten().copied());
        let mut byte_count = 0usize;
        for (i, fmt) in formats.clone().enumerate() {
            match fmt {
                JxlDataFormat::U16 { endianness, .. }
                | JxlDataFormat::F16 { endianness }
                | JxlDataFormat::F32 { endianness }
                    if endianness != Endianness::native() =>
                {
                    return Err(failure("non-native decoder output endianness"));
                }
                _ => {}
            }
            let channels = if i == 0 { layout.channels() } else { 1 };
            let row = width
                .checked_mul(channels)
                .and_then(|v| v.checked_mul(fmt.bytes_per_sample()))
                .ok_or(Error::Limit("output row overflow"))?;
            byte_count = row
                .checked_mul(height)
                .and_then(|n| n.checked_add(byte_count))
                .ok_or(Error::Limit("output frame overflow"))?;
        }
        if byte_count > options.limits.max_output_bytes {
            return Err(Error::Limit("decoded frame allocation"));
        }
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(1 + extras.iter().flatten().count())
            .map_err(|_| Error::Limit("decoded plane list"))?;
        for (i, fmt) in formats.enumerate() {
            let channels = if i == 0 { layout.channels() } else { 1 };
            let samples = count
                .checked_mul(channels)
                .ok_or(Error::Limit("sample count overflow"))?;
            let plane = match fmt {
                JxlDataFormat::U8 { .. } => bytemuck::try_zeroed_vec(samples).map(PixelData::U8),
                JxlDataFormat::U16 { .. } => bytemuck::try_zeroed_vec(samples).map(PixelData::U16),
                JxlDataFormat::F16 { .. } => bytemuck::try_zeroed_vec(samples).map(PixelData::F16),
                JxlDataFormat::F32 { .. } => bytemuck::try_zeroed_vec(samples).map(PixelData::F32),
            }
            .map_err(|_| Error::Limit("decoded sample allocation"))?;
            planes.push(plane);
        }
        Ok(Self {
            planes,
            byte_count,
            width,
            height,
            layout,
        })
    }
    pub(super) fn views(&mut self) -> Result<Vec<JxlOutputBuffer<'_>>> {
        let mut views = Vec::new();
        views
            .try_reserve_exact(self.planes.len())
            .map_err(|_| Error::Limit("decoder output views"))?;
        for plane in &mut self.planes {
            // Typed allocation and packed rows preserve sample alignment for every row.
            let bytes = sample_bytes_mut(plane);
            let row_bytes = bytes.len() / self.height;
            views.push(JxlOutputBuffer::new(bytes, self.height, row_bytes));
        }
        Ok(views)
    }
    pub(super) fn into_frame(
        mut self,
        index: usize,
        header: &JxlFrameHeader,
        complete: bool,
    ) -> Result<DecodedFrame> {
        if self.planes.is_empty() {
            return Err(failure("no color output"));
        }
        let pixels = self.planes.remove(0);
        Ok(DecodedFrame {
            index,
            width: self.width,
            height: self.height,
            layout: self.layout,
            pixels,
            extra_planes: self.planes,
            name: header.name.clone(),
            duration_ticks: header.duration_ticks,
            header: header.clone(),
            complete,
        })
    }
    pub(super) fn copy_frame(
        &self,
        index: usize,
        header: &JxlFrameHeader,
        complete: bool,
    ) -> Result<DecodedFrame> {
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(self.planes.len())
            .map_err(|_| Error::Limit("decoded plane list"))?;
        for plane in &self.planes {
            planes.push(copy_samples(plane)?);
        }
        Self {
            planes,
            byte_count: self.byte_count,
            width: self.width,
            height: self.height,
            layout: self.layout,
        }
        .into_frame(index, header, complete)
    }
}

fn sample_bytes_mut(plane: &mut PixelData) -> &mut [u8] {
    match plane {
        PixelData::U8(samples) => samples,
        PixelData::U16(samples) | PixelData::F16(samples) => bytemuck::cast_slice_mut(samples),
        PixelData::U32(samples) => bytemuck::cast_slice_mut(samples),
        PixelData::F32(samples) => bytemuck::cast_slice_mut(samples),
    }
}

fn copy_samples(plane: &PixelData) -> Result<PixelData> {
    macro_rules! copy {
        ($samples:expr, $variant:ident) => {{
            let mut output = Vec::new();
            output
                .try_reserve_exact($samples.len())
                .map_err(|_| Error::Limit("decoded sample allocation"))?;
            output.extend_from_slice($samples);
            PixelData::$variant(output)
        }};
    }
    Ok(match plane {
        PixelData::U8(samples) => copy!(samples, U8),
        PixelData::U16(samples) => copy!(samples, U16),
        PixelData::U32(samples) => copy!(samples, U32),
        PixelData::F16(samples) => copy!(samples, F16),
        PixelData::F32(samples) => copy!(samples, F32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_decoder::api::{JxlBlendInfo, JxlBlendMode, JxlLayerInfo};

    fn formats() -> [JxlDataFormat; 4] {
        [
            JxlDataFormat::U8 { bit_depth: 8 },
            JxlDataFormat::U16 {
                bit_depth: 16,
                endianness: Endianness::native(),
            },
            JxlDataFormat::F16 {
                endianness: Endianness::native(),
            },
            JxlDataFormat::F32 {
                endianness: Endianness::native(),
            },
        ]
    }

    fn pixel_format(main: JxlDataFormat) -> JxlPixelFormat {
        let [u8, u16, f16, f32] = formats();
        JxlPixelFormat {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(main),
            extra_channel_format: vec![None, Some(u8), Some(u16), None, Some(f16), Some(f32)],
        }
    }

    fn header(size: (usize, usize)) -> JxlFrameHeader {
        JxlFrameHeader {
            name: "typed output".into(),
            duration: Some(125.0),
            size,
            duration_ticks: 7,
            timecode: Some(19),
            is_last: false,
            is_preview: true,
            layer_info: JxlLayerInfo {
                have_crop: true,
                crop_origin: (-3, 2),
                size,
                blend_info: JxlBlendInfo {
                    mode: JxlBlendMode::Replace,
                    source: 0,
                    alpha_channel: 0,
                    clamp: false,
                },
                extra_channel_blend_info: Vec::new(),
                save_as_reference: Some(1),
                save_before_color_transform: true,
            },
        }
    }

    fn bytes(plane: &PixelData) -> &[u8] {
        match plane {
            PixelData::U8(samples) => samples,
            PixelData::U16(samples) | PixelData::F16(samples) => bytemuck::cast_slice(samples),
            PixelData::U32(samples) => bytemuck::cast_slice(samples),
            PixelData::F32(samples) => bytemuck::cast_slice(samples),
        }
    }

    fn pattern(plane: &mut PixelData, offset: usize) -> Vec<u8> {
        let size = usize::from(plane.bits()) / 8;
        // Both float widths include signed zero, subnormals, infinities and NaN payloads.
        let patterns: &[u32] = if size == 4 {
            &[
                0,
                0x8000_0000,
                1,
                0x007f_ffff,
                0x0080_0000,
                0x3f80_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7f80_0001,
                0x7fc0_0123,
                0xff80_0123,
                0xffff_ffff,
            ]
        } else {
            &[
                0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0x7c00, 0xfc00, 0x7c01, 0x7e55, 0xfc01,
                0xffff, 0x007f, 0x0080, 0x00ff, 0x1234,
            ]
        };
        let output = sample_bytes_mut(plane);
        for (i, sample) in output.chunks_exact_mut(size).enumerate() {
            let word = patterns[(i + offset) % patterns.len()];
            match size {
                1 => sample[0] = word as u8,
                2 => sample.copy_from_slice(&(word as u16).to_ne_bytes()),
                4 => sample.copy_from_slice(&word.to_ne_bytes()),
                _ => unreachable!(),
            }
        }
        output.to_vec()
    }

    fn assert_planes(frame: &DecodedFrame, expected: &[Vec<u8>], main: JxlDataFormat) {
        let formats = std::iter::once(main).chain(formats());
        assert_eq!(frame.extra_planes.len(), 4);
        for ((plane, expected), format) in std::iter::once(&frame.pixels)
            .chain(&frame.extra_planes)
            .zip(expected)
            .zip(formats)
        {
            assert!(matches!(
                (plane, format),
                (PixelData::U8(_), JxlDataFormat::U8 { .. })
                    | (PixelData::U16(_), JxlDataFormat::U16 { .. })
                    | (PixelData::F16(_), JxlDataFormat::F16 { .. })
                    | (PixelData::F32(_), JxlDataFormat::F32 { .. })
            ));
            assert_eq!(bytes(plane), expected);
            if let PixelData::F32(samples) = plane {
                let bits: Vec<_> = expected
                    .chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect();
                assert_eq!(
                    samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    bits
                );
            }
        }
    }

    #[test]
    fn typed_frames_move_allocations_and_snapshots_keep_all_bits() {
        for width in [1, 3, 17, 257, 513] {
            for main in formats() {
                let format = pixel_format(main);
                let size = (width, 5);
                let header = header(size);
                let mut buffers = Buffers::new(size, &format, &DecodeOptions::default()).unwrap();
                assert!(
                    buffers
                        .planes
                        .iter()
                        .all(|p| bytes(p).iter().all(|&b| b == 0))
                );
                let expected: Vec<_> = buffers
                    .planes
                    .iter_mut()
                    .enumerate()
                    .map(|(i, plane)| pattern(plane, i))
                    .collect();
                let pointers: Vec<_> = buffers.planes.iter().map(|p| bytes(p).as_ptr()).collect();
                let formats: Vec<_> = std::iter::once(main).chain(formats()).collect();
                {
                    let views = buffers.views().unwrap();
                    assert_eq!(views.len(), 5);
                    for (i, (view, format)) in views.iter().zip(&formats).enumerate() {
                        let channels = if i == 0 { 3 } else { 1 };
                        let row = width * channels * format.bytes_per_sample();
                        assert_eq!(view.byte_size(), (row, size.1));
                        for y in 0..size.1 {
                            assert_eq!(
                                (pointers[i] as usize + y * row) % format.bytes_per_sample(),
                                0
                            );
                        }
                    }
                }
                let snapshot = buffers.copy_frame(9, &header, false).unwrap();
                assert!(!snapshot.complete);
                for (i, plane) in std::iter::once(&snapshot.pixels)
                    .chain(&snapshot.extra_planes)
                    .enumerate()
                {
                    assert_ne!(bytes(plane).as_ptr(), pointers[i]);
                }
                for plane in &mut buffers.planes {
                    sample_bytes_mut(plane).fill(0xa5);
                }
                assert_planes(&snapshot, &expected, main);
                for (plane, expected) in buffers.planes.iter_mut().zip(&expected) {
                    sample_bytes_mut(plane).copy_from_slice(expected);
                }
                // FrameComplete may have full_quality=false; ownership still transfers.
                let frame = buffers.into_frame(9, &header, false).unwrap();
                assert_planes(&frame, &expected, main);
                assert_eq!((frame.width, frame.height), size);
                assert_eq!(frame.layout, PixelLayout::Rgb);
                assert_eq!(frame.index, 9);
                assert!(!frame.complete);
                assert_eq!(frame.name, header.name);
                assert_eq!(frame.duration_ticks, header.duration_ticks);
                assert_eq!(frame.header.duration, header.duration);
                assert_eq!(frame.header.timecode, header.timecode);
                assert_eq!(frame.header.is_last, header.is_last);
                assert_eq!(frame.header.is_preview, header.is_preview);
                assert_eq!(frame.header.size, header.size);
                assert_eq!(frame.header.layer_info, header.layer_info);
                for (i, plane) in std::iter::once(&frame.pixels)
                    .chain(&frame.extra_planes)
                    .enumerate()
                {
                    assert_eq!(bytes(plane).as_ptr(), pointers[i]);
                }
                let mut next = Buffers::new(size, &format, &DecodeOptions::default()).unwrap();
                for plane in &mut next.planes {
                    sample_bytes_mut(plane).fill(0x5a);
                }
                assert_planes(&frame, &expected, main);
                assert_planes(&snapshot, &expected, main);
            }
        }
    }

    #[test]
    fn typed_buffers_keep_limits_and_reject_non_native_storage() {
        let size = (3, 5);
        let mut format = pixel_format(formats()[1]);
        let mut options = DecodeOptions::default();
        options.limits.max_pixels = size.0 * size.1;
        options.limits.max_output_bytes = size.0 * size.1 * (3 * 2 + 1 + 2 + 2 + 4);
        let buffers = Buffers::new(size, &format, &options).unwrap();
        assert_eq!(buffers.byte_count, options.limits.max_output_bytes);
        options.limits.max_output_bytes -= 1;
        assert!(matches!(
            Buffers::new(size, &format, &options),
            Err(Error::Limit(_))
        ));
        options.limits.max_output_bytes += 1;
        options.limits.max_pixels -= 1;
        assert!(matches!(
            Buffers::new(size, &format, &options),
            Err(Error::Limit(_))
        ));
        for size in [(0, 1), (1, 0), (usize::MAX, 2)] {
            assert!(Buffers::new(size, &format, &DecodeOptions::default()).is_err());
        }
        let opposite = if Endianness::native() == Endianness::LittleEndian {
            Endianness::BigEndian
        } else {
            Endianness::LittleEndian
        };
        for main in [
            JxlDataFormat::U16 {
                bit_depth: 16,
                endianness: opposite,
            },
            JxlDataFormat::F16 {
                endianness: opposite,
            },
            JxlDataFormat::F32 {
                endianness: opposite,
            },
        ] {
            format.color_data_format = Some(main);
            assert!(Buffers::new(size, &format, &DecodeOptions::default()).is_err());
        }
    }

    #[test]
    fn backend_formats_use_native_endianness() {
        for depth in [
            JxlBitDepth::Int { bits_per_sample: 8 },
            JxlBitDepth::Int {
                bits_per_sample: 16,
            },
            JxlBitDepth::Float {
                bits_per_sample: 32,
                exponent_bits_per_sample: 8,
            },
        ] {
            for requested in [
                SampleFormat::Native,
                SampleFormat::U8,
                SampleFormat::U8Bits(7),
                SampleFormat::U16,
                SampleFormat::U16Bits(12),
                SampleFormat::F32,
            ] {
                match format(&depth, requested) {
                    JxlDataFormat::U8 { .. } => {}
                    JxlDataFormat::U16 { endianness, .. }
                    | JxlDataFormat::F16 { endianness }
                    | JxlDataFormat::F32 { endianness } => {
                        assert_eq!(endianness, Endianness::native());
                    }
                }
            }
        }
    }
}
