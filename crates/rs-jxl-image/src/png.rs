use std::io::Cursor;

use rs_jxl::color::{
    Chromaticity, ColorEncoding, ColorFields, ColorSpace, Primaries, RenderingIntent,
    TransferFunction, WhitePoint,
};
use rs_jxl::{Image, PixelData, PixelLayout};

use super::{checked_size, metadata, zeroed_bytes};
use crate::Result;

const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const SRGB_CHROMATICITIES: [u32; 8] = [31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000];

#[derive(Debug, Default)]
struct PngColor {
    srgb: Option<u8>,
    gamma: Option<u32>,
    chromaticities: Option<[u32; 8]>,
    icc: bool,
}

impl PngColor {
    fn validate(&self) -> Result<()> {
        if self.srgb.is_some() && self.icc {
            return Err("PNG sRGB and iCCP are mutually exclusive".into());
        }
        if self.srgb.is_some()
            && (self.gamma.is_some_and(|g| g != 45455)
                || self
                    .chromaticities
                    .is_some_and(|c| c != SRGB_CHROMATICITIES))
        {
            return Err("PNG sRGB conflicts with gAMA/cHRM".into());
        }
        Ok(())
    }

    fn encoding(&self, bytes: &[u8], space: ColorSpace) -> Result<ColorEncoding> {
        self.validate()?;
        if self.icc {
            // png ignores malformed iCCP internally. Require an actual profile,
            // never an sRGB fallback. Decode it once, outside the pixel reader.
            let reader = strict_decoder(bytes, false).read_info()?;
            let profile = reader
                .info()
                .icc_profile
                .as_ref()
                .ok_or("malformed PNG iCCP profile")?;
            let mut copy = Vec::new();
            copy.try_reserve_exact(profile.len())?;
            copy.extend_from_slice(profile);
            return Ok(ColorEncoding::icc(space, copy)?);
        }
        // Same unspecified-field defaults as libjxl extras/dec/apng.cc. Explicit
        // gAMA is a pure OETF power curve, NOT the piecewise sRGB transfer curve.
        let mut fields = ColorFields::srgb(space);
        if let Some(intent) = self.srgb {
            fields.rendering_intent = match intent {
                0 => RenderingIntent::Perceptual,
                1 => RenderingIntent::Relative,
                2 => RenderingIntent::Saturation,
                3 => RenderingIntent::Absolute,
                _ => return Err("invalid PNG sRGB intent".into()),
            };
        } else {
            if let Some(gamma) = self.gamma {
                fields.transfer_function = TransferFunction::Gamma(
                    gamma
                        .checked_mul(100)
                        .ok_or("PNG gamma exceeds JPEG XL range")?,
                );
            }
            if let Some(c) = self.chromaticities {
                let xy = |i: usize| Chromaticity {
                    x: (c[i] * 10) as i32,
                    y: (c[i + 1] * 10) as i32,
                };
                fields.white_point = WhitePoint::Custom(xy(0));
                // PNG gray samples do not use RGB primary chromaticities.
                if space == ColorSpace::Rgb {
                    fields.primaries = Some(Primaries::Custom {
                        red: xy(2),
                        green: xy(4),
                        blue: xy(6),
                    });
                }
            }
        }
        let encoding = ColorEncoding::Fields(fields);
        encoding.validate()?;
        Ok(encoding)
    }
}

#[derive(Debug)]
struct CheckedPng {
    dimensions: (u32, u32),
    color: PngColor,
}

fn strict_decoder(bytes: &[u8], ignore_icc: bool) -> png::Decoder<Cursor<&[u8]>> {
    let mut options = png::DecodeOptions::default();
    options.set_ignore_checksums(false);
    options.set_skip_ancillary_crc_failures(false);
    options.set_ignore_text_chunk(true);
    options.set_ignore_iccp_chunk(ignore_icc);
    let mut decoder = png::Decoder::new_with_options(Cursor::new(bytes), options);
    // This budget covers metadata and image buffers, not just ICC. It is a
    // ceiling, not an allocation request; allocate pixels from checked geometry.
    decoder.set_limits(png::Limits { bytes: usize::MAX });
    decoder
}

fn preflight(bytes: &[u8], strip: bool) -> Result<CheckedPng> {
    if !bytes.starts_with(SIGNATURE) {
        return Err("invalid PNG signature".into());
    }
    let mut offset = SIGNATURE.len();
    let mut dimensions = None;
    let mut image_data = false;
    let mut palette = false;
    let mut color = PngColor::default();
    // Inspect all chunks, including metadata after IDAT and malformed ICC that a
    // decoder might ignore. CRC and compressed-pixel validation remain with png.
    while offset < bytes.len() {
        let header = bytes
            .get(offset..offset + 8)
            .ok_or("truncated PNG chunk header")?;
        let length = u32::from_be_bytes(header[..4].try_into()?) as usize;
        if length > i32::MAX as usize {
            return Err("PNG chunk length exceeds the 31-bit format range".into());
        }
        let kind: [u8; 4] = header[4..8].try_into()?;
        if !kind.iter().all(u8::is_ascii_alphabetic) || !kind[2].is_ascii_uppercase() {
            return Err("invalid PNG chunk name".into());
        }
        let end = offset
            .checked_add(12)
            .and_then(|n| n.checked_add(length))
            .filter(|&n| n <= bytes.len())
            .ok_or("truncated or oversized PNG chunk")?;
        let data = &bytes[offset + 8..end - 4];
        if dimensions.is_none() && &kind != b"IHDR" {
            return Err("PNG must start with IHDR".into());
        }
        if matches!(&kind, b"sRGB" | b"gAMA" | b"cHRM" | b"iCCP") && (image_data || palette) {
            return Err("PNG color chunks must precede PLTE and IDAT".into());
        }
        match &kind {
            b"IHDR" => {
                if dimensions.is_some() || data.len() != 13 {
                    return Err("invalid or duplicate PNG IHDR".into());
                }
                let width = u32::from_be_bytes(data[..4].try_into()?);
                let height = u32::from_be_bytes(data[4..8].try_into()?);
                if width > i32::MAX as u32 || height > i32::MAX as u32 {
                    return Err("PNG dimensions exceed the 31-bit format range".into());
                }
                // Check the largest possible expansion (palette/tRNS included).
                let sample_bytes = if data[8] == 16 { 2 } else { 1 };
                checked_size(width, height, 4, sample_bytes)?;
                dimensions = Some((width, height));
            }
            b"IDAT" => image_data = true,
            b"PLTE" => palette = true,
            b"tRNS" => {}
            b"sRGB" => {
                if color.srgb.is_some() || data.len() != 1 || data[0] > 3 {
                    return Err("invalid or duplicate PNG sRGB".into());
                }
                color.srgb = Some(data[0]);
            }
            b"gAMA" => {
                if color.gamma.is_some() || data.len() != 4 {
                    return Err("invalid or duplicate PNG gAMA".into());
                }
                let gamma = u32::from_be_bytes(data.try_into()?);
                if gamma == 0 {
                    return Err("PNG gAMA must be nonzero".into());
                }
                color.gamma = Some(gamma);
            }
            b"cHRM" => {
                if color.chromaticities.is_some() || data.len() != 32 {
                    return Err("invalid or duplicate PNG cHRM".into());
                }
                let mut values = [0u32; 8];
                for (v, chunk) in values.iter_mut().zip(data.chunks_exact(4)) {
                    *v = u32::from_be_bytes(chunk.try_into()?);
                }
                if values
                    .chunks_exact(2)
                    .any(|xy| xy[0] > 100000 || xy[1] > 100000 || xy[0] + xy[1] > 100000)
                    || values[1] == 0
                {
                    return Err("invalid PNG cHRM coordinates".into());
                }
                color.chromaticities = Some(values);
            }
            b"iCCP" => {
                if color.icc {
                    return Err("duplicate PNG iCCP".into());
                }
                let end = data
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or("PNG iCCP is missing the profile-name terminator")?;
                let name = &data[..end];
                if !(1..=79).contains(&name.len())
                    || name.first() == Some(&b' ')
                    || name.last() == Some(&b' ')
                    || name.windows(2).any(|p| p == b"  ")
                    || name.iter().any(|&b| !matches!(b, 32..=126 | 161..=255))
                    || data.get(end + 1) != Some(&0)
                    || data.len() <= end + 2
                {
                    return Err("invalid PNG iCCP name, compression method or empty profile".into());
                }
                color.icc = true;
            }
            b"acTL" | b"fcTL" | b"fdAT" => {
                return Err("animated PNG is unsupported; no frames are silently discarded".into());
            }
            b"IEND" => {
                if !data.is_empty() || !image_data || end != bytes.len() {
                    return Err("invalid PNG IEND, missing pixels, or trailing data".into());
                }
                color.validate()?;
                return Ok(CheckedPng {
                    dimensions: dimensions.ok_or("PNG is missing IHDR")?,
                    color,
                });
            }
            _ if kind[0].is_ascii_lowercase() => {
                metadata(&format!("PNG {}", std::str::from_utf8(&kind)?), strip)?;
            }
            _ => {
                return Err(format!(
                    "unsupported critical PNG chunk {}",
                    std::str::from_utf8(&kind)?
                )
                .into());
            }
        }
        offset = end;
    }
    Err("PNG is missing IEND".into())
}

pub(super) fn decode(bytes: &[u8], strip: bool) -> Result<Image> {
    let checked = preflight(bytes, strip)?;
    let dimensions = checked.dimensions;
    let mut decoder = strict_decoder(bytes, true);
    // EXPAND handles packed gray, palettes and tRNS without STRIP_16 or
    // premultiplication. ICC is inflated separately without an extra size cap.
    decoder.set_transformations(png::Transformations::EXPAND);
    decoder.set_ignore_text_chunk(true);
    decoder.set_ignore_iccp_chunk(true);
    let mut reader = decoder.read_info()?;
    let (color, depth) = reader.output_color_type();
    let layout = match color {
        png::ColorType::Grayscale => PixelLayout::Gray,
        png::ColorType::GrayscaleAlpha => PixelLayout::GrayAlpha,
        png::ColorType::Rgb => PixelLayout::Rgb,
        png::ColorType::Rgba => PixelLayout::Rgba,
        png::ColorType::Indexed => return Err("PNG palette was not expanded".into()),
    };
    let sample_bytes = match depth {
        png::BitDepth::Eight => 1,
        png::BitDepth::Sixteen => 2,
        _ => return Err("PNG packed samples were not expanded".into()),
    };
    let space = if matches!(layout, PixelLayout::Gray | PixelLayout::GrayAlpha) {
        ColorSpace::Gray
    } else {
        ColorSpace::Rgb
    };
    let encoding = checked.color.encoding(bytes, space)?;
    let size = checked_size(dimensions.0, dimensions.1, layout.channels(), sample_bytes)?;
    if reader.output_buffer_size() != Some(size) {
        return Err("PNG decoded buffer size disagrees with validated dimensions".into());
    }
    let mut buffer = zeroed_bytes(size)?;
    let frame = reader.next_frame(&mut buffer)?;
    if (frame.width, frame.height) != dimensions
        || frame.buffer_size() != size
        || frame.color_type != color
        || frame.bit_depth != depth
    {
        return Err("PNG frame differs from the validated image layout".into());
    }
    reader.finish()?;
    let pixels = if sample_bytes == 2 {
        let mut samples = Vec::new();
        samples.try_reserve_exact(size / 2)?;
        for pair in buffer.chunks_exact(2) {
            samples.push(u16::from_be_bytes([pair[0], pair[1]]));
        }
        PixelData::U16(samples)
    } else {
        PixelData::U8(buffer)
    };
    let image = Image::new(dimensions.0, dimensions.1, layout, pixels)?;
    let mut metadata = image.metadata().clone();
    metadata.color_encoding = encoding;
    Ok(image.with_metadata(metadata, Vec::new())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use png::{BitDepth, ColorType};

    fn fixture(
        color: ColorType,
        depth: BitDepth,
        width: u32,
        pixels: &[u8],
        palette: Option<(&[u8], &[u8])>,
        trailing: Option<([u8; 4], &[u8])>,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, 1);
            encoder.set_color(color);
            encoder.set_depth(depth);
            if let Some((palette, alpha)) = palette {
                encoder.set_palette(palette.to_vec());
                encoder.set_trns(alpha.to_vec());
            }
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
            if let Some((kind, data)) = trailing {
                writer
                    .write_chunk(png::chunk::ChunkType(kind), data)
                    .unwrap();
            }
            writer.finish().unwrap();
        }
        bytes
    }

    #[test]
    fn decodes_dimensions_above_the_former_cli_cap() {
        let values = vec![71; 16_385];
        let bytes = fixture(
            ColorType::Grayscale,
            BitDepth::Eight,
            16_385,
            &values,
            None,
            None,
        );
        let image = decode(&bytes, false).unwrap();
        assert_eq!((image.width(), image.height()), (16_385, 1));
        assert_eq!(image.pixels(), &PixelData::U8(values));
    }

    #[test]
    fn large_png_headers_do_not_allocate_an_image_or_hit_a_metadata_budget() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 16_385, 16_385);
            encoder.set_color(ColorType::Rgba);
            encoder.set_depth(BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            // Header inspection does not need a complete pixel stream.
            writer.write_chunk(png::chunk::IDAT, &[0x78, 0x01]).unwrap();
            writer.finish().unwrap();
        }
        assert_eq!(
            preflight(&bytes, false).unwrap().dimensions,
            (16_385, 16_385)
        );
        for ignore_icc in [false, true] {
            let reader = strict_decoder(&bytes, ignore_icc).read_info().unwrap();
            assert_eq!(reader.output_buffer_size(), Some(16_385 * 16_385 * 4));
        }
    }

    #[test]
    fn every_eight_bit_layout_preserves_samples_including_invisible_rgb() {
        for (color, layout, values) in [
            (ColorType::Grayscale, PixelLayout::Gray, vec![0, 255]),
            (
                ColorType::GrayscaleAlpha,
                PixelLayout::GrayAlpha,
                vec![71, 0, 255, 255],
            ),
            (
                ColorType::Rgb,
                PixelLayout::Rgb,
                vec![1, 127, 255, 0, 41, 99],
            ),
            (
                ColorType::Rgba,
                PixelLayout::Rgba,
                vec![19, 87, 231, 0, 0, 1, 2, 255],
            ),
        ] {
            let bytes = fixture(color, BitDepth::Eight, 2, &values, None, None);
            let image = decode(&bytes, false).unwrap();
            assert_eq!(image.layout(), layout);
            assert_eq!(image.pixels(), &PixelData::U8(values));
            assert_eq!((image.width(), image.height(), image.bits()), (2, 1, 8));
        }
    }

    #[test]
    fn every_sixteen_bit_layout_uses_big_endian_without_truncation() {
        for (color, layout) in [
            (ColorType::Grayscale, PixelLayout::Gray),
            (ColorType::GrayscaleAlpha, PixelLayout::GrayAlpha),
            (ColorType::Rgb, PixelLayout::Rgb),
            (ColorType::Rgba, PixelLayout::Rgba),
        ] {
            let samples = [0x1234_u16, 0xabcd, 0xffff, 0][..layout.channels()].to_vec();
            let pixels: Vec<_> = samples
                .iter()
                .flat_map(|sample| sample.to_be_bytes())
                .collect();
            let bytes = fixture(color, BitDepth::Sixteen, 1, &pixels, None, None);
            let image = decode(&bytes, false).unwrap();
            assert_eq!(image.layout(), layout);
            assert_eq!(image.bits(), 16);
            assert_eq!(image.pixels(), &PixelData::U16(samples));
        }
    }

    #[test]
    fn packed_gray_expands_exactly_to_eight_bits() {
        for (depth, width, packed, expected) in [
            (BitDepth::One, 4, vec![0b0101_0000], vec![0, 255, 0, 255]),
            (BitDepth::Two, 4, vec![0b0001_1011], vec![0, 85, 170, 255]),
            (BitDepth::Four, 2, vec![0x1f], vec![17, 255]),
        ] {
            let bytes = fixture(ColorType::Grayscale, depth, width, &packed, None, None);
            assert_eq!(
                decode(&bytes, false).unwrap().pixels(),
                &PixelData::U8(expected)
            );
        }
    }

    #[test]
    fn palette_transparency_keeps_hidden_color() {
        let bytes = fixture(
            ColorType::Indexed,
            BitDepth::One,
            2,
            &[0b0100_0000],
            Some((&[23, 45, 67, 101, 102, 103], &[0, 255])),
            None,
        );
        let image = decode(&bytes, false).unwrap();
        assert_eq!(image.layout(), PixelLayout::Rgba);
        assert_eq!(
            image.pixels(),
            &PixelData::U8(vec![23, 45, 67, 0, 101, 102, 103, 255])
        );
    }

    #[test]
    fn truecolor_sixteen_bit_trns_keeps_all_color_samples() {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(ColorType::Rgb);
            encoder.set_depth(BitDepth::Sixteen);
            encoder.set_trns(vec![0x12, 0x34, 0xab, 0xcd, 0, 1]);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&[0x12, 0x34, 0xab, 0xcd, 0, 1, 0x12, 0x34, 0xab, 0xcd, 0, 2])
                .unwrap();
            writer.finish().unwrap();
        }
        let image = decode(&bytes, false).unwrap();
        assert_eq!(image.layout(), PixelLayout::Rgba);
        assert_eq!(
            image.pixels(),
            &PixelData::U16(vec![0x1234, 0xabcd, 1, 0, 0x1234, 0xabcd, 2, 65535,])
        );
    }

    #[test]
    fn metadata_after_pixels_is_not_silently_dropped() {
        let bytes = fixture(
            ColorType::Grayscale,
            BitDepth::Eight,
            1,
            &[71],
            None,
            Some((*b"tEXt", b"Comment\0test")),
        );
        assert!(
            decode(&bytes, false)
                .unwrap_err()
                .to_string()
                .contains("--strip-metadata")
        );
        assert_eq!(
            decode(&bytes, true).unwrap().pixels(),
            &PixelData::U8(vec![71])
        );
    }

    #[test]
    fn strict_metadata_checks_even_invalid_profiles_and_unknown_chunks() {
        for kind in [*b"eXIf", *b"cICP", *b"mDCV", *b"cLLI", *b"vpAg"] {
            let bytes = fixture(
                ColorType::Grayscale,
                BitDepth::Eight,
                1,
                &[71],
                None,
                Some((kind, b"")),
            );
            let error = preflight(&bytes, false).unwrap_err().to_string();
            assert!(error.contains("--strip-metadata"), "{kind:?}: {error}");
            assert!(preflight(&bytes, true).is_ok());
        }
    }

    fn color_fixture(color: ColorType, chunks: &[([u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(color);
            encoder.set_depth(BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            for (kind, data) in chunks {
                writer
                    .write_chunk(png::chunk::ChunkType(*kind), data)
                    .unwrap();
            }
            writer.write_image_data(&vec![71; color.samples()]).unwrap();
            writer.finish().unwrap();
        }
        bytes
    }

    fn icc_chunk(profile: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut info = png::Info::with_size(1, 1);
            info.bit_depth = BitDepth::Eight;
            info.color_type = ColorType::Rgb;
            info.icc_profile = Some(std::borrow::Cow::Borrowed(profile));
            let mut writer = png::Encoder::with_info(&mut bytes, info)
                .unwrap()
                .write_header()
                .unwrap();
            writer.write_image_data(&[0; 3]).unwrap();
            writer.finish().unwrap();
        }
        let mut pos = 8;
        while pos < bytes.len() {
            let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            if &bytes[pos + 4..pos + 8] == b"iCCP" {
                return bytes[pos + 8..pos + 8 + len].to_vec();
            }
            pos += 12 + len;
        }
        panic!("encoder omitted ICC");
    }

    fn profile(space: &[u8; 4]) -> Vec<u8> {
        let mut p = vec![0; 132];
        p[..4].copy_from_slice(&132u32.to_be_bytes());
        p[16..20].copy_from_slice(space);
        p[20..24].copy_from_slice(b"XYZ ");
        p[36..40].copy_from_slice(b"acsp");
        p
    }

    #[test]
    fn srgb_intents_preserve_color_without_changing_samples_or_alpha() {
        for (intent, expected) in [
            (0, RenderingIntent::Perceptual),
            (1, RenderingIntent::Relative),
            (2, RenderingIntent::Saturation),
            (3, RenderingIntent::Absolute),
        ] {
            for color in [ColorType::GrayscaleAlpha, ColorType::Rgba] {
                let chunks = [
                    (*b"sRGB", vec![intent]),
                    (*b"gAMA", 45455u32.to_be_bytes().to_vec()),
                    (
                        *b"cHRM",
                        SRGB_CHROMATICITIES
                            .iter()
                            .flat_map(|v| v.to_be_bytes())
                            .collect(),
                    ),
                ];
                let bytes = color_fixture(color, &chunks);
                for strip in [false, true] {
                    let image = decode(&bytes, strip).unwrap();
                    let ColorEncoding::Fields(fields) = &image.metadata().color_encoding else {
                        panic!("fields expected");
                    };
                    assert_eq!(fields.rendering_intent, expected);
                    assert_eq!(fields.transfer_function, TransferFunction::Srgb);
                    assert_eq!(image.pixels(), &PixelData::U8(vec![71; color.samples()]));
                    assert_eq!(image.metadata().extra_channels.len(), 1);
                }
            }
        }
    }

    #[test]
    fn gamma_and_chromaticities_remain_explicit_power_curve_and_coordinates() {
        let coordinates: [u32; 8] = [34570, 35850, 68000, 32000, 26500, 69000, 15000, 6000];
        let chunks = [
            (*b"gAMA", 45455u32.to_be_bytes().to_vec()),
            (
                *b"cHRM",
                coordinates.iter().flat_map(|v| v.to_be_bytes()).collect(),
            ),
        ];
        for color in [ColorType::Grayscale, ColorType::Rgb] {
            let image = decode(&color_fixture(color, &chunks), false).unwrap();
            let ColorEncoding::Fields(f) = &image.metadata().color_encoding else {
                panic!("fields expected");
            };
            assert_eq!(f.transfer_function, TransferFunction::Gamma(4_545_500));
            assert_eq!(
                f.white_point,
                WhitePoint::Custom(Chromaticity {
                    x: 345700,
                    y: 358500
                })
            );
            if color == ColorType::Grayscale {
                assert_eq!(f.primaries, None);
            } else {
                assert_eq!(
                    f.primaries,
                    Some(Primaries::Custom {
                        red: Chromaticity {
                            x: 680000,
                            y: 320000
                        },
                        green: Chromaticity {
                            x: 265000,
                            y: 690000
                        },
                        blue: Chromaticity {
                            x: 150000,
                            y: 60000
                        },
                    })
                );
            }
        }
        for gamma in [0u32, 1, 12, 100001, u32::MAX] {
            assert!(
                decode(
                    &color_fixture(ColorType::Rgb, &[(*b"gAMA", gamma.to_be_bytes().to_vec())]),
                    false
                )
                .is_err()
            );
        }
        for gamma in [13u32, 100000] {
            decode(
                &color_fixture(ColorType::Rgb, &[(*b"gAMA", gamma.to_be_bytes().to_vec())]),
                false,
            )
            .unwrap();
        }
    }

    #[test]
    fn icc_bytes_survive_exactly_in_gray_rgb_and_with_fallback_fields() {
        for (color, space) in [(ColorType::Grayscale, b"GRAY"), (ColorType::Rgb, b"RGB ")] {
            let p = profile(space);
            // gAMA/cHRM are fallback descriptions when an ICC is authoritative.
            let chunks = [
                (*b"iCCP", icc_chunk(&p)),
                (*b"gAMA", 100000u32.to_be_bytes().to_vec()),
            ];
            for strip in [false, true] {
                let image = decode(&color_fixture(color, &chunks), strip).unwrap();
                assert_eq!(
                    image.metadata().color_encoding.icc_profile(),
                    Some(p.as_slice())
                );
                assert_eq!(image.pixels(), &PixelData::U8(vec![71; color.samples()]));
            }
            let mismatch = if color == ColorType::Rgb {
                ColorType::Grayscale
            } else {
                ColorType::Rgb
            };
            assert!(decode(&color_fixture(mismatch, &chunks), false).is_err());
        }
    }

    #[test]
    fn rejects_duplicate_conflicting_late_or_corrupt_color_even_when_stripping() {
        let good_icc = icc_chunk(&profile(b"RGB "));
        let gamma = 45455u32.to_be_bytes().to_vec();
        for chunk in [
            (*b"sRGB", vec![0]),
            (*b"gAMA", gamma.clone()),
            (
                *b"cHRM",
                SRGB_CHROMATICITIES
                    .iter()
                    .flat_map(|v| v.to_be_bytes())
                    .collect(),
            ),
            (*b"iCCP", good_icc.clone()),
        ] {
            for strip in [false, true] {
                assert!(
                    decode(
                        &color_fixture(ColorType::Rgb, &[chunk.clone(), chunk.clone()]),
                        strip
                    )
                    .is_err()
                );
                let late = fixture(
                    ColorType::Rgb,
                    BitDepth::Eight,
                    1,
                    &[71; 3],
                    None,
                    Some((chunk.0, &chunk.1)),
                );
                assert!(decode(&late, strip).is_err());
            }
        }
        for chunks in [
            vec![(*b"sRGB", vec![0]), (*b"iCCP", good_icc)],
            vec![
                (*b"sRGB", vec![1]),
                (*b"gAMA", 100000u32.to_be_bytes().to_vec()),
            ],
            vec![(*b"sRGB", vec![4])],
            vec![(*b"iCCP", b"ICC\0\0invalid zlib".to_vec())],
            vec![(*b"iCCP", b" bad name\0\0x".to_vec())],
            vec![(*b"iCCP", b"ICC\0\x01x".to_vec())],
            vec![(*b"iCCP", icc_chunk(b"not an ICC profile"))],
        ] {
            let bytes = color_fixture(ColorType::Rgb, &chunks);
            for strip in [false, true] {
                assert!(decode(&bytes, strip).is_err());
            }
        }
        let mut bytes = color_fixture(ColorType::Rgb, &[(*b"gAMA", gamma)]);
        // IHDR ends at 33, then length/type, 4 gamma bytes, CRC.
        bytes[45] ^= 1;
        assert!(decode(&bytes, true).is_err());
    }

    #[test]
    fn icc_inflation_accepts_profiles_above_the_former_16_mib_cap() {
        let mut large = profile(b"RGB ");
        large.resize(16 * 1024 * 1024 + 4, 0);
        let size = large.len() as u32;
        large[..4].copy_from_slice(&size.to_be_bytes());
        let bytes = color_fixture(ColorType::Rgb, &[(*b"iCCP", icc_chunk(&large))]);
        let image = decode(&bytes, false).unwrap();
        assert_eq!(
            image.metadata().color_encoding.icc_profile(),
            Some(large.as_slice())
        );
    }

    #[test]
    fn animation_is_rejected_even_when_stripping() {
        let bytes = fixture(
            ColorType::Grayscale,
            BitDepth::Eight,
            1,
            &[71],
            None,
            Some((*b"acTL", &[0; 8])),
        );
        assert!(
            preflight(&bytes, true)
                .unwrap_err()
                .to_string()
                .contains("animated")
        );
    }

    #[test]
    fn rejects_truncation_trailing_data_bad_crc_and_huge_dimensions() {
        let bytes = fixture(ColorType::Grayscale, BitDepth::Eight, 1, &[71], None, None);
        for end in 0..bytes.len() {
            assert!(
                decode(&bytes[..end], true).is_err(),
                "accepted prefix {end}"
            );
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode(&trailing, true).is_err());
        let mut corrupt = bytes.clone();
        corrupt[29] ^= 1;
        assert!(decode(&corrupt, true).is_err());
        let mut huge = bytes;
        huge[16..20].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(preflight(&huge, true).is_err());
    }
}
