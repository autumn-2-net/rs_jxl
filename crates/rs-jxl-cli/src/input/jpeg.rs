use std::borrow::Cow;

use rs_jxl::color::{ColorEncoding, ColorSpace as JxlColorSpace};
use rs_jxl::{Image, PixelData, PixelLayout};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};

use super::{MAX_DIMENSION, checked_size, metadata, zeroed_bytes};
use crate::Result;

const MAX_ICC_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug)]
struct CheckedJpeg<'a> {
    width: u32,
    height: u32,
    components: u8,
    bytes: Cow<'a, [u8]>,
    icc: Option<Vec<u8>>,
}

fn next_marker(bytes: &[u8], offset: &mut usize, entropy: bool) -> Result<(usize, u8)> {
    loop {
        if entropy {
            while bytes.get(*offset).is_some_and(|&byte| byte != 0xff) {
                *offset += 1;
            }
        }
        let start = *offset;
        if bytes.get(*offset) != Some(&0xff) {
            return Err("truncated JPEG or missing marker".into());
        }
        while bytes.get(*offset) == Some(&0xff) {
            *offset += 1;
        }
        let marker = *bytes.get(*offset).ok_or("truncated JPEG marker")?;
        *offset += 1;
        if entropy && (marker == 0 || (0xd0..=0xd7).contains(&marker)) {
            continue;
        }
        if marker == 0 || (0xd0..=0xd7).contains(&marker) {
            return Err("unexpected JPEG stuffed byte or restart marker".into());
        }
        return Ok((start, marker));
    }
}

fn frame(data: &[u8]) -> Result<(u32, u32, u8)> {
    if data.len() < 6 {
        return Err("truncated JPEG frame header".into());
    }
    let components = data[5];
    if data[0] != 8 || !matches!(components, 1 | 3) {
        return Err("only 8-bit grayscale/RGB JPEG is supported; CMYK/YCCK and other depths are unsupported".into());
    }
    if data.len() != 6 + 3 * usize::from(components) {
        return Err("invalid JPEG frame component table".into());
    }
    let height = u32::from(u16::from_be_bytes(data[1..3].try_into()?));
    let width = u32::from(u16::from_be_bytes(data[3..5].try_into()?));
    checked_size(width, height, usize::from(components), 1)?;
    let mut blocks = 0;
    for component in data[6..].chunks_exact(3) {
        let (horizontal, vertical) = (component[1] >> 4, component[1] & 15);
        if !(1..=4).contains(&horizontal) || !(1..=4).contains(&vertical) {
            return Err("invalid JPEG sampling factors".into());
        }
        blocks += u32::from(horizontal) * u32::from(vertical);
    }
    if blocks > 10 {
        return Err("JPEG sampling exceeds the maximum blocks per MCU".into());
    }
    Ok((width, height, components))
}

fn preflight(bytes: &[u8], strip: bool) -> Result<CheckedJpeg<'_>> {
    if !bytes.starts_with(&[0xff, 0xd8]) {
        return Err("invalid JPEG signature".into());
    }
    let mut offset = 2;
    let mut dimensions = None;
    let mut scans = 0;
    let mut entropy = false;
    let mut filtered: Option<Vec<u8>> = None;
    let mut copied_until = 0;
    let mut icc_parts: Vec<Option<&[u8]>> = Vec::new();
    let mut icc_bytes = 0usize;
    // Walk through every scan, not just the headers before the first SOS.
    // Removing explicitly discarded APP/COM data also avoids unnecessary
    // decoder-side allocation for ICC and extended XMP payloads.
    loop {
        let (start, marker) = next_marker(bytes, &mut offset, entropy)?;
        entropy = false;
        if marker == 0xd9 {
            if scans == 0 || offset != bytes.len() {
                return Err("JPEG has no scan or has trailing data/additional images".into());
            }
            let (width, height, components) = dimensions.ok_or("JPEG is missing a frame header")?;
            let icc = if icc_parts.is_empty() {
                None
            } else {
                let mut profile = Vec::new();
                profile.try_reserve_exact(icc_bytes)?;
                for part in icc_parts {
                    profile
                        .extend_from_slice(part.ok_or("JPEG ICC is missing a split APP2 segment")?);
                }
                let actual = rs_jxl::icc::validate_profile(&profile)?;
                let expected = if components == 1 {
                    rs_jxl::icc::IccColorSpace::Gray
                } else {
                    rs_jxl::icc::IccColorSpace::Rgb
                };
                if actual != expected {
                    return Err("JPEG ICC color space does not match decoded samples".into());
                }
                Some(profile)
            };
            let bytes = if let Some(mut clean) = filtered {
                clean.extend_from_slice(&bytes[copied_until..]);
                Cow::Owned(clean)
            } else {
                Cow::Borrowed(bytes)
            };
            return Ok(CheckedJpeg {
                width,
                height,
                components,
                bytes,
                icc,
            });
        }
        if marker == 0xd8 || marker == 0x01 {
            return Err("unexpected standalone JPEG marker".into());
        }
        let length_bytes = bytes
            .get(offset..offset + 2)
            .ok_or("truncated JPEG segment length")?;
        let length = usize::from(u16::from_be_bytes(length_bytes.try_into()?));
        if length < 2 {
            return Err("invalid JPEG segment length".into());
        }
        let end = offset
            .checked_add(length)
            .filter(|&end| end <= bytes.len())
            .ok_or("truncated JPEG segment")?;
        let data = &bytes[offset + 2..end];
        let discard = match marker {
            0xc0..=0xc2 => {
                if dimensions.is_some() || scans != 0 {
                    return Err("multiple JPEG frames are unsupported".into());
                }
                dimensions = Some(frame(data)?);
                false
            }
            0xc4 | 0xdb | 0xdd => false,
            0xda => {
                let (_, _, components) = dimensions.ok_or("JPEG scan precedes its frame header")?;
                let count = *data.first().ok_or("empty JPEG scan header")?;
                if count == 0 || count > components || data.len() != 4 + 2 * usize::from(count) {
                    return Err("invalid JPEG scan header".into());
                }
                scans += 1;
                if scans > 100 {
                    return Err("JPEG exceeds the 100 scan limit".into());
                }
                entropy = true;
                false
            }
            0xe0 if data.starts_with(b"JFIF\0") => {
                if data.len() < 14 || scans != 0 {
                    return Err("invalid or late JPEG JFIF header".into());
                }
                let thumbnail = usize::from(data[12]) * usize::from(data[13]) * 3;
                if data.len() != 14 + thumbnail {
                    return Err("invalid JPEG JFIF thumbnail size".into());
                }
                let has_metadata = data[7] != 0 || data[8..10] != data[10..12] || thumbnail != 0;
                if has_metadata {
                    metadata("JPEG JFIF density/aspect/thumbnail", strip)?;
                }
                has_metadata
            }
            0xee if data.starts_with(b"Adobe") => {
                if data.len() != 12 || scans != 0 || data[11] > 1 {
                    return Err("unsupported JPEG Adobe color transform or late marker (CMYK/YCCK is unsupported)".into());
                }
                // APP14 controls pixel decoding and must remain even when stripping.
                if data[7..11] != [0, 0, 0, 0] {
                    metadata("JPEG Adobe flags", strip)?;
                }
                false
            }
            0xe2 if data.starts_with(b"ICC_PROFILE") => {
                if scans != 0 || data.len() <= 14 || !data.starts_with(b"ICC_PROFILE\0") {
                    return Err("malformed or late JPEG ICC APP2 segment".into());
                }
                let sequence = usize::from(data[12]);
                let count = usize::from(data[13]);
                if count == 0 || sequence == 0 || sequence > count {
                    return Err("invalid JPEG ICC split sequence/count".into());
                }
                if icc_parts.is_empty() { icc_parts.resize(count, None); }
                if icc_parts.len() != count || icc_parts[sequence - 1].is_some() {
                    return Err("duplicate or inconsistent JPEG ICC split segments".into());
                }
                icc_bytes = icc_bytes.checked_add(data.len() - 14).filter(|&n| n <= MAX_ICC_BYTES).ok_or("JPEG ICC exceeds 16 MiB")?;
                icc_parts[sequence - 1] = Some(&data[14..]);
                // Keep ICC outside the pixel decoder to avoid its own partial
                // APP2 assembly. Attach the original joined bytes to Image.
                true
            }
            0xe0..=0xef | 0xfe => {
                let kind = if marker == 0xe1 && data.starts_with(b"Exif\0\0") {
                    "EXIF".to_owned()
                } else if marker == 0xfe {
                    "comment".to_owned()
                } else {
                    format!("APP{}", marker - 0xe0)
                };
                metadata(&format!("JPEG {kind}"), strip)?;
                true
            }
            _ => return Err(format!("unsupported JPEG marker FF{marker:02X} (only baseline/extended/progressive Huffman JPEG is supported)").into()),
        };
        if discard {
            if filtered.is_none() {
                let mut clean = Vec::new();
                clean.try_reserve_exact(bytes.len())?;
                filtered = Some(clean);
            }
            let clean = filtered
                .as_mut()
                .ok_or("JPEG metadata filter is unavailable")?;
            clean.extend_from_slice(&bytes[copied_until..start]);
            copied_until = end;
        }
        offset = end;
    }
}

pub(super) fn decode(bytes: &[u8], strip: bool) -> Result<Image> {
    let checked = preflight(bytes, strip)?;
    let (layout, output) = if checked.components == 1 {
        (PixelLayout::Gray, ColorSpace::Luma)
    } else {
        (PixelLayout::Rgb, ColorSpace::RGB)
    };
    let options = DecoderOptions::default()
        .set_strict_mode(true)
        .set_max_width(MAX_DIMENSION as usize)
        .set_max_height(MAX_DIMENSION as usize)
        .jpeg_set_max_scans(100)
        .jpeg_set_out_colorspace(output);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(checked.bytes.as_ref()), options);
    decoder.decode_headers()?;
    if !matches!(
        decoder.input_colorspace(),
        Some(ColorSpace::Luma | ColorSpace::RGB | ColorSpace::YCbCr)
    ) {
        return Err(
            "JPEG decoder reports an unsupported input color space (CMYK/YCCK is unsupported)"
                .into(),
        );
    }
    let info = decoder.info().ok_or("JPEG decoder returned no header")?;
    if (
        u32::from(info.width),
        u32::from(info.height),
        info.components,
    ) != (checked.width, checked.height, checked.components)
        || decoder.output_colorspace() != Some(output)
    {
        return Err("JPEG decoder layout disagrees with validated headers".into());
    }
    let size = checked_size(checked.width, checked.height, layout.channels(), 1)?;
    if decoder.output_buffer_size() != Some(size) {
        return Err("JPEG decoder buffer size disagrees with validated dimensions".into());
    }
    let mut pixels = zeroed_bytes(size)?;
    decoder.decode_into(&mut pixels)?;
    let image = Image::new(checked.width, checked.height, layout, PixelData::U8(pixels))?;
    let mut metadata = image.metadata().clone();
    if let Some(profile) = checked.icc {
        let space = if checked.components == 1 {
            JxlColorSpace::Gray
        } else {
            JxlColorSpace::Rgb
        };
        metadata.color_encoding = ColorEncoding::icc(space, profile)?;
    }
    Ok(image.with_metadata(metadata, Vec::new())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(bytes: &mut Vec<u8>, marker: u8, data: &[u8]) {
        bytes.extend_from_slice(&[0xff, marker]);
        bytes.extend_from_slice(&u16::try_from(data.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(data);
    }

    // One constant MCU, using minimal DC-zero/EOB Huffman tables. This is a
    // deterministic test fixture, not an alternate JPEG encoder implementation.
    fn fixture(components: u8) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8];
        let mut quant = vec![1; 65];
        quant[0] = 0;
        segment(&mut bytes, 0xdb, &quant);
        let mut header = vec![8, 0, 2, 0, 3, components];
        for id in 1..=components {
            header.extend_from_slice(&[id, 0x11, 0]);
        }
        segment(&mut bytes, 0xc0, &header);
        let mut tables = Vec::new();
        for selector in [0, 0x10] {
            tables.extend_from_slice(&[selector, 1]);
            tables.extend_from_slice(&[0; 15]);
            tables.push(0);
        }
        segment(&mut bytes, 0xc4, &tables);
        let mut scan = vec![components];
        for id in 1..=components {
            scan.extend_from_slice(&[id, 0]);
        }
        scan.extend_from_slice(&[0, 63, 0]);
        segment(&mut bytes, 0xda, &scan);
        bytes.push(if components == 1 { 0x3f } else { 0x03 });
        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }

    fn with_metadata(bytes: &[u8], marker: u8, payload: &[u8], late: bool) -> Vec<u8> {
        let offset = if late { bytes.len() - 2 } else { 2 };
        let mut result = bytes[..offset].to_vec();
        segment(&mut result, marker, payload);
        result.extend_from_slice(&bytes[offset..]);
        result
    }

    #[test]
    fn decodes_gray_and_rgb_exactly_as_zune_samples() {
        for components in [1, 3] {
            let bytes = fixture(components);
            let image = decode(&bytes, false).unwrap();
            let colorspace = if components == 1 {
                ColorSpace::Luma
            } else {
                ColorSpace::RGB
            };
            let options = DecoderOptions::default()
                .set_strict_mode(true)
                .jpeg_set_out_colorspace(colorspace);
            let direct = JpegDecoder::new_with_options(ZCursor::new(&bytes), options)
                .decode()
                .unwrap();
            assert_eq!((image.width(), image.height()), (3, 2));
            assert_eq!(image.pixels(), &PixelData::U8(direct));
            assert_eq!(
                image.pixels(),
                &PixelData::U8(vec![128; 6 * usize::from(components)])
            );
        }
    }

    #[test]
    fn exif_is_explicitly_discarded_without_orientation_transform() {
        let plain = fixture(3);
        let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let bytes = with_metadata(&plain, 0xe1, exif, false);
        assert!(
            decode(&bytes, false)
                .unwrap_err()
                .to_string()
                .contains("--strip-metadata")
        );
        let image = decode(&bytes, true).unwrap();
        assert_eq!((image.width(), image.height()), (3, 2));
        assert_eq!(image.pixels(), decode(&plain, false).unwrap().pixels());
    }

    #[test]
    fn catches_unknown_app_and_comments_before_and_after_scan() {
        for (marker, payload) in [
            (0xe1, &b"http://ns.adobe.com/xap/1.0/\0test"[..]),
            (0xed, &b"custom"[..]),
            (0xfe, &b"comment"[..]),
        ] {
            for late in [false, true] {
                let plain = fixture(1);
                let bytes = with_metadata(&plain, marker, payload, late);
                assert!(
                    preflight(&bytes, false)
                        .unwrap_err()
                        .to_string()
                        .contains("--strip-metadata")
                );
                let checked = preflight(&bytes, true).unwrap();
                assert_eq!(checked.bytes.as_ref(), &plain);
                assert_eq!(
                    decode(&bytes, true).unwrap().pixels(),
                    &PixelData::U8(vec![128; 6])
                );
            }
        }
    }

    fn profile(components: u8) -> Vec<u8> {
        let mut p = vec![0; 132];
        p[..4].copy_from_slice(&132u32.to_be_bytes());
        p[16..20].copy_from_slice(if components == 1 { b"GRAY" } else { b"RGB " });
        p[20..24].copy_from_slice(b"XYZ ");
        p[36..40].copy_from_slice(b"acsp");
        p
    }

    fn icc_segment(sequence: u8, count: u8, part: &[u8]) -> Vec<u8> {
        let mut bytes = b"ICC_PROFILE\0".to_vec();
        bytes.extend_from_slice(&[sequence, count]);
        bytes.extend_from_slice(part);
        bytes
    }

    #[test]
    fn reassembles_out_of_order_icc_and_preserves_samples_while_stripping_other_metadata() {
        for components in [1, 3] {
            let plain = fixture(components);
            let p = profile(components);
            let bytes = with_metadata(&plain, 0xe2, &icc_segment(1, 3, &p[..50]), false);
            let bytes = with_metadata(&bytes, 0xe2, &icc_segment(3, 3, &p[100..]), false);
            let bytes = with_metadata(&bytes, 0xe2, &icc_segment(2, 3, &p[50..100]), false);
            let checked = preflight(&bytes, false).unwrap();
            assert_eq!(checked.bytes.as_ref(), plain);
            assert_eq!(checked.icc.as_deref(), Some(p.as_slice()));
            for strip in [false, true] {
                let image = decode(&bytes, strip).unwrap();
                assert_eq!(
                    image.metadata().color_encoding.icc_profile(),
                    Some(p.as_slice())
                );
                assert_eq!(
                    image.pixels(),
                    &PixelData::U8(vec![128; 6 * usize::from(components)])
                );
            }
            let commented = with_metadata(&bytes, 0xfe, b"comment", true);
            assert!(decode(&commented, false).is_err());
            assert_eq!(
                decode(&commented, true)
                    .unwrap()
                    .metadata()
                    .color_encoding
                    .icc_profile(),
                Some(p.as_slice())
            );
        }
    }

    #[test]
    fn rejects_malformed_missing_duplicate_and_conflicting_icc_segments() {
        let plain = fixture(3);
        let p = profile(3);
        for payload in [
            b"ICC_PROFILE\0".to_vec(),
            b"ICC_PROFILE!\x01\x01x".to_vec(),
            icc_segment(0, 1, &p),
            icc_segment(1, 0, &p),
            icc_segment(2, 1, &p),
            icc_segment(1, 2, &p),
            icc_segment(1, 1, b"broken"),
            icc_segment(1, 1, &profile(1)),
            icc_segment(1, 1, b""),
        ] {
            for strip in [false, true] {
                assert!(preflight(&with_metadata(&plain, 0xe2, &payload, false), strip).is_err());
            }
        }
        let first = with_metadata(&plain, 0xe2, &icc_segment(1, 2, &p[..66]), false);
        for second in [icc_segment(1, 2, &p[66..]), icc_segment(2, 3, &p[66..])] {
            let bytes = with_metadata(&first, 0xe2, &second, false);
            for strip in [false, true] {
                assert!(preflight(&bytes, strip).is_err());
            }
        }
        let late = with_metadata(&plain, 0xe2, &icc_segment(1, 1, &p), true);
        for strip in [false, true] {
            assert!(preflight(&late, strip).is_err());
        }
    }

    #[test]
    fn validates_entropy_stuffing_and_multiple_scans() {
        let plain = fixture(1);
        let mut bytes = plain[..plain.len() - 2].to_vec();
        bytes.extend_from_slice(&[0xff, 0x00, 0xff, 0xd0, 0x22]);
        segment(&mut bytes, 0xda, &[1, 1, 0, 0, 63, 0]);
        bytes.extend_from_slice(&[0x3f, 0xff, 0xd9]);
        assert!(preflight(&bytes, false).is_ok());
        let late = with_metadata(&bytes, 0xe2, b"ICC_PROFILE\0", true);
        assert!(preflight(&late, false).is_err());
    }

    #[test]
    fn rejects_every_truncated_prefix_and_extra_images() {
        let bytes = fixture(1);
        for end in 0..bytes.len() {
            assert!(
                preflight(&bytes[..end], true).is_err(),
                "accepted prefix {end}"
            );
        }
        let mut doubled = bytes.clone();
        doubled.extend_from_slice(&bytes);
        assert!(preflight(&doubled, true).is_err());
    }

    #[test]
    fn rejects_unsupported_depth_components_and_limits_before_decode() {
        assert!(frame(&[12, 0, 1, 0, 1, 1, 1, 0x11, 0]).is_err());
        assert!(frame(&[8, 0, 1, 0, 1, 4]).is_err());
        assert!(frame(&[8, 0, 1, 0xff, 0xff, 1, 1, 0x11, 0]).is_err());
        assert!(frame(&[8, 0, 1, 0, 1, 1, 1, 0, 0]).is_err());
    }
}
