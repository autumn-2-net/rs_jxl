// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Bounded JPEG ICC and display metadata preparation.
//!
//! Adapts rs-jxl-cli/src/input/jpeg.rs::preflight to borrowed archival records.
//! Source references: jpeg/enc_jpeg_data.cc::{SetColorEncodingFromJpegData,
//! SetBlobsFromJpegData} and extras/dec/jpg.cc::ReadICCProfile at
//! b866b3defec13494bfbaf705db93581cda118e37.
//! Unlike the frozen transcode caller's malformed-ICC fallback, this follows the
//! current CLI's strict rejection policy. No APP classification or bytes change.

#![forbid(unsafe_code)]

use super::data::{JpegData, JpegRecord};
use super::transcode_layout::{JpegColorTransform, color_transform_from_jpeg};
use crate::color::{ColorEncoding, ColorSpace};
use crate::icc::{IccColorSpace, validate_profile};
use crate::metadata::Orientation;
use crate::{Error, Result};

#[path = "exif.rs"]
mod exif;

// jpeg_data.h::kXMPTag includes the terminating NUL in its 29-byte signature.
const XMP_TAG: &[u8; 29] = b"http://ns.adobe.com/xap/1.0/\0";

fn invalid(reason: &str) -> Error {
    Error::InvalidInput(format!("invalid JPEG color metadata: {reason}"))
}

fn visit_app_payloads<'a>(
    jpeg: &'a JpegData,
    mut visit: impl FnMut(u8, &'a [u8], bool) -> Result<()>,
) -> Result<()> {
    let mut past_header = false;
    let mut marker_codes = jpeg.marker_order.iter();
    let mut apps = jpeg.app_data.iter();
    for record in &jpeg.records {
        let segment = match record {
            JpegRecord::Marker(segment) => segment,
            JpegRecord::Entropy(_) => {
                past_header = true;
                continue;
            }
            // Reader::marker alone appends marker_order. The jbrd adapter's
            // fake 0xff for fill/inter-marker bytes is not a raw JPEG marker.
            JpegRecord::Fill { .. } | JpegRecord::InterMarker { .. } | JpegRecord::Tail { .. } => {
                continue;
            }
        };
        if marker_codes.next() != Some(&segment.marker) {
            return Err(invalid("marker order disagrees with archival records"));
        }
        if matches!(segment.marker, 0xda | 0xd9) {
            past_header = true;
        }
        if !(0xe0..=0xef).contains(&segment.marker) {
            continue;
        }
        let data = &segment.data;
        if data.len() < 2 || usize::from(u16::from_be_bytes([data[0], data[1]])) != data.len() {
            return Err(invalid("APP segment length does not match its record"));
        }
        let app = apps
            .next()
            .ok_or_else(|| invalid("missing saved APP data"))?;
        if app.first() != Some(&segment.marker) || app.get(1..) != Some(data.as_slice()) {
            return Err(invalid("saved APP data disagrees with its record"));
        }
        visit(segment.marker, &data[2..], past_header)?;
    }
    if marker_codes.next().is_some() || apps.next().is_some() {
        return Err(invalid("unmatched saved markers or APP data"));
    }
    Ok(())
}

/// Extracts original ICC bytes, or explicit sRGB gray/RGB when no ICC is present.
///
/// Requires coherent `records`, `marker_order` and `app_data` as produced by the
/// JPEG reader: marker_order has real markers only, not jbrd's synthetic 0xff.
/// This is not a complete JPEG/frame validator. ICC APP2 chunks may
/// arrive out of order, but all must precede the first SOS and have unique 1-based
/// sequence numbers, a consistent count and nonempty payloads.
///
/// `max_profile_bytes` bounds the assembled profile, excluding APP framing.
/// Zero permits only untagged images. Unknown APP/EXIF/XMP records and APP type
/// annotations remain untouched; orientation is neither parsed nor transformed.
pub fn color_encoding(jpeg: &JpegData, max_profile_bytes: usize) -> Result<ColorEncoding> {
    let (color_space, expected_icc) = match jpeg.components.len() {
        1 => (ColorSpace::Gray, IccColorSpace::Gray),
        3 => (ColorSpace::Rgb, IccColorSpace::Rgb),
        _ => return Err(invalid("requires one or three JPEG components")),
    };
    // Count is one byte: bounded bookkeeping, borrowing all chunk payloads.
    let mut parts: [Option<&[u8]>; 255] = [None; 255];
    let mut part_count = 0usize;
    let mut profile_bytes = 0usize;
    visit_app_payloads(jpeg, |marker, payload, past_header| {
        if marker != 0xe2 || !payload.starts_with(b"ICC_PROFILE") {
            return Ok(());
        }
        if past_header || payload.len() <= 14 || !payload.starts_with(b"ICC_PROFILE\0") {
            return Err(invalid("malformed or late ICC APP2 segment"));
        }
        let sequence = usize::from(payload[12]);
        let count = usize::from(payload[13]);
        if count == 0 || sequence == 0 || sequence > count {
            return Err(invalid("invalid ICC split sequence/count"));
        }
        if part_count == 0 {
            part_count = count;
        }
        if part_count != count || parts[sequence - 1].is_some() {
            return Err(invalid("duplicate or inconsistent ICC split segments"));
        }
        profile_bytes = profile_bytes
            .checked_add(payload.len() - 14)
            .filter(|&n| n <= max_profile_bytes)
            .ok_or(Error::Limit("JPEG ICC profile bytes"))?;
        parts[sequence - 1] = Some(&payload[14..]);
        Ok(())
    })?;
    if part_count == 0 {
        return Ok(ColorEncoding::srgb(color_space));
    }
    let parts = &parts[..part_count];
    if parts.iter().any(Option::is_none) {
        return Err(invalid("missing ICC split APP2 segment"));
    }
    let mut profile = Vec::new();
    profile
        .try_reserve_exact(profile_bytes)
        .map_err(|_| Error::Limit("JPEG ICC profile allocation"))?;
    for part in parts {
        profile.extend_from_slice(part.expect("complete ICC parts checked above"));
    }
    // ColorEncoding also permits CMYK with a Black extra channel. This JPEG
    // route permits only RGB/gray, so enforce the stricter CLI check explicitly.
    if validate_profile(&profile)? != expected_icc {
        return Err(invalid("ICC color space does not match JPEG components"));
    }
    ColorEncoding::icc(color_space, profile)
}

/// Display metadata is independent of the unchanged JPEG coding dimensions,
/// coefficient order and archived APP bytes. `color_transform` is a frame field,
/// not an ICC color-space choice. Exif bytes here are an additional display box.
#[derive(Debug)]
pub(crate) struct PreparedMetadata {
    pub color_encoding: ColorEncoding,
    pub color_transform: JpegColorTransform,
    pub orientation: Orientation,
    /// Complete Exif box payload: four zero TIFF-offset bytes, then original TIFF.
    /// Box framing/compression and its cumulative container budget remain caller-owned.
    pub exif_box_payload: Option<Vec<u8>>,
}

/// Composer hook matching JxlEncoderAddJPEGFrame's orientation/Exif policy.
/// Valid EXIF overrides `initial_orientation` even when `keep_exif` is false.
/// Malformed TIFF leaves it unchanged; the first nonempty APP1 Exif blob wins,
/// including blobs after SOS. `max_exif_box_bytes` includes the four-byte prefix
/// and is used only if a box is requested. Source defaults are Identity/true.
///
/// ICC deliberately retains color_encoding()'s strict CLI policy rather than
/// the frozen caller's malformed-chunk fallback/CMS interpretation. No APP
/// types are changed, and keep_exif=false does not strip archival jbrd bytes.
pub(crate) fn prepare_metadata(
    jpeg: &JpegData,
    max_profile_bytes: usize,
    initial_orientation: Orientation,
    keep_exif: bool,
    max_exif_box_bytes: usize,
) -> Result<PreparedMetadata> {
    let color_encoding = color_encoding(jpeg, max_profile_bytes)?;
    let color_transform =
        color_transform_from_jpeg(jpeg).map_err(|error| Error::InvalidInput(error.to_string()))?;
    // color_encoding validated every APP envelope and its archival mirror.
    // SetBlobsFromJpegData selects first NONEMPTY, not first valid, TIFF data.
    let tiff = jpeg.app_data.iter().find_map(|app| {
        if app.first() != Some(&0xe1) {
            return None;
        }
        let tiff = app.get(3..)?.strip_prefix(b"Exif\0\0")?;
        (!tiff.is_empty()).then_some(tiff)
    });
    let orientation = tiff.map_or(initial_orientation, |tiff| {
        exif::interpret(tiff, initial_orientation)
    });
    let exif_box_payload = if let Some(tiff) = tiff.filter(|_| keep_exif) {
        // A valid JPEG APP length already bounds TIFF to 65527 bytes, below
        // JxlEncoderAddJPEGFrame's additional 65535-byte Exif guard.
        let length = tiff
            .len()
            .checked_add(4)
            .filter(|&n| n <= max_exif_box_bytes)
            .ok_or(Error::Limit("JPEG Exif box payload bytes"))?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(length)
            .map_err(|_| Error::Limit("JPEG Exif box payload allocation"))?;
        payload.extend_from_slice(&[0; 4]);
        payload.extend_from_slice(tiff);
        Some(payload)
    } else {
        None
    };
    Ok(PreparedMetadata {
        color_encoding,
        color_transform,
        orientation,
        exif_box_payload,
    })
}

/// Returns the optional `xml ` box payload from SetBlobsFromJpegData's first
/// nonempty exact APP1 XMP signature, including markers after SOS. XML bytes
/// remain opaque: no decoding, validation, rewriting or extended-XMP assembly.
///
/// `keep_xmp` defaults to true in the source; false emits no display box without
/// changing archived APP bytes. `max_xmp_box_bytes` bounds the selected payload
/// before allocation, excluding the JPEG signature and container framing.
/// APP envelopes/order/mirrors are checked even when retention is disabled.
/// ICC and EXIF interpretation remain the separate prepare_metadata() hook.
/// The frozen JPEG route has no JUMBF extraction; APP11 remains Unknown.
pub(crate) fn xmp_box_payload(
    jpeg: &JpegData,
    keep_xmp: bool,
    max_xmp_box_bytes: usize,
) -> Result<Option<Vec<u8>>> {
    let mut selected = None;
    visit_app_payloads(jpeg, |marker, payload, _| {
        if marker == 0xe1 && selected.is_none() {
            selected = payload
                .strip_prefix(XMP_TAG.as_slice())
                .filter(|xmp| !xmp.is_empty());
        }
        Ok(())
    })?;
    let Some(xmp) = selected.filter(|_| keep_xmp) else {
        return Ok(None);
    };
    if xmp.len() > max_xmp_box_bytes {
        return Err(Error::Limit("JPEG XMP box payload bytes"));
    }
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(xmp.len())
        .map_err(|_| Error::Limit("JPEG XMP box payload allocation"))?;
    payload.extend_from_slice(xmp);
    Ok(Some(payload))
}

#[cfg(test)]
mod tests {
    use super::super::data::{AppMarkerType, Component, EntropyData, MarkerSegment};
    use super::super::reader::read_jpeg;
    use super::*;

    fn fixture(components: usize) -> JpegData {
        let mut jpeg = JpegData {
            components: vec![Component::default(); components],
            ..JpegData::default()
        };
        marker(&mut jpeg, 0xd8, &[]);
        jpeg
    }

    fn marker(jpeg: &mut JpegData, code: u8, payload: &[u8]) {
        let data = if matches!(code, 0xd8 | 0xd9) {
            Vec::new()
        } else {
            let mut data = u16::try_from(payload.len() + 2)
                .unwrap()
                .to_be_bytes()
                .to_vec();
            data.extend_from_slice(payload);
            data
        };
        if (0xe0..=0xef).contains(&code) {
            let mut app = vec![code];
            app.extend_from_slice(&data);
            jpeg.app_data.push(app);
            jpeg.app_marker_type.push(AppMarkerType::Unknown);
        }
        jpeg.marker_order.push(code);
        jpeg.records.push(JpegRecord::Marker(MarkerSegment {
            offset: 0,
            marker: code,
            data,
        }));
    }

    // Structural ICC fixture, not a CMS/conformance-tested display profile.
    fn profile(space: &[u8; 4], length: usize) -> Vec<u8> {
        let mut result = vec![0; length];
        result[..4].copy_from_slice(&(length as u32).to_be_bytes());
        result[16..20].copy_from_slice(space);
        result[20..24].copy_from_slice(b"XYZ ");
        result[36..40].copy_from_slice(b"acsp");
        result
    }

    fn icc(sequence: u8, count: u8, part: &[u8]) -> Vec<u8> {
        let mut payload = b"ICC_PROFILE\0".to_vec();
        payload.extend_from_slice(&[sequence, count]);
        payload.extend_from_slice(part);
        payload
    }

    fn exif_payload(value: u16, big: bool) -> Vec<u8> {
        let mut result = b"Exif\0\0".to_vec();
        result.extend_from_slice(if big {
            b"MM\0\x2a\0\0\0\x08"
        } else {
            b"II\x2a\0\x08\0\0\0"
        });
        result.extend_from_slice(if big {
            b"\0\x01\x01\x12\0\x03\0\0\0\x01"
        } else {
            b"\x01\0\x12\x01\x03\0\x01\0\0\0"
        });
        result.extend_from_slice(&if big {
            value.to_be_bytes()
        } else {
            value.to_le_bytes()
        });
        result.extend_from_slice(&[0; 6]);
        result
    }

    fn xmp(payload: &[u8]) -> Vec<u8> {
        let mut result = XMP_TAG.to_vec();
        result.extend_from_slice(payload);
        result
    }

    #[test]
    fn absent_profile_defaults_to_matching_srgb_and_rejects_other_component_counts() {
        for (components, space) in [(1, ColorSpace::Gray), (3, ColorSpace::Rgb)] {
            assert_eq!(
                color_encoding(&fixture(components), 0).unwrap(),
                ColorEncoding::srgb(space)
            );
        }
        for components in [0, 2, 4] {
            assert!(color_encoding(&fixture(components), usize::MAX).is_err());
        }
    }

    #[test]
    fn every_three_part_permutation_assembles_exact_gray_and_rgb_profiles() {
        for (components, space, signature) in [
            (1, ColorSpace::Gray, b"GRAY"),
            (3, ColorSpace::Rgb, b"RGB "),
        ] {
            let p = profile(signature, 132);
            let parts = [&p[..17], &p[17..91], &p[91..]];
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let mut jpeg = fixture(components);
                for i in order {
                    marker(&mut jpeg, 0xe2, &icc(i as u8 + 1, 3, parts[i]));
                    marker(&mut jpeg, 0xe3, b"unrelated");
                }
                let encoding = color_encoding(&jpeg, p.len()).unwrap();
                assert_eq!(encoding.color_space(), space);
                assert_eq!(encoding.icc_profile(), Some(p.as_slice()));
            }
        }
    }

    #[test]
    fn malformed_prefix_numbering_empty_parts_duplicates_and_missing_parts_fail() {
        let p = profile(b"RGB ", 132);
        for payload in [
            b"ICC_PROFILE".to_vec(),
            b"ICC_PROFILE\0".to_vec(),
            b"ICC_PROFILE!\x01\x01x".to_vec(),
            icc(0, 1, &p),
            icc(1, 0, &p),
            icc(2, 1, &p),
            icc(1, 1, &[]),
            icc(1, 2, &p),
            icc(2, 2, &p),
        ] {
            let mut jpeg = fixture(3);
            marker(&mut jpeg, 0xe2, &payload);
            assert!(color_encoding(&jpeg, usize::MAX).is_err());
        }
        for second in [icc(1, 2, &p[66..]), icc(2, 3, &p[66..])] {
            let mut jpeg = fixture(3);
            marker(&mut jpeg, 0xe2, &icc(1, 2, &p[..66]));
            marker(&mut jpeg, 0xe2, &second);
            assert!(color_encoding(&jpeg, usize::MAX).is_err());
        }
        let mut jpeg = fixture(3);
        marker(&mut jpeg, 0xe2, &icc(1, 1, &p));
        marker(&mut jpeg, 0xe2, &icc(1, 1, &p));
        assert!(color_encoding(&jpeg, usize::MAX).is_err());
    }

    #[test]
    fn strict_gray_rgb_matching_rejects_cmyk_and_structurally_invalid_profiles() {
        for (components, signature) in [(1, b"RGB "), (3, b"GRAY"), (1, b"CMYK"), (3, b"CMYK")] {
            let mut jpeg = fixture(components);
            marker(&mut jpeg, 0xe2, &icc(1, 1, &profile(signature, 132)));
            assert!(color_encoding(&jpeg, usize::MAX).is_err());
        }
        let good = profile(b"RGB ", 132);
        for (offset, bytes) in [
            (0, *b"\0\0\0\0"),
            (36, *b"bad!"),
            (20, *b"bad!"),
            (128, 1u32.to_be_bytes()),
        ] {
            let mut p = good.clone();
            p[offset..offset + 4].copy_from_slice(&bytes);
            let mut jpeg = fixture(3);
            marker(&mut jpeg, 0xe2, &icc(1, 1, &p));
            assert!(color_encoding(&jpeg, usize::MAX).is_err());
        }
    }

    #[test]
    fn total_profile_budget_is_checked_across_parts_before_assembly() {
        let p = profile(b"RGB ", 132);
        let mut jpeg = fixture(3);
        marker(&mut jpeg, 0xe2, &icc(2, 2, &p[66..]));
        marker(&mut jpeg, 0xe2, &icc(1, 2, &p[..66]));
        for limit in [0, 65, 66, 131] {
            assert!(matches!(color_encoding(&jpeg, limit), Err(Error::Limit(_))));
        }
        for limit in [132, usize::MAX] {
            assert_eq!(
                color_encoding(&jpeg, limit).unwrap().icc_profile(),
                Some(p.as_slice())
            );
        }
    }

    #[test]
    fn late_icc_is_rejected_while_late_unknown_metadata_remains_opaque() {
        let p = profile(b"RGB ", 132);
        for terminator in [0xda, 0xd9] {
            let mut jpeg = fixture(3);
            marker(&mut jpeg, terminator, &[]);
            marker(&mut jpeg, 0xe2, b"MPF\0opaque");
            assert!(color_encoding(&jpeg, 0).is_ok());
            marker(&mut jpeg, 0xe2, &icc(1, 1, &p));
            assert!(color_encoding(&jpeg, usize::MAX).is_err());
        }
    }

    #[test]
    fn archival_marker_order_app_mirrors_and_segment_lengths_are_checked() {
        let mut original = fixture(3);
        marker(&mut original, 0xe2, &icc(1, 1, &profile(b"RGB ", 132)));
        let mut changed = original.clone();
        changed.marker_order[1] = 0xe1;
        assert!(color_encoding(&changed, usize::MAX).is_err());
        changed = original.clone();
        changed.marker_order.push(0xe0);
        assert!(color_encoding(&changed, usize::MAX).is_err());
        changed = original.clone();
        changed.app_data[0][3] ^= 1;
        assert!(color_encoding(&changed, usize::MAX).is_err());
        changed = original.clone();
        changed.app_data.clear();
        assert!(color_encoding(&changed, usize::MAX).is_err());
        changed = original.clone();
        changed.records.clear();
        assert!(color_encoding(&changed, usize::MAX).is_err());
        for data in [vec![], vec![0], vec![0, 1], vec![0, 3], vec![0, 2, 0]] {
            changed = original.clone();
            let JpegRecord::Marker(segment) = &mut changed.records[1] else {
                panic!()
            };
            segment.data = data;
            assert!(color_encoding(&changed, usize::MAX).is_err());
        }
    }

    #[test]
    fn maximum_chunk_count_in_reverse_order_has_bounded_bookkeeping() {
        let mut p = profile(b"RGB ", 260);
        for (i, byte) in p[132..].iter_mut().enumerate() {
            *byte = i as u8;
        }
        let mut jpeg = fixture(3);
        for sequence in (1u8..=255).rev() {
            let start = usize::from(sequence) - 1;
            let end = if sequence == 255 { p.len() } else { start + 1 };
            marker(&mut jpeg, 0xe2, &icc(sequence, 255, &p[start..end]));
        }
        assert_eq!(
            color_encoding(&jpeg, p.len()).unwrap().icc_profile(),
            Some(p.as_slice())
        );
    }

    #[test]
    fn extraction_does_not_change_archival_bytes_or_unknown_app_types() {
        let mut jpeg = fixture(3);
        marker(&mut jpeg, 0xe1, b"Exif\0\0not a TIFF parser input");
        marker(&mut jpeg, 0xe2, b"MPF\0opaque");
        marker(&mut jpeg, 0xe3, b"ICC_PROFILE\0not an APP2 profile");
        let p = profile(b"RGB ", 132);
        marker(&mut jpeg, 0xe2, &icc(1, 1, &p));
        marker(&mut jpeg, 0xd9, &[]);
        let records = jpeg.records.clone();
        let apps = jpeg.app_data.clone();
        let marker_order = jpeg.marker_order.clone();
        let bytes = jpeg.reconstruct().unwrap();
        assert_eq!(
            color_encoding(&jpeg, p.len()).unwrap().icc_profile(),
            Some(p.as_slice())
        );
        assert_eq!(jpeg.records, records);
        assert_eq!(jpeg.app_data, apps);
        assert_eq!(jpeg.marker_order, marker_order);
        assert_eq!(jpeg.app_marker_type, vec![AppMarkerType::Unknown; 4]);
        assert_eq!(jpeg.reconstruct().unwrap(), bytes);
    }

    #[test]
    fn reader_fill_intermarker_unknown_apps_and_icc_looking_tail_preserve_marker_alignment() {
        let p = profile(b"GRAY", 132);
        let mut source = fixture(1);
        source.inter_marker_data.push(vec![0x21, 0xff, 0, 0x22]);
        source.records.push(JpegRecord::InterMarker {
            offset: 0,
            index: 0,
        });
        source.records.push(JpegRecord::Fill {
            offset: 0,
            count: 2,
        });
        marker(&mut source, 0xe2, &icc(2, 2, &p[66..]));
        marker(&mut source, 0xef, &icc(1, 1, b"not an APP2 profile"));
        marker(&mut source, 0xe2, b"MPF\0\xff\xda\xff\xd9");
        marker(&mut source, 0xfe, b"ICC_PROFILE\0opaque comment");
        source.inter_marker_data.push(vec![0x20, 0xff, 0, 0x30]);
        source.records.push(JpegRecord::InterMarker {
            offset: 0,
            index: 1,
        });
        source.records.push(JpegRecord::Fill {
            offset: 0,
            count: 1,
        });
        marker(&mut source, 0xe2, &icc(1, 2, &p[..66]));

        // One baseline gray block, with category-zero DC and EOB AC codes.
        let mut quant = vec![1; 65];
        quant[0] = 0;
        marker(&mut source, 0xdb, &quant);
        let mut huffman = Vec::new();
        for selector in [0, 0x10] {
            huffman.extend_from_slice(&[selector, 1]);
            huffman.extend_from_slice(&[0; 16]);
        }
        marker(&mut source, 0xc4, &huffman);
        marker(&mut source, 0xc0, &[8, 0, 1, 0, 1, 1, 1, 0x11, 0]);
        marker(&mut source, 0xda, &[1, 1, 0, 0, 63, 0]);
        source.records.push(JpegRecord::Entropy(EntropyData {
            offset: 0,
            scan_index: 0,
            bytes: vec![0x3f],
            restart_markers: Vec::new(),
        }));
        marker(&mut source, 0xe2, b"MPF\0late unknown APP2");
        marker(&mut source, 0xd9, &[]);
        let tail_payload = icc(1, 1, b"not an ICC profile");
        source.tail_data = vec![0xff, 0xe2];
        source
            .tail_data
            .extend_from_slice(&u16::try_from(tail_payload.len() + 2).unwrap().to_be_bytes());
        source.tail_data.extend_from_slice(&tail_payload);
        source.tail_data.extend_from_slice(&[0xff, 0xda]);
        source.records.push(JpegRecord::Tail { offset: 0 });

        let bytes = source.reconstruct().unwrap();
        let jpeg = read_jpeg(&bytes).unwrap();
        assert_eq!(
            jpeg.marker_order,
            [
                0xd8, 0xe2, 0xef, 0xe2, 0xfe, 0xe2, 0xdb, 0xc4, 0xc0, 0xda, 0xe2, 0xd9
            ]
        );
        assert!(!jpeg.marker_order.contains(&0xff));
        let fills: Vec<_> = jpeg
            .records
            .iter()
            .filter_map(|record| match record {
                JpegRecord::Fill { count, .. } => Some(*count),
                _ => None,
            })
            .collect();
        assert_eq!(fills, [2, 1]);
        assert_eq!(jpeg.inter_marker_data, source.inter_marker_data);
        assert_eq!(jpeg.tail_data, source.tail_data);
        assert!(matches!(jpeg.records.last(), Some(JpegRecord::Tail { .. })));
        let records = jpeg.records.clone();
        assert_eq!(
            color_encoding(&jpeg, p.len()).unwrap().icc_profile(),
            Some(p.as_slice())
        );
        assert_eq!(jpeg.app_marker_type, vec![AppMarkerType::Unknown; 5]);
        assert_eq!(jpeg.records, records);
        assert_eq!(jpeg.reconstruct().unwrap(), bytes);

        // WireModel::intermarker inserts these only into its separate wire map.
        let mut wire_order_in_raw_data = jpeg.clone();
        wire_order_in_raw_data.marker_order.insert(1, 0xff);
        assert!(color_encoding(&wire_order_in_raw_data, p.len()).is_err());
    }

    #[test]
    fn prepared_orientation_changes_display_metadata_not_jpeg_coding_or_archival_bytes() {
        for big in [false, true] {
            let mut jpeg = fixture(3);
            jpeg.width = 13;
            jpeg.height = 7;
            for (component, id) in jpeg.components.iter_mut().zip(b"RGB") {
                component.id = u32::from(*id);
                component.coefficients = vec![i16::MIN, -1, 0, 1, i16::MAX];
            }
            let p = profile(b"RGB ", 132);
            marker(&mut jpeg, 0xe2, &icc(1, 1, &p));
            let exif = exif_payload(6, big);
            marker(&mut jpeg, 0xe1, &exif);
            marker(&mut jpeg, 0xe1, &exif_payload(8, !big));
            marker(&mut jpeg, 0xd9, &[]);
            let original = jpeg.clone();
            let bytes = jpeg.reconstruct().unwrap();
            let prepared =
                prepare_metadata(&jpeg, p.len(), Orientation::Identity, true, 30).unwrap();
            assert_eq!(prepared.orientation, Orientation::Rotate90Clockwise);
            assert_eq!(prepared.color_transform, JpegColorTransform::None);
            assert_eq!(prepared.color_encoding.icc_profile(), Some(p.as_slice()));
            let mut expected_box = vec![0; 4];
            expected_box.extend_from_slice(&exif[6..]);
            assert_eq!(prepared.exif_box_payload, Some(expected_box));
            assert_eq!((jpeg.width, jpeg.height), (13, 7));
            assert_eq!(jpeg.components, original.components);
            assert_eq!(jpeg.records, original.records);
            assert_eq!(jpeg.app_data, original.app_data);
            assert_eq!(jpeg.app_marker_type, vec![AppMarkerType::Unknown; 3]);
            assert_eq!(jpeg.reconstruct().unwrap(), bytes);
        }
    }

    #[test]
    fn exif_after_sos_updates_orientation_even_when_box_retention_is_disabled() {
        let mut jpeg = fixture(1);
        let initial = Orientation::FlipVertical;
        let absent = prepare_metadata(&jpeg, 0, initial, true, 0).unwrap();
        assert_eq!(absent.orientation, initial);
        assert!(absent.exif_box_payload.is_none());
        marker(&mut jpeg, 0xda, &[]);
        let exif = exif_payload(8, true);
        marker(&mut jpeg, 0xe1, &exif);
        marker(&mut jpeg, 0xd9, &[]);
        let ignored_box = prepare_metadata(&jpeg, 0, initial, false, 0).unwrap();
        assert_eq!(
            ignored_box.orientation,
            Orientation::Rotate90CounterClockwise
        );
        assert!(ignored_box.exif_box_payload.is_none());
        let retained = prepare_metadata(&jpeg, 0, initial, true, 30).unwrap();
        assert_eq!(retained.orientation, ignored_box.orientation);
        assert_eq!(&retained.exif_box_payload.unwrap()[4..], &exif[6..]);
        assert_eq!(
            retained.color_encoding,
            ColorEncoding::srgb(ColorSpace::Gray)
        );
    }

    #[test]
    fn app1_signature_and_first_nonempty_exif_policy_match_source() {
        let valid = exif_payload(6, false);
        for (code, payload) in [
            (0xe2, valid.clone()),
            (0xef, valid.clone()),
            (0xe1, b"Exif\0\0".to_vec()),
            (0xe1, b"Exif\0".to_vec()),
            (0xe1, b"EXIF\0\0not the source signature".to_vec()),
            (0xe1, b"Exif\0!wrong terminator".to_vec()),
        ] {
            let mut jpeg = fixture(3);
            marker(&mut jpeg, code, &payload);
            marker(&mut jpeg, 0xe1, &valid);
            let prepared = prepare_metadata(&jpeg, 0, Orientation::Identity, true, 30).unwrap();
            assert_eq!(prepared.orientation, Orientation::Rotate90Clockwise);
            assert_eq!(&prepared.exif_box_payload.unwrap()[4..], &valid[6..]);
        }
        // A nonempty but malformed first TIFF blocks a later valid blob.
        for payload in [b"Exif\0\0x".to_vec(), exif_payload(9, false)] {
            let mut jpeg = fixture(3);
            marker(&mut jpeg, 0xe1, &payload);
            marker(&mut jpeg, 0xe1, &valid);
            let prepared = prepare_metadata(&jpeg, 0, Orientation::Rotate180, true, 100).unwrap();
            assert_eq!(prepared.orientation, Orientation::Rotate180);
            assert_eq!(&prepared.exif_box_payload.unwrap()[4..], &payload[6..]);
        }
        let mut jpeg = fixture(3);
        marker(&mut jpeg, 0xe1, &valid);
        jpeg.app_data[0][2] -= 1;
        assert!(prepare_metadata(&jpeg, 0, Orientation::Identity, true, 100).is_err());
    }

    #[test]
    fn exif_box_budget_includes_offset_prefix_and_respects_app_length_bound() {
        let mut jpeg = fixture(1);
        let mut exif = exif_payload(2, false);
        exif.resize(65533, 0); // Largest APP payload; six signature bytes precede TIFF.
        marker(&mut jpeg, 0xe1, &exif);
        let maximum_box = 65531;
        for limit in [0, 4, maximum_box - 1] {
            assert!(matches!(
                prepare_metadata(&jpeg, 0, Orientation::Identity, true, limit),
                Err(Error::Limit(_))
            ));
        }
        let prepared =
            prepare_metadata(&jpeg, 0, Orientation::Identity, true, maximum_box).unwrap();
        assert_eq!(prepared.orientation, Orientation::FlipHorizontal);
        let payload = prepared.exif_box_payload.unwrap();
        assert_eq!(payload.len(), maximum_box);
        assert_eq!(&payload[..4], &[0; 4]);
        assert_eq!(&payload[4..], &exif[6..]);
        assert!(
            prepare_metadata(&jpeg, 0, Orientation::Identity, false, 0)
                .unwrap()
                .exif_box_payload
                .is_none()
        );
    }

    #[test]
    fn jpeg_coding_transform_is_independent_of_icc_space_and_display_orientation() {
        let mut original = fixture(3);
        for (component, &id) in original.components.iter_mut().zip(b"RGB") {
            component.id = u32::from(id);
        }
        let p = profile(b"RGB ", 132);
        marker(&mut original, 0xe2, &icc(1, 1, &p));
        marker(&mut original, 0xe1, &exif_payload(7, false));
        for (app0, first_adobe, expected) in [
            (false, None, JpegColorTransform::None),
            (true, None, JpegColorTransform::YCbCr),
            (false, Some(0), JpegColorTransform::None),
            (false, Some(1), JpegColorTransform::YCbCr),
            (true, Some(0), JpegColorTransform::YCbCr),
        ] {
            let mut jpeg = original.clone();
            if let Some(transform) = first_adobe {
                let mut adobe = b"Adobe\0d\0\0\0\0\0".to_vec();
                adobe[11] = transform;
                marker(&mut jpeg, 0xee, &adobe);
                adobe[11] = 1 - transform;
                marker(&mut jpeg, 0xee, &adobe);
            }
            if app0 {
                marker(&mut jpeg, 0xe0, b"not necessarily JFIF");
            }
            let prepared =
                prepare_metadata(&jpeg, p.len(), Orientation::Identity, false, 0).unwrap();
            assert_eq!(prepared.color_transform, expected);
            assert_eq!(prepared.color_encoding.color_space(), ColorSpace::Rgb);
            assert_eq!(prepared.color_encoding.icc_profile(), Some(p.as_slice()));
            assert_eq!(prepared.orientation, Orientation::AntiTranspose);
        }
    }

    #[test]
    fn xmp_first_nonempty_blob_wins_without_xml_or_text_normalization() {
        let mut jpeg = fixture(3);
        assert_eq!(xmp_box_payload(&jpeg, true, 0).unwrap(), None);
        marker(&mut jpeg, 0xe1, XMP_TAG);
        marker(&mut jpeg, 0xe1, XMP_TAG);
        assert_eq!(xmp_box_payload(&jpeg, true, 0).unwrap(), None);
        // A nonempty malformed XML/text blob still wins over a later valid one.
        let first = b"\xff\0\r\nnot XML\0";
        marker(&mut jpeg, 0xe1, &xmp(first));
        marker(&mut jpeg, 0xe1, &xmp(b"<x:xmpmeta/>"));
        assert_eq!(
            xmp_box_payload(&jpeg, true, first.len())
                .unwrap()
                .as_deref(),
            Some(first.as_slice())
        );
        assert_eq!(xmp_box_payload(&jpeg, false, 0).unwrap(), None);
    }

    #[test]
    fn xmp_requires_exact_null_terminated_app1_signature_and_ignores_extended_xmp() {
        assert_eq!(XMP_TAG.len(), 29);
        for length in 0..XMP_TAG.len() {
            let mut jpeg = fixture(1);
            marker(&mut jpeg, 0xe1, &XMP_TAG[..length]);
            assert_eq!(xmp_box_payload(&jpeg, true, 0).unwrap(), None);
        }
        let mut wrong_terminator = XMP_TAG.to_vec();
        wrong_terminator[28] = b'!';
        wrong_terminator.extend_from_slice(b"<ignored/>");
        for (code, payload) in [
            (0xe0, xmp(b"wrong APP0")),
            (0xe2, xmp(b"wrong APP2")),
            (0xeb, xmp(b"wrong APP11")),
            (0xef, xmp(b"wrong APP15")),
            (0xe1, wrong_terminator),
            (0xe1, b"HTTP://ns.adobe.com/xap/1.0/\0<ignored/>".to_vec()),
            (
                0xe1,
                b"http://ns.adobe.com/xmp/extension/\0unassembled extension".to_vec(),
            ),
        ] {
            let mut jpeg = fixture(3);
            marker(&mut jpeg, code, &payload);
            assert_eq!(xmp_box_payload(&jpeg, true, 0).unwrap(), None);
            marker(&mut jpeg, 0xe1, &xmp(b"<selected/>"));
            assert_eq!(
                xmp_box_payload(&jpeg, true, 11).unwrap().as_deref(),
                Some(b"<selected/>".as_slice())
            );
        }
    }

    #[test]
    fn xmp_payload_limit_excludes_signature_and_ignored_duplicates() {
        let maximum = 65533 - XMP_TAG.len();
        let content = vec![0x81; maximum];
        let mut jpeg = fixture(1);
        marker(&mut jpeg, 0xe1, &xmp(&content));
        marker(&mut jpeg, 0xe1, &xmp(&content));
        for limit in [0, maximum - 1] {
            assert!(matches!(
                xmp_box_payload(&jpeg, true, limit),
                Err(Error::Limit(_))
            ));
        }
        for limit in [maximum, usize::MAX] {
            assert_eq!(
                xmp_box_payload(&jpeg, true, limit).unwrap().as_deref(),
                Some(content.as_slice())
            );
        }
        assert_eq!(xmp_box_payload(&jpeg, false, 0).unwrap(), None);
        let mut small_first = fixture(1);
        marker(&mut small_first, 0xe1, &xmp(b"a"));
        marker(&mut small_first, 0xe1, &xmp(&content));
        assert_eq!(
            xmp_box_payload(&small_first, true, 1).unwrap().as_deref(),
            Some(b"a".as_slice())
        );
    }

    #[test]
    fn xmp_after_entropy_is_selected_but_fill_intermarker_tail_and_app11_remain_opaque() {
        let mut jpeg = fixture(3);
        marker(&mut jpeg, 0xe1, &exif_payload(6, false));
        marker(&mut jpeg, 0xeb, b"JP\0\x01\0\0\0\x01opaque JUMBF");
        marker(&mut jpeg, 0xda, &[]);
        jpeg.records.push(JpegRecord::Entropy(EntropyData {
            offset: 0,
            scan_index: 0,
            bytes: xmp(b"not an APP1 record"),
            restart_markers: Vec::new(),
        }));
        jpeg.records.push(JpegRecord::Fill {
            offset: 0,
            count: 3,
        });
        jpeg.inter_marker_data.push(xmp(b"inter-marker decoy"));
        jpeg.records.push(JpegRecord::InterMarker {
            offset: 0,
            index: 0,
        });
        marker(&mut jpeg, 0xd9, &[]);
        jpeg.tail_data = xmp(b"tail decoy");
        jpeg.records.push(JpegRecord::Tail { offset: 0 });
        assert_eq!(xmp_box_payload(&jpeg, true, 0).unwrap(), None);

        // Insert an actual late APP1 before EOI, without adding fake FF entries
        // for Fill/InterMarker records to raw marker_order.
        let tail = jpeg.records.pop().unwrap();
        let eoi = jpeg.records.pop().unwrap();
        assert_eq!(jpeg.marker_order.pop(), Some(0xd9));
        marker(&mut jpeg, 0xe1, &xmp(b"<late/>"));
        jpeg.marker_order.push(0xd9);
        jpeg.records.extend([eoi, tail]);
        let original = jpeg.clone();
        let original_bytes = jpeg.reconstruct().unwrap();
        assert_eq!(
            xmp_box_payload(&jpeg, true, 7).unwrap().as_deref(),
            Some(b"<late/>".as_slice())
        );
        assert_eq!(xmp_box_payload(&jpeg, false, 0).unwrap(), None);
        assert_eq!(
            prepare_metadata(&jpeg, 0, Orientation::Identity, false, 0)
                .unwrap()
                .orientation,
            Orientation::Rotate90Clockwise
        );
        assert_eq!(jpeg.components, original.components);
        assert_eq!(jpeg.records, original.records);
        assert_eq!(jpeg.app_data, original.app_data);
        assert_eq!(jpeg.app_marker_type, vec![AppMarkerType::Unknown; 3]);
        assert_eq!(jpeg.marker_order, original.marker_order);
        assert_eq!(jpeg.reconstruct().unwrap(), original_bytes);
    }

    #[test]
    fn xmp_checks_all_app_mirrors_and_markers_even_after_selection_or_when_disabled() {
        let mut original = fixture(1);
        marker(&mut original, 0xe1, &xmp(b"<first/>"));
        marker(&mut original, 0xeb, b"opaque APP11");
        let mut cases = Vec::new();
        let mut changed = original.clone();
        changed.marker_order.insert(1, 0xff);
        cases.push(changed);
        let mut changed = original.clone();
        changed.app_data[1][3] ^= 1;
        cases.push(changed);
        let mut changed = original.clone();
        changed.app_data.pop();
        cases.push(changed);
        let mut changed = original.clone();
        changed.records.clear();
        cases.push(changed);
        let mut changed = original.clone();
        changed.marker_order.push(0xe1);
        cases.push(changed);
        for data in [vec![], vec![0], vec![0, 1], vec![0, 3], vec![0, 2, 0]] {
            let mut changed = original.clone();
            let JpegRecord::Marker(segment) = &mut changed.records[2] else {
                panic!()
            };
            segment.data = data;
            cases.push(changed);
        }
        for changed in cases {
            for keep in [false, true] {
                assert!(matches!(
                    xmp_box_payload(&changed, keep, usize::MAX),
                    Err(Error::InvalidInput(_))
                ));
            }
        }
    }
}
