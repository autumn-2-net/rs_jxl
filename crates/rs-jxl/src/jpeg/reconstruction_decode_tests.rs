//! Parser/adapter checks and separate full-container original-byte round trips.

use super::*;
use crate::jpeg::coefficient_reader::read_coefficients;
use crate::jpeg::reconstruction_brotli::compress_parts;
use crate::jpeg::reconstruction_metadata::{
    EncodedMetadata, serialize_reconstruction_metadata,
    serialize_reconstruction_metadata_with_sources,
};

use crate::jpeg::transcode_reference_tests::fixtures;

fn marker(out: &mut Vec<u8>, code: u8, payload: &[u8]) {
    out.extend([255, code]);
    out.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    out.extend_from_slice(payload);
}

fn minimal(padding: u8, restart: bool) -> Vec<u8> {
    let mut out = vec![255, 0xd8];
    let mut quant = vec![3];
    quant.extend([1; 64]);
    marker(&mut out, 0xdb, &quant);
    marker(
        &mut out,
        0xc0,
        &[8, 0, 8, 0, if restart { 16 } else { 8 }, 1, 1, 0x11, 3],
    );
    let mut huffman = Vec::new();
    for selector in [0, 16] {
        huffman.extend([selector, 1]);
        huffman.extend([0; 15]);
        huffman.push(0);
    }
    marker(&mut out, 0xc4, &huffman);
    if restart {
        marker(&mut out, 0xdd, &[0, 1]);
    }
    marker(&mut out, 0xda, &[1, 1, 0, 0, 63, 0]);
    out.push(padding & 63);
    if restart {
        out.extend([255, 0xd0, 0x3f]);
    }
    out.extend([255, 0xd9]);
    out
}

fn captured(jpeg: &JpegData) -> CapturedJpegCoefficients {
    CapturedJpegCoefficients {
        width: jpeg.width,
        height: jpeg.height,
        components: jpeg
            .components
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let scan = jpeg
                    .scan_info
                    .iter()
                    .find(|s| s.components.iter().any(|sc| sc.component_index == i))
                    .unwrap();
                let q = scan.quant_table_indices[c.quant_index as usize].unwrap();
                CapturedJpegComponent {
                    h_samp_factor: c.h_samp_factor,
                    v_samp_factor: c.v_samp_factor,
                    width_in_blocks: c.width_in_blocks,
                    height_in_blocks: c.height_in_blocks,
                    coefficients: c.coefficients.clone(),
                    quantization: jpeg.quant[q].values,
                }
            })
            .collect(),
    }
}

fn compressed(parts: &EncodedMetadata) -> Vec<u8> {
    compress_parts(&parts.header, &parts.payload, 3, 1 << 24).unwrap()
}

fn metadata(jpeg: &JpegData) -> Vec<u8> {
    compressed(&serialize_reconstruction_metadata(jpeg, 1 << 24).unwrap())
}

fn pack(fields: &[(u32, u8)]) -> (Vec<u8>, usize) {
    let count: usize = fields.iter().map(|&(_, bits)| usize::from(bits)).sum();
    let mut bytes = vec![0; count.div_ceil(8)];
    let mut pos = 0;
    for &(value, bits) in fields {
        for i in 0..bits {
            bytes[pos / 8] |= (((value >> i) & 1) as u8) << (pos % 8);
            pos += 1;
        }
    }
    (bytes, count)
}

// Explicit independent VisitFields packing for one 8x8 gray block and DQT slot
// three. Huffman trees each have one real code and the synthetic sentinel.
fn minimal_fields() -> Vec<(u32, u8)> {
    let mut fields = vec![
        (1, 1),
        (27, 6),
        (0, 6),
        (4, 6),
        (26, 6),
        (25, 6),
        (0, 2),
        (0, 1),
        (3, 2),
        (1, 1),
        (0, 2),
        (0, 2),
        (1, 2),
        (0, 3),
    ];
    for (class, last) in [(0, 0), (1, 1)] {
        fields.extend([(class, 1), (0, 2), (last, 1), (0, 2), (2, 2), (0, 3)]);
        fields.extend(std::iter::repeat_n((0, 2), 15));
        fields.extend([(0, 2), (0, 2), (3, 2), (255, 8)]);
    }
    fields.extend([
        (0, 2),
        (0, 6),
        (63, 6),
        (0, 4),
        (0, 4),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 2),
        (0, 1),
    ]);
    fields
}

#[test]
fn independent_wire_fixture_builds_a_model_without_archived_entropy() {
    let (header, _) = pack(&minimal_fields());
    let encoded = compress_parts(&header, &[], 3, 4096).unwrap();
    let limits = ReconstructionDecodeLimits::default();
    let parsed = decode_reconstruction_metadata(&encoded, &limits).unwrap();
    assert_eq!(parsed.component_count(), 1);
    assert_eq!(parsed.scan_count(), 1);
    assert!(!parsed.needs_icc() && !parsed.needs_exif() && !parsed.needs_xmp());
    let frame = CapturedJpegCoefficients {
        width: 8,
        height: 8,
        components: vec![CapturedJpegComponent {
            h_samp_factor: 1,
            v_samp_factor: 1,
            width_in_blocks: 1,
            height_in_blocks: 1,
            coefficients: vec![0; 64],
            quantization: [1; 64],
        }],
    };
    let model = parsed
        .into_jpeg(frame, ReconstructionMetadataSources::default(), &limits)
        .unwrap();
    assert_eq!(model.components[0].quant_index, 3);
    assert!(model.padding_bits.is_empty());
    assert!(model.entropy_data(0).unwrap().bytes.is_empty());
    assert_eq!(
        reconstruct_from_coefficients(&model, &limits.write).unwrap(),
        minimal(63, false)
    );
}

#[test]
fn parser_rejects_every_truncation_trailing_bytes_and_nonzero_alignment() {
    let (header, bits) = pack(&minimal_fields());
    let encoded = compress_parts(&header, &[], 3, 4096).unwrap();
    let limits = ReconstructionDecodeLimits::default();
    for length in 0..encoded.len() {
        assert!(
            decode_reconstruction_metadata(&encoded[..length], &limits).is_err(),
            "prefix {length}"
        );
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(decode_reconstruction_metadata(&trailing, &limits).is_err());
    let mut concatenated = encoded.clone();
    concatenated.extend_from_slice(&encoded[header.len()..]);
    assert!(decode_reconstruction_metadata(&concatenated, &limits).is_err());
    assert_ne!(bits % 8, 0);
    let mut unaligned = header;
    unaligned[bits / 8] |= 1 << (bits % 8);
    assert!(
        decode_reconstruction_metadata(&compress_parts(&unaligned, &[], 3, 4096).unwrap(), &limits)
            .is_err()
    );
}

#[test]
fn parser_rejects_invalid_wire_components_markers_tables_and_sentinels() {
    let limits = ReconstructionDecodeLimits::default();
    for (index, value) in [(0, 0), (2, 3), (6, 3), (11, 3), (18, 0)] {
        let mut fields = minimal_fields();
        fields[index].0 = value;
        let (header, _) = pack(&fields);
        assert!(
            decode_reconstruction_metadata(
                &compress_parts(&header, &[], 3, 4096).unwrap(),
                &limits
            )
            .is_err(),
            "field {index}"
        );
    }
    let mut fields = minimal_fields();
    let sentinel = fields.iter().position(|&v| v == (255, 8)).unwrap();
    fields[sentinel].0 = 254;
    let (header, _) = pack(&fields);
    assert!(
        decode_reconstruction_metadata(&compress_parts(&header, &[], 3, 4096).unwrap(), &limits)
            .is_err()
    );
    let (header, _) = pack(&minimal_fields());
    let bomb = compress_parts(&header, &[0; 8192], 3, 4096).unwrap();
    assert!(matches!(
        decode_reconstruction_metadata(&bomb, &limits),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn parser_and_adapter_enforce_exact_limits_and_coefficient_geometry() {
    let jpeg = read_coefficients(&minimal(42, true)).unwrap();
    let parts = serialize_reconstruction_metadata(&jpeg, 4096).unwrap();
    let encoded = compressed(&parts);
    let limits = ReconstructionDecodeLimits {
        max_jbrd_bytes: encoded.len(),
        max_metadata_bytes: parts.header.len() + parts.payload.len(),
        max_coefficient_values: 128,
        ..Default::default()
    };
    reconstruct_jpeg(
        &encoded,
        captured(&jpeg),
        ReconstructionMetadataSources::default(),
        &limits,
    )
    .unwrap();
    for case in 0..8 {
        let mut bad = limits.clone();
        match case {
            0 => bad.max_jbrd_bytes -= 1,
            1 => bad.max_metadata_bytes -= 1,
            2 => bad.write.max_hint_entries = 0,
            3 => bad.write.max_padding_bits = 0,
            4 => bad.write.structural.max_records = 0,
            5 => bad.write.structural.max_scans = 0,
            6 => bad.write.structural.max_quant_tables = 0,
            _ => bad.write.structural.max_huffman_tables = 0,
        }
        // This sequential fixture need not carry reconstruction hints.
        if case != 2 {
            assert!(
                decode_reconstruction_metadata(&encoded, &bad).is_err(),
                "limit {case}"
            );
        }
    }
    for case in 0..7 {
        let mut frame = captured(&jpeg);
        let mut bad = limits.clone();
        match case {
            0 => frame.width = 0,
            1 => frame.components[0].h_samp_factor = 0,
            2 => frame.components[0].width_in_blocks += 1,
            3 => {
                frame.components[0].coefficients.pop();
            }
            4 => frame.components[0].quantization[0] = 0,
            5 => bad.max_coefficient_values -= 1,
            _ => bad.write.structural.max_restart_markers = 0,
        }
        assert!(
            reconstruct_jpeg(
                &encoded,
                frame,
                ReconstructionMetadataSources::default(),
                &bad
            )
            .is_err(),
            "frame {case}"
        );
    }
}

#[test]
fn empty_dht_quant_slot_mapping_padding_restart_and_tail_survive_adapter() {
    for padding in [0, 42, 63] {
        let original = minimal(padding, true);
        let mut with_metadata = original[..2].to_vec();
        marker(&mut with_metadata, 0xc4, &[]);
        with_metadata.extend_from_slice(&original[2..original.len() - 2]);
        with_metadata.extend([255, 255, 0xd9, 0, 255, 0xda]);
        let jpeg = read_coefficients(&with_metadata).unwrap();
        let limits = ReconstructionDecodeLimits::default();
        let model = decode_reconstruction_metadata(&metadata(&jpeg), &limits)
            .unwrap()
            .into_jpeg(
                captured(&jpeg),
                ReconstructionMetadataSources::default(),
                &limits,
            )
            .unwrap();
        assert!(
            model
                .records
                .iter()
                .all(|r| !matches!(r, JpegRecord::Entropy(e) if !e.bytes.is_empty()))
        );
        assert_eq!(model.entropy_data(0).unwrap().restart_markers.len(), 1);
        assert_eq!(
            reconstruct_from_coefficients(&model, &limits.write).unwrap(),
            with_metadata
        );
    }
}

#[test]
fn deduplicated_app_sources_are_exact_required_and_length_checked() {
    let base = minimal(42, false);
    let profile = b"arbitrary exact original ICC bytes";
    let tiff = b"II\x2a\0\x08\0\0\0\0\0\0\0\0\0";
    let xml = b"<x:xmpmeta>original</x:xmpmeta>";
    let mut exif_box = vec![0; 4];
    exif_box.extend_from_slice(tiff);
    let mut original = base[..2].to_vec();
    for (ordinal, bytes) in [(1, &profile[..5]), (2, &profile[5..])] {
        let mut app = b"ICC_PROFILE\0".to_vec();
        app.extend([ordinal, 2]);
        app.extend_from_slice(bytes);
        marker(&mut original, 0xe2, &app);
    }
    let mut exif = b"Exif\0\0".to_vec();
    exif.extend_from_slice(tiff);
    marker(&mut original, 0xe1, &exif);
    let mut xmp = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
    xmp.extend_from_slice(xml);
    marker(&mut original, 0xe1, &xmp);
    marker(&mut original, 0xfe, b"comment\0\xff");
    original.extend([19, 255, 0, 255, 255]);
    original.extend_from_slice(&base[2..]);
    original.extend([0, 255, 0xd9]);
    let jpeg = read_coefficients(&original).unwrap();
    let sources = ReconstructionMetadataSources {
        icc_profile: Some(profile),
        exif_box_payload: Some(&exif_box),
        xmp_box_payload: Some(xml),
    };
    let parts = serialize_reconstruction_metadata_with_sources(&jpeg, 65536, sources).unwrap();
    let encoded = compressed(&parts);
    let limits = ReconstructionDecodeLimits::default();
    let parsed = decode_reconstruction_metadata(&encoded, &limits).unwrap();
    assert!(parsed.needs_icc() && parsed.needs_exif() && parsed.needs_xmp());
    assert_eq!(
        reconstruct_jpeg(&encoded, captured(&jpeg), sources, &limits).unwrap(),
        original
    );
    // Deduplication changes storage, not the expanded metadata budget. Count
    // each reconstructed APP envelope as well as its externally stored bytes.
    let mut bounded = limits.clone();
    bounded.max_jbrd_bytes = encoded.len();
    bounded.max_metadata_bytes = parts.header.len()
        + parts.payload.len()
        + profile.len()
        + 2 * 17
        + tiff.len()
        + 9
        + xml.len()
        + 32;
    assert_eq!(
        reconstruct_jpeg(&encoded, captured(&jpeg), sources, &bounded).unwrap(),
        original
    );
    bounded.max_metadata_bytes -= 1;
    assert!(matches!(
        reconstruct_jpeg(&encoded, captured(&jpeg), sources, &bounded),
        Err(Error::Limit("expanded metadata"))
    ));
    for missing in [
        ReconstructionMetadataSources {
            icc_profile: None,
            ..sources
        },
        ReconstructionMetadataSources {
            exif_box_payload: None,
            ..sources
        },
        ReconstructionMetadataSources {
            xmp_box_payload: None,
            ..sources
        },
    ] {
        assert!(matches!(
            reconstruct_jpeg(&encoded, captured(&jpeg), missing, &limits),
            Err(Error::MissingSource(_))
        ));
    }
    for bad in [
        ReconstructionMetadataSources {
            icc_profile: Some(&profile[..profile.len() - 1]),
            ..sources
        },
        ReconstructionMetadataSources {
            exif_box_payload: Some(&exif_box[..3]),
            ..sources
        },
        ReconstructionMetadataSources {
            xmp_box_payload: Some(&[]),
            ..sources
        },
    ] {
        assert!(reconstruct_jpeg(&encoded, captured(&jpeg), bad, &limits).is_err());
    }
    let mut longer_profile = profile.to_vec();
    longer_profile.push(0);
    let mut offset_exif = exif_box.clone();
    offset_exif[3] = 1;
    for bad in [
        ReconstructionMetadataSources {
            icc_profile: Some(&longer_profile),
            ..sources
        },
        ReconstructionMetadataSources {
            exif_box_payload: Some(&offset_exif),
            ..sources
        },
    ] {
        assert!(matches!(
            reconstruct_jpeg(&encoded, captured(&jpeg), bad, &limits),
            Err(Error::Invalid(_))
        ));
    }
}

#[test]
fn unknown_app_envelopes_and_brotli_lengths_are_checked() {
    let base = minimal(63, false);
    let mut bytes = base[..2].to_vec();
    marker(&mut bytes, 0xe7, b"opaque");
    bytes.extend_from_slice(&base[2..]);
    let jpeg = read_coefficients(&bytes).unwrap();
    let parts = serialize_reconstruction_metadata(&jpeg, 4096).unwrap();
    let limits = ReconstructionDecodeLimits::default();
    for index in [0, 1, 2] {
        let mut bad = parts.clone();
        bad.payload[index] ^= 1;
        assert!(decode_reconstruction_metadata(&compressed(&bad), &limits).is_err());
    }
    let mut short = parts.clone();
    short.payload.pop();
    assert!(decode_reconstruction_metadata(&compressed(&short), &limits).is_err());
    let mut excess = parts;
    excess.payload.push(0);
    assert!(decode_reconstruction_metadata(&compressed(&excess), &limits).is_err());
}

#[test]
fn arbitrary_bounded_inputs_never_panic() {
    let mut seed = 0x1837_ab09u32;
    let mut limits = ReconstructionDecodeLimits {
        max_jbrd_bytes: 256,
        max_metadata_bytes: 1024,
        ..Default::default()
    };
    limits.write.max_hint_entries = 64;
    limits.write.max_padding_bits = 256;
    limits.write.structural.max_records = 64;
    limits.write.structural.max_scans = 16;
    for length in 0..256 {
        let bytes: Vec<u8> = (0..length)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        let _ = decode_reconstruction_metadata(&bytes, &limits);
    }
}

// These tests cross the actual container + vendored coefficient decoder path.
// No coefficients, metadata parser model or original JPEG are passed to decode.
fn full_roundtrip(name: &str, original: &[u8]) {
    for threads in [1, 4] {
        let encoded = crate::jpeg::transcode(
            original,
            &crate::jpeg::TranscodeOptions {
                threads,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{name}, encode threads={threads}: {e}"));
        let reconstructed = crate::decode::reconstruct_jpeg(
            &encoded,
            &crate::decode::DecodeOptions {
                threads,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{name}, decode threads={threads}: {e}"));
        assert_eq!(
            reconstructed, original,
            "original JPEG bytes: {name}, threads={threads}"
        );
    }
}

#[test]
fn full_container_gray_rgb_subsampling_progressive_restart_and_refinement() {
    for (name, original) in fixtures::fixtures() {
        full_roundtrip(&name, &original);
    }
}

#[test]
fn full_container_icc_exif_xmp_opaque_metadata_fill_and_tail() {
    for (name, original) in fixtures::metadata_fixtures() {
        full_roundtrip(&name, &original);
    }
}

#[test]
fn full_container_nondefault_and_implicit_padding() {
    for padding in [0, 42, 63] {
        full_roundtrip(&format!("padding_{padding}"), &minimal(padding, true));
    }
}
