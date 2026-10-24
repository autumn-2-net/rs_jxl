//! Complete JPEG -> JXL -> original-JPEG checks, using the runtime Rust decoder.
//! No pixel re-encoding, original-JPEG lookup or external runtime decoder.
use rs_jxl::Error;
use rs_jxl::decode::{DecodeOptions, SampleFormat, decode, reconstruct_jpeg};
use rs_jxl::jpeg::{TranscodeOptions, transcode};
use std::io::Write;

fn marker(bytes: &mut Vec<u8>, code: u8, payload: &[u8]) {
    bytes.extend([0xff, code]);
    bytes.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    bytes.extend_from_slice(payload);
}

fn jpeg_fixture(with_sources: bool, tail_length: usize) -> Vec<u8> {
    let mut bytes = vec![0xff, 0xd8];
    if with_sources {
        marker(
            &mut bytes,
            0xe1,
            b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0",
        );
        marker(
            &mut bytes,
            0xe1,
            b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta>exact</x:xmpmeta>",
        );
    }
    let mut quant = vec![0];
    quant.extend([1; 64]);
    marker(&mut bytes, 0xdb, &quant);
    marker(&mut bytes, 0xc0, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
    let mut huffman = Vec::new();
    for selector in [0, 16] {
        huffman.extend([selector, 1]);
        huffman.extend([0; 15]);
        huffman.push(0);
    }
    marker(&mut bytes, 0xc4, &huffman);
    marker(&mut bytes, 0xda, &[1, 1, 0, 0, 63, 0]);
    bytes.extend([0x2a, 0xff, 0xd9]);
    bytes.resize(bytes.len() + tail_length, 0x5a);
    bytes
}

fn encoded(original: &[u8]) -> Vec<u8> {
    transcode(original, &TranscodeOptions::default()).unwrap()
}

// These fixtures are small, finite boxes produced by the encoder above.
fn box_range(bytes: &[u8], kind: [u8; 4]) -> std::ops::Range<usize> {
    let mut offset = 0;
    while offset < bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && size <= bytes.len() - offset);
        if bytes[offset + 4..offset + 8] == kind {
            return offset..offset + size;
        }
        offset += size;
    }
    panic!("missing fixture box {kind:?}");
}

fn append_box(bytes: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
    bytes.extend(u32::try_from(payload.len() + 8).unwrap().to_be_bytes());
    bytes.extend(kind);
    bytes.extend_from_slice(payload);
}

fn compressed_box(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut result = kind.to_vec();
    {
        let mut writer = brotli::CompressorWriter::new(&mut result, 4096, 3, 18);
        writer.write_all(payload).unwrap();
    }
    result
}

fn assert_decode_error(result: rs_jxl::Result<Vec<u8>>) {
    assert!(matches!(result, Err(Error::Decoding(_))), "{result:?}");
}

#[test]
fn exact_jpeg_bytes_ignore_pixel_output_and_progression_options() {
    let original = jpeg_fixture(true, 7);
    let bytes = encoded(&original);
    for threads in [1, 4] {
        for downsampling in [1, 2, 4, 8] {
            let options = DecodeOptions {
                threads,
                downsampling,
                format: SampleFormat::F32,
                adjust_orientation: false,
                render_spot_colors: false,
                include_extra_channels: false,
                coalescing: false,
                include_preview: true,
                allow_partial: true,
                ..Default::default()
            };
            assert_eq!(reconstruct_jpeg(&bytes, &options).unwrap(), original);
        }
    }
    assert_eq!(
        reconstruct_jpeg(&bytes, &DecodeOptions::default()).unwrap(),
        original
    );
}

#[test]
fn every_incomplete_container_prefix_fails_even_with_partial_pixels_enabled() {
    let original = jpeg_fixture(false, 0);
    let bytes = encoded(&original);
    let options = DecodeOptions {
        allow_partial: true,
        downsampling: 8,
        ..Default::default()
    };
    for length in 0..bytes.len() {
        assert!(
            reconstruct_jpeg(&bytes[..length], &options).is_err(),
            "accepted prefix {length}/{}",
            bytes.len()
        );
    }
    assert_eq!(reconstruct_jpeg(&bytes, &options).unwrap(), original);
    for tail in [
        b"\0\0\0\x20xml <x/>".as_slice(),
        b"\0\0\0\x07xml ",
        b"\0\0\0\x01xml \0\0\0\0",
    ] {
        let mut truncated = bytes.clone();
        truncated.extend_from_slice(tail);
        assert_decode_error(reconstruct_jpeg(&truncated, &options));
    }
}

#[test]
fn required_metadata_sources_must_be_present_and_unambiguous() {
    let original = jpeg_fixture(true, 0);
    let bytes = encoded(&original);
    let options = DecodeOptions::default();
    for kind in [*b"jbrd", *b"Exif", *b"xml "] {
        let range = box_range(&bytes, kind);
        let payload = &bytes[range.start + 8..range.end];
        let mut missing = bytes.clone();
        missing.drain(range.clone());
        assert_decode_error(reconstruct_jpeg(&missing, &options));
        for compressed in [false, true] {
            let mut duplicate = bytes.clone();
            if compressed {
                append_box(&mut duplicate, *b"brob", &compressed_box(kind, payload));
            } else {
                append_box(&mut duplicate, kind, payload);
            }
            let error = reconstruct_jpeg(&duplicate, &options).unwrap_err();
            assert!(matches!(error, Error::Decoding(_)), "{error:?}");
            // brob may not wrap jbrd at all, even before source selection.
            if !(compressed && kind == *b"jbrd") {
                assert!(error.to_string().contains("ambiguous duplicate"), "{error}");
            }
        }
    }
}

#[test]
fn unrelated_duplicate_display_boxes_do_not_change_original_bytes() {
    let original = jpeg_fixture(false, 7);
    let mut bytes = encoded(&original);
    for kind in [*b"Exif", *b"xml "] {
        append_box(&mut bytes, kind, b"\0\0\0\0first display metadata");
        append_box(
            &mut bytes,
            *b"brob",
            &compressed_box(kind, b"\0\0\0\0different display metadata"),
        );
    }
    assert_eq!(
        reconstruct_jpeg(&bytes, &DecodeOptions::default()).unwrap(),
        original
    );
}

#[test]
fn required_sources_can_follow_codestream_compressed_or_extend_to_eof() {
    let original = jpeg_fixture(true, 7);
    let bytes = encoded(&original);
    for kind in [*b"jbrd", *b"Exif", *b"xml "] {
        let range = box_range(&bytes, kind);
        let payload = &bytes[range.start + 8..range.end];
        for compressed in [false, true] {
            if compressed && kind == *b"jbrd" {
                continue;
            }
            for unbounded in [false, true] {
                let mut relocated = bytes.clone();
                relocated.drain(range.clone());
                let start = relocated.len();
                if compressed {
                    append_box(&mut relocated, *b"brob", &compressed_box(kind, payload));
                } else {
                    append_box(&mut relocated, kind, payload);
                }
                if unbounded {
                    relocated[start..start + 4].fill(0);
                }
                assert_eq!(
                    reconstruct_jpeg(&relocated, &DecodeOptions::default()).unwrap(),
                    original,
                    "{kind:?}, compressed={compressed}, unbounded={unbounded}"
                );
            }
        }
    }
}

#[test]
fn expanded_metadata_and_final_output_limits_keep_the_limit_error() {
    // The compressed tail fits in the metadata budget, but its expansion does not.
    let original = jpeg_fixture(false, 8192);
    let bytes = encoded(&original);
    let range = box_range(&bytes, *b"jbrd");
    assert!(range.len() - 8 < 4096);
    let mut options = DecodeOptions::default();
    options.limits.max_metadata_bytes = 4096;
    assert!(matches!(
        reconstruct_jpeg(&bytes, &options),
        Err(Error::Limit("expanded metadata"))
    ));
    options = DecodeOptions::default();
    options.limits.max_output_bytes = original.len();
    assert_eq!(reconstruct_jpeg(&bytes, &options).unwrap(), original);
    options.limits.max_output_bytes -= 1;
    assert!(matches!(
        reconstruct_jpeg(&bytes, &options),
        Err(Error::Limit("JPEG output bytes"))
    ));
    for case in 0..5 {
        options = DecodeOptions::default();
        match case {
            0 => options.limits.max_input_bytes = bytes.len() - 1,
            1 => options.limits.max_pixels = 63,
            2 => options.limits.max_samples = 1,
            3 => options.limits.max_frames = 0,
            _ => options.limits.max_metadata_bytes = range.len() - 9,
        }
        let result = reconstruct_jpeg(&bytes, &options);
        assert!(
            matches!(result, Err(Error::Limit(_))),
            "case {case}: {result:?}"
        );
    }
}

#[test]
fn pixel_jxl_is_not_silently_reencoded_as_jpeg() {
    let image = rs_jxl::Image::new(
        2,
        2,
        rs_jxl::PixelLayout::Rgb,
        rs_jxl::PixelData::U8(vec![127; 12]),
    )
    .unwrap();
    let bytes = rs_jxl::encode(&image, &Default::default()).unwrap();
    assert!(reconstruct_jpeg(&bytes, &DecodeOptions::default()).is_err());
}

#[test]
fn reconstruction_rejects_empty_truncated_and_invalid_input() {
    for bytes in [&[][..], &[0xff, 0x0a][..], b"not JPEG XL"] {
        assert!(reconstruct_jpeg(bytes, &DecodeOptions::default()).is_err());
    }
}

#[test]
#[ignore = "explicit RS_JXL_JPEG_INPUTS path list; bounded real-data decoder acceptance"]
fn real_jpegs_restore_original_bytes_from_jxl_alone() {
    let roots = std::env::var_os("RS_JXL_JPEG_INPUTS").expect("set RS_JXL_JPEG_INPUTS");
    let mut paths = Vec::new();
    for path in std::env::split_paths(&roots) {
        if path.is_file() {
            paths.push(path);
        } else {
            for entry in std::fs::read_dir(path).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_file()
                    && entry.path().extension().is_some_and(|ext| {
                        ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg")
                    })
                {
                    paths.push(entry.path());
                }
                assert!(paths.len() <= 64, "select at most 64 JPEG fixtures");
            }
        }
    }
    paths.sort();
    paths.dedup();
    assert!(!paths.is_empty() && paths.len() <= 64);
    for path in paths {
        assert!(std::fs::metadata(&path).unwrap().len() <= 16 * 1024 * 1024);
        let source = std::fs::read(&path).unwrap();
        for effort in [3, 7] {
            let encoded = transcode(
                &source,
                &TranscodeOptions {
                    effort,
                    threads: 2,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("{} e{effort}: {error}", path.display()));
            for threads in [1, 4] {
                let options = DecodeOptions {
                    threads,
                    ..Default::default()
                };
                let restored = reconstruct_jpeg(&encoded, &options).unwrap_or_else(|error| {
                    panic!("{} e{effort} t{threads}: {error}", path.display())
                });
                assert_eq!(restored.len(), source.len(), "{}", path.display());
                assert!(
                    restored == source,
                    "{} e{effort} t{threads}: original bytes differ",
                    path.display()
                );
            }
            // Capture hooks must not make ordinary pixel decoding unusable.
            let pixels = decode(&encoded, &DecodeOptions::default()).unwrap();
            assert_eq!(pixels.frames.len(), 1);
            eprintln!(
                "{} e{effort}: {} bytes -> {} JXL bytes; JPEG exact at threads1/4; pixels decoded",
                path.file_name().unwrap().to_string_lossy(),
                source.len(),
                encoded.len()
            );
        }
    }
}
