//! Command-path coverage for selected PNG frames and full/selected PNG previews.
use super::*;
use clap::Parser;
use rs_jxl::bit_writer::BitWriter;
use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{Frame, FrameOptions};
use rs_jxl::metadata::{Animation, Orientation};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::{EncodeOptions, Image, ModularOptions, encode, encode_frames};
use std::fs;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};

const XMP_BEFORE: &[u8] = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" xmlns:tiff=\"http://ns.adobe.com/tiff/1.0/\" tiff:ImageWidth=\"3\" tiff:ImageLength=\"2\" tiff:Orientation=\"6\">before</x:xmpmeta>";
const XMP_AFTER: &[u8] = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">after</x:xmpmeta>";

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let path = std::env::temp_dir().join(format!(
                "rs-jxl-selected-metadata-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            if fs::create_dir(&path).is_ok() {
                return Self(path);
            }
        }
        panic!("cannot create test directory");
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn args(input: &Path, output: Option<&Path>, tail: &[&str]) -> DecodeArgs {
    let argv = ["rs-jxl", "decode", input.to_str().unwrap()]
        .into_iter()
        .chain(output.map(|p| p.to_str().unwrap()))
        .chain(["--frame", "0", "--quiet"])
        .chain(tail.iter().copied());
    let crate::Command::Decode(args) = crate::Cli::try_parse_from(argv).unwrap().command else {
        panic!("wrong command");
    };
    args
}

fn decoded_pixels(args: &DecodeArgs, bytes: &[u8]) -> DecodedImage {
    let format = args.output_format().unwrap();
    let mut options = args.options(format);
    args.configure_color(&mut options, format).unwrap();
    if format == OutputFormat::Png && options.format == SampleFormat::Native {
        options.format =
            default_integer_format(&decode::read_basic_info(bytes, &options).unwrap()).unwrap();
    }
    match args.frame {
        Some(index) => decode::decode_frame(bytes, index, &options).unwrap(),
        None => decode::decode(bytes, &options).unwrap(),
    }
}

fn encode_options() -> EncodeOptions {
    EncodeOptions {
        max_output_bytes: 4 << 20,
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            lz77: false,
            entropy: Coding::Ans,
            tree_samples: 1,
            max_tree_leaves: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gray(width: u32, height: u32, pixels: Vec<u8>) -> Image {
    let image = Image::new(width, height, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
    let mut metadata = image.metadata().clone();
    metadata.orientation = Orientation::Rotate90Clockwise;
    // Zero prediction without transforms keeps these U8 fixtures in Level 5.
    metadata.modular_16_bit_buffer_sufficient = true;
    image.with_metadata(metadata, Vec::new()).unwrap()
}

fn tiff(orientation: u16) -> Vec<u8> {
    let mut tiff = metadata::orientation_exif(orientation);
    // Distinguish preserved Exif from an orientation-only chunk synthesized by PNG.
    tiff.extend_from_slice(b"selected-metadata-exif");
    tiff
}

fn append_box(bytes: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
    bytes.extend_from_slice(&u32::try_from(payload.len() + 8).unwrap().to_be_bytes());
    bytes.extend_from_slice(&kind);
    bytes.extend_from_slice(payload);
}

fn metadata_box(bytes: &mut Vec<u8>, kind: [u8; 4], payload: &[u8], compressed: bool) {
    if compressed {
        let payload =
            rs_jxl::jpeg::reconstruction_brotli::compress_parts(&kind, payload, 3, 1 << 20)
                .unwrap();
        append_box(bytes, *b"brob", &payload);
    } else {
        append_box(bytes, kind, payload);
    }
}

fn container(raw: &[u8], compressed: bool, exif_after: bool) -> Vec<u8> {
    assert!(raw.starts_with(b"\xff\x0a"));
    let mut bytes = b"\0\0\0\x0cJXL \r\n\x87\n\0\0\0\x14ftypjxl \0\0\0\0jxl ".to_vec();
    let mut exif = vec![0; 4];
    exif.extend_from_slice(&tiff(6));
    metadata_box(&mut bytes, *b"xml ", XMP_BEFORE, compressed);
    if !exif_after {
        metadata_box(&mut bytes, *b"Exif", &exif, compressed);
    }
    append_box(&mut bytes, *b"jxlc", raw);
    metadata_box(&mut bytes, *b"xml ", XMP_AFTER, compressed);
    if exif_after {
        metadata_box(&mut bytes, *b"Exif", &exif, compressed);
    }
    bytes
}

fn animation(images: &[Image]) -> Vec<u8> {
    let images: Vec<_> = images
        .iter()
        .map(|image| {
            let mut metadata = image.metadata().clone();
            metadata.modular_16_bit_buffer_sufficient = true;
            metadata.animation = Some(Animation {
                ticks_per_second_numerator: 10,
                ticks_per_second_denominator: 1,
                loops: 0,
                timecodes: false,
            });
            image.clone().with_metadata(metadata, Vec::new()).unwrap()
        })
        .collect();
    let frames: Vec<_> = images
        .iter()
        .enumerate()
        .map(|(index, image)| Frame {
            image,
            options: FrameOptions {
                duration: 1,
                name: format!("frame-{index}"),
                ..Default::default()
            },
        })
        .collect();
    let bytes = encode_frames(&images[0], &frames, &encode_options()).unwrap();
    assert!(bytes.starts_with(b"\xff\x0a"));
    bytes
}

// Preview fixture header follows tests/decode_backend_layers.rs and the port of
// libjxl metadata.rs/size.h. Copyright JPEG XL Project Authors, BSD-3-Clause;
// retained notice in docs/PORTING_KERNELS.md.
fn image_header(image: &Image, preview: Option<(u32, u32)>) -> Vec<u8> {
    let mut writer = BitWriter::new(4096);
    writer.write(16, 0xaff).unwrap();
    writer.write(1, 0).unwrap();
    writer.write(2, 0).unwrap();
    writer.write(9, u64::from(image.height() - 1)).unwrap();
    writer.write(3, 0).unwrap();
    writer.write(2, 0).unwrap();
    writer.write(9, u64::from(image.width() - 1)).unwrap();
    let metadata = image.metadata();
    if let Some((width, height)) = preview {
        writer.write(1, 0).unwrap();
        writer.write(1, 1).unwrap();
        writer.write(3, metadata.orientation as u64 - 1).unwrap();
        writer.write(1, 0).unwrap(); // intrinsic size absent
        writer.write(1, 1).unwrap(); // preview present
        writer.write(1, 0).unwrap(); // preview div8
        writer.write(2, 0).unwrap();
        writer.write(6, u64::from(height - 1)).unwrap();
        writer.write(3, 0).unwrap();
        writer.write(2, 0).unwrap();
        writer.write(6, u64::from(width - 1)).unwrap();
        writer.write(1, 0).unwrap(); // animation absent
        metadata.bit_depth.write_header(&mut writer).unwrap();
        writer.write(1, 1).unwrap(); // Modular 16-bit buffers
        writer.write(2, 0).unwrap(); // no extra channels
        writer.write(1, 0).unwrap(); // not XYB
        metadata.color_encoding.write_header(&mut writer).unwrap();
        metadata.tone_mapping.write_header(&mut writer).unwrap();
        writer.write(2, 0).unwrap(); // no extensions
    } else {
        metadata.write_header(&mut writer).unwrap();
    }
    rs_jxl::metadata::write_default_transform_data(&mut writer).unwrap();
    writer.zero_pad_to_byte().unwrap();
    writer.into_bytes().unwrap()
}

fn preview_fixture() -> Vec<u8> {
    let preview = gray(2, 1, vec![31, 97]);
    let main = gray(3, 2, vec![1, 2, 3, 4, 5, 6]);
    let mut raw = image_header(&main, Some((2, 1)));
    for image in [&preview, &main] {
        let encoded = encode(image, &encode_options()).unwrap();
        let header = image_header(image, None);
        assert!(encoded.starts_with(&header));
        raw.extend_from_slice(&encoded[header.len()..]);
    }
    raw
}

fn assert_png(path: &Path, image: &DecodedImage, frame: &DecodedFrame, orientation: u16) {
    let mut reader = png::Decoder::new(Cursor::new(fs::read(path).unwrap()))
        .read_info()
        .unwrap();
    let info = reader.info();
    assert!(info.animation_control.is_none());
    assert_eq!(
        (info.width as usize, info.height as usize),
        (frame.width, frame.height)
    );
    assert_eq!(
        info.exif_metadata.as_deref(),
        Some(tiff(orientation).as_slice())
    );
    assert_eq!(info.utf8_text.len(), 2);
    for (chunk, expected) in info.utf8_text.iter().zip([XMP_BEFORE, XMP_AFTER]) {
        assert_eq!(chunk.keyword, "XML:com.adobe.xmp");
        assert_eq!(chunk.get_text().unwrap().as_bytes(), expected);
    }
    assert_eq!(
        info.icc_profile.as_deref().unwrap(),
        image
            .output_profile
            .as_ref()
            .unwrap()
            .try_as_icc()
            .unwrap()
            .as_slice()
    );
    let PixelData::U8(expected) = &frame.pixels else {
        panic!("fixture must use 8-bit output");
    };
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let output = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(&pixels[..output.buffer_size()], expected);
}

#[test]
fn selected_png_preserves_plain_and_compressed_metadata_before_and_after_pixels() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let raw = encode(&gray(3, 2, vec![1, 2, 3, 4, 5, 6]), &encode_options()).unwrap();
    for compressed in [false, true] {
        for exif_after in [false, true] {
            let bytes = container(&raw, compressed, exif_after);
            fs::write(&input, &bytes).unwrap();
            for keep_orientation in [false, true] {
                let output = dir.0.join(format!(
                    "out-{compressed}-{exif_after}-{keep_orientation}.png"
                ));
                let mut a = args(&input, Some(&output), &[]);
                a.keep_orientation = keep_orientation;
                let image = decoded_pixels(&a, &bytes);
                assert!(!image.complete);
                assert!(image.bytes_read < bytes.len());
                run(a).unwrap();
                assert_png(
                    &output,
                    &image,
                    &image.frames[0],
                    if keep_orientation { 6 } else { 1 },
                );
            }
        }
    }
}

#[test]
fn selected_png_sidecars_keep_transformed_profile_and_selected_pixel_accounting() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let output = dir.0.join("out.png");
    let json = dir.0.join("metadata.json");
    let icc = dir.0.join("output.icc");
    let original_icc = dir.0.join("original.icc");
    let images: Vec<_> = [
        vec![32, 96, 192, 240, 128, 16],
        vec![16, 128, 240, 192, 96, 32],
    ]
    .into_iter()
    .map(|pixels| Image::new(2, 1, PixelLayout::Rgb, PixelData::U8(pixels)).unwrap())
    .collect();
    let bytes = container(&animation(&images), false, true);
    fs::write(&input, &bytes).unwrap();
    let a = args(
        &input,
        Some(&output),
        &[
            "--color-space",
            "linear",
            "--bits",
            "8",
            "--display-nits",
            "100",
            "--metadata-out",
            json.to_str().unwrap(),
            "--icc-out",
            icc.to_str().unwrap(),
            "--orig-icc-out",
            original_icc.to_str().unwrap(),
            "--num-reps",
            "2",
        ],
    );
    let image = decoded_pixels(&a, &bytes);
    run(a).unwrap();
    assert_png(&output, &image, &image.frames[0], 1);
    let metadata: serde_json::Value = serde_json::from_slice(&fs::read(json).unwrap()).unwrap();
    assert_eq!(metadata["complete"], false);
    assert_eq!(metadata["bytes_read"], image.bytes_read);
    assert!(image.bytes_read < bytes.len());
    assert_eq!(metadata["frames"].as_array().unwrap().len(), 2);
    assert_eq!(metadata["frames"][1]["name"], "frame-1");
    assert_eq!(metadata["boxes"].as_array().unwrap().len(), 3);
    assert_eq!(
        metadata["output_intensity_target"],
        image.output_intensity_target
    );
    let output_icc = fs::read(icc).unwrap();
    let original_icc = fs::read(original_icc).unwrap();
    assert_eq!(
        output_icc,
        image
            .output_profile
            .as_ref()
            .unwrap()
            .try_as_icc()
            .unwrap()
            .as_slice()
    );
    assert_eq!(
        original_icc,
        image
            .embedded_profile
            .as_ref()
            .unwrap()
            .try_as_icc()
            .unwrap()
            .as_slice()
    );
    assert_ne!(output_icc, original_icc);
}

#[test]
fn png_preview_preserves_metadata_for_selected_full_and_streamed_decode() {
    let mut bytes = container(&preview_fixture(), true, true);
    for kind in [*b"jumb", *b"jbrd", *b"jhgm"] {
        append_box(&mut bytes, kind, b"not representable in PNG");
    }
    for (frame, stream_input) in [(Some(0), false), (None, false), (None, true)] {
        for primary in [None, Some("out.npy"), Some("out.png")] {
            let dir = Directory::new();
            let input = dir.0.join("input.jxl");
            fs::write(&input, &bytes).unwrap();
            let output = primary.map(|name| dir.0.join(name));
            let preview = dir.0.join("preview.png");
            let mut a = args(
                &input,
                output.as_deref(),
                &["--bits", "8", "--preview-out", preview.to_str().unwrap()],
            );
            a.frame = frame;
            a.stream_input = stream_input;
            a.input_chunk_size = stream_input.then_some(17);
            let image = decoded_pixels(&a, &bytes);
            run(a).unwrap();
            assert_png(&preview, &image, image.preview.as_ref().unwrap(), 1);
            if let Some(output) = output {
                if output.extension().unwrap() == "png" {
                    assert_png(&output, &image, &image.frames[0], 1);
                } else {
                    assert!(fs::read(output).unwrap().starts_with(b"\x93NUMPY"));
                }
            }
        }
    }
}

#[test]
fn selected_png_rejects_bad_trailing_metadata_before_any_publication() {
    let raw = preview_fixture();
    let mut writer = rs_jxl::container::ChunkedWriter::new(Vec::new(), 5, 1 << 20).unwrap();
    writer.write_chunk(&raw, true).unwrap();
    let base = writer.finish().unwrap();
    let mut invalid_exif = Vec::new();
    append_box(&mut invalid_exif, *b"Exif", &[0, 0, 0, 1]);
    let mut invalid_xmp = Vec::new();
    append_box(&mut invalid_xmp, *b"xml ", &[0xff]);
    for tail in [
        b"\0\0\0\x20xml <x/>".as_slice(),
        b"\0\0\0\x10brobxml \xff\xff\xff\xff",
        invalid_exif.as_slice(),
        invalid_xmp.as_slice(),
    ] {
        for preview_only in [false, true] {
            for allow_partial in [false, true] {
                let dir = Directory::new();
                let input = dir.0.join("input.jxl");
                let output = dir.0.join(if preview_only { "out.npy" } else { "out.png" });
                let preview = dir.0.join("preview.png");
                let json = dir.0.join("metadata.json");
                let icc = dir.0.join("output.icc");
                let original_icc = dir.0.join("original.icc");
                let mut bytes = base.clone();
                bytes.extend_from_slice(tail);
                fs::write(&input, &bytes).unwrap();
                fs::write(&output, b"keep existing output").unwrap();
                let mut a = args(&input, Some(&output), &["--bits", "8", "--force"]);
                a.allow_partial_files = allow_partial;
                a.preview_out = preview_only.then_some(preview.clone());
                // Exercise the PNG-triggered inspection without sidecars first.
                assert_eq!(decoded_pixels(&a, &bytes).frames.len(), 1);
                assert!(run(a).is_err());
                assert_eq!(fs::read(&output).unwrap(), b"keep existing output");
                assert!(!preview.exists());
                let mut a = args(&input, Some(&output), &["--bits", "8", "--force"]);
                a.allow_partial_files = allow_partial;
                a.preview_out = preview_only.then_some(preview.clone());
                a.metadata_out = Some(json.clone());
                a.icc_out = Some(icc.clone());
                a.orig_icc_out = Some(original_icc.clone());
                assert!(run(a).is_err());
                assert_eq!(fs::read(&output).unwrap(), b"keep existing output");
                assert!(
                    !preview.exists() && !json.exists() && !icc.exists() && !original_icc.exists()
                );
                assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
            }
        }
    }
}

#[test]
fn selected_png_scans_metadata_without_decoding_later_pixels() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let output = dir.0.join("out.png");
    let mut seed = 7u32;
    let noisy = (0..3000)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        })
        .collect();
    let mut raw = animation(&[gray(100, 30, vec![17; 3000]), gray(100, 30, noisy)]);
    let first = decode::decode_frame(&raw, 0, &DecodeOptions::default()).unwrap();
    // The noisy second frame dominates this fixture. Input bytes_read includes
    // decoder read-ahead and cannot be used as the first frame's byte boundary.
    // ANS terminal-state validation detects this damage when pixels are decoded.
    assert!(raw.len() > 1000);
    let start = raw.len() / 2;
    raw[start..].fill(0xff);
    let selected = decode::decode_frame(&raw, 0, &DecodeOptions::default()).unwrap();
    assert_eq!(selected.frames[0].pixels, first.frames[0].pixels);
    assert_eq!(
        decode::inspect(&raw, &DecodeOptions::default())
            .unwrap()
            .frames
            .len(),
        2
    );
    assert!(decode::decode(&raw, &DecodeOptions::default()).is_err());
    let bytes = container(&raw, false, true);
    fs::write(&input, &bytes).unwrap();
    let a = args(&input, Some(&output), &[]);
    let image = decoded_pixels(&a, &bytes);
    run(a).unwrap();
    assert_png(&output, &image, &image.frames[0], 1);
}

#[test]
fn metadata_free_selected_npy_keeps_the_incomplete_later_frame_fast_path() {
    let dir = Directory::new();
    let input = dir.0.join("input.jxl");
    let output = dir.0.join("out.npy");
    let mut bytes = animation(&[
        gray(3, 2, vec![1, 2, 3, 4, 5, 6]),
        gray(3, 2, vec![7, 8, 9, 10, 11, 12]),
    ]);
    bytes.pop();
    assert!(decode::inspect(&bytes, &DecodeOptions::default()).is_err());
    fs::write(&input, &bytes).unwrap();
    let a = args(&input, Some(&output), &[]);
    let image = decoded_pixels(&a, &bytes);
    assert!(!image.complete);
    run(a).unwrap();
    let npy = fs::read(output).unwrap();
    assert!(npy.starts_with(b"\x93NUMPY\x01\x00"));
    let start = 10 + usize::from(u16::from_le_bytes(npy[8..10].try_into().unwrap()));
    let PixelData::U8(expected) = &image.frames[0].pixels else {
        panic!("expected native U8");
    };
    assert_eq!(&npy[start..], expected);
}
