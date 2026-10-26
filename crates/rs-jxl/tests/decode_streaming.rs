//! Persistent-state, EOF and resource regressions. Fixtures stay in memory;
//! pixel samples, not compressed output identity, are the acceptance criterion.

use std::cell::Cell;
use std::io::{self, Cursor, Read, Write};

use jxl_decoder::api::{Event, JxlDecoder, JxlDecoderOptions};
use rs_jxl::bit_writer::BitWriter;
use rs_jxl::container::ChunkedWriter;
use rs_jxl::decode::{
    ColorOptions, ColorProfile, DecodeOptions, DecodedFrame, DecodedImage, SampleFormat,
    StreamDecoder, decode, decode_reader, decode_reader_with, decode_reader_with_chunk_size,
};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::metadata::{Animation, BitDepth, Orientation};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{self, Reference};
use rs_jxl::{
    EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode, encode_frames,
};

fn encoding() -> EncodeOptions {
    EncodeOptions {
        max_output_bytes: 4 << 20,
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            lz77: false,
            entropy: Coding::Prefix,
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
    metadata.modular_16_bit_buffer_sufficient = true;
    image.with_metadata(metadata, vec![]).unwrap()
}

fn raw() -> Vec<u8> {
    let data = encode(&gray(4, 3, (0..12).map(|n| n * 19).collect()), &encoding()).unwrap();
    assert!(data.starts_with(&[0xff, 0x0a]));
    data
}

fn container(raw: &[u8]) -> Vec<u8> {
    let mut writer = ChunkedWriter::new(Vec::new(), 5, 1 << 20).unwrap();
    writer.write_chunk(raw, true).unwrap();
    writer.finish().unwrap()
}

fn sequence(durations: &[u32]) -> Vec<u8> {
    let images: Vec<_> = durations
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let mut pixels = vec![17; 48];
            pixels[index * 3] = 80 + index as u8;
            let image = gray(8, 6, pixels);
            let mut metadata = image.metadata().clone();
            // One-tick display frames must remain below Level 5's 120 fps.
            metadata.animation = Some(Animation {
                ticks_per_second_numerator: 60,
                ..Animation::default()
            });
            image.with_metadata(metadata, vec![]).unwrap()
        })
        .collect();
    let plan = multiframe::plan(&images, durations, Reference::Previous).unwrap();
    let frames: Vec<_> = plan.iter().map(|frame| frame.as_frame()).collect();
    encode_frames(&images[0], &frames, &encoding()).unwrap()
}

fn assert_frame(actual: &DecodedFrame, expected: &DecodedFrame) {
    assert_eq!(actual.index, expected.index);
    assert_eq!(
        (actual.width, actual.height),
        (expected.width, expected.height)
    );
    assert_eq!(actual.layout, expected.layout);
    assert_eq!(actual.pixels, expected.pixels);
    assert_eq!(actual.extra_planes, expected.extra_planes);
    assert_eq!(actual.duration_ticks, expected.duration_ticks);
    assert_eq!(actual.name, expected.name);
    assert_eq!(actual.complete, expected.complete);
    assert_eq!(
        format!("{:?}", actual.header),
        format!("{:?}", expected.header)
    );
}

fn assert_image(actual: &DecodedImage, expected: &DecodedImage) {
    assert_eq!(actual.info.size, expected.info.size);
    assert_eq!(actual.info.bit_depth, expected.info.bit_depth);
    assert_eq!(actual.embedded_profile, expected.embedded_profile);
    assert_eq!(actual.output_profile, expected.output_profile);
    assert_eq!(
        actual.output_intensity_target.to_bits(),
        expected.output_intensity_target.to_bits()
    );
    assert_eq!(actual.complete, expected.complete);
    assert_eq!(actual.frames.len(), expected.frames.len());
    for (actual, expected) in actual.frames.iter().zip(&expected.frames) {
        assert_frame(actual, expected);
    }
    assert_eq!(actual.preview.is_some(), expected.preview.is_some());
    if let (Some(actual), Some(expected)) = (&actual.preview, &expected.preview) {
        assert_frame(actual, expected);
    }
    assert_eq!(actual.boxes.len(), expected.boxes.len());
    for (actual, expected) in actual.boxes.iter().zip(&expected.boxes) {
        assert_eq!(actual.kind, expected.kind);
        assert_eq!(actual.data, expected.data);
    }
}

#[test]
fn bytewise_and_irregular_reads_match_one_shot_samples_and_profiles() {
    let pixels = [
        PixelData::U8(vec![0, 91, 231, 255, 14, 0, 80, 23]),
        PixelData::U16(vec![0, 65535, 1001, 9, 0, 321, 32768, 71]),
        PixelData::F32(
            [
                0, 0x80000000, 0x3f800000, 0x7fc12345, 1, 0xff800000, 0x7f800000, 0x3e800000,
            ]
            .map(f32::from_bits)
            .to_vec(),
        ),
    ];
    for pixels in pixels {
        let source = Image::new(4, 2, PixelLayout::Gray, pixels).unwrap();
        let bytes = encode(&source, &encoding()).unwrap();
        let options = DecodeOptions::default();
        let expected = decode(&bytes, &options).unwrap();
        for chunk in [1, 2, 7, 31, 64 * 1024] {
            let actual = decode_reader_with_chunk_size(bytes.as_slice(), &options, chunk).unwrap();
            assert_image(&actual, &expected);
        }
        assert_image(
            &decode_reader(bytes.as_slice(), &options).unwrap(),
            &expected,
        );
    }
}

#[test]
fn every_two_chunk_boundary_resumes_the_same_decoder() {
    let bytes = raw();
    let expected = decode(&bytes, &DecodeOptions::default()).unwrap();
    for split in 0..=bytes.len() {
        let mut stream = StreamDecoder::new(&DecodeOptions::default()).unwrap();
        stream.push(&bytes[..split]).unwrap();
        let mut frames = stream.take_frames();
        stream.push(&[]).unwrap();
        stream.push(&bytes[split..]).unwrap();
        frames.extend(stream.take_frames());
        let summary = stream.finish().unwrap();
        frames.extend(stream.take_frames());
        assert!(summary.complete);
        assert_eq!(summary.frame_count, 1);
        assert_eq!(frames.len(), 1);
        assert_frame(&frames[0], &expected.frames[0]);
        assert!(stream.basic_info().is_some());
        assert!(stream.push(&[]).is_err());
        assert!(stream.finish().is_err());
    }
}

#[test]
fn frames_are_available_before_remaining_frames_and_before_eof() {
    let bytes = sequence(&[1, 2, 3]);
    let expected = decode(&bytes, &DecodeOptions::default()).unwrap();
    let mut stream = StreamDecoder::new(&DecodeOptions::default()).unwrap();
    let mut frames = Vec::new();
    let mut first_at = None;
    for (offset, byte) in bytes.iter().enumerate() {
        stream.push(std::slice::from_ref(byte)).unwrap();
        for frame in stream.take_frames() {
            first_at.get_or_insert(offset + 1);
            frames.push(frame);
        }
    }
    assert!(
        first_at.unwrap() < bytes.len(),
        "decoder deferred all frames until input end"
    );
    assert_eq!(frames.len(), expected.frames.len());
    for (actual, expected) in frames.iter().zip(&expected.frames) {
        assert_frame(actual, expected);
    }
    assert!(stream.finish().unwrap().complete);
    assert!(stream.take_frames().is_empty());
}

#[test]
fn reader_callbacks_run_between_reads_not_after_full_input_collection() {
    struct Observed<'a> {
        data: &'a [u8],
        position: &'a Cell<usize>,
        saw_eof: &'a Cell<bool>,
    }
    impl Read for Observed<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let count = self.data.len().min(out.len()).min(3);
            out[..count].copy_from_slice(&self.data[..count]);
            self.data = &self.data[count..];
            self.position.set(self.position.get() + count);
            self.saw_eof.set(count == 0);
            Ok(count)
        }
    }
    let mut bytes = container(&raw());
    append_box(&mut bytes, *b"xml ", &vec![b'x'; 4096]);
    let position = Cell::new(0);
    let saw_eof = Cell::new(false);
    let mut count = 0;
    let summary = decode_reader_with(
        Observed {
            data: &bytes,
            position: &position,
            saw_eof: &saw_eof,
        },
        &DecodeOptions::default(),
        17,
        |_| {
            assert!(!saw_eof.get());
            assert!(position.get() < bytes.len());
            count += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(count, 1);
    assert!(summary.complete);
    assert_eq!(summary.boxes[0].data.len(), 4096);
    assert!(saw_eof.get());
}

fn append_box(bytes: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
    bytes.extend_from_slice(&u32::try_from(payload.len() + 8).unwrap().to_be_bytes());
    bytes.extend_from_slice(&kind);
    bytes.extend_from_slice(payload);
}

fn compressed(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut result = kind.to_vec();
    {
        let mut writer = brotli::CompressorWriter::new(&mut result, 4096, 4, 18);
        writer.write_all(payload).unwrap();
    }
    result
}

#[test]
fn metadata_between_chunks_and_unbounded_compressed_tail_wait_for_eof() {
    let raw = raw();
    let mut writer = ChunkedWriter::new(Vec::new(), 5, 1 << 20).unwrap();
    writer.write_metadata(*b"Exif", b"\0\0\0\0exif").unwrap();
    writer.write_chunk(&raw[..2], false).unwrap();
    writer.write_metadata(*b"xml ", b"before").unwrap();
    writer.write_chunk(&raw[2..], true).unwrap();
    let prefix = writer.finish().unwrap();
    let tail = b"last unbounded metadata\0".repeat(100);
    for compressed_tail in [false, true] {
        let mut bytes = prefix.clone();
        append_box(&mut bytes, *b"test", b"ignored");
        append_box(
            &mut bytes,
            *b"brob",
            &compressed(*b"jumb", b"bounded compressed"),
        );
        bytes.extend_from_slice(&0u32.to_be_bytes());
        if compressed_tail {
            bytes.extend_from_slice(b"brob");
            bytes.extend_from_slice(&compressed(*b"xml ", &tail));
        } else {
            bytes.extend_from_slice(b"xml ");
            bytes.extend_from_slice(&tail);
        }
        let expected = decode(&bytes, &DecodeOptions::default()).unwrap();
        for chunk in [1, 5, 8192] {
            let actual =
                decode_reader_with_chunk_size(bytes.as_slice(), &DecodeOptions::default(), chunk)
                    .unwrap();
            assert_image(&actual, &expected);
            assert_eq!(actual.bytes_read, bytes.len());
        }
        let mut stream = StreamDecoder::new(&DecodeOptions::default()).unwrap();
        stream.push(&prefix).unwrap();
        assert_eq!(stream.take_frames().len(), 1);
        for part in bytes[prefix.len()..].chunks(1) {
            stream.push(part).unwrap();
        }
        let summary = stream.finish().unwrap();
        assert_eq!(summary.boxes.last().unwrap().data, tail);
        assert_eq!(summary.bytes_read, bytes.len());
    }
}

fn two_chunk_container(raw: &[u8]) -> (Vec<u8>, usize, usize) {
    let split = raw.len() / 2;
    let mut writer = ChunkedWriter::new(Vec::new(), 5, 1 << 20).unwrap();
    writer.write_chunk(&raw[..split], false).unwrap();
    writer.write_chunk(&raw[split..], true).unwrap();
    let ordered = writer.finish().unwrap();
    let prefix = 32;
    let middle = prefix + 12 + split;
    assert_eq!(&ordered[16..24], b"ftypjxl ");
    assert_eq!(&ordered[24..28], &0u32.to_be_bytes());
    (ordered, prefix, middle)
}

#[test]
fn extended_box_headers_with_ordered_version_zero_chunks() {
    let (mut bytes, _, _) = two_chunk_container(&raw());
    bytes.extend_from_slice(&1u32.to_be_bytes());
    bytes.extend_from_slice(b"xml ");
    bytes.extend_from_slice(&21u64.to_be_bytes());
    bytes.extend_from_slice(b"large");
    let options = DecodeOptions::default();
    let expected = decode(&bytes, &options).unwrap();
    assert_eq!(expected.boxes.len(), 1);
    assert_eq!(expected.boxes[0].data, b"large");
    for chunk in [1, 3, 17] {
        assert_image(
            &decode_reader_with_chunk_size(bytes.as_slice(), &options, chunk).unwrap(),
            &expected,
        );
    }
}

fn out_of_order_container(raw: &[u8], version: u32) -> Vec<u8> {
    let (ordered, prefix, middle) = two_chunk_container(raw);
    let mut bytes = ordered[..prefix].to_vec();
    // ftyp minor version, not the codestream's TOC permutation. The writer
    // intentionally emits version 0; only this extension fixture declares 1.
    bytes[24..28].copy_from_slice(&version.to_be_bytes());
    bytes.extend_from_slice(&ordered[middle..]);
    bytes.extend_from_slice(&ordered[prefix..middle]);
    bytes
}

#[test]
fn out_of_order_jxlp_is_rejected_in_version_zero() {
    let bytes = out_of_order_container(&raw(), 0);
    for allow_partial in [false, true] {
        let options = DecodeOptions {
            allow_partial,
            ..Default::default()
        };
        assert!(matches!(decode(&bytes, &options), Err(Error::Decoding(_))));
        for chunk in [1, 3, 17] {
            assert!(matches!(
                decode_reader_with_chunk_size(bytes.as_slice(), &options, chunk),
                Err(Error::Decoding(_)),
            ));
        }
    }
}

#[test]
fn out_of_order_jxlp_requires_explicit_version_one() {
    // Maintained upstream extension: libjxl PR #4741 and jxl-rs PR #752.
    let options = DecodeOptions::default();
    for raw in [raw(), sequence(&[1, 2, 3])] {
        assert!(raw.starts_with(&[0xff, 0x0a]));
        let expected = decode(&raw, &options).unwrap();
        let bytes = out_of_order_container(&raw, 1);
        assert_image(&decode(&bytes, &options).unwrap(), &expected);
        for chunk in [1, 3, 17] {
            assert_image(
                &decode_reader_with_chunk_size(bytes.as_slice(), &options, chunk).unwrap(),
                &expected,
            );
        }
        // A checkpoint cannot reproduce the already-buffered future chunk.
        // Later frames must not restart an index with missing dependencies.
        assert_backend_seek_index(&bytes, &expected, 0);

        // Version 1 itself does not disable seeking: physically ordered input
        // still publishes complete, usable targets.
        let (mut ordered, _, _) = two_chunk_container(&raw);
        ordered[24..28].copy_from_slice(&1u32.to_be_bytes());
        assert_backend_seek_index(&ordered, &expected, expected.frames.len());
    }
}

fn assert_backend_seek_index(bytes: &[u8], expected: &DecodedImage, count: usize) {
    let mut options = JxlDecoderOptions::default();
    options.scan_frames_only = true;
    let mut decoder = JxlDecoder::new(options);
    decoder.close_input();
    let mut input = bytes;
    let mut headers = 0;
    loop {
        match decoder.process(&mut input, None, None).unwrap() {
            Event::BasicInfo | Event::FrameComplete { .. } => {}
            Event::FrameHeader => {
                assert!(!decoder.frame_header().unwrap().is_preview);
                headers += 1;
            }
            Event::Complete => break,
            Event::NeedMoreInput { .. } => panic!("complete fixture requested more input"),
        }
    }
    assert_eq!(
        headers,
        expected.frames.len(),
        "missing sequential frame events"
    );
    assert_eq!(decoder.scanned_frames().len(), count);
    let targets = decoder.scanned_frames().to_vec();
    for (index, target) in targets.iter().enumerate() {
        assert_eq!(target.index, index);
        assert!(target.file_offset < bytes.len() as u64);
        let offset = decoder.start_new_frame(target.seek_target).unwrap();
        assert!(offset < bytes.len() as u64);
        let mut input = &bytes[offset as usize..];
        loop {
            match decoder.process(&mut input, None, None).unwrap() {
                Event::FrameHeader => {
                    let header = decoder.frame_header().unwrap();
                    assert_eq!(header.duration_ticks, expected.frames[index].duration_ticks);
                    assert_eq!(header.size, expected.frames[index].header.size);
                    break;
                }
                Event::BasicInfo | Event::FrameComplete { .. } => {}
                Event::Complete | Event::NeedMoreInput { .. } => {
                    panic!("published seek target did not reach its frame")
                }
            }
        }
    }
    assert_eq!(
        decoder.scanned_frames().len(),
        count,
        "seeking changed the index"
    );
}

#[test]
fn malformed_trailing_boxes_are_errors_even_after_pixels_were_emitted() {
    for tail in [
        b"\0\0\0\x20xml <x/>".as_slice(),
        b"\0\0\0\x10brobxml \xff\xff\xff\xff",
        b"\0\0\0\x07xml ",
        b"\0\0\0\x08xml",
    ] {
        for allow_partial in [false, true] {
            let mut bytes = container(&raw());
            bytes.extend_from_slice(tail);
            let options = DecodeOptions {
                allow_partial,
                ..Default::default()
            };
            let mut stream = StreamDecoder::new(&options).unwrap();
            let outcome = stream.push(&bytes).and_then(|()| stream.finish());
            assert!(outcome.is_err());
            let frames = stream.take_frames();
            assert_eq!(frames.len(), 1, "earlier frame was lost on tail error");
            assert!(frames[0].complete);
            assert!(stream.push(&[]).is_err());
            assert!(stream.finish().is_err());
        }
    }
}

#[test]
fn truncated_codestreams_are_not_accepted_until_finish_and_partial_is_opt_in() {
    let bytes = raw();
    for end in 0..bytes.len() {
        let mut stream = StreamDecoder::new(&DecodeOptions::default()).unwrap();
        stream.push(&bytes[..end]).unwrap();
        assert!(
            stream.finish().is_err(),
            "accepted prefix {end}/{}",
            bytes.len()
        );
    }
    let bytes = sequence(&[1, 2, 3]);
    let prefix = &bytes[..bytes.len() - 1];
    let options = DecodeOptions {
        allow_partial: true,
        ..Default::default()
    };
    let expected = decode(prefix, &options).unwrap();
    assert!(!expected.complete);
    for chunk in [1, 11, 65536] {
        let actual = decode_reader_with_chunk_size(prefix, &options, chunk).unwrap();
        assert_image(&actual, &expected);
    }
    assert!(decode_reader(b"not JXL".as_slice(), &options).is_err());
}

#[test]
fn input_output_and_frame_limits_are_cumulative_even_when_draining() {
    let bytes = sequence(&[1, 2, 3]);
    let expected = decode(&bytes, &DecodeOptions::default()).unwrap();
    let output_bytes = expected
        .frames
        .iter()
        .map(|f| f.width * f.height)
        .sum::<usize>();
    for (input_limit, output_limit, frame_limit, passes) in [
        (bytes.len(), output_bytes, 3, true),
        (bytes.len() - 1, output_bytes, 3, false),
        (bytes.len(), output_bytes - 1, 3, false),
        (bytes.len(), output_bytes, 2, false),
    ] {
        let mut options = DecodeOptions::default();
        options.limits.max_input_bytes = input_limit;
        options.limits.max_output_bytes = output_limit;
        options.limits.max_frames = frame_limit;
        let mut stream = StreamDecoder::new(&options).unwrap();
        let mut result = Ok(());
        for chunk in bytes.chunks(1) {
            result = stream.push(chunk);
            drop(stream.take_frames());
            if result.is_err() {
                break;
            }
        }
        let result = result.and_then(|()| stream.finish());
        if passes {
            assert!(result.is_ok());
        } else {
            assert!(matches!(result, Err(Error::Limit(_))));
        }
    }
}

#[test]
fn metadata_budget_covers_all_compressed_and_trailing_boxes() {
    let mut bytes = container(&raw());
    let first = vec![b'x'; 8000];
    let last = vec![b'y'; 9000];
    append_box(&mut bytes, *b"brob", &compressed(*b"jumb", &first));
    bytes.extend_from_slice(b"\0\0\0\0brob");
    bytes.extend_from_slice(&compressed(*b"xml ", &last));
    for limit in [0, first.len() + last.len() - 1, first.len() + last.len()] {
        let mut options = DecodeOptions::default();
        options.limits.max_metadata_bytes = limit;
        let result = decode_reader_with_chunk_size(bytes.as_slice(), &options, 3);
        if limit == first.len() + last.len() {
            assert!(result.is_ok());
        } else {
            assert!(matches!(result, Err(Error::Limit(_))));
        }
    }
}

#[test]
fn reader_chunk_size_and_exact_input_limit_are_checked_without_overread() {
    let bytes = raw();
    let mut options = DecodeOptions::default();
    options.limits.max_input_bytes = bytes.len();
    let expected = decode(&bytes, &options).unwrap();
    assert_image(
        &decode_reader_with_chunk_size(bytes.as_slice(), &options, usize::MAX).unwrap(),
        &expected,
    );
    assert!(matches!(
        decode_reader_with_chunk_size(bytes.as_slice(), &options, 0),
        Err(Error::InvalidInput(_))
    ));
    let mut oversized = bytes.clone();
    oversized.extend_from_slice(&[0; 1024]);
    let mut reader = Cursor::new(oversized);
    assert!(matches!(
        decode_reader(&mut reader, &options),
        Err(Error::Limit(_))
    ));
    assert_eq!(reader.position(), bytes.len() as u64 + 1);
    options.limits.max_input_bytes = 0;
    assert!(matches!(
        decode_reader(bytes.as_slice(), &options),
        Err(Error::Limit(_))
    ));
}

#[test]
fn reader_interrupted_retries_and_io_or_callback_errors_are_not_eof() {
    struct Interrupt<R>(R, bool);
    impl<R: Read> Read for Interrupt<R> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.1 = !self.1;
            if self.1 {
                Err(io::ErrorKind::Interrupted.into())
            } else {
                self.0.read(bytes)
            }
        }
    }
    let bytes = container(&raw());
    let options = DecodeOptions {
        allow_partial: true,
        ..Default::default()
    };
    let expected = decode(&bytes, &options).unwrap();
    let actual =
        decode_reader_with_chunk_size(Interrupt(bytes.as_slice(), false), &options, 7).unwrap();
    assert_image(&actual, &expected);
    struct FailAtEnd<'a>(&'a [u8], io::ErrorKind);
    impl Read for FailAtEnd<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() {
                return Err(self.1.into());
            }
            self.0.read(out)
        }
    }
    for kind in [io::ErrorKind::Other, io::ErrorKind::WouldBlock] {
        let mut emitted = 0;
        let result = decode_reader_with(FailAtEnd(&bytes, kind), &options, 1, |_| {
            emitted += 1;
            Ok(())
        });
        assert!(matches!(result, Err(Error::Io(error)) if error.kind() == kind));
        assert_eq!(emitted, 1);
    }
    let mut reader = Cursor::new(&bytes);
    assert!(matches!(
        decode_reader_with(&mut reader, &options, 1, |_| Err(Error::InvalidInput(
            "consumer stopped".into()
        ))),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn color_quantization_and_raw_zero_duration_layers_match_batch_decode() {
    let rgba = Image::new(
        2,
        2,
        PixelLayout::Rgba,
        PixelData::U8(vec![
            77, 90, 111, 0, 200, 10, 15, 128, 1, 2, 3, 255, 91, 52, 36, 127,
        ]),
    )
    .unwrap();
    let bytes = encode(&rgba, &encoding()).unwrap();
    for format in [
        SampleFormat::Native,
        SampleFormat::U8Bits(5),
        SampleFormat::U16,
        SampleFormat::F32,
    ] {
        let options = DecodeOptions {
            format,
            color: ColorOptions {
                output_profile: Some(ColorProfile::parse("linear").unwrap()),
                ..Default::default()
            },
            ..Default::default()
        };
        let expected = decode(&bytes, &options).unwrap();
        assert_image(
            &decode_reader_with_chunk_size(bytes.as_slice(), &options, 1).unwrap(),
            &expected,
        );
    }
    let bytes = sequence(&[0, 2, 0, 3]);
    for coalescing in [false, true] {
        let options = DecodeOptions {
            coalescing,
            ..Default::default()
        };
        let expected = decode(&bytes, &options).unwrap();
        assert_image(
            &decode_reader_with_chunk_size(bytes.as_slice(), &options, 2).unwrap(),
            &expected,
        );
    }
}

#[test]
fn wide_native_integer_streaming_uses_capture_not_float_conversion() {
    for bits in [17, 24, 25, 31] {
        let max = (1u32 << bits) - 1;
        let pixels = PixelData::U32(vec![0, 1, max, max - 1, max / 2, 100003]);
        let image = Image::new(3, 2, PixelLayout::Gray, pixels.clone()).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.bit_depth = BitDepth::new(bits).unwrap();
        let image = image.with_metadata(metadata, vec![]).unwrap();
        let bytes = encode(&image, &encoding()).unwrap();
        let actual =
            decode_reader_with_chunk_size(bytes.as_slice(), &DecodeOptions::default(), 1).unwrap();
        assert!(actual.complete);
        assert_eq!(actual.frames[0].pixels, pixels);
    }
}

// The preview header fixture follows metadata.rs and libjxl size.h. Translated
// header layout: Copyright (c) the JPEG XL Project Authors, BSD-3-Clause; see
// vendor/jxl-rs/LICENSE. No external fixture or codec executable is required.
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
        writer.write(1, 0).unwrap();
        writer.write(1, 1).unwrap();
        writer.write(1, 0).unwrap();
        writer.write(2, 0).unwrap();
        writer.write(6, u64::from(height - 1)).unwrap();
        writer.write(3, 0).unwrap();
        writer.write(2, 0).unwrap();
        writer.write(6, u64::from(width - 1)).unwrap();
        writer.write(1, 0).unwrap();
        metadata.bit_depth.write_header(&mut writer).unwrap();
        writer.write(1, 1).unwrap();
        writer.write(2, 0).unwrap();
        writer.write(1, 0).unwrap();
        metadata.color_encoding.write_header(&mut writer).unwrap();
        metadata.tone_mapping.write_header(&mut writer).unwrap();
        writer.write(2, 0).unwrap();
    } else {
        metadata.write_header(&mut writer).unwrap();
    }
    rs_jxl::metadata::write_default_transform_data(&mut writer).unwrap();
    writer.zero_pad_to_byte().unwrap();
    writer.into_bytes().unwrap()
}

#[test]
fn preview_has_its_own_buffer_and_does_not_increment_display_index() {
    let orient = |image: Image| {
        let mut metadata = image.metadata().clone();
        metadata.orientation = Orientation::Rotate90Clockwise;
        image.with_metadata(metadata, vec![]).unwrap()
    };
    let preview = orient(gray(2, 1, vec![31, 97]));
    let main = orient(gray(3, 2, vec![1, 2, 3, 4, 5, 6]));
    let preview_encoded = encode(&preview, &encoding()).unwrap();
    let main_encoded = encode(&main, &encoding()).unwrap();
    let preview_header = image_header(&preview, None);
    let main_header = image_header(&main, None);
    assert!(preview_encoded.starts_with(&preview_header));
    assert!(main_encoded.starts_with(&main_header));
    let mut bytes = image_header(&main, Some((2, 1)));
    bytes.extend_from_slice(&preview_encoded[preview_header.len()..]);
    bytes.extend_from_slice(&main_encoded[main_header.len()..]);
    for include_preview in [false, true] {
        let options = DecodeOptions {
            include_preview,
            ..Default::default()
        };
        let expected = decode(&bytes, &options).unwrap();
        let actual = decode_reader_with_chunk_size(bytes.as_slice(), &options, 1).unwrap();
        assert_image(&actual, &expected);
        assert_eq!(actual.frames[0].index, 0);
        let mut stream = StreamDecoder::new(&options).unwrap();
        stream.push(&bytes).unwrap();
        let summary = stream.finish().unwrap();
        assert_eq!(summary.frame_count, 1);
        assert_eq!(summary.has_preview, include_preview);
    }
}
