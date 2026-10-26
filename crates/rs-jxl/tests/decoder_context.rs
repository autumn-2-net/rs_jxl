//! Reusable caller-owned resources must not retain any per-input decoder state.
//! Fixtures are encoded in memory.

use rs_jxl::decode::{
    AlphaMode, DecodeOptions, DecodedFrame, DecodedImage, Decoder, SampleFormat, decode,
    decode_frame, reconstruct_jpeg,
};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::metadata::Animation;
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{self, Reference};
use rs_jxl::{
    EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode, encode_frames,
    icc, jpeg,
};

// The shared fixture module's own tests resolve crate::jpeg and crate::icc.
#[allow(dead_code)]
#[path = "support/jpeg_fixtures.rs"]
mod jpeg_fixtures;

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

fn gray(width: u32, height: u32, seed: u8) -> Image {
    let pixels = (0..width * height)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
        .collect();
    let image = Image::new(width, height, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
    let mut metadata = image.metadata().clone();
    metadata.modular_16_bit_buffer_sufficient = true;
    image.with_metadata(metadata, vec![]).unwrap()
}

fn encoded(image: &Image) -> Vec<u8> {
    encode(image, &encoding()).unwrap()
}

fn animation() -> Vec<u8> {
    let images: Vec<_> = (0..3)
        .map(|i| {
            let image = gray(8, 6, 17);
            let PixelData::U8(mut pixels) = image.pixels().clone() else {
                unreachable!()
            };
            pixels[i * 3] ^= 1;
            let mut metadata = image.metadata().clone();
            metadata.animation = Some(Animation {
                ticks_per_second_numerator: 60,
                ..Default::default()
            });
            Image::new(8, 6, PixelLayout::Gray, PixelData::U8(pixels))
                .unwrap()
                .with_metadata(metadata, vec![])
                .unwrap()
        })
        .collect();
    let mut plan = multiframe::plan(&images, &[1, 2, 3], Reference::Previous).unwrap();
    for (index, frame) in plan.iter_mut().enumerate() {
        frame.options.name = format!("context-frame-{index}");
    }
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
    assert_eq!(actual.info.orientation, expected.info.orientation);
    assert_eq!(actual.info.extra_channels, expected.info.extra_channels);
    assert_eq!(actual.info.animation, expected.info.animation);
    assert_eq!(
        actual.info.uses_original_profile,
        expected.info.uses_original_profile
    );
    assert_eq!(actual.info.preview_size, expected.info.preview_size);
    let a = &actual.info.tone_mapping;
    let b = &expected.info.tone_mapping;
    assert_eq!(a.intensity_target.to_bits(), b.intensity_target.to_bits());
    assert_eq!(a.min_nits.to_bits(), b.min_nits.to_bits());
    assert_eq!(a.relative_to_max_display, b.relative_to_max_display);
    assert_eq!(a.linear_below.to_bits(), b.linear_below.to_bits());
    assert_eq!(actual.embedded_profile, expected.embedded_profile);
    assert_eq!(actual.output_profile, expected.output_profile);
    assert_eq!(
        actual.output_intensity_target.to_bits(),
        expected.output_intensity_target.to_bits()
    );
    assert_eq!(actual.complete, expected.complete);
    assert_eq!(actual.bytes_read, expected.bytes_read);
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

fn assert_error_matches<T>(actual: rs_jxl::Result<T>, expected: rs_jxl::Result<T>) {
    let actual = actual.err().expect("reused decoder must reject this input");
    let expected = expected
        .err()
        .expect("fresh decoder must reject this input");
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

#[test]
fn reused_context_matches_fresh_pixels_across_formats_and_sizes() {
    let images = [
        gray(17, 9, 3),
        Image::new(
            2,
            1,
            PixelLayout::Rgba,
            PixelData::U8(vec![17, 93, 241, 0, 99, 33, 4, 255]),
        )
        .unwrap(),
        Image::new(
            3,
            2,
            PixelLayout::Gray,
            PixelData::U16(vec![0, 1, 32767, 32768, 65534, 65535]),
        )
        .unwrap(),
        Image::new(
            3,
            1,
            PixelLayout::Gray,
            PixelData::U32(vec![0, 1 << 24, (1 << 31) - 1]),
        )
        .unwrap(),
        Image::new(
            2,
            2,
            PixelLayout::Gray,
            PixelData::F32(
                [0x80000000, 0x3f800000, 0x7fc12345, 1]
                    .map(f32::from_bits)
                    .to_vec(),
            ),
        )
        .unwrap(),
    ];
    let inputs: Vec<_> = images.iter().map(encoded).collect();
    for threads in [1, 4] {
        let options = DecodeOptions {
            threads,
            ..Default::default()
        };
        let mut decoder = Decoder::new(options.clone()).unwrap();
        assert_eq!(decoder.options().threads, threads);
        assert_eq!(decoder.options().format, SampleFormat::Native);
        for _ in 0..2 {
            for (input, source) in inputs.iter().zip(&images) {
                let fresh = decode(input, &options).unwrap();
                let reused = decoder.decode(input).unwrap();
                assert!(reused.complete);
                assert_eq!(reused.frames.len(), 1);
                assert_eq!(&reused.frames[0].pixels, source.pixels());
                assert_image(&reused, &fresh);
            }
        }
    }
}

#[test]
fn frame_selection_and_animation_references_are_fresh_for_every_input() {
    let sequence = animation();
    let still = encoded(&gray(2, 2, 71));
    for threads in [1, 4] {
        for coalescing in [false, true] {
            let options = DecodeOptions {
                threads,
                coalescing,
                ..Default::default()
            };
            let mut decoder = Decoder::new(options.clone()).unwrap();
            let expected = decode(&sequence, &options).unwrap();
            assert!(expected.complete);
            assert_eq!(expected.frames.len(), 3);
            for index in [2, 0, 1, 2] {
                let selected = decoder.decode_frame(&sequence, index).unwrap();
                assert!(!selected.complete);
                assert_eq!(selected.frames.len(), 1);
                assert_image(
                    &selected,
                    &decode_frame(&sequence, index, &options).unwrap(),
                );
                assert_image(
                    &decoder.decode(&still).unwrap(),
                    &decode(&still, &options).unwrap(),
                );
                assert_image(&decoder.decode(&sequence).unwrap(), &expected);
            }
            assert_error_matches(
                decoder.decode_frame(&sequence, 99),
                decode_frame(&sequence, 99, &options),
            );
            assert_image(&decoder.decode(&sequence).unwrap(), &expected);
        }
    }
}

#[test]
fn malformed_and_truncated_calls_do_not_poison_later_decodes() {
    let sequence = animation();
    let still = encoded(&gray(3, 2, 19));
    for threads in [1, 4] {
        let options = DecodeOptions {
            threads,
            ..Default::default()
        };
        let mut decoder = Decoder::new(options.clone()).unwrap();
        let expected = decode(&still, &options).unwrap();
        for bad in [
            b"not a JXL stream".as_slice(),
            &[],
            &sequence[..1],
            &sequence[..sequence.len() / 2],
        ] {
            assert_error_matches(decoder.decode(bad), decode(bad, &options));
            assert_image(&decoder.decode(&still).unwrap(), &expected);
            assert_error_matches(decoder.decode_frame(bad, 0), decode_frame(bad, 0, &options));
            assert_image(&decoder.decode(&still).unwrap(), &expected);
        }
    }
}

#[test]
fn limits_are_enforced_per_input_and_recover_after_failure() {
    let small = encoded(&gray(2, 2, 11));
    let large = encoded(&gray(8, 8, 97));
    let sequence = animation();
    for threads in [1, 4] {
        for resource in 0..5 {
            let mut options = DecodeOptions {
                threads,
                ..Default::default()
            };
            let mut oversized = small.clone();
            oversized.push(0);
            let bad = match resource {
                0 => {
                    options.limits.max_input_bytes = small.len();
                    &oversized
                }
                1 => {
                    options.limits.max_pixels = 4;
                    &large
                }
                2 => {
                    // The backend budgets padded rows and three color planes.
                    options.limits.max_samples = 128;
                    &large
                }
                3 => {
                    options.limits.max_output_bytes = 32;
                    &large
                }
                _ => {
                    options.limits.max_frames = 1;
                    &sequence
                }
            };
            let mut decoder = Decoder::new(options.clone()).unwrap();
            let expected = decode(&small, &options).unwrap();
            for _ in 0..2 {
                assert!(matches!(decoder.decode(bad), Err(Error::Limit(_))));
                assert_error_matches(decoder.decode(bad), decode(bad, &options));
                assert_image(&decoder.decode(&small).unwrap(), &expected);
            }
        }
        let mut options = DecodeOptions {
            threads,
            ..Default::default()
        };
        options.limits.max_frames = 0;
        let mut decoder = Decoder::new(options.clone()).unwrap();
        assert!(matches!(
            decoder.decode_frame(&small, 0),
            Err(Error::Limit("requested frame index"))
        ));
        assert_error_matches(decoder.decode(&small), decode(&small, &options));
        assert!(matches!(
            decoder.reconstruct_jpeg(&small),
            Err(Error::Limit("JPEG frame count"))
        ));
    }
}

#[test]
fn constructor_rejects_invalid_options_and_free_function_error_order_is_unchanged() {
    let mut cases = Vec::new();
    for threads in [0, 257, usize::MAX] {
        cases.push(DecodeOptions {
            threads,
            ..Default::default()
        });
    }
    for format in [
        SampleFormat::U8Bits(0),
        SampleFormat::U8Bits(9),
        SampleFormat::U16Bits(0),
        SampleFormat::U16Bits(17),
    ] {
        cases.push(DecodeOptions {
            format,
            ..Default::default()
        });
    }
    for downsampling in [0, 3, 16] {
        cases.push(DecodeOptions {
            downsampling,
            ..Default::default()
        });
    }
    cases.push(DecodeOptions {
        coalescing: false,
        alpha_mode: AlphaMode::Premultiplied,
        ..Default::default()
    });
    for options in cases {
        assert!(matches!(Decoder::new(options), Err(Error::InvalidInput(_))));
    }
    let mut options = DecodeOptions {
        threads: 0,
        ..Default::default()
    };
    options.limits.max_input_bytes = 0;
    options.limits.max_frames = 0;
    assert!(matches!(
        decode(&[0], &options),
        Err(Error::Limit("JXL input bytes"))
    ));
    assert!(matches!(
        reconstruct_jpeg(&[0], &options),
        Err(Error::Limit("JXL input bytes"))
    ));
    assert!(matches!(
        decode_frame(&[0], 0, &options),
        Err(Error::Limit("requested frame index"))
    ));
    assert!(matches!(decode(&[], &options), Err(Error::InvalidInput(_))));
    assert!(matches!(
        reconstruct_jpeg(&[], &options),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn one_context_alternates_exact_jpeg_reconstruction_and_pixels_after_errors() {
    let fixtures: Vec<_> = ["gray_seq_17x9", "ycbcr420_seq_17x19_rst2"]
        .into_iter()
        .map(|name| {
            let original = jpeg_fixtures::fixture(name).unwrap();
            let bytes = jpeg::transcode(&original, &jpeg::TranscodeOptions::default()).unwrap();
            (bytes, original)
        })
        .collect();
    let pixels = encoded(&gray(4, 3, 99));
    for threads in [1, 4] {
        let options = DecodeOptions {
            threads,
            ..Default::default()
        };
        let mut decoder = Decoder::new(options.clone()).unwrap();
        let expected_pixels = decode(&pixels, &options).unwrap();
        for _ in 0..2 {
            for (bytes, original) in &fixtures {
                assert_error_matches(
                    decoder.reconstruct_jpeg(&pixels),
                    reconstruct_jpeg(&pixels, &options),
                );
                let truncated = &bytes[..bytes.len() / 2];
                assert_error_matches(
                    decoder.reconstruct_jpeg(truncated),
                    reconstruct_jpeg(truncated, &options),
                );
                assert_eq!(decoder.reconstruct_jpeg(bytes).unwrap(), *original);
                assert_eq!(
                    decoder.reconstruct_jpeg(bytes).unwrap(),
                    reconstruct_jpeg(bytes, &options).unwrap()
                );
                assert_image(&decoder.decode(&pixels).unwrap(), &expected_pixels);
                assert_image(
                    &decoder.decode(bytes).unwrap(),
                    &decode(bytes, &options).unwrap(),
                );
            }
        }
    }
}
