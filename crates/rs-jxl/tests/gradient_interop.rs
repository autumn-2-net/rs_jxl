//! Independent frozen decoder fixtures plus the repaired encoder's low-effort path.
use std::io::Cursor;

use rs_jxl::decode::{DecodeOptions, DecodedImage, Decoder, decode, decode_reader_with_chunk_size};
use rs_jxl::scheduler::{CandidateParams, adapter::project_native};
use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout, encode};

#[path = "fixtures/gradient_streams.rs"]
mod fixtures;

fn unhex(input: &str) -> Vec<u8> {
    let text: Vec<_> = input.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    assert_eq!(text.len() % 2, 0);
    text.chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn image() -> Image {
    let pattern = [0., 0., 0., 0., 0., 0., 0., 2., 1e-10f32];
    Image::new(
        16,
        11,
        PixelLayout::Rgb,
        PixelData::F32((0..528).map(|i| pattern[i % pattern.len()]).collect()),
    )
    .unwrap()
}

fn check(decoded: &DecodedImage, source: &Image) {
    assert!(decoded.complete);
    assert_eq!(decoded.frames.len(), 1);
    let frame = &decoded.frames[0];
    assert_eq!(
        (frame.width, frame.height),
        (source.width() as usize, source.height() as usize)
    );
    assert_eq!(frame.layout, source.layout());
    assert_eq!(&frame.pixels, source.pixels());
}

#[test]
fn official_gradient_float_fixture_decodes_without_a_compatibility_switch() {
    let bytes = unhex(fixtures::OFFICIAL);
    for threads in [1, 4] {
        let options = DecodeOptions {
            threads,
            ..Default::default()
        };
        assert!(!options.legacy_gradient);
        check(&decode(&bytes, &options).unwrap(), &image());
        for chunk in [1, 7, 61] {
            check(
                &decode_reader_with_chunk_size(Cursor::new(&bytes), &options, chunk).unwrap(),
                &image(),
            );
        }
    }
}

#[test]
fn legacy_recovery_is_explicit_for_ans_and_prefix_streams() {
    for fixture in [fixtures::LEGACY_ANS, fixtures::LEGACY_PREFIX] {
        let bytes = unhex(fixture);
        for threads in [1, 4] {
            let options = DecodeOptions {
                threads,
                legacy_gradient: true,
                ..Default::default()
            };
            check(&decode(&bytes, &options).unwrap(), &image());
            check(
                &decode_reader_with_chunk_size(Cursor::new(&bytes), &options, 13).unwrap(),
                &image(),
            );
        }
    }
    assert!(decode(&unhex(fixtures::LEGACY_ANS), &DecodeOptions::default()).is_err());
    // Prefix streams can successfully return wrong samples, so an ANS-error
    // retry is not a safe automatic format detector.
    if let Ok(decoded) = decode(&unhex(fixtures::LEGACY_PREFIX), &DecodeOptions::default()) {
        assert_ne!(&decoded.frames[0].pixels, image().pixels());
    }
}

#[test]
fn reusable_decoder_does_not_leak_gradient_interpretation_across_calls() {
    let mut decoder = Decoder::new(DecodeOptions::default()).unwrap();
    for (bytes, legacy_gradient) in [
        (fixtures::OFFICIAL, false),
        (fixtures::LEGACY_ANS, true),
        (fixtures::OFFICIAL, false),
        (fixtures::LEGACY_PREFIX, true),
    ] {
        decoder
            .set_options(DecodeOptions {
                legacy_gradient,
                ..Default::default()
            })
            .unwrap();
        check(&decoder.decode(&unhex(bytes)).unwrap(), &image());
    }
}

#[test]
fn corrected_low_efforts_preserve_positive_and_negative_float_words() {
    for layout in [
        PixelLayout::Gray,
        PixelLayout::GrayAlpha,
        PixelLayout::Rgb,
        PixelLayout::Rgba,
    ] {
        for pattern in [
            [0., 2., 0., 2., 0., 1e-10f32],
            [-0., -2., 0., -2., 1e-10, -0.],
        ] {
            let source = Image::new(
                17,
                33,
                layout,
                PixelData::F32(
                    (0..17 * 33 * layout.channels())
                        .map(|i| pattern[i % pattern.len()])
                        .collect(),
                ),
            )
            .unwrap();
            for effort in [1, 2] {
                let params = CandidateParams::default().backend(effort).unwrap();
                let options = project_native(&params, &source, &EncodeOptions::default())
                    .unwrap()
                    .exact_options()
                    .unwrap();
                let bytes = encode(&source, &options).unwrap();
                check(&decode(&bytes, &DecodeOptions::default()).unwrap(), &source);
            }
        }
    }
}
