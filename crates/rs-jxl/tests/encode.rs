use rs_jxl::{EncodeOptions, Image, PixelData, PixelLayout, encode};

const LAYOUTS: [PixelLayout; 4] = [
    PixelLayout::Gray,
    PixelLayout::GrayAlpha,
    PixelLayout::Rgb,
    PixelLayout::Rgba,
];

#[test]
fn frames_validate_references_and_animation_without_partial_output() {
    use rs_jxl::{
        encode_frames, encode_frames_to,
        frame::{BlendInfo, Frame, FrameOptions},
        metadata::Animation,
    };
    let image = Image::new(2, 2, PixelLayout::Rgba, PixelData::U8(vec![0; 16])).unwrap();
    let crop = Image::new(1, 1, PixelLayout::Rgba, PixelData::U8(vec![1, 2, 3, 0])).unwrap();
    let options = EncodeOptions::default();
    assert!(encode_frames(&image, &[], &options).is_err());
    let invalid = Frame {
        image: &crop,
        options: FrameOptions {
            blend: BlendInfo {
                source: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    };
    let mut sink = vec![1, 2, 3];
    assert!(encode_frames_to(&image, &[invalid], &options, &mut sink).is_err());
    assert_eq!(sink, vec![1, 2, 3]);
    for bad in [
        FrameOptions {
            duration: 1,
            ..Default::default()
        },
        FrameOptions {
            reference_only: true,
            ..Default::default()
        },
        FrameOptions {
            save_as_reference: 1,
            ..Default::default()
        },
        FrameOptions {
            save_before_color_transform: true,
            ..Default::default()
        },
    ] {
        assert!(
            encode_frames(
                &image,
                &[Frame {
                    image: &image,
                    options: bad
                }],
                &options
            )
            .is_err()
        );
    }
    for animation in [
        Animation {
            ticks_per_second_numerator: 0,
            ..Default::default()
        },
        Animation {
            ticks_per_second_denominator: 0,
            ..Default::default()
        },
    ] {
        assert!(animation.validate().is_err());
    }
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    let canvas = image.clone().with_metadata(metadata, vec![]).unwrap();
    let first = Frame {
        image: &image,
        options: FrameOptions {
            duration: 1,
            save_as_reference: 1,
            ..Default::default()
        },
    };
    let next = Frame {
        image: &crop,
        options: FrameOptions {
            duration: 1,
            blend: BlendInfo {
                source: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    };
    assert!(
        !encode_frames(&canvas, &[first, next], &options)
            .unwrap()
            .is_empty()
    );
}

fn channels(layout: PixelLayout) -> usize {
    match layout {
        PixelLayout::Gray => 1,
        PixelLayout::GrayAlpha => 2,
        PixelLayout::Rgb => 3,
        PixelLayout::Rgba => 4,
    }
}

#[test]
fn image_preserves_dimensions_layout_depth_and_samples() {
    for layout in LAYOUTS {
        let count = 4 * channels(layout);
        let samples: Vec<u8> = (0..count).map(|i| (i * 37 + 11) as u8).collect();
        let image = Image::new(2, 2, layout, PixelData::U8(samples.clone())).unwrap();
        assert_eq!((image.width(), image.height()), (2, 2));
        assert_eq!(image.layout(), layout);
        assert_eq!(image.bits(), 8);
        match image.pixels() {
            PixelData::U8(actual) => assert_eq!(actual, &samples),
            _ => panic!("8-bit input changed sample representation"),
        }

        let values = [0, 1, 255, 256, 32767, 32768, 65280, 65535];
        let samples: Vec<u16> = (0..count).map(|i| values[i % values.len()]).collect();
        let image = Image::new(2, 2, layout, PixelData::U16(samples.clone())).unwrap();
        assert_eq!((image.width(), image.height()), (2, 2));
        assert_eq!(image.layout(), layout);
        assert_eq!(image.bits(), 16);
        match image.pixels() {
            PixelData::U16(actual) => assert_eq!(actual, &samples),
            _ => panic!("16-bit input changed sample representation"),
        }
    }
}

#[test]
fn zero_dimensions_are_rejected() {
    for layout in LAYOUTS {
        for (width, height) in [(0, 0), (0, 1), (1, 0)] {
            for pixels in [PixelData::U8(Vec::new()), PixelData::U16(Vec::new())] {
                assert!(
                    Image::new(width, height, layout, pixels).is_err(),
                    "accepted zero dimension: {width}x{height} {layout:?}"
                );
            }
        }
    }
}

#[test]
fn sample_count_must_match_dimensions_and_all_channels() {
    for layout in LAYOUTS {
        let expected = 2 * 3 * channels(layout);
        for actual in [0, 1, expected - 1, expected + 1] {
            for pixels in [
                PixelData::U8(vec![0; actual]),
                PixelData::U16(vec![0; actual]),
            ] {
                assert!(
                    Image::new(2, 3, layout, pixels).is_err(),
                    "accepted {actual} samples, expected {expected}, layout {layout:?}"
                );
            }
        }
    }
}

#[test]
fn oversized_dimension_products_are_checked_without_allocating_pixels() {
    // Include an RGBA product that wraps exactly to zero on 32/64-bit targets,
    // so an empty input cannot accidentally pass an unchecked length comparison.
    for layout in LAYOUTS {
        for (width, height) in [
            (1 << 31, 1 << 31),
            (u32::MAX, u32::MAX),
            (u32::MAX, 1),
            (1, u32::MAX),
        ] {
            for pixels in [PixelData::U8(Vec::new()), PixelData::U16(Vec::new())] {
                assert!(
                    Image::new(width, height, layout, pixels).is_err(),
                    "accepted huge dimensions with no pixels: {width}x{height} {layout:?}"
                );
            }
        }
    }
}

#[test]
fn invalid_thread_counts_are_rejected() {
    let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![37])).unwrap();
    for threads in [0, 257, usize::MAX] {
        let options = EncodeOptions {
            threads,
            ..Default::default()
        };
        assert!(
            encode(&image, &options).is_err(),
            "accepted invalid thread count {threads}"
        );
    }
}

#[test]
fn pixel_count_limit_is_checked_before_encoding() {
    let image = Image::new(2, 2, PixelLayout::Gray, PixelData::U8(vec![0, 1, 128, 255])).unwrap();
    for limit in [0, 3] {
        let options = EncodeOptions {
            max_pixels: limit,
            ..Default::default()
        };
        assert!(encode(&image, &options).is_err());
    }
    let options = EncodeOptions {
        max_pixels: 4,
        ..Default::default()
    };
    assert!(encode(&image, &options).is_ok());
}

#[test]
fn max_output_bytes_is_a_hard_inclusive_limit() {
    let images = [
        Image::new(2, 2, PixelLayout::Gray, PixelData::U8(vec![0, 1, 128, 255])).unwrap(),
        Image::new(
            2,
            2,
            PixelLayout::Rgba,
            PixelData::U16(vec![
                1, 255, 256, 0, 65535, 32768, 32767, 1, 65280, 257, 65534, 32768, 0, 1, 65535,
                65535,
            ]),
        )
        .unwrap(),
        Image::new(
            257,
            1,
            PixelLayout::Gray,
            PixelData::U8((0..257).map(|i| (i * 37) as u8).collect()),
        )
        .unwrap(),
    ];
    for image in images {
        let options = EncodeOptions {
            threads: 2,
            max_output_bytes: 1024 * 1024,
            ..Default::default()
        };
        let bytes = encode(&image, &options).expect("small valid image must encode");
        assert!(
            bytes.len() > 1,
            "encoder returned an empty or incomplete stream"
        );
        assert!(bytes.len() <= options.max_output_bytes);

        for limit in [0, 1, bytes.len() - 1] {
            let options = EncodeOptions {
                threads: 2,
                max_output_bytes: limit,
                ..Default::default()
            };
            assert!(
                encode(&image, &options).is_err(),
                "encoding succeeded with insufficient output budget {limit}"
            );
        }

        let exact_options = EncodeOptions {
            threads: 2,
            max_output_bytes: bytes.len(),
            ..Default::default()
        };
        let exact = encode(&image, &exact_options).expect("exact output budget must be accepted");
        assert_eq!(exact.len(), bytes.len());
    }
}

fn assert_level_10_container(bytes: &[u8], context: &str) {
    let mut remaining = bytes;
    let mut boxes = Vec::new();
    while !remaining.is_empty() {
        assert!(remaining.len() >= 8, "{context}: truncated box header");
        let declared = u32::from_be_bytes(remaining[..4].try_into().unwrap());
        let (length, header_length) = if declared == 1 {
            assert!(
                remaining.len() >= 16,
                "{context}: truncated extended box length"
            );
            let extended = u64::from_be_bytes(remaining[8..16].try_into().unwrap());
            (
                usize::try_from(extended).expect("box length fits usize"),
                16,
            )
        } else {
            (usize::try_from(declared).unwrap(), 8)
        };
        assert!(
            length >= header_length,
            "{context}: box must declare a complete, explicit length"
        );
        assert!(
            length <= remaining.len(),
            "{context}: box extends beyond the output buffer"
        );
        boxes.push((&remaining[4..8], &remaining[header_length..length]));
        remaining = &remaining[length..];
    }
    // Walking declared lengths through EOF also rejects gaps and trailing bytes.
    assert_eq!(
        boxes.len(),
        4,
        "{context}: expected signature/ftyp/jxll/jxlc"
    );
    assert_eq!(
        &bytes[..4],
        &12_u32.to_be_bytes(),
        "{context}: signature box must use its fixed 12-byte size"
    );
    for ((kind, _), expected) in boxes.iter().zip([b"JXL ", b"ftyp", b"jxll", b"jxlc"]) {
        assert_eq!(*kind, expected.as_slice(), "{context}: box type/order");
    }
    assert_eq!(
        boxes[0].1,
        &[0x0d, 0x0a, 0x87, 0x0a],
        "{context}: signature payload"
    );
    let file_type = boxes[1].1;
    assert!(
        file_type.len() >= 12,
        "{context}: ftyp needs a compatible brand"
    );
    assert_eq!(
        (file_type.len() - 8) % 4,
        0,
        "{context}: truncated ftyp brand"
    );
    assert_eq!(&file_type[..4], b"jxl ", "{context}: major brand");
    assert_eq!(&file_type[4..8], &[0; 4], "{context}: minor version");
    assert!(
        file_type[8..].chunks_exact(4).any(|brand| brand == b"jxl "),
        "{context}: JXL compatibility brand missing"
    );
    assert_eq!(boxes[2].1, &[10], "{context}: jxll must declare Level 10");
    let codestream = boxes[3].1;
    assert!(codestream.len() > 2, "{context}: empty codestream box");
    assert!(
        codestream.starts_with(&[0xff, 0x0a]),
        "{context}: codestream signature"
    );
}

#[test]
fn sixteen_bit_output_declares_level_10_in_a_complete_container() {
    for layout in LAYOUTS {
        for width in [1, 257] {
            // Even the all-zero 1x1 gray image needs Level 10 when typed as U16.
            let samples = (0..width as usize * channels(layout))
                .map(|index| (index as u16).wrapping_mul(257))
                .collect();
            let image = Image::new(width, 1, layout, PixelData::U16(samples)).unwrap();
            for threads in [1, 2] {
                let context = format!("{width}x1 {layout:?} 16-bit, threads={threads}");
                let options = EncodeOptions {
                    threads,
                    ..Default::default()
                };
                let bytes = encode(&image, &options)
                    .unwrap_or_else(|error| panic!("{context}: encode: {error:?}"));
                assert_level_10_container(&bytes, &context);
            }
        }
    }
}

#[test]
fn eight_bit_level_follows_the_declared_modular_buffer_contract() {
    for layout in LAYOUTS {
        for width in [1, 257] {
            let image = Image::new(
                width,
                1,
                layout,
                PixelData::U8(vec![255; width as usize * channels(layout)]),
            )
            .unwrap();
            let bytes = encode(&image, &EncodeOptions::default()).unwrap();
            assert_level_10_container(&bytes, "default 32-bit Modular buffer contract");
            let mut metadata = image.metadata().clone();
            metadata.modular_16_bit_buffer_sufficient = true;
            let image = image.with_metadata(metadata, vec![]).unwrap();
            let bytes = encode(&image, &EncodeOptions::default()).unwrap();
            assert!(bytes.len() > 2);
            assert!(
                bytes.starts_with(&[0xff, 0x0a]),
                "{width}x1 {layout:?}: expected a bare 8-bit codestream"
            );
        }
    }
}
