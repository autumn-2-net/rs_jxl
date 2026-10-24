//! End-to-end tests of the maintained decoder API against our own encoder.
//! Fixtures are deterministic and in memory; no external codec or filesystem.
//! Float comparisons use IEEE words, including signaling NaNs and signed zero.

use std::io::Write;

use rs_jxl::container::ChunkedWriter;
use rs_jxl::decode::{
    DecodeOptions, DecodedFrame, DecodedImage, ExtraChannel, JxlBitDepth, SampleFormat, decode,
    decode_frame,
};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{Frame, FrameOptions};
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType, Orientation};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::multiframe::{self, Reference};
use rs_jxl::{
    EncodeOptions, Error, Image, ModularOptions, PixelData, PixelLayout, encode, encode_frames,
};

const LAYOUTS: [PixelLayout; 4] = [
    PixelLayout::Gray,
    PixelLayout::GrayAlpha,
    PixelLayout::Rgb,
    PixelLayout::Rgba,
];
const REFERENCES: [Reference; 4] = [
    Reference::Previous,
    Reference::First,
    Reference::SmallestCrop,
    Reference::Full,
];

#[test]
fn cms_quantizes_both_primary_alpha_copies_consistently() {
    let source = Image::new(
        2,
        1,
        PixelLayout::Rgba,
        PixelData::F32(vec![0.25, 0.5, 0.75, 0.5, 0.1, 0.2, 0.3, 0.0]),
    )
    .unwrap();
    let bytes = encoded(&source);
    for format in [SampleFormat::U8, SampleFormat::U16, SampleFormat::U8Bits(5)] {
        let result = decode(
            &bytes,
            &DecodeOptions {
                format,
                color: rs_jxl::decode::ColorOptions {
                    output_profile: Some(rs_jxl::decode::ColorProfile::parse("sRGB").unwrap()),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let frame = only_frame(&result);
        assert_pixels(
            &alpha_plane(&frame.pixels, frame.layout),
            &frame.extra_planes[0],
        );
    }
}

#[test]
fn raw_layers_reject_transformations_that_invalidate_blend_headers() {
    let bytes = encode_sequence(
        &sequence_images(true),
        &[0, 1, 0, 1, 0],
        Reference::Previous,
    );
    let mut options = DecodeOptions {
        coalescing: false,
        ..Default::default()
    };
    options.color.output_profile = Some(rs_jxl::decode::ColorProfile::parse("linear").unwrap());
    assert!(matches!(
        decode(&bytes, &options),
        Err(Error::InvalidInput(_))
    ));
    options.color = Default::default();
    options.alpha_mode = rs_jxl::decode::AlphaMode::Premultiplied;
    assert!(matches!(
        decode(&bytes, &options),
        Err(Error::InvalidInput(_))
    ));
    options.alpha_mode = rs_jxl::decode::AlphaMode::AsIs;
    options.color.background = Some([1.0; 3]);
    assert!(matches!(
        decode(&bytes, &options),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn raw_spot_layers_require_unrendered_color_planes() {
    let source = integer_image(3, 2, PixelLayout::Rgb, false);
    let mut metadata = source.metadata().clone();
    let mut spot = ExtraChannelInfo::new(ExtraChannelType::SpotColor, 8).unwrap();
    spot.spot_color = [1.0; 4];
    metadata.extra_channels.push(spot);
    let source = source
        .with_metadata(metadata, vec![PixelData::U8(vec![127; 6])])
        .unwrap();
    let bytes = encoded(&source);
    let mut options = DecodeOptions {
        coalescing: false,
        ..Default::default()
    };
    assert!(matches!(
        decode(&bytes, &options),
        Err(Error::InvalidInput(_))
    ));
    options.render_spot_colors = false;
    let decoded = decode(&bytes, &options).unwrap();
    assert_pixels(&only_frame(&decoded).pixels, source.pixels());
    assert_eq!(only_frame(&decoded).extra_planes, source.extra_planes());
}

#[test]
fn inspection_and_decode_negotiate_the_same_pixel_profile() {
    let bytes = encoded(&integer_image(3, 2, PixelLayout::Rgb, false));
    for format in [
        SampleFormat::Native,
        SampleFormat::U8,
        SampleFormat::U16,
        SampleFormat::F32,
    ] {
        let options = DecodeOptions {
            format,
            ..Default::default()
        };
        let inspection = rs_jxl::decode::inspect(&bytes, &options).unwrap();
        let decoded = decode(&bytes, &options).unwrap();
        assert_eq!(inspection.output_profile, decoded.output_profile);
    }
}

#[test]
fn raw_layers_return_every_crop_without_applying_blend() {
    let images = sequence_images(true);
    let durations = [0, 2, 0, 3, 0];
    for reference in REFERENCES {
        let plan = multiframe::plan(&images, &durations, reference).unwrap();
        let frames: Vec<_> = plan.iter().map(|p| p.as_frame()).collect();
        let bytes = encode_frames(&images[0], &frames, &encode_options()).unwrap();
        let options = DecodeOptions {
            coalescing: false,
            ..Default::default()
        };
        let decoded = decode(&bytes, &options).unwrap();
        assert_eq!(decoded.frames.len(), plan.len());
        for (index, (actual, expected)) in decoded.frames.iter().zip(&plan).enumerate() {
            assert_eq!(actual.index, index);
            assert!(actual.complete);
            assert!(!actual.header.is_preview);
            assert_eq!(
                actual.header.layer_info.crop_origin,
                (
                    expected.options.origin.0 as i64,
                    expected.options.origin.1 as i64
                )
            );
            assert_eq!(
                (actual.width, actual.height),
                (
                    expected.image.width() as usize,
                    expected.image.height() as usize
                )
            );
            assert_eq!(actual.duration_ticks, durations[index]);
            assert_pixels(&actual.pixels, expected.image.pixels());
            assert_pixels(&actual.extra_planes[1], &expected.image.extra_planes()[0]);
            let selected = decode_frame(&bytes, index, &options).unwrap();
            assert_pixels(&selected.frames[0].pixels, expected.image.pixels());
        }
        let scan = rs_jxl::decode::inspect(&bytes, &options).unwrap();
        assert_eq!(scan.frames.len(), plan.len());
        for (header, frame) in scan.frames.iter().zip(&decoded.frames) {
            assert_eq!(header.size, frame.header.size);
            assert_eq!(header.duration_ticks, frame.duration_ticks);
        }
    }
}

#[test]
fn inspection_skips_pixels_and_retains_container_metadata() {
    let bytes = encoded(&integer_image(11, 7, PixelLayout::Rgb, false));
    let options = DecodeOptions::default();
    let info = rs_jxl::decode::inspect(&bytes, &options).unwrap();
    assert_eq!(info.info.size, (11, 7));
    assert_eq!(info.frames.len(), 1);
    assert_eq!(info.frames[0].size, (11, 7));
    assert!(info.preview.is_none());
    assert!(info.bytes_read <= bytes.len());
    // An inspection does not allocate output samples.
    let mut tight = options;
    tight.limits.max_output_bytes = 0;
    assert!(rs_jxl::decode::inspect(&bytes, &tight).is_ok());
    assert!(decode(&bytes, &tight).is_err());
    assert!(rs_jxl::decode::inspect(&bytes[..bytes.len() - 1], &tight).is_err());
}

#[test]
fn requested_integer_bit_depths_are_explicit_and_checked() {
    let image = Image::new(3, 1, PixelLayout::Gray, PixelData::U8(vec![0, 128, 255])).unwrap();
    let bytes = encoded(&image);
    for bits in 1..=16 {
        let format = if bits <= 8 {
            SampleFormat::U8Bits(bits)
        } else {
            SampleFormat::U16Bits(bits)
        };
        let result = decode(
            &bytes,
            &DecodeOptions {
                format,
                ..Default::default()
            },
        )
        .unwrap();
        let max = (1u32 << bits) - 1;
        let expected = [0, ((128.0f64 / 255.0) * max as f64).round() as u32, max];
        match &result.frames[0].pixels {
            PixelData::U8(p) => {
                assert_eq!(p.iter().map(|&x| x as u32).collect::<Vec<_>>(), expected)
            }
            PixelData::U16(p) => {
                assert_eq!(p.iter().map(|&x| x as u32).collect::<Vec<_>>(), expected)
            }
            _ => panic!("unexpected output storage"),
        }
    }
    for format in [
        SampleFormat::U8Bits(0),
        SampleFormat::U8Bits(9),
        SampleFormat::U16Bits(0),
        SampleFormat::U16Bits(17),
    ] {
        assert!(
            decode(
                &bytes,
                &DecodeOptions {
                    format,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}

#[test]
fn partial_input_is_opt_in_and_completed_earlier_frames_remain_available() {
    let images = sequence_images(true);
    let bytes = encode_sequence(&images, &[1, 2, 3, 4, 5], Reference::Previous);
    let truncated = &bytes[..bytes.len() - 1];
    assert!(decode(truncated, &DecodeOptions::default()).is_err());
    let result = decode(
        truncated,
        &DecodeOptions {
            allow_partial: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!result.complete);
    assert!(!result.frames.is_empty());
    assert!(result.frames[0].complete);
    assert_pixels(&result.frames[0].pixels, images[0].pixels());
    assert!(result.boxes.is_empty());
    assert!(
        decode(
            b"not a codestream",
            &DecodeOptions {
                allow_partial: true,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn implicit_level10_compatibility_is_explicit_and_keeps_resource_limits() {
    let base = Image::new(2, 1, PixelLayout::Gray, PixelData::U8(vec![7, 19])).unwrap();
    let mut metadata = base.metadata().clone();
    for _ in 0..5 {
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap());
    }
    let source = base
        .with_metadata(
            metadata,
            (0..5).map(|value| PixelData::U8(vec![value; 2])).collect(),
        )
        .unwrap();
    let mut bytes = encoded(&source);
    let mut offset = 0;
    let mut removed = false;
    while offset < bytes.len() {
        let size = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        assert!(size >= 8 && offset + size <= bytes.len());
        if &bytes[offset + 4..offset + 8] == b"jxll" {
            bytes.drain(offset..offset + size);
            removed = true;
            break;
        }
        offset += size;
    }
    assert!(removed, "fixture must have an explicit Level10 declaration");
    assert!(decode(&bytes, &DecodeOptions::default()).is_err());
    let mut options = DecodeOptions {
        allow_implicit_main10: true,
        ..Default::default()
    };
    let image = decode(&bytes, &options).unwrap();
    assert_eq!(image.frames[0].pixels, *source.pixels());
    assert_eq!(image.frames[0].extra_planes, source.extra_planes());
    options.limits.max_pixels = 1;
    assert!(matches!(decode(&bytes, &options), Err(Error::Limit(_))));
}

fn encode_options() -> EncodeOptions {
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

fn encoded(image: &Image) -> Vec<u8> {
    encode(image, &encode_options()).expect("encode deterministic fixture")
}

fn encoded_raw_gray8(image: &Image) -> Vec<u8> {
    assert_eq!(image.layout(), PixelLayout::Gray);
    assert!(matches!(image.pixels(), PixelData::U8(_)));
    assert!(image.metadata().extra_channels.is_empty());
    let mut metadata = image.metadata().clone();
    // Zero prediction without transforms keeps these U8 intermediates in 16 bits.
    // This permits a bare Level 5 stream instead of the automatic Level 10 wrapper.
    metadata.modular_16_bit_buffer_sufficient = true;
    let image = image.clone().with_metadata(metadata, vec![]).unwrap();
    let bytes = encoded(&image);
    assert!(bytes.starts_with(&[0xff, 0x0a]));
    bytes
}

fn only_frame(image: &DecodedImage) -> &DecodedFrame {
    assert_eq!(image.frames.len(), 1);
    &image.frames[0]
}

fn assert_pixels(actual: &PixelData, expected: &PixelData) {
    // PixelData equality is bitwise for F32, not floating-point equality.
    assert_eq!(actual, expected);
}

fn assert_f32_words(actual: &PixelData, expected: &[u32]) {
    let PixelData::F32(samples) = actual else {
        panic!("expected F32 output, got {actual:?}");
    };
    assert_eq!(samples.len(), expected.len());
    for (i, (&sample, &word)) in samples.iter().zip(expected).enumerate() {
        assert_eq!(sample.to_bits(), word, "sample {i}, expected {word:#010x}");
    }
}

fn alpha_plane(pixels: &PixelData, layout: PixelLayout) -> PixelData {
    assert!(layout.has_alpha());
    let channels = layout.channels();
    macro_rules! plane {
        ($samples:expr, $variant:ident) => {
            PixelData::$variant(
                $samples
                    .chunks_exact(channels)
                    .map(|p| p[channels - 1])
                    .collect(),
            )
        };
    }
    match pixels {
        PixelData::U8(v) => plane!(v, U8),
        PixelData::U16(v) => plane!(v, U16),
        PixelData::U32(v) => plane!(v, U32),
        PixelData::F16(v) => plane!(v, F16),
        PixelData::F32(v) => plane!(v, F32),
    }
}

fn integer_image(width: u32, height: u32, layout: PixelLayout, wide: bool) -> Image {
    let channels = layout.channels();
    let count = width as usize * height as usize * channels;
    let mut samples: Vec<u16> = (0..count)
        .map(|i| {
            let values = [0, 1, 255, 256, 32767, 32768, 65534, 65535];
            if wide {
                values[i % values.len()]
            } else {
                ((i * 37 + 19) % 256) as u16
            }
        })
        .collect();
    if layout.has_alpha() {
        for (i, pixel) in samples.chunks_exact_mut(channels).enumerate() {
            pixel[channels - 1] = match i % 3 {
                0 => 0,
                1 => {
                    if wide {
                        32768
                    } else {
                        128
                    }
                }
                _ => {
                    if wide {
                        65535
                    } else {
                        255
                    }
                }
            };
            if i % 3 == 0 {
                pixel[0] = if wide { 32769 } else { 113 };
            }
        }
    }
    let pixels = if wide {
        PixelData::U16(samples)
    } else {
        PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
    };
    Image::new(width, height, layout, pixels).unwrap()
}

#[test]
fn native_integer_roundtrips_all_layouts_and_hidden_colors() {
    for wide in [false, true] {
        for layout in LAYOUTS {
            for (width, height) in [(1, 1), (1, 5), (7, 3), (17, 2)] {
                let source = integer_image(width, height, layout, wide);
                let bytes = encoded(&source);
                let result = decode(&bytes, &DecodeOptions::default()).unwrap();
                assert!(result.complete);
                assert!(result.boxes.is_empty());
                assert_eq!(result.info.size, (width as usize, height as usize));
                assert_eq!(
                    result.info.bit_depth.bits_per_sample(),
                    if wide { 16 } else { 8 }
                );
                assert!(result.info.animation.is_none());
                assert!(result.embedded_profile.is_some());
                assert!(result.output_profile.is_some());
                let frame = only_frame(&result);
                assert_eq!((frame.width, frame.height), result.info.size);
                assert_eq!(frame.index, 0);
                assert_eq!(frame.layout, layout);
                assert_eq!(frame.duration_ticks, 0);
                assert!(frame.name.is_empty());
                assert_pixels(&frame.pixels, source.pixels());
                assert_eq!(frame.extra_planes.len(), usize::from(layout.has_alpha()));
                if layout.has_alpha() {
                    assert_eq!(result.info.extra_channels[0].ec_type, ExtraChannel::Alpha);
                    assert!(!result.info.extra_channels[0].alpha_associated);
                    assert_pixels(
                        &frame.extra_planes[0],
                        &alpha_plane(source.pixels(), layout),
                    );
                }
            }
        }
    }
}

#[test]
fn native_integer_depths_keep_the_declared_range() {
    for bits in 1..=16 {
        let maximum = ((1u32 << bits) - 1) as u16;
        let values = vec![0, 1, maximum / 2, maximum, maximum - 1, 0];
        let pixels = if bits <= 8 {
            PixelData::U8(values.iter().map(|&v| v as u8).collect())
        } else {
            PixelData::U16(values)
        };
        let source = Image::new(3, 2, PixelLayout::Gray, pixels).unwrap();
        let mut metadata = source.metadata().clone();
        metadata.bit_depth = BitDepth::new(bits).unwrap();
        let source = source.with_metadata(metadata, vec![]).unwrap();
        let result = decode(&encoded(&source), &DecodeOptions::default()).unwrap();
        assert_eq!(result.info.bit_depth.bits_per_sample(), u32::from(bits));
        assert_pixels(&only_frame(&result).pixels, source.pixels());
    }
}

#[test]
fn requested_integer_formats_rescale_and_f32_normalizes() {
    let values = vec![0u8, 1, 17, 127, 128, 254, 255];
    for wide in [false, true] {
        let pixels = if wide {
            PixelData::U16(values.iter().map(|&v| u16::from(v) * 257).collect())
        } else {
            PixelData::U8(values.clone())
        };
        let source = Image::new(7, 1, PixelLayout::Gray, pixels).unwrap();
        let bytes = encoded(&source);
        for format in [SampleFormat::U8, SampleFormat::U16, SampleFormat::F32] {
            let result = decode(
                &bytes,
                &DecodeOptions {
                    format,
                    ..Default::default()
                },
            )
            .unwrap();
            let pixels = &only_frame(&result).pixels;
            match format {
                SampleFormat::U8 => assert_pixels(pixels, &PixelData::U8(values.clone())),
                SampleFormat::U16 => assert_pixels(
                    pixels,
                    &PixelData::U16(values.iter().map(|&v| u16::from(v) * 257).collect()),
                ),
                SampleFormat::F32 => {
                    let PixelData::F32(samples) = pixels else {
                        panic!("expected F32")
                    };
                    assert_eq!(samples.len(), values.len());
                    for (&actual, &expected) in samples.iter().zip(&values) {
                        assert!((actual - f32::from(expected) / 255.0).abs() <= f32::EPSILON);
                    }
                }
                SampleFormat::Native | SampleFormat::U8Bits(_) | SampleFormat::U16Bits(_) => {
                    unreachable!()
                }
            }
        }
    }
}

fn half_to_f32_bits(word: u16) -> u32 {
    let sign = u32::from(word & 0x8000) << 16;
    let exponent = u32::from((word >> 10) & 31);
    let fraction = u32::from(word & 1023);
    match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = 31 - fraction.leading_zeros();
            sign | ((leading + 103) << 23) | ((fraction - (1 << leading)) << (23 - leading))
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    }
}

#[test]
fn all_65536_f16_patterns_expand_to_exact_f32_words() {
    // Odd multiplication permutes the entire domain and mixes special/finite lanes.
    let words: Vec<u16> = (0..=u16::MAX).map(|v| v.wrapping_mul(40503)).collect();
    let expected: Vec<u32> = words.iter().copied().map(half_to_f32_bits).collect();
    let source = Image::new(256, 256, PixelLayout::Gray, PixelData::F16(words)).unwrap();
    let bytes = encoded(&source);
    for (format, threads) in [(SampleFormat::Native, 1), (SampleFormat::F32, 3)] {
        let result = decode(
            &bytes,
            &DecodeOptions {
                format,
                threads,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.complete);
        assert!(matches!(
            result.info.bit_depth,
            JxlBitDepth::Float {
                bits_per_sample: 16,
                exponent_bits_per_sample: 5,
            }
        ));
        assert_f32_words(&only_frame(&result).pixels, &expected);
    }
}

#[test]
fn float_edges_preserve_payloads_and_hidden_colors_in_every_layout() {
    let half = [
        0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7c01, 0x7dff,
        0xfc01, 0xfdff, 0x7e55, 0xffff,
    ];
    let single = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0x7fbf_ffff,
        0xff80_1234,
        0x7fc1_2345,
        0xffff_ffff,
    ];
    for layout in LAYOUTS {
        for is_half in [false, true] {
            let count = 17 * 3 * layout.channels();
            let (pixels, expected) = if is_half {
                let words: Vec<u16> = (0..count).map(|i| half[i % half.len()]).collect();
                let expected = words
                    .iter()
                    .copied()
                    .map(half_to_f32_bits)
                    .collect::<Vec<_>>();
                (PixelData::F16(words), expected)
            } else {
                let words: Vec<u32> = (0..count).map(|i| single[i % single.len()]).collect();
                (
                    PixelData::F32(words.iter().copied().map(f32::from_bits).collect()),
                    words,
                )
            };
            let source = Image::new(17, 3, layout, pixels).unwrap();
            let result = decode(&encoded(&source), &DecodeOptions::default()).unwrap();
            let frame = only_frame(&result);
            assert_eq!(frame.layout, layout);
            assert_f32_words(&frame.pixels, &expected);
            if layout.has_alpha() {
                let alpha: Vec<u32> = expected
                    .chunks_exact(layout.channels())
                    .map(|p| p[layout.channels() - 1])
                    .collect();
                assert_eq!(frame.extra_planes.len(), 1);
                assert_f32_words(&frame.extra_planes[0], &alpha);
            }
        }
    }
}

#[test]
fn native_wide_integers_are_normalized_f32_not_u31_storage() {
    let values = vec![0, 1, 1 << 24, 1 << 30, (1 << 31) - 2, (1 << 31) - 1];
    let source = Image::new(3, 2, PixelLayout::Gray, PixelData::U32(values.clone())).unwrap();
    let result = decode(&encoded(&source), &DecodeOptions::default()).unwrap();
    let PixelData::F32(samples) = &only_frame(&result).pixels else {
        panic!("wide native output must be F32")
    };
    assert_eq!(samples.len(), values.len());
    for (&actual, &value) in samples.iter().zip(&values) {
        let expected = (f64::from(value) / 2_147_483_647.0) as f32;
        assert!((actual - expected).abs() <= f32::EPSILON);
    }
    assert_eq!(samples[4].to_bits(), samples[5].to_bits());
}

#[test]
fn mixed_extra_depths_names_and_authoritative_alpha_planes() {
    let alpha = [0u16, 1, 17, 128, 254, 255];
    let pixels: Vec<u16> = alpha
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| [17 + i as u16, 2048, 4095, a])
        .collect();
    let source = Image::new(3, 2, PixelLayout::Rgba, PixelData::U16(pixels.clone())).unwrap();
    let mut metadata = source.metadata().clone();
    metadata.bit_depth = BitDepth::new(12).unwrap();
    metadata.extra_channels[0].bit_depth = BitDepth::new(8).unwrap();
    metadata.extra_channels[0].name = "opacity\0mask".into();
    let depths = [
        (
            ExtraChannelType::Depth,
            BitDepth::new(16).unwrap(),
            "depth-\u{6df1}",
        ),
        (
            ExtraChannelType::SelectionMask,
            BitDepth::new(1).unwrap(),
            "selection",
        ),
        (
            ExtraChannelType::Thermal,
            BitDepth::floating(16, 5).unwrap(),
            "thermal",
        ),
        (
            ExtraChannelType::Optional,
            BitDepth::floating(32, 8).unwrap(),
            "raw\0float",
        ),
    ];
    for (kind, depth, name) in depths {
        let mut info = ExtraChannelInfo::new(kind, 8).unwrap();
        info.bit_depth = depth;
        info.name = name.into();
        metadata.extra_channels.push(info);
    }
    let half = [0, 0x8000, 0x0001, 0x7c01, 0xfe55, 0x7c00];
    let single = [
        0x8000_0000,
        1,
        0x3f80_0000,
        0xff80_0001,
        0x7fc1_2345,
        0x7f80_0000,
    ];
    let extras = vec![
        PixelData::U16(vec![0, 1, 255, 32768, 65534, 65535]),
        PixelData::U8(vec![0, 1, 0, 1, 1, 0]),
        PixelData::F16(half.to_vec()),
        PixelData::F32(single.iter().copied().map(f32::from_bits).collect()),
    ];
    let source = source.with_metadata(metadata, extras).unwrap();
    let bytes = encoded(&source);
    let result = decode(&bytes, &DecodeOptions::default()).unwrap();
    let frame = only_frame(&result);
    assert_eq!(frame.layout, PixelLayout::Rgba);
    assert_eq!(frame.extra_planes.len(), 5);
    for (actual, expected) in result
        .info
        .extra_channels
        .iter()
        .zip(&source.metadata().extra_channels)
    {
        assert_eq!(actual.name, expected.name);
        assert_eq!(
            actual.bit_depth.bits_per_sample(),
            u32::from(expected.bit_depth.bits_per_sample)
        );
        assert_eq!(actual.dim_shift, 0);
    }
    assert_eq!(
        result
            .info
            .extra_channels
            .iter()
            .map(|e| e.ec_type)
            .collect::<Vec<_>>(),
        [
            ExtraChannel::Alpha,
            ExtraChannel::Depth,
            ExtraChannel::SelectionMask,
            ExtraChannel::Thermal,
            ExtraChannel::Optional
        ]
    );
    assert!(matches!(
        result.info.extra_channels[3].bit_depth,
        JxlBitDepth::Float {
            bits_per_sample: 16,
            exponent_bits_per_sample: 5,
        }
    ));
    assert_pixels(
        &frame.extra_planes[0],
        &PixelData::U8(alpha.iter().map(|&v| v as u8).collect()),
    );
    assert_pixels(&frame.extra_planes[1], &source.extra_planes()[0]);
    assert_pixels(&frame.extra_planes[2], &source.extra_planes()[1]);
    assert_f32_words(
        &frame.extra_planes[3],
        &half.into_iter().map(half_to_f32_bits).collect::<Vec<_>>(),
    );
    assert_f32_words(&frame.extra_planes[4], &single);
    let PixelData::U16(main) = &frame.pixels else {
        panic!("native 12-bit main must use U16")
    };
    for (i, pixel) in main.chunks_exact(4).enumerate() {
        assert_eq!(&pixel[..3], &pixels[i * 4..i * 4 + 3]);
        assert_eq!(
            pixel[3],
            (f32::from(alpha[i]) * (4095.0 / 255.0)).round() as u16
        );
    }

    let no_extras = decode(
        &bytes,
        &DecodeOptions {
            include_extra_channels: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(no_extras.info.extra_channels.len(), 5);
    assert!(only_frame(&no_extras).extra_planes.is_empty());
    assert_pixels(&only_frame(&no_extras).pixels, &frame.pixels);

    let float = decode(
        &bytes,
        &DecodeOptions {
            format: SampleFormat::F32,
            ..Default::default()
        },
    )
    .unwrap();
    let float_frame = only_frame(&float);
    assert!(matches!(float_frame.pixels, PixelData::F32(_)));
    assert!(
        float_frame
            .extra_planes
            .iter()
            .all(|p| matches!(p, PixelData::F32(_)))
    );
    assert_f32_words(&float_frame.extra_planes[4], &single);
}

#[test]
fn subsampled_extra_planes_are_full_canvas_and_report_the_source_shift() {
    for shift in 1..=3 {
        let source = integer_image(5, 3, PixelLayout::Gray, false);
        let mut metadata = source.metadata().clone();
        let mut depth = ExtraChannelInfo::new(ExtraChannelType::Depth, 12).unwrap();
        depth.name = "constant-subsampled".into();
        depth.dim_shift = shift;
        metadata.extra_channels.push(depth);
        let scale = 1usize << shift;
        let count = 5usize.div_ceil(scale) * 3usize.div_ceil(scale);
        let source = source
            .with_metadata(metadata, vec![PixelData::U16(vec![4095; count])])
            .unwrap();
        let result = decode(&encoded(&source), &DecodeOptions::default()).unwrap();
        assert_eq!(result.info.extra_channels[0].dim_shift, u32::from(shift));
        let frame = only_frame(&result);
        assert_pixels(&frame.pixels, source.pixels());
        assert_pixels(&frame.extra_planes[0], &PixelData::U16(vec![4095; 15]));
    }
}

fn oriented_indices(width: usize, height: usize, orientation: u8) -> ((usize, usize), Vec<usize>) {
    let size = if orientation >= 5 {
        (height, width)
    } else {
        (width, height)
    };
    let mut order = vec![usize::MAX; width * height];
    for y in 0..height {
        for x in 0..width {
            let (dx, dy) = match orientation {
                1 => (x, y),
                2 => (width - 1 - x, y),
                3 => (width - 1 - x, height - 1 - y),
                4 => (x, height - 1 - y),
                5 => (y, x),
                6 => (height - 1 - y, x),
                7 => (height - 1 - y, width - 1 - x),
                8 => (y, width - 1 - x),
                _ => unreachable!(),
            };
            order[dy * size.0 + dx] = y * width + x;
        }
    }
    (size, order)
}

#[test]
fn all_orientations_apply_or_keep_color_alpha_and_extra_planes() {
    for (width, height) in [(3, 2), (1, 5), (5, 1)] {
        for orientation in 1..=8 {
            let source = integer_image(width, height, PixelLayout::Rgba, false);
            let mut metadata = source.metadata().clone();
            metadata.orientation = Orientation::try_from(orientation).unwrap();
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
            let depth: Vec<u16> = (0..width * height)
                .map(|i| (i * 5003 + 257) as u16)
                .collect();
            let source = source
                .with_metadata(metadata, vec![PixelData::U16(depth.clone())])
                .unwrap();
            let bytes = encoded(&source);
            let PixelData::U8(color) = source.pixels() else {
                unreachable!()
            };
            for adjust_orientation in [false, true] {
                let result = decode(
                    &bytes,
                    &DecodeOptions {
                        adjust_orientation,
                        ..Default::default()
                    },
                )
                .unwrap();
                let (size, indices) = oriented_indices(
                    width as usize,
                    height as usize,
                    if adjust_orientation { orientation } else { 1 },
                );
                assert_eq!(result.info.size, size);
                assert_eq!(result.info.orientation as u8, orientation);
                let frame = only_frame(&result);
                assert_eq!((frame.width, frame.height), size);
                let expected: Vec<u8> = indices
                    .iter()
                    .flat_map(|&i| color[i * 4..i * 4 + 4].iter().copied())
                    .collect();
                assert_pixels(&frame.pixels, &PixelData::U8(expected));
                assert_pixels(
                    &frame.extra_planes[0],
                    &PixelData::U8(indices.iter().map(|&i| color[i * 4 + 3]).collect()),
                );
                assert_pixels(
                    &frame.extra_planes[1],
                    &PixelData::U16(indices.iter().map(|&i| depth[i]).collect()),
                );
            }
        }
    }
}

fn sequence_images(animated: bool) -> Vec<Image> {
    let base = integer_image(4, 3, PixelLayout::Rgba, false);
    let PixelData::U8(color) = base.pixels() else {
        unreachable!()
    };
    let mut metadata = base.metadata().clone();
    metadata.animation = animated.then_some(Animation {
        ticks_per_second_numerator: 8,
        ticks_per_second_denominator: 2,
        loops: 3,
        timecodes: false,
    });
    let mut depth = ExtraChannelInfo::new(ExtraChannelType::Depth, 12).unwrap();
    depth.name = "sequence-depth".into();
    metadata.extra_channels.push(depth);
    let make = |pixels: Vec<u8>, values: Vec<u16>| {
        Image::new(4, 3, PixelLayout::Rgba, PixelData::U8(pixels))
            .unwrap()
            .with_metadata(metadata.clone(), vec![PixelData::U16(values)])
            .unwrap()
    };
    let depths: Vec<u16> = (0..12).map(|i| i * 257).collect();
    let a = make(color.clone(), depths.clone());
    let mut changed = color.clone();
    changed[0] ^= 1; // Change only invisible color at the top-left corner.
    let b = make(changed, depths.clone());
    let mut changed = color.clone();
    changed[11 * 4 + 2] ^= 7;
    let mut changed_depths = depths;
    changed_depths[11] ^= 1;
    let c = make(changed, changed_depths);
    vec![a.clone(), a, b, c.clone(), c]
}

fn encode_sequence(images: &[Image], durations: &[u32], reference: Reference) -> Vec<u8> {
    let mut plan = multiframe::plan(images, durations, reference).unwrap();
    for (i, frame) in plan.iter_mut().enumerate() {
        frame.options.name = format!("frame-{i}\0\u{5c42}");
    }
    let frames: Vec<Frame<'_>> = plan.iter().map(|frame| frame.as_frame()).collect();
    encode_frames(&images[0], &frames, &encode_options()).unwrap()
}

fn assert_sequence_frame(
    frame: &DecodedFrame,
    source: &Image,
    visible: usize,
    coded: usize,
    duration: u32,
) {
    assert_eq!(frame.index, visible);
    assert_eq!(
        (frame.width, frame.height),
        (source.width() as usize, source.height() as usize)
    );
    assert_eq!(frame.layout, source.layout());
    assert_eq!(frame.duration_ticks, duration);
    assert_eq!(frame.name, format!("frame-{coded}\0\u{5c42}"));
    assert_pixels(&frame.pixels, source.pixels());
    assert_eq!(frame.extra_planes.len(), 2);
    assert_pixels(
        &frame.extra_planes[0],
        &alpha_plane(source.pixels(), source.layout()),
    );
    assert_pixels(&frame.extra_planes[1], &source.extra_planes()[0]);
}

#[test]
fn all_reference_plans_reconstruct_visible_frames_and_zero_duration_layers() {
    for (animated, durations) in [
        (true, [1, 2, 3, 4, 5]),
        (true, [0, 2, 0, 3, 0]),
        (true, [0; 5]),
        (false, [0; 5]),
    ] {
        let images = sequence_images(animated);
        let visible: Vec<usize> = (0..images.len())
            .filter(|&i| durations[i] != 0 || i + 1 == images.len())
            .collect();
        for reference in REFERENCES {
            let bytes = encode_sequence(&images, &durations, reference);
            let result = decode(&bytes, &DecodeOptions::default()).unwrap();
            assert!(result.complete);
            assert_eq!(
                result.frames.len(),
                visible.len(),
                "{reference:?}, {durations:?}"
            );
            if animated {
                let animation = result.info.animation.as_ref().unwrap();
                assert_eq!((animation.tps_numerator, animation.tps_denominator), (8, 2));
                assert_eq!(animation.num_loops, 3);
                assert!(!animation.have_timecodes);
            } else {
                assert!(result.info.animation.is_none());
            }
            for (index, &coded) in visible.iter().enumerate() {
                assert_sequence_frame(
                    &result.frames[index],
                    &images[coded],
                    index,
                    coded,
                    durations[coded],
                );
                let selected = decode_frame(&bytes, index, &DecodeOptions::default()).unwrap();
                assert!(!selected.complete);
                assert!(selected.boxes.is_empty());
                assert_sequence_frame(
                    only_frame(&selected),
                    &images[coded],
                    index,
                    coded,
                    durations[coded],
                );
            }
            assert!(matches!(
                decode_frame(&bytes, visible.len(), &DecodeOptions::default()),
                Err(Error::Decoding(_))
            ));
        }
    }
}

#[test]
fn selected_frame_is_owned_and_does_not_require_a_complete_later_frame() {
    let images = sequence_images(true);
    let bytes = encode_sequence(&images, &[1, 2, 3, 4, 5], Reference::Previous);
    let mut truncated = bytes.clone();
    truncated.truncate(truncated.len() - 1);
    assert!(decode(&truncated, &DecodeOptions::default()).is_err());
    let first = decode_frame(&truncated, 0, &DecodeOptions::default()).unwrap();
    assert_sequence_frame(only_frame(&first), &images[0], 0, 0, 1);
    drop(truncated);
    drop(bytes);
    // Returned samples are owned, not borrowed from input or decoder storage.
    assert_pixels(&only_frame(&first).pixels, images[0].pixels());
}

#[test]
fn reference_only_frames_are_dependencies_not_visible_indices() {
    let images = sequence_images(true);
    let crop = multiframe::crop(
        &images[3],
        multiframe::Rect {
            x: 3,
            y: 2,
            width: 1,
            height: 1,
        },
    )
    .unwrap();
    let frames = [
        Frame {
            image: &images[0],
            options: FrameOptions {
                reference_only: true,
                save_as_reference: 1,
                name: "invisible-reference".into(),
                ..Default::default()
            },
        },
        Frame {
            image: &crop,
            options: FrameOptions {
                origin: (3, 2),
                duration: 7,
                name: "frame-3\0\u{5c42}".into(),
                blend: rs_jxl::frame::BlendInfo {
                    source: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        },
    ];
    let bytes = encode_frames(&images[0], &frames, &encode_options()).unwrap();
    let result = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_sequence_frame(only_frame(&result), &images[3], 0, 3, 7);
    let selected = decode_frame(&bytes, 0, &DecodeOptions::default()).unwrap();
    assert_sequence_frame(only_frame(&selected), &images[3], 0, 3, 7);
    assert!(decode_frame(&bytes, 1, &DecodeOptions::default()).is_err());
}

fn assert_limit(result: rs_jxl::Result<DecodedImage>) {
    match result {
        Err(Error::Limit(_)) => (),
        Err(error) => panic!("expected a resource-limit error, got {error}"),
        Ok(_) => panic!("decoding exceeded its limit"),
    }
}

#[test]
fn input_pixel_and_output_limits_have_inclusive_boundaries() {
    let source = integer_image(4, 3, PixelLayout::Rgba, false);
    let bytes = encoded(&source);
    for maximum in [0, bytes.len() - 1] {
        let mut options = DecodeOptions::default();
        options.limits.max_input_bytes = maximum;
        assert_limit(decode(&bytes, &options));
        assert_limit(decode_frame(&bytes, 0, &options));
    }
    let mut exact = DecodeOptions::default();
    exact.limits.max_input_bytes = bytes.len();
    exact.limits.max_pixels = 12;
    assert_pixels(
        &only_frame(&decode(&bytes, &exact).unwrap()).pixels,
        source.pixels(),
    );
    for maximum in [0, 11] {
        let mut options = DecodeOptions::default();
        options.limits.max_pixels = maximum;
        assert_limit(decode(&bytes, &options));
    }
    for (format, bytes_per_sample) in [
        (SampleFormat::Native, 1),
        (SampleFormat::U16, 2),
        (SampleFormat::F32, 4),
    ] {
        for include_extra_channels in [false, true] {
            let needed = 12 * (4 + usize::from(include_extra_channels)) * bytes_per_sample;
            let mut options = DecodeOptions {
                format,
                include_extra_channels,
                ..Default::default()
            };
            options.limits.max_output_bytes = needed - 1;
            assert_limit(decode(&bytes, &options));
            options.limits.max_output_bytes = needed;
            assert!(decode(&bytes, &options).unwrap().complete);
        }
    }
}

#[test]
fn invalid_threads_and_insufficient_sample_budgets_fail() {
    let source = integer_image(4, 3, PixelLayout::Rgba, false);
    let bytes = encoded(&source);
    for threads in [0, 257, usize::MAX] {
        let options = DecodeOptions {
            threads,
            ..Default::default()
        };
        assert!(matches!(
            decode(&bytes, &options),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            decode_frame(&bytes, 0, &options),
            Err(Error::InvalidInput(_))
        ));
    }
    for maximum in [0, 1, 47] {
        let mut options = DecodeOptions::default();
        options.limits.max_samples = maximum;
        assert!(decode(&bytes, &options).is_err());
        // Disabling returned extra planes must not bypass source sample limits.
        options.include_extra_channels = false;
        assert!(decode(&bytes, &options).is_err());
    }
    let result = decode(
        &bytes,
        &DecodeOptions {
            threads: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_pixels(&only_frame(&result).pixels, source.pixels());
}

#[test]
fn frame_and_retained_output_limits_include_extras_but_not_discarded_frames() {
    let images = sequence_images(true);
    let bytes = encode_sequence(&images, &[1, 2, 3, 4, 5], Reference::Previous);
    // RGBA U8 + authoritative U8 alpha + U12 stored in U16.
    let frame_bytes = 4 * 3 * (4 + 1 + 2);
    let mut options = DecodeOptions::default();
    options.limits.max_output_bytes = frame_bytes * 5;
    options.limits.max_frames = 5;
    assert_eq!(decode(&bytes, &options).unwrap().frames.len(), 5);
    options.limits.max_output_bytes -= 1;
    assert_limit(decode(&bytes, &options));
    options.limits.max_output_bytes = frame_bytes;
    let selected = decode_frame(&bytes, 4, &options).unwrap();
    assert_sequence_frame(only_frame(&selected), &images[4], 4, 4, 5);
    options.limits.max_output_bytes -= 1;
    assert_limit(decode_frame(&bytes, 4, &options));

    for maximum in [0, 4] {
        let mut options = DecodeOptions::default();
        options.limits.max_frames = maximum;
        assert_limit(decode(&bytes, &options));
        assert_limit(decode_frame(&bytes, 4, &options));
    }
    assert_limit(decode_frame(&bytes, usize::MAX, &DecodeOptions::default()));
    assert!(matches!(
        decode_frame(&bytes, 5, &DecodeOptions::default()),
        Err(Error::Decoding(_))
    ));

    for (durations, visible) in [([0, 2, 0, 3, 0], 3), ([0; 5], 1)] {
        let bytes = encode_sequence(&images, &durations, Reference::Previous);
        let mut options = DecodeOptions::default();
        options.limits.max_frames = visible;
        options.limits.max_output_bytes = frame_bytes * visible;
        assert_eq!(decode(&bytes, &options).unwrap().frames.len(), visible);
    }
}

#[test]
fn malformed_inputs_and_every_strict_codestream_prefix_are_errors() {
    for input in [b"".as_slice(), b"not a JXL", b"\xff", b"\xff\x0a"] {
        assert!(decode(input, &DecodeOptions::default()).is_err());
        assert!(decode_frame(input, 0, &DecodeOptions::default()).is_err());
    }
    let source = integer_image(3, 2, PixelLayout::Gray, false);
    let bytes = encoded_raw_gray8(&source);
    for end in 0..bytes.len() {
        assert!(
            decode(&bytes[..end], &DecodeOptions::default()).is_err(),
            "accepted prefix {end}/{}",
            bytes.len()
        );
    }
    assert_pixels(
        &only_frame(&decode(&bytes, &DecodeOptions::default()).unwrap()).pixels,
        source.pixels(),
    );
}

type MetadataBoxes = Vec<([u8; 4], Vec<u8>)>;

fn metadata_container(source: &Image) -> (Vec<u8>, MetadataBoxes) {
    let raw = encoded_raw_gray8(source);
    let exif = b"\0\0\0\0II*\0\x08\0\0\0\0\0\0\0\0\0".to_vec();
    let xml = b"<xmp>before final codestream chunk</xmp>".to_vec();
    let jumb = b"opaque JUMBF payload\0\xff".to_vec();
    let trailing = b"<xmp>after codestream</xmp>".to_vec();
    let mut writer = ChunkedWriter::new(Vec::new(), 5, 1 << 20).unwrap();
    writer.write_metadata(*b"Exif", &exif).unwrap();
    let split = raw.len() / 2;
    writer.write_chunk(&raw[..split], false).unwrap();
    writer.write_metadata(*b"xml ", &xml).unwrap();
    writer.write_chunk(&raw[split..], true).unwrap();
    writer.write_metadata(*b"jumb", &jumb).unwrap();
    writer.write_metadata(*b"xml ", &trailing).unwrap();
    (
        writer.finish().unwrap(),
        vec![
            (*b"Exif", exif),
            (*b"xml ", xml),
            (*b"jumb", jumb),
            (*b"xml ", trailing),
        ],
    )
}

fn assert_boxes(image: &DecodedImage, expected: &[([u8; 4], Vec<u8>)]) {
    assert_eq!(image.boxes.len(), expected.len());
    // The API groups by kind; do not assume physical cross-kind box ordering.
    for kind in [*b"Exif", *b"xml ", *b"jumb"] {
        let actual: Vec<&[u8]> = image
            .boxes
            .iter()
            .filter(|b| b.kind == kind)
            .map(|b| b.data.as_slice())
            .collect();
        let expected: Vec<&[u8]> = expected
            .iter()
            .filter(|(k, _)| *k == kind)
            .map(|(_, data)| data.as_slice())
            .collect();
        assert_eq!(actual, expected, "box {kind:?}");
    }
}

#[test]
fn chunked_container_metadata_before_between_and_after_pixels_is_returned() {
    let source = integer_image(3, 2, PixelLayout::Gray, false);
    let (bytes, expected) = metadata_container(&source);
    let result = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert!(result.complete);
    assert_pixels(&only_frame(&result).pixels, source.pixels());
    assert_boxes(&result, &expected);
    let selected = decode_frame(&bytes, 0, &DecodeOptions::default()).unwrap();
    assert!(!selected.complete);
    assert!(selected.boxes.is_empty());
    assert_pixels(&only_frame(&selected).pixels, source.pixels());

    let required = expected.iter().map(|(_, data)| data.len()).sum::<usize>();
    for maximum in [0, required - 1, required] {
        let mut options = DecodeOptions::default();
        options.limits.max_metadata_bytes = maximum;
        if maximum < required {
            assert_limit(decode(&bytes, &options));
        } else {
            assert_boxes(&decode(&bytes, &options).unwrap(), &expected);
        }
    }
}

fn container_with_codestream(source: &Image) -> Vec<u8> {
    let raw = encoded_raw_gray8(source);
    let mut writer = ChunkedWriter::new(Vec::new(), 5, 1 << 20).unwrap();
    writer.write_chunk(&raw, true).unwrap();
    writer.finish().unwrap()
}

fn append_box(bytes: &mut Vec<u8>, kind: [u8; 4], payload: &[u8]) {
    bytes.extend_from_slice(&u32::try_from(payload.len() + 8).unwrap().to_be_bytes());
    bytes.extend_from_slice(&kind);
    bytes.extend_from_slice(payload);
}

fn compressed_box(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut result = kind.to_vec();
    {
        let mut writer = brotli::CompressorWriter::new(&mut result, 4096, 4, 18);
        writer.write_all(payload).unwrap();
    }
    result
}

#[test]
fn brob_metadata_is_decompressed_and_budgeted_cumulatively() {
    let source = integer_image(2, 2, PixelLayout::Gray, false);
    let mut bytes = container_with_codestream(&source);
    let first = b"<xmp>repeated</xmp>".repeat(32);
    let second = b"JUMBF\0".repeat(64);
    append_box(&mut bytes, *b"brob", &compressed_box(*b"xml ", &first));
    append_box(&mut bytes, *b"brob", &compressed_box(*b"jumb", &second));
    let required = first.len() + second.len();
    let expected = vec![(*b"xml ", first), (*b"jumb", second)];
    for maximum in [0, required - 1, required] {
        let mut options = DecodeOptions::default();
        options.limits.max_metadata_bytes = maximum;
        if maximum < required {
            assert_limit(decode(&bytes, &options));
        } else {
            let result = decode(&bytes, &options).unwrap();
            assert_pixels(&only_frame(&result).pixels, source.pixels());
            assert_boxes(&result, &expected);
        }
    }
}

#[test]
fn unbounded_trailing_metadata_and_unknown_boxes_do_not_change_pixels() {
    let source = integer_image(2, 2, PixelLayout::Gray, false);
    let mut bytes = container_with_codestream(&source);
    append_box(&mut bytes, *b"test", b"unrequested opaque contents");
    let payload = b"<xmp>to end of input</xmp>";
    bytes.extend_from_slice(&0u32.to_be_bytes());
    bytes.extend_from_slice(b"xml ");
    bytes.extend_from_slice(payload);
    let result = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert!(result.complete);
    assert_pixels(&only_frame(&result).pixels, source.pixels());
    assert_boxes(&result, &[(*b"xml ", payload.to_vec())]);
}

#[test]
fn full_decode_rejects_truncated_boxes_but_selected_frame_stops_early() {
    let source = integer_image(3, 2, PixelLayout::Gray, false);
    let good = container_with_codestream(&source);
    for bad_tail in [
        b"\0\0\0\x20xml <x/>".as_slice(),
        b"\0\0\0\x10brobxml \xff\xff\xff\xff",
        b"\0\0\0\x07xml ",
    ] {
        let mut bytes = good.clone();
        bytes.extend_from_slice(bad_tail);
        assert!(decode(&bytes, &DecodeOptions::default()).is_err());
        let selected = decode_frame(&bytes, 0, &DecodeOptions::default()).unwrap();
        assert!(!selected.complete);
        assert_pixels(&only_frame(&selected).pixels, source.pixels());
    }
}
