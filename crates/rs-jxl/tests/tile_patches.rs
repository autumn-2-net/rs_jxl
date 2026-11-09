//! Native sample checks for the exact repeated-tile experiment.
use std::sync::Arc;

use rs_jxl::{
    EncodeOptions, Image, ModularOptions, PixelData, PixelLayout,
    decode::{DecodeOptions, Decoder},
    encode, encode_frames_to,
    frame::{BlendInfo, Frame, FrameOptions},
    metadata::{ExtraChannelInfo, ExtraChannelType},
    modular::{encode::FixedTree, palette::PaletteOptions, predictor::Predictor},
    region_input::{FrameAreaPlan, ImageDescriptor},
    streaming::{encode_frames_streaming_to, encode_indexed_frames_streaming_to},
    tools::patches::{PatchOptions, SourcePatchConfig, TilePatchOptions, TilePatchReceipt},
};

const SIDE: usize = 16;
const LAYOUTS: [PixelLayout; 4] = [
    PixelLayout::Gray,
    PixelLayout::GrayAlpha,
    PixelLayout::Rgb,
    PixelLayout::Rgba,
];

fn options(threads: usize) -> (EncodeOptions, Arc<TilePatchReceipt>) {
    let receipt = Arc::new(TilePatchReceipt::default());
    let options = EncodeOptions {
        threads,
        tile_patches: Some(TilePatchOptions {
            force: true,
            receipt: Some(Arc::clone(&receipt)),
            ..Default::default()
        }),
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            rct: None,
            lz77: false,
            ..Default::default()
        },
        ..Default::default()
    };
    (options, receipt)
}

fn fixture(bits: u8, layout: PixelLayout, width: usize, height: usize, unique: bool) -> Image {
    let mut samples = Vec::with_capacity(width * height * layout.channels());
    for y in 0..height {
        for x in 0..width {
            let tile = (y / SIDE) * width.div_ceil(SIDE) + x / SIDE;
            for c in 0..layout.channels() {
                // Include nonzero hidden colors, zero, extrema, and low-bit variation.
                let alpha = layout.has_alpha() && c == layout.channels() - 1;
                let value = if alpha {
                    if (x % SIDE + y % SIDE).is_multiple_of(3) {
                        0
                    } else {
                        255
                    }
                } else {
                    let local = ((x % SIDE) * 17 + (y % SIDE) * 29 + c * 71) & 255;
                    if unique {
                        (local + tile * 11) & 255
                    } else {
                        local
                    }
                };
                let value = if !alpha && (x >= width / SIDE * SIDE || y >= height / SIDE * SIDE) {
                    (value + x * 7 + y * 13 + 1) & 255
                } else {
                    value
                };
                samples.push(if alpha {
                    if bits == 8 {
                        value as u16
                    } else {
                        value as u16 * 257
                    }
                } else if bits == 8 {
                    value as u16
                } else {
                    ((value << 8) | ((x % SIDE + 3 * (y % SIDE) + c) & 255)) as u16
                });
            }
        }
    }
    Image::new(
        width as u32,
        height as u32,
        layout,
        if bits == 8 {
            PixelData::U8(samples.into_iter().map(|v| v as u8).collect())
        } else {
            PixelData::U16(samples)
        },
    )
    .unwrap()
}

fn change_sample(image: &Image, x: usize, y: usize, channel: usize) -> Image {
    let at = (y * image.width() as usize + x) * image.layout().channels() + channel;
    let mut pixels = image.pixels().clone();
    match &mut pixels {
        PixelData::U8(values) => values[at] ^= 1,
        PixelData::U16(values) => values[at] ^= 1,
        _ => unreachable!(),
    }
    Image::new(image.width(), image.height(), image.layout(), pixels).unwrap()
}

fn assert_decoded(bytes: &[u8], source: &Image) {
    for threads in [1, 4] {
        let mut decoder = Decoder::new(DecodeOptions {
            threads,
            ..Default::default()
        })
        .unwrap();
        let decoded = decoder.decode(bytes).unwrap();
        assert!(decoded.complete);
        assert_eq!(decoded.frames.len(), 1);
        let frame = &decoded.frames[0];
        assert_eq!(
            (frame.width, frame.height),
            (source.width() as usize, source.height() as usize)
        );
        assert_eq!(&frame.pixels, source.pixels(), "decode threads={threads}");
        let alpha = usize::from(source.layout().has_alpha());
        assert_eq!(
            frame.extra_planes.len(),
            source.extra_planes().len() + alpha
        );
        if alpha != 0 {
            let stride = source.layout().channels();
            let expected = match source.pixels() {
                PixelData::U8(v) => {
                    PixelData::U8(v.chunks_exact(stride).map(|p| p[stride - 1]).collect())
                }
                PixelData::U16(v) => {
                    PixelData::U16(v.chunks_exact(stride).map(|p| p[stride - 1]).collect())
                }
                _ => unreachable!(),
            };
            assert_eq!(frame.extra_planes[0], expected);
        }
        assert_eq!(&frame.extra_planes[alpha..], source.extra_planes());
    }
}

#[test]
fn active_tiles_preserve_all_layouts_depths_hidden_colors_and_partial_edges() {
    for bits in [8, 16] {
        for layout in LAYOUTS {
            for threads in [1, 4] {
                // Right/bottom remainders must survive without being erased as full tiles.
                let source = fixture(bits, layout, 65, 33, false);
                let (options, receipt) = options(threads);
                let bytes = encode(&source, &options).unwrap();
                let stats = receipt.snapshot();
                assert!(stats.calls > 0);
                assert!(stats.tiles_examined >= 8);
                assert!(stats.repeated_groups > 0);
                assert!(stats.selected_groups > 0);
                assert!(
                    stats.positions >= 3,
                    "{bits} bits, {layout:?}, threads={threads}"
                );
                assert_eq!(stats.covered_pixels, stats.positions * SIDE * SIDE);
                assert!(stats.atlas_pixels > 0);
                assert!(stats.atlas_pixels <= 1 << 22);
                assert!(stats.seconds.is_finite() && stats.seconds >= 0.0);
                assert_decoded(&bytes, &source);
            }
        }
    }
}

#[test]
fn exactly_three_occurrences_are_required() {
    for count in [1, 2, 3] {
        let source = fixture(8, PixelLayout::Rgb, SIDE * count, SIDE, false);
        let (options, receipt) = options(1);
        let bytes = encode(&source, &options).unwrap();
        let stats = receipt.snapshot();
        assert_eq!(stats.positions, if count == 3 { 3 } else { 0 });
        assert_decoded(&bytes, &source);
    }
}

#[test]
fn alpha_only_and_invisible_rgb_differences_do_not_match() {
    for bits in [8, 16] {
        for layout in [PixelLayout::GrayAlpha, PixelLayout::Rgba] {
            for channel in [0, layout.channels() - 1] {
                // Three exact tiles remain; the fourth differs in exactly one sample.
                let original = fixture(bits, layout, SIDE * 4, SIDE, false);
                let source = change_sample(&original, SIDE * 4 - 1, SIDE - 1, channel);
                let (options, receipt) = options(1);
                let bytes = encode(&source, &options).unwrap();
                assert_eq!(receipt.snapshot().positions, 3);
                assert_decoded(&bytes, &source);

                // With only two exact tiles left, RGB equality cannot override alpha,
                // and zero alpha cannot make differing hidden color equivalent.
                let original = fixture(bits, layout, SIDE * 3, SIDE, false);
                let source = change_sample(&original, SIDE * 3 - 1, SIDE - 1, channel);
                let (options, receipt) = self::options(1);
                let bytes = encode(&source, &options).unwrap();
                assert_eq!(receipt.snapshot().positions, 0);
                assert_decoded(&bytes, &source);
            }
        }
    }
}

#[test]
fn no_repeat_and_subtile_inputs_remain_exact_without_positions() {
    for (width, height, unique) in [(64, 32, true), (15, 33, false), (49, 15, false)] {
        let source = fixture(16, PixelLayout::Rgba, width, height, unique);
        let (options, receipt) = options(4);
        let bytes = encode(&source, &options).unwrap();
        assert_eq!(receipt.snapshot().positions, 0);
        assert_decoded(&bytes, &source);
    }
}

#[test]
fn constant_tiles_do_not_create_a_dictionary() {
    let source = Image::new(64, 32, PixelLayout::Gray, PixelData::U8(vec![173; 64 * 32])).unwrap();
    let (options, receipt) = options(1);
    let bytes = encode(&source, &options).unwrap();
    let stats = receipt.snapshot();
    assert!(stats.constant_tiles > 0);
    assert_eq!(stats.positions, 0);
    assert_eq!(stats.atlas_pixels, 0);
    assert_decoded(&bytes, &source);
}

#[test]
fn screened_rejection_keeps_original_pixels_without_generating_a_reference() {
    let (width, height) = (64usize, 32usize);
    let pixels: Vec<u8> = (0..width * height)
        .map(|p| u8::from((p % width).is_multiple_of(16) && (p / width).is_multiple_of(16)))
        .collect();
    let source = Image::new(
        width as u32,
        height as u32,
        PixelLayout::Gray,
        PixelData::U8(pixels),
    )
    .unwrap();
    let (mut options, receipt) = options(1);
    options.tile_patches.as_mut().unwrap().force = false;
    let bytes = encode(&source, &options).unwrap();
    let stats = receipt.snapshot();
    assert_eq!(stats.admission_rejected, 1);
    assert_eq!(stats.selected_groups, 0);
    assert!(stats.repeated_groups > 0);
    assert_decoded(&bytes, &source);
}

#[test]
fn frame_area_route_uses_the_stronger_admission_margin() {
    let (width, height) = (385usize, 385usize);
    let pixels: Vec<u8> = (0..width * height)
        .map(|p| {
            let mut x = ((p % width % 16) + 16 * (p / width % 16) + 1) as u32;
            x = x.wrapping_mul(0x45d9f3b);
            x ^= x >> 16;
            x = x.wrapping_mul(0x45d9f3b);
            x as u8
        })
        .collect();
    let source = Image::new(
        width as u32,
        height as u32,
        PixelLayout::Gray,
        PixelData::U8(pixels),
    )
    .unwrap();
    let (mut options, receipt) = options(1);
    options.group_size_shift = 0;
    options.modular.palette_policy = Some(PaletteOptions {
        groupwise: true,
        ..Default::default()
    });
    let tiles = options.tile_patches.as_mut().unwrap();
    tiles.force = false;
    tiles.full_frame_min_estimated_savings = 1.0;
    let bytes = encode(&source, &options).unwrap();
    let work = receipt.snapshot();
    assert!(work.estimated_saved_bits > 0.0);
    assert_eq!(work.admission_rejected, 1);
    assert_decoded(&bytes, &source);
    let accepted = Arc::new(TilePatchReceipt::default());
    let tiles = options.tile_patches.as_mut().unwrap();
    tiles.full_frame_min_estimated_savings = 0.02;
    tiles.receipt = Some(accepted.clone());
    let bytes = encode(&source, &options).unwrap();
    assert!(accepted.snapshot().selected_groups > 0);
    assert_decoded(&bytes, &source);
}

#[test]
fn mixed_depth_extras_preserve_replaced_and_wide_pass_through_words() {
    let source = fixture(8, PixelLayout::Rgb, 65, 33, false);
    let mut metadata = source.metadata().clone();
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 12).unwrap());
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 31).unwrap());
    let count = source.width() as usize * source.height() as usize;
    let narrow = PixelData::U16(
        (0..count)
            .map(|i| {
                let (x, y) = (i % 65, i / 65);
                ((x % SIDE) * 193 + (y % SIDE) * 71) as u16 & 4095
            })
            .collect(),
    );
    // Distinct values at corresponding tile positions force genuine pass-through.
    let wide = PixelData::U32(
        (0..count)
            .map(|i| match i {
                0 => 0,
                1 => 1,
                2 => 2_147_483_647,
                _ => 16_777_217 + i as u32 * 3,
            })
            .collect(),
    );
    let source = source.with_metadata(metadata, vec![narrow, wide]).unwrap();
    for threads in [1, 4] {
        let (options, receipt) = options(threads);
        let bytes = encode(&source, &options).unwrap();
        assert!(receipt.snapshot().positions > 0);
        assert_decoded(&bytes, &source);
    }
}

#[test]
fn conflicting_detectors_and_reserved_slot_fail_before_writing() {
    let source = fixture(8, PixelLayout::Rgb, 64, 16, false);
    for source_add in [false, true] {
        let (mut options, _) = options(1);
        if source_add {
            options.source_patches = Some(SourcePatchConfig::default());
        } else {
            options.patches = Some(PatchOptions::default());
        }
        let mut sink = vec![1, 2, 3];
        assert!(encode_frames_to(&source, &[Frame::new(&source)], &options, &mut sink).is_err());
        assert_eq!(sink, [1, 2, 3]);
    }
    let (options, _) = options(1);
    for settings in [
        FrameOptions {
            save_as_reference: 3,
            ..Default::default()
        },
        FrameOptions {
            blend: BlendInfo {
                source: 3,
                ..Default::default()
            },
            ..Default::default()
        },
    ] {
        let mut sink = vec![1, 2, 3];
        let frame = Frame {
            image: &source,
            options: settings,
        };
        let frames = [frame, Frame::new(&source)];
        let error = encode_frames_to(&source, &frames, &options, &mut sink).unwrap_err();
        assert!(error.to_string().contains("reserve reference slot 3"));
        assert_eq!(sink, [1, 2, 3]);
        let mut control = options.clone();
        control.tile_patches = None;
        encode_frames_to(&source, &frames, &control, Vec::new()).unwrap();
    }
}

#[test]
fn frame_streaming_keeps_tiles_but_indexed_and_area_streams_reject_them() {
    let source = fixture(8, PixelLayout::Rgba, 64, 16, false);
    let (options, receipt) = options(1);
    let mut sink = Vec::new();
    encode_frames_streaming_to(&source, &[Frame::new(&source)], &options, &mut sink).unwrap();
    assert!(receipt.snapshot().positions > 0);
    assert_decoded(&sink, &source);

    let mut sink = vec![1, 2, 3];
    assert!(
        encode_indexed_frames_streaming_to(
            &source,
            &[Frame::new(&source)],
            &[0],
            &options,
            &mut sink,
        )
        .is_err()
    );
    assert_eq!(sink, [1, 2, 3]);
    // More than eight groups and explicit groupwise policy make the control eligible.
    let descriptor =
        ImageDescriptor::new(385, 385, source.layout(), source.metadata().clone()).unwrap();
    let mut area_options = options.clone();
    area_options.group_size_shift = 0;
    area_options.modular.palette_policy = Some(PaletteOptions {
        groupwise: true,
        ..Default::default()
    });
    assert!(FrameAreaPlan::new(descriptor.clone(), &area_options).is_err());
    area_options.tile_patches = None;
    assert!(FrameAreaPlan::new(descriptor, &area_options).is_ok());
}
