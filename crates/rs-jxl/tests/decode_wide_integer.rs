//! Exact original integer samples, never an f32 projection oracle.
use rs_jxl::decode::{DecodeOptions, decode, decode_frame};
use rs_jxl::entropy::stream::Coding;
use rs_jxl::frame::{BlendInfo, Frame, FrameOptions};
use rs_jxl::metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::modular::encode::FixedTree;
use rs_jxl::modular::predictor::Predictor;
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout, encode, encode_frames};

fn options() -> EncodeOptions {
    EncodeOptions {
        modular: ModularOptions {
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            rct: None,
            lz77: false,
            entropy: Coding::Prefix,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn image(bits: u8, layout: PixelLayout, width: u32, height: u32) -> Image {
    let max = (1u32 << bits) - 1;
    let edges = [
        0,
        1,
        2,
        max,
        max - 1,
        max / 2,
        max / 2 + 1,
        16_777_215,
        16_777_216,
        16_777_217,
        16_777_219,
        1_073_741_823,
    ];
    let mut state = 0x12345678u32;
    let pixels = (0..width as usize * height as usize * layout.channels())
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (if i < edges.len() { edges[i] } else { state }) & max
        })
        .collect();
    let image = Image::new(width, height, layout, PixelData::U32(pixels)).unwrap();
    let mut metadata = image.metadata().clone();
    metadata.bit_depth = BitDepth::new(bits).unwrap();
    for extra in &mut metadata.extra_channels {
        extra.bit_depth = BitDepth::new(bits).unwrap();
    }
    image.with_metadata(metadata, vec![]).unwrap()
}

#[test]
fn all_wide_depths_preserve_adjacent_values_and_extremes() {
    for bits in 17..=31 {
        for layout in [PixelLayout::Gray, PixelLayout::Rgb, PixelLayout::Rgba] {
            let source = image(bits, layout, 13, 7);
            let bytes = encode(&source, &options()).unwrap();
            for threads in [1, 4] {
                let decoded = decode(
                    &bytes,
                    &DecodeOptions {
                        threads,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(decoded.complete);
                assert_eq!(
                    decoded.frames[0].pixels,
                    *source.pixels(),
                    "depth={bits}, {layout:?}, threads={threads}"
                );
            }
        }
    }
}

#[test]
fn multiple_groups_and_parallel_inverse_transforms_keep_full_words() {
    let source = image(31, PixelLayout::Rgb, 263, 269);
    let bytes = encode(&source, &options()).unwrap();
    for threads in [1, 4] {
        let decoded = decode(
            &bytes,
            &DecodeOptions {
                threads,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(decoded.frames[0].pixels, *source.pixels());
    }
    assert!(decode(&bytes[..bytes.len() / 2], &DecodeOptions::default()).is_err());
}

#[test]
fn wide_extra_plane_with_narrow_main_preserves_its_native_type() {
    let source = Image::new(4, 1, PixelLayout::Gray, PixelData::U8(vec![0, 1, 127, 255])).unwrap();
    let mut metadata = source.metadata().clone();
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 31).unwrap());
    let extra = PixelData::U32(vec![16_777_217, 2_147_483_647, 1, 0]);
    let source = source.with_metadata(metadata, vec![extra.clone()]).unwrap();
    let bytes = encode(&source, &options()).unwrap();
    let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.frames[0].pixels, *source.pixels());
    assert_eq!(decoded.frames[0].extra_planes, [extra]);
}

#[test]
fn unequal_alpha_depth_keeps_authoritative_original_plane() {
    let source = Image::new(4, 1, PixelLayout::Gray, PixelData::U8(vec![3, 7, 19, 251])).unwrap();
    let mut metadata = source.metadata().clone();
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Alpha, 31).unwrap());
    let alpha = PixelData::U32(vec![0, 16_777_217, 1_073_741_823, 2_147_483_647]);
    let source = source.with_metadata(metadata, vec![alpha.clone()]).unwrap();
    let bytes = encode(&source, &options()).unwrap();
    let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.frames[0].layout, PixelLayout::GrayAlpha);
    assert_eq!(decoded.frames[0].extra_planes, [alpha]);
    assert_eq!(
        decoded.frames[0].pixels,
        PixelData::U8(vec![3, 0, 7, 2, 19, 127, 251, 255])
    );
    let display_only = decode(
        &bytes,
        &DecodeOptions {
            include_extra_channels: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(display_only.frames[0].pixels, decoded.frames[0].pixels);
    assert!(display_only.frames[0].extra_planes.is_empty());
}

#[test]
fn cropped_replace_references_and_raw_layers_keep_low_bits() {
    let canvas = image(31, PixelLayout::Gray, 7, 5);
    let crop = image(31, PixelLayout::Gray, 3, 2);
    let frames = [
        Frame {
            image: &canvas,
            options: FrameOptions {
                save_as_reference: 1,
                ..Default::default()
            },
        },
        Frame {
            image: &crop,
            options: FrameOptions {
                origin: (2, 1),
                blend: BlendInfo {
                    source: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        },
    ];
    let bytes = encode_frames(&canvas, &frames, &options()).unwrap();
    let PixelData::U32(mut expected) = canvas.pixels().clone() else {
        unreachable!()
    };
    let PixelData::U32(changed) = crop.pixels() else {
        unreachable!()
    };
    for y in 0..2 {
        expected[(y + 1) * 7 + 2..(y + 1) * 7 + 5].copy_from_slice(&changed[y * 3..y * 3 + 3]);
    }
    let composed = decode(&bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(composed.frames[0].pixels, PixelData::U32(expected));
    let raw = DecodeOptions {
        coalescing: false,
        ..Default::default()
    };
    let decoded = decode(&bytes, &raw).unwrap();
    assert_eq!(decoded.frames[0].pixels, *canvas.pixels());
    assert_eq!(decoded.frames[1].pixels, *crop.pixels());
    assert_eq!(
        decode_frame(&bytes, 1, &raw).unwrap().frames[0].pixels,
        *crop.pixels()
    );
}

#[test]
fn wide_samples_survive_reversible_tools_and_entropy_routes() {
    for bits in [17, 24, 25, 31] {
        let source = image(bits, PixelLayout::Rgb, 13, 7);
        for route in 0..5 {
            // The ported encoder intentionally rejects an overflowing i32
            // squeeze average. Exercise that tool within its accepted domain,
            // still declaring 31 bits and retaining values not exact in f32.
            let source = if route == 3 && bits == 31 {
                let PixelData::U32(values) = source.pixels() else {
                    unreachable!()
                };
                Image::new(
                    13,
                    7,
                    PixelLayout::Rgb,
                    PixelData::U32(values.iter().map(|v| (v >> 2) | 1).collect()),
                )
                .unwrap()
                .with_metadata(source.metadata().clone(), vec![])
                .unwrap()
            } else {
                source.clone()
            };
            let mut settings = options();
            match route {
                0 => {
                    settings.modular.rct = Some(6);
                    settings.modular.entropy = Coding::Ans;
                }
                1 => {
                    settings.modular.rct = Some(41);
                    settings.modular.fixed_tree = Some(FixedTree::GradientLeaf);
                }
                2 => {
                    settings.modular.fixed_tree = Some(FixedTree::Weighted);
                    settings.modular.lz77 = true;
                }
                3 => {
                    settings.modular.squeeze = vec![true, false];
                }
                _ => {
                    settings.modular.palette_colors = Some(256);
                }
            }
            let bytes = encode(&source, &settings).unwrap();
            let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
            assert_eq!(
                decoded.frames[0].pixels,
                *source.pixels(),
                "bits={bits} route={route}"
            );
        }
    }
}

#[test]
fn native_wide_rejects_implicit_render_transforms_and_respects_output_budget() {
    let source = image(31, PixelLayout::Rgba, 4, 3);
    let bytes = encode(&source, &options()).unwrap();
    let mut limited = DecodeOptions::default();
    limited.limits.max_output_bytes = 4 * 3 * 5 * 4 - 1;
    assert!(matches!(
        decode(&bytes, &limited),
        Err(rs_jxl::Error::Limit(_))
    ));
    limited.limits.max_output_bytes += 1;
    assert_eq!(
        decode(&bytes, &limited).unwrap().frames[0].pixels,
        *source.pixels()
    );
    limited.alpha_mode = rs_jxl::decode::AlphaMode::Premultiplied;
    assert!(matches!(
        decode(&bytes, &limited),
        Err(rs_jxl::Error::InvalidInput(_))
    ));
}
