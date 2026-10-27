use rs_jxl::{
    EncodeOptions, Image, PixelData, PixelLayout,
    codec::Encoder,
    decode::{DecodeOptions, decode, decode_frame},
    frame::{BlendInfo, Frame, FrameOptions},
    layers::{self, LayerOptions, default_frame_options},
    metadata::{Animation, ExtraChannelInfo, ExtraChannelType},
};

fn image(width: u32, height: u32, bits: u8) -> Image {
    let count = width as usize * height as usize * 4;
    let pixels = if bits == 8 {
        PixelData::U8(
            (0..count)
                .map(|i| if i % 8 == 3 { 0 } else { (i * 73 + 5) as u8 })
                .collect(),
        )
    } else {
        PixelData::U16(
            (0..count)
                .map(|i| if i % 8 == 3 { 0 } else { (i * 997 + 5) as u16 })
                .collect(),
        )
    };
    Image::new(width, height, PixelLayout::Rgba, pixels).unwrap()
}

fn raw_options() -> DecodeOptions {
    DecodeOptions {
        coalescing: false,
        ..Default::default()
    }
}

fn exact(actual: &PixelData, expected: &PixelData) {
    match (actual, expected) {
        (PixelData::F32(a), PixelData::F16(b)) => {
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().copied().map(half_to_f32_bits).collect::<Vec<_>>()
            );
        }
        (PixelData::F32(a), PixelData::F32(b)) => {
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        _ => assert_eq!(actual, expected),
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
fn original_shapes_hidden_rgb_and_random_layer_access() {
    for bits in [8, 16] {
        for threads in [1, 4] {
            let images = [image(3, 7, bits), image(9, 2, bits), image(1, 4, bits)];
            let options = EncodeOptions {
                threads,
                ..Default::default()
            };
            let output = layers::encode(&images, &options, &Default::default()).unwrap();
            assert_eq!((output.width, output.height), (9, 7));
            assert_eq!(output.layers.len(), images.len());
            let raw = decode(&output.bytes, &raw_options()).unwrap();
            assert_eq!(raw.frames.len(), images.len());
            for (i, (frame, expected)) in raw.frames.iter().zip(&images).enumerate() {
                assert_eq!(
                    (frame.width, frame.height),
                    (expected.width() as usize, expected.height() as usize)
                );
                exact(&frame.pixels, expected.pixels());
                assert_eq!(frame.duration_ticks, 0);
                exact(
                    &decode_frame(&output.bytes, i, &raw_options())
                        .unwrap()
                        .frames[0]
                        .pixels,
                    expected.pixels(),
                );
            }
            let composite = decode(&output.bytes, &Default::default()).unwrap();
            assert_eq!(composite.frames.len(), 1);
            let full = &composite.frames[0];
            assert_eq!((full.width, full.height), (9, 7));
            match (&full.pixels, images[2].pixels()) {
                (PixelData::U8(actual), PixelData::U8(expected)) => {
                    for y in 0..7 {
                        for x in 0..9 {
                            for c in 0..4 {
                                assert_eq!(
                                    actual[(y * 9 + x) * 4 + c],
                                    if x == 0 && y < 4 {
                                        expected[y * 4 + c]
                                    } else {
                                        0
                                    }
                                );
                            }
                        }
                    }
                }
                (PixelData::U16(actual), PixelData::U16(expected)) => {
                    for y in 0..7 {
                        for x in 0..9 {
                            for c in 0..4 {
                                assert_eq!(
                                    actual[(y * 9 + x) * 4 + c],
                                    if x == 0 && y < 4 {
                                        expected[y * 4 + c]
                                    } else {
                                        0
                                    }
                                );
                            }
                        }
                    }
                }
                _ => panic!("unexpected pixels"),
            }
        }
    }
}

#[test]
fn explicit_canvas_offsets_and_clipping_do_not_crop_raw_pixels() {
    let images = [image(7, 4, 8), image(3, 8, 8)];
    let layers = LayerOptions {
        canvas: Some((5, 6)),
        frames: vec![
            FrameOptions {
                origin: (-2, 1),
                duration: 2,
                ..default_frame_options()
            },
            FrameOptions {
                origin: (4, -3),
                duration: 3,
                ..default_frame_options()
            },
        ],
        ..Default::default()
    };
    let output = layers::encode(&images, &Default::default(), &layers).unwrap();
    let raw = decode(&output.bytes, &raw_options()).unwrap();
    assert_eq!(raw.frames.len(), 2);
    for (i, expected) in images.iter().enumerate() {
        exact(&raw.frames[i].pixels, expected.pixels());
        assert_eq!(
            (raw.frames[i].width, raw.frames[i].height),
            (expected.width() as usize, expected.height() as usize)
        );
    }
    let composite = decode(&output.bytes, &Default::default()).unwrap();
    assert_eq!(composite.frames.len(), 2);
    for (index, frame) in composite.frames.iter().enumerate() {
        assert_eq!((frame.width, frame.height), (5, 6));
        assert_eq!(frame.duration_ticks, layers.frames[index].duration);
        let PixelData::U8(actual) = &frame.pixels else {
            panic!()
        };
        let PixelData::U8(source) = images[index].pixels() else {
            panic!()
        };
        let (ox, oy) = layers.frames[index].origin;
        for y in 0..6 {
            for x in 0..5 {
                for c in 0..4 {
                    let (sx, sy) = (x as i32 - ox, y as i32 - oy);
                    let expected = if sx >= 0
                        && sx < images[index].width() as i32
                        && sy >= 0
                        && sy < images[index].height() as i32
                    {
                        source[(sy as usize * images[index].width() as usize + sx as usize) * 4 + c]
                    } else {
                        0
                    };
                    assert_eq!(actual[(y * 5 + x) * 4 + c], expected);
                }
            }
        }
    }
}

#[test]
fn explicit_reference_reuses_background_without_inventing_pixels() {
    let images = [
        Image::new(4, 3, PixelLayout::Gray, PixelData::U8(vec![7; 12])).unwrap(),
        Image::new(2, 1, PixelLayout::Gray, PixelData::U8(vec![9, 11])).unwrap(),
    ];
    let config = LayerOptions {
        frames: vec![
            FrameOptions {
                duration: 1,
                save_as_reference: 2,
                ..default_frame_options()
            },
            FrameOptions {
                origin: (1, 1),
                duration: 1,
                blend: BlendInfo {
                    source: 2,
                    ..Default::default()
                },
                ..default_frame_options()
            },
        ],
        ..Default::default()
    };
    let output = layers::encode(&images, &Default::default(), &config).unwrap();
    let decoded = decode(&output.bytes, &Default::default()).unwrap();
    assert_eq!(
        decoded.frames[1].pixels,
        PixelData::U8(vec![7, 7, 7, 7, 7, 9, 11, 7, 7, 7, 7, 7])
    );
    exact(
        &decode_frame(&output.bytes, 1, &raw_options())
            .unwrap()
            .frames[0]
            .pixels,
        images[1].pixels(),
    );
}

#[test]
fn raw_float_and_wide_integer_words_are_preserved() {
    for (a, b) in [
        (
            PixelData::F32(vec![-0.0, f32::from_bits(0x7fc0_0111), 1.0, f32::INFINITY]),
            PixelData::F32(vec![f32::from_bits(0x7fc0_0123), -0.0]),
        ),
        (
            PixelData::F16(vec![0x8000, 0x7e01, 0x3c00, 0x7c00]),
            PixelData::F16(vec![0x7e02, 0x8000]),
        ),
        (
            PixelData::U32(vec![1, 0x7fff_ffff, 0x0123_4567, 10]),
            PixelData::U32(vec![0x7654_3210, 2]),
        ),
    ] {
        let images = [
            Image::new(2, 2, PixelLayout::Gray, a).unwrap(),
            Image::new(1, 2, PixelLayout::Gray, b).unwrap(),
        ];
        let output = layers::encode(
            &images,
            &Default::default(),
            &LayerOptions {
                canvas: Some((3, 4)),
                ..Default::default()
            },
        )
        .unwrap();
        let decoded = decode(&output.bytes, &raw_options()).unwrap();
        assert_eq!(decoded.frames.len(), 2);
        for (actual, expected) in decoded.frames.iter().zip(&images) {
            exact(&actual.pixels, expected.pixels());
        }
    }
}

#[test]
fn native_extra_channels_with_different_plane_sizes() {
    let images: Vec<_> = [(3, 4), (7, 2)]
        .into_iter()
        .map(|(w, h)| {
            let image = Image::new(
                w,
                h,
                PixelLayout::Gray,
                PixelData::U8(vec![5; w as usize * h as usize]),
            )
            .unwrap();
            let mut metadata = image.metadata().clone();
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
            image
                .with_metadata(
                    metadata,
                    vec![PixelData::U16((0..w * h).map(|i| i as u16 * 997).collect())],
                )
                .unwrap()
        })
        .collect();
    let output = layers::encode(&images, &Default::default(), &Default::default()).unwrap();
    let decoded = decode(
        &output.bytes,
        &DecodeOptions {
            include_extra_channels: true,
            ..raw_options()
        },
    )
    .unwrap();
    for (actual, expected) in decoded.frames.iter().zip(&images) {
        exact(&actual.pixels, expected.pixels());
        assert_eq!(&actual.extra_planes[0], &expected.extra_planes()[0]);
    }
}

#[test]
fn empty_reference_is_standard_zero_background_in_low_level_api() {
    let canvas = Image::new(4, 3, PixelLayout::Gray, PixelData::U8(vec![255; 12])).unwrap();
    let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![17])).unwrap();
    for source in 0..4 {
        let frame = Frame {
            image: &image,
            options: FrameOptions {
                origin: (1, 2),
                blend: BlendInfo {
                    source,
                    ..Default::default()
                },
                ..Default::default()
            },
        };
        let data = rs_jxl::encode_frames(&canvas, &[frame], &Default::default()).unwrap();
        let output = decode(&data, &Default::default()).unwrap();
        assert_eq!(
            output.frames[0].pixels,
            PixelData::U8(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0])
        );
    }
}

#[test]
fn pretransform_sources_remain_invalid_for_frame_blending() {
    let images = [image(4, 3, 8), image(2, 1, 8)];
    let config = LayerOptions {
        frames: vec![
            FrameOptions {
                save_as_reference: 1,
                save_before_color_transform: true,
                ..default_frame_options()
            },
            default_frame_options(),
        ],
        ..Default::default()
    };
    let error = layers::encode(&images, &Default::default(), &config).unwrap_err();
    assert!(error.to_string().contains("pre-transform reference"));
}

#[test]
fn generated_patch_atlases_do_not_change_regular_layer_indices() {
    use rs_jxl::tools::patches::{ImageView, PatchOptions, discover};
    let patch_options = PatchOptions::default();
    let images: Vec<_> = [(48usize, 32usize), (64, 40)]
        .into_iter()
        .map(|(width, height)| {
            let mut pixels = vec![0u8; width * height];
            for x in [8, 28] {
                for y in 8..16 {
                    pixels[y * width + x..y * width + x + 8].fill(173);
                }
            }
            let planes = vec![pixels.iter().map(|&v| i32::from(v)).collect()];
            let view = ImageView {
                width,
                height,
                bit_depth: 8,
                color_channels: 1,
                planes: &planes,
            };
            assert!(discover(view, &patch_options).unwrap().is_some());
            Image::new(
                width as u32,
                height as u32,
                PixelLayout::Gray,
                PixelData::U8(pixels),
            )
            .unwrap()
        })
        .collect();
    let output = layers::encode(
        &images,
        &EncodeOptions {
            patches: Some(patch_options),
            ..Default::default()
        },
        &Default::default(),
    )
    .unwrap();
    let raw = decode(&output.bytes, &raw_options()).unwrap();
    assert_eq!(raw.frames.len(), images.len());
    for (index, expected) in images.iter().enumerate() {
        exact(&raw.frames[index].pixels, expected.pixels());
        exact(
            &decode_frame(&output.bytes, index, &raw_options())
                .unwrap()
                .frames[0]
                .pixels,
            expected.pixels(),
        );
    }
}

#[test]
fn validation_precedes_work_and_context_recovers_after_error() {
    let first = image(3, 4, 8);
    let images = [first.clone(), image(2, 5, 8)];
    let mut context = Encoder::new(2).unwrap();
    let options = EncodeOptions {
        threads: 2,
        ..Default::default()
    };
    assert!(
        context
            .encode_layers(&[], &options, &Default::default())
            .is_err()
    );
    for config in [
        LayerOptions {
            canvas: Some((0, 1)),
            ..Default::default()
        },
        LayerOptions {
            canvas: Some((1u32 << 31, 1)),
            ..Default::default()
        },
        LayerOptions {
            frames: vec![default_frame_options()],
            ..Default::default()
        },
        LayerOptions {
            frames: vec![
                FrameOptions {
                    origin: (-1, 0),
                    ..default_frame_options()
                },
                default_frame_options(),
            ],
            ..Default::default()
        },
        LayerOptions {
            frames: vec![
                default_frame_options(),
                FrameOptions {
                    save_as_reference: 1,
                    ..default_frame_options()
                },
            ],
            ..Default::default()
        },
        LayerOptions {
            frames: vec![
                FrameOptions {
                    reference_only: true,
                    ..default_frame_options()
                },
                default_frame_options(),
            ],
            ..Default::default()
        },
        LayerOptions {
            animation: Some(Animation {
                ticks_per_second_numerator: 0,
                ..Default::default()
            }),
            ..Default::default()
        },
    ] {
        assert!(context.encode_layers(&images, &options, &config).is_err());
    }
    assert!(
        context
            .encode_layers(&images, &Default::default(), &Default::default())
            .is_err()
    );
    let other = Image::new(3, 4, PixelLayout::Rgb, PixelData::U8(vec![0; 36])).unwrap();
    assert!(
        context
            .encode_layers(&[first, other], &options, &Default::default())
            .is_err()
    );
    assert!(
        context
            .encode_layers(&images, &options, &Default::default())
            .is_ok()
    );
}

#[test]
fn exact_output_and_canvas_pixel_limits() {
    let images = [image(3, 4, 16), image(6, 2, 16)];
    let layers = LayerOptions::default();
    let encoded = layers::encode(&images, &Default::default(), &layers).unwrap();
    let exact = EncodeOptions {
        max_pixels: 24,
        max_output_bytes: encoded.bytes.len(),
        ..Default::default()
    };
    assert!(layers::encode(&images, &exact, &layers).is_ok());
    assert!(
        layers::encode(
            &images,
            &EncodeOptions {
                max_output_bytes: exact.max_output_bytes - 1,
                ..exact.clone()
            },
            &layers
        )
        .is_err()
    );
    assert!(
        layers::encode(
            &images,
            &EncodeOptions {
                max_pixels: 23,
                ..exact
            },
            &layers
        )
        .is_err()
    );
}
