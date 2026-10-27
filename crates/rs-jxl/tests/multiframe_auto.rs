//! Exact-sample and frame-timing acceptance, not compressed-byte identity.
use rs_jxl::decode::{DecodeOptions, decode, decode_frame};
use rs_jxl::metadata::{Animation, BitDepth, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::multiframe::{AutoEncodeResult, AutoOptions, Reference, Strategy, encode_auto};
use rs_jxl::{EncodeOptions, Error, Image, PixelData, PixelLayout};

fn sequence(layout: PixelLayout, bits: u8, partial_alpha: bool) -> Vec<Image> {
    let (w, h) = (65, 49);
    let channels = layout.channels();
    let maximum = if bits == 8 { 255 } else { 65535 };
    let mut pixels: Vec<u16> = (0..w * h * channels)
        .map(|i| ((i * 1237 + i / 17 * 2017) as u32 & maximum) as u16)
        .collect();
    if layout.has_alpha() {
        for (i, pixel) in pixels.chunks_exact_mut(channels).enumerate() {
            pixel[channels - 1] = if partial_alpha && i % 3 == 0 {
                0
            } else if partial_alpha && i % 3 == 1 {
                (maximum / 2) as u16
            } else {
                maximum as u16
            };
        }
    }
    let mut frames = Vec::new();
    for index in 0..4 {
        let mut values = pixels.clone();
        if index == 1 || index == 2 {
            for p in [2 * w + 2, (h - 3) * w + w - 3] {
                values[p * channels] ^= 0x55;
            }
        }
        let data = if bits == 8 {
            PixelData::U8(values.into_iter().map(|v| v as u8).collect())
        } else {
            PixelData::U16(values)
        };
        let image = Image::new(w as u32, h as u32, layout, data).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.animation = Some(Animation::default());
        frames.push(image.with_metadata(metadata, vec![]).unwrap());
    }
    frames
}

fn verify(
    images: &[Image],
    durations: &[u32],
    auto: &AutoOptions,
    threads: usize,
) -> AutoEncodeResult {
    let result = encode_auto(
        images,
        durations,
        &EncodeOptions {
            threads,
            ..Default::default()
        },
        auto,
    )
    .unwrap();
    assert_eq!(result.frames.len(), images.len());
    for report in &result.frames {
        let winner = &report.candidates[report.winner_index];
        assert_eq!(
            report.winner_index,
            report
                .candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.encoded_bytes)
                .map(|(index, _)| index)
                .unwrap()
        );
        assert!(winner.encoded_frames > 0);
    }
    let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
    let visible: Vec<_> = images
        .iter()
        .zip(durations)
        .enumerate()
        .filter(|(i, (_, duration))| **duration != 0 || *i + 1 == images.len())
        .collect();
    assert_eq!(decoded.frames.len(), visible.len());
    for (frame, (_, (image, duration))) in decoded.frames.iter().zip(&visible) {
        assert_eq!(&frame.pixels, image.pixels());
        assert_eq!(frame.layout, image.layout());
        assert_eq!(frame.duration_ticks, **duration);
        assert_eq!(frame.width, image.width() as usize);
        assert_eq!(frame.height, image.height() as usize);
    }
    let last = decode_frame(&result.bytes, visible.len() - 1, &DecodeOptions::default()).unwrap();
    assert_eq!(last.frames.len(), 1);
    assert_eq!(&last.frames[0].pixels, images.last().unwrap().pixels());
    result
}

#[test]
fn strategies_preserve_integer_samples_hidden_rgb_and_timing() {
    for bits in [8, 16] {
        for layout in [
            PixelLayout::Gray,
            PixelLayout::GrayAlpha,
            PixelLayout::Rgb,
            PixelLayout::Rgba,
        ] {
            for partial in [false, true] {
                let images = sequence(layout, bits, partial);
                for reference in [
                    Reference::Full,
                    Reference::Previous,
                    Reference::First,
                    Reference::SmallestCrop,
                ] {
                    verify(
                        &images,
                        &[2, 3, 5, 7],
                        &AutoOptions {
                            reference,
                            ..Default::default()
                        },
                        1,
                    );
                }
            }
        }
    }
}

#[test]
fn zero_duration_and_mixed_layers_retain_semantics() {
    let images = sequence(PixelLayout::Rgba, 8, true);
    for durations in [[0, 0, 0, 0], [0, 3, 0, 7], [2, 0, 3, 0]] {
        verify(&images, &durations, &AutoOptions::default(), 4);
    }
    let layers: Vec<_> = images
        .into_iter()
        .map(|image| {
            let mut metadata = image.metadata().clone();
            metadata.animation = None;
            image.with_metadata(metadata, vec![]).unwrap()
        })
        .collect();
    verify(&layers, &[0; 4], &AutoOptions::default(), 1);
    assert!(
        encode_auto(
            &layers,
            &[1; 4],
            &EncodeOptions::default(),
            &AutoOptions::default()
        )
        .is_err()
    );
}

#[test]
fn zero_duration_extra_only_states_preserve_reference_slots() {
    let (w, h) = (65usize, 49usize);
    let points = [2 * w + 2, (h - 3) * w + w - 3, 24 * w + 32];
    let base: Vec<u16> = (0..w * h)
        .map(|i| (i * 1237 + i / 17 * 2017) as u16)
        .collect();
    let images: Vec<_> = [0u8, 3, 7, 4, 0, 5]
        .into_iter()
        .map(|changes| {
            let image = Image::new(
                w as u32,
                h as u32,
                PixelLayout::Gray,
                PixelData::U8(vec![17; w * h]),
            )
            .unwrap();
            let mut metadata = image.metadata().clone();
            metadata.animation = Some(Animation::default());
            metadata
                .extra_channels
                .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
            let mut values = base.clone();
            for (i, &point) in points.iter().enumerate() {
                if changes & (1 << i) != 0 {
                    values[point] ^= 0x55;
                }
            }
            image
                .with_metadata(metadata, vec![PixelData::U16(values)])
                .unwrap()
        })
        .collect();
    let durations = [0, 0, 3, 0, 0, 7];
    for reference in [
        Reference::Previous,
        Reference::First,
        Reference::SmallestCrop,
    ] {
        let result = verify(
            &images,
            &durations,
            &AutoOptions {
                reference,
                tile_size: 16,
                adaptive_rectangles: false,
                ..Default::default()
            },
            1,
        );
        assert_eq!(
            result.frames[1].candidates[result.frames[1].winner_index].encoded_frames, 2,
            "fixture must select multiple rectangles for an invisible logical state"
        );
        let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
        for (frame, index) in decoded.frames.iter().zip([2, 5]) {
            assert_eq!(frame.extra_planes, images[index].extra_planes());
        }
        let selected = decode_frame(&result.bytes, 1, &DecodeOptions::default()).unwrap();
        assert_eq!(selected.frames[0].extra_planes, images[5].extra_planes());

        let raw = decode(
            &result.bytes,
            &DecodeOptions {
                coalescing: false,
                ..Default::default()
            },
        )
        .unwrap();
        let mut layers = raw.frames.iter();
        for (index, report) in result.frames.iter().enumerate() {
            let winner = &report.candidates[report.winner_index];
            let source = match winner.strategy {
                Strategy::Full => 0,
                Strategy::CropFirst
                | Strategy::MaskFirst
                | Strategy::TilesFirst
                | Strategy::MergedFirst => 1,
                _ if index == 1 => 1,
                _ => 2,
            };
            for part in 0..winner.encoded_frames {
                let layer = layers.next().unwrap();
                let final_part = part + 1 == winner.encoded_frames;
                let last = index + 1 == images.len() && final_part;
                assert_eq!(
                    layer.duration_ticks,
                    if final_part { durations[index] } else { 0 }
                );
                assert_eq!(layer.header.is_last, last);
                let info = &layer.header.layer_info;
                assert_eq!(info.blend_info.source, if part == 0 { source } else { 2 });
                assert_eq!(info.extra_channel_blend_info, [info.blend_info]);
                assert_eq!(
                    info.save_as_reference,
                    if last {
                        None
                    } else if index == 0 {
                        Some(1)
                    } else {
                        Some(2)
                    }
                );
                assert!(!info.save_before_color_transform);
            }
        }
        assert!(layers.next().is_none());
    }
}

#[test]
fn opaque_masks_preserve_unequal_color_and_alpha_depths() {
    for (color_bits, alpha_bits) in [(8, 16), (16, 8), (12, 1)] {
        let color_max = ((1u32 << color_bits) - 1) as u16;
        let alpha_max = ((1u32 << alpha_bits) - 1) as u16;
        let images: Vec<_> = sequence(PixelLayout::Rgba, 16, false)
            .into_iter()
            .map(|image| {
                let PixelData::U16(mut values) = image.pixels().clone() else {
                    unreachable!()
                };
                for pixel in values.chunks_exact_mut(4) {
                    for value in &mut pixel[..3] {
                        *value &= color_max;
                    }
                    pixel[3] = alpha_max;
                }
                let mut metadata = image.metadata().clone();
                metadata.bit_depth = BitDepth::new(color_bits).unwrap();
                metadata.extra_channels[0].bit_depth = BitDepth::new(alpha_bits).unwrap();
                Image::new(
                    image.width(),
                    image.height(),
                    image.layout(),
                    PixelData::U16(values),
                )
                .unwrap()
                .with_metadata(metadata, vec![])
                .unwrap()
            })
            .collect();
        let result = encode_auto(
            &images,
            &[0, 2, 0, 3],
            &EncodeOptions::default(),
            &AutoOptions {
                reference: Reference::Previous,
                allow_multi_rect: false,
                adaptive_rectangles: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            result.frames[1].candidates[result.frames[1].winner_index].strategy,
            Strategy::MaskPrevious
        );
        let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.frames.len(), 2);
        assert_eq!(
            decoded.info.bit_depth.bits_per_sample(),
            u32::from(color_bits)
        );
        assert_eq!(
            decoded.info.extra_channels[0].bit_depth.bits_per_sample(),
            u32::from(alpha_bits)
        );
        for (frame, index) in decoded.frames.iter().zip([1, 3]) {
            let PixelData::U16(mut values) = images[index].pixels().clone() else {
                unreachable!()
            };
            // Interleaved alpha follows the color depth; its native plane is authoritative.
            for pixel in values.chunks_exact_mut(4) {
                pixel[3] = color_max;
            }
            let expected = if color_bits <= 8 {
                PixelData::U8(values.into_iter().map(|v| v as u8).collect())
            } else {
                PixelData::U16(values)
            };
            assert_eq!(frame.pixels, expected);
            let count = frame.width * frame.height;
            let alpha = if alpha_bits <= 8 {
                PixelData::U8(vec![alpha_max as u8; count])
            } else {
                PixelData::U16(vec![alpha_max; count])
            };
            assert_eq!(frame.extra_planes, [alpha]);
        }
    }
}

#[test]
fn exact_input_output_and_trial_budget_boundaries() {
    let images: Vec<_> = [0, 1, 0]
        .into_iter()
        .map(|value| {
            let mut samples = vec![0; 15];
            samples[7] = value;
            Image::new(5, 3, PixelLayout::Gray, PixelData::U16(samples)).unwrap()
        })
        .collect();
    // Three full trials plus three 1x1 crops, including the identical first reference.
    let auto = AutoOptions {
        allow_mask: false,
        allow_multi_rect: false,
        adaptive_rectangles: false,
        max_frames: 3,
        max_total_trials: 6,
        max_trial_pixels: 48,
        ..Default::default()
    };
    let options = EncodeOptions {
        max_pixels: 15,
        ..Default::default()
    };
    let result = encode_auto(&images, &[0; 3], &options, &auto).unwrap();
    assert_eq!(
        result
            .frames
            .iter()
            .map(|f| f.candidates.len() + f.pruned.len())
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert_eq!(
        result
            .frames
            .iter()
            .map(|f| {
                f.candidates.iter().map(|c| c.trial_pixels).sum::<u64>()
                    + f.pruned.iter().map(|c| c.trial_pixels).sum::<u64>()
            })
            .sum::<u64>(),
        48
    );
    assert_eq!(&result.bytes[..12], b"\0\0\0\x0cJXL \x0d\x0a\x87\x0a");
    let bounded = EncodeOptions {
        max_output_bytes: result.bytes.len(),
        ..options.clone()
    };
    let bounded_result = encode_auto(&images, &[0; 3], &bounded, &auto).unwrap();
    assert_eq!(bounded_result.bytes.len(), result.bytes.len());
    let decoded = decode(&bounded_result.bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.frames[0].pixels, *images[2].pixels());
    for limited in [
        AutoOptions {
            max_frames: 2,
            ..auto.clone()
        },
        AutoOptions {
            max_total_trials: 5,
            ..auto.clone()
        },
        AutoOptions {
            max_trial_pixels: 47,
            ..auto.clone()
        },
    ] {
        assert!(matches!(
            encode_auto(&images, &[0; 3], &options, &limited),
            Err(Error::Limit(_))
        ));
    }
    for limited in [
        EncodeOptions {
            max_pixels: 14,
            ..options.clone()
        },
        EncodeOptions {
            max_output_bytes: result.bytes.len() - 1,
            ..options.clone()
        },
    ] {
        assert!(matches!(
            encode_auto(&images, &[0; 3], &limited, &auto),
            Err(Error::Limit(_))
        ));
    }
    let different_depth = Image::new(5, 3, PixelLayout::Gray, PixelData::U8(vec![0; 15])).unwrap();
    assert!(matches!(
        encode_auto(
            &[images[0].clone(), different_depth],
            &[0; 2],
            &options,
            &auto
        ),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn scattered_changes_select_multiple_replace_layers() {
    let images = sequence(PixelLayout::Rgba, 8, true);
    let auto = AutoOptions {
        tile_size: 16,
        adaptive_rectangles: false,
        ..Default::default()
    };
    let result = encode_auto(&images, &[1; 4], &EncodeOptions::default(), &auto).unwrap();
    let second = &result.frames[1];
    assert!(
        second
            .candidates
            .iter()
            .any(|c| c.strategy == Strategy::TilesPrevious)
    );
    assert_eq!(
        second.candidates[second.winner_index].strategy,
        Strategy::TilesPrevious
    );
    assert_eq!(second.candidates[second.winner_index].encoded_frames, 2);
    assert!(
        !second
            .candidates
            .iter()
            .map(|c| c.strategy)
            .chain(second.pruned.iter().map(|c| c.strategy))
            .any(|strategy| matches!(strategy, Strategy::MaskPrevious | Strategy::MaskFirst))
    );
    verify(&images, &[1; 4], &auto, 1);
}

#[test]
fn binary_mask_is_tested_only_for_opaque_existing_alpha() {
    for bits in [8, 16] {
        for layout in [PixelLayout::GrayAlpha, PixelLayout::Rgba] {
            let images = sequence(layout, bits, false);
            let auto = AutoOptions {
                allow_multi_rect: false,
                adaptive_rectangles: false,
                ..Default::default()
            };
            let result = encode_auto(&images, &[1; 4], &EncodeOptions::default(), &auto).unwrap();
            let second = &result.frames[1];
            assert!(
                second
                    .candidates
                    .iter()
                    .any(|c| c.strategy == Strategy::MaskPrevious)
            );
            assert_eq!(
                second.candidates[second.winner_index].strategy,
                Strategy::MaskPrevious
            );
            verify(&images, &[1; 4], &auto, 1);
        }
    }
}

#[test]
fn first_reference_and_identical_frames_remain_available() {
    let images = sequence(PixelLayout::Rgb, 16, false);
    let auto = AutoOptions {
        adaptive_rectangles: false,
        ..Default::default()
    };
    let result = encode_auto(&images, &[1; 4], &EncodeOptions::default(), &auto).unwrap();
    let last = result.frames.last().unwrap();
    assert_eq!(
        last.candidates[last.winner_index].strategy,
        Strategy::CropFirst
    );
    let same = &result.frames[2];
    assert_eq!(
        same.candidates[same.winner_index].strategy,
        Strategy::CropPrevious
    );
    verify(&images, &[1; 4], &auto, 1);
}

#[test]
fn planar_extras_participate_and_subsampled_extras_force_full_frames() {
    for shift in [0, 1] {
        let mut images = sequence(PixelLayout::Rgb, 8, false);
        for (i, image) in images.iter_mut().enumerate() {
            let mut metadata = image.metadata().clone();
            let mut ec = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
            ec.dim_shift = shift;
            metadata.extra_channels.push(ec);
            let n = image.width().div_ceil(1 << shift) * image.height().div_ceil(1 << shift);
            let mut values = vec![1000; n as usize];
            values[0] = 1000 + i as u16;
            *image = image
                .clone()
                .with_metadata(metadata, vec![PixelData::U16(values)])
                .unwrap();
        }
        let result = encode_auto(
            &images,
            &[1; 4],
            &EncodeOptions::default(),
            &AutoOptions::default(),
        )
        .unwrap();
        let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
        let full = encode_auto(
            &images,
            &[1; 4],
            &EncodeOptions::default(),
            &AutoOptions {
                reference: Reference::Full,
                ..Default::default()
            },
        )
        .unwrap();
        let full = decode(&full.bytes, &DecodeOptions::default()).unwrap();
        for ((frame, expected), source) in decoded.frames.iter().zip(&full.frames).zip(&images) {
            assert_eq!(&frame.pixels, source.pixels());
            assert_eq!(frame.extra_planes, expected.extra_planes);
        }
        if shift != 0 {
            assert!(
                result
                    .frames
                    .iter()
                    .all(|f| f.candidates.len() == 1 && f.pruned.is_empty())
            );
        }
    }
}

#[test]
fn floating_words_are_not_used_for_alpha_masking() {
    let mut images = sequence(PixelLayout::Rgb, 8, false);
    for image in &mut images {
        let PixelData::U8(values) = image.pixels() else {
            unreachable!()
        };
        let values = values.iter().map(|&v| f32::from(v) / 256.0).collect();
        let mut metadata = image.metadata().clone();
        metadata.bit_depth = BitDepth::floating(32, 8).unwrap();
        *image = Image::new(
            image.width(),
            image.height(),
            image.layout(),
            PixelData::F32(values),
        )
        .unwrap()
        .with_metadata(metadata, vec![])
        .unwrap();
    }
    verify(&images, &[1; 4], &AutoOptions::default(), 1);
}

#[test]
fn float_special_words_are_preserved_by_replace_deltas() {
    let words = [
        0x8000_0000,
        0,
        0x7fc1_2345,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0001,
    ];
    let images: Vec<_> = (0..3)
        .map(|i| {
            let mut values: Vec<f32> = (0..64)
                .map(|p| f32::from_bits(words[p % words.len()]))
                .collect();
            if i == 1 {
                values[2] = f32::from_bits(0x7fc5_4321);
                values[61] = -0.0;
            }
            let image = Image::new(8, 8, PixelLayout::Gray, PixelData::F32(values)).unwrap();
            let mut metadata = image.metadata().clone();
            metadata.animation = Some(Animation::default());
            image.with_metadata(metadata, vec![]).unwrap()
        })
        .collect();
    verify(
        &images,
        &[1; 3],
        &AutoOptions {
            tile_size: 2,
            ..Default::default()
        },
        1,
    );
}

#[test]
fn premultiplied_or_additional_alpha_is_never_implicitly_masked() {
    for extra in [false, true] {
        let mut images = sequence(PixelLayout::Rgba, 16, false);
        for image in &mut images {
            let mut metadata = image.metadata().clone();
            metadata.extra_channels[0].alpha_associated = !extra;
            let planes = if extra {
                metadata
                    .extra_channels
                    .push(ExtraChannelInfo::new(ExtraChannelType::Alpha, 8).unwrap());
                vec![PixelData::U8(vec![
                    128;
                    image.width() as usize
                        * image.height() as usize
                ])]
            } else {
                vec![]
            };
            *image = image.clone().with_metadata(metadata, planes).unwrap();
        }
        let result = encode_auto(
            &images,
            &[1; 4],
            &EncodeOptions::default(),
            &AutoOptions::default(),
        )
        .unwrap();
        assert!(
            result
                .frames
                .iter()
                .flat_map(|f| {
                    f.candidates
                        .iter()
                        .map(|c| c.strategy)
                        .chain(f.pruned.iter().map(|c| c.strategy))
                })
                .all(|strategy| !matches!(strategy, Strategy::MaskPrevious | Strategy::MaskFirst))
        );
        verify(&images, &[1; 4], &AutoOptions::default(), 1);
    }
}

#[test]
fn invalid_inputs_and_sequence_work_limits_are_errors() {
    let images = sequence(PixelLayout::Rgb, 8, false);
    for auto in [
        AutoOptions {
            max_frames: 1,
            ..Default::default()
        },
        AutoOptions {
            max_total_trials: 1,
            ..Default::default()
        },
        AutoOptions {
            max_trial_pixels: 1,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            encode_auto(&images, &[1; 4], &EncodeOptions::default(), &auto),
            Err(Error::Limit(_))
        ));
    }
    assert!(
        encode_auto(
            &images,
            &[1; 3],
            &EncodeOptions::default(),
            &AutoOptions::default()
        )
        .is_err()
    );
    assert!(encode_auto(&[], &[], &EncodeOptions::default(), &AutoOptions::default()).is_err());
    assert!(
        encode_auto(
            &images,
            &[1; 4],
            &EncodeOptions {
                max_output_bytes: 1,
                ..Default::default()
            },
            &AutoOptions::default()
        )
        .is_err()
    );
    assert!(
        encode_auto(
            &images,
            &[1; 4],
            &EncodeOptions::default(),
            &AutoOptions {
                tile_size: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
}
