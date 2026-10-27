use super::*;
use crate::{
    PixelData, PixelLayout, codestream,
    decode::{DecodeOptions, decode},
    encode::{encode_logical_frame_trial, wrap_stream},
    frame::{BlendInfo, FrameOptions},
    metadata::{Animation, ExtraChannelInfo, ExtraChannelType},
};

fn animated(image: Image) -> Image {
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    let extras = image.extra_planes().to_vec();
    image.with_metadata(metadata, extras).unwrap()
}

fn gray(width: u32, height: u32) -> Image {
    Image::new(
        width,
        height,
        PixelLayout::Gray,
        PixelData::U8(
            (0..width * height)
                .map(|i| i.wrapping_mul(17) as u8)
                .collect(),
        ),
    )
    .unwrap()
}

fn assert_decoded(canvas: &Image, frame_bytes: Vec<u8>, options: &EncodeOptions) {
    let mut stream = codestream::image_header(canvas, options.max_output_bytes).unwrap();
    stream.extend(frame_bytes);
    let bytes = wrap_stream(canvas, stream, options.max_output_bytes).unwrap();
    let decoded = decode(&bytes, &Default::default()).unwrap();
    assert_eq!(&decoded.frames[0].pixels, canvas.pixels());
}

fn assert_samples(actual: &PixelData, expected: &PixelData) {
    if let (PixelData::F32(actual), PixelData::F32(expected)) = (actual, expected) {
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    } else {
        assert_eq!(actual, expected);
    }
}

fn assert_cached_roundtrip(image: Image, options: &EncodeOptions, section_count: usize) {
    crate::execution::serial(|| {
        let image = animated(image);
        let mut cache = LogicalFrameCache::default();
        let mut stream = codestream::image_header(&image, options.max_output_bytes).unwrap();
        let initial = Frame {
            image: &image,
            options: FrameOptions {
                duration: 1,
                save_as_reference: 1,
                ..Default::default()
            },
        };
        assert!(!cache.contains(&()));
        stream.extend(
            encode_logical_frame_cached_trial(
                &image,
                &initial,
                options,
                false,
                None,
                &mut cache,
                (),
            )
            .unwrap()
            .into_complete()
            .unwrap(),
        );
        assert!(cache.contains(&()));
        assert_eq!(cache.entry.as_ref().unwrap().1.len(), section_count);
        let stored = cache.entry.as_ref().unwrap().1.as_ptr();
        let last = Frame {
            image: &image,
            options: FrameOptions {
                duration: 300,
                name: "cached final frame".into(),
                ..Default::default()
            },
        };
        let hit =
            encode_logical_frame_cached_trial(&image, &last, options, true, None, &mut cache, ())
                .unwrap()
                .into_complete()
                .unwrap();
        assert!(cache.contains(&()));
        assert_eq!(cache.entry.as_ref().unwrap().1.as_ptr(), stored);
        let ordinary = encode_logical_frame_trial(&image, &last, options, true, None)
            .unwrap()
            .into_complete()
            .unwrap();
        assert_eq!(hit.len(), ordinary.len());
        stream.extend(hit);
        let bytes = wrap_stream(&image, stream, options.max_output_bytes).unwrap();
        let decoded = decode(
            &bytes,
            &DecodeOptions {
                coalescing: false,
                include_extra_channels: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(decoded.complete);
        assert_eq!(decoded.frames.len(), 2);
        for (index, frame) in decoded.frames.iter().enumerate() {
            assert!(frame.complete);
            assert_eq!(
                (frame.width, frame.height, frame.layout),
                (
                    image.width() as usize,
                    image.height() as usize,
                    image.layout()
                )
            );
            assert_eq!(frame.duration_ticks, if index == 0 { 1 } else { 300 });
            assert_eq!(
                frame.name,
                if index == 0 { "" } else { "cached final frame" }
            );
            assert_eq!(frame.header.is_last, index == 1);
            assert_eq!(
                frame.header.layer_info.save_as_reference,
                if index == 0 { Some(1) } else { None }
            );
            assert_samples(&frame.pixels, image.pixels());
            assert_eq!(frame.extra_planes.len(), image.extra_planes().len());
            for (actual, expected) in frame.extra_planes.iter().zip(image.extra_planes()) {
                assert_samples(actual, expected);
            }
        }
    });
}

#[test]
fn cache_hit_preserves_u16_samples() {
    let samples = [0, 1, 255, 256, 32767, 32768, 65534, 65535].repeat(3);
    let image = Image::new(8, 3, PixelLayout::Gray, PixelData::U16(samples)).unwrap();
    assert_cached_roundtrip(image, &EncodeOptions::default(), 1);
}

#[test]
fn cache_hit_preserves_independent_full_resolution_extra_plane() {
    let image = gray(8, 3);
    let mut metadata = image.metadata().clone();
    let mut depth = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
    depth.name = "independent depth".into();
    assert_eq!(depth.dim_shift, 0);
    metadata.extra_channels.push(depth);
    let extra = PixelData::U16((0..24u16).map(|i| i.wrapping_mul(4051)).collect());
    let image = image.with_metadata(metadata, vec![extra]).unwrap();
    assert_cached_roundtrip(image, &EncodeOptions::default(), 1);
}

#[test]
fn cache_hit_preserves_exact_float_words() {
    let words = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x3f80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
    ]
    .repeat(3);
    let image = Image::new(
        8,
        3,
        PixelLayout::Gray,
        PixelData::F32(words.into_iter().map(f32::from_bits).collect()),
    )
    .unwrap();
    assert_cached_roundtrip(image, &EncodeOptions::default(), 1);
}

#[test]
fn cache_hit_reframes_multigroup_progressive_sections() {
    let options = EncodeOptions {
        group_size_shift: 0,
        responsive: true,
        passes: crate::modular::layout::Passes::progressive(),
        ..Default::default()
    };
    // Three 128-wide groups, three passes, one DC group and two globals.
    assert_cached_roundtrip(gray(257, 17), &options, 12);
}

#[test]
fn cached_payload_reframes_sources_saves_origins_timing_and_last() {
    crate::execution::serial(|| {
        let canvas = animated(
            Image::new(
                9,
                7,
                PixelLayout::Rgba,
                PixelData::U8([11, 29, 53, 0].repeat(9 * 7)),
            )
            .unwrap(),
        );
        let patch = animated(
            Image::new(
                3,
                2,
                PixelLayout::Rgba,
                PixelData::U8([231, 34, 17, 0].repeat(3 * 2)),
            )
            .unwrap(),
        );
        for route in 0..5 {
            let mut options = EncodeOptions::default();
            match route {
                0 => {}
                1 => options.shared_tree = true,
                2 => options.global_palette = true,
                3 => options.responsive = true,
                4 => options.modular.palette_policy = Some(Default::default()),
                _ => unreachable!(),
            }
            let initial = Frame {
                image: &canvas,
                options: FrameOptions {
                    duration: 1,
                    save_as_reference: 1,
                    ..Default::default()
                },
            };
            let mut stream = codestream::image_header(&canvas, options.max_output_bytes).unwrap();
            stream.extend(
                encode_logical_frame_trial(&canvas, &initial, &options, false, None)
                    .unwrap()
                    .into_complete()
                    .unwrap(),
            );
            let mut cache = LogicalFrameCache::default();
            let mut expected = [11, 29, 53, 0].repeat(9 * 7);
            let mut visible = vec![(expected.clone(), 1)];
            for (part, (x, y, source, save, duration)) in
                [(2, 1, 1, 2, 0), (4, 3, 2, 1, 300), (0, 4, 1, 0, 7)]
                    .into_iter()
                    .enumerate()
            {
                let last = part == 2;
                let frame = Frame {
                    image: &patch,
                    options: FrameOptions {
                        origin: (x as i32, y as i32),
                        duration,
                        save_as_reference: save,
                        blend: BlendInfo {
                            source,
                            ..Default::default()
                        },
                        name: "x".repeat(part * 40),
                        ..Default::default()
                    },
                };
                assert_eq!(cache.contains(&0u8), part != 0);
                let cached = encode_logical_frame_cached_trial(
                    &canvas, &frame, &options, last, None, &mut cache, 0u8,
                )
                .unwrap()
                .into_complete()
                .unwrap();
                let ordinary = encode_logical_frame_trial(&canvas, &frame, &options, last, None)
                    .unwrap()
                    .into_complete()
                    .unwrap();
                assert_eq!(cached.len(), ordinary.len());
                stream.extend(cached);
                for row in y..y + 2 {
                    for col in x..x + 3 {
                        expected[(row * 9 + col) * 4..(row * 9 + col + 1) * 4]
                            .copy_from_slice(&[231, 34, 17, 0]);
                    }
                }
                if duration != 0 {
                    visible.push((expected.clone(), duration));
                }
            }
            let bytes = wrap_stream(&canvas, stream, options.max_output_bytes).unwrap();
            let decoded = decode(&bytes, &Default::default()).unwrap();
            assert_eq!(decoded.frames.len(), visible.len());
            for (frame, (pixels, duration)) in decoded.frames.iter().zip(visible) {
                assert_eq!(frame.pixels, PixelData::U8(pixels));
                assert_eq!(frame.duration_ticks, duration);
            }
        }
    });
}

#[test]
fn cache_keeps_only_complete_payload_and_rechecks_fresh_trial_budgets() {
    crate::execution::serial(|| {
        let image = gray(257, 3);
        let frame = Frame::new(&image);
        let options = EncodeOptions {
            group_size_shift: 0,
            max_output_bytes: 1 << 20,
            ..Default::default()
        };
        let mut cache = LogicalFrameCache::default();
        let complete = encode_logical_frame_cached_trial(
            &image, &frame, &options, true, None, &mut cache, 0u8,
        )
        .unwrap()
        .into_complete()
        .unwrap();
        let size = complete.len();
        let miss_budget = SectionBudget::new(Some(1), options.max_output_bytes).unwrap();
        assert!(matches!(
            encode_logical_frame_cached_trial(
                &image,
                &frame,
                &options,
                true,
                Some(&miss_budget),
                &mut cache,
                1,
            )
            .unwrap(),
            TrialEncode::Bounded { .. }
        ));
        assert!(!cache.contains(&0));
        assert!(!cache.contains(&1));
        encode_logical_frame_cached_trial(&image, &frame, &options, true, None, &mut cache, 1)
            .unwrap()
            .into_complete()
            .unwrap();
        assert!(!cache.contains(&0));
        assert!(cache.contains(&1));
        let stored = cache.entry.as_ref().unwrap().1.as_ptr();
        for cutoff in [size - 1, size, size + 1] {
            let budget = SectionBudget::new(Some(cutoff), options.max_output_bytes).unwrap();
            let result = encode_logical_frame_cached_trial(
                &image,
                &frame,
                &options,
                true,
                Some(&budget),
                &mut cache,
                1,
            )
            .unwrap();
            if cutoff <= size {
                assert!(
                    matches!(result, TrialEncode::Bounded { lower_bound } if lower_bound == cutoff)
                );
            } else {
                assert_eq!(result.into_complete().unwrap().len(), size);
            }
            assert_eq!(budget.section_bytes(), size);
            assert_eq!(cache.entry.as_ref().unwrap().1.as_ptr(), stored);
        }
        assert_decoded(&image, complete, &options);
    });
}

#[test]
fn cache_hit_hard_limits_outrank_pruning_and_headers_are_not_double_charged() {
    let image = gray(17, 9);
    let frame = Frame::new(&image);
    let options = EncodeOptions {
        max_output_bytes: 1 << 20,
        ..Default::default()
    };
    let mut cache = LogicalFrameCache::default();
    let size =
        encode_logical_frame_cached_trial(&image, &frame, &options, true, None, &mut cache, ())
            .unwrap()
            .into_complete()
            .unwrap()
            .len();
    for (hard_limit, prior_overhead) in [(size - 1, 0), (size, 1)] {
        let budget = SectionBudget::new(Some(1), hard_limit).unwrap();
        budget.retain_overhead(prior_overhead).unwrap();
        assert!(matches!(
            encode_logical_frame_cached_trial(
                &image,
                &frame,
                &options,
                true,
                Some(&budget),
                &mut cache,
                (),
            ),
            Err(Error::Limit(_))
        ));
        assert_eq!(budget.reached(), None);
    }
    let exact_budget = SectionBudget::new(Some(1), size).unwrap();
    assert!(matches!(
        encode_logical_frame_cached_trial(
            &image,
            &frame,
            &options,
            true,
            Some(&exact_budget),
            &mut cache,
            (),
        )
        .unwrap(),
        TrialEncode::Bounded { .. }
    ));
    assert_eq!(exact_budget.section_bytes(), size);
    for hard_limit in [size - 1, size] {
        let limited = EncodeOptions {
            max_output_bytes: hard_limit,
            ..options.clone()
        };
        let result =
            encode_logical_frame_cached_trial(&image, &frame, &limited, true, None, &mut cache, ());
        if hard_limit < size {
            assert!(matches!(result, Err(Error::Limit(_))));
        } else {
            assert_eq!(result.unwrap().into_complete().unwrap().len(), size);
        }
    }
}

#[test]
fn cache_hit_still_validates_frame_settings_and_pixel_limits() {
    let image = gray(7, 5);
    let options = EncodeOptions::default();
    let mut frame = Frame::new(&image);
    let mut cache = LogicalFrameCache::default();
    encode_logical_frame_cached_trial(&image, &frame, &options, true, None, &mut cache, ())
        .unwrap()
        .into_complete()
        .unwrap();
    for invalid in [
        FrameOptions {
            save_as_reference: 1,
            ..Default::default()
        },
        FrameOptions {
            blend: BlendInfo {
                source: 4,
                ..Default::default()
            },
            ..Default::default()
        },
        FrameOptions {
            name: "x".repeat(1072),
            ..Default::default()
        },
    ] {
        frame.options = invalid;
        assert!(matches!(
            encode_logical_frame_cached_trial(&image, &frame, &options, true, None, &mut cache, ()),
            Err(Error::InvalidInput(_))
        ));
    }
    frame.options = Default::default();
    let limited = EncodeOptions {
        max_pixels: 0,
        ..options
    };
    assert!(matches!(
        encode_logical_frame_cached_trial(&image, &frame, &limited, true, None, &mut cache, ()),
        Err(Error::Limit("pixel count"))
    ));
    assert!(cache.contains(&()));
}

fn patch_options(source: bool) -> EncodeOptions {
    let mut options = EncodeOptions::default();
    if source {
        options.source_patches = Some(Default::default());
    } else {
        options.patches = Some(Default::default());
    }
    options
}

#[test]
fn configured_patch_tools_without_plan_reuse_and_revalidate_private_slots() {
    let image = Image::new(
        13,
        11,
        PixelLayout::Rgba,
        PixelData::U8([29, 53, 71, 0].repeat(13 * 11)),
    )
    .unwrap();
    for source in [false, true] {
        let options = patch_options(source);
        assert!(prepare_patch_plan(&image, &options).unwrap().is_none());
        let frame = Frame::new(&image);
        let mut cache = LogicalFrameCache::default();
        let first = encode_logical_frame_cached_trial(
            &image, &frame, &options, true, None, &mut cache, 0u8,
        )
        .unwrap()
        .into_complete()
        .unwrap();
        assert!(cache.contains(&0));
        let stored = cache.entry.as_ref().unwrap().1.as_ptr();
        let budget = SectionBudget::new(Some(first.len() + 1), usize::MAX).unwrap();
        let hit = encode_logical_frame_cached_trial(
            &image,
            &frame,
            &options,
            true,
            Some(&budget),
            &mut cache,
            0,
        )
        .unwrap()
        .into_complete()
        .unwrap();
        let ordinary = encode_logical_frame_trial(&image, &frame, &options, true, None)
            .unwrap()
            .into_complete()
            .unwrap();
        assert_eq!(hit.len(), first.len());
        assert_eq!(hit.len(), ordinary.len());
        assert_eq!(budget.section_bytes(), hit.len());
        assert_eq!(cache.entry.as_ref().unwrap().1.as_ptr(), stored);
        assert_decoded(&image, hit, &options);

        for key in [0, 1] {
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
                FrameOptions {
                    extra_blend: vec![BlendInfo {
                        source: 3,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ] {
                let invalid = Frame {
                    image: &image,
                    options: settings,
                };
                assert!(matches!(
                    encode_logical_frame_cached_trial(
                        &image, &invalid, &options, false, None, &mut cache, key,
                    ),
                    Err(Error::InvalidInput(message)) if message == "patches reserve reference slot 3"
                ));
                assert_eq!(cache.entry.as_ref().unwrap().1.as_ptr(), stored);
            }
        }

        let reference = Frame {
            image: &image,
            options: FrameOptions {
                reference_only: true,
                save_as_reference: 1,
                save_before_color_transform: true,
                ..Default::default()
            },
        };
        let mut sequence = encode_logical_frame_cached_trial(
            &image, &reference, &options, false, None, &mut cache, 0,
        )
        .unwrap()
        .into_complete()
        .unwrap();
        assert!(!cache.contains(&0));
        sequence.extend(
            encode_logical_frame_trial(&image, &frame, &options, true, None)
                .unwrap()
                .into_complete()
                .unwrap(),
        );
        assert_decoded(&image, sequence, &options);
    }
}

#[test]
fn actual_plans_in_both_patch_routes_bypass_and_clear_cached_payloads() {
    let (width, height) = (129usize, 83usize);
    let mut pixels = vec![255; width * height];
    for (x, y) in [(17, 17), (81, 17), (33, 53), (97, 53)] {
        for row in y..y + 5 {
            pixels[row * width + x..row * width + x + 6].fill(63);
        }
    }
    let image = Image::new(
        width as u32,
        height as u32,
        PixelLayout::Gray,
        PixelData::U8(pixels),
    )
    .unwrap();
    let uniform = Image::new(
        width as u32,
        height as u32,
        PixelLayout::Gray,
        PixelData::U8(vec![255; width * height]),
    )
    .unwrap();
    for source in [false, true] {
        let options = patch_options(source);
        assert!(prepare_patch_plan(&image, &options).unwrap().is_some());
        assert!(prepare_patch_plan(&uniform, &options).unwrap().is_none());
        let mut cache = LogicalFrameCache::default();
        encode_logical_frame_cached_trial(
            &image,
            &Frame::new(&uniform),
            &options,
            true,
            None,
            &mut cache,
            0u8,
        )
        .unwrap()
        .into_complete()
        .unwrap();
        assert!(cache.contains(&0));
        let frame = Frame::new(&image);
        let bytes =
            encode_logical_frame_cached_trial(&image, &frame, &options, true, None, &mut cache, 1)
                .unwrap()
                .into_complete()
                .unwrap();
        assert!(!cache.contains(&0));
        assert!(!cache.contains(&1));
        let ordinary = encode_logical_frame_trial(&image, &frame, &options, true, None)
            .unwrap()
            .into_complete()
            .unwrap();
        assert_eq!(bytes.len(), ordinary.len());
        assert_decoded(&image, bytes, &options);
    }
}
