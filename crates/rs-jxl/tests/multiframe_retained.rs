//! Retained-reference acceptance compares decoded samples and measured lengths,
//! never a golden compressed bitstream. Fixtures are small, deterministic states.

use rs_jxl::codec::Encoder;
use rs_jxl::decode::{DecodeOptions, DecodedFrame, JxlBlendMode, decode, decode_frame};
use rs_jxl::metadata::{Animation, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::multiframe::{AutoEncodeResult, AutoOptions, Reference, Strategy, encode_auto};
use rs_jxl::{EncodeOptions, Error, Image, PixelData, PixelLayout};

const WIDTH: usize = 65;
const HEIGHT: usize = 49;
const POINTS: [usize; 2] = [2 * WIDTH + 2, (HEIGHT - 3) * WIDTH + WIDTH - 3];

fn random_words(seed: u32, len: usize) -> Vec<u16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u16
        })
        .collect()
}

fn animated(image: Image) -> Image {
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    let extras = image.extra_planes().to_vec();
    image.with_metadata(metadata, extras).unwrap()
}

// X, A and B have unrelated content; A2 differs from A only at the given points.
fn recurrence(layout: PixelLayout, bits: u8, points: &[usize], opaque: bool) -> Vec<Image> {
    [0x71a2, 0x2319, 0x8b55, 0x2319]
        .into_iter()
        .enumerate()
        .map(|(index, seed)| {
            let channels = layout.channels();
            let maximum = if bits == 8 { 255 } else { 65535 };
            let mut values = random_words(seed, WIDTH * HEIGHT * channels);
            for value in &mut values {
                *value &= maximum;
            }
            if layout.has_alpha() {
                for (pixel, samples) in values.chunks_exact_mut(channels).enumerate() {
                    samples[channels - 1] = if opaque {
                        maximum
                    } else {
                        match pixel % 3 {
                            0 => 0,
                            1 => maximum / 2,
                            _ => maximum,
                        }
                    };
                }
                // The changed color samples below are invisible but significant.
                if !opaque {
                    for &point in points {
                        values[point * channels + channels - 1] = 0;
                    }
                }
            }
            if index == 3 {
                for &point in points {
                    values[point * channels] ^= 0x55;
                }
            }
            let pixels = if bits == 8 {
                PixelData::U8(values.into_iter().map(|value| value as u8).collect())
            } else {
                PixelData::U16(values)
            };
            animated(Image::new(WIDTH as u32, HEIGHT as u32, layout, pixels).unwrap())
        })
        .collect()
}

fn retained() -> AutoOptions {
    AutoOptions {
        retained_reference: true,
        tile_size: 8,
        ..Default::default()
    }
}

fn crop_only() -> AutoOptions {
    AutoOptions {
        allow_mask: false,
        allow_multi_rect: false,
        adaptive_rectangles: false,
        ..retained()
    }
}

fn assert_samples(actual: &PixelData, expected: &PixelData) {
    if let (PixelData::F32(actual), PixelData::F32(expected)) = (actual, expected) {
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    } else {
        assert_eq!(actual, expected);
    }
}

fn assert_frame(frame: &DecodedFrame, image: &Image, duration: u32) {
    assert_eq!((frame.width, frame.height), (WIDTH, HEIGHT));
    assert_eq!(frame.layout, image.layout());
    assert_eq!(frame.duration_ticks, duration);
    assert_samples(&frame.pixels, image.pixels());
    let extra_start = usize::from(image.layout().has_alpha());
    assert_eq!(
        frame.extra_planes.len(),
        extra_start + image.extra_planes().len()
    );
    if extra_start != 0 {
        let channels = image.layout().channels();
        let alpha = match image.pixels() {
            PixelData::U8(values) => PixelData::U8(
                values
                    .chunks_exact(channels)
                    .map(|pixel| pixel[channels - 1])
                    .collect(),
            ),
            PixelData::U16(values) => PixelData::U16(
                values
                    .chunks_exact(channels)
                    .map(|pixel| pixel[channels - 1])
                    .collect(),
            ),
            _ => unreachable!("only integer alpha fixtures"),
        };
        assert_samples(&frame.extra_planes[0], &alpha);
    }
    for (actual, expected) in frame.extra_planes[extra_start..]
        .iter()
        .zip(image.extra_planes())
    {
        assert_samples(actual, expected);
    }
}

fn assert_decoded(result: &AutoEncodeResult, images: &[Image], durations: &[u32]) {
    assert_eq!(result.frames.len(), images.len());
    let options = DecodeOptions {
        include_extra_channels: true,
        ..Default::default()
    };
    let decoded = decode(&result.bytes, &options).unwrap();
    assert!(decoded.complete);
    assert_eq!(
        decoded.info.bit_depth.bits_per_sample(),
        u32::from(images[0].metadata().bit_depth.bits_per_sample)
    );
    assert_eq!(
        decoded.info.animation.is_some(),
        images[0].metadata().animation.is_some()
    );
    let visible: Vec<_> = (0..images.len())
        .filter(|&index| durations[index] != 0 || index + 1 == images.len())
        .collect();
    assert_eq!(decoded.frames.len(), visible.len());
    for (frame, &index) in decoded.frames.iter().zip(&visible) {
        assert_frame(frame, &images[index], durations[index]);
    }
    let selected = decode_frame(&result.bytes, visible.len() - 1, &options).unwrap();
    assert_eq!(selected.frames.len(), 1);
    let index = *visible.last().unwrap();
    assert_frame(&selected.frames[0], &images[index], durations[index]);
}

fn retained_strategy(strategy: Strategy) -> bool {
    matches!(
        strategy,
        Strategy::CropRetained
            | Strategy::MaskRetained
            | Strategy::TilesRetained
            | Strategy::MergedRetained
    )
}

fn check_reference(
    strategy: Strategy,
    reference: Option<usize>,
    source: u8,
    index: usize,
    slots: &[Option<usize>; 4],
) {
    if let Some(reference) = reference {
        assert!(matches!(source, 1 | 2));
        assert!(reference < index);
        assert_eq!(slots[usize::from(source)], Some(reference));
        if retained_strategy(strategy) {
            assert_ne!(reference, 0);
            assert!(reference + 1 < index);
        }
    } else {
        assert_eq!(strategy, Strategy::Full);
        assert_eq!(source, 0);
    }
}

// Check every report against the actual raw frame headers, not guessed slot roles.
fn assert_reference_reports(result: &AutoEncodeResult, durations: &[u32], enabled: bool) {
    let raw = decode(
        &result.bytes,
        &DecodeOptions {
            coalescing: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut layers = raw.frames.iter();
    let mut slots = [None; 4];
    for (index, report) in result.frames.iter().enumerate() {
        assert_eq!(report.retained_reference, enabled);
        assert_eq!(report.references_before, slots);
        assert_eq!(report.references_before[0], None);
        assert_eq!(report.references_before[3], None);
        let winner = &report.candidates[report.winner_index];
        assert_eq!(
            winner.encoded_bytes,
            report
                .candidates
                .iter()
                .map(|candidate| candidate.encoded_bytes)
                .min()
                .unwrap()
        );
        for candidate in &report.candidates {
            check_reference(
                candidate.strategy,
                candidate.reference_index,
                candidate.source,
                index,
                &slots,
            );
            assert_eq!(
                candidate.save_as_reference == 0,
                index + 1 == result.frames.len()
            );
            assert!(candidate.save_as_reference <= 2);
        }
        for candidate in &report.pruned {
            check_reference(
                candidate.strategy,
                candidate.reference_index,
                candidate.source,
                index,
                &slots,
            );
            assert_eq!(
                candidate.save_as_reference == 0,
                index + 1 == result.frames.len()
            );
            assert!(candidate.save_as_reference <= 2);
        }
        for difference in &report.differences {
            assert!(matches!(difference.source, 1 | 2));
            assert_eq!(
                slots[usize::from(difference.source)],
                Some(difference.reference)
            );
        }
        let mut previous_save = None;
        for part in 0..winner.encoded_frames {
            let layer = layers.next().unwrap();
            let final_part = part + 1 == winner.encoded_frames;
            let last = final_part && index + 1 == result.frames.len();
            assert_eq!(
                layer.duration_ticks,
                if final_part { durations[index] } else { 0 }
            );
            assert_eq!(layer.header.is_last, last);
            let info = &layer.header.layer_info;
            assert!(!info.save_before_color_transform);
            assert_eq!(
                info.blend_info.source,
                if part == 0 {
                    u32::from(winner.source)
                } else {
                    previous_save.unwrap()
                }
            );
            assert!(
                info.extra_channel_blend_info
                    .iter()
                    .all(|blend| *blend == info.blend_info)
            );
            if final_part {
                assert_eq!(
                    info.save_as_reference,
                    if last {
                        None
                    } else {
                        Some(u32::from(winner.save_as_reference))
                    }
                );
            }
            if let Some(slot) = info.save_as_reference {
                assert!(matches!(slot, 1 | 2));
                slots[slot as usize] = final_part.then_some(index);
            }
            previous_save = info.save_as_reference;
        }
        assert_eq!(report.references_after, slots);
    }
    assert!(layers.next().is_none());
}

#[test]
fn older_canvas_is_selected_and_beats_legacy_for_gray_and_hidden_rgba() {
    let options = EncodeOptions::default();
    for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
        for bits in [8, 16] {
            let images = recurrence(layout, bits, &POINTS[..1], false);
            let auto = crop_only();
            let result = encode_auto(&images, &[2, 3, 5, 7], &options, &auto).unwrap();
            let legacy = encode_auto(
                &images,
                &[2, 3, 5, 7],
                &options,
                &AutoOptions {
                    retained_reference: false,
                    ..auto
                },
            )
            .unwrap();
            let report = &result.frames[3];
            let winner = &report.candidates[report.winner_index];
            assert_eq!(winner.strategy, Strategy::CropRetained);
            assert_eq!(winner.reference_index, Some(1));
            assert_eq!(report.references_before, [None, Some(2), Some(1), None]);
            assert_eq!(winner.source, 2);
            assert!(
                result.bytes.len() < legacy.bytes.len(),
                "{layout:?}/{bits}: retained={} legacy={}",
                result.bytes.len(),
                legacy.bytes.len()
            );
            assert_decoded(&result, &images, &[2, 3, 5, 7]);
            assert_decoded(&legacy, &images, &[2, 3, 5, 7]);
            assert_reference_reports(&result, &[2, 3, 5, 7], true);
            assert_reference_reports(&legacy, &[2, 3, 5, 7], false);
        }
    }
}

#[test]
fn retained_multirect_and_zero_duration_states_preserve_complete_slot_maps() {
    let mut images = recurrence(PixelLayout::Rgba, 8, &POINTS, false);
    let auto = AutoOptions {
        adaptive_rectangles: false,
        ..retained()
    };
    for durations in [[0; 4], [0, 3, 0, 7], [2, 0, 5, 0]] {
        let result = encode_auto(&images, &durations, &Default::default(), &auto).unwrap();
        let last = &result.frames[3];
        let winner = &last.candidates[last.winner_index];
        assert_eq!(winner.strategy, Strategy::TilesRetained);
        assert!(winner.encoded_frames > 1);
        assert_eq!(last.references_after[usize::from(winner.source)], Some(1));
        assert_eq!(last.references_after[3 - usize::from(winner.source)], None);
        assert_decoded(&result, &images, &durations);
        assert_reference_reports(&result, &durations, true);
    }
    for image in &mut images {
        let mut metadata = image.metadata().clone();
        metadata.animation = None;
        *image = image.clone().with_metadata(metadata, vec![]).unwrap();
    }
    // Every non-display logical state is checked as a completed prefix as well.
    for count in 1..=images.len() {
        let result = encode_auto(
            &images[..count],
            &vec![0; count],
            &Default::default(),
            &auto,
        )
        .unwrap();
        assert_decoded(&result, &images[..count], &vec![0; count]);
        assert_reference_reports(&result, &vec![0; count], true);
    }
}

#[test]
fn completed_multirect_state_can_become_an_older_retained_background() {
    let mut images = recurrence(PixelLayout::Rgba, 8, &POINTS, false);
    images.push(images[2].clone());
    images.push(images[3].clone());
    let durations = [0, 3, 0, 7, 0, 2];
    let result = encode_auto(
        &images,
        &durations,
        &Default::default(),
        &AutoOptions {
            adaptive_rectangles: false,
            ..retained()
        },
    )
    .unwrap();
    let middle = &result.frames[3];
    let winner = &middle.candidates[middle.winner_index];
    assert_eq!(winner.strategy, Strategy::TilesRetained);
    assert!(winner.encoded_frames > 1);
    assert_eq!(
        middle.references_after[usize::from(winner.save_as_reference)],
        Some(3)
    );
    let last = &result.frames[5];
    let winner = &last.candidates[last.winner_index];
    assert_eq!(winner.strategy, Strategy::CropRetained);
    assert_eq!(winner.reference_index, Some(3));
    assert_decoded(&result, &images, &durations);
    assert_reference_reports(&result, &durations, true);
}

#[test]
fn extra_only_changes_use_retained_canvas_and_preserve_full_resolution_depth() {
    let images: Vec<_> = [0x71a2, 0x2319, 0x8b55, 0x2319]
        .into_iter()
        .enumerate()
        .map(|(index, seed)| {
            let image = animated(
                Image::new(
                    WIDTH as u32,
                    HEIGHT as u32,
                    PixelLayout::Gray,
                    PixelData::U8(vec![17; WIDTH * HEIGHT]),
                )
                .unwrap(),
            );
            let mut metadata = image.metadata().clone();
            let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap();
            extra.name = "depth".into();
            metadata.extra_channels.push(extra);
            let mut values = random_words(seed, WIDTH * HEIGHT);
            if index == 3 {
                values[POINTS[0]] ^= 0x55;
            }
            image
                .with_metadata(metadata, vec![PixelData::U16(values)])
                .unwrap()
        })
        .collect();
    let result = encode_auto(&images, &[0, 2, 0, 5], &Default::default(), &crop_only()).unwrap();
    let report = &result.frames[3];
    assert_eq!(
        report.candidates[report.winner_index].strategy,
        Strategy::CropRetained
    );
    assert_eq!(
        report
            .differences
            .iter()
            .find(|diff| diff.reference == 1)
            .unwrap()
            .changed_pixels,
        1
    );
    assert_decoded(&result, &images, &[0, 2, 0, 5]);
    assert_reference_reports(&result, &[0, 2, 0, 5], true);
    let decoded = decode(&result.bytes, &Default::default()).unwrap();
    assert_eq!(decoded.info.extra_channels.len(), 1);
    assert_eq!(decoded.info.extra_channels[0].name, "depth");
}

#[test]
fn retained_float_replacement_preserves_signed_zero_nan_payloads_and_infinities() {
    let special = [0, 0x8000_0000, 0x7fc1_2345, 0x7f80_0000, 0xff80_0000, 1];
    let images: Vec<_> = [0u32, 1, 2, 1]
        .into_iter()
        .enumerate()
        .map(|(index, state)| {
            let mut words: Vec<_> = (0..WIDTH * HEIGHT)
                .map(|pixel| special[(pixel + state as usize) % special.len()])
                .collect();
            if state == 1 {
                words[POINTS[0]] = if index == 3 { 0x8000_0000 } else { 0 };
            }
            animated(
                Image::new(
                    WIDTH as u32,
                    HEIGHT as u32,
                    PixelLayout::Gray,
                    PixelData::F32(words.into_iter().map(f32::from_bits).collect()),
                )
                .unwrap(),
            )
        })
        .collect();
    let result = encode_auto(&images, &[1; 4], &Default::default(), &crop_only()).unwrap();
    let report = &result.frames[3];
    assert_eq!(
        report.candidates[report.winner_index].strategy,
        Strategy::CropRetained
    );
    assert_eq!(
        report
            .differences
            .iter()
            .find(|diff| diff.reference == 1)
            .unwrap()
            .changed_pixels,
        1
    );
    assert_decoded(&result, &images, &[1; 4]);
    assert_reference_reports(&result, &[1; 4], true);
}

#[test]
fn pruning_preserves_retained_winner_lengths_and_samples_with_mask_candidates() {
    for opaque in [false, true] {
        let images = recurrence(PixelLayout::Rgba, 8, &POINTS, opaque);
        let bounded = encode_auto(&images, &[1; 4], &Default::default(), &retained()).unwrap();
        let complete = encode_auto(
            &images,
            &[1; 4],
            &Default::default(),
            &AutoOptions {
                prune: false,
                ..retained()
            },
        )
        .unwrap();
        assert_eq!(bounded.bytes.len(), complete.bytes.len());
        for (a, b) in bounded.frames.iter().zip(&complete.frames) {
            let a_winner = &a.candidates[a.winner_index];
            let b_winner = &b.candidates[b.winner_index];
            assert_eq!(
                (
                    a_winner.strategy,
                    a_winner.reference_index,
                    a_winner.encoded_bytes
                ),
                (
                    b_winner.strategy,
                    b_winner.reference_index,
                    b_winner.encoded_bytes
                )
            );
            assert_eq!(a.references_before, b.references_before);
            assert_eq!(a.references_after, b.references_after);
            assert!(b.pruned.is_empty());
        }
        let report = &complete.frames[3];
        assert_eq!(
            report
                .candidates
                .iter()
                .any(|candidate| candidate.strategy == Strategy::MaskRetained),
            opaque
        );
        assert!(retained_strategy(
            report.candidates[report.winner_index].strategy
        ));
        assert_decoded(&bounded, &images, &[1; 4]);
        assert_decoded(&complete, &images, &[1; 4]);
        assert_reference_reports(&bounded, &[1; 4], true);
        assert_reference_reports(&complete, &[1; 4], true);
    }
}

#[test]
fn disabled_retention_preserves_legacy_reference_roles_and_candidate_labels() {
    assert!(!AutoOptions::default().retained_reference);
    let images = recurrence(PixelLayout::Gray, 8, &POINTS[..1], false);
    let default = encode_auto(&images, &[1; 4], &Default::default(), &Default::default()).unwrap();
    let disabled = encode_auto(
        &images,
        &[1; 4],
        &Default::default(),
        &AutoOptions {
            retained_reference: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(default.bytes.len(), disabled.bytes.len());
    for (index, report) in default.frames.iter().enumerate() {
        assert!(!report.retained_reference);
        assert!(
            report
                .candidates
                .iter()
                .map(|c| c.strategy)
                .chain(report.pruned.iter().map(|c| c.strategy))
                .all(|strategy| !retained_strategy(strategy))
        );
        if index > 0 {
            assert_eq!(report.references_before[1], Some(0));
        }
        if index > 1 {
            assert_eq!(report.references_before[2], Some(index - 1));
        }
    }
    assert_decoded(&default, &images, &[1; 4]);
    assert_decoded(&disabled, &images, &[1; 4]);
    assert_reference_reports(&default, &[1; 4], false);
}

#[test]
fn invalid_reference_settings_and_work_limits_do_not_poison_reused_contexts() {
    let images = recurrence(PixelLayout::Gray, 8, &POINTS[..1], false);
    for threads in [1, 2] {
        let options = EncodeOptions {
            threads,
            ..Default::default()
        };
        let mut encoder = Encoder::new(threads).unwrap();
        for reference in [Reference::First, Reference::Previous, Reference::Full] {
            assert!(matches!(
                encoder.encode_auto(
                    &images,
                    &[1; 4],
                    &options,
                    &AutoOptions {
                        reference,
                        ..retained()
                    }
                ),
                Err(Error::InvalidInput(_))
            ));
        }
        let base = encoder
            .encode_auto(&images, &[1; 4], &options, &crop_only())
            .unwrap();
        let trials: usize = base
            .frames
            .iter()
            .map(|report| report.candidates.len() + report.pruned.len())
            .sum();
        let pixels: u64 = base
            .frames
            .iter()
            .map(|report| {
                report
                    .candidates
                    .iter()
                    .map(|c| c.trial_pixels)
                    .sum::<u64>()
                    + report.pruned.iter().map(|c| c.trial_pixels).sum::<u64>()
            })
            .sum();
        let exact = AutoOptions {
            max_frames: 4,
            max_total_trials: trials,
            max_trial_pixels: pixels,
            ..crop_only()
        };
        for limited in [
            AutoOptions {
                max_frames: 3,
                ..exact.clone()
            },
            AutoOptions {
                max_total_trials: trials - 1,
                ..exact.clone()
            },
            AutoOptions {
                max_trial_pixels: pixels - 1,
                ..exact.clone()
            },
            AutoOptions {
                max_total_trials: 0,
                ..exact.clone()
            },
            AutoOptions {
                max_trial_pixels: 0,
                ..exact.clone()
            },
        ] {
            assert!(matches!(
                encoder.encode_auto(&images, &[1; 4], &options, &limited),
                Err(Error::Limit(_))
            ));
            let recovered = encoder
                .encode_auto(&images, &[1; 4], &options, &exact)
                .unwrap();
            assert_eq!(recovered.bytes.len(), base.bytes.len());
            assert_decoded(&recovered, &images, &[1; 4]);
            assert_reference_reports(&recovered, &[1; 4], true);
        }
        let exact_output = EncodeOptions {
            max_output_bytes: base.bytes.len(),
            ..options.clone()
        };
        let result = encoder
            .encode_auto(&images, &[1; 4], &exact_output, &exact)
            .unwrap();
        assert_eq!(result.bytes.len(), base.bytes.len());
        assert!(matches!(
            encoder.encode_auto(
                &images,
                &[1; 4],
                &EncodeOptions {
                    max_output_bytes: base.bytes.len() - 1,
                    ..options.clone()
                },
                &exact
            ),
            Err(Error::Limit(_))
        ));
        let mut reordered = images.clone();
        reordered.swap(0, 2);
        let fresh = encode_auto(&reordered, &[1; 4], &options, &crop_only()).unwrap();
        let reused = encoder
            .encode_auto(&reordered, &[1; 4], &options, &crop_only())
            .unwrap();
        assert_eq!(reused.bytes.len(), fresh.bytes.len());
        assert_eq!(reused.frames[0].references_before, [None; 4]);
        assert_decoded(&reused, &reordered, &[1; 4]);
        assert_reference_reports(&reused, &[1; 4], true);
    }
}

#[test]
fn retained_replacement_avoids_blending_hidden_color_or_float_words() {
    let images = recurrence(PixelLayout::Rgba, 16, &POINTS, false);
    let result = encode_auto(&images, &[1; 4], &Default::default(), &retained()).unwrap();
    let raw = decode(
        &result.bytes,
        &DecodeOptions {
            coalescing: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        raw.frames
            .iter()
            .all(|frame| frame.header.layer_info.blend_info.mode == JxlBlendMode::Replace)
    );
    assert_decoded(&result, &images, &[1; 4]);
    assert_reference_reports(&result, &[1; 4], true);
}
