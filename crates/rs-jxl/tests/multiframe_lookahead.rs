//! Native lookahead acceptance: exact samples, raw slot transitions and measured
//! lengths, never a golden compressed bitstream. All fixtures fit one group.

use rs_jxl::decode::{DecodeOptions, DecodedFrame, decode};
use rs_jxl::metadata::{Animation, ExtraChannelInfo, ExtraChannelType};
use rs_jxl::multiframe::{
    AutoCandidateReport, AutoEncodeResult, AutoOptions, Rect, Strategy, encode_auto,
};
use rs_jxl::{Error, Image, PixelData, PixelLayout};

const W: usize = 41;
const H: usize = 33;
const P: usize = 2 * W + 2;
const Q: usize = (H - 3) * W + W - 3;
const CENTER: usize = (H / 2) * W + W / 2;
const SEEDS: [u32; 3] = [0x71a2, 0x2319, 0x8b55];

#[derive(Clone, Copy, Debug)]
enum Samples {
    Gray8,
    Rgba8,
    Gray16,
    Extra16,
    Float32,
}

fn state(kind: Samples, seed: u32, changes: &[usize]) -> Image {
    let layout = if matches!(kind, Samples::Rgba8) {
        PixelLayout::Rgba
    } else {
        PixelLayout::Gray
    };
    let channels = layout.channels();
    let mut random = seed;
    let mut words: Vec<u16> = (0..W * H * channels)
        .map(|_| {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as u16
        })
        .collect();
    if layout.has_alpha() {
        for (p, pixel) in words.chunks_exact_mut(channels).enumerate() {
            pixel[3] = if [P, Q, CENTER].contains(&p) {
                0
            } else {
                [0, 127, 255][p % 3]
            };
        }
    }
    let mut extras = Vec::new();
    let pixels = if matches!(kind, Samples::Float32) {
        let special = [
            0,
            0x8000_0000,
            0x7fc1_2345,
            0x7fc5_4321,
            0x7f80_0000,
            0xff80_0000,
            1,
        ];
        let mut bits: Vec<u32> = words
            .iter()
            .map(|&w| special[w as usize % special.len()])
            .collect();
        bits[P] = 0;
        bits[P + 1] = 0x7fc1_2345;
        for &p in changes {
            bits[p] ^= if p == P { 0x8000_0000 } else { 1 };
        }
        PixelData::F32(bits.into_iter().map(f32::from_bits).collect())
    } else {
        for &p in changes {
            words[p * channels] ^= 0x55;
        }
        match kind {
            Samples::Extra16 => {
                extras.push(PixelData::U16(words));
                PixelData::U8(vec![19; W * H])
            }
            Samples::Gray16 => PixelData::U16(words),
            _ => PixelData::U8(words.into_iter().map(|w| w as u8).collect()),
        }
    };
    let image = Image::new(W as u32, H as u32, layout, pixels).unwrap();
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    if matches!(kind, Samples::Extra16) {
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
    }
    image.with_metadata(metadata, extras).unwrap()
}

fn recurrence(kind: Samples, target: usize, changes: &[usize]) -> Vec<Image> {
    let mut images: Vec<_> = SEEDS
        .into_iter()
        .map(|seed| state(kind, seed, &[]))
        .collect();
    images.push(state(kind, SEEDS[target], changes));
    images
}

fn lookahead() -> AutoOptions {
    AutoOptions {
        retained_reference: true,
        reference_lookahead: true,
        tile_size: 8,
        max_rectangles: 8,
        ..Default::default()
    }
}

fn crop_only() -> AutoOptions {
    AutoOptions {
        allow_mask: false,
        allow_multi_rect: false,
        adaptive_rectangles: false,
        ..lookahead()
    }
}

fn words(pixels: &PixelData) -> Vec<u32> {
    match pixels {
        PixelData::U8(v) => v.iter().map(|&x| u32::from(x)).collect(),
        PixelData::U16(v) => v.iter().map(|&x| u32::from(x)).collect(),
        PixelData::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
        _ => panic!("unexpected fixture sample type"),
    }
}

fn assert_frame(frame: &DecodedFrame, image: &Image, duration: u32) {
    assert_eq!(
        (frame.width, frame.height, frame.layout),
        (W, H, image.layout())
    );
    assert_eq!(frame.duration_ticks, duration);
    assert_eq!(
        std::mem::discriminant(&frame.pixels),
        std::mem::discriminant(image.pixels())
    );
    assert_eq!(words(&frame.pixels), words(image.pixels()));
    let alpha = usize::from(image.layout().has_alpha());
    assert_eq!(frame.extra_planes.len(), alpha + image.extra_planes().len());
    if alpha != 0 {
        let channels = image.layout().channels();
        let expected: Vec<_> = words(image.pixels())
            .chunks_exact(channels)
            .map(|p| p[channels - 1])
            .collect();
        assert_eq!(words(&frame.extra_planes[0]), expected);
    }
    for (actual, expected) in frame.extra_planes[alpha..].iter().zip(image.extra_planes()) {
        assert_eq!(words(actual), words(expected));
    }
}

// Independent bitwise geometry, including invisible color and full-size extras.
fn difference(a: &Image, b: &Image) -> (u64, Option<Rect>) {
    let (a_main, b_main) = (words(a.pixels()), words(b.pixels()));
    let extras: Vec<_> = a
        .extra_planes()
        .iter()
        .zip(b.extra_planes())
        .map(|(a, b)| (words(a), words(b)))
        .collect();
    let channels = a.layout().channels();
    let (mut count, mut left, mut top, mut right, mut bottom) = (0, W, H, 0, 0);
    for p in 0..W * H {
        let offset = p * channels;
        if a_main[offset..offset + channels] != b_main[offset..offset + channels]
            || extras.iter().any(|(a, b)| a[p] != b[p])
        {
            count += 1;
            left = left.min(p % W);
            top = top.min(p / W);
            right = right.max(p % W + 1);
            bottom = bottom.max(p / W + 1);
        }
    }
    (
        count,
        (count != 0).then(|| Rect {
            x: left as u32,
            y: top as u32,
            width: (right - left) as u32,
            height: (bottom - top) as u32,
        }),
    )
}

fn score(a: &Image, b: &Image) -> (u64, u64) {
    let (count, bounds) = difference(a, b);
    (
        bounds.map_or(0, |r| u64::from(r.width) * u64::from(r.height)),
        count,
    )
}

fn winner(result: &AutoEncodeResult, index: usize) -> &AutoCandidateReport {
    let report = &result.frames[index];
    &report.candidates[report.winner_index]
}

fn verify(images: &[Image], durations: &[u32], auto: &AutoOptions) -> AutoEncodeResult {
    let result = encode_auto(images, durations, &Default::default(), auto).unwrap();
    assert_eq!(result.frames.len(), images.len());
    let decoded = decode(
        &result.bytes,
        &DecodeOptions {
            include_extra_channels: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(decoded.complete);
    let visible: Vec<_> = (0..images.len())
        .filter(|&i| durations[i] != 0 || i + 1 == images.len())
        .collect();
    assert_eq!(decoded.frames.len(), visible.len());
    for (frame, &i) in decoded.frames.iter().zip(&visible) {
        assert_frame(frame, &images[i], durations[i]);
    }
    let raw = decode(
        &result.bytes,
        &DecodeOptions {
            coalescing: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(raw.complete);
    let mut layers = raw.frames.iter();
    let mut slots = [None; 4];
    for (i, report) in result.frames.iter().enumerate() {
        let last = i + 1 == images.len();
        let best = winner(&result, i);
        assert_eq!(report.retained_reference, auto.retained_reference);
        assert_eq!(report.reference_lookahead, auto.reference_lookahead);
        assert_eq!(report.references_before, slots);
        assert_eq!((slots[0], slots[3]), (None, None));
        let scanning = auto.reference_lookahead && !last;
        assert_eq!(
            report.lookahead_comparisons,
            if scanning {
                1 + slots.iter().flatten().count()
            } else {
                0
            }
        );
        assert_eq!(
            report.reused_payloads == 0,
            report.reused_payload_pixels == 0
        );
        assert!(report.reused_payload_pixels >= report.reused_payloads as u64);
        let keep = if scanning {
            let mut ranked: Vec<_> = (1..=2)
                .filter_map(|slot| {
                    slots[slot].map(|reference| (score(&images[reference], &images[i + 1]), slot))
                })
                .collect();
            ranked.sort_unstable();
            ranked.first().and_then(|&(best, slot)| {
                (best < score(&images[i], &images[i + 1])
                    && ranked.get(1).is_none_or(|&(other, _)| best < other))
                .then_some(slot)
            })
        } else {
            None
        };
        for (reference, source, save) in report
            .candidates
            .iter()
            .map(|c| (c.reference_index, c.source, c.save_as_reference))
            .chain(
                report
                    .pruned
                    .iter()
                    .map(|c| (c.reference_index, c.source, c.save_as_reference)),
            )
        {
            if let Some(reference) = reference {
                assert!(matches!(source, 1 | 2));
                assert!(reference < i);
                assert_eq!(slots[source as usize], Some(reference));
            } else {
                assert_eq!(source, 0);
            }
            assert_eq!(save == 0, last);
            assert!(save <= 2);
            if let Some(keep) = keep {
                assert_eq!(save as usize, 3 - keep);
            }
        }
        for diff in &report.differences {
            assert!(matches!(diff.source, 1 | 2));
            assert_eq!(slots[diff.source as usize], Some(diff.reference));
            assert_eq!(
                (diff.changed_pixels, diff.bounds),
                difference(&images[diff.reference], &images[i])
            );
        }
        assert_eq!(
            best.encoded_bytes,
            report
                .candidates
                .iter()
                .map(|c| c.encoded_bytes)
                .min()
                .unwrap()
        );
        assert!(best.encoded_frames > 0);
        let mut source = u32::from(best.source);
        for part in 0..best.encoded_frames {
            let layer = layers.next().expect("missing raw layer");
            let final_part = part + 1 == best.encoded_frames;
            assert_eq!(
                layer.duration_ticks,
                if final_part { durations[i] } else { 0 }
            );
            assert_eq!(layer.header.is_last, last && final_part);
            let info = &layer.header.layer_info;
            assert!(!info.save_before_color_transform);
            assert_eq!(info.blend_info.source, source);
            assert!(
                info.extra_channel_blend_info
                    .iter()
                    .all(|blend| *blend == info.blend_info)
            );
            if final_part {
                assert_eq!(
                    info.save_as_reference,
                    (!last).then_some(u32::from(best.save_as_reference))
                );
            }
            if let Some(save) = info.save_as_reference {
                assert!(matches!(save, 1 | 2));
                slots[save as usize] = final_part.then_some(i);
                source = save;
            }
        }
        assert_eq!(report.references_after, slots);
        if !last {
            assert_eq!(slots[best.save_as_reference as usize], Some(i));
        }
    }
    assert!(layers.next().is_none());
    result
}

#[test]
fn preserves_x_that_retention_alone_would_evict_and_reduces_actual_size() {
    for kind in [Samples::Rgba8, Samples::Gray16] {
        let images = recurrence(kind, 0, &[P]);
        let enabled = verify(&images, &[2, 3, 5, 7], &crop_only());
        let retained = verify(
            &images,
            &[2, 3, 5, 7],
            &AutoOptions {
                reference_lookahead: false,
                ..crop_only()
            },
        );
        assert_eq!(
            enabled.frames[2].references_after,
            [None, Some(0), Some(2), None]
        );
        assert!(!retained.frames[3].references_before.contains(&Some(0)));
        assert_eq!(winner(&enabled, 3).reference_index, Some(0));
        assert_eq!(winner(&enabled, 3).strategy, Strategy::CropFirst);
        assert!(
            enabled.bytes.len() < retained.bytes.len(),
            "{kind:?}: lookahead={} retained={}",
            enabled.bytes.len(),
            retained.bytes.len()
        );
    }
}

#[test]
fn preserves_the_prior_a_recurrence_benefit() {
    let images = recurrence(Samples::Gray8, 1, &[P]);
    let enabled = verify(&images, &[1; 4], &crop_only());
    let retained = verify(
        &images,
        &[1; 4],
        &AutoOptions {
            reference_lookahead: false,
            ..crop_only()
        },
    );
    let legacy = verify(
        &images,
        &[1; 4],
        &AutoOptions {
            retained_reference: false,
            reference_lookahead: false,
            ..crop_only()
        },
    );
    assert_eq!(winner(&enabled, 3).strategy, Strategy::CropRetained);
    assert_eq!(winner(&enabled, 3).reference_index, Some(1));
    assert_eq!(enabled.bytes.len(), retained.bytes.len());
    assert!(enabled.bytes.len() < legacy.bytes.len());
}

fn multirect_images() -> Vec<Image> {
    // B reads A and overwrites A's slot to keep X for X'. B' then needs the
    // completed two-rectangle B, including both otherwise invisible RGB edits.
    vec![
        state(Samples::Rgba8, SEEDS[0], &[]),
        state(Samples::Rgba8, SEEDS[1], &[]),
        state(Samples::Rgba8, SEEDS[1], &[P, Q]),
        state(Samples::Rgba8, SEEDS[0], &[CENTER]),
        state(Samples::Rgba8, SEEDS[1], &[P, Q, CENTER]),
    ]
}

#[test]
fn multirect_can_read_and_save_the_same_slot_then_supply_a_future_frame() {
    let images = multirect_images();
    let auto = AutoOptions {
        adaptive_rectangles: false,
        ..lookahead()
    };
    for durations in [[0, 3, 0, 5, 7], [2, 0, 5, 0, 0]] {
        let result = verify(&images, &durations, &auto);
        let b = winner(&result, 2);
        assert_eq!(b.strategy, Strategy::TilesPrevious);
        assert_eq!(b.reference_index, Some(1));
        assert_eq!((b.source, b.save_as_reference, b.encoded_frames), (2, 2, 2));
        assert_eq!(
            result.frames[2].references_after,
            [None, Some(0), Some(2), None]
        );
        assert_eq!(winner(&result, 4).reference_index, Some(2));
        assert_eq!(winner(&result, 4).strategy, Strategy::CropRetained);
        assert_eq!(winner(&result, 4).source, b.save_as_reference);
    }
}

#[test]
fn extra_only_and_float_bit_differences_drive_lookahead_and_survive_reused_maps() {
    for kind in [Samples::Extra16, Samples::Float32] {
        let images = recurrence(kind, 0, &[P, P + 1]);
        let result = verify(&images, &[0, 2, 0, 5], &crop_only());
        assert_eq!(winner(&result, 3).reference_index, Some(0));
        let diff = result.frames[3]
            .differences
            .iter()
            .find(|diff| diff.reference == 0)
            .unwrap();
        assert_eq!(diff.changed_pixels, 2);
        assert_eq!(
            diff.bounds,
            Some(Rect {
                x: 2,
                y: 2,
                width: 2,
                height: 1
            })
        );
        if matches!(kind, Samples::Extra16) {
            assert_eq!(images[0].pixels(), images[3].pixels());
        }
    }
}

#[test]
fn lookahead_ranks_bbox_area_before_count_and_does_not_override_old_canvas_ties() {
    // (A edits, next-input edits, X score, A score, destination for B).
    let cases = [
        (
            vec![P + 1, P + W, P + W + 1, P + 2 * W + 2],
            vec![P, P + 1, P + W, P + W + 1],
            (4, 4),
            (9, 2),
            2,
        ),
        (vec![P + 1], vec![P, P + 2], (3, 2), (3, 3), 2),
        (vec![P, P + 2], vec![P], (1, 1), (1, 1), 1),
    ];
    for (a, next, x_score, a_score, destination) in cases {
        let images = vec![
            state(Samples::Gray8, SEEDS[0], &[]),
            state(Samples::Gray8, SEEDS[0], &a),
            state(Samples::Gray8, SEEDS[2], &[]),
            state(Samples::Gray8, SEEDS[0], &next),
        ];
        assert_eq!(score(&images[0], &images[3]), x_score);
        assert_eq!(score(&images[1], &images[3]), a_score);
        let result = verify(&images, &[1; 4], &crop_only());
        assert_eq!(winner(&result, 2).strategy, Strategy::Full);
        assert_eq!(winner(&result, 2).save_as_reference, destination);
    }
}

#[test]
fn an_old_canvas_tied_with_current_does_not_override_normal_eviction() {
    let q = P + 2 * W + 2;
    let images: Vec<_> = [vec![P], vec![q, q + 1], vec![q], vec![]]
        .iter()
        .map(|edits| state(Samples::Gray8, SEEDS[0], edits))
        .collect();
    assert_eq!(score(&images[0], &images[3]), score(&images[2], &images[3]));
    let result = verify(&images, &[1; 4], &crop_only());
    assert_eq!(winner(&result, 2).reference_index, Some(1));
    assert_eq!(
        (
            winner(&result, 2).source,
            winner(&result, 2).save_as_reference
        ),
        (2, 1)
    );
    assert!(!result.frames[2].references_after.contains(&Some(0)));
}

#[test]
fn input_after_the_next_one_cannot_change_prior_choices_or_trial_work() {
    let mut a = recurrence(Samples::Gray8, 0, &[P]);
    a.push(state(Samples::Gray8, SEEDS[1], &[Q]));
    let mut b = a.clone();
    b[4] = state(Samples::Gray8, SEEDS[2], &[Q]);
    let a = verify(&a, &[1; 5], &crop_only());
    let b = verify(&b, &[1; 5], &crop_only());
    for i in 0..3 {
        let (a_report, b_report) = (&a.frames[i], &b.frames[i]);
        assert_eq!(a_report.references_after, b_report.references_after);
        assert_eq!(
            a_report.lookahead_comparisons,
            b_report.lookahead_comparisons
        );
        let signature = |result: &AutoEncodeResult| {
            let c = winner(result, i);
            (
                c.strategy,
                c.reference_index,
                c.source,
                c.save_as_reference,
                c.encoded_bytes,
                c.trial_pixels,
            )
        };
        assert_eq!(signature(&a), signature(&b));
        assert_eq!(a_report.candidates.len(), b_report.candidates.len());
        assert_eq!(a_report.pruned.len(), b_report.pruned.len());
    }
}

#[test]
fn identical_rects_from_distinct_sources_reuse_payloads_with_rebuilt_headers() {
    let images: Vec<_> = [vec![], vec![P + 2], vec![P, P + 4]]
        .iter()
        .map(|edits| state(Samples::Gray8, SEEDS[0], edits))
        .collect();
    let result = verify(
        &images,
        &[2, 3, 5],
        &AutoOptions {
            prune: false,
            ..crop_only()
        },
    );
    let report = &result.frames[2];
    assert!(report.pruned.is_empty());
    for (strategy, reference) in [(Strategy::CropPrevious, 1), (Strategy::CropFirst, 0)] {
        let candidate = report
            .candidates
            .iter()
            .find(|c| c.strategy == strategy)
            .unwrap();
        assert_eq!(candidate.reference_index, Some(reference));
        assert_eq!(candidate.trial_pixels, 5);
    }
    assert_eq!(report.differences[0].bounds, report.differences[1].bounds);
    assert_ne!(report.differences[0].source, report.differences[1].source);
    assert!(report.reused_payloads > 0);
    assert!(report.reused_payload_pixels > 0);
    assert_eq!(
        report.reused_payload_pixels,
        5 * report.reused_payloads as u64
    );
    assert_eq!(result.frames[0].reused_payloads, 0);
    assert_eq!(result.frames[1].reused_payloads, 0);
}

#[test]
fn pruning_with_unlimited_budgets_preserves_lengths_samples_and_slots() {
    let images = multirect_images();
    let auto = lookahead();
    assert_eq!(auto.max_frames, usize::MAX);
    assert_eq!(auto.max_total_trials, usize::MAX);
    assert_eq!(auto.max_trial_pixels, u64::MAX);
    let bounded = verify(&images, &[0, 2, 0, 5, 7], &auto);
    let complete = verify(
        &images,
        &[0, 2, 0, 5, 7],
        &AutoOptions {
            prune: false,
            ..auto
        },
    );
    assert_eq!(bounded.bytes.len(), complete.bytes.len());
    assert!(
        bounded
            .frames
            .iter()
            .any(|report| !report.pruned.is_empty())
    );
    for i in 0..images.len() {
        let (a, b) = (winner(&bounded, i), winner(&complete, i));
        assert_eq!(
            (a.strategy, a.reference_index, a.encoded_bytes),
            (b.strategy, b.reference_index, b.encoded_bytes)
        );
        assert_eq!(
            bounded.frames[i].references_before,
            complete.frames[i].references_before
        );
        assert_eq!(
            bounded.frames[i].references_after,
            complete.frames[i].references_after
        );
        assert_eq!(
            bounded.frames[i].lookahead_comparisons,
            complete.frames[i].lookahead_comparisons
        );
        assert!(complete.frames[i].pruned.is_empty());
    }
}

#[test]
fn lookahead_requires_retained_reference() {
    assert!(!AutoOptions::default().reference_lookahead);
    let image = state(Samples::Gray8, SEEDS[0], &[]);
    let error = encode_auto(
        &[image],
        &[1],
        &Default::default(),
        &AutoOptions {
            reference_lookahead: true,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::InvalidInput(message) if message.contains("reference_lookahead") && message.contains("retained_reference"))
    );
}
