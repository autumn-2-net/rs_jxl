//! Pruning is a work optimization; exact samples and encoded sizes are the oracle.
use rs_jxl::decode::{DecodeOptions, decode, decode_frame};
use rs_jxl::metadata::Animation;
use rs_jxl::multiframe::{
    AutoEncodeResult, AutoOptions, Rect, Reference, Strategy, crop, difference_bounds, encode_auto,
};
use rs_jxl::{EncodeOptions, Error, Image, PixelData, PixelLayout};

fn image(width: u32, height: u32, layout: PixelLayout, pixels: Vec<u8>) -> Image {
    let image = Image::new(width, height, layout, PixelData::U8(pixels)).unwrap();
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(Animation::default());
    image.with_metadata(metadata, vec![]).unwrap()
}

fn fixture(dense: bool, noisy: bool, hidden: bool) -> Vec<Image> {
    let (width, height) = if dense {
        (257usize, 129usize)
    } else {
        (96, 80)
    };
    let mut base = vec![0; width * height * 4];
    let mut state = 0x1234_5678u32;
    for pixel in base.chunks_exact_mut(4) {
        for value in &mut pixel[..3] {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *value = if noisy { state as u8 } else { 0 };
        }
        pixel[3] = if hidden { 0 } else { 255 };
    }
    let mut changed = base.clone();
    for y in 0..height {
        for x in 0..width {
            if dense || (x % 16 == 4 && y % 16 == 4) {
                changed[(y * width + x) * 4] ^= 0x7f;
                if hidden {
                    changed[(y * width + x) * 4 + 1] ^= 0x55;
                }
            }
        }
    }
    [base.clone(), changed.clone(), changed, base]
        .into_iter()
        .map(|pixels| image(width as u32, height as u32, PixelLayout::Rgba, pixels))
        .collect()
}

fn run(images: &[Image], auto: &AutoOptions, threads: usize) -> AutoEncodeResult {
    let result = encode_auto(
        images,
        &vec![1; images.len()],
        &EncodeOptions {
            threads,
            group_size_shift: 0,
            ..Default::default()
        },
        auto,
    )
    .unwrap();
    assert_eq!(result.frames.len(), images.len());
    let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded.frames.len(), images.len());
    for (frame, expected) in decoded.frames.iter().zip(images) {
        assert_eq!(&frame.pixels, expected.pixels());
        assert_eq!(frame.layout, expected.layout());
        assert_eq!(frame.width, expected.width() as usize);
        assert_eq!(frame.height, expected.height() as usize);
        assert_eq!(frame.duration_ticks, 1);
    }
    for index in [0, images.len() - 1] {
        let selected = decode_frame(&result.bytes, index, &DecodeOptions::default()).unwrap();
        assert_eq!(selected.frames.len(), 1);
        assert_eq!(&selected.frames[0].pixels, images[index].pixels());
    }
    for frame in &result.frames {
        assert_eq!(
            frame.winner_index,
            frame
                .candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| candidate.encoded_bytes)
                .unwrap()
                .0
        );
        for candidate in &frame.candidates {
            assert!(candidate.encoded_frames > 0);
            assert!(candidate.trial_pixels >= candidate.encoded_frames as u64);
        }
        let winner = &frame.candidates[frame.winner_index];
        for pruned in &frame.pruned {
            assert!(pruned.lower_bound >= winner.encoded_bytes);
            assert!(pruned.planned_frames > 0);
            assert!(pruned.encoded_frames <= pruned.planned_frames);
            assert!(pruned.trial_pixels >= pruned.encoded_frames as u64);
        }
    }
    result
}

fn work(result: &AutoEncodeResult) -> (usize, u64) {
    result
        .frames
        .iter()
        .fold((0, 0), |(trials, pixels), frame| {
            (
                trials + frame.candidates.len() + frame.pruned.len(),
                pixels
                    + frame.candidates.iter().map(|c| c.trial_pixels).sum::<u64>()
                    + frame.pruned.iter().map(|c| c.trial_pixels).sum::<u64>(),
            )
        })
}

fn compare_pruning(full: &AutoEncodeResult, pruned: &AutoEncodeResult) {
    assert_eq!(pruned.bytes.len(), full.bytes.len());
    assert_eq!(pruned.frames.len(), full.frames.len());
    assert_eq!(work(pruned).0, work(full).0);
    assert!(work(pruned).1 <= work(full).1);
    for (expected, actual) in full.frames.iter().zip(&pruned.frames) {
        assert!(expected.pruned.is_empty());
        let winner = &expected.candidates[expected.winner_index];
        let actual_winner = &actual.candidates[actual.winner_index];
        assert_eq!(actual_winner.strategy, winner.strategy);
        assert_eq!(actual_winner.encoded_bytes, winner.encoded_bytes);
        assert_eq!(actual_winner.encoded_frames, winner.encoded_frames);
        assert_eq!(
            actual.candidates.len() + actual.pruned.len(),
            expected.candidates.len()
        );
        for candidate in &expected.candidates {
            let completed: Vec<_> = actual
                .candidates
                .iter()
                .filter(|c| c.strategy == candidate.strategy)
                .collect();
            let stopped: Vec<_> = actual
                .pruned
                .iter()
                .filter(|c| c.strategy == candidate.strategy)
                .collect();
            assert_eq!(completed.len() + stopped.len(), 1);
            if let Some(completed) = completed.first() {
                assert_eq!(completed.encoded_bytes, candidate.encoded_bytes);
                assert_eq!(completed.encoded_frames, candidate.encoded_frames);
                assert_eq!(completed.trial_pixels, candidate.trial_pixels);
            } else {
                let stopped = stopped[0];
                assert!(stopped.lower_bound <= candidate.encoded_bytes);
                assert_eq!(stopped.planned_frames, candidate.encoded_frames);
                assert!(stopped.trial_pixels <= candidate.trial_pixels);
            }
        }
    }
}

#[test]
fn pruning_defaults_on_and_preserves_dense_and_sparse_results() {
    assert!(AutoOptions::default().prune);
    assert!(AutoOptions::default().adaptive_rectangles);
    for dense in [false, true] {
        let images = fixture(dense, false, false);
        for reference in [
            Reference::Previous,
            Reference::First,
            Reference::SmallestCrop,
        ] {
            for adaptive_rectangles in [false, true] {
                let auto = AutoOptions {
                    reference,
                    tile_size: 16,
                    adaptive_rectangles,
                    prune: false,
                    ..Default::default()
                };
                let full = run(&images, &auto, 1);
                let pruned = run(
                    &images,
                    &AutoOptions {
                        prune: true,
                        ..auto
                    },
                    1,
                );
                compare_pruning(&full, &pruned);
                assert!(
                    !pruned.frames[1].pruned.is_empty(),
                    "the changed frame must actually prune: dense={dense}, {reference:?}"
                );
                if !dense {
                    assert!(work(&pruned).1 < work(&full).1);
                    assert!(
                        pruned.frames[1].pruned.iter().any(|c| {
                            c.planned_frames > 1 && c.encoded_frames < c.planned_frames
                        })
                    );
                }
            }
        }
    }
}

#[test]
fn equal_size_candidates_keep_the_first_winner_across_threads() {
    let base = fixture(false, true, true).remove(0);
    let images = vec![base; 3];
    let auto = AutoOptions {
        prune: false,
        ..Default::default()
    };
    let full = run(&images, &auto, 1);
    let last = &full.frames[2];
    let previous = last
        .candidates
        .iter()
        .position(|c| c.strategy == Strategy::CropPrevious)
        .unwrap();
    let first = last
        .candidates
        .iter()
        .position(|c| c.strategy == Strategy::CropFirst)
        .unwrap();
    assert!(previous < first);
    assert_eq!(
        last.candidates[previous].encoded_bytes,
        last.candidates[first].encoded_bytes
    );
    assert_eq!(last.winner_index, previous);
    for threads in [1, 4, 1] {
        compare_pruning(
            &full,
            &run(
                &images,
                &AutoOptions {
                    prune: true,
                    ..auto.clone()
                },
                threads,
            ),
        );
    }
}

#[test]
fn difference_reports_count_pixels_once_including_hidden_rgb() {
    let images = fixture(false, true, true);
    for reference in [
        Reference::Full,
        Reference::Previous,
        Reference::First,
        Reference::SmallestCrop,
    ] {
        let result = run(
            &images,
            &AutoOptions {
                reference,
                ..Default::default()
            },
            1,
        );
        for (index, report) in result.frames.iter().enumerate() {
            let references = match reference {
                _ if index == 0 => vec![],
                Reference::Full => vec![],
                Reference::Previous => vec![index - 1],
                Reference::First => vec![0],
                Reference::SmallestCrop if index == 1 => vec![0],
                Reference::SmallestCrop => vec![index - 1, 0],
            };
            let mut expected = Vec::new();
            for previous in references {
                let (PixelData::U8(a), PixelData::U8(b)) =
                    (images[previous].pixels(), images[index].pixels())
                else {
                    unreachable!()
                };
                let count = a
                    .chunks_exact(4)
                    .zip(b.chunks_exact(4))
                    .filter(|(a, b)| a != b)
                    .count() as u64;
                let bounds = difference_bounds(&images[previous], &images[index]).unwrap();
                expected.push((
                    previous,
                    count,
                    bounds.map(|b| (b.x, b.y, b.width, b.height)),
                ));
            }
            let mut actual: Vec<_> = report
                .differences
                .iter()
                .map(|d| {
                    (
                        d.reference,
                        d.changed_pixels,
                        d.bounds.map(|b| (b.x, b.y, b.width, b.height)),
                    )
                })
                .collect();
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected, "frame {index}, {reference:?}");
        }
    }
}

#[test]
fn exact_budgets_charge_completed_and_pruned_attempts() {
    let images = fixture(false, false, false);
    let durations = vec![1; images.len()];
    for prune in [false, true] {
        let auto = AutoOptions {
            prune,
            tile_size: 16,
            ..Default::default()
        };
        let unbounded = run(&images, &auto, 1);
        let (trials, pixels) = work(&unbounded);
        let exact = AutoOptions {
            max_frames: images.len(),
            max_total_trials: trials,
            max_trial_pixels: pixels,
            ..auto
        };
        let bounded = run(&images, &exact, 1);
        assert_eq!(work(&bounded), (trials, pixels));
        assert_eq!(bounded.bytes.len(), unbounded.bytes.len());
        for limit in [
            AutoOptions {
                max_frames: 0,
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
            AutoOptions {
                max_frames: images.len() - 1,
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
        ] {
            assert!(matches!(
                encode_auto(
                    &images,
                    &durations,
                    &EncodeOptions {
                        group_size_shift: 0,
                        ..Default::default()
                    },
                    &limit
                ),
                Err(Error::Limit(_))
            ));
        }
        for invalid in [
            AutoOptions {
                tile_size: 0,
                ..exact.clone()
            },
            AutoOptions {
                max_rectangles: 0,
                ..exact.clone()
            },
        ] {
            assert!(matches!(
                encode_auto(&images, &durations, &EncodeOptions::default(), &invalid),
                Err(Error::InvalidInput(_))
            ));
        }
    }
}

fn overflow_fixture() -> Vec<Image> {
    let (width, height) = (129usize, 97usize);
    let mut state = 0x1234_5678u32;
    let base: Vec<u8> = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    let mut changed = base.clone();
    for y in 0..5 {
        for x in 0..8 {
            changed[(y * 16 + 3) * width + x * 16 + 3] ^= 0x55;
        }
    }
    [base, changed]
        .into_iter()
        .map(|pixels| image(width as u32, height as u32, PixelLayout::Gray, pixels))
        .collect()
}

#[test]
fn adaptive_candidates_recover_more_than_thirty_two_changed_tiles() {
    let images = overflow_fixture();
    for (reference, tiles, merged) in [
        (
            Reference::Previous,
            Strategy::TilesPrevious,
            Strategy::MergedPrevious,
        ),
        (
            Reference::First,
            Strategy::TilesFirst,
            Strategy::MergedFirst,
        ),
    ] {
        let auto = AutoOptions {
            reference,
            tile_size: 4,
            max_rectangles: 32,
            prune: false,
            adaptive_rectangles: false,
            ..Default::default()
        };
        let legacy = run(&images, &auto, 1);
        assert!(
            legacy.frames[1]
                .candidates
                .iter()
                .all(|c| c.strategy != tiles)
        );
        let adaptive = run(
            &images,
            &AutoOptions {
                adaptive_rectangles: true,
                ..auto
            },
            1,
        );
        let candidate = adaptive.frames[1]
            .candidates
            .iter()
            .find(|c| c.strategy == merged)
            .expect("adaptive rectangles must recover a bounded candidate after tile overflow");
        assert!(candidate.encoded_frames > 1 && candidate.encoded_frames <= 32);
        assert_eq!(adaptive.frames[1].differences[0].changed_pixels, 40);
        assert!(adaptive.bytes.len() <= legacy.bytes.len());
    }
}

#[test]
fn zero_duration_merged_layers_preserve_samples_and_reference_slots() {
    let mut images = overflow_fixture();
    images.push(images[1].clone());
    let durations = [0, 0, 7];
    for reference in [
        Reference::Previous,
        Reference::First,
        Reference::SmallestCrop,
    ] {
        let mut results = Vec::new();
        for prune in [false, true] {
            let result = encode_auto(
                &images,
                &durations,
                &EncodeOptions {
                    group_size_shift: 0,
                    ..Default::default()
                },
                &AutoOptions {
                    reference,
                    prune,
                    tile_size: 4,
                    ..Default::default()
                },
            )
            .unwrap();
            let middle = &result.frames[1];
            let winner = &middle.candidates[middle.winner_index];
            assert!(matches!(
                winner.strategy,
                Strategy::MergedPrevious | Strategy::MergedFirst
            ));
            assert!(winner.encoded_frames > 1);
            let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
            assert_eq!(decoded.frames.len(), 1);
            assert_eq!(&decoded.frames[0].pixels, images[2].pixels());
            assert_eq!(decoded.frames[0].duration_ticks, 7);
            let selected = decode_frame(&result.bytes, 0, &DecodeOptions::default()).unwrap();
            assert_eq!(selected.frames.len(), 1);
            assert_eq!(&selected.frames[0].pixels, images[2].pixels());
            let raw = decode(
                &result.bytes,
                &DecodeOptions {
                    coalescing: false,
                    ..Default::default()
                },
            )
            .unwrap();
            let start = result.frames[0].candidates[result.frames[0].winner_index].encoded_frames;
            for (part, layer) in raw.frames[start..start + winner.encoded_frames]
                .iter()
                .enumerate()
            {
                let info = &layer.header.layer_info;
                let expected = crop(
                    &images[1],
                    Rect {
                        x: info.crop_origin.0 as u32,
                        y: info.crop_origin.1 as u32,
                        width: layer.width as u32,
                        height: layer.height as u32,
                    },
                )
                .unwrap();
                assert_eq!(&layer.pixels, expected.pixels());
                assert_eq!(layer.duration_ticks, 0);
                assert!(!layer.header.is_last);
                assert_eq!(info.save_as_reference, Some(2));
                assert_eq!(info.blend_info.source, if part == 0 { 1 } else { 2 });
            }
            results.push(result);
        }
        compare_pruning(&results[0], &results[1]);
    }
}

#[test]
fn adaptive_search_retains_every_legacy_candidate_without_size_regression() {
    for (dense, noisy, hidden) in [
        (false, false, false),
        (true, false, false),
        (false, true, true),
        (true, true, true),
    ] {
        let images = fixture(dense, noisy, hidden);
        for reference in [
            Reference::Previous,
            Reference::First,
            Reference::SmallestCrop,
        ] {
            let auto = AutoOptions {
                reference,
                tile_size: 16,
                prune: false,
                adaptive_rectangles: false,
                ..Default::default()
            };
            let legacy = run(&images, &auto, 1);
            let adaptive_options = AutoOptions {
                adaptive_rectangles: true,
                ..auto
            };
            let adaptive = run(&images, &adaptive_options, 1);
            assert!(adaptive.bytes.len() <= legacy.bytes.len());
            for (old, new) in legacy.frames.iter().zip(&adaptive.frames) {
                for candidate in &old.candidates {
                    let retained = new
                        .candidates
                        .iter()
                        .find(|c| c.strategy == candidate.strategy)
                        .expect("adaptive search must retain legacy plans");
                    assert_eq!(retained.encoded_bytes, candidate.encoded_bytes);
                    assert_eq!(retained.encoded_frames, candidate.encoded_frames);
                    assert_eq!(retained.trial_pixels, candidate.trial_pixels);
                }
            }
            let pruned = run(
                &images,
                &AutoOptions {
                    prune: true,
                    ..adaptive_options
                },
                1,
            );
            compare_pruning(&adaptive, &pruned);
            assert!(pruned.bytes.len() <= legacy.bytes.len());
        }
    }
}
