//! Bounded independent Replace-layer trial dispatch must not change byte winners.
use rs_jxl::decode::{DecodeOptions, decode};
use rs_jxl::metadata::Animation;
use rs_jxl::modular::{encode::FixedTree, predictor::Predictor};
use rs_jxl::multiframe::{AutoOptions, Reference, encode_auto};
use rs_jxl::{EncodeOptions, Image, ModularOptions, PixelData, PixelLayout};
fn fixtures() -> Vec<Image> {
    let mut random = 0x793a5621u32;
    let first = vec![0; 128 * 128 * 3];
    let second: Vec<u8> = (0..first.len())
        .map(|_| {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random as u8
        })
        .collect();
    let mut third = second.clone();
    for y in (8..128).step_by(16) {
        for x in (8..128).step_by(16) {
            third[(y * 128 + x) * 3] ^= 127;
        }
    }
    [first, second, third]
        .into_iter()
        .map(|pixels| {
            let image = Image::new(128, 128, PixelLayout::Rgb, PixelData::U8(pixels)).unwrap();
            let mut metadata = image.metadata().clone();
            metadata.animation = Some(Animation::default());
            image.with_metadata(metadata, vec![]).unwrap()
        })
        .collect()
}
fn options(threads: usize) -> EncodeOptions {
    EncodeOptions {
        threads,
        modular: ModularOptions {
            rct: None,
            lz77: false,
            predictor: Predictor::Zero,
            fixed_tree: Some(FixedTree::Zero),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn auto(prune: bool) -> AutoOptions {
    AutoOptions {
        reference: Reference::Previous,
        allow_mask: false,
        tile_size: 16,
        max_rectangles: 16,
        prune,
        ..Default::default()
    }
}
#[test]
fn parallel_rectangles_keep_canonical_bytes_and_frames() {
    let images = fixtures();
    let durations = [2, 3, 5];
    let serial = encode_auto(&images, &durations, &options(1), &auto(false)).unwrap();
    assert!(
        serial
            .frames
            .iter()
            .flat_map(|f| &f.candidates)
            .any(|c| c.encoded_frames > 1),
        "fixture must actually encode multipart trials"
    );
    for prune in [false, true] {
        let parallel = encode_auto(&images, &durations, &options(4), &auto(prune)).unwrap();
        assert_eq!(parallel.bytes, serial.bytes, "prune={prune}");
        let decoded = decode(&parallel.bytes, &DecodeOptions::default()).unwrap();
        assert!(decoded.complete);
        assert_eq!(decoded.frames.len(), 3);
        for ((frame, source), duration) in decoded.frames.iter().zip(&images).zip(durations) {
            assert!(frame.complete);
            assert_eq!(frame.duration_ticks, duration);
            assert_eq!(frame.pixels, *source.pixels());
        }
    }
}

#[test]
fn screened_search_charges_one_trial_per_input_for_byte_and_word_samples() {
    let bytes = fixtures();
    let words: Vec<_> = bytes
        .iter()
        .map(|image| {
            let PixelData::U8(samples) = image.pixels() else {
                unreachable!()
            };
            let word_image = Image::new(
                image.width(),
                image.height(),
                image.layout(),
                PixelData::U16(samples.iter().map(|&v| u16::from(v) * 257).collect()),
            )
            .unwrap();
            let mut metadata = word_image.metadata().clone();
            metadata.animation = image.metadata().animation.clone();
            word_image.with_metadata(metadata, vec![]).unwrap()
        })
        .collect();
    for images in [&bytes, &words] {
        let search = AutoOptions {
            screen_candidates: true,
            max_total_trials: images.len(),
            ..Default::default()
        };
        let mut finite = options(4);
        finite.max_output_bytes = 1 << 20;
        let result = encode_auto(images, &[2, 3, 5], &finite, &search).unwrap();
        assert!(
            result
                .frames
                .iter()
                .all(|f| f.candidates.len() == 1 && f.pruned.is_empty())
        );
        let decoded = decode(&result.bytes, &DecodeOptions::default()).unwrap();
        assert_eq!(decoded.frames.len(), images.len());
        for (actual, expected) in decoded.frames.iter().zip(images) {
            assert_eq!(&actual.pixels, expected.pixels());
        }
    }
}
#[test]
fn parallel_trial_work_and_output_budgets_still_fail() {
    let images = fixtures();
    let durations = [2, 3, 5];
    let mut policy = auto(false);
    assert!(policy.adaptive_rate_feedback);
    let complete = encode_auto(&images, &durations, &options(1), &policy).unwrap();
    // With pruning disabled canonical candidate/report order is execution
    // order. Put the cap strictly inside the first multipart trial, after all
    // preceding complete trials, rather than failing on the initial frame.
    let mut before = 0u64;
    let multipart = complete
        .frames
        .iter()
        .flat_map(|f| &f.candidates)
        .find_map(|candidate| {
            if candidate.encoded_frames > 1 {
                Some(candidate.trial_pixels)
            } else {
                before += candidate.trial_pixels;
                None
            }
        })
        .expect("fixture must submit multipart work");
    assert!(multipart > 1);
    policy.max_trial_pixels = before + multipart - 1;
    assert!(policy.max_trial_pixels > 128 * 128);
    assert!(matches!(
        encode_auto(&images, &durations, &options(4), &policy),
        Err(rs_jxl::Error::Limit("automatic trial pixel work"))
    ));
    let total = complete
        .frames
        .iter()
        .flat_map(|f| &f.candidates)
        .map(|c| c.trial_pixels)
        .sum();
    policy.max_trial_pixels = total;
    assert_eq!(
        encode_auto(&images, &durations, &options(4), &policy)
            .unwrap()
            .bytes,
        complete.bytes,
        "exact total trial-pixel cap"
    );
    // This remains an end-to-end output-limit guard, not evidence of hitting
    // the parallel append path: the small cap may reject an earlier frame.
    let mut limits = options(4);
    limits.max_output_bytes = 128;
    assert!(encode_auto(&images, &durations, &limits, &auto(false)).is_err());

    // A finite work limit must allow the same pruned search as the serial
    // encoder, even when a speculative rectangle could exceed the budget.
    let mut pruned = auto(true);
    let serial = encode_auto(&images, &durations, &options(1), &pruned).unwrap();
    pruned.max_trial_pixels = serial
        .frames
        .iter()
        .map(|f| {
            f.candidates.iter().map(|c| c.trial_pixels).sum::<u64>()
                + f.pruned.iter().map(|c| c.trial_pixels).sum::<u64>()
        })
        .sum();
    assert_eq!(
        encode_auto(&images, &durations, &options(4), &pruned)
            .unwrap()
            .bytes,
        serial.bytes
    );
}
