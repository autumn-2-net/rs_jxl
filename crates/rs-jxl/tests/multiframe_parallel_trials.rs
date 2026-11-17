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
fn parallel_unbounded_rectangles_keep_canonical_bytes_and_frames() {
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
}
