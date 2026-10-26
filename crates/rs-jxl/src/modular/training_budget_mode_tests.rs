use super::*;
use crate::modular::ma_train::MaBudgetMode;

fn options(source: bool) -> ModularOptions {
    let mut policy = SamplingPolicy::from_source(&sample::SourcePolicy::default()).unwrap();
    policy.nb_repeats = 1.0;
    policy.properties = vec![3];
    policy.node_threshold = 0.0;
    ModularOptions {
        predictor: Predictor::Zero,
        rct: None,
        lz77: false,
        tree_samples: 32,
        max_tree_leaves: 64,
        max_merged_tree_leaves: Some(128),
        ma_max_leaves: Some(1),
        split_threshold: 0.0,
        sampling: source.then_some(policy),
        ..Default::default()
    }
}

fn channels() -> [Channel; 1] {
    [Channel {
        width: 32,
        height: 1,
        hshift: 0,
        vshift: 0,
        samples: (0..32).map(|i| i32::from(i >= 16)).collect(),
    }]
}

#[test]
fn ma_budget_mode_source_and_legacy_defaults_and_overrides_reach_prepared_learning() {
    let channels = channels();
    let streams = [TrainingStream {
        channels: &channels,
        stream_id: 21,
        wp: weighted::Header::default(),
    }];
    for source in [false, true] {
        let mut options = options(source);
        assert_eq!(learn(&streams, &options).is_err(), source);
        options.ma_budget_mode = Some(MaBudgetMode::Strict);
        assert!(matches!(learn(&streams, &options), Err(Error::Limit(_))));
        options.ma_budget_mode = Some(MaBudgetMode::Bounded);
        assert_eq!(learn(&streams, &options).unwrap().leaves(), 1);
        assert_eq!(learn_local(&streams, &options).unwrap().leaves(), 1);
        options.ma_budget_mode = Some(MaBudgetMode::Strict);
        options.ma_max_leaves = Some(2);
        assert_eq!(learn(&streams, &options).unwrap().leaves(), 2);
        if source {
            // The exhaustion policy never changes selected-sample accounting.
            options.ma_budget_mode = Some(MaBudgetMode::Bounded);
            options.sample_limit = Some(31);
            assert!(matches!(learn(&streams, &options), Err(Error::Limit(_))));
        }
    }
}

#[test]
fn ma_budget_mode_explicit_constant_suffix_cannot_be_dropped_to_fit() {
    let channels: Vec<_> = [7, 9]
        .into_iter()
        .map(|value| Channel {
            width: 32,
            height: 1,
            hshift: 0,
            vshift: 0,
            samples: vec![value; 32],
        })
        .collect();
    let streams = [TrainingStream {
        channels: &channels,
        stream_id: 0,
        wp: weighted::Header::default(),
    }];
    for source in [false, true] {
        for mode in [MaBudgetMode::Strict, MaBudgetMode::Bounded] {
            let mut options = options(source);
            options.source_constant_channels = true;
            options.ma_budget_mode = Some(mode);
            assert!(matches!(learn(&streams, &options), Err(Error::Limit(_))));
            options.ma_max_leaves = Some(2);
            let tree = learn(&streams, &options).unwrap();
            assert_eq!(tree.leaves(), 2);
            assert_eq!(tree.lookup(&[0, 0]).unwrap().2, 7);
            assert_eq!(tree.lookup(&[1, 0]).unwrap().2, 9);
        }
    }
}

#[test]
fn ma_budget_mode_validates_and_clamps_additional_cap_without_changing_defaults() {
    let defaults = ModularOptions::default();
    assert_eq!(defaults.ma_budget_mode, None);
    assert_eq!(defaults.ma_max_leaves, None);
    let mut options = options(true);
    for bad in [0, ma_train::MAX_TREE_LEAVES + 1, usize::MAX] {
        options.ma_max_leaves = Some(bad);
        assert!(options.validate().is_err());
    }
    for (merged, cap, expected) in [
        (None, None, 64),
        (None, Some(128), 64),
        (Some(128), Some(256), 128),
        (Some(128), Some(2), 2),
        (Some(1), Some(2), 1),
    ] {
        options.max_merged_tree_leaves = merged;
        options.ma_max_leaves = cap;
        options.ma_budget_mode = Some(MaBudgetMode::Bounded);
        options.validate().unwrap();
        let train = encode::train_options(&options);
        assert_eq!(train.tree_leaf_limit(), expected);
        assert_eq!(train.budget_mode, options.ma_budget_mode);
    }
}

#[test]
fn ma_budget_mode_bounded_encodes_roundtrip_including_invisible_rgb() {
    use crate::decode::{DecodeOptions, SampleFormat, decode};
    use crate::{EncodeOptions, Image, PixelData, PixelLayout};
    for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
        let mut pixels = Vec::new();
        for i in 0..256 {
            for channel in 0..layout.channels() {
                pixels.push(if channel == 3 {
                    0
                } else {
                    ((i % 64 >= 32) as u8) * 128 + channel as u8 * 20
                });
            }
        }
        let image = Image::new(64, 4, layout, PixelData::U8(pixels)).unwrap();
        for threads in [1, 4] {
            for shared_tree in [false, true] {
                let mut encode_options = EncodeOptions {
                    threads,
                    shared_tree,
                    modular: options(true),
                    ..Default::default()
                };
                encode_options.modular.ma_budget_mode = Some(MaBudgetMode::Strict);
                assert!(matches!(
                    crate::encode(&image, &encode_options),
                    Err(Error::Limit(_))
                ));
                encode_options.modular.ma_budget_mode = Some(MaBudgetMode::Bounded);
                let bytes = crate::encode(&image, &encode_options).unwrap();
                let decoded = decode(
                    &bytes,
                    &DecodeOptions {
                        format: SampleFormat::U8,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(decoded.frames.len(), 1);
                assert_eq!(&decoded.frames[0].pixels, image.pixels());
            }
        }
    }
}
