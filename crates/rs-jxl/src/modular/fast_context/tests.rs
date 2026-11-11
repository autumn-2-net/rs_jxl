use super::*;
use crate::decode::{DecodeOptions, decode};
use crate::modular::{encode::ModularOptions, group::Channel, training, weighted};
use crate::{EncodeOptions, Image, PixelData, PixelLayout};

fn fast(mode: Mode, bias: Bias) -> Options {
    Options {
        mode,
        bias,
        scan: Scan::Auto,
        predictors: PredictorSet::Basic,
        cross_channel: false,
        receipt: Some(Arc::new(Receipt::default())),
    }
}

#[test]
fn every_plan_roundtrips_integer_depths_and_hidden_rgb() {
    for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
        for bits in [8, 16, 31] {
            let max = (1u32 << bits) - 1;
            let words: Vec<_> = (0..35 * 7 * layout.channels())
                .map(|i| {
                    if layout == PixelLayout::Rgba && i % 4 == 3 {
                        0
                    } else {
                        ((i as u32).wrapping_mul(0x9e3779b9) ^ ((i / 35) as u32 * 101)) & max
                    }
                })
                .collect();
            let pixels = match bits {
                8 => PixelData::U8(words.iter().map(|&v| v as u8).collect()),
                16 => PixelData::U16(words.iter().map(|&v| v as u16).collect()),
                _ => PixelData::U32(words),
            };
            let image = Image::new(35, 7, layout, pixels).unwrap();
            for (mode, bias, predictors) in [
                (Mode::Leaf, Bias::Off),
                (Mode::Gradient12, Bias::Off),
                (Mode::Directional12, Bias::Off),
                (Mode::Auto12, Bias::Off),
                (Mode::Auto12, Bias::Static),
                (Mode::Auto12, Bias::Adaptive),
            ]
            .into_iter()
            .flat_map(|(mode, bias)| {
                [PredictorSet::Basic, PredictorSet::Full].map(|predictors| (mode, bias, predictors))
            }) {
                let mut fast = fast(mode, bias);
                fast.predictors = predictors;
                let options = EncodeOptions {
                    threads: 2,
                    modular: ModularOptions {
                        fast_context: Some(fast.clone()),
                        max_merged_tree_leaves: Some(256),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let bytes = crate::encode(&image, &options).unwrap();
                let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
                assert_eq!(
                    decoded.frames[0].pixels,
                    *image.pixels(),
                    "{bits} {layout:?} {mode:?} {bias:?} {predictors:?}"
                );
                assert!(fast.receipt.as_ref().unwrap().snapshot().trees > 0);
            }
        }
    }
}

#[test]
fn full_roundtrips_hidden_rgba_with_every_weighted_header_preset() {
    for bits in [8, 16, 31] {
        let max = (1u32 << bits) - 1;
        let words: Vec<_> = (0..35 * 9 * 4)
            .map(|i| {
                if i % 4 == 3 {
                    0
                } else {
                    ((i as u32).wrapping_mul(0x9371_05ab) ^ (i / 7) as u32) & max
                }
            })
            .collect();
        let pixels = match bits {
            8 => PixelData::U8(words.iter().map(|&v| v as u8).collect()),
            16 => PixelData::U16(words.iter().map(|&v| v as u16).collect()),
            _ => PixelData::U32(words),
        };
        let image = Image::new(35, 9, PixelLayout::Rgba, pixels).unwrap();
        for preset in 0..5 {
            let mut fast = fast(Mode::Auto12, Bias::Adaptive);
            fast.predictors = PredictorSet::Full;
            let options = EncodeOptions {
                modular: ModularOptions {
                    fast_context: Some(fast),
                    weighted_preset: preset,
                    max_merged_tree_leaves: Some(256),
                    ..Default::default()
                },
                ..Default::default()
            };
            let bytes = crate::encode(&image, &options).unwrap();
            let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
            assert_eq!(decoded.frames[0].pixels, *image.pixels(), "{bits} {preset}");
        }
    }
}

#[test]
fn bounds_limits_and_noncontiguous_group_ids_are_respected() {
    let channels = [Channel {
        width: 33,
        height: 3,
        hshift: 0,
        vshift: 0,
        samples: (0..99).map(|i| (i * 71) % 255).collect(),
    }];
    let streams = [
        training::TrainingStream {
            channels: &channels,
            wp: weighted::Header::default(),
            stream_id: 19,
        },
        training::TrainingStream {
            channels: &channels,
            wp: weighted::Header::default(),
            stream_id: 3,
        },
    ];
    for limit in [1, 2, 7, 64] {
        let mut options = ModularOptions {
            fast_context: Some(fast(Mode::Gradient12, Bias::Static)),
            ma_max_leaves: Some(limit),
            max_merged_tree_leaves: Some(64),
            sample_limit: Some(198),
            ..Default::default()
        };
        let tree = training::learn(&streams, &options).unwrap();
        assert!(tree.leaves() <= limit);
        assert!(tree.nodes().iter().all(|node| match node {
            Node::Split { property, .. } => [0, 10, 11].contains(property),
            Node::Leaf {
                predictor,
                multiplier,
                ..
            } => PREDICTORS.contains(predictor) && *multiplier == 1,
        }));
        options.sample_limit = Some(197);
        assert!(matches!(
            training::learn(&streams, &options),
            Err(Error::Limit(_))
        ));
        options.sample_limit = None;
        assert!(
            training::learn_with_budget(
                &streams,
                &options,
                Some(&training::SampleBudget::new(197)),
                true
            )
            .is_err()
        );
    }
}

#[test]
fn conflicting_predefined_trees_fail_instead_of_silently_ignoring_experiment() {
    let mut options = ModularOptions::default();
    assert!(options.fast_context.is_none());
    options.fast_context = Some(fast(Mode::Auto12, Bias::Off));
    options.fixed_tree = Some(crate::modular::encode::FixedTree::GradientLeaf);
    assert!(options.validate().is_err());
    options.fixed_tree = None;
    options.custom_tree = Some(Tree::channel_predictors(&[Predictor::Zero]).unwrap());
    assert!(options.validate().is_err());
}

#[test]
fn cross_channel_axis_uses_available_budget_and_has_its_own_receipt() {
    let mut random = 91u32;
    let pixels: Vec<_> = (0..67 * 31)
        .map(|_| {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            (random >> 16) as i32
        })
        .collect();
    let plane = Channel {
        width: 67,
        height: 31,
        hshift: 0,
        vshift: 0,
        samples: pixels,
    };
    let channels = [plane.clone(), plane];
    let streams = [training::TrainingStream {
        channels: &channels,
        wp: weighted::Header::default(),
        stream_id: 19,
    }];
    for previous in [0, 1] {
        let receipt = Arc::new(Receipt::default());
        let options = ModularOptions {
            previous_channels: previous,
            fast_context: Some(Options {
                cross_channel: true,
                receipt: Some(receipt.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tree = training::learn(&streams, &options).unwrap();
        let selected = tree
            .nodes()
            .iter()
            .any(|node| matches!(node, Node::Split { property: 18, .. }));
        assert_eq!(selected, previous != 0);
        assert_eq!(receipt.snapshot().cross_channel != 0, selected);
    }
    assert!(
        Options {
            mode: Mode::Leaf,
            cross_channel: true,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn cross_channel_plans_and_offsets_roundtrip_hidden_rgb_and_integer_depths() {
    use crate::modular::refinement::{OffsetMode, OffsetOptions};
    for bits in [8, 16, 31] {
        let max = (1u32 << bits) - 1;
        let words: Vec<_> = (0..35 * 9 * 4)
            .map(|i| {
                if i % 4 == 3 {
                    0
                } else {
                    (((i / 4) as u32).wrapping_mul(0x9e3779b9) ^ (i % 4) as u32 * 71) & max
                }
            })
            .collect();
        let pixels = match bits {
            8 => PixelData::U8(words.iter().map(|&v| v as u8).collect()),
            16 => PixelData::U16(words.iter().map(|&v| v as u16).collect()),
            _ => PixelData::U32(words),
        };
        let image = Image::new(35, 9, PixelLayout::Rgba, pixels).unwrap();
        for predictors in [PredictorSet::Basic, PredictorSet::Full] {
            let mut options = EncodeOptions::default();
            options.threads = 2;
            options.modular.previous_channels = 1;
            options.modular.fast_context = Some(Options {
                mode: Mode::Auto12,
                bias: Bias::Static,
                predictors,
                cross_channel: true,
                ..Default::default()
            });
            options.modular.refinement.offsets = Some(OffsetOptions {
                mode: OffsetMode::Distribution,
                ..Default::default()
            });
            let bytes = crate::encode(&image, &options).unwrap();
            let decoded = decode(&bytes, &DecodeOptions::default()).unwrap();
            assert_eq!(
                decoded.frames[0].pixels,
                *image.pixels(),
                "{bits} {predictors:?}"
            );
        }
    }
}

#[test]
fn quantile_coordinates_cover_wide_values_without_new_property_ids() {
    let pixels = [
        -1_000_000,
        1_000_000,
        17,
        -10,
        i32::MIN,
        i32::MAX,
        7,
        -900_000,
    ];
    let view = ChannelView {
        channel: 0,
        group: 7,
        width: 4,
        height: 2,
        stride: 4,
        pixels: &pixels,
        is_meta: false,
    };
    for property in [10, 11] {
        let coordinates = thresholds(&[&view], property, &Previous::new()).unwrap();
        assert!(coordinates.windows(2).all(|p| p[0] < p[1]));
        assert!(coordinates.len() < 12);
        assert!(coordinates.iter().all(|&v| v < i32::MAX));
        assert!(coordinates.iter().any(|v| v.unsigned_abs() > 65535));
    }
}
