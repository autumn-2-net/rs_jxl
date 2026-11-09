use super::*;
use crate::modular::{
    ma_train::{self, MaBudgetMode, Sample},
    predictor::Predictor,
    tree::Tree,
};

fn sample(x: i32, gradient: i32, residuals: &[i32]) -> Sample {
    let mut properties = vec![0; 16];
    properties[3] = x;
    properties[10] = gradient;
    Sample {
        properties,
        residuals: residuals.to_vec(),
        weight: 100,
    }
}

fn options(mode: SeedMode) -> TrainOptions {
    TrainOptions {
        properties: vec![3, 10],
        split_threshold: 0.0,
        parallel: false,
        seed: Some(SeedOptions {
            mode,
            depth: 1,
            receipt: Some(Arc::new(SeedReceipt::default())),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn check(tree: &Tree, samples: &[Sample], expected: &[Predictor]) {
    for (sample, &predictor) in samples.iter().zip(expected) {
        assert_eq!(tree.lookup(&sample.properties).unwrap().1, predictor);
    }
}

#[test]
fn defaults_and_invalid_configuration() {
    assert!(TrainOptions::default().seed.is_none());
    assert!(
        crate::modular::encode::ModularOptions::default()
            .source_ma_seed
            .is_none()
    );
    for depth in [0, 9] {
        assert!(
            SeedOptions {
                depth,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    for split_bits in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(
            SeedOptions {
                split_bits,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
    let mut invalid = options(SeedMode::Repair);
    invalid.search.histogram_bins = 4;
    assert!(matches!(
        ma_train::train(&[], &invalid),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn refinement_can_use_properties_outside_the_initial_dictionary() {
    let samples = [sample(-7, 0, &[0, 400]), sample(13, 0, &[400, 0])];
    let only = ma_train::train(&samples, &options(SeedMode::Only)).unwrap();
    assert_eq!(only.leaves(), 1);
    for mode in [SeedMode::Refine, SeedMode::Repair] {
        let options = options(mode);
        let tree = ma_train::train(&samples, &options).unwrap();
        assert_eq!(tree.leaves(), 2);
        check(&tree, &samples, &[Predictor::Weighted, Predictor::Gradient]);
        let stats = options.seed.unwrap().receipt.unwrap().snapshot();
        assert_eq!(stats.learners, 1);
        assert_eq!(stats.prepared_records, 2);
        assert_eq!(stats.refined_leaves, 1);
        assert!(stats.histogram_reuses > 0);
    }
}

#[test]
fn seed_depth_does_not_limit_final_depth_and_all_predictors_remain_available() {
    let predictors = [
        Predictor::Weighted,
        Predictor::Gradient,
        Predictor::Top,
        Predictor::Average0,
    ];
    let samples: Vec<_> = (0..4)
        .map(|i| {
            let mut residuals = vec![400; 4];
            residuals[i] = 0;
            sample((i % 2) as i32, (i / 2) as i32, &residuals)
        })
        .collect();
    for bytes in [0, 32 << 20] {
        for mode in [SeedMode::Only, SeedMode::Refine, SeedMode::Repair] {
            let mut options = options(mode);
            options.predictors = predictors.to_vec();
            options.histogram_cache_bytes = bytes;
            let tree = ma_train::train(&samples, &options).unwrap();
            if mode == SeedMode::Only {
                assert_eq!(tree.leaves(), 2);
            } else {
                assert_eq!(tree.leaves(), 4);
                check(&tree, &samples, &predictors);
            }
        }
    }
}

#[test]
fn repair_replaces_a_misleading_seed_partition() {
    let mut samples = Vec::new();
    for x in 0..2 {
        for gradient in 0..2 {
            let mut s = sample(x, gradient, if x == 0 { &[0, 400] } else { &[400, 0] });
            s.weight = if x == gradient { 1000 } else { 100 };
            samples.push(s);
        }
    }
    let options = options(SeedMode::Repair);
    let tree = ma_train::train(&samples, &options).unwrap();
    assert_eq!(tree.leaves(), 2);
    check(
        &tree,
        &samples,
        &[
            Predictor::Weighted,
            Predictor::Weighted,
            Predictor::Gradient,
            Predictor::Gradient,
        ],
    );
    let stats = options.seed.unwrap().receipt.unwrap().snapshot();
    assert_eq!(stats.seed_splits, 1);
    assert_eq!(stats.repaired_branches, 1);
}

#[test]
fn seed_model_cost_can_reject_unhelpful_branch_and_budgets_are_explicit() {
    let samples = [sample(0, -7, &[0, 400]), sample(0, 13, &[400, 0])];
    let mut options = options(SeedMode::Only);
    options.seed.as_mut().unwrap().split_bits = 1e9;
    assert_eq!(ma_train::train(&samples, &options).unwrap().leaves(), 1);
    assert_eq!(
        options
            .seed
            .as_ref()
            .unwrap()
            .receipt
            .as_ref()
            .unwrap()
            .snapshot()
            .merged_branches,
        1
    );
    options.seed.as_mut().unwrap().split_bits = 0.0;
    options.max_leaves = 1;
    options.budget_mode = Some(MaBudgetMode::Bounded);
    assert_eq!(ma_train::train(&samples, &options).unwrap().leaves(), 1);
    options.budget_mode = Some(MaBudgetMode::Strict);
    assert!(ma_train::train(&samples, &options).is_err());
    options.seed.as_mut().unwrap().split_bits = 1e9;
    for limit in [1, 2] {
        options.max_leaves = limit;
        assert_eq!(ma_train::train(&samples, &options).unwrap().leaves(), 1);
    }
}

#[test]
fn empty_preparation_keeps_a_legal_leaf() {
    for mode in [SeedMode::Only, SeedMode::Refine, SeedMode::Repair] {
        assert_eq!(ma_train::train(&[], &options(mode)).unwrap().leaves(), 1);
    }
}

#[test]
fn multiplier_and_constant_partitions_survive_grafting() {
    use ma_train::{ConstantChannel, MultiplierInfo};
    for mode in [SeedMode::Only, SeedMode::Refine, SeedMode::Repair] {
        let mut options = options(mode);
        options.properties = vec![0, 1, 10];
        options.static_property_range = Some([[0, 3], [0, 1]]);
        options.max_leaves = 6;
        options.multiplier_info = vec![
            MultiplierInfo {
                range: [[0, 1], [0, 1]],
                multiplier: 3,
            },
            MultiplierInfo {
                range: [[1, 3], [0, 1]],
                multiplier: 5,
            },
        ];
        options.constant_channels = vec![ConstantChannel {
            channel: 2,
            value: -11,
        }];
        let mut samples = Vec::new();
        for channel in 0..2 {
            let multiplier = if channel == 0 { 3 } else { 5 };
            for gradient in 0..2 {
                let mut s = sample(
                    0,
                    gradient,
                    if gradient == 0 { &[0, 300] } else { &[300, 0] },
                );
                s.properties[0] = channel;
                for value in &mut s.residuals {
                    *value *= multiplier;
                }
                samples.push(s);
            }
        }
        let tree = ma_train::train(&samples, &options).unwrap();
        for s in &samples {
            let (_, _, offset, multiplier) = tree.lookup(&s.properties).unwrap();
            assert_eq!(offset, 0);
            assert_eq!(multiplier, if s.properties[0] == 0 { 3 } else { 5 });
        }
        let (_, predictor, offset, multiplier) = tree.lookup(&[2, 0]).unwrap();
        assert_eq!((predictor, offset, multiplier), (Predictor::Zero, -11, 1));
    }
}

#[test]
fn seeded_native_encoder_roundtrips_depths_and_invisible_rgb() {
    use crate::decode::{DecodeOptions, decode};
    use crate::modular::encode::ModularOptions;
    use crate::{EncodeOptions, Image, PixelData, PixelLayout};
    for bits in [8, 16, 31] {
        for layout in [PixelLayout::Gray, PixelLayout::Rgba] {
            let mask = (1u32 << bits) - 1;
            let words: Vec<_> = (0..35 * 7 * layout.channels())
                .map(|i| {
                    if layout == PixelLayout::Rgba && i % 4 == 3 {
                        0
                    } else {
                        (i as u32).wrapping_mul(0x9e3779b9) & mask
                    }
                })
                .collect();
            let pixels = match bits {
                8 => PixelData::U8(words.iter().map(|&v| v as u8).collect()),
                16 => PixelData::U16(words.iter().map(|&v| v as u16).collect()),
                _ => PixelData::U32(words),
            };
            let image = Image::new(35, 7, layout, pixels).unwrap();
            for mode in [SeedMode::Only, SeedMode::Refine, SeedMode::Repair] {
                let receipt = Arc::new(SeedReceipt::default());
                let options = EncodeOptions {
                    threads: 2,
                    modular: ModularOptions {
                        predictor: Predictor::Variable,
                        source_ma_seed: Some(SeedOptions {
                            mode,
                            receipt: Some(receipt.clone()),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let encoded = crate::encode(&image, &options).unwrap();
                let decoded = decode(&encoded, &DecodeOptions::default()).unwrap();
                assert_eq!(
                    decoded.frames[0].pixels,
                    *image.pixels(),
                    "{mode:?} {bits} {layout:?}"
                );
                assert!(receipt.snapshot().learners > 0);
            }
        }
    }
}
