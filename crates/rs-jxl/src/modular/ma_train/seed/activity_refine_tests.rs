use super::*;
use crate::modular::{
    ma_train::{self, MaBudgetMode, Sample},
    predictor::Predictor,
};

fn options(topology: SeedTopology, mode: SeedMode) -> TrainOptions {
    TrainOptions {
        predictors: (0..14).map(|id| Predictor::try_from(id).unwrap()).collect(),
        properties: vec![3, 10, 11],
        split_threshold: 0.0,
        parallel: false,
        seed: Some(SeedOptions {
            topology,
            mode,
            depth: 1,
            receipt: Some(Arc::new(SeedReceipt::default())),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn sample(predictor: usize, x: i32, v: i32, h: i32) -> Sample {
    let mut properties = vec![0; 16];
    properties[3] = x;
    properties[10] = v;
    properties[11] = h;
    let mut residuals = vec![4096; 14];
    residuals[predictor] = 0;
    Sample {
        properties,
        residuals,
        weight: 1000,
    }
}

#[test]
fn activity_initialization_does_not_restrict_any_predictor_or_final_depth() {
    let samples: Vec<_> = (0..14).map(|p| sample(p, p as i32, 0, 0)).collect();
    for topology in [SeedTopology::Activity25, SeedTopology::Activity81] {
        let initial = ma_train::train(&samples, &options(topology, SeedMode::Only)).unwrap();
        assert_eq!(initial.leaves(), 1);
        for mode in [SeedMode::Refine, SeedMode::Repair] {
            let opts = options(topology, mode);
            let tree = ma_train::train(&samples, &opts).unwrap();
            assert_eq!(tree.leaves(), 14);
            for (p, sample) in samples.iter().enumerate() {
                assert_eq!(
                    tree.lookup(&sample.properties).unwrap().1,
                    Predictor::try_from(p as u32).unwrap()
                );
            }
            assert!(
                tree.nodes()
                    .iter()
                    .any(|n| matches!(n, Node::Split { property: 3, .. }))
            );
            assert!(
                opts.seed
                    .as_ref()
                    .unwrap()
                    .receipt
                    .as_ref()
                    .unwrap()
                    .snapshot()
                    .refined_leaves
                    > 0
            );
        }
    }
}

#[test]
fn joint_seed_grafting_refines_the_original_record_domains() {
    let samples: Vec<_> = (0..8)
        .map(|p| {
            sample(
                p,
                (p % 2) as i32,
                if p & 2 == 0 { -8 } else { 8 },
                if p & 4 == 0 { -8 } else { 8 },
            )
        })
        .collect();
    for topology in [SeedTopology::Activity25, SeedTopology::Activity81] {
        for mode in [SeedMode::Refine, SeedMode::Repair] {
            let opts = options(topology, mode);
            let tree = ma_train::train(&samples, &opts).unwrap();
            for (p, sample) in samples.iter().enumerate() {
                assert_eq!(
                    tree.lookup(&sample.properties).unwrap().1,
                    Predictor::try_from(p as u32).unwrap()
                );
            }
            let snapshot = opts
                .seed
                .as_ref()
                .unwrap()
                .receipt
                .as_ref()
                .unwrap()
                .snapshot();
            assert!(snapshot.seed_splits > 0);
            assert!(snapshot.refined_leaves > 0);
            assert!(snapshot.histogram_reuses > 0);
        }
    }
}

#[test]
fn activity_seed_respects_explicit_budgets_and_empty_populations() {
    let samples = [sample(6, 0, -8, -8), sample(13, 1, 8, 8)];
    for topology in [SeedTopology::Activity25, SeedTopology::Activity81] {
        let mut opts = options(topology, SeedMode::Only);
        assert_eq!(ma_train::train(&[], &opts).unwrap().leaves(), 1);
        opts.max_leaves = 1;
        opts.budget_mode = Some(MaBudgetMode::Bounded);
        assert_eq!(ma_train::train(&samples, &opts).unwrap().leaves(), 1);
        opts.budget_mode = Some(MaBudgetMode::Strict);
        assert!(ma_train::train(&samples, &opts).is_err());
    }
}

#[test]
fn activity_refinement_roundtrips_integer_depths_and_hidden_rgb() {
    use crate::modular::encode::ModularOptions;
    use crate::{EncodeOptions, Image, PixelData, PixelLayout};
    for topology in [SeedTopology::Activity25, SeedTopology::Activity81] {
        for mode in [SeedMode::Refine, SeedMode::Repair] {
            for bits in [8, 16, 31] {
                let words: Vec<_> = (0..35 * 9 * 4)
                    .map(|i| {
                        if i % 4 == 3 {
                            0
                        } else {
                            (i as u32).wrapping_mul(0x9e37_79b9) & ((1u32 << bits) - 1)
                        }
                    })
                    .collect();
                let pixels = match bits {
                    8 => PixelData::U8(words.iter().map(|&x| x as u8).collect()),
                    16 => PixelData::U16(words.iter().map(|&x| x as u16).collect()),
                    _ => PixelData::U32(words),
                };
                let image = Image::new(35, 9, PixelLayout::Rgba, pixels).unwrap();
                let opts = EncodeOptions {
                    threads: 2,
                    modular: ModularOptions {
                        predictor: Predictor::Variable,
                        source_ma_seed: Some(SeedOptions {
                            topology,
                            mode,
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let encoded = crate::encode(&image, &opts).unwrap();
                let decoded = crate::decode::decode(&encoded, &Default::default()).unwrap();
                assert_eq!(
                    decoded.frames[0].pixels,
                    *image.pixels(),
                    "{topology:?} {mode:?} {bits}"
                );
            }
        }
    }
}

#[test]
fn activity_regression_nonunit_multiplier_skeleton_survives_grafting() {
    use ma_train::{ConstantChannel, MultiplierInfo};

    for topology in [SeedTopology::Activity25, SeedTopology::Activity81] {
        for mode in [SeedMode::Only, SeedMode::Refine, SeedMode::Repair] {
            let mut opts = options(topology, mode);
            opts.properties = vec![0, 1, 3, 10, 11];
            opts.static_property_range = Some([[0, 3], [0, 1]]);
            opts.max_leaves = 5;
            opts.budget_mode = Some(MaBudgetMode::Strict);
            opts.multiplier_info = vec![
                MultiplierInfo {
                    range: [[0, 1], [0, 1]],
                    multiplier: 3,
                },
                MultiplierInfo {
                    range: [[1, 3], [0, 1]],
                    multiplier: 5,
                },
            ];
            opts.constant_channels = vec![ConstantChannel {
                channel: 2,
                value: -11,
            }];
            let mut samples = Vec::new();
            for channel in 0..2 {
                let multiplier = if channel == 0 { 3 } else { 5 };
                for v in [-8, 8] {
                    let mut s = sample(if v < 0 { 6 } else { 13 }, 0, v, 0);
                    s.properties[0] = channel;
                    for residual in &mut s.residuals {
                        *residual *= multiplier;
                    }
                    samples.push(s);
                }
            }
            let tree = ma_train::train(&samples, &opts).unwrap();
            assert_eq!(tree.leaves(), 5, "{topology:?} {mode:?}");
            for s in &samples {
                let (_, predictor, offset, multiplier) = tree.lookup(&s.properties).unwrap();
                assert_eq!(
                    predictor,
                    if s.properties[10] < 0 {
                        Predictor::Weighted
                    } else {
                        Predictor::Average4
                    }
                );
                assert_eq!(offset, 0);
                assert_eq!(multiplier, if s.properties[0] == 0 { 3 } else { 5 });
            }
            let (_, predictor, offset, multiplier) = tree.lookup(&[2, 0]).unwrap();
            assert_eq!((predictor, offset, multiplier), (Predictor::Zero, -11, 1));
            let snapshot = opts
                .seed
                .as_ref()
                .unwrap()
                .receipt
                .as_ref()
                .unwrap()
                .snapshot();
            assert_eq!(snapshot.seed_splits, 2);
            if mode != SeedMode::Only {
                assert!(snapshot.refined_leaves >= 4);
            }
        }
    }
}

#[test]
fn activity_regression_partial_budget_collapse_releases_slots_for_later_roots() {
    use ma_train::MultiplierInfo;

    // The greater-channel root is attached first. Its XOR needs both levels:
    // truncating after the first split leaves identical child distributions.
    // Collapsing it must release the slot needed by channel zero's binary seed.
    let mut samples = Vec::new();
    for channel in 0..2 {
        for v in [-8, 8] {
            for h in [-8, 8] {
                let winner = if channel == 0 {
                    v < 0
                } else {
                    (v < 0) == (h < 0)
                };
                let mut s = sample(if winner { 6 } else { 13 }, 0, v, h);
                s.properties[0] = channel;
                for residual in &mut s.residuals {
                    *residual *= if channel == 0 { 3 } else { 5 };
                }
                samples.push(s);
            }
        }
    }
    for topology in [SeedTopology::Activity25, SeedTopology::Activity81] {
        for mode in [SeedMode::Only, SeedMode::Refine, SeedMode::Repair] {
            let mut opts = options(topology, mode);
            opts.properties = vec![0, 1, 3, 10, 11];
            opts.static_property_range = Some([[0, 2], [0, 1]]);
            opts.max_leaves = 3;
            opts.budget_mode = Some(MaBudgetMode::Bounded);
            opts.multiplier_info = vec![
                MultiplierInfo {
                    range: [[0, 1], [0, 1]],
                    multiplier: 3,
                },
                MultiplierInfo {
                    range: [[1, 2], [0, 1]],
                    multiplier: 5,
                },
            ];
            let tree = ma_train::train(&samples, &opts).unwrap();
            assert_eq!(tree.leaves(), 3, "{topology:?} {mode:?}");
            for s in &samples {
                let (_, predictor, offset, multiplier) = tree.lookup(&s.properties).unwrap();
                assert_eq!(offset, 0);
                assert_eq!(multiplier, if s.properties[0] == 0 { 3 } else { 5 });
                if s.properties[0] == 0 {
                    assert_eq!(
                        predictor,
                        if s.properties[10] < 0 {
                            Predictor::Weighted
                        } else {
                            Predictor::Average4
                        }
                    );
                } else {
                    assert_eq!(predictor, Predictor::Weighted);
                }
            }
            let snapshot = opts
                .seed
                .as_ref()
                .unwrap()
                .receipt
                .as_ref()
                .unwrap()
                .snapshot();
            assert_eq!(snapshot.seed_splits, 2);
            assert_eq!(snapshot.merged_branches, 1);

            opts.budget_mode = Some(MaBudgetMode::Strict);
            assert!(matches!(
                ma_train::train(&samples, &opts),
                Err(Error::Limit(_))
            ));
            opts.max_leaves = 6;
            let full = ma_train::train(&samples, &opts).unwrap();
            assert_eq!(full.leaves(), 6);
            for s in &samples {
                let winner = if s.properties[0] == 0 {
                    s.properties[10] < 0
                } else {
                    (s.properties[10] < 0) == (s.properties[11] < 0)
                };
                assert_eq!(
                    full.lookup(&s.properties).unwrap().1,
                    if winner {
                        Predictor::Weighted
                    } else {
                        Predictor::Average4
                    }
                );
            }
        }
    }
}
