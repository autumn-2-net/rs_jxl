use super::super::{Bias, PredictorSet, Receipt, Scan};
use super::*;
use crate::modular::{encode::ModularOptions, group::Channel, training, weighted};
use crate::{EncodeOptions, Image, PixelData, PixelLayout};
use std::sync::Arc;

#[test]
fn uniform_populations_collapse_and_joint_populations_do_not() {
    for thresholds in [COARSE, FINE] {
        let n = thresholds.len() + 1;
        for joint in [false, true] {
            let cells = (0..n * n)
                .map(|i| {
                    std::array::from_fn(|_| {
                        let mut hist = Histogram::default();
                        let residual = if joint && ((i / n >= n / 2) != (i % n >= n / 2)) {
                            7
                        } else {
                            0
                        };
                        for _ in 0..512 {
                            hist.add(residual, false);
                        }
                        hist
                    })
                })
                .collect();
            let mut grid = Grid::new(cells, n);
            let bits = grid.solve(0, 0);
            assert_eq!(grid.state(0, 0).hist[0].count, (n * n * 512) as u64);
            assert!(bits <= grid.state(0, 0).leaf.bits);
            let mut nodes = Vec::new();
            grid.append(&mut nodes, thresholds, 0, 0);
            let tree = Tree::new(nodes).unwrap();
            if !joint {
                assert_eq!(tree.leaves(), 1);
            } else {
                assert!(tree.leaves() > 1);
                assert!(
                    tree.nodes()
                        .iter()
                        .any(|n| matches!(n, Node::Split { property: 10, .. }))
                );
                assert!(
                    tree.nodes()
                        .iter()
                        .any(|n| matches!(n, Node::Split { property: 11, .. }))
                );
            }
        }
    }
}

#[test]
fn thresholds_and_tree_lookup_have_identical_boundary_semantics() {
    for thresholds in [COARSE, FINE] {
        let n = thresholds.len() + 1;
        let cells = (0..n * n)
            .map(|i| {
                std::array::from_fn(|p| {
                    let mut hist = Histogram::default();
                    for j in 0..2048 {
                        hist.add(if p == i % 4 { 0 } else { j }, false);
                    }
                    hist
                })
            })
            .collect();
        let mut grid = Grid::new(cells, n);
        grid.solve(0, 0);
        let mut nodes = Vec::new();
        grid.append(&mut nodes, thresholds, 0, 0);
        let tree = Tree::new(nodes).unwrap();
        let values: Vec<_> = [i32::MIN, i32::MAX]
            .into_iter()
            .chain(thresholds.iter().flat_map(|&v| [v - 1, v, v + 1]))
            .collect();
        for &v in &values {
            for &h in &values {
                let mut props = [0; 12];
                props[10] = v;
                props[11] = h;
                let (_, actual, offset, multiplier) = tree.lookup(&props).unwrap();
                let vi = thresholds.partition_point(|&t| v > t);
                let hi = thresholds.partition_point(|&t| h > t);
                assert_eq!(actual, PREDICTORS[(vi * n + hi) % 4]);
                assert_eq!((offset, multiplier), (0, 1));
            }
        }
    }
}

#[test]
fn integer_depths_hidden_rgb_and_narrow_edges_roundtrip() {
    for mode in [Mode::Activity25, Mode::Activity81] {
        for (width, height) in [(1, 7), (17, 9), (35, 1)] {
            for bits in [8, 16, 31] {
                let words: Vec<_> = (0..width * height * 4)
                    .map(|i| {
                        if i % 4 == 3 {
                            if i % 12 == 3 { 0 } else { (1u32 << bits) - 1 }
                        } else {
                            ((i as u32).wrapping_mul(0x9e37_79b9) ^ (i / 7) as u32)
                                & ((1u32 << bits) - 1)
                        }
                    })
                    .collect();
                let data = match bits {
                    8 => PixelData::U8(words.iter().map(|&v| v as u8).collect()),
                    16 => PixelData::U16(words.iter().map(|&v| v as u16).collect()),
                    _ => PixelData::U32(words),
                };
                let image =
                    Image::new(width as u32, height as u32, PixelLayout::Rgba, data).unwrap();
                let receipt = Arc::new(Receipt::default());
                let options = EncodeOptions {
                    threads: 2,
                    modular: ModularOptions {
                        fast_context: Some(Options {
                            mode,
                            receipt: Some(receipt.clone()),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let bytes = crate::encode(&image, &options).unwrap();
                let decoded = crate::decode::decode(&bytes, &Default::default()).unwrap();
                assert_eq!(
                    decoded.frames[0].pixels,
                    *image.pixels(),
                    "{mode:?} {width}x{height} {bits}"
                );
                assert!(receipt.snapshot().trees > 0);
            }
        }
    }
}

#[test]
fn stride_empty_views_and_simd_keep_the_same_statistics() {
    let data: Vec<i32> = (0..41 * 5)
        .map(|i| (i as i32).wrapping_mul(0x7163_22f9))
        .collect();
    let view = ChannelView {
        channel: 0,
        group: 19,
        width: 35,
        height: 5,
        stride: 41,
        pixels: &data,
        is_meta: false,
    };
    let empty = ChannelView {
        channel: 0,
        group: 3,
        width: 0,
        height: 0,
        stride: 0,
        pixels: &[],
        is_meta: true,
    };
    let scalar = scan(
        &[&view, &empty],
        FINE,
        &Options {
            mode: Mode::Activity81,
            scan: Scan::Scalar,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(scalar.hist[0].count, 175);
    for backend in [Scan::Auto, Scan::Avx2, Scan::Avx512] {
        if !backend.backend().available() {
            continue;
        }
        let plan = scan(
            &[&view, &empty],
            FINE,
            &Options {
                mode: Mode::Activity81,
                scan: backend,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(plan.hist, scalar.hist);
        assert_eq!(plan.bits, scalar.bits);
        assert_eq!(format!("{:?}", plan.nodes), format!("{:?}", scalar.nodes));
    }
}

#[test]
fn empty_height_never_allocates_width_sized_scratch() {
    let empty = ChannelView {
        channel: 0,
        group: 19,
        width: usize::MAX,
        height: 0,
        stride: usize::MAX,
        pixels: &[],
        is_meta: false,
    };
    for mode in [Mode::Activity25, Mode::Activity81] {
        let tree = learn(
            &[empty],
            1,
            1,
            false,
            &Options {
                mode,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(tree.leaves(), 1);
    }
}

#[test]
fn explicit_budgets_and_conflicting_options_are_not_ignored() {
    let channels: Vec<_> = (0..3)
        .map(|c| Channel {
            width: 33,
            height: 9,
            hshift: 0,
            vshift: 0,
            samples: (0..297).map(|i| (i * 71 + c * 31) % 255).collect(),
        })
        .collect();
    let streams = [training::TrainingStream {
        channels: &channels,
        wp: weighted::Header::default(),
        stream_id: 19,
    }];
    for mode in [Mode::Activity25, Mode::Activity81] {
        for limit in [1, 2, 7, 256] {
            let mut options = ModularOptions {
                fast_context: Some(Options {
                    mode,
                    ..Default::default()
                }),
                ma_max_leaves: Some(limit),
                ..Default::default()
            };
            let tree = training::learn(&streams, &options).unwrap();
            assert!(tree.leaves() <= limit);
            options.sample_limit = Some(33 * 9 * 3 - 1);
            assert!(training::learn(&streams, &options).is_err());
        }
        let views = [ChannelView {
            channel: 0,
            group: 19,
            width: 33,
            height: 9,
            stride: 33,
            pixels: &channels[0].samples,
            is_meta: false,
        }];
        assert!(
            learn(
                &views,
                3,
                1,
                true,
                &Options {
                    mode,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            Options {
                mode,
                bias: Bias::Static,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Options {
                mode,
                predictors: PredictorSet::Full,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Options {
                mode,
                cross_channel: true,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
