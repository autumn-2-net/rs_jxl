use super::*;
use crate::modular::{encode, group::Channel, predictor::Predictor, weighted};

fn leaf(offset: i32, multiplier: u32) -> Node {
    Node::Leaf {
        predictor: Predictor::Zero,
        offset,
        multiplier,
    }
}

fn plane(values: Vec<i32>) -> Channel {
    Channel {
        width: values.len(),
        height: 1,
        hshift: 0,
        vshift: 0,
        samples: values,
    }
}

fn stream(channels: &[Channel]) -> TrainingStream<'_> {
    TrainingStream {
        channels,
        wp: weighted::Header::default(),
        stream_id: 0,
    }
}

#[test]
fn mean_is_full_population_and_reservoir_is_bounded_and_deterministic() {
    let mut a = Samples::new(7);
    let mut b = Samples::new(7);
    for i in 0..10_000 {
        a.add(i, 32).unwrap();
        b.add(i, 32).unwrap();
    }
    assert_eq!(a.mean(), 5000);
    assert_eq!(a.count, 10_000);
    assert_eq!(a.values.len(), 32);
    assert_eq!(a.values, b.values);
    assert!(a.values.iter().any(|&x| x > 9000));
    for value in [i32::MIN, -1, 0, 1, i32::MAX] {
        let mut s = Samples::new(0);
        for _ in 0..3 {
            s.add(i64::from(value), 1).unwrap();
        }
        assert_eq!(s.mean(), value);
    }
    let mut s = Samples::new(0);
    s.add(-1, 2).unwrap();
    s.add(0, 2).unwrap();
    assert_eq!(s.mean(), -1);
}

#[test]
fn unit_residual_wrapping_matches_wide_encoder_arithmetic() {
    for pixel in [i32::MIN, -1, 0, i32::MAX] {
        for prediction in [i32::MIN as i64 * 2, 0, i32::MAX as i64 * 2] {
            for offset in [i32::MIN, -33, 0, 33, i32::MAX] {
                let raw = (i64::from(pixel) - prediction) as i32;
                assert_eq!(
                    raw.wrapping_sub(offset),
                    (i64::from(pixel) - prediction - i64::from(offset)) as i32
                );
            }
        }
    }
}

#[test]
fn proposal_statistics_keep_wide_residuals_until_after_averaging() {
    let mut samples = Samples::new(0);
    samples.add(2147483648, 2).unwrap();
    samples.add(2147483646, 2).unwrap();
    assert_eq!(samples.mean(), i32::MAX);
    assert!(samples.bits(i32::MAX) < samples.bits(0));
}

#[test]
fn zero_residual_leaf_is_preserved_when_offset_labels_could_merge() {
    let tree = Tree::new(vec![
        Node::Split {
            property: 0,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        Node::Split {
            property: 0,
            threshold: 1,
            greater: 3,
            less_equal: 4,
        },
        leaf(0, 1),
        leaf(1, 1),
        leaf(1, 1),
    ])
    .unwrap();
    let channels = [
        plane(vec![0; 100]),
        plane(vec![1; 100]),
        plane(vec![1; 100]),
    ];
    assert!(
        fit(&tree, &[stream(&channels)], 0, &OffsetOptions::default())
            .unwrap()
            .0
            .is_none()
    );
}

#[test]
fn sequential_offset_costs_telescope_to_shared_histogram_change() {
    let mut counts = [0; TOKENS];
    counts[0] = 5;
    counts[2] = 3;
    let cost = |c: &[u64; TOKENS]| 8.0 * 8.0f64.log2() - c.iter().map(|&n| xlogx(n)).sum::<f64>();
    let before = cost(&counts);
    let mut delta = 0.0;
    for _ in 0..4 {
        delta += offset_cost(&counts, 1);
        counts[0] -= 1;
        counts[2] += 1;
    }
    assert!((delta - (cost(&counts) - before)).abs() < 1e-10);
}

#[test]
fn token_cost_not_absolute_error_decides_offsets() {
    let mut s = Samples::new(0);
    for value in [1, 2, 1, 2] {
        s.add(value, 4).unwrap();
    }
    // Signed-token relabeling has identical entropy and zero extra bits.
    assert_eq!(s.bits(0), s.bits(1));
    assert_eq!(s.bits(0), s.bits(2));
    let channels = [plane(vec![1, 2].repeat(256))];
    let tree = Tree::new(vec![leaf(0, 1)]).unwrap();
    let (proposal, _) = fit(&tree, &[stream(&channels)], 0, &OffsetOptions::default()).unwrap();
    assert!(proposal.is_none());
}

#[test]
fn biased_residuals_propose_offsets_without_changing_predictors() {
    let tree = Tree::new(vec![leaf(0, 1)]).unwrap();
    let channels = [plane((0..1024).map(|i| 100 + i % 3).collect())];
    for mode in [OffsetMode::Mean, OffsetMode::Median, OffsetMode::Entropy] {
        let (proposal, stats) = fit(
            &tree,
            &[stream(&channels)],
            0,
            &OffsetOptions {
                mode,
                samples_per_leaf: 128,
            },
        )
        .unwrap();
        let proposal = proposal.unwrap();
        assert!(matches!(
            proposal.nodes()[0],
            Node::Leaf {
                predictor: Predictor::Zero,
                offset: 98..=103,
                multiplier: 1
            }
        ));
        assert_eq!(stats.leaves, 1);
        assert_eq!(stats.population, 1024);
        assert_eq!(stats.samples, 128);
    }
}

#[test]
fn preserves_existing_offsets_multipliers_and_zero_constants() {
    for (offset, multiplier, value) in [(73, 1, 73), (0, 2, 100), (0, 3, 102), (0, 1, 0)] {
        let tree = Tree::new(vec![leaf(offset, multiplier)]).unwrap();
        let channels = [plane(vec![value; 100])];
        let (proposal, _) = fit(&tree, &[stream(&channels)], 0, &OffsetOptions::default()).unwrap();
        assert!(proposal.is_none());
    }
}

#[test]
fn distribution_refits_existing_bias_and_can_return_to_zero() {
    for (before, value) in [(17, 1000), (-137, 0), (i32::MAX, i32::MIN + 4096)] {
        let tree = Tree::new(vec![leaf(before, 1)]).unwrap();
        let channels = [plane(vec![value; 2048])];
        let (proposal, stats) = fit(
            &tree,
            &[stream(&channels)],
            0,
            &OffsetOptions {
                mode: OffsetMode::Distribution,
                ..Default::default()
            },
        )
        .unwrap();
        let proposal = proposal.unwrap();
        assert!(
            matches!(proposal.nodes()[0], Node::Leaf { offset, multiplier: 1, .. } if offset == value)
        );
        assert_eq!(stats.leaves, 1);
        assert_eq!(stats.population, 2048);
    }
}

#[test]
fn distribution_preserves_perfect_nonzero_bias_and_nonunit_multipliers() {
    for (offset, multiplier, value) in [(73, 1, 73), (i32::MIN, 1, i32::MIN), (0, 2, 100)] {
        let tree = Tree::new(vec![leaf(offset, multiplier)]).unwrap();
        let channels = [plane(vec![value; 128])];
        let (proposal, stats) = fit(
            &tree,
            &[stream(&channels)],
            0,
            &OffsetOptions {
                mode: OffsetMode::Distribution,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(proposal.is_none());
        assert_eq!(stats.leaves, 0);
    }
}

#[test]
fn distribution_keeps_nonzero_bias_when_wrapped_residual_cost_ties() {
    let tree = Tree::new(vec![leaf(i32::MAX, 1)]).unwrap();
    let channels = [plane(vec![i32::MIN; 2048])];
    // The wrapped residual is 1, not a huge signed distance. Both offsets have
    // zero residual extra bits here, so moving to MIN is not a strict gain.
    let (proposal, _) = fit(
        &tree,
        &[stream(&channels)],
        0,
        &OffsetOptions {
            mode: OffsetMode::Distribution,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(proposal.is_none());
}

#[test]
fn fast_context_offset_fitting_is_explicit_but_custom_trees_remain_exact() {
    use crate::modular::fast_context::{Bias, Options};
    for bias in [Bias::Off, Bias::Static, Bias::Adaptive] {
        let mut options = encode::ModularOptions::default();
        options.fast_context = Some(Options {
            bias,
            ..Default::default()
        });
        options.refinement.offsets = Some(OffsetOptions {
            mode: OffsetMode::Distribution,
            ..Default::default()
        });
        assert!(options.validate().is_ok());
        options.custom_tree = Some(Tree::new(vec![leaf(73, 1)]).unwrap());
        assert!(options.validate().is_err());
    }
}

#[test]
fn bfs_contexts_are_not_node_indices() {
    let tree = Tree::new(vec![
        Node::Split {
            property: 0,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        Node::Split {
            property: 0,
            threshold: 1,
            greater: 3,
            less_equal: 4,
        },
        leaf(0, 1),
        leaf(0, 1),
        leaf(0, 1),
    ])
    .unwrap();
    let channels: Vec<_> = [100, 200, 300]
        .into_iter()
        .map(|v| plane(vec![v; 512]))
        .collect();
    let (proposal, _) = fit(
        &tree,
        &[stream(&channels)],
        0,
        &OffsetOptions {
            mode: OffsetMode::Mean,
            ..Default::default()
        },
    )
    .unwrap();
    let proposal = proposal.unwrap();
    for (node, expected) in [(2, 100), (3, 300), (4, 200)] {
        assert!(matches!(proposal.nodes()[node], Node::Leaf { offset, .. } if offset == expected));
    }
}

#[test]
fn validates_inputs_and_explicit_tree_conflicts() {
    let tree = Tree::new(vec![leaf(0, 1)]).unwrap();
    let channels = [plane(vec![100; 100])];
    let view = stream(&channels);
    assert!(
        fit(
            &tree,
            &[view, stream(&channels)],
            0,
            &OffsetOptions::default()
        )
        .is_err()
    );
    assert!(fit(&tree, &[], 61, &OffsetOptions::default()).is_err());
    assert!(
        fit(
            &tree,
            &[],
            0,
            &OffsetOptions {
                samples_per_leaf: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    let mut broken = plane(vec![100; 100]);
    broken.width = 99;
    assert!(fit(&tree, &[stream(&[broken])], 0, &OffsetOptions::default()).is_err());
    let mut options = encode::ModularOptions::default();
    options.refinement.offsets = Some(OffsetOptions::default());
    options.custom_tree = Some(tree);
    assert!(options.validate().is_err());
}

#[test]
fn weighted_traversal_is_deterministic_including_wide_residuals() {
    let tree = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Weighted,
        offset: 0,
        multiplier: 1,
    }])
    .unwrap();
    let channels = [plane(vec![i32::MIN, i32::MAX, 100, -300].repeat(128))];
    let run = || fit(&tree, &[stream(&channels)], 0, &OffsetOptions::default()).unwrap();
    let (a, sa) = run();
    let (b, sb) = run();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(sa.samples, sb.samples);
    assert_eq!(sa.population, 512);
}

#[test]
fn replacement_cost_tracks_nonzero_offsets_and_shared_token_extra_bits() {
    let mut offsets = [0, 0, 17, 17, -9, i32::MAX, i32::MIN];
    let mut counts = [0; TOKENS];
    for offset in offsets {
        counts[UINT.encode(pack_signed(offset)).token as usize] += 1;
    }
    let cost = |offsets: &[i32], counts: &[u64; TOKENS]| {
        xlogx(offsets.len() as u64) - counts.iter().map(|&n| xlogx(n)).sum::<f64>()
            + offsets
                .iter()
                .map(|&v| f64::from(UINT.encode(pack_signed(v)).nbits))
                .sum::<f64>()
    };
    for (index, new) in [(0, 17), (1, -9), (2, 0), (3, 18), (5, i32::MIN), (6, 0)] {
        let old = offsets[index];
        let before = cost(&offsets, &counts);
        let delta = replacement_cost(&counts, old, new);
        counts[UINT.encode(pack_signed(old)).token as usize] -= 1;
        counts[UINT.encode(pack_signed(new)).token as usize] += 1;
        offsets[index] = new;
        assert!((cost(&offsets, &counts) - before - delta).abs() < 1e-10);
    }
}

#[test]
fn distribution_keeps_constants_and_nonunit_leaves_and_checks_configs() {
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        ..Default::default()
    };
    for (offset, multiplier, value) in [(73, 1, 73), (0, 2, 100), (0, 3, 102), (0, 1, 0)] {
        let tree = Tree::new(vec![leaf(offset, multiplier)]).unwrap();
        let channels = [plane(vec![value; 100])];
        assert!(
            fit(&tree, &[stream(&channels)], 0, &options)
                .unwrap()
                .0
                .is_none()
        );
        assert!(fit_with_configs(&tree, &[stream(&channels)], 0, &options, Some(&[])).is_err());
    }
}

#[test]
fn distribution_uses_each_bfs_context_configuration_deterministically() {
    let tree = Tree::new(vec![
        Node::Split {
            property: 0,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        leaf(0, 1),
        leaf(0, 1),
    ])
    .unwrap();
    let channels = [
        plane(vec![100, 101, 102, 499].repeat(256)),
        plane(vec![i32::MIN, i32::MAX, 73, 74].repeat(256)),
    ];
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        samples_per_leaf: 32,
    };
    let configs = [
        HybridUintConfig::new(0, 0, 0).unwrap(),
        HybridUintConfig::new(15, 0, 0).unwrap(),
    ];
    let run =
        || fit_with_configs(&tree, &[stream(&channels)], 0, &options, Some(&configs)).unwrap();
    let (a, sa) = run();
    let (b, sb) = run();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(sa.population, 2048);
    assert_eq!(sa.samples, 64);
    assert_eq!(sa.candidates, sb.candidates);
}

#[test]
fn distribution_configuration_routes_to_bfs_context_not_node_index() {
    let tree = Tree::new(vec![
        Node::Split {
            property: 0,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        Node::Split {
            property: 0,
            threshold: 1,
            greater: 3,
            less_equal: 4,
        },
        leaf(0, 1),
        leaf(0, 1),
        leaf(0, 1),
    ])
    .unwrap();
    let channels =
        [100, 200, 300].map(|center| plane(vec![center, center + 1, center + 2].repeat(128)));
    let direct = HybridUintConfig::new(15, 0, 0).unwrap();
    let compact = HybridUintConfig::new(0, 0, 0).unwrap();
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        ..Default::default()
    };
    // BFS contexts are node 2 / node 3 / node 4, belonging to channels 0 / 2 / 1.
    // Direct symbols cannot benefit from merely translating these small values;
    // the compact mapping pays extra bits until its biased leaf is centered.
    for (configs, selected_node, center) in [
        ([direct, compact, direct], 3, 300),
        ([compact, direct, direct], 2, 100),
    ] {
        let (proposal, stats) =
            fit_with_configs(&tree, &[stream(&channels)], 0, &options, Some(&configs)).unwrap();
        let proposal = proposal.expect("compact context has a useful bias");
        assert_eq!(stats.leaves, 1);
        for node in 2..=4 {
            let Node::Leaf { offset, .. } = proposal.nodes()[node] else {
                unreachable!()
            };
            if node == selected_node {
                assert!((center - 2..=center + 3).contains(&offset));
            } else {
                assert_eq!(offset, 0);
            }
        }
    }
}
