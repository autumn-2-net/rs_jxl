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
