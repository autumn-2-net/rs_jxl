use super::*;
use crate::modular::group::Channel;
use crate::modular::predictor::{Neighbors, clamped_gradient};
use crate::modular::tree::Properties;
use crate::modular::weighted;

fn plane(width: usize, height: usize, sample: impl Fn(usize, usize) -> i32) -> Channel {
    let mut samples = Vec::new();
    for y in 0..height {
        for x in 0..width {
            samples.push(sample(x, y));
        }
    }
    Channel {
        width,
        height,
        hshift: 0,
        vshift: 0,
        samples,
    }
}

fn leaf(predictor: Predictor) -> Node {
    Node::Leaf {
        predictor,
        offset: 0,
        multiplier: 1,
    }
}

fn stream(channels: &[Channel]) -> TrainingStream<'_> {
    TrainingStream {
        channels,
        wp: weighted::Header::default(),
        stream_id: 0,
    }
}

fn single(predictor: Predictor) -> Tree {
    Tree::new(vec![leaf(predictor)]).unwrap()
}

// Independent scalar raster oracle: deliberately no encode::visit, row helper,
// eligibility-gated WP updates, stream sorting or production histogram adds.
fn assert_raster_histograms(
    tree: &Tree,
    streams: &[TrainingStream<'_>],
    allowed: &[Predictor],
    previous_channels: usize,
) -> Vec<Option<LeafStats>> {
    let actual = collect_stats(tree, streams, allowed, previous_channels).unwrap();
    let mut expected: Vec<Vec<Histogram>> = actual
        .iter()
        .map(|leaf| {
            leaf.as_ref().map_or_else(Vec::new, |leaf| {
                leaf.candidates.iter().map(|_| Histogram::new()).collect()
            })
        })
        .collect();
    for stream in streams {
        for (c, channel) in stream.channels.iter().enumerate() {
            if channel.width == 0 || channel.height == 0 {
                continue;
            }
            let mut state = weighted::State::new(stream.wp, channel.width).unwrap();
            let mut property_state = Properties::default();
            for y in 0..channel.height {
                for x in 0..channel.width {
                    let pos = y * channel.width + x;
                    let value = channel.samples[pos];
                    let n =
                        Neighbors::from_plane(&channel.samples, channel.width, channel.width, x, y)
                            .unwrap();
                    let (wp, error) = state
                        .predict_and_update(
                            value, x, y, n.top, n.left, n.topright, n.topleft, n.toptop,
                        )
                        .unwrap();
                    let mut props = vec![0; 16 + 4 * previous_channels];
                    props[..16].copy_from_slice(&property_state.calculate(
                        n,
                        x,
                        y,
                        c,
                        stream.stream_id,
                        error,
                    ));
                    let mut slot = 0;
                    for reference in stream.channels[..c].iter().rev() {
                        if slot == previous_channels {
                            break;
                        }
                        if reference.width != channel.width
                            || reference.height != channel.height
                            || reference.hshift != channel.hshift
                            || reference.vshift != channel.vshift
                        {
                            continue;
                        }
                        let v = reference.samples[pos];
                        let left = if x == 0 {
                            0
                        } else {
                            reference.samples[pos - 1]
                        };
                        let top = if y == 0 {
                            left
                        } else {
                            reference.samples[pos - channel.width]
                        };
                        let nw = if x == 0 || y == 0 {
                            left
                        } else {
                            reference.samples[pos - channel.width - 1]
                        };
                        let residual = v.wrapping_sub(clamped_gradient(left, top, nw));
                        props[16 + 4 * slot..20 + 4 * slot].copy_from_slice(&[
                            v.wrapping_abs(),
                            v,
                            residual.wrapping_abs(),
                            residual,
                        ]);
                        slot += 1;
                    }
                    let (context, _, _, _) = tree.lookup(&props).unwrap();
                    if let Some(leaf) = &actual[context] {
                        for (p, candidate) in leaf.candidates.iter().enumerate() {
                            let prediction =
                                predict_one(candidate.predictor, n, Some(i64::from(wp))).unwrap();
                            let residual = (i64::from(value) - prediction) as i32;
                            let encoded = UINT.encode(pack_signed(residual));
                            let histogram = &mut expected[context][p];
                            histogram.counts[encoded.token as usize] += 1;
                            histogram.count += 1;
                            histogram.extra += u64::from(encoded.nbits);
                        }
                    }
                }
            }
        }
    }
    for (context, leaf) in actual.iter().enumerate() {
        if let Some(leaf) = leaf {
            for (p, candidate) in leaf.candidates.iter().enumerate() {
                let expected = &expected[context][p];
                assert_eq!(candidate.histogram.count, expected.count);
                assert_eq!(candidate.histogram.extra, expected.extra);
                assert_eq!(candidate.histogram.counts, expected.counts);
            }
        }
    }
    actual
}

#[test]
fn no_op_for_empty_domain_no_alternatives_and_exact_ties() {
    let channels = [plane(9, 5, |_, _| 0)];
    let tree = single(Predictor::Left);
    let allowed = [
        Predictor::Weighted,
        Predictor::Gradient,
        Predictor::Weighted,
    ];
    assert!(refit(&tree, &[], &allowed, 0).unwrap().is_none());
    assert!(
        refit(&tree, &[stream(&channels)], &[], 0)
            .unwrap()
            .is_none()
    );
    assert!(
        refit(&tree, &[stream(&channels)], &[Predictor::Top], 0)
            .unwrap()
            .is_none()
    );
    assert!(
        refit(&tree, &[stream(&channels)], &allowed, 0)
            .unwrap()
            .is_none()
    );
    let stats = collect_stats(&tree, &[stream(&channels)], &allowed, 0).unwrap();
    let candidates = &stats[0].as_ref().unwrap().candidates;
    assert_eq!(candidates.len(), 3);
    assert!(candidates.iter().all(|p| p.histogram.bits() == 0.0));
    // A nonzero singleton has identical predictions and extra-bit cost too.
    let channels = [plane(1, 1, |_, _| 12345)];
    assert!(
        refit(&tree, &[stream(&channels)], &allowed, 0)
            .unwrap()
            .is_none()
    );
    let gradient = single(Predictor::Gradient);
    assert!(
        refit(&gradient, &[stream(&channels)], &[Predictor::Gradient], 0)
            .unwrap()
            .is_none()
    );
}

#[test]
fn full_population_ramp_improves_with_each_permitted_predictor() {
    let channels = [plane(64, 32, |x, y| (13 * x + 7 * y) as i32)];
    let tree = single(Predictor::Zero);
    for predictor in [Predictor::Gradient, Predictor::Weighted] {
        let fitted = refit(&tree, &[stream(&channels)], &[predictor], 0)
            .unwrap()
            .expect("ramp improves over Zero");
        assert_eq!(fitted.lookup(&[0; 16]).unwrap().1, predictor);
        let stats = assert_raster_histograms(&tree, &[stream(&channels)], &[predictor], 0);
        let candidates = &stats[0].as_ref().unwrap().candidates;
        assert_eq!(candidates[0].histogram.count, 64 * 32);
        assert!(candidates[1].histogram.bits() < candidates[0].histogram.bits());
    }
    // The borrowed original remains unchanged even when it is not permitted.
    assert_eq!(tree.lookup(&[0; 16]).unwrap().1, Predictor::Zero);
}

#[test]
fn preserves_topology_non_bfs_node_mapping_and_unsampled_leaf() {
    let tree =
        Tree::channel_predictors(&[Predictor::Zero, Predictor::Left, Predictor::Top]).unwrap();
    let channels = [
        plane(64, 20, |x, y| (x * 17 + y * 9) as i32),
        plane(0, 0, |_, _| 0),
        plane(0, 0, |_, _| 0),
    ];
    let fitted = refit(&tree, &[stream(&channels)], &[Predictor::Gradient], 0)
        .unwrap()
        .unwrap();
    assert_eq!(fitted.lookup(&[0]).unwrap().1, Predictor::Gradient);
    assert_eq!(fitted.lookup(&[1]).unwrap().1, Predictor::Left);
    assert_eq!(fitted.lookup(&[2]).unwrap().1, Predictor::Top);
    assert_eq!(fitted.nodes().len(), tree.nodes().len());
    assert_eq!(fitted.leaves(), tree.leaves());
    for c in 0..3 {
        assert_eq!(tree.lookup(&[c]).unwrap().0, fitted.lookup(&[c]).unwrap().0);
    }
    for (original, changed) in tree.nodes().iter().zip(fitted.nodes()) {
        if matches!(original, Node::Split { .. }) {
            assert_eq!(format!("{original:?}"), format!("{changed:?}"));
        }
    }
}

#[test]
fn offset_and_multiplier_leaves_are_never_changed() {
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
        leaf(Predictor::Zero),
        Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 3,
        },
        Node::Leaf {
            predictor: Predictor::Zero,
            offset: 19,
            multiplier: 1,
        },
    ])
    .unwrap();
    let channels = [
        plane(48, 24, |x, y| (9 * x + 3 * y) as i32),
        plane(48, 24, |x, y| (9 * x + 3 * y + 19) as i32),
        plane(48, 24, |x, y| (27 * x + 9 * y) as i32),
    ];
    let fitted = refit(
        &tree,
        &[stream(&channels)],
        &[Predictor::Gradient, Predictor::Weighted],
        0,
    )
    .unwrap()
    .unwrap();
    assert_ne!(fitted.lookup(&[0]).unwrap().1, Predictor::Zero);
    assert_eq!(tree.lookup(&[1]).unwrap(), fitted.lookup(&[1]).unwrap());
    assert_eq!(tree.lookup(&[2]).unwrap(), fitted.lookup(&[2]).unwrap());
    for node in [tree.nodes()[3].clone(), tree.nodes()[4].clone()] {
        let fixed = Tree::new(vec![node]).unwrap();
        assert!(
            refit(&fixed, &[stream(&channels)], &[Predictor::Gradient], 0)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn wp_history_crosses_ineligible_pixels_and_drives_property15_without_wp_candidate() {
    let channels = [plane(43, 31, |x, y| {
        ((x * 71 + y * 137 + x * y * 29) % 4093) as i32 - 2000
    })];
    let tree = Tree::new(vec![
        Node::Split {
            property: 3,
            threshold: 17,
            greater: 3,
            less_equal: 1,
        },
        Node::Leaf {
            predictor: Predictor::Zero,
            offset: 7,
            multiplier: 1,
        },
        leaf(Predictor::Zero),
        Node::Split {
            property: 15,
            threshold: 0,
            greater: 4,
            less_equal: 2,
        },
        leaf(Predictor::Zero),
    ])
    .unwrap();
    let streams = [
        TrainingStream {
            channels: &channels,
            wp: weighted::Header::preset(3).unwrap(),
            stream_id: 91,
        },
        TrainingStream {
            channels: &channels,
            wp: weighted::Header::preset(1).unwrap(),
            stream_id: 7,
        },
    ];
    for allowed in [vec![Predictor::Gradient], vec![Predictor::Weighted]] {
        let stats = assert_raster_histograms(&tree, &streams, &allowed, 0);
        assert!(stats[0].is_none());
        let counts: Vec<_> = stats
            .iter()
            .flatten()
            .map(|leaf| leaf.candidates[0].histogram.count)
            .collect();
        assert!(counts.iter().all(|&count| count != 0));
        assert_eq!(counts.iter().sum::<u64>(), 2 * (43 - 18) * 31);
    }
    // An original WP predictor must also keep state when only Gradient is allowed.
    let weighted_tree = single(Predictor::Weighted);
    assert_raster_histograms(&weighted_tree, &streams, &[Predictor::Gradient], 0);
}

#[test]
fn keeps_actual_stream_channel_ids_metadata_and_matching_previous_channels() {
    let mut metadata = plane(32, 12, |_, _| -777);
    metadata.hshift = -1;
    metadata.vshift = -1;
    let reference = plane(32, 12, |x, _| if x % 2 == 0 { -100 } else { 100 });
    let mut mismatched = plane(32, 12, |_, _| 9999);
    mismatched.hshift = 1;
    let channels = [
        metadata,
        reference,
        mismatched,
        plane(32, 12, |x, y| (x * 17 + y) as i32),
    ];
    let tree = Tree::new(vec![
        Node::Split {
            property: 1,
            threshold: 40,
            greater: 1,
            less_equal: 2,
        },
        leaf(Predictor::Zero),
        Node::Split {
            property: 0,
            threshold: 2,
            greater: 3,
            less_equal: 4,
        },
        Node::Split {
            property: 17,
            threshold: 0,
            greater: 5,
            less_equal: 6,
        },
        Node::Leaf {
            predictor: Predictor::Left,
            offset: 5,
            multiplier: 1,
        },
        leaf(Predictor::Zero),
        leaf(Predictor::Zero),
    ])
    .unwrap();
    let streams = [
        TrainingStream {
            channels: &channels,
            wp: weighted::Header::preset(4).unwrap(),
            stream_id: 91,
        },
        TrainingStream {
            channels: &channels,
            wp: weighted::Header::preset(2).unwrap(),
            stream_id: 7,
        },
    ];
    let stats = assert_raster_histograms(&tree, &streams, &[Predictor::Weighted], 2);
    assert_eq!(
        stats[0].as_ref().unwrap().candidates[0].histogram.count,
        4 * 32 * 12
    );
    assert!(stats[1].is_none());
    for context in [2, 3] {
        assert_eq!(
            stats[context].as_ref().unwrap().candidates[0]
                .histogram
                .count,
            16 * 12
        );
    }
}

#[test]
fn wide_predictor_differences_wrap_before_tokenization() {
    let channels = [plane(9, 4, |x, y| {
        [i32::MIN, i32::MAX, 0, -1, 1][(x + y) % 5]
    })];
    let tree = single(Predictor::Average4);
    assert_raster_histograms(&tree, &[stream(&channels)], &[Predictor::Gradient], 0);
    let mut histogram = Histogram::new();
    for residual in [i32::MIN, i32::MAX, 0, -1, 1] {
        histogram.add(residual).unwrap();
    }
    assert_eq!(histogram.count, 5);
    assert!(histogram.bits().is_finite());
    let mut constant = Histogram::new();
    for _ in 0..10 {
        constant.add(i32::MIN).unwrap();
    }
    assert_eq!(constant.bits(), 10.0 * 28.0);
    let mut binary = Histogram::new();
    binary.add(0).unwrap();
    binary.add(1).unwrap();
    assert_eq!(binary.bits(), 2.0);
}

#[test]
fn entropy_ties_are_independent_of_token_labels() {
    let populations = [1, 3, 7, 11, 19, 23, 79];
    let mut original = Histogram::new();
    let mut permuted = Histogram::new();
    for (token, &population) in populations.iter().enumerate() {
        original.counts[token] = population;
        permuted.counts[15 - token] = population;
        original.count += population;
        permuted.count += population;
    }
    assert_eq!(original.bits(), permuted.bits());
}

#[test]
fn invalid_domains_return_errors_instead_of_partial_refits() {
    let tree = single(Predictor::Zero);
    let mut bad = plane(2, 3, |_, _| 0);
    bad.samples.pop();
    assert!(refit(&tree, &[stream(&[bad])], &[Predictor::Gradient], 0).is_err());
    let channels = [plane(2, 3, |_, _| 0)];
    assert!(
        refit(
            &tree,
            &[stream(&channels), stream(&channels)],
            &[Predictor::Gradient],
            0
        )
        .is_err()
    );
    assert!(refit(&tree, &[stream(&channels)], &[Predictor::Gradient], 61).is_err());
    let tree = Tree::new(vec![
        Node::Split {
            property: 16,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        leaf(Predictor::Zero),
        leaf(Predictor::Zero),
    ])
    .unwrap();
    assert!(refit(&tree, &[stream(&channels)], &[Predictor::Gradient], 0).is_err());
}
