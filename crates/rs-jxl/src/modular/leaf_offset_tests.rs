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

#[test]
fn literal_fit_matches_pixel_walk_for_signed_depths_offsets_and_stream_order() {
    use crate::entropy::stream::Token;
    for predictor in [
        Predictor::Zero,
        Predictor::Left,
        Predictor::Top,
        Predictor::Gradient,
        Predictor::Select,
        Predictor::Average0,
        Predictor::Average1,
        Predictor::Average2,
        Predictor::Average3,
        Predictor::Average4,
        Predictor::TopRight,
        Predictor::TopLeft,
        Predictor::LeftLeft,
    ] {
        for scale in [1, 257] {
            let make_leaf = |offset| Node::Leaf {
                predictor,
                offset,
                multiplier: 1,
            };
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
                make_leaf(i32::MAX),
                make_leaf(-37),
                make_leaf(i32::MIN),
            ])
            .unwrap();
            let make_channels = |seed| {
                (0..3)
                    .map(|c| Channel {
                        width: 17,
                        height: 3,
                        hshift: 0,
                        vshift: 0,
                        samples: (0..51)
                            .map(|i| ((i * 73 + c * 31 + seed) % 256 - 128) * scale)
                            .collect(),
                    })
                    .collect::<Vec<_>>()
            };
            let a = make_channels(7);
            let b = make_channels(49);
            let views = [
                TrainingStream {
                    stream_id: 9,
                    ..stream(&a)
                },
                TrainingStream {
                    stream_id: 2,
                    ..stream(&b)
                },
            ];
            let encode_options = encode::ModularOptions {
                lz77: false,
                ..Default::default()
            };
            let mut tokens: Vec<Vec<Token>> = vec![vec![]];
            for view in &views {
                tokens.push(
                    encode::group_stream(
                        view.channels,
                        &tree,
                        &encode_options,
                        &view.wp,
                        view.stream_id,
                    )
                    .unwrap()
                    .0,
                );
                tokens.push(vec![]);
            }
            let options = OffsetOptions {
                mode: OffsetMode::Distribution,
                samples_per_leaf: 7,
            };
            let (mut pixel_samples, _) = collect_leaves(&tree, &options).unwrap();
            let (mut token_samples, _) = collect_leaves(&tree, &options).unwrap();
            for (view, encoded) in [(&views[1], &tokens[3]), (&views[0], &tokens[1])] {
                encode::visit(
                    view.channels,
                    &view.wp,
                    false,
                    0,
                    view.stream_id,
                    |_, _, neighbors, weighted, properties, value| {
                        let (context, predictor, _, _) = tree.lookup(properties)?;
                        let prediction =
                            predict_one(predictor, neighbors, Some(i64::from(weighted))).unwrap();
                        pixel_samples[context]
                            .as_mut()
                            .unwrap()
                            .add(i64::from(value) - prediction, 7)
                    },
                )
                .unwrap();
                for token in encoded {
                    let sample = token_samples[token.context].as_mut().unwrap();
                    let signed = ((token.value >> 1) as i32) ^ -((token.value & 1) as i32);
                    sample
                        .add(i64::from(signed.wrapping_add(sample.baseline)), 7)
                        .unwrap();
                }
            }
            for (pixel, token) in pixel_samples
                .iter()
                .flatten()
                .zip(token_samples.iter().flatten())
            {
                assert_eq!(
                    (pixel.count, pixel.sum, pixel.random, pixel.all_zero),
                    (token.count, token.sum, token.random, token.all_zero)
                );
                assert_eq!(pixel.values, token.values);
            }
            let configs = [
                HybridUintConfig::new(0, 0, 0).unwrap(),
                HybridUintConfig::default(),
                HybridUintConfig::new(4, 1, 2).unwrap(),
            ];
            let (pixel, ps) = fit_with_configs(&tree, &views, 0, &options, Some(&configs)).unwrap();
            let (cached, cs) = fit_tokens(&tree, &views, 0, &tokens, &options, Some(&configs))
                .unwrap()
                .expect("safe literals");
            let mut raw = vec![vec![]];
            for view in &views {
                raw.push(
                    super::super::tokens::tokenize(
                        view.channels,
                        &tree,
                        &view.wp,
                        0,
                        view.stream_id,
                        false,
                    )
                    .unwrap(),
                );
                raw.push(vec![]);
            }
            let (raw_fit, rs) = fit_literals(&tree, &views, 0, &raw, &options, Some(&configs))
                .unwrap()
                .expect("safe pre-LZ literals");
            assert_eq!(format!("{pixel:?}"), format!("{raw_fit:?}"));
            assert_eq!(
                (ps.samples, ps.population, ps.candidates, ps.leaves),
                (rs.samples, rs.population, rs.candidates, rs.leaves)
            );
            assert_eq!(format!("{pixel:?}"), format!("{cached:?}"));
            assert_eq!(
                (ps.samples, ps.population, ps.candidates, ps.leaves),
                (cs.samples, cs.population, cs.candidates, cs.leaves)
            );
            assert_eq!(cs.samples, 21);
            assert_eq!(cs.population, 306);
            // Check every remapped value against fresh native tokenization,
            // independently of whether the distribution proxy proposes a win.
            let mut nodes = tree.nodes().to_vec();
            for (node, offset) in [(2, i32::MIN), (3, 81), (4, i32::MAX)] {
                if let Node::Leaf { offset: value, .. } = &mut nodes[node] {
                    *value = offset;
                }
            }
            let after = Tree::new(nodes).unwrap();
            assert_eq!(remap_tokens(&tree, &after, &mut tokens).unwrap(), Some(306));
            for (view, cached) in views.iter().zip(tokens.iter().filter(|s| !s.is_empty())) {
                let fresh = encode::group_stream(
                    view.channels,
                    &after,
                    &encode_options,
                    &view.wp,
                    view.stream_id,
                )
                .unwrap()
                .0;
                assert_eq!(*cached, fresh);
            }
        }
    }
}

#[test]
fn literal_fit_falls_back_for_wide_average4_and_nonunit_data() {
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        ..Default::default()
    };
    for (predictor, multiplier, values) in [
        (Predictor::Average4, 1, vec![i32::MIN, 0]),
        (Predictor::Zero, 2, vec![100, 102]),
    ] {
        let tree = Tree::new(vec![Node::Leaf {
            predictor,
            offset: 0,
            multiplier,
        }])
        .unwrap();
        let channels = [plane(values)];
        let view = stream(&channels);
        let encoding = encode::ModularOptions {
            lz77: false,
            ..Default::default()
        };
        let tokens = vec![
            encode::group_stream(&channels, &tree, &encoding, &view.wp, 0)
                .unwrap()
                .0,
        ];
        assert!(
            fit_tokens(
                &tree,
                std::slice::from_ref(&view),
                0,
                &tokens,
                &options,
                None
            )
            .unwrap()
            .is_none()
        );
        let raw =
            vec![super::super::tokens::tokenize(&channels, &tree, &view.wp, 0, 0, false).unwrap()];
        assert!(
            fit_literals(&tree, std::slice::from_ref(&view), 0, &raw, &options, None)
                .unwrap()
                .is_none()
        );
        // Falling back still uses wide means and normal multiplier eligibility.
        fit(&tree, &[stream(&channels)], 0, &options).unwrap();
    }
}

#[test]
fn raw_fit_rejects_misassociated_streams_and_invalid_contexts() {
    use crate::tools::lz77::Literal;
    let tree = Tree::new(vec![leaf(17, 1)]).unwrap();
    let channels = [plane(vec![1000; 3])];
    let view = stream(&channels);
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        ..Default::default()
    };
    let raw = super::super::tokens::tokenize(&channels, &tree, &view.wp, 0, 0, false).unwrap();
    for invalid in [
        vec![],
        vec![raw[..2].to_vec()],
        vec![raw[..2].to_vec(), raw[2..].to_vec()],
        vec![vec![
            Literal {
                context: 1,
                value: 0
            };
            3
        ]],
    ] {
        assert!(
            fit_literals(
                &tree,
                std::slice::from_ref(&view),
                0,
                &invalid,
                &options,
                None
            )
            .unwrap()
            .is_none()
        );
    }
    assert!(
        fit_literals(
            &tree,
            std::slice::from_ref(&view),
            0,
            &[raw],
            &options,
            None
        )
        .unwrap()
        .is_some()
    );
}

#[test]
fn raw_fit_recovers_weighted_and_wide_residuals_from_pixels() {
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        samples_per_leaf: 7,
    };
    for id in 0..14 {
        let predictor = Predictor::try_from(id).unwrap();
        for wide in [false, true] {
            for offset in [i32::MIN, -37, i32::MAX] {
                let tree = Tree::new(vec![Node::Leaf {
                    predictor,
                    offset,
                    multiplier: 1,
                }])
                .unwrap();
                let channels = [Channel {
                    width: 19,
                    height: 3,
                    hshift: 0,
                    vshift: 0,
                    samples: (0..57)
                        .map(|i| {
                            if wide {
                                [i32::MIN, i32::MAX, -1, 0, 1][i % 5]
                            } else {
                                ((i * 7919) % 65536) as i32 - 32768
                            }
                        })
                        .collect(),
                }];
                let view = stream(&channels);
                let raw = vec![
                    super::super::tokens::tokenize(&channels, &tree, &view.wp, 0, 0, false)
                        .unwrap(),
                ];
                let configs = [HybridUintConfig::new(4, 1, 2).unwrap()];
                let views = std::slice::from_ref(&view);
                let cached = fit_literals(&tree, views, 0, &raw, &options, Some(&configs)).unwrap();
                if wide && predictor == Predictor::Average4 {
                    assert!(cached.is_none());
                    continue;
                }
                let (cached, cs) = cached.expect("i32 predictor is exactly recoverable");
                let (original, os) =
                    fit_with_configs(&tree, views, 0, &options, Some(&configs)).unwrap();
                assert_eq!(
                    format!("{cached:?}"),
                    format!("{original:?}"),
                    "{predictor:?} wide={wide} offset={offset}"
                );
                assert_eq!(
                    (cs.samples, cs.population, cs.leaves, cs.candidates),
                    (os.samples, os.population, os.leaves, os.candidates)
                );
            }
        }
    }
}

#[test]
fn single_pass_reuse_rejects_late_invalid_tokens_without_changing_inputs() {
    use crate::entropy::stream::Token;

    let tree = Tree::new(vec![leaf(-37, 1)]).unwrap();
    let channels = [plane((0..129).map(|i| i * 17 - 1000).collect())];
    let view = stream(&channels);
    let views = std::slice::from_ref(&view);
    let options = OffsetOptions {
        mode: OffsetMode::DistributionSinglePass,
        samples_per_leaf: 7,
    };
    let raw = super::super::tokens::tokenize(&channels, &tree, &view.wp, 0, 0, false).unwrap();
    let mut invalid_raw = raw.clone();
    invalid_raw.last_mut().unwrap().context = u32::MAX;
    let streams = [vec![], invalid_raw, vec![]];
    let unchanged = streams.clone();
    assert!(
        fit_literals(&tree, views, 0, &streams, &options, None)
            .unwrap()
            .is_none()
    );
    assert_eq!(streams, unchanged);

    let tokens: Vec<_> = raw
        .iter()
        .map(|t| Token::literal(t.context as usize, t.value))
        .collect();
    for length_token in [false, true] {
        let mut invalid = tokens.clone();
        let last = invalid.last_mut().unwrap();
        if length_token {
            last.is_length = true;
        } else {
            last.context = usize::MAX;
        }
        let streams = [vec![], invalid, vec![]];
        let unchanged = streams.clone();
        assert!(
            fit_tokens(&tree, views, 0, &streams, &options, None)
                .unwrap()
                .is_none()
        );
        assert_eq!(streams, unchanged);
    }

    let (expected, es) = fit_with_configs(&tree, views, 0, &options, None).unwrap();
    let (actual, actual_stats) = fit_literals(&tree, views, 0, &[raw], &options, None)
        .unwrap()
        .expect("valid literals still use the cached fitter");
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    assert_eq!(
        (
            actual_stats.samples,
            actual_stats.population,
            actual_stats.candidates,
            actual_stats.leaves
        ),
        (es.samples, es.population, es.candidates, es.leaves)
    );
}

#[test]
fn distribution_parallel_scores_preserve_order_and_signaling_sweeps() {
    fn build(nodes: &mut Vec<Node>, first: usize, end: usize) -> usize {
        let index = nodes.len();
        nodes.push(leaf((first as i32 % 7) - 3, 1));
        if end - first > 1 {
            let mid = (first + end) / 2;
            let less_equal = build(nodes, first, mid);
            let greater = build(nodes, mid, end);
            nodes[index] = Node::Split {
                property: 0,
                threshold: mid as i32 - 1,
                greater,
                less_equal,
            };
        }
        index
    }
    let mut nodes = Vec::new();
    build(&mut nodes, 0, 137);
    let tree = Tree::new(nodes).unwrap();
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        samples_per_leaf: 73,
    };
    let configs: Vec<_> = (0..tree.leaves())
        .map(|i| match i % 3 {
            0 => HybridUintConfig::default(),
            1 => HybridUintConfig::new(4, 1, 2).unwrap(),
            _ => HybridUintConfig::new(0, 0, 0).unwrap(),
        })
        .collect();
    let run = || {
        let (mut samples, counts) = collect_leaves(&tree, &options).unwrap();
        for (i, sample) in samples.iter_mut().enumerate() {
            let sample = sample.as_mut().unwrap();
            for j in 0..(173 + i % 71) {
                let value = if i % 11 == 0 {
                    sample.baseline as i64
                } else {
                    (i * 37 + j * 13 % 17) as i64 - 8192
                };
                sample.add(value, options.samples_per_leaf).unwrap();
            }
        }
        let (proposal, stats) =
            fit_distribution(&tree, &mut samples, counts, Some(&configs)).unwrap();
        (
            format!("{proposal:?}"),
            stats.leaves,
            stats.samples,
            stats.population,
            stats.candidates,
        )
    };
    let expected = crate::execution::serial(run);
    assert!(expected.1 > 0);
    for workers in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        assert_eq!(crate::execution::with_pool(&pool, run), expected);
    }
}

#[test]
fn raw_remapping_preserves_wide_residuals_and_rejects_incompatible_trees() {
    let before = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Left,
        offset: i32::MAX,
        multiplier: 1,
    }])
    .unwrap();
    let after = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Left,
        offset: i32::MIN,
        multiplier: 1,
    }])
    .unwrap();
    let channels = [plane(vec![i32::MIN, i32::MAX, 0, -1].repeat(32))];
    let wp = weighted::Header::default();
    let mut raw = vec![
        super::super::tokens::tokenize(&channels, &before, &wp, 0, 5, false).unwrap(),
        vec![],
    ];
    assert_eq!(
        remap_literals(&before, &after, &mut raw).unwrap(),
        Some(128)
    );
    assert_eq!(
        raw[0],
        super::super::tokens::tokenize(&channels, &after, &wp, 0, 5, false).unwrap()
    );
    for incompatible in [leaf(0, 2), leaf(0, 1)] {
        let other = Tree::new(vec![incompatible]).unwrap();
        let unchanged = raw.clone();
        assert_eq!(remap_literals(&before, &other, &mut raw).unwrap(), None);
        assert_eq!(raw, unchanged);
    }
}

#[test]
fn stateless_literal_fit_bound_prevents_modulo_aliases_at_extreme_offsets() {
    use crate::modular::predictor::Neighbors;
    const B: i32 = (1 << 28) - 1;
    for predictor in [
        Predictor::Zero,
        Predictor::Left,
        Predictor::Top,
        Predictor::Gradient,
        Predictor::Select,
        Predictor::Average0,
        Predictor::Average1,
        Predictor::Average2,
        Predictor::Average3,
        Predictor::Average4,
        Predictor::TopRight,
        Predictor::TopLeft,
        Predictor::LeftLeft,
    ] {
        for mask in 0..128 {
            let n = |bit| if mask & (1 << bit) == 0 { -B } else { B };
            let neighbors = Neighbors {
                left: n(0),
                top: n(1),
                toptop: n(2),
                topleft: n(3),
                topright: n(4),
                leftleft: n(5),
                toprightright: n(6),
            };
            let prediction = predict_one(predictor, neighbors, None).unwrap();
            for pixel in [-B, B] {
                let residual = i64::from(pixel) - prediction;
                assert!(i32::try_from(residual).is_ok());
                for offset in [i32::MIN, -17, 0, 17, i32::MAX] {
                    let packed = pack_signed((residual - i64::from(offset)) as i32);
                    let signed = ((packed >> 1) as i32) ^ -((packed & 1) as i32);
                    assert_eq!(i64::from(signed.wrapping_add(offset)), residual);
                }
            }
        }
    }
}
