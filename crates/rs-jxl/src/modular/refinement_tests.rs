use super::*;
use crate::modular::predictor::Predictor;
use crate::modular::tree::Node;
use std::cell::Cell;

fn tree() -> Tree {
    Tree::new(vec![Node::Leaf {
        predictor: Predictor::Gradient,
        offset: 0,
        multiplier: 1,
    }])
    .unwrap()
}

#[test]
fn context_map_trials_reuse_tokens_and_require_smaller_serialized_output() {
    use crate::entropy::cluster::ClusterStrategy;
    fn branches(nodes: &mut Vec<Node>, lo: i32, hi: i32) -> usize {
        let id = nodes.len();
        nodes.push(Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        });
        if hi - lo > 1 {
            let mid = (lo + hi) / 2;
            let greater = branches(nodes, mid, hi);
            let less_equal = branches(nodes, lo, mid);
            nodes[id] = Node::Split {
                property: 2,
                threshold: mid - 1,
                greater,
                less_equal,
            };
        }
        id
    }
    let mut nodes = Vec::new();
    branches(&mut nodes, 0, 1024);
    let tree = Tree::new(nodes).unwrap();
    let mut tokens = Vec::new();
    let mut state = 19u32;
    for context in 0..1024 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let id = if context < 2 {
            context as u32
        } else {
            state >> 31
        };
        let scale = if context < 2 { 150 } else { 1 };
        tokens.extend(std::iter::repeat_n(Token::literal(context, id), 2 * scale));
        tokens.extend(std::iter::repeat_n(Token::literal(context, 1 - id), scale));
    }
    for candidate_bits in [80, 100, 120] {
        let receipt = Arc::new(Receipt::default());
        let mut options = encode::ModularOptions::default();
        options.entropy = Coding::Ans;
        options.entropy_options.clustering = ClusterStrategy::Fast;
        options.refinement = Options {
            context_map: Some(ContextMapOptions::default()),
            receipt: Some(receipt.clone()),
            ..Default::default()
        };
        let tokenize_calls = Cell::new(0);
        let write_calls = Cell::new(0);
        let selected = choose(
            None,
            &tree,
            &[],
            &options,
            |_| {
                tokenize_calls.set(tokenize_calls.get() + 1);
                Ok((vec![tokens.clone(), vec![]], false))
            },
            |_, _, _, _| {
                let trial = write_calls.get();
                write_calls.set(trial + 1);
                Ok((if trial == 0 { 100 } else { candidate_bits }, trial))
            },
        )
        .unwrap();
        let stats = receipt.snapshot();
        assert_eq!(tokenize_calls.get(), 1);
        assert!(write_calls.get() >= 2);
        assert_eq!(selected != 0, candidate_bits < 100);
        assert_eq!(stats.context_map_wins, usize::from(candidate_bits < 100));
        assert_eq!(stats.saved_bits, 100 - candidate_bits.min(100));
        assert!(stats.context_map_proposals > 0);
        assert!(stats.context_map_proposals <= 2);
    }
}

#[test]
fn offset_trials_select_serialized_size_not_the_proxy() {
    use crate::modular::{group::Channel, weighted};
    let tree = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Zero,
        offset: 0,
        multiplier: 1,
    }])
    .unwrap();
    let channels = [Channel {
        width: 1024,
        height: 1,
        hshift: 0,
        vshift: 0,
        samples: vec![100; 1024],
    }];
    let views = [training::TrainingStream {
        channels: &channels,
        wp: weighted::Header::default(),
        stream_id: 0,
    }];
    for proposed_bits in [80, 100, 120] {
        let receipt = Arc::new(Receipt::default());
        let options = encode::ModularOptions {
            refinement: Options {
                offsets: Some(OffsetOptions {
                    mode: OffsetMode::Mean,
                    ..Default::default()
                }),
                receipt: Some(receipt.clone()),
                ..Default::default()
            },
            ..Default::default()
        };
        let selected = choose(
            None,
            &tree,
            &views,
            &options,
            |_| Ok((vec![vec![Token::literal(0, 0); 1024]], false)),
            |tree, _, _, _| {
                let Node::Leaf { offset, .. } = tree.nodes()[0] else {
                    unreachable!()
                };
                Ok((if offset == 0 { 100 } else { proposed_bits }, offset))
            },
        )
        .unwrap();
        assert_eq!(selected, if proposed_bits < 100 { 100 } else { 0 });
        let stats = receipt.snapshot();
        assert_eq!(stats.offset_proposals, 1);
        assert_eq!(stats.offset_wins, usize::from(proposed_bits < 100));
        assert_eq!(stats.models, 2);
        assert_eq!(stats.refit_wins, 0);
    }
}

#[test]
fn distribution_offset_trial_uses_real_config_and_keeps_exact_cost_fallback() {
    use crate::entropy::{HybridUintConfig, options::UintConfigStrategy};
    use crate::modular::{group::Channel, weighted};
    let tree = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Zero,
        offset: 17,
        multiplier: 1,
    }])
    .unwrap();
    let channels = [Channel {
        width: 512,
        height: 1,
        hshift: 0,
        vshift: 0,
        samples: vec![100; 512],
    }];
    let views = [training::TrainingStream {
        channels: &channels,
        wp: weighted::Header::default(),
        stream_id: 0,
    }];
    for candidate_bits in [80, 100, 120] {
        let receipt = Arc::new(Receipt::default());
        let mut options = encode::ModularOptions::default();
        options.fast_context = Some(crate::modular::fast_context::Options {
            bias: crate::modular::fast_context::Bias::Static,
            ..Default::default()
        });
        options.entropy_options.uint_config =
            UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap());
        options.refinement.offsets = Some(OffsetOptions {
            mode: OffsetMode::Distribution,
            ..Default::default()
        });
        options.refinement.receipt = Some(receipt.clone());
        let calls = Cell::new(0);
        let selected = choose(
            None,
            &tree,
            &views,
            &options,
            |_| {
                calls.set(calls.get() + 1);
                Ok((vec![vec![Token::literal(0, 200); 512]], false))
            },
            |tree, _, _, _| {
                let Node::Leaf { offset, .. } = tree.nodes()[0] else {
                    unreachable!()
                };
                Ok((if offset == 17 { 100 } else { candidate_bits }, offset))
            },
        )
        .unwrap();
        assert_eq!(selected != 17, candidate_bits < 100);
        assert_eq!(calls.get(), 2);
        assert_eq!(
            receipt.snapshot().offset_wins,
            usize::from(candidate_bits < 100)
        );
    }
}

#[test]
fn exact_entropy_trial_reuses_tokens_and_only_charges_winning_header() {
    for values in [
        vec![0; 2000],
        (0..3000).map(|i| (i * 73 % 39) as u32).collect(),
    ] {
        let streams = vec![values.into_iter().map(|v| Token::literal(0, v)).collect()];
        let calls = Cell::new(0);
        let receipt = Arc::new(Receipt::default());
        let costs = Arc::new(encode::receipt_cost::ModelCostReceipt::default());
        let options = encode::ModularOptions {
            entropy: Coding::Source,
            lz77: false,
            refinement: Options {
                entropy: true,
                receipt: Some(receipt.clone()),
                ..Default::default()
            },
            model_cost_receipt: Some(costs.clone()),
            ..Default::default()
        };
        let serialize = |tree: &Tree,
                         model: &Model,
                         streams: &[Vec<Token>],
                         options: &encode::ModularOptions| {
            let mut writer = BitWriter::new(1 << 20);
            encode::write_tree_and_model(tree, model, &mut writer, options)?;
            model.write_stream(&streams[0], &mut writer)?;
            Ok((
                writer.bits_written(),
                (writer.bits_written(), model.uses_ans()),
            ))
        };
        let (bits, ans) = choose(
            None,
            &tree(),
            &[],
            &options,
            |_| {
                calls.set(calls.get() + 1);
                Ok((streams.clone(), false))
            },
            serialize,
        )
        .unwrap();
        let mut raw_options = options.clone();
        raw_options.model_cost_receipt = None;
        let source =
            Model::build_with_options(&streams, 1, false, Coding::Source, &options.entropy_options)
                .unwrap();
        let alternate = Model::build_with_options(
            &streams,
            1,
            false,
            if source.uses_ans() {
                Coding::Prefix
            } else {
                Coding::Ans
            },
            &options.entropy_options,
        )
        .unwrap();
        let original = serialize(&tree(), &source, &streams, &raw_options)
            .unwrap()
            .0;
        let other = serialize(&tree(), &alternate, &streams, &raw_options)
            .unwrap()
            .0;
        assert_eq!(bits, original.min(other));
        assert_eq!(calls.get(), 1);
        assert_eq!(receipt.snapshot().models, 2);
        assert_eq!(receipt.snapshot().saved_bits, original - bits);
        let selected = if ans == source.uses_ans() {
            source
        } else {
            alternate
        };
        let mut header = BitWriter::new(1 << 20);
        encode::write_tree_and_model(&tree(), &selected, &mut header, &raw_options).unwrap();
        assert_eq!(costs.bits(), header.bits_written());
    }
}

#[test]
fn ties_retain_baseline_and_do_not_claim_a_win() {
    let receipt = Arc::new(Receipt::default());
    let options = encode::ModularOptions {
        refinement: Options {
            entropy: true,
            receipt: Some(receipt.clone()),
            ..Default::default()
        },
        ..Default::default()
    };
    let streams = vec![vec![Token::literal(0, 0); 1000]];
    let base = Model::build_with_options(
        &streams,
        1,
        false,
        options.entropy,
        &options.entropy_options,
    )
    .unwrap();
    let selected = choose(
        None,
        &tree(),
        &[],
        &options,
        |_| Ok((streams.clone(), false)),
        |_, model, _, _| Ok((100, model.uses_ans())),
    )
    .unwrap();
    assert_eq!(selected, base.uses_ans());
    assert_eq!(receipt.snapshot().entropy_wins, 0);
    assert_eq!(receipt.snapshot().saved_bits, 0);
}

#[test]
fn oversized_optional_output_keeps_baseline_but_allocation_errors_propagate() {
    use crate::bit_writer::BitWriterError;
    for error in [BitWriterError::SizeLimit, BitWriterError::Allocation] {
        let options = encode::ModularOptions {
            refinement: Options {
                entropy: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let calls = Cell::new(0);
        let result = choose(
            None,
            &tree(),
            &[],
            &options,
            |_| Ok((vec![vec![Token::literal(0, 0); 100]], false)),
            |_, _, _, _| {
                calls.set(calls.get() + 1);
                if calls.get() == 1 {
                    Ok((8, 19))
                } else {
                    Err(error.into())
                }
            },
        );
        if error == BitWriterError::SizeLimit {
            assert_eq!(result.unwrap(), 19);
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn refit_obeys_source_predictor_policy_and_rejects_larger_actual_output() {
    use crate::modular::group::Channel;
    use crate::modular::sample::{SamplingPolicy, SourcePolicy, TreeMode};
    let planes = vec![Channel {
        width: 64,
        height: 32,
        hshift: 0,
        vshift: 0,
        samples: (0..2048).map(|i| (i % 64) * 13 + (i / 64) * 7).collect(),
    }];
    let views = [training::TrainingStream {
        channels: &planes,
        wp: weighted::Header::default(),
        stream_id: 0,
    }];
    let original = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Zero,
        offset: 0,
        multiplier: 1,
    }])
    .unwrap();
    for (mode, permitted) in [
        (TreeMode::WeightedOnly, Predictor::Weighted),
        (TreeMode::NoWeighted, Predictor::Gradient),
    ] {
        for proposal_cost in [80, 120] {
            let receipt = Arc::new(Receipt::default());
            let options = encode::ModularOptions {
                predictor: Predictor::Best,
                sampling: Some(
                    SamplingPolicy::from_source(&SourcePolicy {
                        tree_mode: mode,
                        ..Default::default()
                    })
                    .unwrap(),
                ),
                refinement: Options {
                    leaves: true,
                    receipt: Some(receipt.clone()),
                    ..Default::default()
                },
                ..Default::default()
            };
            let selected = choose(
                None,
                &original,
                &views,
                &options,
                |_| Ok((vec![vec![Token::literal(0, 0); 100]], false)),
                |tree, _, _, _| {
                    let predictor = tree.predictors()[0];
                    if predictor != Predictor::Zero {
                        assert_eq!(predictor, permitted);
                    }
                    Ok((
                        if predictor == Predictor::Zero {
                            100
                        } else {
                            proposal_cost
                        },
                        predictor,
                    ))
                },
            )
            .unwrap();
            assert_eq!(receipt.snapshot().refit_proposals, 1);
            assert_eq!(
                selected,
                if proposal_cost < 100 {
                    permitted
                } else {
                    Predictor::Zero
                }
            );
            assert_eq!(
                receipt.snapshot().refit_wins,
                usize::from(proposal_cost < 100)
            );
        }
    }
}

#[test]
fn unaligned_local_trials_preserve_prefix_and_have_no_larger_payload() {
    use crate::modular::group::Channel;
    let planes = vec![Channel {
        width: 32,
        height: 16,
        hshift: 0,
        vshift: 0,
        samples: (0..512).map(|i| i * 37 % 256).collect(),
    }];
    let wp = weighted::Header::default();
    let mut options = encode::ModularOptions {
        lz77: false,
        ..Default::default()
    };
    let (tokens, lz77) = encode::group_stream(&planes, &tree(), &options, &wp, 0).unwrap();
    let streams = [tokens];
    let model =
        Model::build_with_options(&streams, 1, lz77, options.entropy, &options.entropy_options)
            .unwrap();
    for prefix in 0..8 {
        let mut base = BitWriter::new(1 << 20);
        base.write(prefix, (1 << prefix) - 1).unwrap();
        encode::write_tree_and_model(&tree(), &model, &mut base, &options).unwrap();
        model.write_stream(&streams[0], &mut base).unwrap();
        let mut refined = BitWriter::new(1 << 20);
        refined.write(prefix, (1 << prefix) - 1).unwrap();
        options.refinement.entropy = true;
        write_local(&tree(), &planes, &wp, 0, &options, &mut refined).unwrap();
        assert!(refined.bits_written() <= base.bits_written());
        refined.zero_pad_to_byte().unwrap();
        assert_eq!(
            refined.bytes().unwrap()[0] & ((1 << prefix) - 1),
            (1 << prefix) - 1
        );
    }
}

#[test]
fn local_source_lz_offset_reuse_reparses_and_preserves_actual_cost_selection() {
    use crate::entropy::{HybridUintConfig, options::UintConfigStrategy};
    use crate::modular::group::Channel;
    use crate::tools::lz77::{SourceConfig, SourceLz77Method};
    let tree = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Zero,
        offset: 17,
        multiplier: 1,
    }])
    .unwrap();
    let planes = [Channel {
        width: 64,
        height: 16,
        hshift: 0,
        vshift: 0,
        samples: (0..1024).map(|i| 1000 + i % 3).collect(),
    }];
    let wp = weighted::Header::default();
    let receipt = Arc::new(Receipt::default());
    let mut options = encode::ModularOptions::default();
    options.entropy_options.uint_config =
        UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap());
    options.source_lz77 = Some(SourceConfig {
        method: SourceLz77Method::Optc256,
        length_uint_config: options.entropy_options.lz77_length_config,
        min_symbol: Some(options.entropy_options.lz77_min_symbol),
        ..Default::default()
    });
    options.refinement.offsets = Some(OffsetOptions {
        mode: OffsetMode::Distribution,
        ..Default::default()
    });
    options.refinement.receipt = Some(receipt.clone());
    let mut optimized = BitWriter::new(1 << 20);
    write_local(&tree, &planes, &wp, 4, &options, &mut optimized).unwrap();
    let stats = receipt.snapshot();
    assert_eq!(stats.offset_proposals, 1);
    assert_eq!(stats.offset_raw_reused_tokens, 1024);
    assert_eq!(stats.offset_token_fits, 1);
    // The uncached comparison invokes native tokenization and source parsing
    // afresh for every tree, independently of the production raw cache.
    options.refinement.receipt = None;
    let views = [training::TrainingStream {
        channels: &planes,
        wp,
        stream_id: 4,
    }];
    let fresh_bits = choose(
        None,
        &tree,
        &views,
        &options,
        |candidate| {
            let (tokens, enabled) = encode::group_stream(&planes, candidate, &options, &wp, 4)?;
            Ok((vec![tokens], enabled))
        },
        |candidate, model, streams, scratch| {
            let mut writer = BitWriter::new(1 << 20);
            encode::write_tree_and_model(candidate, model, &mut writer, scratch)?;
            model.write_stream(&streams[0], &mut writer)?;
            Ok((writer.bits_written(), writer.bits_written()))
        },
    )
    .unwrap();
    assert_eq!(optimized.bits_written(), fresh_bits);
    assert!(stats.models >= 2);
}

#[test]
fn literal_offset_reuse_keeps_strict_actual_size_fallback() {
    use crate::entropy::{HybridUintConfig, options::UintConfigStrategy};
    use crate::modular::group::Channel;
    let tree = Tree::new(vec![Node::Leaf {
        predictor: Predictor::Zero,
        offset: 17,
        multiplier: 1,
    }])
    .unwrap();
    let planes = [Channel {
        width: 512,
        height: 1,
        hshift: 0,
        vshift: 0,
        samples: vec![1000; 512],
    }];
    let wp = weighted::Header::default();
    let views = [training::TrainingStream {
        channels: &planes,
        wp,
        stream_id: 0,
    }];
    for candidate_bits in [80, 100, 120] {
        let receipt = Arc::new(Receipt::default());
        let mut options = encode::ModularOptions {
            lz77: false,
            ..Default::default()
        };
        options.entropy_options.uint_config =
            UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap());
        options.refinement.offsets = Some(OffsetOptions {
            mode: OffsetMode::Distribution,
            ..Default::default()
        });
        options.refinement.receipt = Some(receipt.clone());
        let calls = Cell::new(0);
        let selected = choose(
            None,
            &tree,
            &views,
            &options,
            |candidate| {
                calls.set(calls.get() + 1);
                let (tokens, enabled) = encode::group_stream(&planes, candidate, &options, &wp, 0)?;
                Ok((vec![tokens], enabled))
            },
            |candidate, _, _, _| {
                let Node::Leaf { offset, .. } = candidate.nodes()[0] else {
                    unreachable!()
                };
                Ok((if offset == 17 { 100 } else { candidate_bits }, offset))
            },
        )
        .unwrap();
        assert_eq!(selected != 17, candidate_bits < 100);
        assert_eq!(calls.get(), 1);
        assert_eq!(receipt.snapshot().offset_token_fits, 1);
        assert_eq!(receipt.snapshot().offset_reused_tokens, 512);
    }
}

#[test]
fn offset_routing_fallback_retokenizes_instead_of_reusing_contexts() {
    use crate::entropy::{HybridUintConfig, options::UintConfigStrategy};
    use crate::modular::group::Channel;
    let tree = Tree::new(vec![
        Node::Split {
            property: 9,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        },
        Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        },
    ])
    .unwrap();
    let a = 1 << 30;
    // Repeat the overflow fixture in independent channels so the fitted bias
    // saves enough residual bits to pay for signaling a nonzero offset.
    let planes = vec![
        Channel {
            width: 2,
            height: 2,
            hshift: 0,
            vshift: 0,
            samples: vec![0, a, a, a + 1000],
        };
        128
    ];
    let wp = weighted::Header::default();
    let views = [training::TrainingStream {
        channels: &planes,
        wp,
        stream_id: 0,
    }];
    for lz77 in [false, true] {
        for refit in [false, true] {
            let receipt = Arc::new(Receipt::default());
            let mut options = encode::ModularOptions {
                lz77,
                ..Default::default()
            };
            options.entropy_options.uint_config =
                UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap());
            options.refinement.offsets = Some(OffsetOptions {
                mode: OffsetMode::Distribution,
                ..Default::default()
            });
            options.refinement.leaves = refit;
            options.refinement.receipt = Some(receipt.clone());
            let mut actual = BitWriter::new(1 << 20);
            write_local(&tree, &planes, &wp, 0, &options, &mut actual).unwrap();
            let stats = receipt.snapshot();
            assert_eq!(stats.offset_proposals, 1, "lz={lz77}, refit={refit}");
            assert_eq!(stats.offset_token_fits, 0);
            assert_eq!(stats.offset_raw_reused_tokens, 0);
            assert_eq!(stats.offset_reused_tokens, 0);
            options.refinement.receipt = None;
            let calls = Cell::new(0);
            let expected = choose(
                None,
                &tree,
                &views,
                &options,
                |candidate| {
                    calls.set(calls.get() + 1);
                    let (tokens, enabled) =
                        encode::group_stream(&planes, candidate, &options, &wp, 0)?;
                    Ok((vec![tokens], enabled))
                },
                |candidate, model, streams, scratch| {
                    let fresh = encode::group_stream(&planes, candidate, &options, &wp, 0)?.0;
                    assert_eq!(streams[0], fresh);
                    let mut writer = BitWriter::new(1 << 20);
                    encode::write_tree_and_model(candidate, model, &mut writer, scratch)?;
                    model.write_stream(&streams[0], &mut writer)?;
                    Ok((writer.bits_written(), writer.bits_written()))
                },
            )
            .unwrap();
            assert!(calls.get() >= 2);
            assert_eq!(actual.bits_written(), expected);
        }
    }
}
