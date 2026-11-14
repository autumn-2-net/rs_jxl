use super::*;
use crate::modular::{group::Channel, predictor::Predictor, tree::Node};

fn fixture(value: i32) -> (Tree, [Channel; 1]) {
    (
        Tree::new(vec![Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        }])
        .unwrap(),
        [Channel {
            width: 64,
            height: 16,
            hshift: 0,
            vshift: 0,
            samples: vec![value; 1024],
        }],
    )
}

#[test]
fn single_pass_runs_only_one_backend_and_does_not_claim_measured_savings() {
    for value in [0, 1000] {
        for lz77 in [false, true] {
            let (tree, planes) = fixture(value);
            let wp = weighted::Header::default();
            let views = [training::TrainingStream {
                channels: &planes,
                wp,
                stream_id: 0,
            }];
            let receipt = Arc::new(Receipt::default());
            let mut options = encode::ModularOptions {
                lz77,
                ..Default::default()
            };
            options.refinement.offsets = Some(OffsetOptions {
                mode: OffsetMode::DistributionSinglePass,
                ..Default::default()
            });
            options.refinement.receipt = Some(receipt.clone());
            let raw = super::super::tokens::tokenize(&planes, &tree, &wp, 0, 0, false).unwrap();
            let cache = LiteralCache::new(Some(vec![raw]));
            let parses = Cell::new(0);
            let writes = Cell::new(0);
            let selected = choose(
                Some(&cache),
                &tree,
                &views,
                &options,
                |candidate| {
                    parses.set(parses.get() + 1);
                    let (tokens, enabled) =
                        encode::group_stream(&planes, candidate, &options, &wp, 0)?;
                    Ok((vec![tokens], enabled))
                },
                |tree, _, _, _| {
                    writes.set(writes.get() + 1);
                    let Node::Leaf { offset, .. } = tree.nodes()[0] else {
                        unreachable!()
                    };
                    // Direct mode must not encode the original to reject this result.
                    Ok((if offset == 0 { 80 } else { 120 }, offset))
                },
            )
            .unwrap();
            assert_eq!(parses.get(), 1);
            assert_eq!(writes.get(), 1);
            assert_eq!(selected != 0, value != 0);
            let stats = receipt.snapshot();
            assert_eq!(stats.models, 1);
            assert_eq!(stats.offset_single_pass_domains, 1);
            assert_eq!(stats.offset_token_fits, 1);
            assert_eq!(stats.offset_applied, usize::from(value != 0));
            assert_eq!(stats.offset_wins, 0);
            assert_eq!(stats.saved_bits, 0);
        }
    }
}

#[test]
fn single_pass_local_cached_residuals_match_fresh_final_tree_tokens() {
    for lz77 in [false, true] {
        let (tree, planes) = fixture(1000);
        let wp = weighted::Header::default();
        let receipt = Arc::new(Receipt::default());
        let mut options = encode::ModularOptions {
            lz77,
            ..Default::default()
        };
        options.refinement.offsets = Some(OffsetOptions {
            mode: OffsetMode::DistributionSinglePass,
            ..Default::default()
        });
        options.refinement.receipt = Some(receipt.clone());
        let mut actual = BitWriter::new(1 << 20);
        write_local(&tree, &planes, &wp, 0, &options, &mut actual).unwrap();
        let stats = receipt.snapshot();
        assert_eq!(stats.models, 1);
        assert_eq!(stats.offset_applied, 1);
        assert_eq!(stats.offset_raw_reused_tokens, 1024);
        let views = [training::TrainingStream {
            channels: &planes,
            wp,
            stream_id: 0,
        }];
        let refined = super::super::leaf_offset::fit(
            &tree,
            &views,
            0,
            options.refinement.offsets.as_ref().unwrap(),
        )
        .unwrap()
        .0
        .unwrap();
        let (tokens, enabled) = encode::group_stream(&planes, &refined, &options, &wp, 0).unwrap();
        let model = Model::build_with_options(
            &[tokens.clone()],
            refined.leaves() + usize::from(enabled),
            enabled,
            options.entropy,
            &options.entropy_options,
        )
        .unwrap();
        let mut expected = BitWriter::new(1 << 20);
        encode::write_tree_and_model(&refined, &model, &mut expected, &options).unwrap();
        model.write_stream(&tokens, &mut expected).unwrap();
        assert_eq!(actual.bits_written(), expected.bits_written());
        actual.zero_pad_to_byte().unwrap();
        expected.zero_pad_to_byte().unwrap();
        assert_eq!(actual.into_bytes().unwrap(), expected.into_bytes().unwrap());
    }
}

#[test]
fn single_pass_propagates_output_errors_without_a_hidden_baseline() {
    let (tree, planes) = fixture(1000);
    let views = [training::TrainingStream {
        channels: &planes,
        wp: weighted::Header::default(),
        stream_id: 0,
    }];
    let mut options = encode::ModularOptions::default();
    options.refinement.offsets = Some(OffsetOptions {
        mode: OffsetMode::DistributionSinglePass,
        ..Default::default()
    });
    let calls = Cell::new(0);
    let result = choose::<()>(
        None,
        &tree,
        &views,
        &options,
        |_| Ok((vec![vec![Token::literal(0, 0); 1024]], false)),
        |_, _, _, _| {
            calls.set(calls.get() + 1);
            Err(Error::Limit("encoded output size"))
        },
    );
    assert!(matches!(result, Err(Error::Limit("encoded output size"))));
    assert_eq!(calls.get(), 1);
}
