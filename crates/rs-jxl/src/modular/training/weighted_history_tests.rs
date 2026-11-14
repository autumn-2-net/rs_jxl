// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; the complete upstream notice is in docs/PORTING_KERNELS.md.

use super::*;

fn check_shared_weighted_histories(threads: usize, partial: bool) {
    use crate::modular::{tokens, weighted_history::History};

    let first = [channel(256, 256, -17), channel(257, 129, 7)];
    let second = first.clone();
    let third = first.clone();
    let headers = [0, 2, 4].map(|preset| weighted::Header::preset(preset).unwrap());
    assert_ne!(headers[0], headers[1]);
    assert_ne!(headers[0], headers[2]);
    assert_ne!(headers[1], headers[2]);
    let streams = [
        TrainingStream {
            channels: &first,
            wp: headers[0],
            stream_id: 37,
        },
        TrainingStream {
            channels: &second,
            wp: headers[1],
            stream_id: 5,
        },
        TrainingStream {
            channels: &third,
            wp: headers[2],
            stream_id: 19,
        },
    ];
    for c in 0..first.len() {
        assert_eq!(first[c].samples, second[c].samples);
        assert_eq!(first[c].samples, third[c].samples);
        assert_ne!(first[c].samples.as_ptr(), second[c].samples.as_ptr());
        assert_ne!(first[c].samples.as_ptr(), third[c].samples.as_ptr());
    }
    let mut opts = options();
    opts.predictor = Predictor::Weighted;
    opts.previous_channels = 1;
    opts.source_gather = Some(GatherOptions {
        workers: threads,
        sparse: true,
        ..Default::default()
    });
    let policy = opts.sampling.as_mut().unwrap();
    policy.nb_repeats = 0.125;
    policy.previous_channels = 1;
    policy.properties = vec![0, 1, 9, 15, 17, 19];
    policy.max_channel_size = if partial { 256 } else { usize::MAX };

    let check = || {
        let history37 = History::new(&first, headers[0], &opts).unwrap();
        let history19 = History::new(&third, headers[2], &opts).unwrap();
        // Cache positions deliberately differ from sorted learning stream positions.
        let histories = [&history37, &history19];
        let plain = learn(&streams, &opts).unwrap();
        let cached = learn_with_histories(&streams, &opts, &histories).unwrap();
        assert_eq!(cached.leaves(), plain.leaves());
        assert_eq!(cached.tokens(), plain.tokens());
        assert!(history37.matches(&first, &headers[0]));
        assert!(history19.matches(&third, &headers[2]));
        assert!(!history37.matches(&second, &headers[0]));
        assert!(!history37.matches(&first, &headers[1]));
        assert!(!history19.matches(&second, &headers[2]));

        for (input, history) in [
            (&streams[0], Some(&history37)),
            (&streams[1], None),
            (&streams[2], Some(&history19)),
        ] {
            if let Some(history) = history {
                for c in 0..input.channels.len() {
                    let values = history.channel(c);
                    assert_eq!(values.is_some(), !partial || c == 0);
                    if let Some(values) = values {
                        assert_eq!(values.len(), input.channels[c].samples.len());
                        encode::visit_channel(
                            input.channels,
                            c,
                            &input.wp,
                            true,
                            opts.previous_channels,
                            input.stream_id,
                            |_, pos, _, prediction, props, _| {
                                assert_eq!(values[pos], [prediction, props[15]]);
                                Ok(())
                            },
                        )
                        .unwrap();
                    }
                }
            }
            let expected = tokens::tokenize(
                input.channels,
                &plain,
                &input.wp,
                opts.previous_channels,
                input.stream_id,
                false,
            )
            .unwrap();
            for parallel in [false, true] {
                let actual = tokens::tokenize_with_history(
                    input.channels,
                    &cached,
                    &input.wp,
                    opts.previous_channels,
                    input.stream_id,
                    parallel,
                    history,
                )
                .unwrap();
                assert_eq!(
                    actual.len(),
                    input
                        .channels
                        .iter()
                        .map(|c| c.samples.len())
                        .sum::<usize>()
                );
                assert_eq!(actual, expected);
            }
        }
        // An unrelated cache must also fall back when explicitly supplied.
        assert_eq!(
            tokens::tokenize_with_history(
                &second,
                &cached,
                &headers[1],
                opts.previous_channels,
                streams[1].stream_id,
                true,
                Some(&history37),
            )
            .unwrap(),
            tokens::tokenize(
                &second,
                &plain,
                &headers[1],
                opts.previous_channels,
                streams[1].stream_id,
                false,
            )
            .unwrap()
        );
    };
    if threads == 1 {
        execution::serial(check);
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(check);
    }
}

#[test]
fn weighted_history_shared_learning_matches_uncached_with_unsorted_streams() {
    for threads in [1, 4] {
        check_shared_weighted_histories(threads, false);
    }
}

#[test]
fn weighted_history_shared_learning_partial_capture_falls_back_per_channel() {
    for threads in [1, 4] {
        check_shared_weighted_histories(threads, true);
    }
}

#[test]
fn weighted_history_reuses_exact_gather_predictions_and_tokens() {
    use crate::modular::{tokens, tree::Node, weighted_history::History};
    for width in [1, 2, 8, 9, 17, 257] {
        let channels = [deep_channel(width, 19), deep_channel(width, 19)];
        for preset in 0..5 {
            let header = weighted::Header::preset(preset).unwrap();
            let mut opts = options();
            opts.previous_channels = 1;
            let policy = opts.sampling.as_mut().unwrap();
            policy.previous_channels = 1;
            policy.max_channel_size = usize::MAX;
            policy.properties = (0..20).collect();
            let policy = opts.sampling.as_ref().unwrap();
            let input = TrainingStream {
                channels: &channels,
                wp: header,
                stream_id: 0,
            };
            let streams = [&input];
            let predictors = [Predictor::Weighted, Predictor::Gradient];
            let quantization = gather_quantization(&streams, policy);
            let source = SourceGather {
                streams: &streams,
                policy,
                predictors: &predictors,
                variable_channels: 2,
                sample_limit: usize::MAX,
                quantization: &quantization,
                budget: None,
            };
            let history = History::new(&channels, header, &opts).unwrap();
            let gathered = source
                .collect_prepared_with_history(&GatherOptions::default(), Some(&history))
                .unwrap();
            let plain = source.collect_prepared(&GatherOptions::default()).unwrap();
            assert_sample_sequences(
                &gathered.samples.into_samples().unwrap(),
                &plain.samples.into_samples().unwrap(),
            );
            for c in 0..2 {
                let cache = history.channel(c).unwrap();
                assert_eq!(cache.len(), channels[c].samples.len());
                encode::visit_channel(
                    &channels,
                    c,
                    &header,
                    true,
                    1,
                    37,
                    |_, pos, _, prediction, props, _| {
                        assert_eq!(cache[pos], [prediction, props[15]]);
                        Ok(())
                    },
                )
                .unwrap();
            }
            let mut trees = vec![
                Tree::channel_predictors(&[Predictor::Weighted]).unwrap(),
                Tree::channel_predictors(&[Predictor::Gradient]).unwrap(),
                crate::modular::fixed::build(
                    16,
                    crate::modular::fixed::FixedTreeMode::Weighted {
                        total_pixels: width * 19,
                    },
                )
                .unwrap(),
            ];
            for property in [1, 8, 15, 19] {
                trees.push(
                    Tree::new(vec![
                        Node::Split {
                            property,
                            threshold: 8,
                            greater: 1,
                            less_equal: 2,
                        },
                        Node::Leaf {
                            predictor: Predictor::Weighted,
                            offset: 11,
                            multiplier: 1,
                        },
                        Node::Leaf {
                            predictor: Predictor::Gradient,
                            offset: -3,
                            multiplier: 1,
                        },
                    ])
                    .unwrap(),
                );
            }
            for tree in trees {
                let expected = tokens::tokenize(&channels, &tree, &header, 1, 37, false).unwrap();
                let actual = tokens::tokenize_with_history(
                    &channels,
                    &tree,
                    &header,
                    1,
                    37,
                    false,
                    Some(&history),
                )
                .unwrap();
                assert_eq!(actual, expected);
                // Different allocations or predictor headers must never reuse cached values.
                let copied = channels.clone();
                assert!(!history.matches(&copied, &header));
                assert_eq!(
                    tokens::tokenize_with_history(
                        &copied,
                        &tree,
                        &header,
                        1,
                        37,
                        false,
                        Some(&history)
                    )
                    .unwrap(),
                    expected
                );
                let other = weighted::Header::preset((preset + 1) % 5).unwrap();
                assert!(!history.matches(&channels, &other));
                assert_eq!(
                    tokens::tokenize_with_history(
                        &channels,
                        &tree,
                        &other,
                        1,
                        37,
                        false,
                        Some(&history)
                    )
                    .unwrap(),
                    tokens::tokenize(&channels, &tree, &other, 1, 37, false).unwrap()
                );
            }
        }
    }
}

#[test]
fn weighted_history_local_learning_and_parallel_tokenization_preserve_results() {
    use crate::modular::{tokens, weighted_history::History};
    let channels = [
        channel(257, 129, -17),
        channel(257, 129, 7),
        channel(257, 129, 71),
    ];
    let header = weighted::Header::default();
    let mut opts = options();
    opts.source_gather = Some(GatherOptions::default());
    opts.sampling.as_mut().unwrap().max_channel_size = usize::MAX;
    let input = stream(&channels, 0);
    let history = History::new(&channels, header, &opts).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let tree = pool
        .install(|| learn_local_with_history(&[input], &opts, Some(&history)))
        .unwrap();
    let plain = pool
        .install(|| learn_local(&[stream(&channels, 0)], &opts))
        .unwrap();
    assert_eq!(tree.leaves(), plain.leaves());
    assert!(history.channel(0).is_some());
    let expected = tokens::tokenize(&channels, &plain, &header, 0, 19, false).unwrap();
    assert_eq!(
        pool.install(|| tokens::tokenize_with_history(
            &channels,
            &tree,
            &header,
            0,
            19,
            true,
            Some(&history)
        ))
        .unwrap(),
        expected
    );
    let mut striped = opts.clone();
    striped.source_gather.as_mut().unwrap().stripes = 2;
    assert!(History::new(&channels, header, &striped).is_none());
    let mut zero = opts.clone();
    zero.sampling.as_mut().unwrap().nb_repeats = 0.0;
    assert!(History::new(&channels, header, &zero).is_none());
}

#[test]
fn weighted_history_parallel_gather_publishes_complete_channels() {
    use crate::modular::weighted_history::History;
    let channels = [channel(256, 256, -17), channel(256, 256, 7)];
    let header = weighted::Header::default();
    let mut opts = options();
    opts.predictor = Predictor::Weighted;
    let policy = opts.sampling.as_mut().unwrap();
    policy.nb_repeats = 0.125;
    policy.properties = vec![15];
    let policy = opts.sampling.as_ref().unwrap();
    let input = stream(&channels, 0);
    let streams = [&input];
    let predictors = [Predictor::Weighted];
    let quantization = gather_quantization(&streams, policy);
    let source = SourceGather {
        streams: &streams,
        policy,
        predictors: &predictors,
        variable_channels: channels.len(),
        sample_limit: usize::MAX,
        quantization: &quantization,
        budget: None,
    };
    let gather = GatherOptions {
        workers: 2,
        sparse: true,
        ..Default::default()
    };
    let fraction = sample::tree_sample_fraction(65_536, policy.nb_repeats).unwrap();
    let waves = gather
        .plan(
            &[65_536; 2],
            &[fraction * 65_536.0 + 64.0; 2],
            predictors.len(),
            &policy.properties,
            true,
        )
        .unwrap();
    assert_eq!(waves.len(), 1);
    assert_eq!(waves[0].chunks, vec![0..1, 1..2]);
    let history = History::new(&channels, header, &opts).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let actual = pool
        .install(|| source.collect_prepared_with_history(&gather, Some(&history)))
        .unwrap();
    assert!(actual.parallel_waves > 0);
    assert_eq!(actual.serial_tasks, 0);
    let expected = execution::serial(|| source.collect_prepared(&gather)).unwrap();
    assert_eq!(actual.stats, expected.stats);
    assert_eq!(
        weighted_population(&actual.samples.into_samples().unwrap(), &quantization),
        weighted_population(&expected.samples.into_samples().unwrap(), &quantization),
    );
    for (c, plane) in channels.iter().enumerate() {
        let cached = history.channel(c).expect("parallel channel published");
        assert_eq!(cached.len(), plane.samples.len());
        encode::visit_channel(
            &channels,
            c,
            &header,
            true,
            0,
            19,
            |_, pos, _, prediction, props, _| {
                assert_eq!(cached[pos], [prediction, props[15]]);
                Ok(())
            },
        )
        .unwrap();
    }
}

#[test]
fn weighted_history_sample_limit_never_publishes_partial_channel() {
    use crate::modular::weighted_history::History;
    let channels = [channel(8, 2, -17), channel(8, 4, 7)];
    let header = weighted::Header::default();
    let mut opts = options();
    let policy = opts.sampling.as_mut().unwrap();
    policy.nb_repeats = 1.0;
    policy.properties = vec![15];
    let policy = opts.sampling.as_ref().unwrap();
    let input = stream(&channels, 0);
    let streams = [&input];
    let predictors = [Predictor::Weighted];
    let quantization = gather_quantization(&streams, policy);
    let limit = channels[0].samples.len() + 7;
    let budget = SampleBudget::new(limit);
    let mut source = SourceGather {
        streams: &streams,
        policy,
        predictors: &predictors,
        variable_channels: channels.len(),
        sample_limit: limit,
        quantization: &quantization,
        budget: Some(&budget),
    };
    let gather = GatherOptions {
        workers: 1,
        ..Default::default()
    };
    let history = History::new(&channels, header, &opts).unwrap();
    assert!(matches!(
        source.collect_prepared_with_history(&gather, Some(&history)),
        Err(Error::Limit("grouped MA selected-sample budget exceeded"))
    ));
    assert_eq!(budget.remaining.load(Ordering::Relaxed), 0);
    assert_eq!(history.channel(0).unwrap().len(), channels[0].samples.len());
    assert!(history.channel(1).is_none());
    // A failed capture must leave the slot available for a complete retry.
    source.sample_limit = usize::MAX;
    source.budget = None;
    source
        .collect_prepared_with_history(&gather, Some(&history))
        .unwrap();
    assert_eq!(history.channel(1).unwrap().len(), channels[1].samples.len());
}

#[test]
fn weighted_history_partial_channel_prefix_keeps_complete_token_output() {
    use crate::modular::{tokens, weighted_history::History};
    let mut channels = [channel(8, 8, -17), channel(17, 4, 7), channel(8, 8, 0)];
    channels[2].samples.fill(7);
    let header = weighted::Header::default();
    // The size cutoff stops at channel 1, including the smaller channel after
    // it. The variable-channel cutoff instead omits only the constant suffix.
    for (maximum, variable_channels, cached_channels) in [(8, 3, 1), (usize::MAX, 2, 2)] {
        let mut opts = options();
        let policy = opts.sampling.as_mut().unwrap();
        policy.nb_repeats = 1.0;
        policy.properties = vec![15];
        policy.max_channel_size = maximum;
        let policy = opts.sampling.as_ref().unwrap();
        let input = stream(&channels, 0);
        let streams = [&input];
        let predictors = [Predictor::Weighted];
        let quantization = gather_quantization(&streams, policy);
        let source = SourceGather {
            streams: &streams,
            policy,
            predictors: &predictors,
            variable_channels,
            sample_limit: usize::MAX,
            quantization: &quantization,
            budget: None,
        };
        let history = History::new(&channels, header, &opts).unwrap();
        let actual = source
            .collect_prepared_with_history(&GatherOptions::default(), Some(&history))
            .unwrap();
        assert_eq!(
            actual.stats.visited_pixels,
            channels[..cached_channels]
                .iter()
                .map(|c| c.samples.len() as u64)
                .sum::<u64>()
        );
        for c in 0..channels.len() {
            assert_eq!(history.channel(c).is_some(), c < cached_channels);
        }
        for predictor in [Predictor::Weighted, Predictor::Gradient] {
            let tree = Tree::channel_predictors(&[predictor]).unwrap();
            let expected = tokens::tokenize(&channels, &tree, &header, 0, 19, false).unwrap();
            let actual = tokens::tokenize_with_history(
                &channels,
                &tree,
                &header,
                0,
                19,
                false,
                Some(&history),
            )
            .unwrap();
            assert_eq!(
                actual.len(),
                channels.iter().map(|c| c.samples.len()).sum::<usize>()
            );
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn weighted_history_eligibility_checks_aggregate_budget_and_unsupported_policies() {
    use crate::modular::weighted_history::History;
    let header = weighted::Header::default();
    let opts = options();
    let samples = (8 << 20) / std::mem::size_of::<[i32; 2]>();
    let mut channels = [
        Channel {
            width: samples - 1,
            height: 1,
            hshift: 0,
            vshift: 0,
            samples: vec![0; samples - 1],
        },
        channel(1, 1, 0),
    ];
    assert!(History::new(&channels, header, &opts).is_some());
    channels[1].samples.push(0);
    channels[1].width = 2;
    assert!(History::new(&channels, header, &opts).is_none());
    assert!(History::new(&[], header, &opts).is_none());
    assert!(History::new(&[channel(1, 0, 0)], header, &opts).is_none());

    let small = [channel(8, 4, 0)];
    assert!(History::new(&small, header, &opts).is_some());
    let policies: &[(&str, fn(&mut ModularOptions))] = &[
        ("no sampling", |o| o.sampling = None),
        ("zero sampling", |o| {
            o.sampling.as_mut().unwrap().nb_repeats = 0.0
        }),
        ("fixed tree", |o| {
            o.fixed_tree = Some(encode::FixedTree::Weighted)
        }),
        ("custom tree", |o| {
            o.custom_tree = Some(Tree::channel_predictors(&[Predictor::Weighted]).unwrap())
        }),
        ("tree groups", |o| o.tree_groups = Some(Default::default())),
        ("block context", |o| {
            o.block_context = Some(Default::default())
        }),
        ("fast context", |o| {
            o.fast_context = Some(Default::default())
        }),
        ("entropy refinement", |o| o.refinement.entropy = true),
        ("leaf refinement", |o| o.refinement.leaves = true),
        ("offset refinement", |o| {
            o.refinement.offsets = Some(Default::default())
        }),
        ("context refinement", |o| {
            o.refinement.context_map = Some(Default::default())
        }),
        ("stripes", |o| {
            o.source_gather = Some(GatherOptions {
                stripes: 2,
                ..Default::default()
            })
        }),
        ("no weighted predictor or property", |o| {
            o.predictor = Predictor::Gradient;
            o.sampling.as_mut().unwrap().properties = vec![9];
        }),
    ];
    for &(name, change) in policies {
        let mut unsupported = opts.clone();
        change(&mut unsupported);
        assert!(
            History::new(&small, header, &unsupported).is_none(),
            "{name}"
        );
    }
    // Property 15 alone still needs WP, even with a non-weighted predictor.
    let mut property_only = opts;
    property_only.predictor = Predictor::Gradient;
    property_only.sampling.as_mut().unwrap().properties = vec![15];
    assert!(History::new(&small, header, &property_only).is_some());
}
