use super::*;
use crate::entropy::stream::Coding;
use crate::modular::predictor::Predictor;

fn plane(values: &[i32]) -> Channel {
    Channel {
        width: values.len(),
        height: 1,
        hshift: 0,
        vshift: 0,
        samples: values.to_vec(),
    }
}

fn policy(predictor: Predictor) -> ModularOptions {
    ModularOptions {
        custom_tree: Some(Tree::channel_predictors(&[predictor]).unwrap()),
        lz77: false,
        rct: None,
        entropy: Coding::Prefix,
        ..Default::default()
    }
}

fn build_three() -> Shared {
    let first = [plane(&[1, 2, 5, -2])];
    let middle = [plane(&[4, 4, 4, 4])];
    let last = [plane(&[7, -3, 8])];
    let streams = [
        Stream {
            id: 2,
            channels: &first,
            transforms: &[],
        },
        Stream {
            id: 7,
            channels: &middle,
            transforms: &[],
        },
        Stream {
            id: 20,
            channels: &last,
            transforms: &[],
        },
    ];
    let zero = policy(Predictor::Zero);
    let left = policy(Predictor::Left);
    let top = policy(Predictor::Top);
    Shared::build(
        &[
            Domain {
                start: 1,
                streams: &streams[..1],
                options: &zero,
            },
            Domain {
                start: 6,
                streams: &streams[1..2],
                options: &left,
            },
            Domain {
                start: 19,
                streams: &streams[2..],
                options: &top,
            },
        ],
        40,
        &zero,
        8,
        1024,
    )
    .unwrap()
}

#[test]
fn domains_keep_actual_stream_ids_and_rebuild_contexts() {
    let shared = build_three();
    let mut properties = [0; 16];
    let mut contexts = Vec::new();
    for (id, predictor) in [
        (2, Predictor::Zero),
        (7, Predictor::Left),
        (20, Predictor::Top),
    ] {
        properties[1] = id;
        let (context, found, offset, multiplier) = shared.tree.lookup(&properties).unwrap();
        assert_eq!((found, offset, multiplier), (predictor, 0, 1));
        contexts.push(context);
        let stream = shared.streams.iter().find(|s| s.id == id as usize).unwrap();
        assert!(stream.tokens.iter().all(|t| t.context == context));
    }
    contexts.sort_unstable();
    assert_eq!(contexts, [0, 1, 2]);
    assert_eq!(shared.tree.leaves(), 3);
    // Left prediction resets at each channel's first sample.
    assert_eq!(
        shared.streams[1]
            .tokens
            .iter()
            .map(|t| t.value)
            .collect::<Vec<_>>(),
        [8, 0, 0, 0]
    );
}

#[test]
fn embedded_streams_preserve_all_bit_offsets_and_empty_slots_write_nothing() {
    let shared = build_three();
    let write = |w: &mut BitWriter| {
        shared.write_global_info(w).unwrap();
        for id in 0..40 {
            shared.write_stream(id, w).unwrap();
        }
    };
    let mut plain = BitWriter::new(1 << 20);
    write(&mut plain);
    let bits = plain.bits_written();
    plain.zero_pad_to_byte().unwrap();
    let plain = plain.into_bytes().unwrap();
    for offset in 0..8 {
        let mut writer = BitWriter::new(1 << 20);
        writer.write(offset, (1 << offset) - 1).unwrap();
        let before = writer.bits_written();
        shared.write_stream(3, &mut writer).unwrap();
        assert_eq!(writer.bits_written(), before);
        write(&mut writer);
        assert_eq!(writer.bits_written(), bits + usize::from(offset));
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        for i in 0..bits {
            let pos = i + usize::from(offset);
            assert_eq!(
                (plain[i / 8] >> (i % 8)) & 1,
                (bytes[pos / 8] >> (pos % 8)) & 1
            );
        }
    }
}

#[test]
fn malformed_domains_and_budgets_fail_before_traversal() {
    let policy = policy(Predictor::Zero);
    let planes = [plane(&[1, 2, 3])];
    let streams = [Stream {
        id: 3,
        channels: &planes,
        transforms: &[],
    }];
    let domains = [Domain {
        start: 2,
        streams: &streams,
        options: &policy,
    }];
    assert!(Shared::build(&domains, 4, &policy, 8, 2).is_err());
    assert!(Shared::build(&domains, 3, &policy, 8, 3).is_err());
    assert!(Shared::build(&domains, 4, &policy, 0, 3).is_err());
    assert!(Shared::build(&domains, 4, &policy, 8, 3).is_ok());
    let duplicate = [
        Domain {
            start: 2,
            streams: &streams,
            options: &policy,
        },
        Domain {
            start: 2,
            streams: &streams,
            options: &policy,
        },
    ];
    assert!(Shared::build(&duplicate, 4, &policy, 8, 6).is_err());
    let empty = [Domain {
        start: 1,
        streams: &[],
        options: &policy,
    }];
    assert!(Shared::build(&empty, 4, &policy, 8, 10).is_err());
    let invalid = [Channel {
        width: 9,
        ..plane(&[1, 2])
    }];
    let invalid_stream = [Stream {
        id: 1,
        channels: &invalid,
        transforms: &[],
    }];
    let invalid_domain = [Domain {
        start: 1,
        streams: &invalid_stream,
        options: &policy,
    }];
    assert!(Shared::build(&invalid_domain, 4, &policy, 8, 10).is_err());
}

#[test]
fn shared_source_parser_handles_distinct_widths_and_constant_domains() {
    use crate::entropy::options::SpeedTier;
    use crate::jpeg::modular_policy::StreamContext;
    let context = StreamContext {
        num_streams: 40,
        ..Default::default()
    };
    let dc = context
        .options_for_dc(SpeedTier::Tortoise as i8, 0)
        .unwrap();
    let meta = context
        .options_for_ac_metadata(SpeedTier::Tortoise as i8, 0)
        .unwrap();
    let planes = [plane(&vec![0; 1024])];
    let narrow = [plane(&[0; 8])];
    let first = [Stream {
        id: 1,
        channels: &planes,
        transforms: &[],
    }];
    let second = [Stream {
        id: 12,
        channels: &narrow,
        transforms: &[],
    }];
    let domains = [
        Domain {
            start: 1,
            streams: &first,
            options: &dc,
        },
        Domain {
            start: 11,
            streams: &second,
            options: &meta,
        },
    ];
    let shared = Shared::build(&domains, 40, &dc, 8, 1032).unwrap();
    let mut w = BitWriter::new(1 << 20);
    shared.write_global_info(&mut w).unwrap();
    shared.write_stream(1, &mut w).unwrap();
    shared.write_stream(12, &mut w).unwrap();
    assert!(w.bits_written() > 0);
}

#[test]
fn cooperative_domain_dispatch_bounds_roots_and_borrows_only_the_callers_pool() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Active<'a>(&'a AtomicUsize);
    impl Drop for Active<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let items: Vec<_> = (0..19).collect();
    for (tasks, scratch_bytes, cap) in [(8, 32 << 20, 1), (2, 128 << 20, 2), (8, 1, 1)] {
        let runner = LearnerExecution {
            pool: Some(&pool),
            cooperative: Some(WorkerLimits {
                tasks,
                scratch_bytes,
            }),
        };
        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let visited: Vec<_> = items.iter().map(|_| AtomicUsize::new(0)).collect();
        let actual = runner
            .run(&items, |i, &value, parallel| {
                assert!(parallel);
                assert!(!execution::is_serial());
                assert!(pool.current_thread_index().is_some());
                assert_eq!(visited[i].fetch_add(1, Ordering::SeqCst), 0);
                let active = live.fetch_add(1, Ordering::SeqCst) + 1;
                let _active = Active(&live);
                peak.fetch_max(active, Ordering::SeqCst);
                assert!(active <= cap);
                let scratch = vec![value; 257];
                let children = execution::map(&scratch, |&sample| {
                    assert!(pool.current_thread_index().is_some());
                    std::thread::yield_now();
                    sample + 1
                });
                assert_eq!(scratch, vec![value; 257]);
                assert_eq!(children, vec![value + 1; 257]);
                Ok(value * value)
            })
            .unwrap();
        assert_eq!(actual, items.iter().map(|n| n * n).collect::<Vec<_>>());
        assert!(peak.load(Ordering::SeqCst) > 0);
        assert!(peak.load(Ordering::SeqCst) <= cap);
        assert_eq!(live.load(Ordering::SeqCst), 0);
        assert!(visited.iter().all(|n| n.load(Ordering::SeqCst) == 1));
    }
    // One useful domain and default-off both retain serial learner searches.
    for cooperative in [None, Some(WorkerLimits::default())] {
        LearnerExecution {
            pool: Some(&pool),
            cooperative,
        }
        .run(&[0], |_, _, parallel| {
            assert!(!parallel);
            Ok(())
        })
        .unwrap();
    }
    LearnerExecution {
        pool: Some(&pool),
        cooperative: None,
    }
    .run(&items, |_, _, parallel| {
        assert!(!parallel);
        Ok(())
    })
    .unwrap();
}

#[test]
fn cooperative_null_scopes_limits_and_errors_do_not_dispatch_extra_roots() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let caller = std::thread::current().id();
    let check = |runner: LearnerExecution<'_>| {
        runner
            .run(&[1, 2, 3], |_, &n, parallel| {
                assert!(!parallel);
                assert!(execution::is_serial());
                assert_eq!(std::thread::current().id(), caller);
                execution::map(&[n; 8], |_| {
                    assert_eq!(std::thread::current().id(), caller);
                });
                Ok(n)
            })
            .unwrap()
    };
    assert_eq!(
        check(LearnerExecution {
            pool: None,
            cooperative: Some(WorkerLimits::default())
        }),
        [1, 2, 3]
    );
    assert_eq!(
        execution::serial(|| check(LearnerExecution {
            pool: Some(&pool),
            cooperative: Some(WorkerLimits::default())
        })),
        [1, 2, 3]
    );
    for limits in [
        WorkerLimits {
            tasks: 0,
            scratch_bytes: 1,
        },
        WorkerLimits {
            tasks: 65,
            scratch_bytes: 1,
        },
        WorkerLimits {
            tasks: 1,
            scratch_bytes: 0,
        },
    ] {
        assert!(
            LearnerExecution {
                pool: None,
                cooperative: Some(limits)
            }
            .run(&[0], |_, _, _| -> Result<()> {
                panic!("invalid limits reached a learner")
            })
            .is_err()
        );
    }
    let runner = LearnerExecution {
        pool: Some(&pool),
        cooperative: Some(WorkerLimits {
            tasks: 1,
            scratch_bytes: 1,
        }),
    };
    let started = AtomicUsize::new(0);
    let error = runner
        .run(&[0; 17], |_, _, parallel| -> Result<()> {
            assert!(parallel);
            started.fetch_add(1, Ordering::SeqCst);
            Err(Error::Limit("learner stopped"))
        })
        .unwrap_err();
    assert!(matches!(error, Error::Limit("learner stopped")));
    assert_eq!(started.load(Ordering::SeqCst), 1);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = runner.run(&[0; 17], |_, _, _| -> Result<()> {
                panic!("learner unwind")
            });
        }))
        .is_err()
    );
    assert_eq!(runner.run(&[2, 3], |_, &n, _| Ok(n * n)).unwrap(), [4, 9]);
    assert!(!execution::is_serial());
}

fn auxiliary_bytes(shared: &Shared) -> Vec<u8> {
    let mut writer = BitWriter::new(1 << 24);
    shared.write_global_info(&mut writer).unwrap();
    for id in 0..40 {
        shared.write_stream(id, &mut writer).unwrap();
    }
    writer.zero_pad_to_byte().unwrap();
    writer.into_bytes().unwrap()
}

#[test]
fn cooperative_learning_preserves_domains_effective_trees_tokens_and_sample_ceilings() {
    use crate::modular::sample::{SamplingPolicy, SourcePolicy};
    let planes: Vec<_> = [11usize, 37]
        .into_iter()
        .map(|salt| {
            vec![Channel {
                width: 64,
                height: 64,
                hshift: 0,
                vshift: 0,
                samples: (0..4096)
                    .map(|i| {
                        let x = i % 64;
                        let y = i / 64;
                        if x < 32 {
                            (x * 3 + y * 7 + salt) as i32
                        } else {
                            ((x * 157 + y * salt + x * y * 11) % 4093) as i32 - 2048
                        }
                    })
                    .collect(),
            }]
        })
        .collect();
    let constant = [plane(&[3; 16])];
    let streams = [
        Stream {
            id: 2,
            channels: &planes[0],
            transforms: &[],
        },
        Stream {
            id: 19,
            channels: &planes[1],
            transforms: &[],
        },
        Stream {
            id: 35,
            channels: &constant,
            transforms: &[],
        },
    ];
    let fixed = policy(Predictor::Zero);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for fraction in [0.0, 0.125, 1.0] {
        let learned = ModularOptions {
            predictor: Predictor::Gradient,
            custom_tree: None,
            max_tree_leaves: super::super::ma_train::MAX_LEAVES,
            sample_limit: Some(8192),
            sampling: Some(
                SamplingPolicy::from_source(&SourcePolicy {
                    speed_tier: 1,
                    predictor: Predictor::Gradient,
                    num_streams: 40,
                    learning_fraction: fraction,
                    ..Default::default()
                })
                .unwrap(),
            ),
            ..policy(Predictor::Gradient)
        };
        let domains = [
            Domain {
                start: 1,
                streams: &streams[..1],
                options: &learned,
            },
            Domain {
                start: 11,
                streams: &[],
                options: &learned,
            },
            Domain {
                start: 18,
                streams: &streams[1..2],
                options: &learned,
            },
            Domain {
                start: 30,
                streams: &streams[2..],
                options: &fixed,
            },
        ];
        let expected =
            execution::serial(|| Shared::build(&domains, 40, &learned, 8, 8208)).unwrap();
        let expected_bytes = auxiliary_bytes(&expected);
        if fraction == 1.0 {
            assert!(expected.tree.leaves() > 3);
        }
        for (borrowed, cooperative) in [
            (None, Some(WorkerLimits::default())),
            (Some(&pool), None),
            (
                Some(&pool),
                Some(WorkerLimits {
                    tasks: 2,
                    scratch_bytes: 64 << 20,
                }),
            ),
        ] {
            let actual = Shared::build_with_execution(
                &domains,
                40,
                &learned,
                8,
                8208,
                LearnerExecution {
                    pool: borrowed,
                    cooperative,
                },
            )
            .unwrap();
            assert_eq!(actual.tree.tokens(), expected.tree.tokens());
            assert_eq!(auxiliary_bytes(&actual), expected_bytes);
            for (actual, expected) in actual.streams.iter().zip(&expected.streams) {
                assert_eq!(actual.id, expected.id);
                assert_eq!(actual.tokens, expected.tokens);
            }
            // Independently tokenize with the effective merged tree, including
            // real ID gaps and the predefined domain's rebased contexts.
            for stream in &streams {
                let wp = encode::weighted_header(stream.channels, &learned).unwrap();
                let raw = tokens::tokenize(stream.channels, &actual.tree, &wp, 0, stream.id, false)
                    .unwrap();
                let expected: Vec<_> = raw
                    .into_iter()
                    .map(|t| Token::literal(t.context as usize, t.value))
                    .collect();
                assert_eq!(
                    actual
                        .streams
                        .iter()
                        .find(|s| s.id == stream.id)
                        .unwrap()
                        .tokens,
                    expected
                );
            }
        }
        if fraction == 1.0 {
            let mut limited = learned.clone();
            limited.sample_limit = Some(8191);
            for borrowed in [None, Some(&pool)] {
                let result = Shared::build_with_execution(
                    &domains,
                    40,
                    &limited,
                    8,
                    8208,
                    LearnerExecution {
                        pool: borrowed,
                        cooperative: Some(WorkerLimits::default()),
                    },
                );
                assert!(matches!(
                    result,
                    Err(Error::Limit("grouped MA selected-sample budget exceeded"))
                ));
            }
            limited.sample_limit = Some(4095);
            let local_domains = [
                Domain {
                    start: 1,
                    streams: &streams[..1],
                    options: &limited,
                },
                Domain {
                    start: 18,
                    streams: &streams[1..2],
                    options: &learned,
                },
            ];
            assert!(matches!(
                Shared::build_with_execution(
                    &local_domains,
                    40,
                    &learned,
                    8,
                    8192,
                    LearnerExecution {
                        pool: Some(&pool),
                        cooperative: Some(WorkerLimits::default())
                    }
                ),
                Err(Error::Limit(_))
            ));
        }
    }
}
