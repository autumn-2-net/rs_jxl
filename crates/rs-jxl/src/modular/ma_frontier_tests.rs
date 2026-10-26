use super::*;

const RECORDS: usize = 16_384;

fn fixture(low_records: usize, wide: bool) -> (TreeSamples, TrainOptions) {
    assert!(low_records.is_multiple_of(16));
    let order: Vec<_> = (0..RECORDS).map(|i| (i * 4051) % RECORDS).collect();
    let mut properties = vec![PropertyColumn::Packed(
        order.iter().map(|&i| u8::from(i >= low_records)).collect(),
    )];
    for bit in 0..4 {
        properties.push(PropertyColumn::Packed(
            order.iter().map(|&i| ((i >> bit) & 1) as u8).collect(),
        ));
    }
    properties.push(PropertyColumn::Packed(vec![0; RECORDS]));
    if wide {
        for column in &mut properties {
            *column = PropertyColumn::Wide(column.to_wide());
        }
    }
    // A costly wrong predictor forces the group split before the four label
    // bits. The odd permutation makes both child loans depend on real swaps.
    let bad = HybridUintConfig::new(4, 1, 2).unwrap().encode(1 << 30);
    let residuals = (0..2)
        .map(|predictor| {
            order
                .iter()
                .map(|&i| {
                    if predictor == usize::from(i >= low_records) {
                        ResidualToken {
                            token: (i % 16) as u8,
                            nbits: 0,
                        }
                    } else {
                        ResidualToken {
                            token: bad.token as u8,
                            nbits: bad.nbits as u8,
                        }
                    }
                })
                .collect()
        })
        .collect();
    (
        TreeSamples {
            properties,
            residuals,
            counts: order.iter().map(|&i| 1 + ((i / 16) % 3) as u16).collect(),
            thresholds: vec![vec![0]; 6],
        },
        TrainOptions {
            predictors: vec![Predictor::Zero, Predictor::Left],
            properties: (6..12).collect(),
            max_leaves: 64,
            require_complete_search: true,
            split_threshold: 1.0,
            parallel: true,
            parallel_tools: ParallelOptions {
                subtrees: true,
                balanced: true,
                small_subtrees: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
}

fn reference(samples: &TreeSamples, options: &TrainOptions) -> Tree {
    let options = TrainOptions {
        parallel: false,
        parallel_tools: ParallelOptions::default(),
        histogram_cache_bytes: 0,
        reuse_parent_histograms: false,
        ..options.clone()
    };
    Tree::new(train_nodes(&mut samples.clone(), &options).unwrap()).unwrap()
}

fn learn(
    pool: &rayon::ThreadPool,
    samples: &TreeSamples,
    options: &TrainOptions,
) -> (Tree, TrainingStats) {
    let stats = TrainingStats::default();
    let nodes = pool
        .install(|| train_nodes_with_stats(&mut samples.clone(), options, &stats))
        .unwrap();
    (Tree::new(nodes).unwrap(), stats)
}

fn assert_same_semantics(
    actual: &Tree,
    expected: &Tree,
    samples: &TreeSamples,
    options: &TrainOptions,
) {
    assert_eq!(actual.leaves(), expected.leaves());
    let mut forward = vec![None; actual.leaves()];
    let mut reverse = vec![None; expected.leaves()];
    let mut actual_mass = vec![0u64; actual.leaves()];
    let mut expected_mass = vec![0u64; expected.leaves()];
    for i in 0..samples.counts.len() {
        let mut properties = [0; 12];
        for (column, &property) in samples.properties.iter().zip(&options.properties) {
            properties[property] = column.value(i) as i32;
        }
        let (a, ap, ao, am) = actual.lookup(&properties).unwrap();
        let (e, ep, eo, em) = expected.lookup(&properties).unwrap();
        assert_eq!((ap, ao, am), (ep, eo, em), "leaf decision at record {i}");
        // Compare the complete partition, not physical node or context IDs.
        assert_eq!(*forward[a].get_or_insert(e), e, "merged baseline leaves");
        assert_eq!(*reverse[e].get_or_insert(a), a, "split baseline leaf");
        actual_mass[a] += u64::from(samples.counts[i]);
        expected_mass[e] += u64::from(samples.counts[i]);
    }
    assert!(reverse.iter().all(Option::is_some));
    for (a, e) in forward.into_iter().enumerate() {
        assert_eq!(actual_mass[a], expected_mass[e.unwrap()]);
    }
    // Every binary combination occurs in the fixture. Also exercise values on
    // both sides of its strict threshold, outside the sampled 0/1 values.
    for label in 0..32 {
        for (low, high) in [(-1, 1), (0, 2)] {
            let mut properties = [0; 12];
            for bit in 0..5 {
                properties[6 + bit] = if label & (1 << bit) == 0 { low } else { high };
            }
            let (a, ap, ao, am) = actual.lookup(&properties).unwrap();
            let (e, ep, eo, em) = expected.lookup(&properties).unwrap();
            assert_eq!((ap, ao, am), (ep, eo, em));
            assert_eq!(reverse[e], Some(a));
        }
    }
}

fn assert_fixture_root(tree: &Tree, samples: &TreeSamples, low: usize) {
    assert_eq!(tree.leaves(), 32);
    assert!(matches!(
        tree.nodes()[0],
        Node::Split {
            property: 6,
            threshold: 0,
            ..
        }
    ));
    assert_eq!(
        (0..RECORDS)
            .filter(|&i| samples.properties[0].value(i) == 0)
            .count(),
        low
    );
    assert!(RECORDS > 8192 && low.min(RECORDS - low) >= 1024);
}

#[test]
fn ma_frontier_options_default_off_and_apply_independently() {
    for tools in [
        ParallelOptions::default(),
        TrainOptions::default().parallel_tools,
        TrainOptions::variable().parallel_tools,
    ] {
        assert!(!tools.dynamic_subtrees);
        assert!(!tools.inherit_histograms);
    }
    for dynamic_subtrees in [false, true] {
        for inherit_histograms in [false, true] {
            let tools = ParallelOptions {
                subtrees: true,
                balanced: true,
                small_subtrees: true,
                dynamic_subtrees,
                inherit_histograms,
                ..Default::default()
            };
            let mut options = TrainOptions::default();
            tools.apply(&mut options);
            assert_eq!(options.parallel_tools.dynamic_subtrees, dynamic_subtrees);
            assert_eq!(
                options.parallel_tools.inherit_histograms,
                inherit_histograms
            );
            assert!(options.parallel_tools.subtrees);
            assert!(options.parallel_tools.balanced && options.parallel_tools.small_subtrees);
        }
    }
}

#[test]
fn ma_frontier_complete_search_matches_weighted_packed_and_wide_baselines() {
    for threads in [2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for wide in [false, true] {
            let (samples, mut options) = fixture(RECORDS / 2, wide);
            let expected = reference(&samples, &options);
            assert_fixture_root(&expected, &samples, RECORDS / 2);
            for dynamic in [false, true] {
                for inherit in [false, true] {
                    options.parallel_tools.dynamic_subtrees = dynamic;
                    options.parallel_tools.inherit_histograms = inherit;
                    let (actual, stats) = learn(&pool, &samples, &options);
                    assert_same_semantics(&actual, &expected, &samples, &options);
                    assert_eq!(stats.dynamic_forks.load(Ordering::Relaxed) > 0, dynamic);
                    if !inherit {
                        assert_eq!(stats.histogram_handoffs.load(Ordering::Relaxed), 0);
                    }
                }
            }
        }
    }
}

#[test]
fn ma_frontier_unbalanced_handoffs_preserve_search_and_respect_cache_budget() {
    for threads in [2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for wide in [false, true] {
            let (samples, mut options) = fixture(2048, wide);
            let expected = reference(&samples, &options);
            assert_fixture_root(&expected, &samples, 2048);
            let cache_bytes = options.histogram_cache_bytes;
            for dynamic in [false, true] {
                options.parallel_tools.dynamic_subtrees = dynamic;
                // Static grain 65536 detaches the cache-bearing 14336-record
                // child; balanced dynamic grain admits both root children.
                options.parallel_tools.balanced = dynamic;
                for inherit in [false, true] {
                    options.parallel_tools.inherit_histograms = inherit;
                    for bytes in [0, cache_bytes] {
                        options.histogram_cache_bytes = bytes;
                        let (actual, stats) = learn(&pool, &samples, &options);
                        assert_same_semantics(&actual, &expected, &samples, &options);
                        assert_eq!(stats.dynamic_forks.load(Ordering::Relaxed) > 0, dynamic);
                        if !dynamic {
                            assert!(stats.detached_subtrees.load(Ordering::Relaxed) > 0);
                        }
                        assert_eq!(
                            stats.histogram_handoffs.load(Ordering::Relaxed) > 0,
                            inherit && bytes != 0,
                            "handoff activation: threads={threads} wide={wide} dynamic={dynamic} inherit={inherit} cache={bytes}"
                        );
                        if bytes == 0 {
                            assert_eq!(stats.cache_builds.load(Ordering::Relaxed), 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ma_frontier_bounded_mode_retains_original_leaf_ceiling_and_decisions() {
    for threads in [2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (samples, mut options) = fixture(RECORDS / 2, threads == 4);
        options.budget_mode = Some(MaBudgetMode::Bounded);
        options.max_leaves = 1;
        assert!(options.require_complete_search);
        assert!(!options.complete_search());
        for cap in [1, 2, 7, 32] {
            options.max_merged_leaves = Some(cap);
            options.parallel_tools.dynamic_subtrees = false;
            options.parallel_tools.inherit_histograms = false;
            let (expected, _) = learn(&pool, &samples, &options);
            assert_eq!(expected.leaves(), cap);
            for dynamic in [false, true] {
                for inherit in [false, true] {
                    options.parallel_tools.dynamic_subtrees = dynamic;
                    options.parallel_tools.inherit_histograms = inherit;
                    let (actual, stats) = learn(&pool, &samples, &options);
                    assert_eq!(actual.leaves(), cap);
                    assert_same_semantics(&actual, &expected, &samples, &options);
                    assert_eq!(stats.dynamic_forks.load(Ordering::Relaxed), 0);
                }
            }
        }
    }
}

#[test]
fn ma_frontier_null_runner_and_disabled_search_gates_do_not_fork() {
    for threads in [2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (samples, mut options) = fixture(RECORDS / 2, threads == 4);
        let expected = reference(&samples, &options);
        options.parallel_tools.dynamic_subtrees = true;
        options.parallel_tools.inherit_histograms = true;
        let stats = TrainingStats::default();
        let actual = pool.install(|| {
            execution::serial(|| {
                assert_eq!(execution::worker_count(), 1);
                Tree::new(train_nodes_with_stats(&mut samples.clone(), &options, &stats).unwrap())
                    .unwrap()
            })
        });
        assert_same_semantics(&actual, &expected, &samples, &options);
        assert_eq!(stats.dynamic_forks.load(Ordering::Relaxed), 0);
        assert_eq!(stats.histogram_handoffs.load(Ordering::Relaxed), 0);
        assert_eq!(stats.detached_subtrees.load(Ordering::Relaxed), 0);
        assert_eq!(stats.ready_nodes.load(Ordering::Relaxed), 0);
        for (parallel, subtrees, complete) in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            options.parallel = parallel;
            options.parallel_tools.subtrees = subtrees;
            options.require_complete_search = complete;
            let (actual, stats) = learn(&pool, &samples, &options);
            assert_same_semantics(&actual, &expected, &samples, &options);
            assert_eq!(stats.dynamic_forks.load(Ordering::Relaxed), 0);
            if !parallel || !subtrees {
                assert_eq!(stats.histogram_handoffs.load(Ordering::Relaxed), 0);
            }
        }
    }
}

#[test]
fn ma_frontier_independent_calls_recover_after_budget_and_cache_errors() {
    for threads in [2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (samples, mut options) = fixture(RECORDS / 2, threads == 4);
        let expected = reference(&samples, &options);
        options.parallel_tools.dynamic_subtrees = true;
        options.parallel_tools.inherit_histograms = true;
        options.budget_mode = Some(MaBudgetMode::Strict);
        let mut working = samples.clone();
        let limited = TrainOptions {
            max_leaves: 31,
            ..options.clone()
        };
        let failed = TrainingStats::default();
        assert!(matches!(
            pool.install(|| train_nodes_with_stats(&mut working, &limited, &failed)),
            Err(Error::Limit(_))
        ));
        assert!(failed.dynamic_forks.load(Ordering::Relaxed) > 0);

        let fills = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&fills);
        let injected = TrainingStats {
            initial_cache_hook: Some(Box::new(move |property, stage| {
                if property == 0
                    && stage == InitialCacheStage::Filled
                    && seen.fetch_add(1, Ordering::Relaxed) != 0
                {
                    return Err(Error::Limit("injected frontier child cache failure"));
                }
                Ok(())
            })),
            ..Default::default()
        };
        assert!(matches!(
            pool.install(|| train_nodes_with_stats(&mut working, &options, &injected)),
            Err(Error::Limit("injected frontier child cache failure"))
        ));
        assert!(fills.load(Ordering::Relaxed) >= 2);
        assert!(injected.dynamic_forks.load(Ordering::Relaxed) > 0);

        // Reuse the physically reordered columns after each failure and again
        // after success; invocation-local permits and caches must not leak.
        for _ in 0..2 {
            let stats = TrainingStats::default();
            let actual = Tree::new(
                pool.install(|| train_nodes_with_stats(&mut working, &options, &stats))
                    .unwrap(),
            )
            .unwrap();
            assert_same_semantics(&actual, &expected, &samples, &options);
            assert!(stats.dynamic_forks.load(Ordering::Relaxed) > 0);
        }
    }
}
