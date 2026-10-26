// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.

use super::super::{
    ConstantChannel, MAX_LEAVES, MAX_TREE_LEAVES, MAX_TREE_NODES, ParallelOptions, ResidualToken,
    Tree, global_tree_leaf_limit, local_tree_leaf_limit, train_nodes_with_stats,
    train_with_constants,
};
use super::*;
use std::sync::atomic::Ordering;

fn fixture(channels: usize, bits: usize, records: usize) -> (TreeSamples, TrainOptions) {
    let mut properties = vec![PropertyColumn::Wide(
        (0..records).map(|i| (i % channels) as u32).collect(),
    )];
    for bit in 0..bits {
        properties.push(PropertyColumn::Packed(
            (0..records)
                .map(|i| (((i / channels) >> bit) & 1) as u8)
                .collect(),
        ));
    }
    let residuals: Vec<_> = (0..records)
        .map(|i| {
            let token = ((i / channels) % (1 << bits)) as u8;
            ResidualToken {
                token,
                nbits: if token < 16 { 0 } else { 1 + (token - 16) / 8 },
            }
        })
        .collect();
    let mut thresholds = vec![(0..channels.saturating_sub(1) as i32).collect()];
    thresholds.extend((0..bits).map(|_| vec![0]));
    (
        TreeSamples {
            properties,
            residuals: vec![residuals.clone(), residuals],
            counts: vec![1; records],
            thresholds,
        },
        TrainOptions {
            predictors: vec![Predictor::Zero, Predictor::Left],
            properties: std::iter::once(0).chain(6..6 + bits).collect(),
            static_property_range: Some([[0, channels as u32], [0, 1]]),
            max_leaves: 2,
            max_merged_leaves: Some(channels * (1 << bits)),
            split_threshold: 1.0,
            require_complete_search: true,
            parallel: false,
            parallel_tools: ParallelOptions {
                subtrees: true,
                small_subtrees: true,
                balanced: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
}

fn signature(nodes: Vec<Node>) -> Vec<(usize, u32)> {
    Tree::new(nodes)
        .unwrap()
        .tokens()
        .iter()
        .map(|token| (token.context, token.value))
        .collect()
}

#[test]
fn ma_budget_decoder_bounds_count_nodes_not_leaves_and_do_not_overflow() {
    assert_eq!(global_tree_leaf_limit(15, 1, 1), 512);
    assert_eq!(global_tree_leaf_limit(16, 1, 1), 513);
    assert_eq!(global_tree_leaf_limit(32, 1, 1), 513);
    assert_eq!(global_tree_leaf_limit(1024, 1024, 3), 98_816);
    assert_eq!(
        global_tree_leaf_limit(usize::MAX, usize::MAX, usize::MAX),
        MAX_TREE_LEAVES
    );
    assert_eq!(local_tree_leaf_limit(0), 512);
    assert_eq!(local_tree_leaf_limit(1), 513);
    assert_eq!(local_tree_leaf_limit(2), 513);
    assert_eq!(local_tree_leaf_limit(usize::MAX), 1 << 19);
    assert_eq!(merged_node_count(3, 3, 3).unwrap(), 5);
    assert!(merged_node_count(3, 3, 2).is_err());
    assert_eq!(
        merged_node_count(MAX_TREE_NODES - 1, 1, MAX_TREE_LEAVES).unwrap(),
        MAX_TREE_NODES - 1
    );
    assert!(matches!(
        merged_node_count(MAX_TREE_NODES - 1, 3, usize::MAX),
        Err(Error::Limit("merged MA tree exceeds decoder node limit"))
    ));
    assert!(merged_node_count(usize::MAX, 3, usize::MAX).is_err());
    assert!(merged_node_count(0, 1, 1).is_err());
    assert!(merged_node_count(1, 0, 1).is_err());
}

#[test]
fn ma_budget_local_allowances_reserve_all_other_roots_without_equal_quotas() {
    assert_eq!(graft_leaf_allowance(1, 8).unwrap(), 8);
    assert_eq!(graft_leaf_allowance(7, 8).unwrap(), 5);
    assert_eq!(graft_leaf_allowance(7, 4).unwrap(), 1);
    assert!(graft_leaf_allowance(7, 3).is_err());
    assert!(graft_leaf_allowance(0, 4).is_err());
    assert!(graft_leaf_allowance(1, 0).is_err());
}

#[test]
fn ma_budget_complete_serial_and_detached_search_share_one_global_ceiling() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (samples, mut options) = fixture(1, 5, 16_384);
    options.validate().unwrap();
    let expected =
        train_nodes_with_stats(&mut samples.clone(), &options, &TrainingStats::default()).unwrap();
    assert_eq!(Tree::new(expected.clone()).unwrap().leaves(), 32);
    let expected = signature(expected);
    options.parallel = true;
    let stats = TrainingStats::default();
    let actual = pool
        .install(|| train_nodes_with_stats(&mut samples.clone(), &options, &stats))
        .unwrap();
    assert!(stats.detached_subtrees.load(Ordering::Relaxed) > 0);
    assert_eq!(signature(actual), expected);
    for global in [None, Some(31)] {
        options.max_merged_leaves = global;
        for parallel in [false, true] {
            options.parallel = parallel;
            assert!(matches!(
                pool.install(|| train_nodes_with_stats(
                    &mut samples.clone(),
                    &options,
                    &TrainingStats::default(),
                )),
                Err(Error::Limit(_))
            ));
        }
    }
}

#[test]
fn ma_budget_channel_roots_and_nested_subtrees_do_not_reset_the_global_limit() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (samples, mut options) = fixture(2, 5, 32_768);
    options.channel_tree = ChannelTreeMode::Channels;
    let expected =
        train_nodes_with_stats(&mut samples.clone(), &options, &TrainingStats::default()).unwrap();
    assert_eq!(Tree::new(expected.clone()).unwrap().leaves(), 64);
    let expected = signature(expected);
    options.parallel = true;
    let stats = TrainingStats::default();
    let actual = pool
        .install(|| train_nodes_with_stats(&mut samples.clone(), &options, &stats))
        .unwrap();
    assert_eq!(stats.channel_roots.load(Ordering::Relaxed), 2);
    assert!(stats.detached_subtrees.load(Ordering::Relaxed) > 0);
    assert_eq!(signature(actual), expected);
    options.max_merged_leaves = Some(63);
    for parallel in [false, true] {
        options.parallel = parallel;
        assert!(matches!(
            pool.install(|| train_nodes_with_stats(
                &mut samples.clone(),
                &options,
                &TrainingStats::default(),
            )),
            Err(Error::Limit(_))
        ));
    }
}

#[test]
fn ma_budget_constant_suffix_is_reserved_in_the_global_not_legacy_budget() {
    let (samples, mut options) = fixture(2, 3, 8192);
    options.channel_tree = ChannelTreeMode::Channels;
    options.static_property_range = Some([[0, 3], [0, 1]]);
    options.constant_channels = vec![ConstantChannel {
        channel: 2,
        value: 7,
    }];
    options.max_merged_leaves = Some(17);
    options.validate().unwrap();
    let tree = train_with_constants(&options, |learning| {
        assert_eq!(learning.max_leaves, 2);
        assert_eq!(learning.max_merged_leaves, Some(16));
        assert!(learning.constant_channels.is_empty());
        train_nodes_with_stats(&mut samples.clone(), learning, &TrainingStats::default())
    })
    .unwrap();
    assert_eq!(tree.leaves(), 17);
    let (_, predictor, offset, multiplier) = tree.lookup(&[2; 9]).unwrap();
    assert_eq!((predictor, offset, multiplier), (Predictor::Zero, 7, 1));
    options.max_merged_leaves = Some(16);
    assert!(
        train_with_constants(&options, |learning| {
            train_nodes_with_stats(&mut samples.clone(), learning, &TrainingStats::default())
        })
        .is_err()
    );
}

fn balanced_tree(leaves: usize) -> Vec<Node> {
    fn add(nodes: &mut Vec<Node>, begin: usize, end: usize) -> usize {
        let index = nodes.len();
        nodes.push(leaf(Predictor::Zero));
        if end - begin > 1 {
            let mid = begin + (end - begin) / 2;
            let greater = add(nodes, mid, end);
            let less_equal = add(nodes, begin, mid);
            nodes[index] = Node::Split {
                property: 6,
                threshold: mid as i32 - 1,
                greater,
                less_equal,
            };
        }
        index
    }
    let mut nodes = Vec::new();
    add(&mut nodes, 0, leaves);
    nodes
}

#[test]
fn ma_budget_merge_above_legacy_limit_preserves_explicit_ceiling_and_contexts() {
    const FORMER_LEARNER_LIMIT: usize = 16_384;
    let part_leaves = FORMER_LEARNER_LIMIT / 2 + 1;
    let learned = balanced_tree(part_leaves);
    let mut nodes = vec![leaf(Predictor::Zero)];
    let (high, low) = split(&mut nodes, 0, 0, 0, Predictor::Zero, 2).unwrap();
    graft(&mut nodes, high, learned.clone(), FORMER_LEARNER_LIMIT).unwrap();
    let before = signature(nodes.clone());
    assert!(graft(&mut nodes, low, learned.clone(), FORMER_LEARNER_LIMIT).is_err());
    assert_eq!(signature(nodes.clone()), before);
    graft(&mut nodes, low, learned, part_leaves * 2).unwrap();
    let tree = Tree::new(nodes).unwrap();
    assert_eq!(tree.leaves(), FORMER_LEARNER_LIMIT + 2);
    let mut contexts = std::collections::BTreeSet::new();
    for channel in 0..2 {
        for value in 0..part_leaves {
            let mut properties = [0; 7];
            properties[0] = channel;
            properties[6] = value as i32;
            assert!(contexts.insert(tree.lookup(&properties).unwrap().0));
        }
    }
    assert_eq!(contexts.len(), FORMER_LEARNER_LIMIT + 2);
}

#[test]
fn ma_budget_public_options_validate_legacy_and_global_limits_independently() {
    for global in [0, MAX_TREE_LEAVES + 1, usize::MAX] {
        let options = TrainOptions {
            max_merged_leaves: Some(global),
            ..Default::default()
        };
        assert!(options.validate().is_err());
    }
    for legacy in [0, MAX_LEAVES + 1, usize::MAX] {
        let options = TrainOptions {
            max_leaves: legacy,
            max_merged_leaves: Some(MAX_TREE_LEAVES),
            ..Default::default()
        };
        assert!(options.validate().is_err());
    }
    TrainOptions {
        max_merged_leaves: Some(MAX_TREE_LEAVES),
        ..Default::default()
    }
    .validate()
    .unwrap();
}
