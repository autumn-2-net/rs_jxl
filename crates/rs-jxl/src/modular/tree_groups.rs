// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; the complete upstream notice is in docs/PORTING_KERNELS.md.

//! enc_ma_tree_groups.h::MaTreeGroupBoundaries and enc_modular.cc's learned
//! ComputeTree/MergeTrees routes. Shards are independent learners with their own
//! quantile populations, constants and threshold scaling, not a replay of one
//! global tree. Every learner receives the original stream IDs.
//!
//! Empty supplied groups are removed before optional sharding. Missing stream
//! IDs represent zero work without allocating a dense ID-indexed array. Only a
//! single useful group is sharded; existing multiple groups remain intact.
//! Fixed-tree, JPEG/lossy routing and model/audit selection belong to the caller.
//! All selected samples share one hard budget. At most eight learners execute
//! concurrently, each retaining its own bounded histogram cache. Merge never
//! truncates a learned tree and reconstructs decoder context IDs via Tree::new.

use std::ops::Range;

use crate::execution;

use crate::{Error, Result};

use super::encode::ModularOptions;
use super::ma_train;
use super::training::{self, SampleBudget, TrainingStream};
use super::tree::{Node, Tree};

pub const MAX_SHARDS: usize = 8;
/// Resource ceiling for eight full learner budgets, not a bitstream limit.
pub const MAX_MERGED_LEAVES: usize = MAX_SHARDS * ma_train::MAX_LEAVES;

#[derive(Clone, Debug)]
pub struct TreeGroupOptions {
    /// Typed equivalent of JXL_MA_TREE_SHARDS: only 1, 2, 4, or 8.
    pub requested_shards: usize,
    pub min_work: u64,
    pub parallel: bool,
    /// Total raw leaves after merge. Entropy clustering is a separate stage.
    pub max_merged_leaves: usize,
}

impl Default for TreeGroupOptions {
    fn default() -> Self {
        Self {
            requested_shards: 1,
            min_work: 1 << 20,
            parallel: true,
            max_merged_leaves: MAX_MERGED_LEAVES,
        }
    }
}

impl TreeGroupOptions {
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.requested_shards, 1 | 2 | 4 | 8) {
            return Err(Error::InvalidInput(
                "MA tree shards must be 1, 2, 4, or 8".into(),
            ));
        }
        validate_budget(self.max_merged_leaves)
    }
}

fn validate_budget(max_leaves: usize) -> Result<()> {
    if !(1..=MAX_MERGED_LEAVES).contains(&max_leaves) {
        return Err(Error::InvalidInput("invalid merged MA leaf budget".into()));
    }
    Ok(())
}

/// Source ordered work partition: no item splitting and no minimum-work
/// violations at a cut. The unsplit range may be smaller than min_work.
/// Empty input retains the source sentinel [0, 0]; it is not a trainable shard.
pub fn boundaries(
    work: &[u64],
    requested: usize,
    min_work: u64,
    limit: usize,
) -> Result<Vec<usize>> {
    let total = work.iter().try_fold(0u64, |sum, &count| {
        sum.checked_add(count)
            .ok_or(Error::Limit("MA group work overflow"))
    })?;
    let min_work = min_work.max(1);
    let shards = (requested.min(limit) as u128).min(u128::from(total / min_work)) as usize;
    let mut result = vec![0];
    if shards > 1 {
        let mut prefix = 0;
        let mut previous = 0;
        for (i, &count) in work.iter().take(work.len().saturating_sub(1)).enumerate() {
            prefix += count;
            let target = (u128::from(total) * result.len() as u128).div_ceil(shards as u128);
            if result.len() < shards
                && u128::from(prefix) >= target
                && prefix - previous >= min_work
                && total - prefix >= min_work
            {
                result.push(i + 1);
                previous = prefix;
            }
        }
    }
    result.push(work.len());
    Ok(result)
}

fn validate_splits(splits: &[usize]) -> Result<()> {
    if splits.len() < 2
        || splits.iter().any(|&id| id > i32::MAX as usize + 1)
        || splits.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(Error::InvalidInput(
            "invalid MA stream split boundaries".into(),
        ));
    }
    Ok(())
}

/// Balanced greater-first MergeTrees. Each input root is node zero; all child
/// indices are relocated and all contexts are rebuilt, never copied as IDs.
/// Splits are original stream IDs (inclusive starts plus exclusive final end).
pub fn merge(trees: Vec<Tree>, stream_splits: &[usize], max_leaves: usize) -> Result<Tree> {
    validate_budget(max_leaves)?;
    validate_splits(stream_splits)?;
    if trees.len() != stream_splits.len() - 1 {
        return Err(Error::InvalidInput("MA tree/split count mismatch".into()));
    }
    let leaves = trees.iter().try_fold(0usize, |sum, tree| {
        sum.checked_add(tree.leaves())
            .ok_or(Error::Limit("merged MA leaf count overflow"))
    })?;
    if leaves > max_leaves {
        return Err(Error::Limit("merged MA leaf budget exceeded"));
    }
    if trees.len() == 1 {
        return Ok(trees.into_iter().next().expect("one tree"));
    }
    let node_count = leaves
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or(Error::Limit("merged MA node count overflow"))?;
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(node_count)
        .map_err(|_| Error::Limit("merged MA allocation"))?;
    let mut inputs: Vec<_> = trees
        .into_iter()
        .map(|tree| Some(tree.into_nodes()))
        .collect();
    fn append(
        inputs: &mut [Option<Vec<Node>>],
        splits: &[usize],
        begin: usize,
        end: usize,
        nodes: &mut Vec<Node>,
    ) -> usize {
        let root = nodes.len();
        if end == begin + 1 {
            for mut node in inputs[begin].take().expect("each tree merged once") {
                if let Node::Split {
                    greater,
                    less_equal,
                    ..
                } = &mut node
                {
                    *greater += root;
                    *less_equal += root;
                }
                nodes.push(node);
            }
        } else {
            let mid = (begin + end) / 2;
            let threshold = (splits[mid] - 1) as i32;
            nodes.push(Node::Split {
                property: 1,
                threshold,
                greater: 0,
                less_equal: 0,
            });
            let greater = append(inputs, splits, mid, end, nodes);
            let less_equal = append(inputs, splits, begin, mid, nodes);
            nodes[root] = Node::Split {
                property: 1,
                threshold,
                greater,
                less_equal,
            };
        }
        root
    }
    let count = inputs.len();
    append(&mut inputs, stream_splits, 0, count, &mut nodes);
    debug_assert_eq!(nodes.len(), node_count);
    Tree::new(nodes)
}

struct Plan<'a> {
    streams: Vec<TrainingStream<'a>>,
    splits: Vec<usize>,
    ranges: Vec<Range<usize>>,
}

fn has_pixels(stream: &TrainingStream<'_>) -> bool {
    stream
        .channels
        .iter()
        .any(|ch| ch.width != 0 && ch.height != 0)
}

fn stream_work(stream: &TrainingStream<'_>, max_channel_size: usize) -> Result<u64> {
    let mut total = 0u64;
    let mut stopped = false;
    for channel in stream.channels {
        let area = channel
            .width
            .checked_mul(channel.height)
            .ok_or(Error::Limit("MA group channel dimensions overflow"))?;
        if area != channel.samples.len() {
            return Err(Error::InvalidInput(
                "MA group channel dimensions/buffer mismatch".into(),
            ));
        }
        if channel.hshift >= 0
            && channel.vshift >= 0
            && (channel.width > max_channel_size || channel.height > max_channel_size)
        {
            stopped = true;
        }
        if !stopped {
            total = total
                .checked_add(area as u64)
                .ok_or(Error::Limit("MA group work overflow"))?;
        }
    }
    Ok(total)
}

fn plan<'a>(
    streams: &[TrainingStream<'a>],
    options: &ModularOptions,
    stream_splits: &[usize],
    grouping: &TreeGroupOptions,
) -> Result<Plan<'a>> {
    options.validate()?;
    grouping.validate()?;
    if streams.is_empty() {
        return Err(Error::InvalidInput("empty MA group stream domain".into()));
    }
    let mut streams: Vec<_> = streams
        .iter()
        .map(|stream| TrainingStream {
            channels: stream.channels,
            wp: stream.wp,
            stream_id: stream.stream_id,
        })
        .collect();
    streams.sort_unstable_by_key(|stream| stream.stream_id);
    let mut previous = None;
    let max_channel_size = options
        .sampling
        .as_ref()
        .map_or(usize::MAX, |p| p.max_channel_size);
    let mut work = Vec::with_capacity(streams.len());
    for stream in &streams {
        if stream.stream_id > i32::MAX as usize || previous == Some(stream.stream_id) {
            return Err(Error::InvalidInput(
                "invalid/duplicate MA group stream ID".into(),
            ));
        }
        previous = Some(stream.stream_id);
        work.push(stream_work(stream, max_channel_size)?);
    }
    let supplied = if stream_splits.is_empty() {
        vec![0, streams.last().expect("nonempty streams").stream_id + 1]
    } else {
        stream_splits.to_vec()
    };
    validate_splits(&supplied)?;
    if streams[0].stream_id < supplied[0]
        || streams.last().expect("nonempty streams").stream_id >= *supplied.last().unwrap()
    {
        return Err(Error::InvalidInput(
            "MA stream outside supplied group domain".into(),
        ));
    }
    let mut splits = Vec::new();
    let mut begin = 0;
    for pair in supplied.windows(2) {
        let end = streams.partition_point(|stream| stream.stream_id < pair[1]);
        if streams[begin..end].iter().any(has_pixels) {
            splits.push(pair[0]);
        }
        begin = end;
    }
    // ComputeTree emits no tree in this case. The caller should omit the tree
    // route, not replace missing training data with a fabricated fixed tree.
    if splits.is_empty() {
        return Err(Error::InvalidInput(
            "MA group domain has no pixel streams".into(),
        ));
    }
    splits.push(*supplied.last().unwrap());
    if grouping.requested_shards > 1 && splits.len() == 2 {
        let begin = streams.partition_point(|stream| stream.stream_id < splits[0]);
        let end = streams.partition_point(|stream| stream.stream_id < splits[1]);
        let positions = boundaries(
            &work[begin..end],
            grouping.requested_shards,
            grouping.min_work,
            MAX_SHARDS,
        )?;
        let mut sharded = vec![splits[0]];
        for &position in &positions[1..positions.len() - 1] {
            // A source cut occurs immediately after positive work, before any
            // omitted zero-work IDs. Do not substitute the next existing ID.
            sharded.push(streams[begin + position - 1].stream_id + 1);
        }
        sharded.push(splits[1]);
        splits = sharded;
    }
    if splits.len() - 1 > grouping.max_merged_leaves {
        return Err(Error::Limit("MA groups exceed merged leaf budget"));
    }
    let mut ranges = Vec::with_capacity(splits.len() - 1);
    for pair in splits.windows(2) {
        let mut begin = streams.partition_point(|stream| stream.stream_id < pair[0]);
        let mut end = streams.partition_point(|stream| stream.stream_id < pair[1]);
        while begin < end && !has_pixels(&streams[begin]) {
            begin += 1;
        }
        while begin < end && !has_pixels(&streams[end - 1]) {
            end -= 1;
        }
        if begin == end {
            return Err(Error::Encoding(
                "MA partition produced an empty learner".into(),
            ));
        }
        ranges.push(begin..end);
    }
    Ok(Plan {
        streams,
        splits,
        ranges,
    })
}

/// ComputeTree's learned group route. Empty stream_splits selects one initial
/// whole-domain range. The caller must gate fixed/JPEG/lossy/model/audit routes.
/// options.max_tree_leaves applies per learner; max_merged_leaves applies to the
/// result; options.sample_limit is shared globally, never multiplied by shards.
pub(super) fn learn(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    stream_splits: &[usize],
    grouping: &TreeGroupOptions,
) -> Result<Tree> {
    learn_with_gather(streams, options, stream_splits, grouping, None)
}

pub(super) fn learn_with_gather(
    streams: &[TrainingStream<'_>],
    options: &ModularOptions,
    stream_splits: &[usize],
    grouping: &TreeGroupOptions,
    gather: Option<&super::sample::GatherOptions>,
) -> Result<Tree> {
    let plan = plan(streams, options, stream_splits, grouping)?;
    let budget = SampleBudget::new(options.sample_limit);
    let mut trees = Vec::with_capacity(plan.ranges.len());
    let process = |range: &Range<usize>| {
        if let Some(gather) = gather {
            return training::learn_with_budget_and_gather(
                &plan.streams[range.clone()],
                options,
                Some(&budget),
                plan.ranges.len() == 1,
                Some(gather),
            );
        }
        training::learn_with_budget(
            &plan.streams[range.clone()],
            options,
            Some(&budget),
            plan.ranges.len() == 1,
        )
    };
    // Also bounds concurrent histogram caches for caller-supplied group routes.
    for wave in plan.ranges.chunks(MAX_SHARDS) {
        let results: Vec<Result<Tree>> = if grouping.parallel && wave.len() > 1 {
            execution::map(wave, process)
        } else {
            wave.iter().map(&process).collect()
        };
        for result in results {
            trees.push(result?);
        }
    }
    merge(trees, &plan.splits, grouping.max_merged_leaves)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::super::{
        group::Channel,
        predictor::Predictor,
        sample::{SamplingPolicy, SourcePolicy},
        weighted,
    };
    use super::*;

    fn channel(values: Vec<i32>) -> Channel {
        Channel {
            width: values.len(),
            height: 1,
            hshift: 0,
            vshift: 0,
            samples: values,
        }
    }

    fn stream(channels: &[Channel], stream_id: usize) -> TrainingStream<'_> {
        TrainingStream {
            channels,
            stream_id,
            wp: weighted::Header::default(),
        }
    }

    fn options() -> ModularOptions {
        let mut sampling = SamplingPolicy::from_source(&SourcePolicy::default()).unwrap();
        sampling.nb_repeats = 1.0;
        sampling.properties = vec![1, 3];
        sampling.max_property_values = 256;
        sampling.node_threshold = 0.0;
        ModularOptions {
            predictor: Predictor::Gradient,
            max_tree_leaves: 64,
            sampling: Some(sampling),
            ..ModularOptions::default()
        }
    }

    fn grouping() -> TreeGroupOptions {
        TreeGroupOptions {
            requested_shards: 2,
            min_work: 1,
            parallel: false,
            ..TreeGroupOptions::default()
        }
    }

    fn signature(tree: &Tree) -> Vec<(usize, u32)> {
        tree.tokens()
            .iter()
            .map(|token| (token.context, token.value))
            .collect()
    }

    fn fixture() -> Vec<Vec<Channel>> {
        (0..4)
            .map(|group| {
                vec![channel(
                    (1..=64)
                        .map(|x| (if x <= 32 { x } else { 32 + 4 * (x - 32) }) + group * 32)
                        .collect(),
                )]
            })
            .collect()
    }

    #[test]
    fn source_work_boundaries_match_targets_minimums_and_limits() {
        let unit = 1 << 20;
        assert_eq!(
            boundaries(&[unit; 8], 8, unit, 8).unwrap(),
            (0..=8).collect::<Vec<_>>()
        );
        assert_eq!(boundaries(&[unit; 8], 8, unit, 2).unwrap(), [0, 4, 8]);
        assert_eq!(boundaries(&[6, 1, 1, 6], 4, 3, 8).unwrap(), [0, 1, 4]);
        assert_eq!(boundaries(&[0, 4, 0, 4, 0], 2, 4, 8).unwrap(), [0, 2, 5]);
        assert_eq!(boundaries(&[5, 5], 8, 10, 8).unwrap(), [0, 2]);
        assert_eq!(boundaries(&[100], 8, 1, 8).unwrap(), [0, 1]);
        assert_eq!(boundaries(&[0; 5], 8, 1, 8).unwrap(), [0, 5]);
        assert_eq!(boundaries(&[], 8, 1, 8).unwrap(), [0, 0]);
        assert_eq!(boundaries(&[1, 1], 2, 0, 8).unwrap(), [0, 1, 2]);
        assert_eq!(boundaries(&[u64::MAX], usize::MAX, 1, 8).unwrap(), [0, 1]);
        assert!(matches!(
            boundaries(&[u64::MAX, 1], 2, 1, 8),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn sparse_ids_retain_dense_source_cut_positions_without_dense_allocation() {
        let planes = [channel(vec![0, 1, 2, 3])];
        let streams = [
            stream(&planes, 31),
            stream(&[], 30),
            stream(&planes, 0),
            stream(&planes, 21),
        ];
        let grouping = TreeGroupOptions {
            requested_shards: 4,
            min_work: 4,
            ..grouping()
        };
        let sparse = plan(&streams, &options(), &[], &grouping).unwrap();
        let mut dense = vec![0; 32];
        for id in [0, 21, 31] {
            dense[id] = 4;
        }
        assert_eq!(sparse.splits, boundaries(&dense, 4, 4, 8).unwrap());
        assert_eq!(sparse.splits, [0, 1, 22, 32]);
        assert_eq!(
            sparse
                .ranges
                .iter()
                .map(|range| sparse.streams[range.start].stream_id)
                .collect::<Vec<_>>(),
            [0, 21, 31]
        );
        let streams = [stream(&planes, i32::MAX as usize), stream(&planes, 0)];
        let plan = plan(&streams, &options(), &[], &grouping).unwrap();
        assert_eq!(plan.splits, [0, 1, i32::MAX as usize + 1]);
        assert_eq!(plan.streams.len(), 2);
    }

    #[test]
    fn existing_useful_groups_preserve_routes_and_trim_only_empty_edges() {
        let planes = [channel(vec![0, 1, 2, 3])];
        let streams = [
            stream(&[], 0),
            stream(&planes, 12),
            stream(&[], 15),
            stream(&planes, 18),
            stream(&planes, 25),
            stream(&[], 29),
            stream(&[], 37),
        ];
        let grouping = TreeGroupOptions {
            requested_shards: 8,
            ..grouping()
        };
        let plan = plan(&streams, &options(), &[0, 10, 20, 30, 40], &grouping).unwrap();
        assert_eq!(plan.splits, [10, 20, 40]);
        assert_eq!(
            plan.streams[plan.ranges[0].clone()]
                .iter()
                .map(|s| s.stream_id)
                .collect::<Vec<_>>(),
            [12, 15, 18]
        );
        assert_eq!(
            plan.streams[plan.ranges[1].clone()]
                .iter()
                .map(|s| s.stream_id)
                .collect::<Vec<_>>(),
            [25]
        );
    }

    #[test]
    fn work_counts_metadata_width_one_and_stops_at_oversized_nonmetadata() {
        let channels = [
            Channel {
                hshift: -1,
                ..channel(vec![0; 8])
            },
            channel(vec![0; 8]),
            channel(vec![0; 2]),
        ];
        assert_eq!(stream_work(&stream(&channels, 21), 4).unwrap(), 8);
        let channels = [Channel {
            width: 1,
            height: 5,
            ..channel(vec![0; 5])
        }];
        assert_eq!(stream_work(&stream(&channels, 21), 4).unwrap(), 0);
        assert_eq!(stream_work(&stream(&channels, 21), 8).unwrap(), 5);
        let malformed = [
            channel(vec![0; 8]),
            Channel {
                samples: vec![0],
                ..channel(vec![0; 2])
            },
        ];
        assert!(stream_work(&stream(&malformed, 21), 4).is_err());
    }

    fn branch_tree(offset: i32) -> Tree {
        Tree::new(vec![
            Node::Split {
                property: 3,
                threshold: 3,
                greater: 4,
                less_equal: 1,
            },
            Node::Split {
                property: 0,
                threshold: 0,
                greater: 2,
                less_equal: 3,
            },
            Node::Leaf {
                predictor: Predictor::Gradient,
                offset,
                multiplier: 3,
            },
            Node::Leaf {
                predictor: Predictor::Zero,
                offset: -offset,
                multiplier: 2,
            },
            Node::Leaf {
                predictor: Predictor::Weighted,
                offset: offset + 1,
                multiplier: 1,
            },
        ])
        .unwrap()
    }

    #[test]
    fn merge_relocates_children_preserves_leaf_parameters_and_rebuilds_contexts() {
        let trees: Vec<_> = [7, 13, 19].into_iter().map(branch_tree).collect();
        let merged = merge(trees.clone(), &[0, 22, 1_000_001, i32::MAX as usize + 1], 9).unwrap();
        assert_eq!(merged.leaves(), 9);
        assert!(matches!(
            merged.nodes()[0],
            Node::Split {
                property: 1,
                threshold: 21,
                ..
            }
        ));
        let mut contexts = BTreeMap::new();
        for (group, ids) in [[0, 21], [22, 1_000_000], [1_000_001, i32::MAX]]
            .into_iter()
            .enumerate()
        {
            for id in ids {
                for channel in 0..2 {
                    for x in [0, 3, 4, 7] {
                        let properties = [channel, id, 0, x];
                        let expected = trees[group].lookup(&properties).unwrap();
                        let actual = merged.lookup(&properties).unwrap();
                        assert_eq!(
                            (actual.1, actual.2, actual.3),
                            (expected.1, expected.2, expected.3)
                        );
                        if let Some(previous) = contexts.insert((group, expected.0), actual.0) {
                            assert_eq!(previous, actual.0);
                        }
                    }
                }
            }
        }
        assert_eq!(
            contexts.values().copied().collect::<BTreeSet<_>>(),
            (0..9).collect::<BTreeSet<_>>()
        );
        assert_eq!(
            signature(&merged),
            signature(&Tree::new(merged.clone().into_nodes()).unwrap())
        );
        assert_eq!(
            signature(&merge(vec![trees[0].clone()], &[0, 22], 3).unwrap()),
            signature(&trees[0])
        );
    }

    #[test]
    fn grouped_learning_equals_independent_source_learners_then_stream_id_merge() {
        let planes = fixture();
        let streams: Vec<_> = planes
            .iter()
            .zip([0, 21, 1_000_000, 1_000_021])
            .map(|(planes, id)| stream(planes, id))
            .collect();
        let options = options();
        let grouping = grouping();
        let plan = plan(&streams, &options, &[], &grouping).unwrap();
        assert_eq!(plan.splits, [0, 22, 1_000_022]);
        let separate: Vec<_> = plan
            .ranges
            .iter()
            .map(|range| training::learn(&plan.streams[range.clone()], &options).unwrap())
            .collect();
        assert!(separate.iter().all(|tree| tree.leaves() > 1));
        let expected = merge(separate, &plan.splits, grouping.max_merged_leaves).unwrap();
        let actual = learn(&streams, &options, &[], &grouping).unwrap();
        assert_eq!(signature(&actual), signature(&expected));
        assert_eq!(actual.predictors(), [Predictor::Gradient]);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let parallel = pool
            .install(|| {
                learn(
                    &streams,
                    &options,
                    &[],
                    &TreeGroupOptions {
                        parallel: true,
                        ..grouping.clone()
                    },
                )
            })
            .unwrap();
        assert_eq!(signature(&actual), signature(&parallel));
        let unsharded = learn(&streams, &options, &[], &TreeGroupOptions::default()).unwrap();
        assert_eq!(
            signature(&unsharded),
            signature(&training::learn(&streams, &options).unwrap())
        );
    }

    #[test]
    fn global_sample_and_merge_budgets_do_not_truncate_complete_learners() {
        let planes = fixture();
        let streams: Vec<_> = planes
            .iter()
            .zip([0, 21, 22, 23])
            .map(|(planes, id)| stream(planes, id))
            .collect();
        let mut options = options();
        // The merge must still use property 1 when it is not a learned feature.
        options.sampling.as_mut().unwrap().properties = vec![3];
        options.sample_limit = 256;
        let grouping = grouping();
        assert!(learn(&streams, &options, &[], &grouping).is_ok());
        options.sample_limit = 255;
        assert!(matches!(
            learn(&streams, &options, &[], &grouping),
            Err(Error::Limit(_))
        ));
        options.sample_limit = 256;
        assert!(matches!(
            learn(
                &streams,
                &options,
                &[],
                &TreeGroupOptions {
                    max_merged_leaves: 2,
                    ..grouping.clone()
                }
            ),
            Err(Error::Limit(_))
        ));
        options.max_tree_leaves = 1;
        assert!(matches!(
            learn(&streams, &options, &[], &grouping),
            Err(Error::Limit(_))
        ));
        options.sampling.as_mut().unwrap().node_threshold = 1.0e6;
        let finished = learn(
            &streams,
            &options,
            &[],
            &TreeGroupOptions {
                max_merged_leaves: 2,
                ..grouping
            },
        )
        .unwrap();
        assert_eq!(finished.leaves(), 2);
        assert_eq!(finished.predictors(), [Predictor::Gradient]);
    }

    #[test]
    fn training_dispatch_uses_optional_groups_and_explicit_routes_without_recursion() {
        let planes = fixture();
        let streams: Vec<_> = planes
            .iter()
            .zip([0, 21, 22, 23])
            .map(|(planes, id)| stream(planes, id))
            .collect();
        let mut options = options();
        let groups = grouping();
        let expected = learn(&streams, &options, &[], &groups).unwrap();
        options.tree_groups = Some(groups);
        assert_eq!(
            signature(&training::learn(&streams, &options).unwrap()),
            signature(&expected)
        );
        options.tree_group_splits = vec![0, 22, 24];
        assert_eq!(
            signature(&training::learn(&streams, &options).unwrap()),
            signature(&expected)
        );
        options.tree_groups = None;
        options.tree_group_splits.clear();
        assert_eq!(
            signature(&training::learn(&streams, &options).unwrap()),
            signature(&training::learn_with_budget(&streams, &options, None, true).unwrap())
        );
    }

    #[test]
    fn independent_constant_proofs_keep_original_stream_ids_and_offsets() {
        let first = [channel(vec![-3; 64])];
        let last = [channel(vec![7; 64])];
        let streams = [stream(&last, 1_000_000), stream(&first, 21)];
        let mut options = options();
        options.source_constant_channels = true;
        options.max_tree_leaves = 1;
        options.sample_limit = 1;
        let tree = learn(&streams, &options, &[], &grouping()).unwrap();
        assert_eq!(tree.leaves(), 2);
        for (id, value) in [(21, -3), (1_000_000, 7)] {
            let (_, predictor, offset, multiplier) = tree.lookup(&[0, id]).unwrap();
            assert_eq!((predictor, offset, multiplier), (Predictor::Zero, value, 1));
        }
    }

    #[test]
    fn rejects_invalid_domains_routes_and_merge_budgets() {
        let planes = [channel(vec![0, 1])];
        let options = options();
        let grouping = grouping();
        assert!(learn(&[], &options, &[], &grouping).is_err());
        assert!(learn(&[stream(&[], 0)], &options, &[], &grouping).is_err());
        assert!(
            learn(
                &[stream(&planes, 21), stream(&planes, 21)],
                &options,
                &[],
                &grouping
            )
            .is_err()
        );
        assert!(learn(&[stream(&planes, 21)], &options, &[0, 21], &grouping).is_err());
        assert!(learn(&[stream(&planes, 21)], &options, &[0, 22, 22], &grouping).is_err());
        assert!(
            TreeGroupOptions {
                requested_shards: 3,
                ..grouping.clone()
            }
            .validate()
            .is_err()
        );
        for max_leaves in [0, MAX_MERGED_LEAVES + 1] {
            assert!(merge(vec![branch_tree(0)], &[0, 1], max_leaves).is_err());
        }
        assert!(merge(vec![branch_tree(0)], &[0, 1, 2], 10).is_err());
        assert!(matches!(
            merge(vec![branch_tree(0), branch_tree(1)], &[0, 21, 22], 5),
            Err(Error::Limit(_))
        ));
        let invalid_domain = Tree::new(vec![
            Node::Split {
                property: 1,
                threshold: 30,
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
        // A caller-supplied subtree may not introduce an unreachable group split.
        assert!(merge(vec![invalid_domain, branch_tree(0)], &[0, 21, 22], 8).is_err());
    }
}
