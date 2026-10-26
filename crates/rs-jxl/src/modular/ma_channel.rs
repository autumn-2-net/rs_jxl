// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; retained notice in docs/PORTING_KERNELS.md.
//! enc_ma.cc::FindBestSplit channel/row partition wrapper, frozen b866b3d.

#[cfg(test)]
use super::TreeSamples;

use super::{
    Node, Predictor, PropertyColumn, PropertyValue, SampleStorage, Samples, ScratchPool,
    StaticPropRange, TrainOptions, TrainingNode, TrainingStats, leaf, partition_node,
    train_roots_with_layout,
};
use crate::{Error, Result};

#[cfg(test)]
#[path = "ma_channel_budget_tests.rs"]
mod budget_tests;

/// Resolved source experiment. Metadata/final-only/probe eligibility belongs
/// to the caller; ineligible calls must resolve to Disabled before training.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChannelTreeMode {
    #[default]
    Disabled,
    /// JXL_MA_CHANNEL_TREE=1.
    Channels,
    /// JXL_MA_CHANNEL_TREE=2: at most one row split per channel partition.
    ChannelsAndRows,
    /// JXL_MA_CHANNEL_TREE=3: at most two levels of row splits.
    ChannelsAndRowsTwice,
}

impl ChannelTreeMode {
    fn row_depth(self) -> usize {
        match self {
            Self::Disabled | Self::Channels => 0,
            Self::ChannelsAndRows => 1,
            Self::ChannelsAndRowsTwice => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Part {
    node: usize,
    begin: usize,
    end: usize,
    lo: usize,
    hi: usize,
    range: StaticPropRange,
    depth: usize,
}

struct Plan {
    nodes: Vec<Node>,
    jobs: Vec<Part>,
    indices: Vec<usize>,
    channels: usize,
}

fn record_bins(
    column: &PropertyColumn<
        impl std::ops::DerefMut<Target = [u8]>,
        impl std::ops::DerefMut<Target = [u32]>,
    >,
    begin: usize,
    end: usize,
    bins: usize,
) -> Result<Vec<usize>> {
    fn collect<T: PropertyValue>(values: &[T], counts: &mut [usize]) {
        for &value in values {
            counts[value.index()] += 1;
        }
    }
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(bins)
        .map_err(|_| Error::Limit("MA channel partition histogram allocation"))?;
    counts.resize(bins, 0);
    match column {
        PropertyColumn::Packed(values) => collect(&values[begin..end], &mut counts),
        PropertyColumn::Wide(values) => collect(&values[begin..end], &mut counts),
    }
    Ok(counts)
}

fn row_cut(bins: &[usize], records: usize) -> Option<(usize, usize)> {
    if records < 16384 {
        return None;
    }
    let mut lower = 0;
    for (quant, &count) in bins.iter().take(bins.len().saturating_sub(1)).enumerate() {
        lower += count;
        if lower >= records.div_ceil(2) {
            return (lower >= 4096 && records - lower >= 4096).then_some((quant, lower));
        }
    }
    None
}

fn split(
    nodes: &mut Vec<Node>,
    node: usize,
    property: usize,
    threshold: i32,
    predictor: Predictor,
    max_leaves: usize,
) -> Result<(usize, usize)> {
    if nodes.len().div_ceil(2) >= max_leaves {
        return Err(Error::Limit(
            "required MA channel partitions exceed leaf budget",
        ));
    }
    nodes
        .try_reserve(2)
        .map_err(|_| Error::Limit("MA channel skeleton allocation"))?;
    let greater = nodes.len();
    let less_equal = greater + 1;
    nodes.push(leaf(predictor));
    nodes.push(leaf(predictor));
    nodes[node] = Node::Split {
        property,
        threshold,
        greater,
        less_equal,
    };
    Ok((greater, less_equal))
}

fn plan(samples: &mut Samples<impl SampleStorage>, options: &TrainOptions) -> Result<Option<Plan>> {
    let size = samples.counts.len();
    if options.channel_tree == ChannelTreeMode::Disabled
        || !options.multiplier_info.is_empty()
        || size < 8192
        || options.properties.first() != Some(&0)
        || samples.thresholds[0].len() >= 32
    {
        return Ok(None);
    }
    let counts = record_bins(
        &samples.properties[0],
        0,
        size,
        samples.thresholds[0].len() + 1,
    )?;
    let occupied: Vec<_> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| (n != 0).then_some(i))
        .collect();
    if occupied.len() <= 1 {
        return Ok(None);
    }
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(size)
        .map_err(|_| Error::Limit("MA channel indices allocation"))?;
    indices.extend(0..size);
    let initial = Part {
        node: 0,
        begin: 0,
        end: size,
        lo: 0,
        hi: occupied.len(),
        range: options.static_property_range.unwrap_or([[0, u32::MAX]; 2]),
        depth: 0,
    };
    let predictor = options.predictors[0];
    let spatial = options
        .properties
        .iter()
        .position(|&property| property == 2);
    let mut nodes = vec![leaf(predictor)];
    let mut pending = vec![initial];
    let mut jobs = Vec::new();
    while let Some(part) = pending.pop() {
        if part.hi - part.lo == 1 {
            if part.depth < options.channel_tree.row_depth()
                && part.end - part.begin >= 16384
                && let Some(property) = spatial
            {
                let bins = record_bins(
                    &samples.properties[property],
                    part.begin,
                    part.end,
                    samples.thresholds[property].len() + 1,
                )?;
                if let Some((quant, lower)) = row_cut(&bins, part.end - part.begin) {
                    let cut = part.begin + lower;
                    let (greater, less_equal) = split(
                        &mut nodes,
                        part.node,
                        2,
                        samples.thresholds[property][quant],
                        predictor,
                        options.max_leaves,
                    )?;
                    partition_node::<true>(
                        samples,
                        &mut indices[part.begin..part.end],
                        property,
                        quant,
                        lower,
                    );
                    pending.push(Part {
                        node: less_equal,
                        end: cut,
                        depth: part.depth + 1,
                        ..part
                    });
                    pending.push(Part {
                        node: greater,
                        begin: cut,
                        depth: part.depth + 1,
                        ..part
                    });
                    continue;
                }
            }
            jobs.push(part);
            continue;
        }
        let mid = part.lo + (part.hi - part.lo) / 2;
        let lower: usize = occupied[part.lo..mid].iter().map(|&bin| counts[bin]).sum();
        let cut = part.begin + lower;
        let quant = occupied[mid - 1];
        let threshold = samples.thresholds[0][quant];
        let boundary = u32::try_from(threshold)
            .ok()
            .and_then(|n| n.checked_add(1))
            .filter(|&n| n < part.range[0][1])
            .ok_or_else(|| Error::Encoding("MA channel threshold outside static range".into()))?;
        let (greater, less_equal) = split(
            &mut nodes,
            part.node,
            0,
            threshold,
            predictor,
            options.max_leaves,
        )?;
        partition_node::<true>(samples, &mut indices[part.begin..part.end], 0, quant, lower);
        let mut low = part.range;
        let mut high = part.range;
        low[0][1] = boundary;
        high[0][0] = boundary;
        pending.push(Part {
            node: less_equal,
            end: cut,
            hi: mid,
            range: low,
            ..part
        });
        pending.push(Part {
            node: greater,
            begin: cut,
            lo: mid,
            range: high,
            ..part
        });
    }
    Ok(Some(Plan {
        nodes,
        jobs,
        indices,
        channels: occupied.len(),
    }))
}

pub(super) fn graft(
    nodes: &mut Vec<Node>,
    destination: usize,
    learned: Vec<Node>,
    max_leaves: usize,
) -> Result<()> {
    merged_node_count(nodes.len(), learned.len(), max_leaves)?;
    nodes
        .try_reserve(learned.len() - 1)
        .map_err(|_| Error::Limit("MA channel graft allocation"))?;
    let offset = nodes.len() - 1;
    for (i, mut node) in learned.into_iter().enumerate() {
        if let Node::Split {
            greater,
            less_equal,
            ..
        } = &mut node
        {
            *greater += offset;
            *less_equal += offset;
        }
        if i == 0 {
            nodes[destination] = node;
        } else {
            nodes.push(node);
        }
    }
    Ok(())
}

fn merged_node_count(current: usize, learned: usize, max_leaves: usize) -> Result<usize> {
    if current == 0 || learned == 0 {
        return Err(Error::Encoding("empty MA graft tree".into()));
    }
    let final_size = current
        .checked_add(learned - 1)
        .ok_or(Error::Limit("MA merged node count overflow"))?;
    if final_size > super::MAX_TREE_NODES {
        return Err(Error::Limit("merged MA tree exceeds decoder node limit"));
    }
    if final_size.div_ceil(2) > max_leaves {
        crate::profile::note(
            "ma.merge_limit",
            format_args!(
                "current_nodes={current} learned_nodes={learned} merged_nodes={final_size} max_leaves={max_leaves}"
            ),
        );
        return Err(Error::Limit("merged MA tree exceeds leaf budget"));
    }
    Ok(final_size)
}

/// Every other leaf is already occupied or reserved for an unfinished job.
/// Give a learner all remaining capacity, never an equal partition quota.
pub(super) fn graft_leaf_allowance(nodes: usize, max_leaves: usize) -> Result<usize> {
    let reserved = nodes
        .div_ceil(2)
        .checked_sub(1)
        .ok_or_else(|| Error::Encoding("empty MA graft destination tree".into()))?;
    max_leaves
        .checked_sub(reserved)
        .filter(|&limit| limit != 0)
        .ok_or(Error::Limit("MA skeleton exceeds leaf budget"))
}

pub(super) fn train(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    scratch_pool: &ScratchPool,
) -> Result<Option<Vec<Node>>> {
    let Some(mut plan) = plan(samples, options)? else {
        return Ok(None);
    };
    let _scope = crate::profile::scope("ma.channel_trees");
    // Bounded roots consume the shared remainder in plan order. Independent
    // allowances can oversubscribe the merge, and pruning learned trees would
    // lose their seed predictors or mandatory partitions.
    let parallel_roots = options.parallel
        && options.budget_mode != Some(super::MaBudgetMode::Bounded)
        && !crate::execution::is_serial();
    let root_limit = if parallel_roots {
        options.parallel_tools.roots(plan.jobs.len())
    } else {
        1
    };
    crate::profile::note(
        "ma.channel_trees",
        format_args!(
            "channels={} jobs={} samples={} row_depth={} root_limit={}",
            plan.channels,
            plan.jobs.len(),
            samples.counts.len(),
            options.channel_tree.row_depth(),
            root_limit,
        ),
    );
    if parallel_roots {
        let local_limit = graft_leaf_allowance(plan.nodes.len(), options.max_leaves)?;
        // Partitioning is complete. Loan every column's disjoint intervals;
        // quantizers stay shared. Only local index descriptors are rebuilt.
        drop(plan.indices);
        let mut spatial_order: Vec<_> = plan.jobs.into_iter().enumerate().collect();
        spatial_order.sort_by_key(|(_, part)| part.begin);
        let ranges: Vec<_> = spatial_order
            .iter()
            .map(|(_, part)| part.begin..part.end)
            .collect();
        let views = samples.ranges_mut(&ranges)?;
        let mut jobs: Vec<_> = spatial_order
            .into_iter()
            .zip(views)
            .map(|((index, part), view)| (index, part, view))
            .collect();
        jobs.sort_by_key(|(index, _, _)| *index);
        let destinations: Vec<_> = jobs.iter().map(|(_, part, _)| part.node).collect();
        let order: Vec<_> = (0..jobs.len()).collect();
        let learned = super::ma_parallel::run(
            jobs,
            &order,
            options.parallel_tools.workers.tasks,
            root_limit,
            options.parallel_tools.quiet,
            |(_, part, mut view)| {
                let learning = TrainOptions {
                    max_leaves: local_limit,
                    static_property_range: Some(part.range),
                    ..options.clone()
                };
                let slots = options
                    .properties
                    .len()
                    .min(crate::execution::worker_count())
                    * if options.parallel_nodes { 2 } else { 1 };
                let mut indices = Vec::new();
                indices
                    .try_reserve_exact(view.counts.len())
                    .map_err(|_| Error::Limit("MA channel local indices allocation"))?;
                indices.extend(0..view.counts.len());
                stats
                    .channel_roots
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                train_roots_with_layout::<true>(
                    &mut view,
                    &learning,
                    stats,
                    &ScratchPool::new(slots),
                    &mut indices,
                    vec![leaf(options.predictors[0])],
                    vec![TrainingNode {
                        node: 0,
                        begin: 0,
                        end: part.end - part.begin,
                        predictor: 0,
                        multiplier: 1,
                    }],
                )
            },
        )?;
        for (destination, learned) in destinations.into_iter().zip(learned) {
            graft(&mut plan.nodes, destination, learned, options.max_leaves)?;
        }
        return Ok(Some(plan.nodes));
    }
    for part in plan.jobs {
        // Preserve absolute indices for deterministic race hashes and deferred
        // cache membership. A learner only mutates this prepartitioned range.
        let learning = TrainOptions {
            max_leaves: graft_leaf_allowance(plan.nodes.len(), options.max_leaves)?,
            static_property_range: Some(part.range),
            ..options.clone()
        };
        let learned = train_roots_with_layout::<true>(
            samples,
            &learning,
            stats,
            scratch_pool,
            &mut plan.indices,
            vec![leaf(options.predictors[0])],
            vec![TrainingNode {
                node: 0,
                begin: part.begin,
                end: part.end,
                predictor: 0,
                multiplier: 1,
            }],
        )?;
        // The local allowance reserves other roots; the graft still enforces
        // the invocation ceiling after all independent work has completed.
        graft(&mut plan.nodes, part.node, learned, options.max_leaves)?;
    }
    Ok(Some(plan.nodes))
}

#[cfg(test)]
mod tests {
    use super::super::{
        ConstantChannel, InitialCacheStage, MultiplierInfo, ResidualToken, SearchOptions, Tree,
        train_nodes_with_layout, train_nodes_with_stats, train_with_constants,
    };
    use super::*;
    use std::sync::atomic::Ordering;

    fn fixture(records: &[usize], ybins: usize) -> (TreeSamples, TrainOptions) {
        let total: usize = records.iter().sum();
        let mut samples = TreeSamples {
            properties: vec![
                PropertyColumn::Wide(Vec::with_capacity(total)),
                PropertyColumn::Wide(Vec::with_capacity(total)),
                PropertyColumn::Packed(Vec::with_capacity(total)),
                PropertyColumn::Packed(Vec::with_capacity(total)),
            ],
            residuals: vec![Vec::with_capacity(total), Vec::with_capacity(total)],
            counts: Vec::with_capacity(total),
            thresholds: vec![
                (0..records.len().saturating_sub(1) as i32).collect(),
                vec![0],
                (0..ybins.saturating_sub(1) as i32).collect(),
                vec![0],
            ],
        };
        for row in 0..records.iter().copied().max().unwrap_or(0) {
            for (channel, &count) in records.iter().enumerate() {
                if row >= count {
                    continue;
                }
                samples.properties[0].push(channel as u32);
                samples.properties[1].push((row % 2) as u32);
                samples.properties[2].push((row % ybins) as u32);
                samples.properties[3].push((row % 2) as u32);
                samples.residuals[0].push(ResidualToken {
                    token: (row % 16) as u8,
                    nbits: 0,
                });
                samples.residuals[1].push(ResidualToken {
                    token: ((row + channel) % 16) as u8,
                    nbits: 0,
                });
                samples.counts.push(1);
            }
        }
        let options = TrainOptions {
            predictors: vec![Predictor::Zero, Predictor::Left],
            properties: vec![0, 1, 2, 6],
            channel_tree: ChannelTreeMode::Channels,
            static_property_range: Some([[0, records.len() as u32], [0, 2]]),
            parallel: false,
            split_threshold: 1e15,
            require_complete_search: true,
            ..TrainOptions::default()
        };
        (samples, options)
    }

    fn signature(samples: &TreeSamples) -> Vec<Vec<u32>> {
        let mut rows: Vec<_> = (0..samples.counts.len())
            .map(|i| {
                let mut row = vec![u32::from(samples.counts[i])];
                for column in &samples.residuals {
                    row.extend([u32::from(column[i].token), u32::from(column[i].nbits)]);
                }
                row.extend(samples.properties.iter().map(|column| column.value(i)));
                row
            })
            .collect();
        rows.sort();
        rows
    }

    fn tree_signature(nodes: Vec<Node>) -> Vec<(usize, u32)> {
        Tree::new(nodes)
            .unwrap()
            .tokens()
            .iter()
            .map(|t| (t.context, t.value))
            .collect()
    }

    fn binary_residuals(samples: &mut TreeSamples, options: &mut TrainOptions, only_high: bool) {
        samples.residuals.truncate(1);
        options.predictors.truncate(1);
        options.split_threshold = 1.0;
        for i in 0..samples.counts.len() {
            let active = !only_high || samples.properties[0].value(i) != 0;
            samples.residuals[0][i].token = if active && samples.properties[3].value(i) != 0 {
                15
            } else {
                0
            };
        }
    }

    #[test]
    fn channel_tree_source_gates_record_count_bins_order_and_multipliers() {
        assert_eq!(
            TrainOptions::default().channel_tree,
            ChannelTreeMode::Disabled
        );
        for (counts, enabled) in [
            ([4095, 4096], false),
            ([4096, 4096], true),
            ([8192, 0], false),
        ] {
            let (mut samples, options) = fixture(&counts, 64);
            samples.counts.fill(u16::MAX);
            assert_eq!(plan(&mut samples, &options).unwrap().is_some(), enabled);
        }
        let (samples, baseline) = fixture(&[4096, 4096], 64);
        for bins in [2, 32, 33] {
            let mut samples = samples.clone();
            samples.thresholds[0] = (0..bins - 1).collect();
            assert_eq!(plan(&mut samples, &baseline).unwrap().is_some(), bins <= 32);
        }
        for route in 0..3 {
            let mut samples = samples.clone();
            let original = signature(&samples);
            let mut options = baseline.clone();
            match route {
                0 => options.channel_tree = ChannelTreeMode::Disabled,
                1 => options.properties.swap(0, 1),
                _ => options.multiplier_info.push(MultiplierInfo {
                    range: [[0, 2], [0, 2]],
                    multiplier: 1,
                }),
            }
            assert!(plan(&mut samples, &options).unwrap().is_none());
            assert_eq!(signature(&samples), original);
        }
    }

    #[test]
    fn channel_tree_rows_use_record_median_and_exact_depth_side_boundaries() {
        assert_eq!(row_cut(&[8192, 8191], 16383), None);
        assert_eq!(row_cut(&[8192, 8192], 16384), Some((0, 8192)));
        assert_eq!(row_cut(&[12288, 4096], 16384), Some((0, 12288)));
        assert_eq!(row_cut(&[12289, 4095], 16384), None);
        assert_eq!(row_cut(&[4095, 8193, 4096], 16384), Some((1, 12288)));
        assert_eq!(row_cut(&[0, 8192, 0, 8192], 16384), Some((1, 8192)));
        assert_eq!(row_cut(&[0, 16384], 16384), None);
        let (samples, baseline) = fixture(&[32768, 32768], 64);
        for (mode, jobs) in [
            (ChannelTreeMode::Channels, 2),
            (ChannelTreeMode::ChannelsAndRows, 4),
            (ChannelTreeMode::ChannelsAndRowsTwice, 8),
        ] {
            let options = TrainOptions {
                channel_tree: mode,
                ..baseline.clone()
            };
            let plain = plan(&mut samples.clone(), &options).unwrap().unwrap();
            let mut weighted = samples.clone();
            for i in 0..weighted.counts.len() {
                if weighted.properties[2].value(i) == 0 {
                    weighted.counts[i] = u16::MAX;
                }
            }
            let weighted = plan(&mut weighted, &options).unwrap().unwrap();
            assert_eq!(plain.jobs.len(), jobs);
            assert_eq!(weighted.jobs.len(), jobs);
            assert_eq!(tree_signature(plain.nodes), tree_signature(weighted.nodes));
            assert!(plain.jobs.iter().all(|p| p.depth == mode.row_depth()));
        }
        for per_channel in [16383, 16384] {
            let (mut samples, mut options) = fixture(&[per_channel, per_channel], 64);
            samples.counts.fill(u16::MAX);
            options.channel_tree = ChannelTreeMode::ChannelsAndRowsTwice;
            let planned = plan(&mut samples, &options).unwrap().unwrap();
            assert_eq!(planned.jobs.len(), if per_channel == 16384 { 4 } else { 2 });
        }
        let mut options = baseline;
        options.channel_tree = ChannelTreeMode::ChannelsAndRowsTwice;
        options.properties[2] = 3;
        assert_eq!(
            plan(&mut samples.clone(), &options)
                .unwrap()
                .unwrap()
                .jobs
                .len(),
            2
        );
    }

    #[test]
    fn channel_tree_partition_preserves_columns_quantizers_and_sparse_domains() {
        let (mut samples, options) = fixture(&[100, 0, 100, 50000, 100], 64);
        for (i, count) in samples.counts.iter_mut().enumerate() {
            *count = if i % 17 == 0 {
                u16::MAX
            } else {
                1 + (i % 7) as u16
            };
        }
        let before = signature(&samples);
        let thresholds = samples.thresholds.clone();
        let pointers: Vec<_> = samples
            .properties
            .iter()
            .map(PropertyColumn::as_ptr)
            .collect();
        let count_ptr = samples.counts.as_ptr();
        let planned = plan(&mut samples, &options).unwrap().unwrap();
        assert!(matches!(
            planned.nodes[0],
            Node::Split {
                property: 0,
                threshold: 2,
                ..
            }
        ));
        assert_eq!(
            planned
                .jobs
                .iter()
                .map(|p| samples.properties[0].value(p.begin))
                .collect::<Vec<_>>(),
            [4, 3, 2, 0]
        );
        let mut ranges: Vec<_> = planned.jobs.iter().map(|p| (p.begin, p.end)).collect();
        ranges.sort_unstable();
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, samples.counts.len());
        assert!(ranges.windows(2).all(|pair| pair[0].1 == pair[1].0));
        for part in &planned.jobs {
            let channel = samples.properties[0].value(part.begin);
            assert!((part.begin..part.end).all(|i| samples.properties[0].value(i) == channel));
            assert!(part.range[0][0] <= channel && channel < part.range[0][1]);
            assert_eq!(part.range[1], [0, 2]);
        }
        assert_eq!(
            planned.indices,
            (0..samples.counts.len()).collect::<Vec<_>>()
        );
        assert_eq!(signature(&samples), before);
        assert_eq!(samples.thresholds, thresholds);
        assert_eq!(samples.counts.as_ptr(), count_ptr);
        assert_eq!(
            samples
                .properties
                .iter()
                .map(PropertyColumn::as_ptr)
                .collect::<Vec<_>>(),
            pointers
        );
        Tree::new(planned.nodes).unwrap();
    }

    #[test]
    fn channel_tree_graft_and_global_budget_never_assign_equal_partition_quotas() {
        let (mut samples, mut options) = fixture(&[4096, 4096], 64);
        options.max_leaves = 1;
        assert!(matches!(plan(&mut samples, &options), Err(Error::Limit(_))));
        options.max_leaves = 3;
        binary_residuals(&mut samples, &mut options, true);
        let tree = Tree::new(
            train_nodes_with_stats(&mut samples, &options, &TrainingStats::default()).unwrap(),
        )
        .unwrap();
        assert_eq!(tree.leaves(), 3);
        assert_eq!(
            tree.lookup(&[0, 0, 0, 0, 0, 0, 1]).unwrap().1,
            Predictor::Zero
        );

        let (mut samples, mut options) = fixture(&[4096, 4096], 64);
        binary_residuals(&mut samples, &mut options, false);
        options.max_leaves = 3;
        assert!(matches!(
            train_nodes_with_stats(&mut samples, &options, &TrainingStats::default()),
            Err(Error::Limit(_))
        ));
        options.max_leaves = 4;
        assert_eq!(
            Tree::new(
                train_nodes_with_stats(&mut samples, &options, &TrainingStats::default()).unwrap()
            )
            .unwrap()
            .leaves(),
            4
        );

        let mut nodes = vec![leaf(Predictor::Zero)];
        let (greater, _) = split(&mut nodes, 0, 0, 0, Predictor::Zero, 3).unwrap();
        let learned = vec![
            Node::Split {
                property: 6,
                threshold: 0,
                greater: 1,
                less_equal: 2,
            },
            leaf(Predictor::Left),
            leaf(Predictor::Zero),
        ];
        let before = tree_signature(nodes.clone());
        assert!(matches!(
            graft(&mut nodes, greater, learned.clone(), 2),
            Err(Error::Limit(_))
        ));
        assert_eq!(tree_signature(nodes.clone()), before);
        graft(&mut nodes, greater, learned, 3).unwrap();
        let tree = Tree::new(nodes).unwrap();
        assert_eq!(tree.leaves(), 3);
        assert_eq!(
            tree.lookup(&[1, 0, 0, 0, 0, 0, 1]).unwrap().1,
            Predictor::Left
        );
        assert_eq!(
            tree.lookup(&[0, 0, 0, 0, 0, 0, 1]).unwrap().1,
            Predictor::Zero
        );
    }

    #[test]
    fn channel_tree_partition_learning_retains_absolute_race_offsets() {
        let (mut samples, mut options) = fixture(&[262144, 262144], 64);
        binary_residuals(&mut samples, &mut options, false);
        for property in [5, 7, 8] {
            options.properties.push(property);
            samples.properties.push(samples.properties[3].clone());
            samples.thresholds.push(vec![0]);
        }
        options.search = SearchOptions {
            statistical_race: true,
            ..Default::default()
        };
        options.max_leaves = 4;
        let stats = TrainingStats::default();
        let nodes = train_nodes_with_stats(&mut samples, &options, &stats).unwrap();
        assert_eq!(Tree::new(nodes).unwrap().leaves(), 4);
        assert_eq!(
            *stats.race_ranges.lock().unwrap(),
            [(262144, 524288), (0, 262144)]
        );
        assert_eq!(stats.race_nodes.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn channel_tree_fallbacks_and_cached_parallel_properties_preserve_model() {
        for counts in [[4095, 4096], [8192, 0]] {
            let (samples, options) = fixture(&counts, 64);
            let actual =
                train_nodes_with_stats(&mut samples.clone(), &options, &TrainingStats::default())
                    .unwrap();
            let expected = train_nodes_with_layout::<true>(
                &mut samples.clone(),
                &options,
                &TrainingStats::default(),
                &ScratchPool::new(1),
            )
            .unwrap();
            assert_eq!(tree_signature(actual), tree_signature(expected));
        }
        let (samples, mut options) = fixture(&[8192, 8192], 64);
        let mut baseline = samples.clone();
        binary_residuals(&mut baseline, &mut options, false);
        let expected = tree_signature(
            train_nodes_with_stats(&mut baseline.clone(), &options, &TrainingStats::default())
                .unwrap(),
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        for cache in [0, 32 << 20] {
            options.histogram_cache_bytes = cache;
            options.parallel = true;
            let actual = pool
                .install(|| {
                    train_nodes_with_stats(
                        &mut baseline.clone(),
                        &options,
                        &TrainingStats::default(),
                    )
                })
                .unwrap();
            assert_eq!(tree_signature(actual), expected);
        }
    }

    #[test]
    fn channel_tree_failed_partition_releases_scratch_for_retry() {
        let (mut samples, mut options) = fixture(&[8192, 8192], 64);
        binary_residuals(&mut samples, &mut options, false);
        let scratch = ScratchPool::new(1);
        let failing = TrainingStats {
            initial_cache_hook: Some(Box::new(|_, stage| {
                if stage == InitialCacheStage::Filled {
                    return Err(Error::Limit("injected channel cache failure"));
                }
                Ok(())
            })),
            ..TrainingStats::default()
        };
        assert!(matches!(
            train(&mut samples, &options, &failing, &scratch),
            Err(Error::Limit(_))
        ));
        assert_eq!(scratch.idle.lock().unwrap().len(), 1);
        let created = scratch.created.load(Ordering::Relaxed);
        let nodes = train(&mut samples, &options, &TrainingStats::default(), &scratch)
            .unwrap()
            .unwrap();
        assert_eq!(Tree::new(nodes).unwrap().leaves(), 4);
        assert_eq!(scratch.created.load(Ordering::Relaxed), created);
        assert!(scratch.reused.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn channel_tree_constant_suffix_reserves_global_leaf_budget() {
        let (mut samples, mut options) = fixture(&[4096, 4096], 64);
        samples.thresholds[0].push(1);
        options.static_property_range = Some([[0, 3], [0, 2]]);
        options.constant_channels = vec![ConstantChannel {
            channel: 2,
            value: 7,
        }];
        options.max_leaves = 3;
        let tree = train_with_constants(&options, |learning| {
            assert_eq!(learning.static_property_range, Some([[0, 2], [0, 2]]));
            assert_eq!(learning.max_leaves, 2);
            assert!(learning.constant_channels.is_empty());
            train_nodes_with_stats(&mut samples, learning, &TrainingStats::default())
        })
        .unwrap();
        assert_eq!(tree.leaves(), 3);
        let (_, predictor, offset, multiplier) = tree.lookup(&[2, 0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!((predictor, offset, multiplier), (Predictor::Zero, 7, 1));
        options.max_leaves = 2;
        assert!(matches!(
            train_with_constants(&options, |learning| {
                train_nodes_with_stats(&mut samples, learning, &TrainingStats::default())
            }),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn channel_tree_cooperative_rows_share_columns_and_preserve_serial_model() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (mode, jobs) in [
            (ChannelTreeMode::Channels, 2),
            (ChannelTreeMode::ChannelsAndRows, 4),
            (ChannelTreeMode::ChannelsAndRowsTwice, 8),
        ] {
            let (mut samples, mut options) = fixture(&[32768, 32768], 64);
            binary_residuals(&mut samples, &mut options, false);
            for (i, count) in samples.counts.iter_mut().enumerate() {
                *count = (i % 7 + 1) as u16;
            }
            options.channel_tree = mode;
            options.max_leaves = jobs * 2;
            let expected = tree_signature(
                train_nodes_with_stats(&mut samples.clone(), &options, &TrainingStats::default())
                    .unwrap(),
            );
            let before = signature(&samples);
            options.parallel = true;
            options.parallel_tools.workers.tasks = 3;
            options.parallel_tools.workers.scratch_bytes =
                2 * super::super::ma_parallel::ROOT_SCRATCH_BYTES;
            options.parallel_tools.quiet = true;
            for null in [false, true] {
                let stats = TrainingStats::default();
                let mut columns = samples.clone();
                let counts_pointer = columns.counts.as_ptr();
                let property_pointers: Vec<_> = columns
                    .properties
                    .iter()
                    .map(PropertyColumn::as_ptr)
                    .collect();
                let nodes = pool
                    .install(|| {
                        let mut work = || train_nodes_with_stats(&mut columns, &options, &stats);
                        if null {
                            crate::execution::serial(work)
                        } else {
                            work()
                        }
                    })
                    .unwrap();
                assert_eq!(tree_signature(nodes), expected);
                assert_eq!(signature(&columns), before);
                assert_eq!(columns.counts.as_ptr(), counts_pointer);
                assert_eq!(
                    columns
                        .properties
                        .iter()
                        .map(PropertyColumn::as_ptr)
                        .collect::<Vec<_>>(),
                    property_pointers
                );
                assert_eq!(
                    stats.channel_roots.load(Ordering::Relaxed),
                    if null { 0 } else { jobs }
                );
            }
        }
    }

    #[test]
    fn channel_tree_cooperative_failure_joins_column_loans_before_retry() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let (mut samples, mut options) = fixture(&[8192, 8192], 64);
        binary_residuals(&mut samples, &mut options, false);
        options.parallel = true;
        let original = signature(&samples);
        let stats = TrainingStats {
            initial_cache_hook: Some(Box::new(|_, stage| {
                if stage == InitialCacheStage::Filled {
                    Err(Error::Limit("injected cooperative cache error"))
                } else {
                    Ok(())
                }
            })),
            ..TrainingStats::default()
        };
        assert!(matches!(
            pool.install(|| train_nodes_with_stats(&mut samples, &options, &stats)),
            Err(Error::Limit("injected cooperative cache error"))
        ));
        assert_eq!(signature(&samples), original);
        let actual = pool
            .install(|| train_nodes_with_stats(&mut samples, &options, &TrainingStats::default()))
            .unwrap();
        assert_eq!(Tree::new(actual).unwrap().leaves(), 4);
    }
}
