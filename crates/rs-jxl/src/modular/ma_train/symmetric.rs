// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; notice in docs/PORTING_KERNELS.md.
//! PR5007 interval proposals expanded into standard strict-greater binary nodes.

use super::learner::scaled_leaf;
use super::{
    Node, Ordering, PendingNode, SampleStorage, Samples, Split, TrainOptions, TrainingNode,
    TrainingStats, parent_histograms, partition_node, subtract_parent_histograms,
};
use crate::Result;

#[allow(clippy::too_many_arguments)]
pub(super) fn commit<const REORDER: bool>(
    samples: &mut Samples<impl SampleStorage>,
    options: &TrainOptions,
    stats: &TrainingStats,
    indices: &mut [usize],
    nodes: &mut Vec<Node>,
    mut state: PendingNode,
    split: Split,
) -> Result<[PendingNode; 3]> {
    let interval = split.symmetric.expect("two-sided proposal");
    let job = state.job;
    let begin = job.begin;
    let end = job.end;
    let low_end = begin + interval.low_count;
    let mid_end = begin + split.below_count;
    debug_assert!(begin < low_end && low_end < mid_end && mid_end < end);
    debug_assert!(interval.low_bin < split.bin);
    debug_assert!(options.properties[split.property] >= 2);
    partition_node::<REORDER>(
        samples,
        &mut indices[begin..end],
        split.property,
        split.bin,
        split.below_count,
    );
    partition_node::<REORDER>(
        samples,
        &mut indices[begin..mid_end],
        split.property,
        interval.low_bin,
        interval.low_count,
    );

    let high = nodes.len();
    let midlow = high + 1;
    let mid = high + 2;
    let low = high + 3;
    nodes.push(scaled_leaf(
        options.predictors[split.above_predictor],
        job.multiplier,
    ));
    nodes.push(Node::Split {
        property: options.properties[split.property],
        threshold: samples.thresholds[split.property][interval.low_bin],
        greater: mid,
        less_equal: low,
    });
    nodes.push(scaled_leaf(
        options.predictors[interval.middle_predictor],
        job.multiplier,
    ));
    nodes.push(scaled_leaf(
        options.predictors[split.below_predictor],
        job.multiplier,
    ));
    nodes[job.node] = Node::Split {
        property: options.properties[split.property],
        threshold: samples.thresholds[split.property][split.bin],
        greater: high,
        less_equal: midlow,
    };
    let mut children = [
        PendingNode::new(TrainingNode {
            node: low,
            end: low_end,
            predictor: split.below_predictor,
            ..job
        }),
        PendingNode::new(TrainingNode {
            node: mid,
            begin: low_end,
            end: mid_end,
            predictor: interval.middle_predictor,
            ..job
        }),
        PendingNode::new(TrainingNode {
            node: high,
            begin: mid_end,
            predictor: split.above_predictor,
            ..job
        }),
    ];
    for child in &mut children {
        child.range = state.range;
        child.redundant_properties = state.redundant_properties.clone();
    }
    let largest = (0..3)
        .max_by_key(|&i| children[i].job.end - children[i].job.begin)
        .unwrap();
    let retained = children[largest].job.end - children[largest].job.begin;
    if options.reuse_parent_histograms && retained >= 512 {
        let mut parent = state.parents.take().expect("split has parent histograms");
        for (index, child) in children.iter_mut().enumerate() {
            if index == largest {
                continue;
            }
            let counts = parent_histograms(samples, &indices[child.job.begin..child.job.end]);
            parent = subtract_parent_histograms(parent, &counts)?;
            child.parents = Some(counts);
        }
        children[largest].parents = Some(parent);
        stats.parent_reuses.fetch_add(1, Ordering::Relaxed);
    }
    // The existing deferred subtraction descriptor is one contiguous interval.
    // Reuse it for an outer child; a middle child's complement has two ranges.
    if largest != 1 && retained >= options.histogram_cache_thresholds().1 {
        children[largest].cache = state.cache.take();
        if children[largest].cache.is_some() {
            children[largest].removed = Some(if largest == 0 {
                low_end..end
            } else {
                begin..mid_end
            });
        }
    }
    Ok(children)
}
