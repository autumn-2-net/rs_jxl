//! Opt-in full-population predictor refit with frozen MA topology.
//!
//! Native encoder experiment using the existing Rust traversal and predictors.
//! The rate estimate excludes tree/model headers, clustering and LZ77. Callers
//! must encode both trees and select the smaller complete result.

use std::collections::VecDeque;

use crate::entropy::HybridUintConfig;
use crate::{Error, Result};

use super::encode;
use super::predictor::{Predictor, predict_one};
use super::training::TrainingStream;
use super::tree::{Node, Tree, pack_signed};

const UINT: HybridUintConfig = match HybridUintConfig::new(4, 1, 2) {
    Ok(config) => config,
    Err(_) => panic!("valid refit HybridUint configuration"),
};
// Direct tokens plus eight tokens for each exponent 4..=31.
const TOKENS: usize = 16 + 28 * 8;

struct Histogram {
    counts: [u64; TOKENS],
    count: u64,
    extra: u64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            counts: [0; TOKENS],
            count: 0,
            extra: 0,
        }
    }

    fn add(&mut self, residual: i32) -> Result<()> {
        let encoded = UINT.encode(pack_signed(residual));
        self.count = self
            .count
            .checked_add(1)
            .ok_or(Error::Limit("MA leaf refit population overflow"))?;
        self.extra = self
            .extra
            .checked_add(u64::from(encoded.nbits))
            .ok_or(Error::Limit("MA leaf refit extra-bit overflow"))?;
        self.counts[encoded.token as usize] += 1;
        Ok(())
    }

    fn bits(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let total = self.count as f64;
        // Equal populations under different token labels must score identically.
        let mut counts = self.counts;
        counts.sort_unstable();
        // Per-bin form avoids cancellation for a single occupied token.
        self.extra as f64
            + counts
                .iter()
                .filter(|&&count| count != 0)
                .map(|&count| {
                    let count = count as f64;
                    count * (total / count).log2()
                })
                .sum::<f64>()
    }
}

struct Candidate {
    predictor: Predictor,
    histogram: Histogram,
}

struct LeafStats {
    node_index: usize,
    // Original is always first, including originals outside the allowed set.
    candidates: Vec<Candidate>,
}

/// Refit only zero-offset, unit-multiplier leaves. `allowed` supplies concrete
/// predictors; only Gradient and Weighted are considered as replacements.
/// Every supplied pixel participates, with channel-local causal WP state.
/// Returns None when no strictly improving predictor changes are proposed.
/// Memory is O(nodes + eligible leaves * candidates * 240), plus WP row state;
/// there is no sampling, hidden population cap or retained residual plane.
pub(super) fn refit(
    tree: &Tree,
    streams: &[TrainingStream<'_>],
    allowed: &[Predictor],
    previous_channels: usize,
) -> Result<Option<Tree>> {
    let stats = collect_stats(tree, streams, allowed, previous_channels)?;
    let mut replacements = Vec::new();
    for leaf in stats.iter().flatten() {
        let original = &leaf.candidates[0];
        if original.histogram.count == 0 {
            continue;
        }
        let mut best = original;
        let mut best_bits = best.histogram.bits();
        for candidate in &leaf.candidates[1..] {
            let bits = candidate.histogram.bits();
            if bits < best_bits {
                best = candidate;
                best_bits = bits;
            }
        }
        if best.predictor != original.predictor {
            replacements.push((leaf.node_index, best.predictor));
        }
    }
    if replacements.is_empty() {
        return Ok(None);
    }
    let mut nodes = tree.nodes().to_vec();
    for (index, replacement) in replacements {
        if let Node::Leaf { predictor, .. } = &mut nodes[index] {
            *predictor = replacement;
        }
    }
    Tree::new(nodes).map(Some)
}

fn collect_stats(
    tree: &Tree,
    streams: &[TrainingStream<'_>],
    allowed: &[Predictor],
    previous_channels: usize,
) -> Result<Vec<Option<LeafStats>>> {
    let ordered = ordered_streams(tree, streams, previous_channels)?;
    collect_validated_stats(tree, &ordered, allowed, previous_channels)
}

pub(super) fn ordered_streams<'a, 'b>(
    tree: &Tree,
    streams: &'a [TrainingStream<'b>],
    previous_channels: usize,
) -> Result<Vec<&'a TrainingStream<'b>>> {
    // Match the native encoder's property-slot bound, not a resource policy.
    if previous_channels > 60 {
        return Err(Error::InvalidInput("previous_channels exceeds 60".into()));
    }
    for node in tree.nodes() {
        if let Node::Split { property, .. } = node
            && *property >= 16 + 4 * previous_channels
        {
            return Err(Error::InvalidInput(
                "MA leaf refit tree requires unavailable properties".into(),
            ));
        }
    }
    let mut ordered: Vec<_> = streams.iter().collect();
    ordered.sort_unstable_by_key(|stream| stream.stream_id);
    let mut previous_id = None;
    for stream in &ordered {
        if stream.stream_id > i32::MAX as usize || previous_id == Some(stream.stream_id) {
            return Err(Error::InvalidInput(
                "invalid/duplicate MA leaf refit stream ID".into(),
            ));
        }
        previous_id = Some(stream.stream_id);
        if stream.channels.len() > i32::MAX as usize {
            return Err(Error::Limit("MA leaf refit channel IDs exceed i32"));
        }
        stream.wp.validate()?;
        for channel in stream.channels {
            let area = channel
                .width
                .checked_mul(channel.height)
                .ok_or(Error::Limit("MA leaf refit plane dimensions overflow"))?;
            if area != channel.samples.len() {
                return Err(Error::InvalidInput(
                    "MA leaf refit plane dimensions/buffer mismatch".into(),
                ));
            }
        }
    }

    Ok(ordered)
}

fn collect_validated_stats(
    tree: &Tree,
    ordered: &[&TrainingStream<'_>],
    allowed: &[Predictor],
    previous_channels: usize,
) -> Result<Vec<Option<LeafStats>>> {
    let mut stats = Vec::new();
    stats
        .try_reserve_exact(tree.leaves())
        .map_err(|_| Error::Limit("MA leaf refit leaf allocation"))?;
    let mut queue = VecDeque::from([0]);
    let mut any = false;
    // Tree::lookup yields BFS leaf contexts, NOT source node indices.
    while let Some(node_index) = queue.pop_front() {
        match tree.nodes()[node_index] {
            Node::Split {
                greater,
                less_equal,
                ..
            } => {
                queue.push_back(greater);
                queue.push_back(less_equal);
            }
            Node::Leaf {
                predictor,
                offset,
                multiplier,
            } => {
                let mut candidates = Vec::new();
                let alternatives = [Predictor::Gradient, Predictor::Weighted];
                if offset == 0
                    && multiplier == 1
                    && alternatives
                        .iter()
                        .any(|&p| p != predictor && allowed.contains(&p))
                {
                    candidates
                        .try_reserve_exact(3)
                        .map_err(|_| Error::Limit("MA leaf refit histogram allocation"))?;
                    candidates.push(Candidate {
                        predictor,
                        histogram: Histogram::new(),
                    });
                    for alternative in alternatives {
                        if alternative != predictor && allowed.contains(&alternative) {
                            candidates.push(Candidate {
                                predictor: alternative,
                                histogram: Histogram::new(),
                            });
                        }
                    }
                }
                if candidates.is_empty() {
                    stats.push(None);
                } else {
                    any = true;
                    stats.push(Some(LeafStats {
                        node_index,
                        candidates,
                    }));
                }
            }
        }
    }
    if !any {
        return Ok(stats);
    }
    let use_wp = tree.uses_weighted()
        || stats.iter().flatten().any(|leaf| {
            leaf.candidates
                .iter()
                .any(|candidate| candidate.predictor == Predictor::Weighted)
        });
    for stream in ordered {
        encode::visit(
            stream.channels,
            &stream.wp,
            use_wp,
            previous_channels,
            stream.stream_id,
            |_, _, neighbors, wp_prediction, properties, value| {
                let (context, _, _, _) = tree.lookup(properties)?;
                if let Some(leaf) = &mut stats[context] {
                    for candidate in &mut leaf.candidates {
                        let prediction = predict_one(
                            candidate.predictor,
                            neighbors,
                            Some(i64::from(wp_prediction)),
                        )
                        .map_err(|e| Error::Encoding(e.to_string()))?;
                        // Unit multiplier: narrow the wide difference modulo 2^32,
                        // exactly as the native encoder does before PackSigned.
                        candidate
                            .histogram
                            .add((i64::from(value) - prediction) as i32)?;
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok(stats)
}

#[cfg(test)]
#[path = "leaf_refit_tests.rs"]
mod tests;
