//! Static leaf bias proposals, not a new predictor or decoder-side adaptation.
//! A sampled HybridUint estimate ranks proposals; refinement::choose is the
//! authority and compares actual serialized tree/model/residual costs.

use std::collections::VecDeque;

use crate::entropy::HybridUintConfig;
use crate::{Error, Result};

use super::predictor::predict_one;
use super::training::TrainingStream;
use super::tree::{Node, Tree, pack_signed};

mod distribution;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OffsetMode {
    Mean,
    Median,
    #[default]
    Entropy,
    /// Broader sample-distribution proposals using the baseline literal mapping.
    /// Can refit existing unit-multiplier offsets, including shallow MA plans.
    Distribution,
}

#[derive(Debug, Clone)]
pub struct OffsetOptions {
    pub mode: OffsetMode,
    /// Reservoir capacity per eligible leaf. Only bounds fitting statistics;
    /// it never limits image size, MA training or encoded leaf count.
    pub samples_per_leaf: usize,
}

impl Default for OffsetOptions {
    fn default() -> Self {
        Self {
            mode: OffsetMode::Entropy,
            samples_per_leaf: 1024,
        }
    }
}

impl OffsetOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.samples_per_leaf == 0 {
            return Err(Error::InvalidInput(
                "leaf offset samples must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub(super) struct FitStats {
    pub leaves: usize,
    pub samples: usize,
    pub population: usize,
    pub candidates: usize,
}

const UINT: HybridUintConfig = match HybridUintConfig::new(4, 1, 2) {
    Ok(config) => config,
    Err(_) => panic!("valid offset HybridUint configuration"),
};
const TOKENS: usize = 240;

struct Samples {
    node: usize,
    baseline: i32,
    count: u64,
    sum: i128,
    random: u64,
    all_zero: bool,
    values: Vec<i64>,
}

impl Samples {
    fn new(node: usize) -> Self {
        Self {
            node,
            baseline: 0,
            count: 0,
            sum: 0,
            random: node as u64,
            all_zero: true,
            values: Vec::new(),
        }
    }

    fn add(&mut self, residual: i64, capacity: usize) -> Result<()> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or(Error::Limit("offset population overflow"))?;
        self.sum += i128::from(residual);
        self.all_zero &= (residual as i32) == self.baseline;
        if self.values.len() < capacity {
            if self.values.len() == self.values.capacity() {
                self.values
                    .try_reserve_exact((capacity - self.values.len()).min(64))
                    .map_err(|_| Error::Limit("offset sample allocation"))?;
            }
            self.values.push(residual);
        } else {
            // Algorithm R with a deterministic SplitMix64 stream per leaf.
            // Full-population means are independent of the reservoir.
            self.random = self.random.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.random;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^= x >> 31;
            let index = ((u128::from(x) * u128::from(self.count)) >> 64) as u64;
            if index < self.values.len() as u64 {
                self.values[index as usize] = residual;
            }
        }
        Ok(())
    }

    fn mean(&self) -> i32 {
        if self.count == 0 {
            return 0;
        }
        let count = i128::from(self.count);
        ((self.sum + self.sum.signum() * (count / 2)) / count) as i32
    }

    fn candidates(&mut self, mode: OffsetMode) -> Vec<i32> {
        self.values.sort_unstable();
        let mean = self.mean();
        let median = self.values[(self.values.len() - 1) / 2] as i32;
        let mut result = vec![0];
        match mode {
            OffsetMode::Mean => result.push(mean),
            OffsetMode::Median => result.push(median),
            OffsetMode::Entropy | OffsetMode::Distribution => {
                for center in [mean, median] {
                    for delta in [0, -1, 1, -2, 2] {
                        if let Some(offset) = center.checked_add(delta) {
                            if !result.contains(&offset) {
                                result.push(offset);
                            }
                        }
                    }
                }
            }
        }
        result.dedup();
        result
    }

    fn bits(&self, offset: i32) -> f64 {
        let mut counts = [0u64; TOKENS];
        let mut extra = 0u64;
        for &residual in &self.values {
            // Unit multiplier: match encoder narrowing AFTER wide subtraction.
            let encoded = UINT.encode(pack_signed((residual - i64::from(offset)) as i32));
            counts[encoded.token as usize] += 1;
            extra += u64::from(encoded.nbits);
        }
        counts.sort_unstable();
        let n = self.values.len() as f64;
        let entropy: f64 = counts
            .into_iter()
            .filter(|&c| c != 0)
            .map(|c| c as f64 * (n / c as f64).log2())
            .sum();
        (extra as f64 + entropy) * self.count as f64 / n
    }
}

fn xlogx(n: u64) -> f64 {
    if n < 2 {
        0.0
    } else {
        n as f64 * (n as f64).log2()
    }
}

// Tree entropy context 3 contains ALL leaf offsets, including zero. This is
// only its marginal token estimate, not an independent fixed-width field cost.
fn offset_cost(counts: &[u64; TOKENS], offset: i32) -> f64 {
    let new = UINT.encode(pack_signed(offset));
    let token = new.token as usize;
    if token == 0 {
        return new.nbits as f64;
    }
    xlogx(counts[0]) - xlogx(counts[0] - 1) + xlogx(counts[token]) - xlogx(counts[token] + 1)
        + new.nbits as f64
}

pub(super) fn fit(
    tree: &Tree,
    streams: &[TrainingStream<'_>],
    previous_channels: usize,
    options: &OffsetOptions,
) -> Result<(Option<Tree>, FitStats)> {
    fit_with_configs(tree, streams, previous_channels, options, None)
}

pub(super) fn fit_with_configs(
    tree: &Tree,
    streams: &[TrainingStream<'_>],
    previous_channels: usize,
    options: &OffsetOptions,
    configs: Option<&[HybridUintConfig]>,
) -> Result<(Option<Tree>, FitStats)> {
    options.validate()?;
    if configs.is_some_and(|configs| configs.len() != tree.leaves()) {
        return Err(Error::InvalidInput(
            "offset context configuration count".into(),
        ));
    }
    let ordered = super::leaf_refit::ordered_streams(tree, streams, previous_channels)?;
    let mut leaves = Vec::new();
    leaves
        .try_reserve_exact(tree.leaves())
        .map_err(|_| Error::Limit("offset leaf allocation"))?;
    let mut offset_counts = [0u64; TOKENS];
    let mut queue = VecDeque::from([0]);
    while let Some(node) = queue.pop_front() {
        match tree.nodes()[node] {
            Node::Split {
                greater,
                less_equal,
                ..
            } => {
                queue.push_back(greater);
                queue.push_back(less_equal);
            }
            Node::Leaf {
                offset, multiplier, ..
            } => {
                offset_counts[UINT.encode(pack_signed(offset)).token as usize] += 1;
                leaves.push(
                    (multiplier == 1 && (offset == 0 || options.mode == OffsetMode::Distribution))
                        .then(|| Samples {
                            baseline: offset,
                            ..Samples::new(node)
                        }),
                );
            }
        }
    }
    let mut stats = FitStats::default();
    if leaves.iter().all(Option::is_none) {
        return Ok((None, stats));
    }
    for stream in ordered {
        super::encode::visit(
            stream.channels,
            &stream.wp,
            tree.uses_weighted(),
            previous_channels,
            stream.stream_id,
            |_, _, neighbors, wp_prediction, properties, value| {
                let (context, predictor, _, _) = tree.lookup(properties)?;
                if let Some(leaf) = &mut leaves[context] {
                    let prediction =
                        predict_one(predictor, neighbors, Some(i64::from(wp_prediction)))
                            .map_err(|e| Error::Encoding(e.to_string()))?;
                    leaf.add(i64::from(value) - prediction, options.samples_per_leaf)?;
                }
                Ok(())
            },
        )?;
    }
    if options.mode == OffsetMode::Distribution {
        return fit_distribution(tree, &mut leaves, offset_counts, configs);
    }
    let mut nodes = tree.nodes().to_vec();
    for leaf in leaves.iter_mut().flatten().filter(|s| s.count != 0) {
        stats.population = stats
            .population
            .checked_add(
                usize::try_from(leaf.count)
                    .map_err(|_| Error::Limit("offset population overflow"))?,
            )
            .ok_or(Error::Limit("offset population overflow"))?;
        stats.samples += leaf.values.len();
        // Preserve exact zero residuals even if merging offset-token labels
        // saves header bits but disables a fast decoding path.
        if leaf.all_zero {
            continue;
        }
        let candidates = leaf.candidates(options.mode);
        stats.candidates += candidates.len();
        let mut best = 0;
        let mut best_bits = leaf.bits(0);
        for offset in candidates.into_iter().filter(|&b| b != 0) {
            let bits = leaf.bits(offset) + offset_cost(&offset_counts, offset);
            if bits + 1e-9 * best_bits.abs().max(1.0) < best_bits {
                best = offset;
                best_bits = bits;
            }
        }
        if best != 0 {
            if let Node::Leaf { offset, .. } = &mut nodes[leaf.node] {
                *offset = best;
            }
            stats.leaves += 1;
            offset_counts[0] -= 1;
            offset_counts[UINT.encode(pack_signed(best)).token as usize] += 1;
        }
    }
    let proposal = if stats.leaves == 0 {
        None
    } else {
        Some(Tree::new(nodes)?)
    };
    Ok((proposal, stats))
}

fn replacement_cost(counts: &[u64; TOKENS], old: i32, new: i32) -> f64 {
    let old = UINT.encode(pack_signed(old));
    let new = UINT.encode(pack_signed(new));
    let extra = f64::from(new.nbits) - f64::from(old.nbits);
    if old.token == new.token {
        return extra;
    }
    let removed = counts[old.token as usize];
    let added = counts[new.token as usize];
    xlogx(removed) - xlogx(removed - 1) + xlogx(added) - xlogx(added + 1) + extra
}

fn fit_distribution(
    tree: &Tree,
    leaves: &mut [Option<Samples>],
    mut counts: [u64; TOKENS],
    configs: Option<&[HybridUintConfig]>,
) -> Result<(Option<Tree>, FitStats)> {
    struct Proposal {
        node: usize,
        costs: Vec<(i32, f64)>,
        selected: usize,
    }
    let mut proposals = Vec::new();
    let mut stats = FitStats::default();
    for (context, leaf) in leaves.iter_mut().enumerate() {
        let Some(leaf) = leaf.as_mut().filter(|leaf| leaf.count != 0) else {
            continue;
        };
        stats.population = stats
            .population
            .checked_add(
                usize::try_from(leaf.count)
                    .map_err(|_| Error::Limit("offset population overflow"))?,
            )
            .ok_or(Error::Limit("offset population overflow"))?;
        stats.samples += leaf.values.len();
        if leaf.all_zero {
            continue;
        }
        let config = configs.map_or_else(HybridUintConfig::default, |configs| configs[context]);
        let costs = distribution::scores(leaf, config)?;
        debug_assert_eq!(costs[0].0, leaf.baseline);
        stats.candidates += costs.len();
        proposals.push(Proposal {
            node: leaf.node,
            costs,
            selected: 0,
        });
    }
    // Revisit only cached scores, not pixels: offsets share a signaling histogram.
    // Alternating sweeps reduce the one-way BFS bias without another encode.
    for sweep in 0..3 {
        let mut changed = false;
        for position in 0..proposals.len() {
            let index = if sweep % 2 == 0 {
                position
            } else {
                proposals.len() - 1 - position
            };
            let proposal = &mut proposals[index];
            let (old, mut best_bits) = proposal.costs[proposal.selected];
            let mut best = proposal.selected;
            for (candidate, &(offset, residual_bits)) in proposal.costs.iter().enumerate() {
                let bits = residual_bits + replacement_cost(&counts, old, offset);
                if bits + 1e-9 * best_bits.abs().max(1.0) < best_bits {
                    best = candidate;
                    best_bits = bits;
                }
            }
            if best != proposal.selected {
                counts[UINT.encode(pack_signed(old)).token as usize] -= 1;
                counts[UINT.encode(pack_signed(proposal.costs[best].0)).token as usize] += 1;
                proposal.selected = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut nodes = tree.nodes().to_vec();
    for proposal in proposals {
        let selected = proposal.costs[proposal.selected].0;
        if selected != proposal.costs[0].0 {
            if let Node::Leaf { offset, .. } = &mut nodes[proposal.node] {
                *offset = selected;
            }
            stats.leaves += 1;
        }
    }
    Ok((
        if stats.leaves == 0 {
            None
        } else {
            Some(Tree::new(nodes)?)
        },
        stats,
    ))
}

#[cfg(test)]
#[path = "leaf_offset_tests.rs"]
mod tests;
