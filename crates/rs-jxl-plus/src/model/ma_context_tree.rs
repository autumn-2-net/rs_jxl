//! MA partitions for dynamic calibration of an already predicted binary event.
//!
//! The base probability is an offset in the fitting objective. Training does
//! not encode coefficients, replace the base model, or store fitted biases in
//! tree leaves. Runtime calibration starts afresh in each selected context.

use crate::{Error, Result};
use rs_jxl::modular::{
    predictor::Predictor,
    tree::{Node, Tree},
};

const BIASES: [i32; 9] = [-256, -128, -64, -32, 0, 32, 64, 128, 256];
const MIN_CHILD: usize = 128;
const MIN_GAIN_BITS: f64 = 16.0;
const MAX_THRESHOLDS: usize = 32;
const MAX_LEAVES: usize = 8;
const MAX_OBSERVATIONS: usize = 32768;
const PROBABILITY_BINS: usize = 16;
// Summation order can perturb mathematically equal gains at roundoff scale.
// Resolve those ties by source node, property, then threshold instead.
const TIE_EPSILON: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
pub(super) struct Observation {
    pub(super) properties: [i32; 8],
    pub(super) probability: u32,
    pub(super) bit: i32,
}

/// Integer odds correction using the same frozen squash table as the model.
/// Biases saturate to the permitted range. A zero bias is exactly the original
/// probability, with no squash/stretch quantization or rounding.
pub(super) fn corrected_probability(p: u32, delta: i32) -> u32 {
    if delta == 0 {
        return p;
    }
    let p = p.clamp(1, 65535) as u64;
    let delta = delta.clamp(-512, 512);
    let s = crate::probability::tables().squash[(2048 + delta) as usize] as u64;
    let one = p * s;
    let denominator = one + (65536 - p) * (4096 - s);
    ((65536 * one + denominator / 2) / denominator).clamp(1, 65535) as u32
}

type BiasLosses = [f64; BIASES.len()];
type Losses = [BiasLosses; PROBABILITY_BINS];
struct Leaf {
    node: usize,
    observations: Vec<usize>,
    losses: Losses,
}
#[derive(Clone, Copy)]
struct Split {
    leaf: usize,
    node: usize,
    property: usize,
    threshold: i32,
    gain: f64,
}
impl Split {
    fn improves(self, previous: Option<Self>) -> bool {
        previous.is_none_or(|old| {
            self.gain > old.gain + TIE_EPSILON
                || ((self.gain - old.gain).abs() <= TIE_EPSILON
                    && (self.node, self.property, self.threshold)
                        < (old.node, old.property, old.threshold))
        })
    }
}

fn leaf_node() -> Node {
    Node::Leaf {
        predictor: Predictor::Zero,
        offset: 0,
        multiplier: 1,
    }
}
fn best_loss(losses: &Losses) -> f64 {
    // Runtime already calibrates each base-probability bin independently.
    // A tree split must improve on that baseline, rather than rediscover it.
    losses
        .iter()
        .map(|bin| bin.iter().copied().fold(f64::INFINITY, f64::min))
        .sum()
}
fn add(sum: &mut Losses, probability: u32, row: &BiasLosses) {
    for (total, &value) in sum[(probability >> 12) as usize].iter_mut().zip(row) {
        *total += value;
    }
}
fn reserved<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("MA calibration training allocation"))?;
    Ok(values)
}

fn best_split(
    leaf_index: usize,
    leaf: &Leaf,
    observations: &[Observation],
    losses: &[BiasLosses],
) -> Result<Option<Split>> {
    let n = leaf.observations.len();
    if n < 2 * MIN_CHILD {
        return Ok(None);
    }
    let parent_loss = best_loss(&leaf.losses);
    let mut best = None;
    let mut sorted = reserved(n)?;
    for property in 0..8 {
        sorted.clear();
        sorted.extend_from_slice(&leaf.observations);
        // The explicit observation index is a deterministic secondary key.
        sorted.sort_unstable_by_key(|&i| (observations[i].properties[property], i));
        if observations[sorted[0]].properties[property]
            == observations[sorted[n - 1]].properties[property]
        {
            continue;
        }

        // Quantiles span only ranks that can satisfy the child-size bound.
        // Move duplicate values to their final position: a threshold must
        // never divide observations having the same property value.
        let mut boundaries = Vec::with_capacity(MAX_THRESHOLDS);
        for q in 0..MAX_THRESHOLDS {
            let rank = MIN_CHILD + (n - 2 * MIN_CHILD) * q / (MAX_THRESHOLDS - 1);
            let threshold = observations[sorted[rank - 1]].properties[property];
            let count =
                sorted.partition_point(|&i| observations[i].properties[property] <= threshold);
            if count >= MIN_CHILD
                && n - count >= MIN_CHILD
                && boundaries
                    .last()
                    .is_none_or(|&(previous, _)| previous != count)
            {
                boundaries.push((count, threshold));
            }
        }
        let mut prefix = [[0.0; BIASES.len()]; PROBABILITY_BINS];
        let mut consumed = 0;
        for (count, threshold) in boundaries {
            for &i in &sorted[consumed..count] {
                add(&mut prefix, observations[i].probability, &losses[i]);
            }
            consumed = count;
            let mut suffix = [[0.0; BIASES.len()]; PROBABILITY_BINS];
            for bin in 0..PROBABILITY_BINS {
                for b in 0..BIASES.len() {
                    suffix[bin][b] = leaf.losses[bin][b] - prefix[bin][b];
                }
            }
            let gain = parent_loss - best_loss(&prefix) - best_loss(&suffix);
            let candidate = Split {
                leaf: leaf_index,
                node: leaf.node,
                property,
                threshold,
                gain,
            };
            if gain >= MIN_GAIN_BITS && candidate.improves(best) {
                best = Some(candidate);
            }
        }
    }
    Ok(best)
}

/// Greedily select at most eight contexts by conditional Bernoulli log loss,
/// fitting nine candidate biases independently within sixteen probability bins.
/// Only properties 0..7 occur in splits; every leaf is an inert MA leaf whose
/// context ID selects an independent runtime calibration state.
pub(super) fn train(observations: &[Observation], max_leaves: usize) -> Result<Tree> {
    if max_leaves == 0 || max_leaves > MAX_LEAVES {
        return Err(Error::Invalid("MA calibration leaf cap"));
    }
    if observations.len() > MAX_OBSERVATIONS {
        return Err(Error::Limit("MA calibration training observations"));
    }
    let mut losses = reserved(observations.len())?;
    let mut total = [[0.0; BIASES.len()]; PROBABILITY_BINS];
    for observation in observations {
        if !(1..=65535).contains(&observation.probability) || !matches!(observation.bit, 0 | 1) {
            return Err(Error::Invalid("MA calibration observation"));
        }
        let row = BIASES.map(|bias| {
            let q = corrected_probability(observation.probability, bias);
            let likelihood = if observation.bit != 0 { q } else { 65536 - q };
            16.0 - (likelihood as f64).log2()
        });
        add(&mut total, observation.probability, &row);
        losses.push(row);
    }
    let mut indices = reserved(observations.len())?;
    indices.extend(0..observations.len());
    let mut nodes = vec![leaf_node()];
    let mut leaves = vec![Leaf {
        node: 0,
        observations: indices,
        losses: total,
    }];
    while leaves.len() < max_leaves {
        let mut best = None;
        for (i, leaf) in leaves.iter().enumerate() {
            if let Some(candidate) = best_split(i, leaf, observations, &losses)?
                && candidate.improves(best)
            {
                best = Some(candidate);
            }
        }
        let Some(split) = best else { break };
        let leaf = leaves.remove(split.leaf);
        let mut greater = Leaf {
            node: nodes.len(),
            observations: reserved(leaf.observations.len())?,
            losses: [[0.0; BIASES.len()]; PROBABILITY_BINS],
        };
        let mut less_equal = Leaf {
            node: nodes.len() + 1,
            observations: reserved(leaf.observations.len())?,
            losses: [[0.0; BIASES.len()]; PROBABILITY_BINS],
        };
        for i in leaf.observations {
            let child = if observations[i].properties[split.property] > split.threshold {
                &mut greater
            } else {
                &mut less_equal
            };
            child.observations.push(i);
            add(&mut child.losses, observations[i].probability, &losses[i]);
        }
        nodes[split.node] = Node::Split {
            property: split.property,
            threshold: split.threshold,
            greater: greater.node,
            less_equal: less_equal.node,
        };
        nodes.push(leaf_node());
        nodes.push(leaf_node());
        leaves.push(greater);
        leaves.push(less_equal);
    }
    Tree::new(nodes).map_err(|_| Error::Internal("MA calibration tree construction"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observation(group: i32, probability: u32, bit: i32) -> Observation {
        Observation {
            properties: [group, 0, 0, 0, 0, 0, 0, 0],
            probability,
            bit,
        }
    }
    #[test]
    fn equal_base_probability_opposite_errors_split() {
        let mut observations = Vec::new();
        for group in 0..2 {
            for _ in 0..256 {
                observations.push(observation(group, 32768, group));
            }
        }
        let tree = train(&observations, 8).unwrap();
        assert_eq!(tree.leaves(), 2);
        assert!(matches!(
            tree.nodes()[0],
            Node::Split {
                property: 0,
                threshold: 0,
                ..
            }
        ));
        assert_ne!(
            tree.lookup(&observations[0].properties).unwrap().0,
            tree.lookup(&observations[256].properties).unwrap().0
        );
        for node in tree.nodes() {
            if let Node::Leaf {
                predictor,
                offset,
                multiplier,
            } = node
            {
                assert_eq!(*predictor, Predictor::Zero);
                assert_eq!((*offset, *multiplier), (0, 1));
            }
        }
        assert_eq!(train(&observations, 1).unwrap().leaves(), 1);
    }
    #[test]
    fn calibrated_different_base_probabilities_do_not_split() {
        let mut observations = Vec::new();
        for i in 0..1024 {
            observations.push(observation(0, 16384, i32::from(i % 4 == 0)));
            observations.push(observation(1, 49152, i32::from(i % 4 != 0)));
        }
        assert_eq!(train(&observations, 8).unwrap().leaves(), 1);
    }
    #[test]
    fn shared_per_bin_miscalibration_with_different_mixtures_does_not_split() {
        let mut observations = Vec::new();
        for group in 0..2 {
            // Opposite bin mixtures distinguish the properties, but each bin
            // has the same error in both groups: P(1)=1/2 despite a base P(1)
            // of 1/4 or 3/4. Single's sixteen bias states already explain this.
            for (p, count) in [
                (16384, if group == 0 { 2304 } else { 256 }),
                (49152, if group == 0 { 256 } else { 2304 }),
            ] {
                for i in 0..count {
                    observations.push(observation(group, p, i % 2));
                }
            }
        }
        assert_eq!(train(&observations, 8).unwrap().leaves(), 1);
    }
    #[test]
    fn zero_bias_is_exact_and_odds_correction_is_bounded() {
        for p in 1..=65535 {
            assert_eq!(corrected_probability(p, 0), p);
        }
        for delta in [-512, -256, -32, 32, 256, 512] {
            assert_eq!(
                corrected_probability(32768, delta),
                16 * crate::probability::tables().squash[(2048 + delta) as usize] as u32
            );
            for p in [1, 2, 16384, 32768, 49152, 65534, 65535] {
                let q = corrected_probability(p, delta);
                assert!((1..=65535).contains(&q));
                if delta < 0 {
                    assert!(q <= p);
                } else {
                    assert!(q >= p);
                }
            }
        }
        assert_eq!(
            corrected_probability(32768, i32::MIN),
            corrected_probability(32768, -512)
        );
        assert_eq!(
            corrected_probability(32768, i32::MAX),
            corrected_probability(32768, 512)
        );
    }
    #[test]
    fn child_size_and_input_bounds() {
        let observations: Vec<_> = (0..255)
            .map(|i| observation(i32::from(i >= 127), 32768, i32::from(i >= 127)))
            .collect();
        assert_eq!(train(&observations, 8).unwrap().leaves(), 1);
        assert_eq!(train(&[], 8).unwrap().leaves(), 1);
        assert!(train(&[], 0).is_err());
        assert!(train(&[observation(0, 0, 0)], 8).is_err());
        assert!(train(&[observation(0, 32768, 2)], 8).is_err());
    }
    #[test]
    fn equivalent_properties_have_deterministic_ties() {
        let observations: Vec<_> = (0..512)
            .map(|i| {
                let group = i32::from(i >= 256);
                Observation {
                    properties: [group; 8],
                    probability: 32768,
                    bit: group,
                }
            })
            .collect();
        let a = train(&observations, 8).unwrap();
        let b = train(&observations, 8).unwrap();
        assert!(matches!(
            a.nodes()[0],
            Node::Split {
                property: 0,
                threshold: 0,
                ..
            }
        ));
        for observation in &observations {
            assert_eq!(
                a.lookup(&observation.properties).unwrap(),
                b.lookup(&observation.properties).unwrap()
            );
        }
    }
}
