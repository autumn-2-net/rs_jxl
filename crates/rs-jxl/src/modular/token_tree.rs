// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Specialize static channel/group properties without renumbering wire contexts.
use super::{Neighbors, Node, Predictor, Tree};
use crate::{Error, Result};

// libjxl encoding.h::TreeToLookupTable uses this range after static filtering.
const GRADIENT_MIN: i32 = -8192;
const GRADIENT_MAX: i32 = 8191;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Leaf {
    pub context: usize,
    pub predictor: Predictor,
    pub offset: i32,
    pub multiplier: u32,
}

#[cfg(test)]
#[path = "token_tree/gradient_tests.rs"]
mod gradient_tests;

// Tree::new bounds nodes to 2^22 and properties to 255. Keep the hot walk
// compact; leaf payloads are touched only after the decisions are finished.
#[derive(Clone, Copy, Default)]
struct Decision {
    property: u32,
    threshold: i32,
    greater: u32,
    less_equal: u32,
}

pub(crate) struct ChannelTree {
    nodes: Vec<Decision>,
    leaves: Vec<Leaf>,
    pub predictors: Vec<Predictor>,
    pub weighted: bool,
    pub previous_channels: usize,
    gradient_lut: bool,
}

/// A bounded direct context map for the fixed Gradient/WP trees. This changes
/// only traversal; the original tree and its wire context numbers are retained.
pub(in crate::modular) struct SimpleContexts {
    pub predictor: Predictor,
    minimum: i32,
    contexts: Vec<u32>,
}

/// Compact routing for the common low-effort Weighted/Gradient learner.
pub(in crate::modular) struct PairContexts<'a> {
    nodes: Vec<Decision>,
    leaves: &'a [Leaf],
    pub gradient_lut: bool,
}

impl PairContexts<'_> {
    #[inline]
    pub fn lookup(&self, gradient: i32, left_delta: i32, wp_error: i32) -> Leaf {
        let properties = [gradient, left_delta, wp_error];
        let mut index = 0;
        loop {
            let node = self.nodes[index];
            if node.property == u32::MAX {
                return self.leaves[node.greater as usize];
            }
            index = if properties[node.property as usize] > node.threshold {
                node.greater
            } else {
                node.less_equal
            } as usize;
        }
    }
}

impl SimpleContexts {
    #[inline]
    pub fn lookup(&self, value: i64) -> u32 {
        let index =
            (value - i64::from(self.minimum)).clamp(0, (self.contexts.len() - 1) as i64) as usize;
        self.contexts[index]
    }
}

impl ChannelTree {
    /// Conservative across static channel/group choices. Offsets can switch
    /// gradient-LUT routing to generic wrapping properties (or back again).
    pub(in crate::modular) fn offset_sensitive_routing(tree: &Tree) -> bool {
        // Flags: 1 = eligible specialization; 2 = it contains a property-9
        // decision. A static Gradient leaf alone has no routing distinction.
        let mut pending = Vec::<(usize, bool, Option<u8>)>::new();
        let mut index = 0;
        'visit: loop {
            let mut flags = match tree.nodes[index] {
                Node::Leaf {
                    predictor,
                    offset,
                    multiplier,
                } => u8::from(predictor == Predictor::Gradient && offset == 0 && multiplier == 1),
                Node::Split {
                    property,
                    threshold,
                    greater,
                    less_equal,
                } => {
                    if property <= 1
                        || (property == 9 && (GRADIENT_MIN - 1..GRADIENT_MAX).contains(&threshold))
                    {
                        pending.push((less_equal, property == 9, None));
                        index = greater;
                        continue;
                    }
                    0
                }
            };
            while let Some((second, both, first)) = pending.pop() {
                if let Some(first) = first {
                    flags = if both {
                        if first & flags & 1 != 0 { 3 } else { 0 }
                    } else {
                        first | flags
                    };
                } else if (both && flags == 0) || (!both && flags == 3) {
                    continue;
                } else {
                    pending.push((0, both, Some(flags)));
                    index = second;
                    continue 'visit;
                }
            }
            return flags & 2 != 0;
        }
    }

    pub fn new(tree: &Tree, channel: usize, group: usize) -> Self {
        let mut result = Self {
            nodes: vec![Decision::default()],
            leaves: Vec::new(),
            predictors: Vec::new(),
            weighted: false,
            previous_channels: 0,
            gradient_lut: true,
        };
        let mut pending = vec![(0, 0)];
        while let Some((mut source, destination)) = pending.pop() {
            while let Node::Split {
                property,
                threshold,
                greater,
                less_equal,
            } = tree.nodes[source]
            {
                if property > 1 {
                    break;
                }
                let value = if property == 0 { channel } else { group } as i32;
                source = if value > threshold {
                    greater
                } else {
                    less_equal
                };
            }
            result.nodes[destination] = match tree.nodes[source] {
                Node::Leaf {
                    predictor,
                    offset,
                    multiplier,
                } => {
                    result.gradient_lut &=
                        predictor == Predictor::Gradient && offset == 0 && multiplier == 1;
                    result.weighted |= predictor == Predictor::Weighted;
                    if !predictor.has_weighted() && !result.predictors.contains(&predictor) {
                        result.predictors.push(predictor);
                    }
                    let index = result.leaves.len();
                    result.leaves.push(Leaf {
                        context: tree.contexts[source],
                        predictor,
                        offset,
                        multiplier,
                    });
                    Decision {
                        property: u32::MAX,
                        greater: index as u32,
                        ..Default::default()
                    }
                }
                Node::Split {
                    property,
                    threshold,
                    greater,
                    less_equal,
                } => {
                    result.gradient_lut &=
                        property == 9 && (GRADIENT_MIN - 1..GRADIENT_MAX).contains(&threshold);
                    result.weighted |= property == 15;
                    if property >= 16 {
                        result.previous_channels =
                            result.previous_channels.max((property - 16) / 4 + 1);
                    }
                    let first = result.nodes.len();
                    result.nodes.push(Decision::default());
                    result.nodes.push(Decision::default());
                    pending.push((less_equal, first + 1));
                    pending.push((greater, first));
                    Decision {
                        property: property as u32,
                        threshold,
                        greater: first as u32,
                        less_equal: (first + 1) as u32,
                    }
                }
            };
        }
        result
    }

    pub fn leaf(&self) -> Option<Leaf> {
        if self.nodes[0].property == u32::MAX {
            Some(self.leaves[self.nodes[0].greater as usize])
        } else {
            None
        }
    }

    pub fn pair_contexts(&self) -> Option<PairContexts<'_>> {
        if self
            .nodes
            .iter()
            .any(|n| !matches!(n.property, 9 | 10 | 15 | u32::MAX))
            || self.leaves.iter().any(|leaf| {
                !matches!(leaf.predictor, Predictor::Gradient | Predictor::Weighted)
                    || leaf.offset != 0
                    || leaf.multiplier != 1
            })
        {
            return None;
        }
        let nodes = self
            .nodes
            .iter()
            .map(|node| Decision {
                property: match node.property {
                    9 => 0,
                    10 => 1,
                    15 => 2,
                    _ => u32::MAX,
                },
                ..*node
            })
            .collect();
        Some(PairContexts {
            nodes,
            leaves: &self.leaves,
            gradient_lut: self.gradient_lut,
        })
    }

    pub fn simple_contexts(&self) -> Option<SimpleContexts> {
        let predictor = self.leaves.first()?.predictor;
        let property = match predictor {
            Predictor::Gradient if self.gradient_lut => 9,
            Predictor::Weighted => 15,
            _ => return None,
        };
        if self
            .leaves
            .iter()
            .any(|leaf| leaf.predictor != predictor || leaf.offset != 0 || leaf.multiplier != 1)
        {
            return None;
        }
        let mut minimum = i32::MAX;
        let mut maximum = i32::MIN;
        for node in &self.nodes {
            if node.property == u32::MAX {
                continue;
            }
            if node.property != property {
                return None;
            }
            minimum = minimum.min(node.threshold);
            maximum = maximum.max(node.threshold);
        }
        if minimum == i32::MAX {
            minimum = 0;
            maximum = 0;
        } else {
            maximum = maximum.checked_add(1)?;
        }
        let length = usize::try_from(i64::from(maximum) - i64::from(minimum) + 1).ok()?;
        if length > 16384 {
            return None;
        }
        let mut contexts = vec![0; length];
        let mut pending = vec![(0usize, minimum, maximum)];
        while let Some((index, low, high)) = pending.pop() {
            let node = self.nodes[index];
            if node.property == u32::MAX {
                contexts[(low - minimum) as usize..=(high - minimum) as usize]
                    .fill(self.leaves[node.greater as usize].context as u32);
            } else {
                if low <= node.threshold {
                    pending.push((node.less_equal as usize, low, high.min(node.threshold)));
                }
                if high > node.threshold {
                    pending.push((node.greater as usize, low.max(node.threshold + 1), high));
                }
            }
        }
        Some(SimpleContexts {
            predictor,
            minimum,
            contexts,
        })
    }

    pub fn lookup(&self, properties: &[i32]) -> Result<Leaf> {
        let mut index = 0;
        loop {
            let node = self.nodes[index];
            if node.property == u32::MAX {
                return Ok(self.leaves[node.greater as usize]);
            }
            let value = *properties
                .get(node.property as usize)
                .ok_or_else(|| Error::Encoding("missing MA property".into()))?;
            index = if value > node.threshold {
                node.greater
            } else {
                node.less_equal
            } as usize;
        }
    }

    pub fn lookup_pixel(&self, properties: &[i32], neighbors: Neighbors) -> Result<Leaf> {
        if !self.gradient_lut {
            return self.lookup(properties);
        }
        // Generic property storage wraps to i32; libjxl's gradient LUT instead
        // clamps a wide intermediate. Mirror that distinction only at emission.
        let gradient = (i64::from(neighbors.left) + i64::from(neighbors.top)
            - i64::from(neighbors.topleft))
        .clamp(i64::from(GRADIENT_MIN), i64::from(GRADIENT_MAX)) as i32;
        let mut index = 0;
        loop {
            let node = self.nodes[index];
            if node.property == u32::MAX {
                return Ok(self.leaves[node.greater as usize]);
            }
            index = if gradient > node.threshold {
                node.greater
            } else {
                node.less_equal
            } as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_decisions_fit_four_per_cache_line() {
        assert_eq!(std::mem::size_of::<Decision>(), 16);
    }
    #[test]
    fn specialization_keeps_wire_context_and_prunes_unreachable_weighted_paths() {
        let tree = Tree::new(vec![
            Node::Split {
                property: 0,
                threshold: 0,
                greater: 1,
                less_equal: 2,
            },
            Node::Split {
                property: 1,
                threshold: 18,
                greater: 3,
                less_equal: 4,
            },
            Node::Leaf {
                predictor: Predictor::Weighted,
                offset: 0,
                multiplier: 1,
            },
            Node::Split {
                property: 17,
                threshold: 4,
                greater: 5,
                less_equal: 6,
            },
            Node::Leaf {
                predictor: Predictor::Top,
                offset: 3,
                multiplier: 2,
            },
            Node::Leaf {
                predictor: Predictor::Gradient,
                offset: -7,
                multiplier: 1,
            },
            Node::Leaf {
                predictor: Predictor::Zero,
                offset: 9,
                multiplier: 3,
            },
        ])
        .unwrap();
        for channel in 0..3 {
            for group in [0, 18, 19] {
                let plan = ChannelTree::new(&tree, channel, group);
                assert_eq!(plan.weighted, channel == 0);
                assert_eq!(
                    plan.previous_channels,
                    usize::from(channel > 0 && group > 18)
                );
                for value in [-100, 4, 5, 100] {
                    let mut properties = [0; 20];
                    properties[0] = channel as i32;
                    properties[1] = group as i32;
                    properties[17] = value;
                    let a = tree.lookup(&properties).unwrap();
                    let b = plan.lookup(&properties).unwrap();
                    assert_eq!(a, (b.context, b.predictor, b.offset, b.multiplier));
                }
            }
        }
    }
}
