// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Specialize static channel/group properties without renumbering wire contexts.
use super::{Node, Predictor, Tree};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Leaf {
    pub context: usize,
    pub predictor: Predictor,
    pub offset: i32,
    pub multiplier: u32,
}

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
}

impl ChannelTree {
    pub fn new(tree: &Tree, channel: usize, group: usize) -> Self {
        let mut result = Self {
            nodes: vec![Decision::default()],
            leaves: Vec::new(),
            predictors: Vec::new(),
            weighted: false,
            previous_channels: 0,
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
