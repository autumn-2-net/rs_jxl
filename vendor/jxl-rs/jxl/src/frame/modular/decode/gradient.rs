// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::VecDeque;

use crate::error::Result;
use crate::frame::modular::Predictor;
use crate::frame::modular::predict::PredictionData;
use crate::frame::modular::tree::TreeNode;

const PROPERTY_MIN: i32 = -8192;
const PROPERTY_MAX: i32 = 8191;

/// Semantic eligibility from libjxl FilterTree + TreeToLookupTable, independent
/// of the smaller Rust LUT optimization. Static branches are pruned, but all
/// remaining branches (even empty LUT intervals) must pass these checks.
pub(super) fn eligible(
    nodes: &[TreeNode],
    channel: usize,
    stream: usize,
    queue: &mut VecDeque<u32>,
) -> Result<bool> {
    queue.clear();
    queue.try_reserve(nodes.len())?;
    queue.push_back(0);
    while let Some(pos) = queue.pop_front() {
        match nodes[pos as usize] {
            TreeNode::Split {
                property,
                val,
                left,
                right,
            } if property < 2 => {
                let value = (if property == 0 { channel } else { stream }) as i32;
                queue.push_front(if value > val { left } else { right });
            }
            TreeNode::Split {
                property: 9,
                val,
                left,
                right,
            } if (PROPERTY_MIN - 1..PROPERTY_MAX).contains(&val) => {
                // C++ uses absolute range bounds, not strict subdivision of
                // the current interval. In particular, -8193 is eligible.
                queue.push_back(left);
                queue.push_back(right);
            }
            TreeNode::Leaf {
                predictor: Predictor::Gradient,
                offset: 0,
                multiplier: 1,
                ..
            } => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

#[inline(always)]
pub(super) fn property(data: PredictionData) -> i32 {
    (data.left as i64 + data.top as i64 - data.topleft as i64)
        .clamp(PROPERTY_MIN as i64, PROPERTY_MAX as i64) as i32
}

/// The tree is already statically pruned, clustered, and eligible. Clamping is
/// required even for traversal: a split at -8193 has an empty lower LUT branch.
#[inline(always)]
pub(super) fn context(nodes: &[TreeNode], property: i32) -> u32 {
    let mut pos = 0;
    loop {
        match nodes[pos] {
            TreeNode::Split {
                val, left, right, ..
            } => pos = (if property > val { left } else { right }) as usize,
            TreeNode::Leaf { id, .. } => return id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(id: u32) -> TreeNode {
        TreeNode::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
            id,
        }
    }

    fn split(property: u8, val: i32, left: u32, right: u32) -> TreeNode {
        TreeNode::Split {
            property,
            val,
            left,
            right,
        }
    }

    fn is_eligible(nodes: &[TreeNode], channel: usize, stream: usize) -> bool {
        eligible(nodes, channel, stream, &mut VecDeque::new()).unwrap()
    }

    #[test]
    fn exact_cpp_threshold_bounds() {
        for (threshold, expected) in [
            (i32::MIN, false),
            (-8194, false),
            (-8193, true),
            (-8192, true),
            (-1024, true),
            (1023, true),
            (8190, true),
            (8191, false),
            (i32::MAX, false),
        ] {
            let tree = [split(9, threshold, 1, 2), leaf(1), leaf(2)];
            assert_eq!(is_eligible(&tree, 0, 0), expected, "{threshold}");
        }
    }

    #[test]
    fn empty_intervals_do_not_disqualify_a_tree() {
        // The root's lower side is empty in the LUT domain. C++ still checks
        // its leaves, and does not reject a redundant split within the domain.
        let tree = [
            split(9, -8193, 1, 2),
            leaf(1),
            split(9, 0, 3, 4),
            leaf(3),
            leaf(4),
        ];
        assert!(is_eligible(&tree, 0, 0));
        let mut bad_leaf = tree;
        if let TreeNode::Leaf { offset, .. } = &mut bad_leaf[3] {
            *offset = 1;
        }
        assert!(!is_eligible(&bad_leaf, 0, 0));
    }

    #[test]
    fn only_property9_and_unit_gradient_leaves_qualify() {
        for (predictor, offset, multiplier) in [
            (Predictor::Gradient, 0, 1),
            (Predictor::Gradient, 1, 1),
            (Predictor::Gradient, -1, 1),
            (Predictor::Gradient, 0, 2),
            (Predictor::Weighted, 0, 1),
            (Predictor::Zero, 0, 1),
        ] {
            let tree = [
                split(9, 0, 1, 2),
                leaf(1),
                TreeNode::Leaf {
                    predictor,
                    offset,
                    multiplier,
                    id: 2,
                },
            ];
            assert_eq!(
                is_eligible(&tree, 0, 0),
                predictor == Predictor::Gradient && offset == 0 && multiplier == 1
            );
        }
        for property in 2..=u8::MAX {
            let tree = [split(property, 0, 1, 2), leaf(1), leaf(2)];
            assert_eq!(is_eligible(&tree, 0, 0), property == 9);
        }
    }

    #[test]
    fn static_pruning_precedes_eligibility() {
        let tree = [
            split(0, 0, 1, 2),
            split(1, 7, 3, 4),
            split(15, 0, 5, 6),
            split(9, 7000, 7, 8),
            split(9, 8191, 9, 10),
            leaf(5),
            leaf(6),
            leaf(7),
            leaf(8),
            leaf(9),
            leaf(10),
        ];
        assert!(is_eligible(&tree, 1, 8));
        assert!(!is_eligible(&tree, 1, 7));
        assert!(!is_eligible(&tree, 0, 8));
    }

    #[test]
    fn wide_property_is_clamped_before_traversal() {
        let tree = [split(9, 0, 1, 2), leaf(1), leaf(2)];
        for (left, top, topleft, expected) in [
            (0x4000_0000, 0x4000_0000, 0, 8191),
            (i32::MIN, i32::MIN, i32::MAX, -8192),
            (i32::MAX, i32::MAX, i32::MIN, 8191),
            (10, 20, 5, 25),
        ] {
            let data = PredictionData {
                left,
                top,
                topleft,
                ..PredictionData::default()
            };
            assert_eq!(property(data), expected);
            assert_eq!(
                context(&tree, property(data)),
                if expected > 0 { 1 } else { 2 }
            );
        }
        let edge_tree = [split(9, -8193, 1, 2), leaf(1), leaf(2)];
        assert_eq!(context(&edge_tree, PROPERTY_MIN), 1);
    }
}
