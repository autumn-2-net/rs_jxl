// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; the complete upstream notice is in docs/PORTING_KERNELS.md.

//! Predefined Modular pixel trees from frozen b866b3defec13494bfbaf705db93581cda118e37,
//! lib/jxl/modular/encoding/enc_encoding.cc::MakeFixedTree and PredefinedTree.
//!
//! This is the explicit fixed-tree path, not a substitute for MA learning.
//! Zero maps to kTrivialTreeNoPredictor; Weighted and Gradient map to kWPFixedDC
//! and kGradientFixedDC. VarDCT AC-metadata/transcode trees are outside this
//! pixel-lossless module's scope.
//!
//! total_pixels is the sum of eligible transformed channel areas for the stream,
//! not the image's spatial area or the number of MA samples. As in the source,
//! count metadata channels even when oversized, stop at the first oversized
//! non-metadata channel, and normalize an empty population to one. The caller
//! supplies this count and the stream's original bit depth. No adaptive-training
//! leaf limit is applied: the source full template has 34 leaves.

use std::collections::VecDeque;

use crate::{Error, Result};

use super::predictor::Predictor;
use super::tree::{Node, Tree};

const CUTOFFS: [i32; 33] = [
    -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0, 1, 3, 5, 7,
    11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedTreeMode {
    /// PredefinedTree(kTrivialTreeNoPredictor): one Zero leaf.
    Zero,
    /// PredefinedTree(kWPFixedDC): split on weighted prediction error (15).
    Weighted { total_pixels: usize },
    /// PredefinedTree(kGradientFixedDC): property 9, or property 18 when any
    /// previous-channel properties are configured. The latter is deliberately
    /// the first reference's absolute gradient residual, not property 19.
    Gradient {
        total_pixels: usize,
        previous_channels: usize,
    },
}

/// Build the source template, retaining breadth-first greater-child-first order.
/// `bits` is 1..=32. Zero total pixels uses the source caller's max(total, 1).
/// A nonzero previous-channel count requires property 18 at decoder lookup,
/// even when a particular channel has no matching reference (then its value is 0).
pub fn build(bits: u8, mode: FixedTreeMode) -> Result<Tree> {
    if !(1..=32).contains(&bits) {
        return Err(Error::InvalidInput(
            "fixed MA bit depth must be 1..=32".into(),
        ));
    }
    let (property, predictor, total_pixels) = match mode {
        FixedTreeMode::Zero => return Tree::new(vec![leaf(Predictor::Zero)]),
        FixedTreeMode::Weighted { total_pixels } => (15, Predictor::Weighted, total_pixels),
        FixedTreeMode::Gradient {
            total_pixels,
            previous_channels,
        } => (
            if previous_channels > 0 { 18 } else { 9 },
            Predictor::Gradient,
            total_pixels,
        ),
    };
    // CeilLog2Nonzero without rounding total_pixels up (which could overflow).
    let log_pixels = usize::BITS - (total_pixels.max(1) - 1).leading_zeros();
    let min_gap = 8 * 14usize.saturating_sub(log_pixels as usize);
    let multiplier = 1i32 << bits.saturating_sub(11).min(4);
    let mut nodes = vec![leaf(predictor)];
    let mut queue = VecDeque::from([(0usize, CUTOFFS.len(), 0usize)]);
    while let Some((begin, end, position)) = queue.pop_front() {
        if begin + min_gap >= end {
            continue;
        }
        let split = (begin + end) / 2;
        let greater = nodes.len();
        let less_equal = greater + 1;
        nodes[position] = Node::Split {
            property,
            threshold: CUTOFFS[split] * multiplier,
            greater,
            less_equal,
        };
        queue.push_back((split + 1, end, greater));
        nodes.push(leaf(predictor));
        queue.push_back((begin, split, less_equal));
        nodes.push(leaf(predictor));
    }
    Tree::new(nodes)
}

fn leaf(predictor: Predictor) -> Node {
    Node::Leaf {
        predictor,
        offset: 0,
        multiplier: 1,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::super::tree::pack_signed;
    use super::*;

    fn signature(tree: &Tree) -> Vec<(usize, u32)> {
        tree.tokens()
            .iter()
            .map(|token| (token.context, token.value))
            .collect()
    }

    fn splits(tree: &Tree) -> Vec<(usize, i32)> {
        let tokens = tree.tokens();
        let mut result = Vec::new();
        let mut position = 0;
        while position < tokens.len() {
            assert_eq!(tokens[position].context, 1);
            let property = tokens[position].value;
            if property == 0 {
                position += 5;
            } else {
                let packed = tokens[position + 1].value;
                assert_eq!(tokens[position + 1].context, 0);
                result.push((
                    property as usize - 1,
                    (packed >> 1) as i32 ^ -((packed & 1) as i32),
                ));
                position += 2;
            }
        }
        result
    }

    #[test]
    fn source_zero_template_is_one_zero_predictor_leaf() {
        for bits in [1, 8, 16, 32] {
            let tree = build(bits, FixedTreeMode::Zero).unwrap();
            assert_eq!(signature(&tree), [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
            assert_eq!(tree.lookup(&[]).unwrap(), (0, Predictor::Zero, 0, 1));
            assert!(!tree.uses_weighted());
        }
    }

    #[test]
    fn small_weighted_and_gradient_templates_keep_their_predictor() {
        for (mode, predictor) in [
            (
                FixedTreeMode::Weighted { total_pixels: 0 },
                Predictor::Weighted,
            ),
            (
                FixedTreeMode::Gradient {
                    total_pixels: 0,
                    previous_channels: 1,
                },
                Predictor::Gradient,
            ),
        ] {
            let tree = build(16, mode).unwrap();
            assert_eq!(tree.leaves(), 1);
            assert_eq!(tree.lookup(&[]).unwrap(), (0, predictor, 0, 1));
            assert_eq!(tree.uses_weighted(), predictor == Predictor::Weighted);
        }
    }

    #[test]
    fn source_population_pruning_uses_ceil_log2_at_every_transition() {
        for (pixels, leaves) in [
            (0, 1),
            (1, 1),
            (256, 1),
            (512, 1),
            (513, 2),
            (1024, 2),
            (1025, 2),
            (2048, 2),
            (2049, 2),
            (4096, 2),
            (4097, 4),
            (8192, 4),
            (8193, 34),
            (16_384, 34),
            (usize::MAX, 34),
        ] {
            for mode in [
                FixedTreeMode::Weighted {
                    total_pixels: pixels,
                },
                FixedTreeMode::Gradient {
                    total_pixels: pixels,
                    previous_channels: 0,
                },
            ] {
                assert_eq!(
                    build(8, mode).unwrap().leaves(),
                    leaves,
                    "population {pixels}"
                );
            }
        }
    }

    #[test]
    fn pruned_tree_serialization_and_context_order_match_source_breadth_first() {
        let tree = build(8, FixedTreeMode::Weighted { total_pixels: 4097 }).unwrap();
        assert_eq!(splits(&tree), [(15, 0), (15, 47), (15, -31)]);
        let mut expected = vec![
            (1, 16),
            (0, 0),
            (1, 16),
            (0, pack_signed(47)),
            (1, 16),
            (0, pack_signed(-31)),
        ];
        for _ in 0..4 {
            expected.extend([(1, 0), (2, 6), (3, 0), (4, 0), (5, 0)]);
        }
        assert_eq!(signature(&tree), expected);
        for (value, context) in [(48, 0), (47, 1), (1, 1), (0, 2), (-30, 2), (-31, 3)] {
            let mut properties = [0; 19];
            properties[15] = value;
            assert_eq!(
                tree.lookup(&properties).unwrap(),
                (context, Predictor::Weighted, 0, 1)
            );
        }
    }

    #[test]
    fn full_templates_have_the_source_cutoffs_and_inclusive_thresholds() {
        let expected_cutoffs = [
            -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0, 1,
            3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
        ];
        for (mode, property, predictor) in [
            (
                FixedTreeMode::Weighted {
                    total_pixels: 16_384,
                },
                15,
                Predictor::Weighted,
            ),
            (
                FixedTreeMode::Gradient {
                    total_pixels: 16_384,
                    previous_channels: 0,
                },
                9,
                Predictor::Gradient,
            ),
            (
                FixedTreeMode::Gradient {
                    total_pixels: 16_384,
                    previous_channels: 2,
                },
                18,
                Predictor::Gradient,
            ),
        ] {
            let tree = build(8, mode).unwrap();
            assert_eq!(tree.predictors(), [predictor]);
            let mut thresholds: Vec<_> = splits(&tree)
                .into_iter()
                .map(|(p, threshold)| {
                    assert_eq!(p, property);
                    threshold
                })
                .collect();
            thresholds.sort_unstable();
            assert_eq!(thresholds, expected_cutoffs);
            let mut properties = [0; 19];
            let mut contexts = BTreeSet::new();
            properties[property] = i32::MIN;
            let mut previous = tree.lookup(&properties).unwrap();
            contexts.insert(previous.0);
            for cutoff in expected_cutoffs {
                properties[property] = cutoff;
                assert_eq!(tree.lookup(&properties).unwrap(), previous);
                properties[property] = cutoff + 1;
                let above = tree.lookup(&properties).unwrap();
                assert_ne!(above.0, previous.0);
                assert_eq!((above.1, above.2, above.3), (predictor, 0, 1));
                contexts.insert(above.0);
                previous = above;
            }
            properties[property] = i32::MAX;
            assert_eq!(tree.lookup(&properties).unwrap(), previous);
            assert_eq!(contexts, (0..34).collect());
        }
    }

    #[test]
    fn bit_depth_scales_signed_cutoffs_but_never_leaf_multipliers() {
        for (bits, scale) in [
            (1, 1),
            (8, 1),
            (11, 1),
            (12, 2),
            (13, 4),
            (14, 8),
            (15, 16),
            (16, 16),
            (32, 16),
        ] {
            let tree = build(
                bits,
                FixedTreeMode::Gradient {
                    total_pixels: 16_384,
                    previous_channels: 0,
                },
            )
            .unwrap();
            let mut actual: Vec<_> = splits(&tree).into_iter().map(|(_, v)| v).collect();
            actual.sort_unstable();
            assert_eq!(actual, CUTOFFS.map(|v| v * scale));
            for value in [i32::MIN, -8000, -1, 0, 1, 8000, i32::MAX] {
                let mut properties = [0; 19];
                properties[9] = value;
                assert_eq!(tree.lookup(&properties).unwrap().3, 1);
            }
        }
    }

    #[test]
    fn previous_channel_flag_selects_absolute_reference_property_not_its_signed_neighbor() {
        let tree = build(
            8,
            FixedTreeMode::Gradient {
                total_pixels: 1024,
                previous_channels: 1,
            },
        )
        .unwrap();
        assert_eq!(splits(&tree), [(18, 0)]);
        let mut properties = [0; 20];
        let initial = tree.lookup(&properties).unwrap();
        properties[9] = 123;
        properties[19] = 123;
        assert_eq!(tree.lookup(&properties).unwrap(), initial);
        properties[18] = 1;
        assert_ne!(tree.lookup(&properties).unwrap().0, initial.0);
        assert_eq!(
            signature(&tree),
            signature(
                &build(
                    8,
                    FixedTreeMode::Gradient {
                        total_pixels: 1024,
                        previous_channels: usize::MAX
                    }
                )
                .unwrap()
            )
        );
        assert!(!tree.uses_weighted());
    }

    #[test]
    fn invalid_bit_depth_is_rejected_for_all_modes() {
        for bits in [0, 33, u8::MAX] {
            for mode in [
                FixedTreeMode::Zero,
                FixedTreeMode::Weighted { total_pixels: 1 },
                FixedTreeMode::Gradient {
                    total_pixels: 1,
                    previous_channels: 0,
                },
            ] {
                assert!(build(bits, mode).is_err());
            }
        }
    }
}
