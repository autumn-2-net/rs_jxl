// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::VecDeque;

use super::*;

fn leaf(id: u32) -> TreeNode {
    TreeNode::Leaf {
        predictor: Predictor::Gradient,
        multiplier: 1,
        offset: 0,
        id,
    }
}

fn tree(threshold: i32) -> [TreeNode; 3] {
    [
        TreeNode::Split {
            property: GRADIENT_PROPERTY,
            val: threshold,
            left: 1,
            right: 2,
        },
        leaf(1),
        leaf(2),
    ]
}

fn overflow(positive: bool) -> PredictionData {
    PredictionData {
        left: if positive { 0x4000_0000 } else { -0x4000_0001 },
        top: if positive { 0x4000_0000 } else { -0x4000_0000 },
        ..PredictionData::default()
    }
}

#[test]
fn legacy_small_lut_eligibility_is_unchanged() {
    let mut lut = [0; LUT_TABLE_SIZE];
    let mut stack = Vec::new();
    assert_eq!(LUT_MIN_SPLITVAL, -1024);
    assert_eq!(LUT_MAX_SPLITVAL, 1023);
    for (threshold, expected) in [
        (-8193, false),
        (-1025, false),
        (-1024, true),
        (-1023, true),
        (0, true),
        (1022, true),
        (1023, false),
        (8190, false),
    ] {
        assert_eq!(
            GradientOnly::<_, false>::new(&tree(threshold), ReaderGeneric, &mut lut, &mut stack)
                .is_some(),
            expected,
            "{threshold}"
        );
    }

    let redundant = [
        TreeNode::Split {
            property: 9,
            val: 0,
            left: 1,
            right: 2,
        },
        TreeNode::Split {
            property: 9,
            val: 0,
            left: 3,
            right: 4,
        },
        leaf(2),
        leaf(3),
        leaf(4),
    ];
    assert!(gradient::eligible(&redundant, 0, 0, &mut VecDeque::new()).unwrap());
    assert!(
        GradientOnly::<_, false>::new(&redundant, ReaderGeneric, &mut lut, &mut stack).is_none()
    );
    let wide =
        GradientOnly::<_, true>::new(&redundant, ReaderGeneric, &mut lut, &mut stack).unwrap();
    assert!(matches!(wide.lookup, GradientLookup::Tree(_)));
    assert_eq!(wide.cluster(overflow(true)), 3);
    assert_eq!(wide.cluster(overflow(false)), 2);
}

#[test]
fn gradient_lut_overflow_is_wide_by_default_and_wrapping_in_legacy_mode() {
    let tree = tree(0);
    let mut lut = [0; LUT_TABLE_SIZE];
    let mut stack = Vec::new();
    {
        let wide =
            GradientOnly::<_, true>::new(&tree, ReaderGeneric, &mut lut, &mut stack).unwrap();
        assert!(matches!(wide.lookup, GradientLookup::Lut(_)));
        assert_eq!(wide.cluster(overflow(true)), 1);
        assert_eq!(wide.cluster(overflow(false)), 2);
    }
    let legacy = GradientOnly::<_, false>::new(&tree, ReaderGeneric, &mut lut, &mut stack).unwrap();
    assert!(matches!(legacy.lookup, GradientLookup::Lut(_)));
    assert_eq!(legacy.cluster(overflow(true)), 2);
    assert_eq!(legacy.cluster(overflow(false)), 1);
}

#[test]
fn eligible_trees_outside_the_small_lut_use_clamped_wide_traversal() {
    let mut lut = [0; LUT_TABLE_SIZE];
    let mut stack = Vec::new();
    for threshold in [-8193, -8192, -1025, 1023, 4096, 8190] {
        let tree = tree(threshold);
        assert!(gradient::eligible(&tree, 0, 0, &mut VecDeque::new()).unwrap());
        assert!(
            GradientOnly::<_, false>::new(&tree, ReaderGeneric, &mut lut, &mut stack).is_none()
        );
        let wide =
            GradientOnly::<_, true>::new(&tree, ReaderGeneric, &mut lut, &mut stack).unwrap();
        assert!(matches!(wide.lookup, GradientLookup::Tree(_)));
        assert_eq!(wide.cluster(overflow(true)), 1, "{threshold}");
        assert_eq!(
            wide.cluster(overflow(false)),
            if threshold == -8193 { 1 } else { 2 },
            "{threshold}"
        );
    }
}

#[test]
fn generic_properties_keep_i32_wrapping_for_ineligible_trees() {
    for (predictor, offset, multiplier, threshold) in [
        (Predictor::Gradient, 1, 1, 0),
        (Predictor::Gradient, 0, 2, 0),
        (Predictor::Zero, 0, 1, 0),
        (Predictor::Gradient, 0, 1, -8194),
        (Predictor::Gradient, 0, 1, 8191),
    ] {
        let mut tree = tree(threshold);
        tree[2] = TreeNode::Leaf {
            predictor,
            offset,
            multiplier,
            id: 2,
        };
        assert!(!gradient::eligible(&tree, 0, 0, &mut VecDeque::new()).unwrap());
        let mut flat = Vec::new();
        Tree::build_flat_tree_into(&tree, &mut flat, &mut VecDeque::new()).unwrap();
        let references = References {
            data: &mut [],
            num_ref_props: 0,
        };
        let mut properties = [0; 256];
        let result = predict_flat(
            &flat,
            overflow(true),
            None,
            (1, 1),
            &references,
            &mut properties,
        );
        assert_eq!(properties[9], i32::MIN);
        assert_eq!(result.context, 2);
        assert_eq!(result.multiplier, multiplier);
        assert_eq!(
            result.guess,
            if predictor == Predictor::Zero {
                0
            } else {
                0x4000_0000
            } + offset as i64
        );
    }
}

#[test]
fn mixed_property_tree_keeps_generic_wrapping() {
    let tree = [
        TreeNode::Split {
            property: 9,
            val: 0,
            left: 1,
            right: 2,
        },
        TreeNode::Split {
            property: 3,
            val: 0,
            left: 3,
            right: 4,
        },
        leaf(2),
        leaf(3),
        leaf(4),
    ];
    assert!(!gradient::eligible(&tree, 0, 0, &mut VecDeque::new()).unwrap());
    let mut flat = Vec::new();
    Tree::build_flat_tree_into(&tree, &mut flat, &mut VecDeque::new()).unwrap();
    let references = References {
        data: &mut [],
        num_ref_props: 0,
    };
    let mut properties = [0; 256];
    let result = predict_flat(
        &flat,
        overflow(true),
        None,
        (1, 1),
        &references,
        &mut properties,
    );
    assert_eq!(properties[9], i32::MIN);
    assert_eq!(result.context, 2);
}
