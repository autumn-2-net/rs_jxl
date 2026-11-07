use super::*;

fn leaf(offset: i32, multiplier: u32) -> Node {
    Node::Leaf {
        predictor: Predictor::Gradient,
        offset,
        multiplier,
    }
}

fn split(threshold: i32, offset: i32, multiplier: u32) -> Tree {
    Tree::new(vec![
        Node::Split {
            property: 9,
            threshold,
            greater: 1,
            less_equal: 2,
        },
        leaf(offset, multiplier),
        leaf(offset, multiplier),
    ])
    .unwrap()
}

fn properties(n: Neighbors) -> [i32; 16] {
    let mut properties = [0; 16];
    properties[9] = n.left.wrapping_add(n.top).wrapping_sub(n.topleft);
    properties
}

fn boundaries() -> [Neighbors; 5] {
    [
        Neighbors {
            left: 0x4000_0000,
            top: 0x4000_0000,
            ..Default::default()
        },
        Neighbors {
            left: -0x4000_0000,
            top: -0x4000_0000,
            topleft: 1,
            ..Default::default()
        },
        Neighbors {
            left: i32::MAX,
            top: i32::MAX,
            topleft: i32::MIN,
            ..Default::default()
        },
        Neighbors {
            left: i32::MIN,
            top: i32::MIN,
            topleft: i32::MAX,
            ..Default::default()
        },
        Neighbors {
            left: 31,
            top: 17,
            topleft: 25,
            ..Default::default()
        },
    ]
}

#[test]
fn gradient_lut_context_uses_wide_clipped_property() {
    for threshold in [-8193, -8192, -1025, -1024, -1, 0, 1023, 1024, 8190] {
        let tree = split(threshold, 0, 1);
        let plan = ChannelTree::new(&tree, 0, 0);
        assert!(plan.gradient_lut, "threshold {threshold}");
        for n in boundaries() {
            let wide = i64::from(n.left) + i64::from(n.top) - i64::from(n.topleft);
            let expected = usize::from(wide.clamp(-8192, 8191) <= i64::from(threshold));
            assert_eq!(
                plan.lookup_pixel(&properties(n), n).unwrap().context,
                expected
            );
        }
    }
}

#[test]
fn generic_tree_properties_still_wrap() {
    for (threshold, offset, multiplier) in [(8191, 0, 1), (-8194, 0, 1), (0, 1, 1), (0, 0, 3)] {
        let tree = split(threshold, offset, multiplier);
        let plan = ChannelTree::new(&tree, 0, 0);
        assert!(!plan.gradient_lut);
        for n in boundaries() {
            let p = properties(n);
            assert_eq!(
                plan.lookup_pixel(&p, n).unwrap().context,
                tree.lookup(&p).unwrap().0
            );
        }
    }
    let tree = Tree::new(vec![
        Node::Split {
            property: 9,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        leaf(0, 1),
        Node::Leaf {
            predictor: Predictor::Zero,
            offset: 0,
            multiplier: 1,
        },
    ])
    .unwrap();
    assert!(!ChannelTree::new(&tree, 0, 0).gradient_lut);
}

#[test]
fn static_filtering_controls_gradient_eligibility() {
    let tree = Tree::new(vec![
        Node::Split {
            property: 0,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        Node::Split {
            property: 1,
            threshold: 4,
            greater: 3,
            less_equal: 4,
        },
        Node::Leaf {
            predictor: Predictor::Weighted,
            offset: 0,
            multiplier: 1,
        },
        Node::Split {
            property: 9,
            threshold: 0,
            greater: 5,
            less_equal: 6,
        },
        leaf(7, 1),
        leaf(0, 1),
        leaf(0, 1),
    ])
    .unwrap();
    assert!(!ChannelTree::new(&tree, 0, 5).gradient_lut);
    assert!(!ChannelTree::new(&tree, 1, 4).gradient_lut);
    let plan = ChannelTree::new(&tree, 1, 5);
    assert!(plan.gradient_lut);
    let n = boundaries()[0];
    let mut p = properties(n);
    p[0] = 1;
    p[1] = 5;
    p[9] = 8191;
    assert_eq!(
        plan.lookup_pixel(&properties(n), n).unwrap().context,
        tree.lookup(&p).unwrap().0
    );
}

#[test]
fn clipped_empty_gradient_branch_does_not_force_generic_arithmetic() {
    let tree = Tree::new(vec![
        Node::Split {
            property: 9,
            threshold: -8193,
            greater: 1,
            less_equal: 2,
        },
        Node::Split {
            property: 9,
            threshold: 0,
            greater: 3,
            less_equal: 4,
        },
        leaf(0, 1),
        leaf(0, 1),
        leaf(0, 1),
    ])
    .unwrap();
    let plan = ChannelTree::new(&tree, 0, 0);
    assert!(plan.gradient_lut);
    let n = boundaries()[0];
    let mut p = properties(n);
    p[9] = 8191;
    assert_eq!(
        plan.lookup_pixel(&properties(n), n).unwrap().context,
        tree.lookup(&p).unwrap().0
    );
}
