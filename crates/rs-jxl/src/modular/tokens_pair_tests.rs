use super::*;
use crate::modular::{
    encode,
    predictor::predict_one,
    tree::{Node, Tree},
    weighted,
};

fn tree() -> Tree {
    Tree::new(vec![
        Node::Split {
            property: 0,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        Node::Leaf {
            predictor: Predictor::Left,
            offset: 7,
            multiplier: 3,
        },
        Node::Split {
            property: 9,
            threshold: 16,
            greater: 3,
            less_equal: 4,
        },
        Node::Split {
            property: 15,
            threshold: -3,
            greater: 5,
            less_equal: 6,
        },
        Node::Split {
            property: 10,
            threshold: 1,
            greater: 7,
            less_equal: 8,
        },
        Node::Leaf {
            predictor: Predictor::Weighted,
            offset: 0,
            multiplier: 1,
        },
        Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        },
        Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 1,
        },
        Node::Leaf {
            predictor: Predictor::Weighted,
            offset: 0,
            multiplier: 1,
        },
    ])
    .unwrap()
}

#[test]
fn cached_pair_matches_generic_wide_predictions_and_wire_contexts() {
    let tree = tree();
    let plan = ChannelTree::new(&tree, 0, 19);
    assert!(plan.pair_contexts().is_some());
    for width in [1, 2, 3, 8, 17, 257] {
        for height in [1, 2, 9] {
            let mut random = 0x145fabeu32;
            let plane = Channel {
                width,
                height,
                hshift: 0,
                vshift: 0,
                samples: (0..width * height)
                    .map(|i| {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        match i % 5 {
                            0 => i32::MIN,
                            1 => i32::MAX,
                            2 => 0,
                            _ => random as i32,
                        }
                    })
                    .collect(),
            };
            for preset in 0..5 {
                let header = weighted::Header::preset(preset).unwrap();
                let mut history = Vec::new();
                let mut expected = Vec::new();
                encode::visit_channel(
                    std::slice::from_ref(&plane),
                    0,
                    &header,
                    true,
                    0,
                    19,
                    |_, _, neighbors, wp, properties, value| {
                        history.push([wp, properties[15]]);
                        let leaf = plan.lookup_pixel(properties, neighbors)?;
                        let prediction =
                            predict_one(leaf.predictor, neighbors, Some(i64::from(wp))).unwrap();
                        expected.push(Literal {
                            context: leaf.context as u32,
                            value: pack_signed((i64::from(value) - prediction) as i32),
                        });
                        Ok(())
                    },
                )
                .unwrap();
                let mut actual = Vec::new();
                assert!(emit(&plane, &plan, Some(&history), |t| actual.push(t)).unwrap());
                assert_eq!(actual, expected, "{width}x{height}, preset={preset}");
                let mut integrated = Vec::new();
                super::super::channel(
                    std::slice::from_ref(&plane),
                    0,
                    &tree,
                    &header,
                    0,
                    19,
                    &mut super::super::Scratch::default(),
                    Some(&history),
                    |t| integrated.push(t),
                )
                .unwrap();
                assert_eq!(integrated, expected);
            }
        }
    }
}

#[test]
fn statically_filtered_gradient_lut_preserves_wide_routing() {
    let leaf = Node::Leaf {
        predictor: Predictor::Gradient,
        offset: 0,
        multiplier: 1,
    };
    let tree = Tree::new(vec![
        Node::Split {
            property: 0,
            threshold: 0,
            greater: 1,
            less_equal: 2,
        },
        Node::Leaf {
            predictor: Predictor::Top,
            offset: 3,
            multiplier: 1,
        },
        Node::Split {
            property: 9,
            threshold: 0,
            greater: 3,
            less_equal: 4,
        },
        leaf.clone(),
        leaf,
    ])
    .unwrap();
    let plan = ChannelTree::new(&tree, 0, 0);
    assert!(plan.pair_contexts().unwrap().gradient_lut);
    assert!(ChannelTree::new(&tree, 1, 0).pair_contexts().is_none());
    let plane = Channel {
        width: 4,
        height: 3,
        hshift: 0,
        vshift: 0,
        samples: vec![
            i32::MIN,
            i32::MAX,
            8191,
            -8192,
            i32::MAX,
            i32::MAX,
            -8193,
            8192,
            i32::MIN,
            i32::MIN,
            0,
            -1,
        ],
    };
    let mut expected = Vec::new();
    encode::visit_channel(
        std::slice::from_ref(&plane),
        0,
        &weighted::Header::default(),
        false,
        0,
        0,
        |_, _, neighbors, _, properties, value| {
            let leaf = plan.lookup_pixel(properties, neighbors)?;
            let prediction = predict_one(leaf.predictor, neighbors, None).unwrap();
            expected.push(Literal {
                context: leaf.context as u32,
                value: pack_signed((i64::from(value) - prediction) as i32),
            });
            Ok(())
        },
    )
    .unwrap();
    let mut actual = Vec::new();
    assert!(emit(&plane, &plan, Some(&[[0; 2]; 12]), |t| actual.push(t)).unwrap());
    assert_eq!(actual, expected);
}

#[test]
fn ineligible_trees_and_missing_history_fall_back() {
    let base = tree();
    assert!(ChannelTree::new(&base, 1, 0).pair_contexts().is_none());
    for bad in [
        Node::Split {
            property: 8,
            threshold: 16,
            greater: 3,
            less_equal: 4,
        },
        Node::Split {
            property: 19,
            threshold: 16,
            greater: 3,
            less_equal: 4,
        },
    ] {
        let mut nodes = base.nodes().to_vec();
        nodes[2] = bad;
        assert!(
            ChannelTree::new(&Tree::new(nodes).unwrap(), 0, 0)
                .pair_contexts()
                .is_none()
        );
    }
    for bad in [
        Node::Leaf {
            predictor: Predictor::Weighted,
            offset: 1,
            multiplier: 1,
        },
        Node::Leaf {
            predictor: Predictor::Gradient,
            offset: 0,
            multiplier: 2,
        },
        Node::Leaf {
            predictor: Predictor::Top,
            offset: 0,
            multiplier: 1,
        },
    ] {
        let mut nodes = base.nodes().to_vec();
        nodes[5] = bad;
        assert!(
            ChannelTree::new(&Tree::new(nodes).unwrap(), 0, 0)
                .pair_contexts()
                .is_none()
        );
    }
    let plan = ChannelTree::new(&base, 0, 0);
    let plane = Channel {
        width: 1,
        height: 1,
        hshift: 0,
        vshift: 0,
        samples: vec![0],
    };
    assert!(!emit(&plane, &plan, None, |_| panic!("unexpected token")).unwrap());
    assert!(emit(&plane, &plan, Some(&[]), |_| panic!("unexpected token")).is_err());
}
