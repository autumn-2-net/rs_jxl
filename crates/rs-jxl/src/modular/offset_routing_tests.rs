//! Offset reuse must not cross native gradient-LUT routing boundaries.

use super::{
    FitStats, OffsetMode, OffsetOptions, fit_literals, fit_tokens, fit_with_configs,
    remap_literals, remap_tokens,
};
use crate::entropy::{HybridUintConfig, stream::Token};
use crate::modular::{
    encode,
    group::Channel,
    predictor::{Predictor, predict_one},
    training::TrainingStream,
    tree::{Node, Tree, pack_signed},
    weighted,
};
use crate::tools::lz77::Literal;

struct RoutingCase {
    name: &'static str,
    threshold: i32,
    pixels: [i32; 4],
    generic_contexts: [u32; 4],
    native_contexts: [u32; 4],
}

fn cases() -> [RoutingCase; 5] {
    const A: i32 = 1 << 30;
    [
        RoutingCase {
            name: "positive overflow",
            threshold: 0,
            pixels: [0, A, A, A + 1000],
            generic_contexts: [1, 1, 1, 1],
            native_contexts: [1, 1, 1, 0],
        },
        RoutingCase {
            name: "negative overflow",
            threshold: 0,
            pixels: [1, -A, -A, -A + 1000],
            generic_contexts: [1, 0, 0, 0],
            native_contexts: [1, 0, 0, 1],
        },
        RoutingCase {
            name: "below lower clip boundary",
            threshold: -8193,
            pixels: [0, -5000, -5000, -4000],
            generic_contexts: [0, 0, 0, 1],
            native_contexts: [0, 0, 0, 0],
        },
        RoutingCase {
            name: "at lower split threshold",
            threshold: -8193,
            pixels: [0, -4097, -4096, -3097],
            generic_contexts: [0, 0, 0, 1],
            native_contexts: [0, 0, 0, 0],
        },
        RoutingCase {
            name: "at lower clip boundary",
            threshold: -8193,
            pixels: [0, -4096, -4096, -3096],
            generic_contexts: [0, 0, 0, 0],
            native_contexts: [0, 0, 0, 0],
        },
    ]
}

fn gradient_leaf(offset: i32) -> Node {
    Node::Leaf {
        predictor: Predictor::Gradient,
        offset,
        multiplier: 1,
    }
}

fn split_tree(threshold: i32, offsets: [i32; 2]) -> Tree {
    Tree::new(vec![
        Node::Split {
            property: 9,
            threshold,
            greater: 1,
            less_equal: 2,
        },
        gradient_leaf(offsets[0]),
        gradient_leaf(offsets[1]),
    ])
    .unwrap()
}

fn channels(case: &RoutingCase) -> [Channel; 1] {
    [Channel {
        width: 2,
        height: 2,
        hshift: 0,
        vshift: 0,
        samples: case.pixels.to_vec(),
    }]
}

fn native(channels: &[Channel], tree: &Tree) -> Vec<Literal> {
    crate::modular::tokens::tokenize(channels, tree, &weighted::Header::default(), 0, 0, false)
        .unwrap()
}

fn generic(channels: &[Channel], tree: &Tree) -> Vec<Literal> {
    let mut result = Vec::new();
    encode::visit(
        channels,
        &weighted::Header::default(),
        tree.uses_weighted(),
        0,
        0,
        |_, _, neighbors, weighted, properties, pixel| {
            let (context, predictor, offset, multiplier) = tree.lookup(properties)?;
            assert_eq!(multiplier, 1);
            let prediction = predict_one(predictor, neighbors, Some(i64::from(weighted))).unwrap();
            let residual = i64::from(pixel) - prediction - i64::from(offset);
            result.push(Literal {
                context: context as u32,
                value: pack_signed(residual as i32),
            });
            Ok(())
        },
    )
    .unwrap();
    result
}

fn contexts(literals: &[Literal]) -> Vec<u32> {
    literals.iter().map(|literal| literal.context).collect()
}

fn tokens(literals: &[Literal]) -> Vec<Token> {
    literals
        .iter()
        .map(|literal| Token::literal(literal.context as usize, literal.value))
        .collect()
}

fn assert_remap_falls_back(channels: &[Channel], before: &Tree, after: &Tree) {
    let original = vec![native(channels, before)];
    let mut raw = original.clone();
    assert_eq!(remap_literals(before, after, &mut raw).unwrap(), None);
    assert_eq!(raw, original, "raw fallback must leave input unchanged");

    let original = vec![tokens(&original[0])];
    let mut encoded = original.clone();
    assert_eq!(remap_tokens(before, after, &mut encoded).unwrap(), None);
    assert_eq!(
        encoded, original,
        "token fallback must leave input unchanged"
    );
}

fn assert_fit_falls_back(channels: &[Channel], tree: &Tree) {
    let views = [TrainingStream {
        channels,
        wp: weighted::Header::default(),
        stream_id: 0,
    }];
    let raw = vec![native(channels, tree)];
    let encoded = vec![tokens(&raw[0])];
    let configs = vec![HybridUintConfig::new(4, 1, 2).unwrap(); tree.leaves()];
    for samples_per_leaf in [1, 7] {
        let options = OffsetOptions {
            mode: OffsetMode::Distribution,
            samples_per_leaf,
        };
        assert!(
            fit_tokens(tree, &views, 0, &encoded, &options, Some(&configs))
                .unwrap()
                .is_none(),
            "possible gradient-LUT routing must use the pixel fitter"
        );
        assert!(
            fit_literals(tree, &views, 0, &raw, &options, Some(&configs))
                .unwrap()
                .is_none(),
            "the pre-LZ fitter needs the same routing guard"
        );
        assert_eq!(raw[0], native(channels, tree));
        assert_eq!(encoded[0], tokens(&raw[0]));
    }
}

#[test]
fn gradient_prop9_native_and_generic_contexts_cross_offset_routing_boundary() {
    for case in cases() {
        let channels = channels(&case);
        let baseline = split_tree(case.threshold, [0, 0]);
        let modified = split_tree(case.threshold, [1000, -37]);
        let generic_before = generic(&channels, &baseline);
        let native_before = native(&channels, &baseline);
        let generic_after = generic(&channels, &modified);
        let native_after = native(&channels, &modified);
        assert_eq!(
            contexts(&generic_before),
            case.generic_contexts,
            "{}",
            case.name
        );
        assert_eq!(
            contexts(&native_before),
            case.native_contexts,
            "{}",
            case.name
        );
        assert_eq!(
            contexts(&generic_after),
            case.generic_contexts,
            "{}",
            case.name
        );
        assert_eq!(native_after, generic_after, "{}", case.name);
        // Every fixture's last pre-offset residual is exactly 1000.
        assert_eq!(native_before[3].value, pack_signed(1000), "{}", case.name);
    }
}

#[test]
fn gradient_prop9_remapping_declines_both_directions_without_mutation() {
    for case in cases() {
        let channels = channels(&case);
        let baseline = split_tree(case.threshold, [0, 0]);
        let modified = split_tree(case.threshold, [1000, -37]);
        assert_remap_falls_back(&channels, &baseline, &modified);
        assert_remap_falls_back(&channels, &modified, &baseline);
    }
}

#[test]
fn gradient_prop9_cached_fitting_requires_routing_fallback() {
    for case in cases() {
        // Even the -8192 control requests fallback: eligibility is tree-based.
        assert_fit_falls_back(&channels(&case), &split_tree(case.threshold, [0, 0]));
    }
}

#[test]
fn static_channel_and_group_splits_do_not_hide_gradient_routing() {
    for property in [0, 1] {
        let make = |offset| {
            Tree::new(vec![
                Node::Split {
                    property,
                    threshold: 0,
                    greater: 1,
                    less_equal: 2,
                },
                Node::Leaf {
                    predictor: Predictor::Zero,
                    offset: 0,
                    multiplier: 1,
                },
                Node::Split {
                    property: 9,
                    threshold: 0,
                    greater: 3,
                    less_equal: 4,
                },
                gradient_leaf(offset),
                gradient_leaf(offset),
            ])
            .unwrap()
        };
        let channels = channels(&cases()[0]);
        let baseline = make(0);
        let modified = make(1000);
        assert_eq!(contexts(&native(&channels, &baseline)), [2, 2, 2, 1]);
        assert_eq!(contexts(&generic(&channels, &baseline)), [2, 2, 2, 2]);
        assert_eq!(native(&channels, &modified), generic(&channels, &modified));
        assert_fit_falls_back(&channels, &baseline);
        assert_remap_falls_back(&channels, &baseline, &modified);
        assert_remap_falls_back(&channels, &modified, &baseline);
    }
}

#[test]
fn static_choices_below_prop9_keep_possible_gradient_routing() {
    for property in [0, 1] {
        let make = |offset| {
            Tree::new(vec![
                Node::Split {
                    property: 9,
                    threshold: 0,
                    greater: 1,
                    less_equal: 2,
                },
                Node::Split {
                    property,
                    threshold: 0,
                    greater: 3,
                    less_equal: 4,
                },
                gradient_leaf(offset),
                Node::Leaf {
                    predictor: Predictor::Zero,
                    offset: 0,
                    multiplier: 1,
                },
                gradient_leaf(offset),
            ])
            .unwrap()
        };
        let channels = channels(&cases()[0]);
        let baseline = make(0);
        let modified = make(1000);
        assert_eq!(contexts(&native(&channels, &baseline)), [0, 0, 0, 2]);
        assert_eq!(contexts(&generic(&channels, &baseline)), [0, 0, 0, 0]);
        assert_fit_falls_back(&channels, &baseline);
        assert_remap_falls_back(&channels, &baseline, &modified);
        assert_remap_falls_back(&channels, &modified, &baseline);
    }
}

#[test]
fn gradient_prop9_noop_remapping_also_uses_conservative_fallback() {
    for case in cases() {
        let tree = split_tree(case.threshold, [0, 0]);
        assert_remap_falls_back(&channels(&case), &tree, &tree);
    }
}

fn assert_same_fit(actual: &(Option<Tree>, FitStats), expected: &(Option<Tree>, FitStats)) {
    assert_eq!(format!("{:?}", actual.0), format!("{:?}", expected.0));
    let stats = |s: &FitStats| (s.leaves, s.samples, s.population, s.candidates);
    assert_eq!(stats(&actual.1), stats(&expected.1));
}

fn assert_safe_reuse(baseline: &Tree, modified: &Tree) {
    let channels = channels(&cases()[0]);
    let views = [TrainingStream {
        channels: &channels,
        wp: weighted::Header::default(),
        stream_id: 0,
    }];
    let options = OffsetOptions {
        mode: OffsetMode::Distribution,
        samples_per_leaf: 7,
    };
    let configs = vec![HybridUintConfig::new(4, 1, 2).unwrap(); baseline.leaves()];
    for (before, after) in [(baseline, modified), (modified, baseline)] {
        let mut raw = vec![native(&channels, before)];
        let mut encoded = vec![tokens(&raw[0])];
        assert_eq!(raw[0], generic(&channels, before));
        let expected = fit_with_configs(before, &views, 0, &options, Some(&configs)).unwrap();
        assert_same_fit(
            &fit_literals(before, &views, 0, &raw, &options, Some(&configs))
                .unwrap()
                .expect("no possible gradient-LUT routing change"),
            &expected,
        );
        assert_same_fit(
            &fit_tokens(before, &views, 0, &encoded, &options, Some(&configs))
                .unwrap()
                .expect("no possible gradient-LUT routing change"),
            &expected,
        );
        let fresh = native(&channels, after);
        assert_eq!(remap_literals(before, after, &mut raw).unwrap(), Some(4));
        assert_eq!(raw[0], fresh);
        assert_eq!(remap_tokens(before, after, &mut encoded).unwrap(), Some(4));
        assert_eq!(encoded[0], tokens(&fresh));
    }
}

#[test]
fn gradient_without_prop9_keeps_literal_fitting_and_remapping() {
    assert_safe_reuse(
        &Tree::new(vec![gradient_leaf(0)]).unwrap(),
        &Tree::new(vec![gradient_leaf(1000)]).unwrap(),
    );
}

#[test]
fn ineligible_prop9_thresholds_keep_generic_routing_and_reuse() {
    for threshold in [-8194, 8191] {
        assert_safe_reuse(
            &split_tree(threshold, [0, 0]),
            &split_tree(threshold, [1000, -37]),
        );
    }
}

#[test]
fn prop9_requires_both_branches_to_be_gradient_eligible() {
    for other in [1, 2] {
        let make = |offset| {
            let mut nodes = split_tree(0, [offset, offset]).nodes().to_vec();
            nodes[other] = Node::Leaf {
                predictor: Predictor::Zero,
                offset,
                multiplier: 1,
            };
            Tree::new(nodes).unwrap()
        };
        assert_safe_reuse(&make(0), &make(1000));
    }
}
