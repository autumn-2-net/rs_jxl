// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.

use super::*;
use crate::modular::ma_train::MAX_LEAVES;
use crate::scheduler::{
    adapter::{NativeResourcePolicy, source_patch_reference_policy},
    params::{CandidateParams, Override},
};
use crate::{EncodeOptions, PixelData, PixelLayout};

const CANVAS_LEAVES: usize = 66_048;
const ATLAS_LEAVES: usize = 520;
const SAMPLE_LIMIT: usize = 1_234_567;

fn canvas() -> Image {
    Image::new(
        1024,
        512,
        PixelLayout::Rgba,
        PixelData::U16(vec![17; 1024 * 512 * 4]),
    )
    .unwrap()
}

fn policy(canvas: &Image, explicit_limit: Option<usize>) -> SourcePatchReferencePolicy {
    let params = CandidateParams {
        speed_tier: 4,
        buffering: SourceBuffering::FullFrame,
        patches: Override::On,
        predictor: Some(Predictor::Zero),
        ..Default::default()
    };
    let mut base = EncodeOptions::default();
    base.modular.sample_limit = SAMPLE_LIMIT;
    source_patch_reference_policy(
        &params,
        canvas,
        &base,
        NativeResourcePolicy {
            max_tree_leaves: explicit_limit,
        },
    )
    .unwrap()
}

#[test]
fn atlas_budget_large_canvas_bound_does_not_leak_into_signed_reference() {
    let canvas = canvas();
    let atlas = canvas
        .with_modular_planes(64, 1, vec![vec![-32768; 64]; 4])
        .unwrap();
    assert!(atlas.pixels().is_empty() && atlas.extra_planes().is_empty());
    let policy = policy(&canvas, None);
    assert_eq!(policy.modular.max_tree_leaves, MAX_LEAVES);
    assert_eq!(policy.modular.max_merged_tree_leaves, Some(CANVAS_LEAVES));

    let reference = policy.resolve(&atlas).unwrap();
    // 1024 + 64 * 1 * 4 / 16 = 1040 decoder nodes, hence at most 520 leaves.
    // Empty external PixelData must not replace the signed atlas's geometry.
    assert_eq!(reference.modular.max_merged_tree_leaves, Some(ATLAS_LEAVES));
    assert_eq!(reference.modular.max_tree_leaves, MAX_LEAVES);
    assert_eq!(reference.modular.sample_limit, SAMPLE_LIMIT);
    assert_eq!(reference.modular.predictor, Predictor::Gradient);
    assert_eq!(atlas.modular_sample(0, 0), -32768);
    assert_eq!(atlas.modular_sample(3, 63), -32768);
    assert_eq!(atlas.metadata(), canvas.metadata());

    // Resolving a smaller atlas must not permanently shrink the reusable recipe.
    assert_eq!(policy.modular.max_merged_tree_leaves, Some(CANVAS_LEAVES));
    assert!(policy.modular.sampling.is_none());
    assert_eq!(
        policy
            .resolve(&canvas)
            .unwrap()
            .modular
            .max_merged_tree_leaves,
        Some(CANVAS_LEAVES)
    );
    assert_eq!(
        policy
            .resolve(&atlas)
            .unwrap()
            .modular
            .max_merged_tree_leaves,
        Some(ATLAS_LEAVES)
    );
}

#[test]
fn atlas_budget_explicit_caps_are_retained_including_max_leaves() {
    let canvas = canvas();
    let atlas = canvas
        .with_modular_planes(64, 1, vec![vec![-1; 64]; 4])
        .unwrap();
    for (limit, atlas_limit) in [
        (1, 1),
        (64, 64),
        (ATLAS_LEAVES - 1, ATLAS_LEAVES - 1),
        (ATLAS_LEAVES, ATLAS_LEAVES),
        (ATLAS_LEAVES + 1, ATLAS_LEAVES),
        (MAX_LEAVES, ATLAS_LEAVES),
    ] {
        let policy = policy(&canvas, Some(limit));
        assert_eq!(policy.modular.max_tree_leaves, limit);
        assert_eq!(policy.modular.max_merged_tree_leaves, Some(limit));
        let reference = policy.resolve(&atlas).unwrap();
        assert_eq!(reference.modular.max_tree_leaves, limit);
        assert_eq!(reference.modular.max_merged_tree_leaves, Some(atlas_limit));
        assert_eq!(reference.modular.sample_limit, SAMPLE_LIMIT);
        assert_eq!(policy.modular.max_merged_tree_leaves, Some(limit));
        // In particular, Some(MAX_LEAVES) must not become the auto canvas bound.
        assert_eq!(
            policy
                .resolve(&canvas)
                .unwrap()
                .modular
                .max_merged_tree_leaves,
            Some(limit)
        );
    }
}

#[test]
fn atlas_budget_without_separate_merged_limit_keeps_legacy_caller_ceiling() {
    let canvas = canvas();
    let atlas = canvas
        .with_modular_planes(64, 1, vec![vec![0; 64]; 4])
        .unwrap();
    let mut policy = policy(&canvas, None);
    policy.modular.max_merged_tree_leaves = None;
    for (limit, atlas_limit) in [(64, 64), (MAX_LEAVES, ATLAS_LEAVES)] {
        policy.modular.max_tree_leaves = limit;
        let reference = policy.resolve(&atlas).unwrap();
        assert_eq!(reference.modular.max_tree_leaves, limit);
        assert_eq!(reference.modular.max_merged_tree_leaves, Some(atlas_limit));
        assert_eq!(reference.modular.sample_limit, SAMPLE_LIMIT);
        assert!(policy.modular.max_merged_tree_leaves.is_none());
    }
}
