use super::*;
use crate::modular::{encode::train_options, ma_train::MaBudgetMode};
use crate::scheduler::adapter::{
    NativeResourcePolicy, project_native_with_resources, source_patch_reference_policy,
};
use crate::scheduler::params::{CandidateParams, Override};
use crate::{EncodeOptions, PixelData, PixelLayout};

#[test]
fn ma_budget_mode_source_projection_and_reference_clones_keep_mode_and_cap() {
    let image = Image::new(
        64,
        16,
        PixelLayout::Rgb,
        PixelData::U8(vec![7; 64 * 16 * 3]),
    )
    .unwrap();
    let atlas = image
        .with_modular_planes(16, 1, vec![vec![-7; 16]; 3])
        .unwrap();
    for mode in [
        None,
        Some(MaBudgetMode::Strict),
        Some(MaBudgetMode::Bounded),
    ] {
        for cap in [None, Some(1), Some(7)] {
            let mut base = EncodeOptions::default();
            base.modular.ma_budget_mode = mode;
            base.modular.ma_max_leaves = cap;
            for effort in [6, 9, 10, 11] {
                let mut params = CandidateParams::default().backend(effort).unwrap();
                params.predictor = Some(Predictor::Zero);
                params.buffering = SourceBuffering::FullFrame;
                params.patches = Override::Off;
                let projection = project_native_with_resources(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap();
                let modular = &projection.options.as_ref().unwrap().modular;
                assert_eq!(modular.ma_budget_mode, mode);
                assert_eq!(modular.ma_max_leaves, cap);
                let projected_limit = modular.max_merged_tree_leaves.unwrap();
                assert_eq!(
                    train_options(modular).tree_leaf_limit(),
                    cap.map_or(projected_limit, |cap| cap.min(projected_limit))
                );

                let recipe = source_patch_reference_policy(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap();
                let reference = recipe.resolve(&atlas).unwrap();
                assert_eq!(reference.modular.ma_budget_mode, mode);
                assert_eq!(reference.modular.ma_max_leaves, cap);
                if let Some(cap) = cap {
                    assert!(reference.modular.max_merged_tree_leaves.unwrap() <= cap);
                }
                assert_eq!(base.modular.ma_max_leaves, cap);
                assert_eq!(recipe.modular.ma_max_leaves, cap);
            }
        }
    }
}
