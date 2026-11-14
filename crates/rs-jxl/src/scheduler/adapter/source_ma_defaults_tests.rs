use super::*;
use crate::modular::ma_train::{DescriptionCost, ScoringOptions, TreeLearning};
use crate::modular::refinement::{OffsetMode, OffsetOptions};
use crate::modular::tree::{Node, Tree};
use crate::{PixelData, PixelLayout};

fn project(effort: u8, base: &EncodeOptions) -> EncodeOptions {
    let image = Image::new(16, 16, PixelLayout::Gray, PixelData::U8(vec![7; 256])).unwrap();
    let params = CandidateParams::default().backend(effort).unwrap();
    project_native(&params, &image, base)
        .unwrap()
        .exact_options()
        .unwrap()
}

#[test]
fn ordinary_e9_alone_gets_single_pass_offsets_and_half_description() {
    let base = EncodeOptions::default();
    assert!(base.modular.source_ma_defaults);
    assert!(base.modular.refinement.offsets.is_none());
    assert_eq!(base.modular.source_ma_scoring, ScoringOptions::default());
    for effort in [8, 9, 10] {
        let options = project(effort, &base);
        if effort == 9 {
            assert_eq!(
                options.modular.refinement.offsets.unwrap().mode,
                OffsetMode::DistributionSinglePass
            );
            assert_eq!(
                options.modular.source_ma_scoring.description,
                DescriptionCost::Half
            );
        } else {
            assert!(options.modular.refinement.offsets.is_none());
            assert_eq!(options.modular.source_ma_scoring, ScoringOptions::default());
        }
    }
    let mut disabled = base.clone();
    disabled.modular.source_ma_defaults = false;
    let options = project(9, &disabled);
    assert!(!options.modular.source_ma_defaults);
    assert!(options.modular.refinement.offsets.is_none());
    assert_eq!(options.modular.source_ma_scoring, ScoringOptions::default());
}

#[test]
fn explicit_offsets_and_scoring_survive_defaults_and_opt_out() {
    for enabled in [false, true] {
        for mode in [
            OffsetMode::Entropy,
            OffsetMode::Distribution,
            OffsetMode::DistributionSinglePass,
        ] {
            for description in [
                DescriptionCost::Quarter,
                DescriptionCost::Half,
                DescriptionCost::Full,
            ] {
                let mut base = EncodeOptions::default();
                base.modular.source_ma_defaults = enabled;
                base.modular.refinement.offsets = Some(OffsetOptions {
                    mode,
                    samples_per_leaf: 37,
                });
                base.modular.source_ma_scoring = ScoringOptions {
                    count_log: true,
                    description,
                    ..Default::default()
                };
                let options = project(9, &base);
                let offsets = options.modular.refinement.offsets.unwrap();
                assert_eq!(offsets.mode, mode);
                assert_eq!(offsets.samples_per_leaf, 37);
                assert_eq!(
                    options.modular.source_ma_scoring,
                    base.modular.source_ma_scoring
                );
            }
        }
    }
}

#[test]
fn automatic_profile_fills_each_missing_setting_independently() {
    let mut base = EncodeOptions::default();
    base.modular.source_ma_scoring.description = DescriptionCost::Full;
    let options = project(9, &base);
    assert_eq!(
        options.modular.source_ma_scoring.description,
        DescriptionCost::Full
    );
    assert_eq!(
        options.modular.refinement.offsets.unwrap().mode,
        OffsetMode::DistributionSinglePass
    );

    base.modular.source_ma_scoring = ScoringOptions {
        count_log: true,
        ..Default::default()
    };
    base.modular.refinement.offsets = Some(OffsetOptions::default());
    let options = project(9, &base);
    assert!(options.modular.source_ma_scoring.count_log);
    assert_eq!(
        options.modular.source_ma_scoring.description,
        DescriptionCost::Half
    );
    assert_eq!(
        options.modular.refinement.offsets.unwrap().mode,
        OffsetMode::Entropy
    );
}

#[test]
fn explicit_tree_routes_and_alternative_learners_do_not_get_defaults() {
    for case in 0..11 {
        let mut base = EncodeOptions::default();
        match case {
            0 => base.modular.fixed_tree = Some(FixedTree::GradientLeaf),
            1 => {
                base.modular.custom_tree = Some(
                    Tree::new(vec![Node::Leaf {
                        predictor: Predictor::Gradient,
                        offset: 0,
                        multiplier: 1,
                    }])
                    .unwrap(),
                );
            }
            2 => base.modular.fast_context = Some(Default::default()),
            3 => base.modular.block_context = Some(Default::default()),
            4 => base.modular.source_ma_seed = Some(Default::default()),
            5 => base.modular.source_ma_lookahead = Some(Default::default()),
            6 => base.modular.tree_learning = TreeLearning::Cost,
            7 => base.modular.tree_learning = TreeLearning::Symmetric,
            8 => base.modular.tree_groups = Some(Default::default()),
            9 => base.modular.source_ma_search.histogram_bins = 16,
            10 => {
                base.modular.source_channel_tree =
                    crate::modular::ma_train::ChannelTreeMode::Channels;
            }
            _ => unreachable!(),
        }
        let options = project(9, &base);
        assert!(options.modular.refinement.offsets.is_none(), "case {case}");
        assert_eq!(
            options.modular.source_ma_scoring,
            ScoringOptions::default(),
            "case {case}"
        );
        options.modular.validate().unwrap();
    }
}

#[test]
fn patch_reference_helpers_keep_main_profile_policy_and_explicit_settings() {
    let image = Image::new(16, 16, PixelLayout::Gray, PixelData::U8(vec![7; 256])).unwrap();
    let profile = |options: &crate::modular::encode::ModularOptions| {
        (
            options.source_ma_defaults,
            options
                .refinement
                .offsets
                .as_ref()
                .map(|offsets| (offsets.mode, offsets.samples_per_leaf)),
            options.source_ma_scoring,
        )
    };
    for effort in [8, 9, 10] {
        for enabled in [false, true] {
            for explicit in [false, true] {
                let mut base = EncodeOptions::default();
                base.modular.source_ma_defaults = enabled;
                if explicit {
                    base.modular.refinement.offsets = Some(OffsetOptions {
                        mode: OffsetMode::Distribution,
                        samples_per_leaf: 37,
                    });
                    base.modular.source_ma_scoring.description = DescriptionCost::Full;
                }
                let params = CandidateParams::default().backend(effort).unwrap();
                let main = project_native_with_resources(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap()
                .exact_options()
                .unwrap();
                let policy = source_patch_reference_policy(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap();
                let resolved = policy.resolve(&image).unwrap();
                let direct = project_source_patch_reference(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap();
                for options in [&policy.modular, &resolved.modular, &direct.modular] {
                    assert_eq!(profile(options), profile(&main.modular));
                }
            }
        }
    }
}
