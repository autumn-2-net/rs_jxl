use super::*;
use crate::modular::{
    encode::ModularOptions,
    group::Channel,
    layout::Passes,
    palette::{PaletteStage, PaletteState, apply_palettes},
    source_options::SourceBuffering,
};
use crate::{PixelData, PixelLayout};

fn image(width: u32, height: u32) -> Image {
    Image::new(
        width,
        height,
        PixelLayout::Rgba,
        PixelData::U8(
            (0..width as usize * height as usize * 4)
                .map(|i| (i.wrapping_mul(13) ^ (i / 7)) as u8)
                .collect(),
        ),
    )
    .unwrap()
}

#[test]
fn responsive_selects_global_script_local_trees_and_nonstreaming_entropy() {
    let image = image(1025, 1);
    for speed_tier in -1..=9 {
        for buffering in [SourceBuffering::Auto, SourceBuffering::Streaming] {
            let params = CandidateParams {
                responsive: true,
                speed_tier,
                buffering,
                group_size_shift: Some(0),
                patches: Override::Off,
                ..Default::default()
            };
            let projection = project_native(&params, &image, &EncodeOptions::default()).unwrap();
            let training = projection.training.clone();
            let options = projection.exact_options().unwrap();
            assert!(options.responsive);
            assert!(options.modular.squeeze.is_empty());
            assert_eq!(options.passes, Passes::default());
            assert_eq!(options.shared_tree, speed_tier <= 0);
            assert!(options.global_palette);
            let palette = options.modular.palette_policy.as_ref().unwrap();
            assert!(palette.responsive);
            assert!(!palette.groupwise);
            assert!(!options.modular.entropy_options.streaming_mode);
            assert!(options.modular.source_tree_entropy);
            assert_eq!(options.modular.predictor, Predictor::Zero);
            assert_eq!(options.modular.rct, Some(6));
            assert_eq!(options.modular.rct_candidates, None);
            assert_eq!(options.modular.weighted_candidates, None);
            assert!(options.source_patches.is_none());
            let tier = params.modular_speed_tier();
            assert_eq!(palette.speed_tier, tier);
            assert_eq!(
                options.modular.fixed_tree,
                match tier {
                    7 => Some(FixedTree::Weighted),
                    8 => Some(FixedTree::Gradient),
                    _ => None,
                }
            );
            let sampling = options.modular.sampling.as_ref().unwrap();
            assert_eq!(sampling.nb_repeats, training.fraction);
            assert_eq!(sampling.node_threshold, (75 + 14 * i32::from(tier)) as f32);
            assert_eq!(sampling.max_channel_size, 128);
            assert_eq!(
                sampling.properties,
                training
                    .properties
                    .iter()
                    .map(|&p| p as usize)
                    .collect::<Vec<_>>()
            );
            assert_eq!(&sampling.properties[..3], &[0, 1, 4]);
        }
    }
}

#[test]
fn responsive_retains_explicit_predictor_rct_and_disables_palette_stages() {
    let image = image(16, 16);
    for predictor in [Predictor::Zero, Predictor::Gradient, Predictor::Weighted] {
        for (rct, expected) in [(None, Some(6)), (Some(0), None), (Some(17), Some(17))] {
            let params = CandidateParams {
                responsive: true,
                speed_tier: 1,
                predictor: Some(predictor),
                rct,
                palette_colors: 70000,
                delta_palette: DeltaPalette::ExactLeft,
                patches: Override::Off,
                ..Default::default()
            };
            let options = project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            assert_eq!(options.modular.predictor, predictor);
            assert_eq!(options.modular.rct, expected);
            assert_eq!(options.modular.rct_candidates, None);
            assert_eq!(
                options.modular.weighted_candidates,
                (predictor == Predictor::Weighted).then_some(5)
            );
            let palette = options.modular.palette_policy.as_ref().unwrap();
            let original = vec![
                Channel {
                    samples: vec![0; 16],
                    width: 4,
                    height: 4,
                    hshift: 0,
                    vshift: 0,
                };
                3
            ];
            for stage in [PaletteStage::PreRct, PaletteStage::PostRct] {
                let mut channels = original.clone();
                let mut state = PaletteState {
                    meta_channels: 0,
                    max_bitdepth: 9,
                    max_value: 255,
                };
                assert!(
                    apply_palettes(&mut channels, &mut state, 8, stage, palette)
                        .unwrap()
                        .is_empty()
                );
                assert_eq!(state.meta_channels, 0);
                assert_eq!(channels.len(), 3);
            }
        }
    }
}

#[test]
fn local_directions_never_impersonate_responsive_or_silently_disappear() {
    let image = image(16, 16);
    for responsive in [false, true] {
        for squeeze in [vec![true], vec![true, false, true], vec![true; 31]] {
            let params = CandidateParams {
                responsive,
                squeeze,
                patches: Override::Off,
                ..Default::default()
            };
            assert_eq!(
                params.resolved_predictor(),
                if responsive {
                    Predictor::Zero
                } else {
                    Predictor::Best
                }
            );
            let projection = project_native(&params, &image, &EncodeOptions::default()).unwrap();
            assert_eq!(projection.state(), AdaptationState::Unavailable);
            assert!(
                projection
                    .gaps
                    .contains(&CapabilityGap::ResponsiveTransformOrchestration)
            );
            assert!(
                !projection
                    .gaps
                    .contains(&CapabilityGap::MoreThanThirtySqueezePasses)
            );
            assert!(projection.exact_options().is_err());
            assert!(
                source_patch_reference_policy(
                    &params,
                    &image,
                    &EncodeOptions::default(),
                    NativeResourcePolicy::default()
                )
                .is_err()
            );
        }
    }
}

#[test]
fn default_candidate_does_not_inherit_responsive_local_passes_or_progression() {
    let base = EncodeOptions {
        responsive: true,
        passes: Passes::progressive(),
        modular: ModularOptions {
            squeeze: vec![true, false],
            palette_policy: Some(PaletteOptions {
                responsive: true,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let params = CandidateParams {
        patches: Override::Off,
        ..Default::default()
    };
    let options = project_native(&params, &image(16, 16), &base)
        .unwrap()
        .exact_options()
        .unwrap();
    assert!(!options.responsive);
    assert!(!options.modular.palette_policy.as_ref().unwrap().responsive);
    assert!(options.modular.squeeze.is_empty());
    assert!(options.shared_tree);
    assert_eq!(options.passes, Passes::default());
    assert_eq!(options.modular.predictor, Predictor::Best);
    assert!(base.responsive);
    assert_eq!(base.modular.squeeze, vec![true, false]);
}

#[test]
fn patch_atlas_clears_responsive_before_reselecting_rct_lz_sampling_and_route() {
    let image = image(2304, 1);
    let params = CandidateParams {
        responsive: true,
        speed_tier: 4,
        learning_fraction: 0.2,
        ..Default::default()
    };
    let options = project_native(&params, &image, &EncodeOptions::default())
        .unwrap()
        .exact_options()
        .unwrap();
    assert!(!options.shared_tree);
    assert!(options.source_patches.is_some());
    assert_eq!(
        options.modular.source_lz77.as_ref().unwrap().method,
        SourceLz77Method::Lz77b3w3t
    );
    let recipe = options.source_patch_reference.unwrap();
    assert!(!recipe.sampling.responsive);
    assert_eq!(recipe.sampling.learning_fraction, 0.2);
    for (width, streaming) in [(64, false), (2304, true)] {
        let atlas = image
            .with_modular_planes(width, 1, vec![vec![-1; width as usize]; 4])
            .unwrap();
        assert!(atlas.pixels().is_empty());
        let resolved = recipe.resolve(&atlas).unwrap();
        let modular = resolved.modular;
        assert!(!modular.palette_policy.as_ref().unwrap().responsive);
        assert_eq!(
            modular.palette_policy.as_ref().unwrap().groupwise,
            streaming
        );
        assert_eq!(modular.entropy_options.streaming_mode, streaming);
        assert_eq!(modular.predictor, Predictor::Gradient);
        assert_eq!(modular.rct, None);
        assert_eq!(modular.rct_candidates, Some(5));
        assert_eq!(
            modular.source_lz77.as_ref().unwrap().method,
            SourceLz77Method::Lz77b1w3t
        );
        let sampling = modular.sampling.as_ref().unwrap();
        assert_eq!(sampling.nb_repeats, 0.2f32 * 0.7f32);
        assert_eq!(
            sampling.properties,
            if streaming {
                vec![0, 1, 15, 9, 10]
            } else {
                vec![0, 15, 9, 10, 11]
            }
        );
        assert_eq!(atlas.modular_sample(0, 0), -1);
    }
}

#[test]
fn responsive_identity_is_explicit_and_not_an_automatic_search_option() {
    let defaults = CandidateParams::default();
    let responsive = CandidateParams {
        responsive: true,
        ..defaults.clone()
    };
    assert_ne!(defaults, responsive);
    assert!(super::super::policy::default_pixel_options(&defaults));
    assert!(!super::super::policy::default_pixel_options(&responsive));
    assert!(super::super::policy::PixelSearch::new(&image(16, 16), responsive, None).is_err());
}

#[test]
fn responsive_fixed_tiers_reach_the_native_encoder() {
    let image = image(257, 17);
    for speed_tier in [7, 8] {
        let params = CandidateParams {
            responsive: true,
            speed_tier,
            group_size_shift: Some(0),
            patches: Override::Off,
            ..Default::default()
        };
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        let bytes = crate::encode(&image, &options).unwrap();
        assert!(!bytes.is_empty());
    }
}
