use super::{
    adapter::{
        CapabilityGap, NativeResourcePolicy, project_native, project_source_patch_reference,
        source_patch_reference_policy,
    },
    params::{CandidateParams, DeltaPalette, Override, SourceBuffering},
};
use crate::{
    EncodeOptions, Image, PixelData, PixelLayout,
    modular::{predictor::Predictor, source_options::one_pass_geometry},
    tools::lz77::SourceLz77Method,
};

fn original() -> Image {
    Image::new(
        2304,
        1,
        PixelLayout::Rgba,
        PixelData::U16(vec![17; 2304 * 4]),
    )
    .unwrap()
}

#[test]
fn signed_atlas_recipe_reselects_lz_groups_and_properties_without_external_samples() {
    let image = original();
    let atlas = image
        .with_modular_planes(64, 1, vec![vec![-32768; 64]; 4])
        .unwrap();
    assert!(atlas.pixels().is_empty() && atlas.extra_planes().is_empty());
    let params = CandidateParams {
        speed_tier: 4,
        patches: Override::On,
        predictor: Some(Predictor::Zero),
        learning_fraction: 0.2,
        max_properties: 3,
        delta_palette: DeltaPalette::ExactLeft,
        lz77_careful_first_pass: true,
        bucket_matcher: true,
        ..Default::default()
    };
    let base = EncodeOptions::default();
    let main = project_native(&params, &image, &base).unwrap();
    assert!(!main.gaps.contains(&CapabilityGap::AutomaticPatchPolicy));
    assert!(!main.gaps.contains(&CapabilityGap::LzEnabledState));
    assert!(main.gaps.is_empty());
    let main = main.options.unwrap();
    assert_eq!(main.group_size_shift, 1);
    assert_eq!(
        main.modular.source_lz77.as_ref().unwrap().method,
        SourceLz77Method::Lz77b3w3t
    );
    assert_eq!(
        &main.modular.sampling.as_ref().unwrap().properties[..5],
        &[0, 1, 15, 9, 10]
    );
    let recipe = main.source_patch_reference.unwrap();
    assert!(recipe.modular.sampling.is_none());
    assert_eq!(recipe.group_size_shift, None);
    assert_eq!(recipe.sampling.learning_fraction, 0.2);
    let resolved = recipe.resolve(&atlas).unwrap();
    let direct =
        project_source_patch_reference(&params, &atlas, &base, NativeResourcePolicy::default())
            .unwrap();
    assert_eq!(resolved.group_size_shift, 0);
    assert_eq!(direct.group_size_shift, resolved.group_size_shift);
    let modular = &resolved.modular;
    assert_eq!(modular.predictor, Predictor::Gradient);
    assert_eq!(modular.weighted_candidates, None);
    assert_eq!(modular.rct_candidates, Some(5));
    let lz = modular.source_lz77.as_ref().unwrap();
    assert_eq!(lz.method, SourceLz77Method::Lz77b1w3t);
    assert!(lz.careful_first_pass && lz.bucket_matcher);
    assert_eq!(
        lz.length_uint_config,
        modular.entropy_options.lz77_length_config
    );
    assert_eq!(lz.min_symbol(), modular.entropy_options.lz77_min_symbol);
    let sampling = modular.sampling.as_ref().unwrap();
    assert_eq!(sampling.properties, vec![0, 15, 9, 10, 11, 19, 23, 27]);
    assert_eq!(
        sampling.properties,
        direct.modular.sampling.as_ref().unwrap().properties
    );
    assert_eq!(sampling.max_channel_size, 128);
    assert_eq!(sampling.previous_channels, 3);
    assert_eq!(sampling.nb_repeats, 0.2f32 * 0.7f32);
    assert_eq!(atlas.metadata(), image.metadata());
    assert_eq!(atlas.modular_sample(0, 0), -32768);
    assert_eq!(atlas.modular_sample(3, 63), -32768);
    assert!(atlas.pixels().is_empty() && atlas.extra_planes().is_empty());
    assert_eq!(params.predictor, Some(Predictor::Zero));
}

#[test]
fn reference_recipe_retains_explicit_group_resource_limits_and_entropy_overrides() {
    let image = original();
    let atlas = image.with_modular_planes(1, 1, vec![vec![-1]; 4]).unwrap();
    let mut base = EncodeOptions::default();
    base.modular.sample_limit = 12345;
    for (precise_entropy, probe_lz77, method) in [
        (false, 0, SourceLz77Method::Lz77b1w3t),
        (true, 0, SourceLz77Method::Optc256),
        (true, 1, SourceLz77Method::Lz77b3w3t),
        (false, 2, SourceLz77Method::Optc8),
    ] {
        let params = CandidateParams {
            speed_tier: 4,
            group_size_shift: Some(3),
            predictor: Some(Predictor::Zero),
            precise_entropy,
            probe_lz77,
            ..Default::default()
        };
        let policy = source_patch_reference_policy(
            &params,
            &image,
            &base,
            NativeResourcePolicy {
                max_tree_leaves: Some(64),
            },
        )
        .unwrap();
        let reference = policy.resolve(&atlas).unwrap();
        assert_eq!(reference.group_size_shift, 3);
        assert_eq!(
            reference
                .modular
                .sampling
                .as_ref()
                .unwrap()
                .max_channel_size,
            1024
        );
        assert_eq!(
            reference.modular.source_lz77.as_ref().unwrap().method,
            method
        );
        assert_eq!(reference.modular.max_tree_leaves, 64);
        assert_eq!(reference.modular.sample_limit, 12345);
        assert!(reference.modular.custom_tree.is_none());
        assert!(reference.modular.tree_groups.is_none());
        assert!(reference.modular.tree_group_splits.is_empty());
    }
}

#[test]
fn inherited_reference_recipe_is_cleared_or_replaced_by_the_source_candidate() {
    let image = original();
    let mut base = EncodeOptions::default();
    let unrelated = CandidateParams {
        speed_tier: 0,
        group_size_shift: Some(3),
        ..Default::default()
    };
    base.source_patch_reference = Some(
        source_patch_reference_policy(&unrelated, &image, &base, NativeResourcePolicy::default())
            .unwrap(),
    );
    for (tier, patches, enabled) in [
        (3, Override::Default, true),
        (3, Override::Off, false),
        (7, Override::Default, false),
    ] {
        let params = CandidateParams {
            speed_tier: tier,
            buffering: SourceBuffering::FullFrame,
            patches,
            ..Default::default()
        };
        let options = project_native(&params, &image, &base)
            .unwrap()
            .options
            .unwrap();
        assert_eq!(options.source_patch_reference.is_some(), enabled);
        if let Some(policy) = options.source_patch_reference {
            assert_eq!(policy.group_size_shift, None);
            assert_eq!(policy.sampling.speed_tier, tier);
        }
    }
    assert_eq!(
        base.source_patch_reference
            .as_ref()
            .unwrap()
            .group_size_shift,
        Some(3)
    );
}

#[test]
fn reference_geometry_preserves_source_boundaries_and_rejects_invalid_requests() {
    for (width, height, shift) in [
        (128, 128, 0),
        (129, 128, 1),
        (256, 256, 1),
        (257, 256, 2),
        (400, 400, 2),
        (401, 400, 1),
    ] {
        let geometry = one_pass_geometry(width, height, None).unwrap();
        assert_eq!(geometry.group_size_shift, shift);
    }
    assert_eq!(one_pass_geometry(2304, 1, None).unwrap().num_streams, 33);
    assert_eq!(one_pass_geometry(64, 1, None).unwrap().num_streams, 22);
    assert!(one_pass_geometry(0, 1, None).is_err());
    assert!(one_pass_geometry(1, 1, Some(4)).is_err());
}

#[test]
fn reference_metadata_projection_does_not_enable_patches_for_high_color_depth() {
    for pixels in [PixelData::U32(vec![1]), PixelData::F32(vec![-0.0])] {
        let image = Image::new(1, 1, PixelLayout::Gray, pixels).unwrap();
        let original = image.clone();
        assert!(super::scan::validate_source_input(&image).is_err());
        for patches in [Override::Default, Override::On] {
            let params = CandidateParams {
                patches,
                ..Default::default()
            };
            let options = project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            assert!(options.source_patches.is_none());
            assert!(options.source_patch_reference.is_none());
            // A manually requested settings projection is not a request to run
            // discovery. Only the source outer gate decides whether an atlas exists.
            let reference = project_source_patch_reference(
                &params,
                &image,
                &EncodeOptions::default(),
                NativeResourcePolicy::default(),
            )
            .unwrap();
            assert_eq!(reference.modular.predictor, Predictor::Gradient);
            assert!(reference.modular.sampling.is_some());
            assert_eq!(image, original);
        }
    }
}

#[test]
fn reference_projection_rejects_local_squeeze_scripts() {
    let params = CandidateParams {
        squeeze: vec![true],
        ..Default::default()
    };
    assert!(
        source_patch_reference_policy(
            &params,
            &original(),
            &EncodeOptions::default(),
            NativeResourcePolicy::default()
        )
        .is_err()
    );
}

#[test]
fn signed_atlas_resolves_original_buffering_with_patches_off_and_its_own_geometry() {
    let image = original();
    for (tier, buffering, width, expected_streaming) in [
        (1, SourceBuffering::Auto, 64, false),
        (1, SourceBuffering::Auto, 2304, true),
        (1, SourceBuffering::FullFrame, 2304, false),
        (0, SourceBuffering::Auto, 2304, false),
        (0, SourceBuffering::Streaming, 2304, true),
        (1, SourceBuffering::StreamingAbove2048, 2048, false),
        (1, SourceBuffering::StreamingAbove2048, 2049, true),
    ] {
        let params = CandidateParams {
            speed_tier: tier,
            buffering,
            patches: Override::On,
            ..Default::default()
        };
        let main = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert!(!main.modular.palette_policy.as_ref().unwrap().groupwise);
        assert!(!main.modular.entropy_options.streaming_mode);
        let recipe = main.source_patch_reference.unwrap();
        assert_eq!(recipe.buffering, buffering);
        let atlas = image
            .with_modular_planes(width, 1, vec![vec![-1; width as usize]; 4])
            .unwrap();
        assert!(atlas.pixels().is_empty());
        let resolved = recipe.resolve(&atlas).unwrap();
        assert_eq!(
            resolved.modular.palette_policy.as_ref().unwrap().groupwise,
            expected_streaming
        );
        assert_eq!(
            resolved.modular.entropy_options.streaming_mode,
            expected_streaming
        );
        assert_eq!(resolved.modular.predictor, Predictor::Gradient);
        assert_eq!(
            resolved.modular.sampling.as_ref().unwrap().nb_repeats,
            0.5f32 * 1.3f32
        );
        assert_eq!(resolved.group_size_shift, if width <= 128 { 0 } else { 1 });
        assert_eq!(atlas.modular_sample(0, 0), -1);
        assert!(atlas.pixels().is_empty());
    }
}
