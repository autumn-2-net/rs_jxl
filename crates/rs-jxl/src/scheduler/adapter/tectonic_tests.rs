use super::*;
use crate::modular::source_options::SourceBuffering;
use crate::{PixelData, PixelLayout};

#[test]
fn palette_tools_survive_candidate_and_reference_projection() {
    use crate::modular::palette::{PaletteGeometry, PaletteMapping, PaletteOrder, PaletteTools};
    let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![19; 8 * 8 * 3])).unwrap();
    for mode in 0..=9 {
        let tools = PaletteTools {
            order: PaletteOrder::from_source_mode(mode).unwrap(),
            geometry: PaletteGeometry::from_source_mode(mode % 3).unwrap(),
            mapping: PaletteMapping::Flat,
        };
        let mut base = EncodeOptions::default();
        base.modular.palette_policy = Some(PaletteOptions {
            tools,
            ..Default::default()
        });
        for effort in [3, 9, 11] {
            let params = CandidateParams::default().backend(effort).unwrap();
            let options = project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap();
            assert_eq!(
                options.modular.palette_policy.as_ref().unwrap().tools,
                tools
            );
            let atlas = project_source_patch_reference(
                &params,
                &image,
                &base,
                NativeResourcePolicy::default(),
            )
            .unwrap();
            assert_eq!(atlas.modular.palette_policy.as_ref().unwrap().tools, tools);
        }
    }
}

#[test]
fn explicit_force_palette_survives_candidate_and_reference_projection() {
    let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![19; 8 * 8 * 3])).unwrap();
    let mut base = EncodeOptions::default();
    base.modular.palette_policy = Some(PaletteOptions {
        force_palette: true,
        ..Default::default()
    });
    let original_policy = base.modular.palette_policy.clone();
    for palette_colors in [1, 10000, 65536] {
        for predictor in [None, Some(Predictor::Zero)] {
            let params = CandidateParams {
                speed_tier: 1,
                palette_colors,
                predictor,
                ..Default::default()
            };
            let original_params = params.clone();
            for modular in [
                project_native(&params, &image, &base)
                    .unwrap()
                    .exact_options()
                    .unwrap()
                    .modular,
                project_source_patch_reference(
                    &params,
                    &image,
                    &base,
                    NativeResourcePolicy::default(),
                )
                .unwrap()
                .modular,
            ] {
                let policy = modular.palette_policy.unwrap();
                assert!(policy.force_palette);
                assert_eq!(policy.palette_colors, palette_colors as i32);
                assert!(!policy.zero_tokens);
            }
            assert_eq!(params, original_params);
            assert_eq!(base.modular.palette_policy, original_policy);
        }
    }
}

#[test]
fn force_palette_defaults_and_outside_caps_do_not_enable_override() {
    let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![19; 8 * 8 * 3])).unwrap();
    for (force_palette, palette_colors) in [
        (None, 10000),
        (Some(false), 10000),
        (Some(true), 0),
        (Some(true), 65537),
        (Some(true), 70000),
        (Some(true), 70911),
    ] {
        let mut base = EncodeOptions::default();
        base.modular.palette_policy = force_palette.map(|force_palette| PaletteOptions {
            force_palette,
            ..Default::default()
        });
        let original_policy = base.modular.palette_policy.clone();
        let params = CandidateParams {
            speed_tier: 1,
            palette_colors,
            predictor: Some(Predictor::Zero),
            ..Default::default()
        };
        let original_params = params.clone();
        for modular in [
            project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap()
                .modular,
            project_source_patch_reference(&params, &image, &base, NativeResourcePolicy::default())
                .unwrap()
                .modular,
        ] {
            let policy = modular.palette_policy.unwrap();
            assert!(!policy.force_palette);
            assert_eq!(policy.palette_colors, palette_colors as i32);
            assert!(!policy.zero_tokens);
        }
        assert_eq!(params, original_params);
        assert_eq!(base.modular.palette_policy, original_policy);
    }
}

#[test]
fn explicit_lz_tools_survive_candidates_without_overriding_wire_policy() {
    use crate::tools::lz77::{SourceIndexMode, SourceParallelism};
    let image = Image::new(
        32,
        32,
        PixelLayout::Rgb,
        PixelData::U8(vec![19; 32 * 32 * 3]),
    )
    .unwrap();
    let mut base = EncodeOptions::default();
    let parallelism = Some(SourceParallelism {
        max_workers: 3,
        ..Default::default()
    });
    base.modular.source_lz77 = Some(SourceConfig {
        stream_arena: true,
        stream_arena_quiet: true,
        index_mode: SourceIndexMode::All,
        parallelism,
        index_word_match: true,
        min_symbol: Some(99),
        ..Default::default()
    });
    for effort in [5, 9, 11] {
        let mut params = CandidateParams::default().backend(effort).unwrap();
        params.predictor = Some(Predictor::Zero);
        for options in [
            project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap()
                .modular,
            project_source_patch_reference(&params, &image, &base, NativeResourcePolicy::default())
                .unwrap()
                .modular,
        ] {
            let lz = options.source_lz77.as_ref().unwrap();
            assert!(lz.stream_arena && lz.stream_arena_quiet && lz.index_word_match);
            assert_eq!(lz.parallelism, parallelism);
            assert_eq!(lz.index_mode, SourceIndexMode::All);
            assert_eq!(
                lz.length_uint_config,
                options.entropy_options.lz77_length_config
            );
            assert_eq!(lz.min_symbol(), options.entropy_options.lz77_min_symbol);
            assert_ne!(lz.min_symbol(), 99);
        }
    }
    let params = CandidateParams {
        speed_tier: 7,
        ..Default::default()
    };
    let options = project_native(&params, &image, &base)
        .unwrap()
        .exact_options()
        .unwrap();
    assert!(options.modular.source_lz77.is_none());
    assert!(!options.modular.lz77);
}

#[test]
fn negative_tier_reaches_sampling_palette_entropy_lz_and_patch_reference() {
    let image = Image::new(2304, 1, PixelLayout::Rgb, PixelData::U8(vec![7; 2304 * 3])).unwrap();
    let params = CandidateParams::default().backend(11).unwrap();
    let projection = project_native(&params, &image, &EncodeOptions::default()).unwrap();
    assert_eq!(projection.training.node_threshold, 61);
    assert_eq!(projection.lz_method, LzMethod::Optc256);
    let options = projection.exact_options().unwrap();
    assert!(options.shared_tree && options.global_palette);
    assert_eq!(options.modular.predictor, Predictor::Variable);
    assert!(options.modular.fixed_tree.is_none());
    assert_eq!(options.modular.rct_candidates, Some(19));
    assert_eq!(options.modular.weighted_candidates, Some(5));
    let palette = options.modular.palette_policy.as_ref().unwrap();
    assert_eq!(palette.speed_tier, -1);
    assert!(!palette.groupwise);
    let sampling = options.modular.sampling.as_ref().unwrap();
    assert_eq!(sampling.properties, vec![0, 1, 15]);
    assert_eq!(sampling.max_property_values, 32);
    assert_eq!(sampling.node_threshold, 61.0);
    assert_eq!(sampling.nb_repeats, 0.5f32 * 0.3f32);
    assert_eq!(
        options.modular.entropy_options,
        EntropyOptions::for_modular(ModularEntropyParams {
            speed_tier: SpeedTier::TectonicPlate,
            ..Default::default()
        })
    );
    let lz = options.modular.source_lz77.as_ref().unwrap();
    assert_eq!(lz.method, SourceLz77Method::Optc256);
    assert_eq!(
        lz.length_uint_config,
        options.modular.entropy_options.lz77_length_config
    );
    assert_eq!(
        lz.min_symbol(),
        options.modular.entropy_options.lz77_min_symbol
    );
    assert_eq!(options.source_patches.as_ref().unwrap().speed_tier, -1);
    let recipe = options.source_patch_reference.unwrap();
    assert_eq!(recipe.sampling.speed_tier, -1);
    let atlas = image
        .with_modular_planes(16, 1, vec![vec![-1; 16]; 3])
        .unwrap();
    let reference = recipe.resolve(&atlas).unwrap();
    assert_eq!(reference.modular.predictor, Predictor::Gradient);
    assert_eq!(
        reference
            .modular
            .palette_policy
            .as_ref()
            .unwrap()
            .speed_tier,
        -1
    );
    assert_eq!(
        reference.modular.sampling.as_ref().unwrap().node_threshold,
        61.0
    );
    assert_eq!(
        reference.modular.sampling.as_ref().unwrap().nb_repeats,
        0.5f32 * 0.3f32
    );
    assert_eq!(reference.modular.rct_candidates, Some(19));
    assert_eq!(reference.modular.weighted_candidates, None);
    assert!(atlas.pixels().is_empty());
}

#[test]
fn tectonic_explicit_streaming_keeps_signed_backend_policy() {
    let image = Image::new(1025, 1, PixelLayout::Gray, PixelData::U8(vec![5; 1025])).unwrap();
    let params = CandidateParams {
        speed_tier: -1,
        buffering: SourceBuffering::Streaming,
        group_size_shift: Some(0),
        patches: Override::Off,
        ..Default::default()
    };
    let options = project_native(&params, &image, &EncodeOptions::default())
        .unwrap()
        .exact_options()
        .unwrap();
    assert!(options.modular.palette_policy.as_ref().unwrap().groupwise);
    assert!(options.modular.entropy_options.streaming_mode);
    assert_eq!(
        options.modular.palette_policy.as_ref().unwrap().speed_tier,
        -1
    );
    assert_eq!(
        options.modular.sampling.as_ref().unwrap().node_threshold,
        61.0
    );
    assert_eq!(options.modular.predictor, Predictor::Variable);
    assert!(options.source_patches.is_none());
}

#[test]
fn explicit_tectonic_candidate_reaches_native_encoder_without_effort_aliasing() {
    let image = Image::new(8, 8, PixelLayout::Rgb, PixelData::U8(vec![9; 8 * 8 * 3])).unwrap();
    let params = CandidateParams {
        speed_tier: -1,
        patches: Override::Off,
        ..Default::default()
    };
    let options = project_native(&params, &image, &EncodeOptions::default())
        .unwrap()
        .exact_options()
        .unwrap();
    assert_eq!(
        options
            .modular
            .sampling
            .as_ref()
            .unwrap()
            .max_property_values,
        32
    );
    assert!(!crate::encode(&image, &options).unwrap().is_empty());
    assert_eq!(params.speed_tier, -1);
}
