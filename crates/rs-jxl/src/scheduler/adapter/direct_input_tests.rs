use super::*;
use crate::metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType};
use crate::{PixelData, PixelLayout};

#[test]
fn explicit_candidates_accept_typed_depths_without_relaxing_search_gate() {
    let mut images = vec![
        Image::new(
            2,
            1,
            PixelLayout::Gray,
            PixelData::U32(vec![0, 0x7fff_ffff]),
        )
        .unwrap(),
        Image::new(
            2,
            1,
            PixelLayout::Gray,
            PixelData::F16(vec![0x8000, 0x3c00]),
        )
        .unwrap(),
        Image::new(
            2,
            1,
            PixelLayout::Gray,
            PixelData::F32(vec![f32::from_bits(0x7fc1_2345), -0.0]),
        )
        .unwrap(),
    ];
    let custom = Image::new(2, 1, PixelLayout::Gray, PixelData::F32(vec![0.5, -1.0])).unwrap();
    for (bits, exponent) in [(5, 2), (13, 5), (16, 6), (24, 7), (32, 8)] {
        let mut metadata = custom.metadata().clone();
        metadata.bit_depth = BitDepth::floating(bits, exponent).unwrap();
        images.push(custom.clone().with_metadata(metadata, vec![]).unwrap());
    }
    let wide_storage = Image::new(2, 1, PixelLayout::Gray, PixelData::U32(vec![0, 65535])).unwrap();
    let mut metadata = wide_storage.metadata().clone();
    metadata.bit_depth = BitDepth::new(16).unwrap();
    images.push(wide_storage.with_metadata(metadata, vec![]).unwrap());
    for image in images {
        let original = image.clone();
        assert!(crate::scheduler::scan::validate_source_input(&image).is_err());
        for speed_tier in [-1, 0, 1, 3, 7, 8] {
            let params = CandidateParams {
                speed_tier,
                patches: Override::Off,
                ..Default::default()
            };
            let options = project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            assert_eq!(options.modular.predictor, params.resolved_predictor());
            assert!(options.modular.palette_policy.is_some());
            assert!(options.modular.sampling.is_some());
            assert!(options.source_patches.is_none());
            assert_eq!(image, original, "projection must not convert sample values");
        }
    }
}

#[test]
fn high_color_depth_retains_source_patch_outer_gate_even_when_forced_on() {
    for pixels in [PixelData::U32(vec![0]), PixelData::F32(vec![-0.0])] {
        let image = Image::new(1, 1, PixelLayout::Gray, pixels).unwrap();
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
        }
    }
}

#[test]
fn float_extras_do_not_switch_integer_patch_color_and_atlas_is_metadata_only() {
    let base = Image::new(2, 1, PixelLayout::Rgb, PixelData::U8(vec![9; 6])).unwrap();
    let mut metadata = base.metadata().clone();
    let extras = [
        PixelData::F32(vec![f32::from_bits(0x7fc1_2345), -0.0]),
        PixelData::F16(vec![0x8000, 0x3c00]),
        PixelData::U32(vec![0x7fff_ffff, 0]),
    ];
    for plane in &extras {
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        extra.bit_depth = plane.bit_depth();
        metadata.extra_channels.push(extra);
    }
    let image = base.with_metadata(metadata, extras.to_vec()).unwrap();
    assert!(crate::scheduler::scan::validate_source_input(&image).is_err());
    let params = CandidateParams {
        max_properties: 4,
        ..Default::default()
    };
    let options = project_native(&params, &image, &EncodeOptions::default())
        .unwrap()
        .exact_options()
        .unwrap();
    let patches = options.source_patches.as_ref().unwrap();
    assert!(
        !patches.floating_point,
        "only COLOR controls the detector domain"
    );
    assert_eq!(patches.extra_channel_dim_shifts, [0, 0, 0]);
    let mut planes = vec![vec![0; 8]; 6];
    for color in planes.iter_mut().take(3) {
        color.fill(-1);
    }
    let atlas = image.with_modular_planes(4, 2, planes).unwrap();
    assert!(atlas.pixels().is_empty());
    assert!(atlas.extra_planes().is_empty());
    let resolved = options
        .source_patch_reference
        .as_ref()
        .unwrap()
        .resolve(&atlas)
        .unwrap();
    assert_eq!(resolved.modular.predictor, Predictor::Gradient);
    assert_eq!(
        resolved
            .modular
            .sampling
            .as_ref()
            .unwrap()
            .previous_channels,
        4
    );
    assert_eq!(atlas.metadata(), image.metadata());
}

#[test]
fn float_patch_color_preserves_declared_words_and_resolves_atlas_geometry() {
    let mut images = vec![
        Image::new(
            2,
            1,
            PixelLayout::Gray,
            PixelData::F16(vec![0x8000, 0x3c00]),
        )
        .unwrap(),
    ];
    let custom = Image::new(
        2,
        1,
        PixelLayout::Rgb,
        PixelData::F32(vec![0.5, -0.0, -1.0, -0.5, 1.0, 0.0]),
    )
    .unwrap();
    for (bits, exponent) in [(5, 2), (13, 5), (16, 5), (16, 6)] {
        let mut metadata = custom.metadata().clone();
        metadata.bit_depth = BitDepth::floating(bits, exponent).unwrap();
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 31).unwrap());
        images.push(
            custom
                .clone()
                .with_metadata(metadata, vec![PixelData::U32(vec![0, 0x7fff_ffff])])
                .unwrap(),
        );
    }
    let params = CandidateParams {
        speed_tier: 4,
        group_size_shift: Some(0),
        patches: Override::On,
        predictor: Some(Predictor::Zero),
        max_properties: 3,
        learning_fraction: 0.2,
        ..Default::default()
    };
    for image in images {
        let original = image.clone();
        let mut invalid_planes = vec![vec![0]; image.channel_count()];
        invalid_planes[0][0] = (1u32 << image.bits()) as i32;
        assert!(image.with_modular_planes(1, 1, invalid_planes).is_err());
        assert!(crate::scheduler::scan::validate_source_input(&image).is_err());
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        let patches = options.source_patches.as_ref().unwrap();
        assert!(patches.floating_point);
        assert_eq!(
            patches.extra_channel_dim_shifts,
            vec![0; image.metadata().extra_channels.len()]
        );
        assert!(
            !options
                .modular
                .sampling
                .as_ref()
                .unwrap()
                .properties
                .contains(&1)
        );
        let recipe = options.source_patch_reference.as_ref().unwrap();
        assert_eq!(recipe.sampling.predictor, Predictor::Gradient);
        assert_eq!(recipe.sampling.learning_fraction, 0.2);
        assert!(recipe.modular.sampling.is_none());
        for (width, height, streaming) in [(4, 2, false), (1025, 1, true)] {
            let count = (width * height) as usize;
            let planes = (0..image.channel_count())
                .map(|channel| {
                    (0..count)
                        .map(|i| image.modular_sample(channel, i % 2))
                        .collect()
                })
                .collect();
            let atlas = image.with_modular_planes(width, height, planes).unwrap();
            assert!(atlas.pixels().is_empty() && atlas.extra_planes().is_empty());
            assert_eq!(atlas.metadata(), image.metadata());
            let reference = recipe.resolve(&atlas).unwrap();
            assert_eq!(reference.group_size_shift, 0);
            let modular = &reference.modular;
            assert_eq!(modular.predictor, Predictor::Gradient);
            assert_eq!(
                modular.source_lz77.as_ref().unwrap().method,
                SourceLz77Method::Lz77b1w3t
            );
            assert_eq!(modular.entropy_options.streaming_mode, streaming);
            assert_eq!(
                modular.palette_policy.as_ref().unwrap().groupwise,
                streaming
            );
            let sampling = modular.sampling.as_ref().unwrap();
            assert_eq!(sampling.properties.contains(&1), streaming);
            assert_eq!(sampling.max_channel_size, 128);
            assert_eq!(sampling.previous_channels, 3.min(image.channel_count() - 1));
            assert_eq!(sampling.nb_repeats, 0.2f32 * 0.7f32);
            for channel in 0..image.channel_count() {
                assert_eq!(atlas.channel_depth(channel), image.channel_depth(channel));
                assert_eq!(
                    atlas.modular_sample(channel, count - 1),
                    image.modular_sample(channel, (count - 1) % 2)
                );
            }
        }
        assert_eq!(image, original);
    }
}

#[test]
fn subsampled_source_atlas_policy_remains_explicit() {
    let image = Image::new(2, 2, PixelLayout::Gray, PixelData::U8(vec![0; 4])).unwrap();
    let mut metadata = image.metadata().clone();
    let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
    extra.dim_shift = 1;
    metadata.extra_channels.push(extra);
    let image = image
        .with_metadata(metadata, vec![PixelData::U8(vec![0])])
        .unwrap();
    assert!(matches!(
        project_native(&CandidateParams::default(), &image, &EncodeOptions::default()),
        Err(Error::InvalidInput(message)) if message.contains("frozen RoundtripPatchFrame")
    ));
    let params = CandidateParams {
        patches: Override::Off,
        ..Default::default()
    };
    assert!(
        project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .is_ok()
    );
}
