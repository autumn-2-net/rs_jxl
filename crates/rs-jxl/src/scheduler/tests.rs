use super::{
    adapter::{CapabilityGap, project_native},
    params::{DeltaPalette, LzMethod, Override},
    policy::{PixelSearch, SearchAction},
    regions::{SampleSpec, copy_sample},
    *,
};
use crate::{EncodeOptions, Image, PixelData, PixelLayout, modular::predictor::Predictor};

fn gray(w: u32, h: u32, value: u8) -> Image {
    Image::new(
        w,
        h,
        PixelLayout::Gray,
        PixelData::U8(vec![value; w as usize * h as usize]),
    )
    .unwrap()
}

#[test]
fn flat_scan_and_region_geometry() {
    let image = gray(1024, 512, 7);
    let scan = scan_lossless_input(&image, ScanOptions::default());
    assert_eq!(
        (scan.sampled_colors, scan.sampled_flat, scan.sampled_pixels),
        (1, 1024, 1024)
    );
    assert!(scan.literal_candidate);
    assert!(scan.tiles.len() <= 4096);
    assert_eq!(
        scan.tiles.iter().map(|t| t.observations).sum::<u64>(),
        2 * 1024 * 512
    );
    assert!(
        scan.tiles
            .iter()
            .all(|t| t.horizontal == 0 && t.vertical == 0)
    );
    for r in scan.regions() {
        assert!(r.x + r.width <= 1024 && r.y + r.height <= 512);
    }
}

#[test]
fn hidden_rgb_and_high_sample_bits_are_scanned() {
    let image = Image::new(
        2,
        1,
        PixelLayout::Rgba,
        PixelData::U16(vec![0, 0, 0, 0, 256, 0, 0, 0]),
    )
    .unwrap();
    let scan = scan_lossless_input(
        &image,
        ScanOptions {
            palette_scope: true,
        },
    );
    assert_eq!(scan.tiles[0].horizontal, 1);
    assert_eq!(scan.palette_color_lower_bound, 2);
    assert_eq!(scan.sampled_colors, 2);
}

#[test]
fn source_plan_constants_and_lz_decisions() {
    let params = CandidateParams::default();
    let plan = lossless_probe_plan(&params, PlanOptions::default());
    assert_eq!(plan.len(), 15);
    assert_eq!(plan[0].params.palette_colors, 70000);
    assert_eq!(plan[0].params.group_size_shift, Some(3));
    assert_eq!(plan[0].params.lz_method(), LzMethod::Optc256);
    assert_eq!(plan[8].params.max_properties, 2);
    assert_eq!(plan[8].params.predictor, Some(Predictor::Variable));
    assert_eq!(plan[14].name, "raw-top");
    assert_eq!(
        lossless_probe_plan(&params, PlanOptions { extended_rct: true }).len(),
        18
    );
    assert_eq!(params.backend(3).unwrap().lz_method(), LzMethod::None);
    assert_eq!(params.backend(7).unwrap().lz_method(), LzMethod::Lz77b1w3t);
    assert_eq!(params.backend(8).unwrap().lz_method(), LzMethod::Lz77b3w3t);
    assert_eq!(
        params.backend(9).unwrap().automatic_rcts(),
        &[
            0, 6, 5, 10, 26, 40, 12, 19, 8, 4, 9, 15, 16, 17, 32, 33, 2, 1, 3
        ]
    );
}

#[test]
fn atlas_keeps_original_samples_and_metadata() {
    let image = Image::new(
        3,
        2,
        PixelLayout::Gray,
        PixelData::U16(vec![10, 11, 12, 20, 21, 22]),
    )
    .unwrap();
    let spec = SampleSpec {
        regions: vec![
            ScanRegion {
                x: 1,
                y: 1,
                width: 2,
                height: 1,
            },
            ScanRegion {
                x: 0,
                y: 0,
                width: 2,
                height: 1,
            },
        ],
    };
    let sample = copy_sample(&image, &spec, 4).unwrap();
    assert_eq!(sample.pixels(), &PixelData::U16(vec![21, 22, 10, 11]));
    assert_eq!(sample.metadata(), image.metadata());
    assert!(copy_sample(&image, &spec, 3).is_err());
}

#[test]
fn metadata_and_planar_extras_survive_cropping_and_scan() {
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
    let image = gray(2, 2, 9);
    let mut metadata = image.metadata().clone();
    metadata
        .extra_channels
        .push(ExtraChannelInfo::new(ExtraChannelType::Alpha, 16).unwrap());
    let image = image
        .with_metadata(metadata, vec![PixelData::U16(vec![0, 256, 0, 512])])
        .unwrap();
    let scan = scan_lossless_input(&image, ScanOptions::default());
    assert_eq!(scan.tiles[0].horizontal, 2);
    assert_eq!(scan.tiles[0].vertical, 1);
    assert_eq!(scan.tiles[0].observations, 16);
    let sample = copy_sample(
        &image,
        &SampleSpec {
            regions: vec![ScanRegion {
                x: 1,
                y: 0,
                width: 1,
                height: 2,
            }],
        },
        2,
    )
    .unwrap();
    assert_eq!(sample.metadata(), image.metadata());
    assert_eq!(sample.extra_planes(), &[PixelData::U16(vec![256, 512])]);
}

#[test]
fn new_sample_storage_is_cropped_bitwise_with_metadata_and_extras() {
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
    let cases = [
        (
            PixelData::U32(vec![0, 0x7fff_ffff, 1, 0x4000_0000]),
            PixelData::U32(vec![0x7fff_ffff, 0x4000_0000]),
        ),
        (
            PixelData::F16(vec![0, 0x8000, 0x7c00, 0x7e35]),
            PixelData::F16(vec![0x8000, 0x7e35]),
        ),
        (
            PixelData::F32(vec![0.0, -0.0, f32::INFINITY, f32::from_bits(0x7fc1_2345)]),
            PixelData::F32(vec![-0.0, f32::from_bits(0x7fc1_2345)]),
        ),
    ];
    let spec = SampleSpec {
        regions: vec![ScanRegion {
            x: 1,
            y: 0,
            width: 1,
            height: 2,
        }],
    };
    for (storage, expected) in cases {
        let color = Image::new(2, 2, PixelLayout::Gray, storage.clone()).unwrap();
        let sample = copy_sample(&color, &spec, 2).unwrap();
        assert_eq!(sample.pixels(), &expected);
        assert_eq!(sample.metadata(), color.metadata());

        let base = gray(2, 2, 3);
        let mut metadata = base.metadata().clone();
        let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        info.bit_depth = storage.bit_depth();
        metadata.extra_channels.push(info);
        let image = base.with_metadata(metadata, vec![storage]).unwrap();
        let sample = copy_sample(&image, &spec, 2).unwrap();
        assert_eq!(sample.extra_planes(), &[expected]);
        assert_eq!(sample.metadata(), image.metadata());
        assert_eq!(sample.channel_count(), image.channel_count());
    }
}

fn assert_automatic_source_input_rejected(image: &Image) {
    let params = CandidateParams::default();
    assert!(scan::validate_source_input(image).is_err());
    let scan = scan_lossless_input(
        image,
        ScanOptions {
            palette_scope: true,
        },
    );
    assert!(!scan.available());
    assert!(!scan.literal_candidate);
    assert!(!scan::literal_evidence(image));
    assert!(PixelSearch::new(image, params.clone(), None).is_err());
    assert!(
        CandidateScreen::new(image, None, ScreenOptions::default(), |_, _| {
            panic!("unsupported input reached probe encoder")
        })
        .is_err()
    );
    assert!(
        small_probe::probe_lossless_candidates(
            image,
            &params,
            None,
            PlanOptions::default(),
            false,
            ProbeLimits::default(),
            |_, _| panic!("unsupported input reached small probe encoder"),
        )
        .is_err()
    );
    assert!(
        small_probe::confirm_lossless_candidate(
            image,
            &params,
            &params,
            None,
            ProbeLimits::default(),
            |_, _| panic!("unsupported input reached confirmation encoder"),
        )
        .is_err()
    );
    assert!(
        driver::run_pixel_search::<_, _, ()>(
            image,
            &params,
            &DriverOptions::default(),
            |_, _, _| panic!("unsupported input reached full encoder"),
            |_, _| panic!("unsupported input reached driver probe encoder"),
        )
        .is_err()
    );
}

#[test]
fn automatic_source_input_rejection_precedes_tiny_image_and_callback_shortcuts() {
    use crate::metadata::{BitDepth, ExtraChannelInfo, ExtraChannelType};
    for storage in [
        PixelData::U32(vec![0x7fff_ffff]),
        PixelData::F16(vec![0x7e35]),
        PixelData::F32(vec![f32::from_bits(0x7fc1_2345)]),
    ] {
        let color = Image::new(1, 1, PixelLayout::Gray, storage.clone()).unwrap();
        assert_automatic_source_input_rejected(&color);
        let base = gray(1, 1, 0);
        let mut metadata = base.metadata().clone();
        let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        info.bit_depth = storage.bit_depth();
        metadata.extra_channels.push(info);
        assert_automatic_source_input_rejected(
            &base.with_metadata(metadata, vec![storage]).unwrap(),
        );
    }
    let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U32(vec![7])).unwrap();
    let mut metadata = image.metadata().clone();
    metadata.bit_depth = BitDepth {
        bits_per_sample: 16,
        exponent_bits_per_sample: 0,
    };
    let image = image.with_metadata(metadata, vec![]).unwrap();
    assert_automatic_source_input_rejected(&image);
    assert!(matches!(scan::validate_source_input(&image),
        Err(crate::Error::InvalidInput(message)) if message.contains("U32 storage")));
}

#[test]
fn explicit_source_projection_does_not_inherit_automatic_sample_restrictions() {
    for pixels in [
        PixelData::U32(vec![0x7fff_ffff]),
        PixelData::F16(vec![0x7e35]),
        PixelData::F32(vec![f32::from_bits(0x7fc1_2345)]),
    ] {
        let image = Image::new(1, 1, PixelLayout::Gray, pixels).unwrap();
        let original = image.clone();
        assert!(scan::validate_source_input(&image).is_err());
        let params = CandidateParams {
            patches: Override::Off,
            ..Default::default()
        };
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert!(options.source_patches.is_none());
        assert!(options.modular.palette_policy.is_some());
        assert_eq!(options.modular.predictor, params.resolved_predictor());
        assert_eq!(image, original);
        if image.bits() > 16 {
            // Source ComputeEncodingData's depth gate applies even to On.
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
}

#[test]
fn screen_caches_and_never_combines_cheap_and_exact_scores() {
    let image = gray(1024, 512, 4);
    let mut screen = CandidateScreen::new(
        &image,
        None,
        ScreenOptions::default(),
        |p: &CandidateParams, _: &Image| Ok(if p.speed_tier >= 3 { 100 } else { 10 }),
    )
    .unwrap();
    let p = CandidateParams {
        speed_tier: 1,
        ..Default::default()
    };
    let ranked = screen
        .rank(vec![Candidate::new("one", p.clone())], &p, true)
        .unwrap();
    let exact = screen.score(&p).unwrap();
    assert_eq!(ranked[0].context_bytes, Some(exact.iter().sum()));
    let before = screen.encodes();
    screen.score(&p).unwrap();
    assert_eq!(screen.encodes(), before);
}

#[test]
fn work_limit_is_checked_before_callback() {
    let image = gray(1024, 512, 4);
    let options = ScreenOptions {
        limits: ProbeLimits {
            max_encodes: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut called = false;
    let mut screen =
        CandidateScreen::new(&image, None, options, |_: &CandidateParams, _: &Image| {
            called = true;
            Ok(10)
        })
        .unwrap();
    assert!(screen.score(&CandidateParams::default()).is_err());
    drop(screen);
    assert!(!called);
}

#[test]
fn native_projection_does_not_clamp_palette_or_hide_gaps() {
    let image = gray(64, 64, 0);
    let plan = lossless_probe_plan(&CandidateParams::default(), PlanOptions::default());
    let projection = project_native(&plan[0].params, &image, &EncodeOptions::default()).unwrap();
    assert_eq!(
        projection
            .options
            .as_ref()
            .unwrap()
            .modular
            .palette_policy
            .as_ref()
            .unwrap()
            .palette_colors,
        70000
    );
    assert_eq!(
        projection.state(),
        adapter::AdaptationState::SourceEquivalent
    );
    assert!(!projection.gaps.contains(&CapabilityGap::LzEnabledState));
    assert!(!projection.gaps.contains(&CapabilityGap::TreeEntropyPolicy));
    assert!(projection.exact_options().is_ok());
    let mut delta = plan[0].params.clone();
    delta.delta_palette = DeltaPalette::ExactLeft;
    let projection = project_native(&delta, &image, &EncodeOptions::default()).unwrap();
    assert_eq!(
        projection.state(),
        adapter::AdaptationState::SourceEquivalent
    );
    assert!(
        !projection
            .gaps
            .contains(&CapabilityGap::ExactDeltaAbove65536)
    );
    delta.palette_colors = 70912;
    let projection = project_native(&delta, &image, &EncodeOptions::default()).unwrap();
    assert_eq!(projection.state(), adapter::AdaptationState::Unavailable);
    assert!(projection.gaps.contains(&CapabilityGap::PaletteAbove70911));
    let mut p = plan[8].params.clone();
    p.palette_colors = 256;
    p.delta_palette = DeltaPalette::ExactLeft;
    let projection = project_native(&p, &image, &EncodeOptions::default()).unwrap();
    let options = projection.options.unwrap();
    assert_eq!(
        options.modular.palette_policy.unwrap().mode,
        crate::modular::palette::PaletteMode::ExactDelta(Predictor::Left)
    );
    assert!(options.shared_tree);
}

#[test]
fn policy_keeps_completed_winner_and_explores_learning() {
    let image = gray(128, 128, 0);
    let mut policy = PixelSearch::new(
        &image,
        CandidateParams {
            speed_tier: 1,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let initial = policy.initial_plan().unwrap();
    assert_eq!(initial[0].params.speed_tier, 7);
    policy.record_completed(initial[0].clone(), 1000).unwrap();
    policy.record_completed(initial[1].clone(), 100).unwrap();
    let actions = policy.stage_actions(3).unwrap();
    assert!(
        actions
            .iter()
            .any(|a| matches!(a, SearchAction::Trial(t) if t.params.speed_tier == 3))
    );
    let request = policy.resolve(&actions[0]).unwrap();
    assert!(!policy.record_completed(request, 1100).unwrap());
    assert_eq!(policy.winner().unwrap().bytes, 100);
}

#[test]
fn calibration_requires_consistent_real_bytes() {
    assert!(calibration::calibration_agreement(
        &[90.0, 100.0],
        &[90, 100],
        0
    ));
    assert!(!calibration::calibration_agreement(
        &[90.0, 100.0],
        &[101, 100],
        0
    ));
    assert!(probe::promising(&[80, 130], &[100, 100]));
    assert!(!probe::promising(&[100, 100], &[100, 100]));
}

#[test]
fn small_probe_uses_source_ninety_encode_budget_and_stable_ties() {
    let image = gray(256, 256, 1);
    let result = small_probe::probe_lossless_candidates(
        &image,
        &CandidateParams::default(),
        None,
        PlanOptions::default(),
        false,
        ProbeLimits::default(),
        |_, _| Ok(100),
    )
    .unwrap();
    assert_eq!(result.samples, 9);
    assert_eq!(result.screen_samples, 3);
    assert_eq!(result.shortlisted, 6);
    assert_eq!(result.encodes, 90);
    assert_eq!(result.ranked.len(), 6);
    assert_eq!(result.ranked[0].name, "large-palette-zero");
    assert!(result.ranked.iter().all(|c| c.bytes == Some(900)));
}

#[test]
fn source_palette_scope_stops_at_lower_bound_not_a_false_fit() {
    let mut data = Vec::new();
    for value in 0u32..(256 * 300) {
        data.extend_from_slice(&[value as u8, (value >> 8) as u8, (value >> 16) as u8]);
    }
    let image = Image::new(256, 300, PixelLayout::Rgb, PixelData::U8(data)).unwrap();
    let scan = scan_lossless_input(
        &image,
        ScanOptions {
            palette_scope: true,
        },
    );
    assert_eq!(
        scan.palette_color_lower_bound,
        scan::PALETTE_SCOPE_LIMIT + 1
    );
}

#[test]
fn score_only_planner_requests_the_executable_large_frame_driver() {
    let image = gray(1024, 512, 0);
    let mut policy = PixelSearch::new(&image, CandidateParams::default(), None).unwrap();
    for request in policy.initial_plan().unwrap() {
        policy.record_completed(request, 1000).unwrap();
    }
    for action in policy.stage_actions(3).unwrap() {
        if let Some(request) = policy.resolve(&action) {
            policy.record_completed(request, 900).unwrap();
        }
    }
    assert!(matches!(
        policy.stage_actions(4).unwrap().as_slice(),
        [SearchAction::ExecuteLargeFrameDriver { stage: 4 }]
    ));
    assert_eq!(policy.winner().unwrap().bytes, 900);
}
