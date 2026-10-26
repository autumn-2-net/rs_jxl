use super::{
    adapter::project_native,
    calibration::diverse_frontier,
    driver::{DriverOptions, run_pixel_search},
    params::{Candidate, CandidateParams, Override, SourceBuffering},
    policy::{PixelSearch, default_pixel_options},
    search::{EncodedOutput, Search, SearchLimits, TrialOutput},
};
use crate::{
    EncodeOptions, Image, PixelData, PixelLayout, Result,
    modular::{
        orchestration::{GroupwiseDecision, groupwise_decision},
        source_options::{SourceFrameRoute, one_pass_geometry},
    },
};

fn image(width: u32) -> Image {
    Image::new(
        width,
        1,
        PixelLayout::Rgba,
        PixelData::U8(vec![7; width as usize * 4]),
    )
    .unwrap()
}

#[test]
fn pixel_gate_preserves_buffering_speed_size_and_forced_patch_precedence() {
    use SourceBuffering::{Auto, FullFrame, Streaming, StreamingAbove2048};
    for (w, h, shift, tier, buffering, patches, responsive, streams) in [
        (1024, 1, 0, 1, Auto, false, false, false),
        (1025, 1, 0, 1, Auto, false, false, true),
        (1025, 1, 0, 0, Auto, false, false, false),
        (1025, 1, 0, 0, Streaming, false, false, true),
        (1025, 1, 0, 1, FullFrame, false, false, false),
        (1025, 1, 0, 1, Auto, true, false, false),
        (1025, 1, 0, 8, Streaming, true, false, false),
        (1025, 1, 0, 1, Streaming, false, true, false),
        (2048, 1, 0, 1, StreamingAbove2048, false, false, false),
        (2049, 1, 0, 1, StreamingAbove2048, false, false, true),
        (1, 2049, 0, 1, StreamingAbove2048, false, false, true),
        (2049, 1, 3, 1, StreamingAbove2048, false, false, false),
    ] {
        let geometry = one_pass_geometry(w, h, Some(shift)).unwrap();
        assert_eq!(
            geometry.pixel_route(buffering, tier, patches, responsive)
                == SourceFrameRoute::StreamingFrameAreas,
            streams,
            "{w}x{h} shift={shift} tier={tier} buffering={buffering:?} patches={patches}"
        );
    }
    let geometry = one_pass_geometry(3840, 2160, None).unwrap();
    assert_eq!(
        (
            geometry.group_dim,
            geometry.num_groups,
            geometry.num_dc_groups,
            geometry.num_streams
        ),
        (256, 135, 4, 165)
    );
}

#[test]
fn native_projection_selects_frame_areas_without_starting_patch_detection() {
    for (width, tier, patches, streaming, detector) in [
        (1024, 1, Override::Default, false, true),
        (1025, 1, Override::Default, true, false),
        (1025, 1, Override::Off, true, false),
        (1025, 1, Override::On, false, true),
        (1025, 0, Override::Default, false, true),
        (1025, 8, Override::On, false, false),
    ] {
        let image = image(width);
        let params = CandidateParams {
            speed_tier: tier,
            group_size_shift: Some(0),
            patches,
            ..Default::default()
        };
        let options = project_native(&params, &image, &EncodeOptions::default())
            .unwrap()
            .exact_options()
            .unwrap();
        assert_eq!(
            options.modular.palette_policy.as_ref().unwrap().groupwise,
            streaming
        );
        assert_eq!(options.modular.entropy_options.streaming_mode, streaming);
        assert_eq!(options.source_patches.is_some(), detector);
        assert_eq!(options.source_patch_reference.is_some(), detector);
        assert!(options.patches.is_none());
        assert!(options.shared_tree && options.global_palette);
        assert_eq!(
            groupwise_decision(
                &image,
                &options.modular,
                options.group_size_shift,
                &options.passes,
                options.responsive,
                false
            )
            .unwrap()
                == GroupwiseDecision::FrameAreas,
            streaming
        );
        if tier == 1 {
            let sampling = options.modular.sampling.as_ref().unwrap();
            assert_eq!(sampling.nb_repeats, 0.5f32 * 1.3f32);
            assert_eq!(sampling.max_channel_size, 128);
            assert_eq!(
                sampling.properties,
                vec![0, 1, 15, 9, 10, 11, 12, 13, 14, 2, 3, 4, 5, 6, 7, 8]
            );
            assert_eq!(options.modular.rct_candidates, Some(19));
            assert_eq!(options.modular.weighted_candidates, Some(5));
            assert!(options.modular.source_lz77.is_some());
        }
    }
}

#[test]
fn root_stream_count_controls_properties_without_cropping_or_fraction_reduction() {
    let base = EncodeOptions {
        modular: crate::modular::encode::ModularOptions {
            source_constant_channels: true,
            sample_limit: Some(33554432),
            ..Default::default()
        },
        ..Default::default()
    };
    for buffering in [SourceBuffering::Auto, SourceBuffering::FullFrame] {
        let params = CandidateParams {
            speed_tier: 4,
            buffering,
            group_size_shift: Some(0),
            patches: Override::Off,
            ..Default::default()
        };
        // 9 root groups and 2 DC areas => 33 streams. Resolving one area or
        // local group instead would drop static property 1 at this tier.
        let options = project_native(&params, &image(1025), &base)
            .unwrap()
            .exact_options()
            .unwrap();
        let sampling = options.modular.sampling.as_ref().unwrap();
        assert_eq!(sampling.properties, vec![0, 1, 15, 9, 10]);
        assert_eq!(sampling.nb_repeats, 0.5f32 * 0.7f32);
        assert_eq!(options.modular.sample_limit, Some(33554432));
        assert!(!options.modular.source_constant_channels);
    }
    assert!(base.modular.source_constant_channels);
}

#[test]
fn buffering_participates_in_identity_and_search_eligibility() {
    let image = image(16);
    let mut search = Search::new(SearchLimits::default());
    let mut candidates = Vec::new();
    for buffering in [
        SourceBuffering::Auto,
        SourceBuffering::FullFrame,
        SourceBuffering::StreamingAbove2048,
        SourceBuffering::Streaming,
    ] {
        let params = CandidateParams {
            buffering,
            ..Default::default()
        };
        let eligible = matches!(
            buffering,
            SourceBuffering::Auto | SourceBuffering::FullFrame
        );
        assert_eq!(default_pixel_options(&params), eligible);
        assert_eq!(
            PixelSearch::new(&image, params.clone(), None).is_ok(),
            eligible
        );
        if !eligible {
            assert!(
                run_pixel_search(
                    &image,
                    &params,
                    &DriverOptions::default(),
                    |_, _, _| -> Result<TrialOutput<()>> {
                        panic!("ineligible full search encoded")
                    },
                    |_, _| panic!("ineligible full search probed")
                )
                .is_err()
            );
        }
        assert!(search.find(&params).is_none());
        search
            .refine(&image, 1, params.clone(), false, false, &mut |_, _, _| {
                Ok(TrialOutput::Complete(EncodedOutput {
                    bytes: vec![1],
                    artifact: (),
                }))
            })
            .unwrap();
        candidates.push(Candidate::new("buffering", params));
    }
    assert_eq!(search.trials().len(), 4);
    let ranked = diverse_frontier(vec![
        candidates[0].clone(),
        candidates[0].clone(),
        candidates[1].clone(),
    ]);
    assert_eq!(ranked[1].params.buffering, SourceBuffering::FullFrame);
    for (index, candidate) in candidates.iter().enumerate() {
        assert_eq!(search.find(&candidate.params), Some(index));
    }
}
