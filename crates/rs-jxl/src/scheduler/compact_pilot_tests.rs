//! Scheduling oracles, not codec compatibility or performance measurements.
use super::*;
use crate::scheduler::search::EncodedOutput;
use crate::{PixelData, PixelLayout};

fn image(width: u32, height: u32, layout: PixelLayout) -> Image {
    Image::new(
        width,
        height,
        layout,
        PixelData::U8(vec![
            19;
            width as usize * height as usize * layout.channels()
        ]),
    )
    .unwrap()
}

fn options() -> DriverOptions {
    let mut options = DriverOptions::scout();
    options.scout_pilot_first = true;
    options.scout_small_full = true;
    options.scout_overlap_automatic = true;
    options.screen.limits.max_encodes = 0;
    options.screen.limits.max_encoded_pixels = 0;
    options.screen.limits.max_resident_sample_pixels = 0;
    options
}

fn family(params: &CandidateParams) -> usize {
    match params.predictor {
        None if params.group_size_shift == Some(2) => 1,
        None => 0,
        Some(Predictor::Zero) if params.learning_fraction == 0.0 => 3,
        Some(Predictor::Zero) => 2,
        Some(Predictor::Variable) if params.rct == Some(0) => 4,
        Some(Predictor::Variable) if params.rct == Some(6) => 5,
        _ => panic!("unexpected compact candidate: {params:?}"),
    }
}

fn complete(bytes: usize, artifact: usize) -> TrialOutput<usize> {
    TrialOutput::Complete(EncodedOutput {
        bytes: vec![artifact as u8; bytes],
        artifact,
    })
}

#[test]
fn small_full_is_default_off_and_requires_pilot_first() {
    assert!(!DriverOptions::default().scout_small_full);
    assert!(!DriverOptions::scout().scout_small_full);
    let mut options = DriverOptions::scout();
    options.scout_small_full = true;
    let error = run_pixel_search::<_, _, ()>(
        &image(80, 80, PixelLayout::Rgb),
        &CandidateParams::default().backend(9).unwrap(),
        &options,
        |_, _, _| panic!("invalid options reached full encoder"),
        |_, _| panic!("invalid options reached probe encoder"),
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidInput(message)
        if message == "pilot refinements require pilot-first"));
}

#[test]
fn every_family_can_win_without_any_probe_or_winner_reencode() {
    let image = image(80, 80, PixelLayout::Rgb);
    for winner in 0..6 {
        let mut options = options();
        options.full_limits.max_full_encodes = 6;
        options.full_limits.max_full_pixels = 6 * 6400;
        let mut seen = [false; 6];
        let mut best = None;
        let result = run_pixel_search_with_batches(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, full, budget| {
                assert_eq!((full.width(), full.height()), (80, 80));
                assert_eq!(params.speed_tier, 1);
                assert_eq!(budget.beat_bytes, best);
                let family = family(params);
                assert!(!seen[family], "candidate was re-encoded");
                seen[family] = true;
                let size = if family == winner { 42 } else { 100 + family };
                best = Some(best.map_or(size, |old: usize| old.min(size)));
                Ok(complete(size, family))
            },
            |_, _| panic!("compact tournament used a probe"),
            BatchEncoders {
                planned_probe: Some(Box::new(|_, _| panic!("batched probe"))),
                prefix: Some(Box::new(|_| panic!("prefix probe"))),
                pilot_overlap: Some(Box::new(|_| panic!("overlap front"))),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(seen.into_iter().all(|seen| seen));
        assert_eq!(result.bytes, vec![winner as u8; 42]);
        assert_eq!(result.artifact, winner);
        assert_eq!(family(&result.params), winner);
        assert_eq!(result.report.full_encodes, 6);
        assert_eq!(result.report.full_pixels, 6 * 6400);
        assert_eq!(result.report.probe_encodes, 0);
        assert_eq!(result.report.probe_pixels, 0);
        assert_eq!(result.report.screen_batches, 0);
        assert_eq!(
            result.report.trials.iter().filter(|t| t.baseline).count(),
            1
        );
        assert!(result.report.events.iter().any(|event| matches!(
            event,
            SearchEvent::Route {
                name: "adaptive-scout-small-full",
                candidates: 6
            }
        )));
    }
}

#[test]
fn compact_parallel_waves_keep_all_candidates_and_obey_worker_and_scratch_caps() {
    use std::cell::RefCell;
    for (workers, scratch, expected_width) in
        [(1, 256 << 20, 1), (4, 256 << 20, 4), (8, 66 << 20, 2)]
    {
        let image = image(80, 80, PixelLayout::Rgb);
        let mut options = options();
        options.scout_parallel_finalists = true;
        options.workers.tasks = workers;
        options.workers.scratch_bytes = scratch;
        let calls = RefCell::new(Vec::new());
        let waves = RefCell::new(Vec::new());
        let result = run_pixel_search_with_batches(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                assert!(budget.beat_bytes.is_none());
                let id = family(params);
                calls.borrow_mut().push(id);
                Ok(complete(100, id))
            },
            |_, _| panic!("compact tournament used a probe"),
            BatchEncoders {
                full: Some(Box::new(|_, tasks| {
                    waves.borrow_mut().push(tasks.len());
                    assert!(tasks.len() <= expected_width);
                    tasks
                        .iter()
                        .map(|task| {
                            assert!(!task.cooperative);
                            assert!(task.budget.beat_bytes.is_some());
                            let id = family(task.params);
                            calls.borrow_mut().push(id);
                            Ok(complete(if id == 5 { 42 } else { 100 }, id))
                        })
                        .collect()
                })),
                ..Default::default()
            },
        )
        .unwrap();
        let mut calls = calls.into_inner();
        calls.sort_unstable();
        assert_eq!(calls, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(waves.borrow()[0], expected_width.min(5));
        assert_eq!(result.artifact, 5);
        assert_eq!(result.bytes.len(), 42);
        assert_eq!(result.report.full_encodes, 6);
        assert_eq!(result.report.probe_encodes, 0);
    }
}

#[test]
fn effort_ten_promotes_winner_and_automatic_without_losing_completed_prefix() {
    let image = image(80, 80, PixelLayout::Rgb);
    for winner in [0, 5] {
        for improves in [false, true] {
            let mut calls = Vec::new();
            let result = run_pixel_search(
                &image,
                &CandidateParams::default().backend(10).unwrap(),
                &options(),
                |params, _, _| {
                    let family = family(params);
                    calls.push((family, params.speed_tier));
                    let size = match (params.speed_tier, family == winner, improves) {
                        (1, true, _) => 42,
                        (0, true, true) => 40,
                        (0 | 1, _, _) => 100,
                        _ => panic!("backend effort lowered"),
                    };
                    Ok(complete(size, family))
                },
                |_, _| panic!("effort-ten compact probe"),
            )
            .unwrap();
            assert!(calls[..6].iter().all(|&(_, tier)| tier == 1));
            let mut tail = vec![(winner, 0)];
            if winner != 0 {
                tail.push((0, 0));
            }
            assert_eq!(&calls[6..], tail.as_slice());
            assert_eq!(result.report.full_encodes, 6 + tail.len());
            assert_eq!(result.params.speed_tier, if improves { 0 } else { 1 });
            assert_eq!(family(&result.params), winner);
            assert_eq!(
                result.bytes,
                vec![winner as u8; if improves { 40 } else { 42 }]
            );
            assert_eq!(result.artifact, winner);
            assert_eq!(result.report.probe_encodes, 0);
        }
    }
}

#[test]
fn compact_geometry_and_disabled_flag_keep_legacy_entry_gates() {
    for (width, height, eligible) in [
        (64, 64, false),
        (64, 65, false),
        (80, 80, true),
        (128, 96, true),
        (96, 128, true),
        (128, 97, false),
        (97, 128, false),
        (16, 1024, false),
    ] {
        let image = image(width, height, PixelLayout::Rgb);
        assert_eq!(compact_geometry(&image).unwrap(), eligible);
        for enabled in [false, true] {
            let mut options = options();
            options.scout_small_full = enabled;
            let mut tier = None;
            let error = run_pixel_search::<_, _, ()>(
                &image,
                &CandidateParams::default().backend(9).unwrap(),
                &options,
                |params, _, _| {
                    tier = Some(params.speed_tier);
                    Err(Error::Encoding("entry reached".into()))
                },
                |_, _| panic!("entry gate invoked probes"),
            )
            .unwrap_err();
            assert!(matches!(error, Error::Encoding(message) if message == "entry reached"));
            assert_eq!(tier, Some(if enabled && eligible { 1 } else { 7 }));
        }
    }
}

#[test]
fn unsupported_extra_geometry_does_not_enter_compact_tournament() {
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
    for (count, shift) in [(1, 1), (9, 0)] {
        let image = image(80, 80, PixelLayout::Rgb);
        let mut metadata = image.metadata().clone();
        let mut extra = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        extra.dim_shift = shift;
        metadata.extra_channels = vec![extra; count];
        let samples = (80usize >> shift).pow(2);
        let image = image
            .with_metadata(metadata, vec![PixelData::U8(vec![7; samples]); count])
            .unwrap();
        assert!(!compact_geometry(&image).unwrap());
        let error = run_pixel_search::<_, _, ()>(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options(),
            |params, _, _| {
                assert_eq!(params.speed_tier, 7);
                Err(Error::Encoding("legacy entry".into()))
            },
            |_, _| panic!("extra geometry probe"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Encoding(message) if message == "legacy entry"));
    }
}

#[test]
fn compact_plan_preserves_extended_rcts_and_only_explicit_native_aliases() {
    let input = CandidateParams::default().backend(9).unwrap();
    for layout in [PixelLayout::Gray, PixelLayout::GrayAlpha, PixelLayout::Rgb] {
        for aliases in [false, true] {
            for extended in [false, true] {
                let mut options = options();
                options.scout_native_rct_aliases = aliases;
                options.plan.extended_rct = extended;
                let image = image(80, 80, layout);
                let plan = compact_plan(&image, &input, &options).unwrap();
                let alias = aliases && image.channel_count() < 3;
                assert_eq!(
                    plan.len(),
                    6 + if extended { 3 } else { 0 } - usize::from(alias)
                );
                assert!(plan.iter().all(|params| params.speed_tier == 1));
                assert!(
                    plan.iter()
                        .any(|params| params == &literal(&input, 9, 0.0).unwrap())
                );
                let rcts: Vec<_> = plan
                    .iter()
                    .filter(|params| params.predictor == Some(Predictor::Variable))
                    .map(|params| params.rct.unwrap())
                    .collect();
                let mut expected = vec![0];
                if !alias {
                    expected.push(6);
                }
                if extended {
                    expected.extend([10, 17, 26]);
                }
                assert_eq!(rcts, expected);
            }
        }
    }
}

#[test]
fn each_explicit_extended_rct_can_win_without_probes() {
    let image = image(80, 80, PixelLayout::Rgb);
    for winner in [10, 17, 26] {
        let mut options = options();
        options.plan.extended_rct = true;
        let mut rcts = Vec::new();
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, _| {
                assert_eq!(params.speed_tier, 1);
                if params.predictor == Some(Predictor::Variable) {
                    let rct = params.rct.unwrap();
                    rcts.push(rct);
                    Ok(complete(if rct == winner { 42 } else { 200 }, rct as usize))
                } else {
                    // Cross-channel families lose initially, but explicit
                    // extended transforms must still get their full trials.
                    Ok(complete(100, 100))
                }
            },
            |_, _| panic!("extended RCT compact probe"),
        )
        .unwrap();
        assert_eq!(rcts, [0, 6, 10, 17, 26]);
        assert_eq!(result.params.rct, Some(winner));
        assert_eq!(result.params.predictor, Some(Predictor::Variable));
        assert_eq!(result.artifact, winner as usize);
        assert_eq!(result.bytes, vec![winner as u8; 42]);
        assert_eq!(result.report.full_encodes, 9);
        assert_eq!(result.report.probe_encodes, 0);
    }
}

#[test]
fn full_limits_are_enforced_without_starting_probes() {
    let image = image(80, 80, PixelLayout::Rgb);
    for (encodes, pixels, output_bytes, expected_calls) in [
        (5, 6 * 6400, 100, 5),
        (6, 6 * 6400 - 1, 100, 5),
        (6, 6 * 6400, 0, 0),
        (6, 6 * 6400, 99, 1),
    ] {
        let mut options = options();
        options.full_limits = SearchLimits {
            max_full_encodes: encodes,
            max_full_pixels: pixels,
            max_output_bytes: output_bytes,
        };
        let mut calls = 0;
        let error = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |_, _, budget| {
                calls += 1;
                assert_eq!(budget.max_output_bytes, output_bytes);
                Ok(complete(100, 0))
            },
            |_, _| panic!("limits triggered a probe"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Limit(_)));
        assert_eq!(calls, expected_calls);
    }
}

#[test]
fn real_byte_bounds_and_stable_ties_keep_the_completed_baseline() {
    let image = image(80, 80, PixelLayout::Rgb);
    for bounded in [false, true] {
        let mut first = None;
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options(),
            |params, _, budget| {
                if first.is_none() {
                    assert_eq!(budget.beat_bytes, None);
                    first = Some(params.clone());
                    Ok(complete(100, family(params)))
                } else {
                    assert_eq!(budget.beat_bytes, Some(100));
                    Ok(if bounded {
                        TrialOutput::Bounded { lower_bound: 100 }
                    } else {
                        complete(100, family(params))
                    })
                }
            },
            |_, _| panic!("bounds/ties invoked probes"),
        )
        .unwrap();
        assert_eq!(result.params, first.unwrap());
        assert_eq!(result.artifact, 2);
        assert_eq!(result.bytes, vec![2; 100]);
        assert_eq!(result.report.winner_index, 0);
        assert_eq!(result.report.full_encodes, 6);
        assert!(
            result.report.trials[1..]
                .iter()
                .all(|trial| trial.lower_bound == bounded.then_some(100))
        );
    }
}

#[test]
fn invalid_bounds_and_encoder_errors_are_not_accepted_as_outputs() {
    let image = image(80, 80, PixelLayout::Rgb);
    for failure in 0..3 {
        let mut calls = 0;
        let error = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options(),
            |_, _, _| {
                calls += 1;
                match (failure, calls) {
                    (0, _) => Ok(TrialOutput::Bounded { lower_bound: 100 }),
                    (_, 1) => Ok(complete(100, 0)),
                    (1, _) => Ok(TrialOutput::Bounded { lower_bound: 99 }),
                    _ => Err(Error::Encoding("candidate failed".into())),
                }
            },
            |_, _| panic!("failed compact encode invoked probes"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Encoding(_)));
        assert_eq!(calls, if failure == 0 { 1 } else { 2 });
    }
}
