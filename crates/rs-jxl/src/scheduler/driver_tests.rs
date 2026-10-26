//! Policy tests use deterministic size oracles, not mock codec acceptance.
use super::{
    driver::{BackendSemantics, DriverOptions, SearchEvent, run_pixel_search},
    params::{CandidateParams, Override},
    search::{EncodedOutput, Search, SearchLimits, TrialBudget, TrialOutput},
};
use crate::{Error, Image, PixelData, PixelLayout, modular::predictor::Predictor};

fn image() -> Image {
    Image::new(
        1024,
        512,
        PixelLayout::Gray,
        PixelData::U8(vec![1; 1024 * 512]),
    )
    .unwrap()
}
fn complete(bytes: usize, artifact: usize) -> TrialOutput<usize> {
    TrialOutput::Complete(EncodedOutput {
        bytes: vec![artifact as u8; bytes],
        artifact,
    })
}

fn first_boundary_candidate(height: u32, options: &DriverOptions) -> (CandidateParams, usize) {
    let image = Image::new(
        512,
        height,
        PixelLayout::Gray,
        PixelData::U8(vec![7; 512 * height as usize]),
    )
    .unwrap();
    let mut first = None;
    let mut probes = 0;
    let result = run_pixel_search::<_, _, ()>(
        &image,
        &CandidateParams::default().backend(9).unwrap(),
        options,
        |params, _, budget| {
            assert_eq!(budget.beat_bytes, None);
            first = Some(params.clone());
            Err(Error::Encoding("boundary test reached first full".into()))
        },
        |params, _| {
            probes += 1;
            Ok(
                if params.predictor == Some(Predictor::Zero) && params.learning_fraction == 0.0 {
                    10
                } else {
                    100
                },
            )
        },
    );
    assert!(matches!(result, Err(Error::Encoding(message))
        if message == "boundary test reached first full"));
    (first.unwrap(), probes)
}

#[test]
fn pilot_scan_and_screen_share_effective_adaptive_boundary() {
    for explicit_adaptive in [false, true] {
        let mut options = DriverOptions::scout();
        options.scout_pilot_first = true;
        options.preflight.enabled = true;
        options.screen.adaptive_coverage = explicit_adaptive;
        options.screen.regions.full_width_strips = true;
        for height in [255, 256, 257] {
            let (first, probes) = first_boundary_candidate(height, &options);
            if height < 256 {
                assert_eq!(first, CandidateParams::default().backend(3).unwrap());
                assert_eq!(probes, 0);
            } else {
                // This candidate exists only when coverage supplies flat-palette
                // evidence, not merely when the pilot's screen is available.
                assert_eq!(first.speed_tier, 1);
                assert_eq!(first.predictor, Some(Predictor::Zero));
                assert_eq!(first.learning_fraction, 0.0);
                assert!(probes > 0);
            }
        }
    }
}

#[test]
fn pilot_scan_keeps_ordinary_threshold_and_explicit_scan_opt_out() {
    let mut options = DriverOptions::scout();
    options.scout_pilot_first = true;
    options.screen.regions.full_width_strips = true;
    for height in [1023, 1024] {
        let (first, probes) = first_boundary_candidate(height, &options);
        if height < 1024 {
            assert_eq!(first, CandidateParams::default().backend(3).unwrap());
            assert_eq!(probes, 0);
        } else {
            assert_eq!(first.predictor, Some(Predictor::Zero));
            assert_eq!(first.learning_fraction, 0.0);
            assert!(probes > 0);
        }
    }
    options.preflight.enabled = true;
    options.coverage_scan = false;
    let (first, probes) = first_boundary_candidate(256, &options);
    assert_eq!(first, CandidateParams::default().backend(9).unwrap());
    assert!(probes > 0);
    options = DriverOptions::scout();
    let (first, probes) = first_boundary_candidate(256, &options);
    assert_eq!(first, CandidateParams::default().backend(3).unwrap());
    assert_eq!(probes, 0);
}

#[test]
fn retained_output_is_the_winning_invocation_and_ties_are_stable() {
    let image = image();
    let mut search = Search::new(SearchLimits::default());
    let params = CandidateParams::default();
    let mut calls = 0;
    let mut encode = |_: &CandidateParams, _: &Image, _: TrialBudget| {
        calls += 1;
        Ok(complete(20, calls))
    };
    search
        .refine(&image, 5, params.clone(), true, false, &mut encode)
        .unwrap();
    search
        .refine(&image, 2, params.clone(), true, false, &mut encode)
        .unwrap();
    search
        .refine(&image, 2, params, true, false, &mut encode)
        .unwrap();
    let output = search.finish().unwrap();
    assert_eq!(output.artifact, 2);
    assert_eq!(output.bytes, vec![2; 20]);
    assert_eq!(output.winner_index, 1);
    assert_eq!(calls, 3);
}

#[test]
fn real_bound_is_not_a_size_or_a_winner_and_is_reused() {
    let image = image();
    let mut search = Search::new(SearchLimits::default());
    let baseline = CandidateParams::default();
    search
        .refine(
            &image,
            1,
            baseline.clone(),
            true,
            false,
            &mut |_, _, budget| {
                assert_eq!(budget.beat_bytes, None);
                Ok(complete(100, 1))
            },
        )
        .unwrap();
    let challenger = baseline.backend(9).unwrap();
    let size = search
        .refine(
            &image,
            3,
            challenger.clone(),
            false,
            false,
            &mut |_, _, budget| {
                assert_eq!(budget.beat_bytes, Some(100));
                Ok(TrialOutput::Bounded { lower_bound: 123 })
            },
        )
        .unwrap();
    assert_eq!(size, None);
    assert_eq!(search.trials()[1].lower_bound, Some(100));
    assert_eq!(search.best_before(4, false), Some(100));
    search
        .refine(&image, 4, challenger, false, false, &mut |_, _, _| {
            panic!("bounded candidate rerun")
        })
        .unwrap();
    assert_eq!(search.finish().unwrap().artifact, 1);
}

#[test]
fn baseline_and_calibration_cannot_be_reported_as_bounded() {
    for calibration in [false, true] {
        let image = image();
        let mut search: Search<()> = Search::new(SearchLimits::default());
        let result = search.refine(
            &image,
            1,
            CandidateParams::default(),
            !calibration,
            calibration,
            &mut |_, _, budget| {
                assert_eq!(budget.beat_bytes, None);
                Ok(TrialOutput::Bounded { lower_bound: 100 })
            },
        );
        assert!(result.is_err());
    }
}

#[test]
fn callback_failure_cannot_replace_retained_output() {
    let image = image();
    let mut search = Search::new(SearchLimits::default());
    let params = CandidateParams::default();
    search
        .refine(&image, 1, params.clone(), true, false, &mut |_, _, _| {
            Ok(complete(100, 8))
        })
        .unwrap();
    let result = search.refine(
        &image,
        3,
        params.backend(9).unwrap(),
        false,
        false,
        &mut |_, _, _| Err(Error::Encoding("backend failure".into())),
    );
    assert!(result.is_err());
    assert_eq!(search.finish().unwrap().bytes, vec![8; 100]);
}

#[test]
fn failed_attempts_and_invalid_bounds_consume_work_without_becoming_sizes() {
    let image = image();
    let mut search = Search::new(SearchLimits {
        max_full_encodes: 2,
        ..Default::default()
    });
    let params = CandidateParams::default();
    search
        .refine(&image, 1, params.clone(), true, false, &mut |_, _, _| {
            Ok(complete(100, 1))
        })
        .unwrap();
    assert!(
        search
            .refine(
                &image,
                3,
                params.backend(9).unwrap(),
                false,
                false,
                &mut |_, _, _| Ok(TrialOutput::Bounded { lower_bound: 99 })
            )
            .is_err()
    );
    assert!(
        search
            .refine(
                &image,
                4,
                params.backend(8).unwrap(),
                false,
                false,
                &mut |_, _, _| panic!("work limit did not stop callback")
            )
            .is_err()
    );
    let output = search.finish().unwrap();
    assert_eq!(output.trials.len(), 1);
    assert_eq!(output.full_encodes, 2);
    assert_eq!(output.bytes.len(), 100);
}

#[test]
fn calibration_does_not_become_a_related_family_anchor() {
    let image = image();
    let mut search = Search::new(SearchLimits::default());
    let params = CandidateParams::default();
    search
        .refine(
            &image,
            1,
            params.backend(7).unwrap(),
            true,
            false,
            &mut |_, _, _| Ok(complete(100, 1)),
        )
        .unwrap();
    search
        .refine(
            &image,
            4,
            params.backend(4).unwrap(),
            false,
            true,
            &mut |_, _, b| {
                assert_eq!(b.beat_bytes, None);
                Ok(complete(90, 2))
            },
        )
        .unwrap();
    assert_eq!(search.related(&params.backend(9).unwrap()), Some(0));
    assert_eq!(search.lower_backend(&params.backend(9).unwrap()), Some(0));
    assert_eq!(search.best_before(5, true), Some(100));
    assert_eq!(search.best_before(5, false), Some(90));
    assert_eq!(search.finish().unwrap().artifact, 2);
}

#[test]
fn sibling_race_measures_whole_e4_witnesses_and_replays_original_at_e10() {
    let image = image();
    for (level, proposal_bytes, enabled, deferred) in [
        (9, 10025, true, false),
        (9, 10026, true, true),
        (10, 10026, true, true),
        (9, 10026, false, false),
    ] {
        let mut calls = Vec::new();
        let params = CandidateParams {
            speed_tier: 10 - level,
            ..Default::default()
        };
        let options = DriverOptions {
            coverage_scan: true,
            sibling_race: enabled,
            ..Default::default()
        };
        let result = run_pixel_search(
            &image,
            &params,
            &options,
            |p, sample, budget| {
                assert_eq!(
                    (sample.width(), sample.height()),
                    (image.width(), image.height())
                );
                calls.push(p.clone());
                let bytes = if p.predictor == Some(Predictor::Variable) && p.palette_colors == 0 {
                    if p.speed_tier == 6 && p.learning_fraction == 0.01 {
                        assert_eq!(budget.beat_bytes, None);
                        if p.rct == Some(0) {
                            10000
                        } else {
                            proposal_bytes
                        }
                    } else if p.speed_tier == 1
                        && p.learning_fraction == 0.1
                        && p.group_size_shift == Some(3)
                    {
                        if p.rct == Some(0) { 9000 } else { 8000 }
                    } else {
                        12000
                    }
                } else {
                    20000
                };
                Ok(complete(bytes, calls.len()))
            },
            |p, _| {
                Ok(
                    if p.predictor == Some(Predictor::Variable) && p.palette_colors == 0 {
                        if p.rct == Some(0) { 10 } else { 11 }
                    } else {
                        100
                    },
                )
            },
        )
        .unwrap();
        let races: Vec<_> = result
            .report
            .events
            .iter()
            .filter_map(|event| match event {
                SearchEvent::SiblingRace {
                    stage,
                    params,
                    control_bytes,
                    proposal_bytes,
                    deferred,
                } => Some((*stage, params, *control_bytes, *proposal_bytes, *deferred)),
                _ => None,
            })
            .collect();
        if !enabled {
            assert!(races.is_empty());
            assert!(!result.report.trials.iter().any(|trial| trial.calibration));
            continue;
        }
        assert_eq!(races.len(), 1);
        let (stage, candidate, control, proposal, was_deferred) = races[0];
        assert_eq!(stage, 5);
        assert_eq!(
            (control, proposal, was_deferred),
            (10000, proposal_bytes, deferred)
        );
        let witnesses: Vec<_> = result
            .report
            .trials
            .iter()
            .filter(|t| t.calibration)
            .collect();
        assert_eq!(witnesses.len(), 2);
        assert!(witnesses.iter().all(|t| !t.baseline
            && t.params.speed_tier == 6
            && t.params.learning_fraction == 0.01));
        if deferred && level == 10 {
            let replay = result
                .report
                .trials
                .iter()
                .find(|t| t.params == *candidate)
                .unwrap();
            assert_eq!(replay.stage, 10);
            assert!(!replay.calibration);
            assert_eq!(result.params, *candidate);
            assert_eq!(result.bytes, vec![result.artifact as u8; 8000]);
            assert_eq!(calls[result.artifact - 1], *candidate);
            assert!(!result.report.events.iter().any(|event| matches!(event,
                SearchEvent::Screened { stage: 10, params, .. } if params == candidate)));
        } else if deferred {
            assert!(!result.report.trials.iter().any(|t| t.params == *candidate));
            assert_eq!(result.bytes.len(), 9000);
        }
    }
}

fn full_size(params: &CandidateParams) -> usize {
    if params.predictor == Some(Predictor::Zero)
        && params.palette_colors == 10000
        && params.learning_fraction == 0.0
    {
        700
    } else if params.predictor.is_none() {
        300 + usize::try_from(params.speed_tier).unwrap() * 100
    } else {
        500 + usize::try_from(params.speed_tier).unwrap() * 20
    }
}

#[test]
fn large_frame_driver_executes_frontier_and_keeps_effort_prefix() {
    let image = image();
    let options = DriverOptions {
        coverage_scan: true,
        ..Default::default()
    };
    let run = |level: u8| {
        let mut calls = Vec::new();
        let result = run_pixel_search(
            &image,
            &CandidateParams {
                speed_tier: 10 - level as i8,
                ..Default::default()
            },
            &options,
            |p, _, _| {
                calls.push(p.clone());
                Ok(complete(full_size(p), calls.len()))
            },
            |_, _| Ok(100),
        )
        .unwrap();
        assert_eq!(
            result.bytes,
            vec![result.artifact as u8; result.bytes.len()]
        );
        assert_eq!(result.params, calls[result.artifact - 1]);
        assert_eq!(
            result.report.backend_semantics,
            BackendSemantics::Unverified
        );
        assert!(
            result
                .report
                .events
                .iter()
                .any(|e| matches!(e, SearchEvent::EarlyRacing { stage: 4, .. }))
        );
        (result, calls)
    };
    let (short, short_calls) = run(4);
    let (long, long_calls) = run(10);
    assert_eq!(&long_calls[..short_calls.len()], short_calls.as_slice());
    assert!(long.bytes.len() <= short.bytes.len());
    assert!(long.report.trials.iter().any(|t| t.stage >= 4));
}

#[test]
fn default_large_frame_without_literal_competition_uses_refinement_gate() {
    let image = image();
    let options = DriverOptions {
        coverage_scan: true,
        ..Default::default()
    };
    let result = run_pixel_search(
        &image,
        &CandidateParams {
            speed_tier: 1,
            ..Default::default()
        },
        &options,
        |p, _, _| {
            Ok(complete(
                if p.predictor == Some(Predictor::Zero) {
                    10000
                } else {
                    1000
                },
                1,
            ))
        },
        |_, _| Ok(100),
    )
    .unwrap();
    assert!(
        !result
            .report
            .events
            .iter()
            .any(|e| matches!(e, SearchEvent::EarlyRacing { .. }))
    );
    assert!(result.report.events.iter().any(|e| matches!(
        e,
        SearchEvent::Screened {
            stage: 5,
            neighborhood: true,
            accepted: true,
            ..
        }
    )));
}

#[test]
fn rejected_raw_calibration_is_a_full_witness_not_a_bound_or_promoted_candidate() {
    use super::driver::CalibrationOptions;
    let image = image();
    let options = DriverOptions {
        coverage_scan: true,
        calibration: CalibrationOptions {
            enabled: true,
            fast: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = run_pixel_search(
        &image,
        &CandidateParams {
            speed_tier: 6,
            ..Default::default()
        },
        &options,
        |p, _, budget| {
            if p.predictor == Some(Predictor::Variable) && p.palette_colors == 0 {
                assert_eq!(p.speed_tier, 6);
                assert_eq!(budget.beat_bytes, None);
                Ok(complete(20000, 4))
            } else {
                Ok(complete(
                    if p.predictor.is_none() && p.speed_tier == 3 {
                        9000
                    } else {
                        10000
                    },
                    1,
                ))
            }
        },
        |p, _| {
            Ok(
                if p.predictor == Some(Predictor::Variable) && p.rct == Some(0) {
                    50
                } else {
                    100
                },
            )
        },
    )
    .unwrap();
    assert!(
        result
            .report
            .trials
            .iter()
            .any(|t| t.stage == 4 && t.calibration && t.params.speed_tier == 6)
    );
    assert!(result.report.events.iter().any(|e| matches!(
        e,
        SearchEvent::Calibration {
            kind: "cheap-family",
            accepted: false,
            ..
        }
    )));
    assert!(
        !result
            .report
            .trials
            .iter()
            .any(|t| t.params.predictor == Some(Predictor::Variable) && t.params.speed_tier == 1)
    );
    assert_eq!(result.bytes.len(), 9000);
}

#[test]
fn fast_palette_preselection_replays_deferred_original_plans_at_ten() {
    let image = image();
    let options = DriverOptions {
        coverage_scan: true,
        palette_preselect: true,
        ..Default::default()
    };
    let result = run_pixel_search(
        &image,
        &CandidateParams {
            speed_tier: 0,
            ..Default::default()
        },
        &options,
        |p, _, _| {
            let size = if p.predictor != Some(Predictor::Zero) {
                10000
            } else if p.learning_fraction == 0.0 {
                2000
            } else if p.learning_fraction == 0.01 {
                1500
            } else if p.learning_fraction == 0.03 {
                1600
            } else if p.learning_fraction == 1.0 {
                1400
            } else {
                1700
            };
            Ok(complete(size, 1))
        },
        |p, _| {
            Ok(if p.learning_fraction == 0.03 {
                50
            } else if p.learning_fraction == 1.0 {
                150
            } else {
                100
            })
        },
    )
    .unwrap();
    assert!(
        result
            .report
            .events
            .iter()
            .any(|e| matches!(e, SearchEvent::FastLiteral { enabled: true }))
    );
    assert!(
        result
            .report
            .trials
            .iter()
            .any(|t| t.stage == 10 && t.params.learning_fraction == 0.01)
    );
    assert!(
        result
            .report
            .trials
            .iter()
            .any(|t| t.stage == 10 && t.params.learning_fraction == 1.0)
    );
    assert_eq!(result.bytes.len(), 1400);
}

#[test]
fn native_adapter_activates_source_fields_without_claiming_complete_equivalence() {
    use super::adapter::{CapabilityGap, project_native};
    let image = Image::new(
        512,
        512,
        PixelLayout::Rgb,
        PixelData::U8(vec![4; 512 * 512 * 3]),
    )
    .unwrap();
    let params = CandidateParams {
        speed_tier: 1,
        patches: Override::Off,
        palette_colors: 0,
        channel_colors_percent: 0.0,
        channel_colors_pre_transform_percent: 0.0,
        ..Default::default()
    };
    let projection = project_native(&params, &image, &crate::EncodeOptions::default()).unwrap();
    let options = projection.options.as_ref().unwrap();
    assert_eq!(options.modular.rct_candidates, Some(19));
    assert_eq!(options.modular.weighted_candidates, Some(5));
    assert_eq!(
        options.modular.sampling.as_ref().unwrap().nb_repeats,
        0.5 * 1.3
    );
    assert_eq!(
        options.modular.max_tree_leaves,
        crate::modular::ma_train::MAX_LEAVES
    );
    assert!(options.modular.fixed_tree.is_none());
    assert!(
        !projection
            .gaps
            .contains(&CapabilityGap::AutomaticRctCostSearch)
    );
    assert!(
        !projection
            .gaps
            .contains(&CapabilityGap::WeightedPresetSearch)
    );
    assert!(
        !projection
            .gaps
            .contains(&CapabilityGap::PerChannelRandomTrainingSamples)
    );
    assert!(!projection.gaps.contains(&CapabilityGap::EntropyTierPolicy));
    assert!(
        !projection
            .gaps
            .contains(&CapabilityGap::NativeTreeLeafLimit)
    );
    assert!(options.modular.source_tree_entropy);
    assert!(!projection.gaps.contains(&CapabilityGap::TreeEntropyPolicy));
    assert!(projection.gaps.is_empty());
    assert!(projection.exact_options().is_ok());
}

#[test]
fn native_resource_ceiling_is_explicit_and_not_inferred_from_direct_presets() {
    use super::adapter::{NativeResourcePolicy, project_native, project_native_with_resources};
    let params = CandidateParams {
        patches: Override::Off,
        ..Default::default()
    };
    let image = image();
    let mut base = crate::EncodeOptions::default();
    base.modular.max_tree_leaves = 16;
    base.modular.sample_limit = 123456;
    let projection = project_native(&params, &image, &base).unwrap();
    assert_eq!(
        projection.resources.max_tree_leaves,
        crate::modular::ma_train::MAX_LEAVES
    );
    assert_eq!(projection.resources.sample_limit, 123456);
    let projection = project_native_with_resources(
        &params,
        &image,
        &base,
        NativeResourcePolicy {
            max_tree_leaves: Some(64),
        },
    )
    .unwrap();
    let options = projection.options.unwrap();
    assert_eq!(projection.resources.max_tree_leaves, 64);
    assert_eq!(options.modular.max_tree_leaves, 64);
    assert_eq!(options.modular.sample_limit, 123456);
    assert!(options.modular.sampling.is_some());
    for limit in [0, crate::modular::ma_train::MAX_LEAVES + 1] {
        assert!(
            project_native_with_resources(
                &params,
                &image,
                &base,
                NativeResourcePolicy {
                    max_tree_leaves: Some(limit)
                },
            )
            .is_err()
        );
    }
}

#[test]
fn fixed_tiers_select_source_templates_independently_of_predictor_and_leaf_budget() {
    use super::adapter::{CapabilityGap, project_native};
    use crate::modular::encode::FixedTree;
    let image = image();
    for (tier, kind) in [
        (7, FixedTree::Weighted),
        (8, FixedTree::Gradient),
        (9, FixedTree::Gradient),
    ] {
        let params = CandidateParams {
            speed_tier: tier,
            predictor: Some(Predictor::Zero),
            ..Default::default()
        };
        let projection = project_native(&params, &image, &crate::EncodeOptions::default()).unwrap();
        assert_eq!(
            projection.options.as_ref().unwrap().modular.fixed_tree,
            Some(kind)
        );
        assert!(!projection.gaps.contains(&CapabilityGap::FixedBackendTrees));
        assert!(
            !projection
                .gaps
                .contains(&CapabilityGap::NativeTreeLeafLimit)
        );
    }
    let params = CandidateParams {
        speed_tier: 9,
        ..Default::default()
    };
    let projection = project_native(&params, &image, &crate::EncodeOptions::default()).unwrap();
    assert!(!projection.gaps.contains(&CapabilityGap::FixedBackendTrees));
}

#[test]
fn inherited_custom_tree_cannot_replace_source_candidate_policy() {
    use super::adapter::project_native;
    use crate::modular::tree::Tree;
    let image = image();
    let mut base = crate::EncodeOptions::default();
    base.modular.custom_tree = Some(Tree::channel_predictors(&[Predictor::Zero]).unwrap());
    for tier in [0, 1, 3, 7, 8, 9] {
        let params = CandidateParams {
            speed_tier: tier,
            patches: Override::Off,
            ..Default::default()
        };
        let options = project_native(&params, &image, &base)
            .unwrap()
            .options
            .unwrap();
        assert!(options.modular.custom_tree.is_none());
        assert_eq!(options.modular.fixed_tree.is_some(), tier >= 7);
        assert!(options.modular.source_tree_entropy);
    }
}

#[test]
fn inherited_tree_groups_and_splits_cannot_replace_source_candidate_policy() {
    use super::adapter::project_native;
    use crate::modular::tree_groups::TreeGroupOptions;
    let image = image();
    for (has_groups, has_splits) in [(true, false), (true, true), (false, true)] {
        let mut base = crate::EncodeOptions::default();
        base.modular.tree_groups = has_groups.then_some(TreeGroupOptions {
            requested_shards: 4,
            min_work: 1,
            parallel: false,
            ..Default::default()
        });
        if has_splits {
            base.modular.tree_group_splits = vec![0, 4, 16];
        }
        let original_splits = base.modular.tree_group_splits.clone();
        for tier in 0..=9 {
            let params = CandidateParams {
                speed_tier: tier,
                patches: Override::Off,
                ..Default::default()
            };
            let options = project_native(&params, &image, &base)
                .unwrap()
                .options
                .unwrap();
            assert!(options.modular.tree_groups.is_none());
            assert!(options.modular.tree_group_splits.is_empty());
            assert_eq!(options.modular.fixed_tree.is_some(), tier >= 7);
            assert!(options.shared_tree);
            assert_eq!(base.modular.tree_groups.is_some(), has_groups);
            assert_eq!(base.modular.tree_group_splits, original_splits);
        }
    }
}

#[test]
fn source_palette_modes_and_channel_only_policy_are_not_legacy_palette_proxies() {
    use super::{
        adapter::{CapabilityGap, project_native},
        params::DeltaPalette,
    };
    use crate::modular::palette::PaletteMode;
    let image = image();
    for setting in 0..=6 {
        let params = CandidateParams {
            palette_colors: 70911,
            delta_palette: DeltaPalette::from_source_setting(setting).unwrap(),
            ..Default::default()
        };
        let projection = project_native(&params, &image, &crate::EncodeOptions::default()).unwrap();
        let options = projection.options.as_ref().unwrap();
        let palette = options.modular.palette_policy.as_ref().unwrap();
        assert_eq!(palette.palette_colors, 70911);
        assert_eq!(
            palette.mode,
            PaletteMode::from_source_mode(setting).unwrap()
        );
        assert_eq!(palette.channel_colors_percent, 80.0);
        assert_eq!(palette.channel_colors_pre_transform_percent, 95.0);
        assert_eq!(palette.speed_tier, 3);
        assert!(options.modular.palette_colors.is_none());
        assert!(options.modular.palette_predictor.is_none());
        for gap in [
            CapabilityGap::GlobalAndChannelPaletteHeuristics,
            CapabilityGap::PaletteCostRejection,
            CapabilityGap::PaletteRctTransformOrder,
            CapabilityGap::ExactZeroDeltaInsertion,
            CapabilityGap::ExactDeltaAbove65536,
        ] {
            assert!(!projection.gaps.contains(&gap));
        }
    }
    let params = CandidateParams {
        palette_colors: 0,
        ..Default::default()
    };
    let projection = project_native(&params, &image, &crate::EncodeOptions::default()).unwrap();
    assert_eq!(
        projection
            .options
            .unwrap()
            .modular
            .palette_policy
            .unwrap()
            .channel_colors_percent,
        80.0
    );
}

#[test]
fn source_lz_parser_receives_method_and_shared_wire_configuration() {
    use super::adapter::{CapabilityGap, project_native};
    use crate::tools::lz77::SourceLz77Method;
    let image = image();
    for (tier, probe, expected) in [
        (5, 0, SourceLz77Method::Lz77b1w3t),
        (3, 0, SourceLz77Method::Lz77b3w3t),
        (1, 0, SourceLz77Method::Optc256),
        (1, 2, SourceLz77Method::Optc8),
    ] {
        let params = CandidateParams {
            speed_tier: tier,
            predictor: Some(Predictor::Zero),
            probe_lz77: probe,
            bucket_matcher: true,
            lz77_careful_first_pass: true,
            ..Default::default()
        };
        let projection = project_native(&params, &image, &crate::EncodeOptions::default()).unwrap();
        let options = &projection.options.as_ref().unwrap().modular;
        let source = options.source_lz77.as_ref().unwrap();
        assert_eq!(source.method, expected);
        assert!(source.bucket_matcher && source.careful_first_pass);
        assert_eq!(
            source.length_uint_config,
            options.entropy_options.lz77_length_config
        );
        assert_eq!(source.min_symbol(), options.entropy_options.lz77_min_symbol);
        let literal_config = match options.entropy_options.uint_config {
            crate::entropy::options::UintConfigStrategy::Fixed(config) => config,
            _ => Default::default(),
        };
        assert_eq!(source.literal_uint_config, literal_config);
        assert_eq!(options.entropy, crate::entropy::stream::Coding::Source);
        for gap in [
            CapabilityGap::LzMethodConfiguration,
            CapabilityGap::CarefulFirstPass,
            CapabilityGap::BucketMatcherSelection,
            CapabilityGap::LzEntropyWireAlignment,
            CapabilityGap::EntropyAlphabetSearch,
        ] {
            assert!(!projection.gaps.contains(&gap));
        }
    }
    let params = CandidateParams {
        speed_tier: 7,
        predictor: Some(Predictor::Zero),
        ..Default::default()
    };
    let mut base = crate::EncodeOptions::default();
    base.modular.source_lz77 = Some(Default::default());
    let projection = project_native(&params, &image, &base).unwrap();
    let options = projection.options.unwrap();
    assert!(!options.modular.lz77);
    assert!(options.modular.source_lz77.is_none());
}

#[test]
fn source_patch_projection_uses_outer_gate_and_never_inherits_replace_policy() {
    use super::adapter::{CapabilityGap, project_native};
    let image = image();
    let base = crate::EncodeOptions {
        patches: Some(Default::default()),
        source_patches: Some(crate::tools::patches::SourcePatchConfig {
            min_peak: 1,
            fingerprint_divisor: 4,
            ..Default::default()
        }),
        responsive: true,
        ..Default::default()
    };
    for tier in 0..=9 {
        for request in [Override::Default, Override::On, Override::Off] {
            let params = CandidateParams {
                speed_tier: tier,
                patches: request,
                ..Default::default()
            };
            let projection = project_native(&params, &image, &base).unwrap();
            let options = projection.options.as_ref().unwrap();
            assert!(options.patches.is_none());
            assert!(!options.responsive);
            assert_eq!(options.passes.num_passes, 1);
            let enabled = tier < 6 && request != Override::Off;
            assert_eq!(options.source_patches.is_some(), enabled);
            assert_eq!(options.source_patch_reference.is_some(), enabled);
            assert!(projection.gaps.is_empty());
            assert!(
                !projection
                    .gaps
                    .contains(&CapabilityGap::AutomaticPatchPolicy)
            );
            if let Some(config) = &options.source_patches {
                assert_eq!(config.speed_tier, tier);
                assert!(config.enabled && config.do_color && config.fast);
                assert!(!config.floating_point && !config.groupwise);
                assert_eq!(config.decoding_speed_tier, 0);
                assert_eq!(config.min_peak, 2);
                assert_eq!(config.fingerprint_divisor, 1);
            }
        }
    }
}

#[test]
fn source_palette_responsive_route_is_explicitly_unavailable() {
    use super::adapter::{AdaptationState, CapabilityGap, project_native};
    let image = image();
    let params = CandidateParams {
        squeeze: vec![true, false, true],
        ..Default::default()
    };
    let projection = project_native(&params, &image, &crate::EncodeOptions::default()).unwrap();
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
}
