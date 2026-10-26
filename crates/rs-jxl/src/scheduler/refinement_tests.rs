// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
use super::super::super::search::EncodedOutput;
use super::*;
use crate::{PixelData, PixelLayout, modular::predictor::Predictor};

type Full<'a> =
    Box<dyn FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<()>> + 'a>;
type Harness<'a> = Driver<'a, Full<'a>, ProbeCallback<'a>, ()>;

struct Cost {
    params: CandidateParams,
    full: usize,
    main: usize,
    context: usize,
}

fn cost(params: &CandidateParams, full: usize, main: usize, context: usize) -> Cost {
    Cost {
        params: params.clone(),
        full,
        main,
        context,
    }
}

fn image(pixels: u32) -> Image {
    Image::new(
        512,
        pixels / 512,
        PixelLayout::Gray,
        PixelData::U8(vec![0; pixels as usize]),
    )
    .unwrap()
}

fn raw(input: &CandidateParams) -> CandidateParams {
    CandidateParams {
        palette_colors: 0,
        predictor: Some(Predictor::Variable),
        rct: Some(0),
        ..input.backend(9).unwrap()
    }
}

fn calibrated() -> DriverOptions {
    DriverOptions {
        calibration: CalibrationOptions {
            enabled: true,
            fast: true,
            routes: true,
            refinements: true,
        },
        ..Default::default()
    }
}

fn fixture<'a>(
    image: &'a Image,
    input: &'a CandidateParams,
    options: &'a DriverOptions,
    costs: Vec<Cost>,
) -> Harness<'a> {
    let costs = Rc::new(costs);
    let full_costs = Rc::clone(&costs);
    let encode: Full<'a> = Box::new(move |p, _, _| {
        let bytes = full_costs
            .iter()
            .find(|c| c.params == *p)
            .map_or(100_000, |c| c.full);
        Ok(TrialOutput::Complete(EncodedOutput {
            bytes: vec![0; bytes],
            artifact: (),
        }))
    });
    let callback: ProbeCallback<'a> = Box::new(move |p, sample| {
        Ok(costs.iter().find(|c| c.params == *p).map_or(1000, |c| {
            if sample.width() == 128 && sample.height() == 128 {
                c.context
            } else {
                c.main
            }
        }))
    });
    let probe = Rc::new(RefCell::new(ProbeWork {
        callback,
        limits: options.screen.limits,
        encodes: 0,
        pixels: 0,
    }));
    let shared = Rc::clone(&probe);
    let main: ProbeCallback<'a> = Box::new(move |p, sample| shared.borrow_mut().run(p, sample));
    let shared = Rc::clone(&probe);
    let context: ProbeCallback<'a> = Box::new(move |p, sample| shared.borrow_mut().run(p, sample));
    let mut screen_options = options.screen.clone();
    screen_options.adaptive_coverage = true;
    screen_options.regions.context_side = 64;
    let screen = CandidateScreen::new(image, None, screen_options.clone(), main).unwrap();
    screen_options.regions.context_side = 128;
    let context_screen = CandidateScreen::new(image, None, screen_options, context).unwrap();
    Driver {
        image,
        input,
        options,
        scan: None,
        level: 11,
        search: Search::new(options.full_limits),
        encode,
        full_batch: None,
        pilot_overlap: None,
        probe,
        probe_batch: None,
        prefix_batch: None,
        screen,
        context_screen,
        auxiliary_screen_batches: 0,
        routes: ActiveRoutes::default(),
        calibration_feedback: Vec::new(),
        early_racing: false,
        early_family_winner: false,
        pending_automatic: false,
        analysis: ProbeResult::default(),
        frontier: Vec::new(),
        deferred: Vec::new(),
        confirmation_cache: None,
        events: Vec::new(),
    }
}

fn has_event(driver: &Harness<'_>, name: &str, keep: bool) -> bool {
    driver.events.iter().any(|e| {
        matches!(e, SearchEvent::Calibration {kind, accepted, ..}
        if *kind == name && *accepted == keep)
    })
}

fn deferred(driver: &Harness<'_>, candidate: &CandidateParams, screen: bool) {
    assert_eq!(driver.deferred.len(), 1);
    assert_eq!(driver.deferred[0].params, *candidate);
    assert_eq!(driver.deferred[0].screen, screen);
    assert!(driver.search.find(candidate).is_none());
}

#[test]
fn statistical_projection_matches_source_goldens_and_rejects_missing_evidence() {
    // modular_test.cc's ConservativeLosslessProjection fixtures.
    assert!((evidence::conservative_projection(1000, &[120; 3], &[100; 3]) - 1100.0).abs() < 1e-9);
    assert_eq!(
        evidence::conservative_projection(1000, &[120; 2], &[100; 2]),
        0.0
    );
    assert_eq!(
        evidence::conservative_projection(1000, &[120; 3], &[0; 3]),
        0.0
    );
    assert_eq!(
        evidence::conservative_projection(1000, &[120; 3], &[100; 4]),
        0.0
    );
    assert_eq!(
        evidence::conservative_projection(1000, &[0; 3], &[100; 3]),
        0.0
    );
    assert!(evidence::conservative_projection(1000, &[180, 120, 80], &[100; 3]) < 1000.0);
}

#[test]
fn context_gate_preserves_source_thresholds_and_local_wins() {
    let control = [10000; 3];
    assert!(evidence::context_defers(&[9000; 3], &control, &[9900; 3], &control).unwrap());
    assert!(evidence::context_defers(&[9800; 3], &control, &[9800; 3], &control).unwrap());
    assert!(!evidence::context_defers(&[9799; 3], &control, &[9800; 3], &control).unwrap());
    assert!(
        !evidence::context_defers(&[9000; 3], &control, &[9600, 11000, 11000], &control).unwrap()
    );
    assert!(
        evidence::context_defers(&[9000; 3], &control, &[9700, 11000, 11000], &control).unwrap()
    );
    assert!(!evidence::context_defers(&[], &[], &[], &[]).unwrap());
    assert!(evidence::context_defers(&[1], &[1, 2], &[1], &[1]).is_err());
    assert!(evidence::context_defers(&[1], &[1], &[1], &[1, 2]).is_err());
    assert!(evidence::context_defers(&[1], &[usize::MAX, 1], &[], &[]).is_err());
}

#[test]
fn peer_context_gate_requires_strict_total_loss_and_no_local_escape() {
    let control = [10000; 3];
    assert!(!evidence::peer_context_loses(&[10100; 3], &control).unwrap());
    assert!(evidence::peer_context_loses(&[10101; 3], &control).unwrap());
    assert!(!evidence::peer_context_loses(&[9949, 11000, 11000], &control).unwrap());
    assert!(evidence::peer_context_loses(&[9950, 11000, 11000], &control).unwrap());
    assert!(evidence::peer_context_loses(&[1], &[1, 2]).is_err());
    assert!(evidence::peer_context_loses(&[usize::MAX, 1], &[1, 1]).is_err());
}

#[test]
fn small_automatic_witness_bypasses_screen_and_keeps_complete_baseline() {
    let image = image(1 << 17);
    let input = CandidateParams::default();
    let options = calibrated();
    let candidate = input.backend(9).unwrap();
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![cost(&candidate, 90_000, 5000, 5000)],
    );
    d.full(1, input.backend(3).unwrap(), true, false).unwrap();
    d.routes.preflight_yield = true;
    assert_eq!(
        d.refine(4, candidate, false, false, false).unwrap(),
        Some(90_000)
    );
    assert_eq!(d.probe.borrow().encodes, 0);
    assert!(d.search.trials().last().unwrap().baseline);
}

#[test]
fn paired_rct_veto_survives_neighborhood_but_multiscale_or_exploration_can_admit() {
    let image = image(1 << 19);
    let input = CandidateParams::default();
    let options = DriverOptions::default();
    let incumbent = raw(&input);
    let candidate = CandidateParams {
        rct: Some(6),
        ..incumbent.clone()
    };
    for (multiscale, exploration, keep) in [
        (false, false, false),
        (true, false, true),
        (false, true, true),
    ] {
        let mut d = fixture(
            &image,
            &input,
            &options,
            vec![cost(&candidate, 90_000, 994, 999)],
        );
        d.full(3, incumbent.clone(), true, false).unwrap();
        d.routes.broad_scope = true;
        d.routes.multiscale_rank = multiscale;
        let result = d
            .refine(5, candidate.clone(), false, exploration, true)
            .unwrap();
        assert_eq!(result.is_some(), keep);
        if !keep {
            deferred(&d, &candidate, false);
        }
        assert!(d.context_screen.encodes() > 0);
    }
}

#[test]
fn paired_rct_context_uses_the_same_cumulative_probe_limit() {
    let image = image(1 << 19);
    let input = CandidateParams::default();
    let options = DriverOptions::default();
    let incumbent = raw(&input);
    let candidate = CandidateParams {
        rct: Some(6),
        ..incumbent.clone()
    };
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![cost(&candidate, 90_000, 990, 990)],
    );
    d.full(3, incumbent.clone(), true, false).unwrap();
    d.routes.capacity_mismatch = true;
    let count = d.screen.score(&incumbent).unwrap().len();
    d.probe.borrow_mut().limits.max_encodes = count * 2;
    assert!(matches!(
        d.refine(5, candidate.clone(), false, false, false),
        Err(Error::Limit("cumulative scheduler probe work"))
    ));
    assert!(d.search.find(&candidate).is_none());
    assert!(d.deferred.is_empty());
}

#[test]
fn early_context_disappearing_gain_defers_without_full_calibration() {
    let image = image(1 << 17);
    let input = CandidateParams::default();
    let options = calibrated();
    let candidate = raw(&input);
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![cost(&candidate, 90_000, 900, 990)],
    );
    d.full(3, input.backend(9).unwrap(), true, false).unwrap();
    d.routes.preflight_yield = true;
    d.routes.context_feedback = true;
    d.early_racing = true;
    assert_eq!(
        d.refine(4, candidate.clone(), false, false, false).unwrap(),
        None
    );
    deferred(&d, &candidate, false);
    assert_eq!(d.search.trials().len(), 1);
    assert!(has_event(&d, "context-validation", false));
}

#[test]
fn successful_early_context_skips_full_calibration_instead_of_running_both() {
    let image = image(1 << 17);
    let input = CandidateParams::default();
    let options = calibrated();
    let candidate = raw(&input);
    let medium = candidate.backend(7).unwrap();
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![
            cost(&candidate, 90_000, 990, 960),
            cost(&medium, 200_000, 1000, 1000),
        ],
    );
    d.full(3, input.backend(9).unwrap(), true, false).unwrap();
    d.routes.preflight_yield = true;
    d.routes.context_feedback = true;
    d.early_racing = true;
    assert_eq!(
        d.refine(4, candidate, false, false, false).unwrap(),
        Some(90_000)
    );
    assert!(has_event(&d, "context-validation", true));
    assert_eq!(d.search.trials().len(), 2);
    assert!(d.search.find(&medium).is_none());
}

#[test]
fn compact_neighborhood_is_deferred_then_admitted_at_e10() {
    let image = image(1 << 19);
    let input = CandidateParams::default();
    let options = DriverOptions::default();
    let incumbent = input.backend(9).unwrap();
    let candidate = CandidateParams {
        group_size_shift: Some(2),
        ..incumbent.clone()
    };
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![cost(&candidate, 90_000, 990, 990)],
    );
    d.full(3, incumbent, true, false).unwrap();
    d.routes.preflight_yield = true;
    assert_eq!(
        d.refine(9, candidate.clone(), false, false, true).unwrap(),
        None
    );
    deferred(&d, &candidate, true);
    assert_eq!(
        d.refine(10, candidate, false, false, true).unwrap(),
        Some(90_000)
    );
}

#[test]
fn medium_feedback_transfers_to_same_raw_rct_sibling_only() {
    let image = image(1 << 19);
    let input = CandidateParams::default();
    let options = DriverOptions {
        sibling_transfer: true,
        ..calibrated()
    };
    let candidate = raw(&input);
    let medium = candidate.backend(7).unwrap();
    let sibling = CandidateParams {
        rct: Some(6),
        ..candidate.clone()
    };
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![
            cost(&candidate, 90_000, 990, 990),
            cost(&medium, 150_000, 1000, 1000),
            cost(&sibling, 90_000, 990, 990),
        ],
    );
    d.full(3, input.backend(9).unwrap(), true, false).unwrap();
    d.routes.preflight_yield = true;
    assert_eq!(
        d.refine(5, candidate.clone(), false, false, false).unwrap(),
        None
    );
    assert_eq!(d.calibration_feedback.len(), 1);
    assert_eq!(d.calibration_feedback[0].params, candidate);
    let count = d.search.trials().len();
    assert_eq!(
        d.refine(6, sibling.clone(), false, false, false).unwrap(),
        None
    );
    assert_eq!(d.search.trials().len(), count);
    assert_eq!(d.deferred.last().unwrap().params, sibling);
    assert!(!d.deferred.last().unwrap().screen);
    assert!(has_event(&d, "sibling-transfer", false));
}

#[test]
fn lower_backend_deep_peer_gain_can_overrule_an_optimistic_proxy_loss() {
    let image = image(2 << 20);
    let input = CandidateParams::default();
    let options = calibrated();
    let candidate = input.backend(9).unwrap();
    let incumbent = CandidateParams {
        group_size_shift: Some(2),
        ..candidate.clone()
    };
    let parent = candidate.backend(7).unwrap();
    for preflight in [false, true] {
        let mut d = fixture(
            &image,
            &input,
            &options,
            vec![
                cost(&candidate, 90_000, 990, 990),
                cost(&parent, 120_000, 1000, 1000),
            ],
        );
        d.full(2, parent.clone(), true, false).unwrap();
        d.full(3, incumbent.clone(), true, false).unwrap();
        d.routes.preflight_yield = preflight;
        let result = d.refine(8, candidate.clone(), false, false, false).unwrap();
        assert_eq!(result.is_some(), preflight);
        if !preflight {
            deferred(&d, &candidate, true);
        }
        assert!(has_event(&d, "lower-backend", preflight));
    }
}

#[test]
fn measured_lower_backend_replaces_redundant_training_fraction_calibration() {
    let image = image(2 << 20);
    let input = CandidateParams::default();
    let options = calibrated();
    let incumbent = input.backend(9).unwrap();
    let candidate = CandidateParams {
        learning_fraction: 2.0,
        ..incumbent.clone()
    };
    let parent = candidate.backend(7).unwrap();
    let mut cheap = candidate.backend(4).unwrap();
    cheap.learning_fraction = 0.01;
    for preflight in [false, true] {
        let mut d = fixture(
            &image,
            &input,
            &options,
            vec![
                cost(&candidate, 90_000, 990, 990),
                cost(&parent, 110_000, 1200, 1200),
                cost(&cheap, 200_000, 1000, 1000),
            ],
        );
        d.full(2, parent.clone(), true, false).unwrap();
        d.full(3, incumbent.clone(), true, false).unwrap();
        d.routes.preflight_yield = preflight;
        let result = d.refine(8, candidate.clone(), false, false, false).unwrap();
        assert_eq!(result.is_some(), preflight);
        assert_eq!(d.search.find(&cheap).is_none(), preflight);
        if !preflight {
            deferred(&d, &candidate, false);
        }
    }
}

#[test]
fn automatic_yield_compares_context_to_the_completed_automatic_parent() {
    let image = image(2 << 20);
    let input = CandidateParams::default();
    let options = calibrated();
    let candidate = input.backend(9).unwrap();
    let parent = candidate.backend(7).unwrap();
    let incumbent = raw(&input);
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![
            cost(&candidate, 90_000, 1000, 900),
            cost(&parent, 120_000, 1100, 1000),
            cost(&incumbent, 100_000, 1050, 2000),
        ],
    );
    d.full(2, parent, true, false).unwrap();
    d.full(3, incumbent, true, false).unwrap();
    d.routes.preflight_yield = true;
    d.routes.automatic_yield = true;
    assert_eq!(
        d.refine(8, candidate.clone(), false, true, false).unwrap(),
        None
    );
    deferred(&d, &candidate, false);
    assert!(has_event(&d, "automatic-yield", false));
}

#[test]
fn deep_peer_context_loss_changes_replay_to_unscreened() {
    let image = image(2 << 20);
    let input = CandidateParams::default();
    let options = calibrated();
    let candidate = input.backend(9).unwrap();
    let incumbent = CandidateParams {
        group_size_shift: Some(2),
        ..candidate.clone()
    };
    let parent = candidate.backend(7).unwrap();
    let mut d = fixture(
        &image,
        &input,
        &options,
        vec![
            cost(&candidate, 90_000, 1020, 1030),
            cost(&parent, 120_000, 1300, 1000),
        ],
    );
    d.full(2, parent, true, false).unwrap();
    d.full(3, incumbent, true, false).unwrap();
    d.routes.preflight_yield = true;
    d.routes.context_feedback = true;
    assert_eq!(
        d.refine(8, candidate.clone(), false, true, false).unwrap(),
        None
    );
    deferred(&d, &candidate, false);
    assert!(has_event(&d, "peer-context", false));
    assert!(has_event(&d, "lower-backend", false));
}

#[test]
fn plan_first_marginal_geometry_deduplicates_exact_deferred_plan() {
    let image = image(2 << 20);
    let input = CandidateParams::default();
    let options = calibrated();
    let incumbent = raw(&input);
    let candidate = CandidateParams {
        group_size_shift: Some(2),
        ..incumbent.clone()
    };
    let mut d = fixture(&image, &input, &options, Vec::new());
    d.full(3, incumbent, true, false).unwrap();
    d.routes.preflight_yield = true;
    d.routes.plan_first = true;
    for _ in 0..2 {
        assert_eq!(
            d.refine(9, candidate.clone(), false, false, true).unwrap(),
            None
        );
        deferred(&d, &candidate, false);
    }
    assert_eq!(d.search.trials().len(), 1);
}

#[test]
fn reused_geometry_control_and_closer_first_preserve_calibration_order() {
    let image = image(2 << 20);
    let input = CandidateParams::default();
    let options = calibrated();
    let incumbent = raw(&input);
    let candidate = CandidateParams {
        group_size_shift: Some(2),
        ..incumbent.clone()
    };
    let mut control = incumbent.backend(4).unwrap();
    control.learning_fraction = 0.01;
    let mut cheap = candidate.backend(4).unwrap();
    cheap.learning_fraction = 0.01;
    let medium = candidate.backend(7).unwrap();
    for closer in [false, true] {
        let mut d = fixture(
            &image,
            &input,
            &options,
            vec![
                cost(&candidate, 90_000, 990, 990),
                cost(&control, 150_000, 1000, 1000),
                cost(&cheap, 149_000, 950, 950),
                cost(&medium, 100_000, 1000, 1000),
            ],
        );
        d.full(3, incumbent.clone(), true, false).unwrap();
        d.calibration_full(7, control.clone()).unwrap();
        d.routes.preflight_yield = true;
        d.routes.calibration_reuse = true;
        d.routes.calibration_order = closer;
        assert_eq!(
            d.refine(9, candidate.clone(), false, false, true).unwrap(),
            Some(90_000)
        );
        let trials = d.search.trials();
        assert_eq!(
            trials[2].params,
            if closer {
                medium.clone()
            } else {
                cheap.clone()
            }
        );
        assert_eq!(
            trials[3].params,
            if closer {
                cheap.clone()
            } else {
                medium.clone()
            }
        );
        assert!(has_event(&d, "paired-geometry-ratio", true));
        assert!(has_event(&d, "medium-e7", true));
    }
}

#[test]
fn statistical_palette_peer_defers_without_encoding_another_cheap_twin() {
    let image = image(1 << 19);
    let input = CandidateParams::default();
    let incumbent = raw(&input);
    let candidate = CandidateParams {
        palette_colors: 256,
        predictor: Some(Predictor::Zero),
        learning_fraction: 0.1,
        ..input.backend(9).unwrap()
    };
    let mut cheap = candidate.backend(4).unwrap();
    cheap.learning_fraction = 0.01;
    let peer = CandidateParams {
        palette_colors: 512,
        ..cheap.clone()
    };
    for statistical in [false, true] {
        let options = DriverOptions {
            statistical_feedback: statistical,
            ..calibrated()
        };
        let mut d = fixture(
            &image,
            &input,
            &options,
            vec![
                cost(&candidate, 90_000, 990, 990),
                cost(&cheap, 100_000, 1000, 1000),
                cost(&peer, 150_000, 1000, 1000),
            ],
        );
        d.full(3, incumbent.clone(), true, false).unwrap();
        d.calibration_full(4, peer.clone()).unwrap();
        let result = d.refine(8, candidate.clone(), false, false, false).unwrap();
        assert_eq!(result.is_none(), statistical);
        if statistical {
            deferred(&d, &candidate, true);
            assert!(d.search.find(&cheap).is_none());
            assert!(has_event(&d, "statistical-peer", false));
        }
    }
}

#[test]
fn strong_small_candidate_avoids_e4_extrapolation_and_e11_is_deep() {
    let image = image(1 << 17);
    let input = CandidateParams::default();
    let options = calibrated();
    for effort in [9, 11] {
        let candidate = raw(&input).backend(effort).unwrap();
        let mut cheap = candidate.backend(4).unwrap();
        cheap.learning_fraction = 0.01;
        for preflight in [false, true] {
            let mut d = fixture(
                &image,
                &input,
                &options,
                vec![
                    cost(&candidate, 90_000, 940, 940),
                    cost(&cheap, 200_000, 1000, 1000),
                ],
            );
            d.full(3, input.backend(9).unwrap(), true, false).unwrap();
            d.routes.preflight_yield = preflight;
            let result = d.refine(8, candidate.clone(), false, false, false).unwrap();
            assert_eq!(result.is_some(), preflight);
            assert_eq!(d.search.find(&cheap).is_none(), preflight);
        }
    }
}

#[test]
fn early_promotion_needs_three_percent_gain_unless_plan_first_commits() {
    let image = image(1 << 19);
    let input = CandidateParams::default();
    let options = calibrated();
    let candidate = raw(&input);
    let mut cheap = candidate.backend(4).unwrap();
    cheap.learning_fraction = 0.01;
    for (stage, early, committed, keep) in [
        (4, false, false, true),
        (3, true, false, false),
        (3, true, true, true),
    ] {
        let mut d = fixture(
            &image,
            &input,
            &options,
            vec![
                cost(&candidate, 90_000, 980, 980),
                cost(&cheap, 100_000, 1000, 1000),
            ],
        );
        d.full(1, input.backend(3).unwrap(), true, false).unwrap();
        d.routes.plan_first = committed;
        let result = d
            .refine_promoted(stage, candidate.clone(), false, false, false, early)
            .unwrap();
        assert_eq!(result.is_some(), keep);
        assert_eq!(d.search.find(&cheap).is_none(), committed);
    }
}
