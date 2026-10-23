//! Search-policy tests use mock sizes, not codec correctness claims.
use super::{
    CandidateParams, DirectFrontier, DriverOptions, EncodedOutput, FrontierOptions,
    PreflightOptions, ScopeRouting, TrialOutput, WorkerLimits,
    batch::BatchEncoders,
    driver::{CalibrationOptions, SearchEvent},
    params::Override,
    run_pixel_search, run_pixel_search_with_batches,
    search::{SearchLimits, TrialBudget},
};
use crate::{Error, Image, PixelData, PixelLayout, modular::predictor::Predictor};
use std::cell::Cell;

fn image(w: u32, h: u32) -> Image {
    Image::new(
        w,
        h,
        PixelLayout::Gray,
        PixelData::U8(vec![37; w as usize * h as usize]),
    )
    .unwrap()
}

fn input() -> CandidateParams {
    CandidateParams {
        speed_tier: 1,
        ..Default::default()
    }
}

fn output(p: &CandidateParams, size: usize) -> TrialOutput<CandidateParams> {
    TrialOutput::Complete(EncodedOutput {
        bytes: vec![0; size],
        artifact: p.clone(),
    })
}

fn route(result: &super::SearchResult<CandidateParams>, name: &str) -> bool {
    result
        .report
        .events
        .iter()
        .any(|e| matches!(e, SearchEvent::Route {name: n, ..} if *n == name))
}

#[test]
fn global_tournament_calibrates_all_plans_before_full_winner_and_retains_witness() {
    let image = image(128, 64);
    let options = DriverOptions {
        frontier: FrontierOptions {
            global_tournament: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut calls = Vec::new();
    let result = run_pixel_search(
        &image,
        &input(),
        &options,
        |p, _, b| {
            calls.push((p.clone(), b));
            Ok(output(
                p,
                if p.speed_tier == 3 && p.rct == Some(17) {
                    80
                } else {
                    100
                },
            ))
        },
        |_, _| panic!("global tournament does not crop"),
    )
    .unwrap();
    assert!(route(&result, "global-tournament"));
    assert_eq!(calls.len(), 9);
    assert!(
        calls[..8]
            .iter()
            .all(|(p, b)| p.speed_tier == 3 && b.beat_bytes.is_none())
    );
    assert_eq!(calls[8].0.rct, Some(17));
    assert_eq!(calls[8].0.speed_tier, 1);
    assert_eq!(calls[8].1.beat_bytes, Some(80));
    assert_eq!(result.bytes.len(), 80);
    assert_eq!(result.params, result.artifact);
    assert!(result.report.trials[result.report.winner_index].calibration);
}

#[test]
fn tournament_bounded_challenger_is_not_a_failed_encode() {
    let options = DriverOptions {
        frontier: FrontierOptions {
            global_tournament: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = run_pixel_search(
        &image(128, 64),
        &input(),
        &options,
        |p, _, b| {
            Ok(if let Some(limit) = b.beat_bytes {
                TrialOutput::Bounded { lower_bound: limit }
            } else {
                output(p, 100)
            })
        },
        |_, _| panic!("unexpected crop"),
    )
    .unwrap();
    assert_eq!(result.bytes.len(), 100);
    assert_eq!(result.report.trials.last().unwrap().lower_bound, Some(100));
}

#[test]
fn global_batch_obeys_worker_limit_and_commits_source_order() {
    let image = image(128, 64);
    let batches = Cell::new(0);
    let options = DriverOptions {
        frontier: FrontierOptions {
            global_tournament: true,
            ..Default::default()
        },
        workers: WorkerLimits {
            tasks: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = run_pixel_search_with_batches(
        &image,
        &input(),
        &options,
        |p, _, _| Ok(output(p, 100)),
        |_, _| panic!("unexpected crop"),
        BatchEncoders {
            full: Some(Box::new(|_, tasks| {
                batches.set(batches.get() + 1);
                assert!(tasks.len() <= 3);
                assert!(tasks.iter().all(|t| t.budget.beat_bytes.is_none()));
                Ok(tasks.iter().map(|t| output(t.params, 100)).collect())
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(batches.get(), 3);
    assert_eq!(result.report.full_encodes, 9);
    assert_eq!(result.report.trials[0].params.predictor, None);
    assert_eq!(result.report.trials[7].params.rct, Some(26));
    assert_eq!(result.report.winner_index, 0);
}

#[test]
fn tournament_batch_work_limit_is_checked_before_dispatch() {
    let options = DriverOptions {
        frontier: FrontierOptions {
            global_tournament: true,
            ..Default::default()
        },
        full_limits: SearchLimits {
            max_full_encodes: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = run_pixel_search_with_batches(
        &image(128, 64),
        &input(),
        &options,
        |_, _, _| -> crate::Result<TrialOutput<()>> { panic!("dispatch after limit") },
        |_, _| panic!("unexpected crop"),
        BatchEncoders {
            full: Some(Box::new(|_, _| panic!("dispatch after limit"))),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(Error::Limit(_))));
}

#[test]
fn compact_frontier_skips_full_calibration_and_accepts_only_full_output() {
    let image = image(1024, 512);
    for direct in [
        DirectFrontier::Sampled,
        DirectFrontier::Calibrated,
        DirectFrontier::Extended,
        DirectFrontier::Shrink,
        DirectFrontier::Dense,
    ] {
        let options = DriverOptions {
            frontier: FrontierOptions {
                direct,
                compact: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = run_pixel_search(
            &image,
            &input(),
            &options,
            |p, _, b: TrialBudget| {
                Ok(if let Some(limit) = b.beat_bytes {
                    TrialOutput::Bounded { lower_bound: limit }
                } else {
                    output(p, 120)
                })
            },
            |_, _| Ok(10),
        )
        .unwrap();
        assert!(route(&result, "direct-frontier"));
        assert_eq!(result.bytes.len(), 120);
        assert!(!result.report.trials.iter().any(|t| t.calibration));
        assert!((1..=2).contains(&result.report.full_encodes));
        assert!(result.report.probe_encodes > 0);
        assert_eq!(result.params, result.artifact);
    }
}

#[test]
fn large_direct_frontier_retains_complete_calibration_even_when_finalists_are_bounded() {
    let pixels = (0..1536 * 1536)
        .flat_map(|i| {
            let v = (i as u32).wrapping_mul(2654435761);
            [v as u8, (v >> 8) as u8, (v >> 16) as u8]
        })
        .collect();
    let image = Image::new(1536, 1536, PixelLayout::Rgb, PixelData::U8(pixels)).unwrap();
    for direct in [
        DirectFrontier::Calibrated,
        DirectFrontier::Extended,
        DirectFrontier::Shrink,
        DirectFrontier::Dense,
    ] {
        let options = DriverOptions {
            frontier: FrontierOptions {
                direct,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = run_pixel_search(
            &image,
            &input(),
            &options,
            |p, _, b| {
                Ok(if let Some(limit) = b.beat_bytes {
                    TrialOutput::Bounded { lower_bound: limit }
                } else {
                    output(p, 100)
                })
            },
            |_, _| Ok(10),
        )
        .unwrap();
        assert!(route(&result, "direct-frontier"));
        assert!(
            result
                .report
                .trials
                .iter()
                .any(|t| t.calibration && t.baseline && t.stage == 8)
        );
        assert!(
            result
                .report
                .trials
                .iter()
                .any(|t| t.lower_bound.is_some() && t.stage == 9)
        );
        assert_eq!(result.bytes.len(), 100);
    }
}

#[test]
fn strip_frontier_keeps_two_tied_full_trials_and_counts_probe_work() {
    let options = DriverOptions {
        frontier: FrontierOptions {
            strip_tournament: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = run_pixel_search(
        &image(1024, 512),
        &input(),
        &options,
        |p, _, _| Ok(output(p, 100)),
        |_, _| Ok(10),
    )
    .unwrap();
    assert!(route(&result, "strip-tournament"));
    assert!((1..=2).contains(&result.report.full_encodes));
    assert!(result.report.probe_encodes > 0 && result.report.probe_pixels > 0);
}

#[test]
fn strip_batch_report_includes_its_separate_screen() {
    let options = DriverOptions {
        frontier: FrontierOptions {
            strip_tournament: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let batches = Cell::new(0);
    let result = run_pixel_search_with_batches(
        &image(1024, 512),
        &input(),
        &options,
        |p, _, _| Ok(output(p, 100)),
        |_, _| Ok(10),
        BatchEncoders {
            planned_probe: Some(Box::new(|tasks, _| {
                batches.set(batches.get() + 1);
                Ok(vec![10; tasks.len()])
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(batches.get() > 0);
    assert_eq!(result.report.screen_batches, batches.get());
}

#[test]
fn enabled_preflight_routes_share_work_budget_and_retain_complete_winner() {
    let options = DriverOptions {
        coverage_scan: true,
        statistical_feedback: true,
        sibling_transfer: true,
        calibration: CalibrationOptions {
            enabled: true,
            fast: true,
            routes: true,
            refinements: true,
        },
        preflight: PreflightOptions {
            enabled: true,
            anchor_first: true,
            scope: ScopeRouting::Broad,
            multiscale_rank: true,
            calibration_reuse: true,
            early_family: true,
            calibration_order: true,
            plan_first: true,
            plan_buckets: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let calls = Cell::new(0);
    let result = run_pixel_search(
        &image(1024, 512),
        &input(),
        &options,
        |p, _, _| {
            Ok(output(
                p,
                if p.predictor == Some(Predictor::Variable) && p.palette_colors == 0 {
                    40
                } else {
                    100
                },
            ))
        },
        |p, _| {
            calls.set(calls.get() + 1);
            Ok(
                if p.predictor == Some(Predictor::Variable) && p.palette_colors == 0 {
                    10
                } else {
                    40
                },
            )
        },
    )
    .unwrap();
    assert_eq!(result.report.probe_encodes, calls.get());
    assert_eq!(result.params, result.artifact);
    assert_eq!(result.bytes.len(), 40);
    assert!(route(&result, "plan-first"));
    assert!(
        !result.params.bucket_matcher,
        "scope routing keeps original chain matcher"
    );
    assert_eq!(result.params.patches, Override::Off);
    let mut limited = options;
    limited.screen.limits.max_encodes = 1;
    let result = run_pixel_search(
        &image(1024, 512),
        &input(),
        &limited,
        |p, _, _| Ok(output(p, 100)),
        |_, _| Ok(10),
    );
    assert!(matches!(result, Err(Error::Limit(_))));
}

#[test]
fn plan_first_retains_bucket_leader_and_defers_chain_replay() {
    let options = DriverOptions {
        calibration: CalibrationOptions {
            enabled: true,
            fast: true,
            routes: true,
            refinements: true,
        },
        preflight: PreflightOptions {
            enabled: true,
            calibration_reuse: true,
            early_family: true,
            plan_first: true,
            plan_buckets: true,
            ..Default::default()
        },
        ..Default::default()
    };
    for tier in [1, 0] {
        let input = CandidateParams {
            speed_tier: tier,
            ..input()
        };
        let result = run_pixel_search(
            &image(2048, 1024),
            &input,
            &options,
            |p, _, _| Ok(output(p, if p.bucket_matcher { 40 } else { 100 })),
            |p, _| {
                Ok(
                    if p.predictor == Some(Predictor::Variable) && p.palette_colors == 0 {
                        10
                    } else {
                        40
                    },
                )
            },
        )
        .unwrap();
        assert!(route(&result, "plan-first"));
        assert!(result.params.bucket_matcher);
        let mut chain = result.params.clone();
        chain.bucket_matcher = false;
        let replay: Vec<_> = result
            .report
            .trials
            .iter()
            .filter(|t| t.params == chain)
            .collect();
        if tier == 1 {
            assert!(
                replay.is_empty(),
                "chain leader must be deferred beyond e9: {replay:?}"
            );
        } else {
            assert_eq!(replay.len(), 1);
            assert_eq!(replay[0].stage, 10);
        }
    }
}
