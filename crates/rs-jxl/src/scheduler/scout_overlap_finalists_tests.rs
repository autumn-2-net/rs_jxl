use super::tests::{assert_retained, complete, family, fixture};
use super::*;
use crate::scheduler::batch::pilot_handoff::{PilotFront, PilotHandoff, PilotRequest, PilotUsage};

fn options() -> (Image, DriverOptions) {
    let (image, mut options) = fixture();
    options.scout_overlap_automatic = true;
    options.scout_parallel_finalists = true;
    options.scout_initial_wave = true;
    options.workers = WorkerLimits {
        tasks: 16,
        scratch_bytes: 256 << 20,
    };
    (image, options)
}

fn handoff(
    request: &PilotRequest<'_>,
    scores: &[(usize, [usize; 3])],
) -> PilotHandoff<(usize, i8)> {
    let ranked: Vec<_> = scores
        .iter()
        .map(|(id, scores)| {
            let mut candidate = request
                .plan
                .iter()
                .find(|c| family(&c.params) == *id)
                .unwrap()
                .clone();
            candidate.context_bytes = Some(scores.iter().sum());
            (candidate, scores.to_vec())
        })
        .collect();
    let events = vec![SearchEvent::Frontier {
        candidates: ranked.iter().map(|(c, _)| c.clone()).collect(),
    }];
    PilotHandoff {
        automatic: Ok(complete(request.automatic, 100)),
        front: PilotFront {
            ranked: Ok(ranked),
            events,
            usage: PilotUsage::default(),
        },
    }
}

fn assert_jobs(result: &SearchResult<(usize, i8)>, jobs: usize) {
    assert_eq!(result.report.full_encodes, jobs);
    assert_eq!(result.report.trials.len(), jobs);
    assert_eq!(result.report.full_pixels, jobs as u64 * 1024 * 512);
    assert_eq!(
        result.report.trials.iter().filter(|t| t.baseline).count(),
        1
    );
    assert!(result.report.events.iter().any(|event| matches!(event,
        SearchEvent::Route { name: "adaptive-scout-pilot-first", candidates } if *candidates == jobs
    )));
}

#[test]
fn losing_overlap_front_keeps_original_reference_and_skips_cached_auto_slots() {
    let (image, options) = options();
    let input = CandidateParams::default().backend(9).unwrap();
    let mut waves = Vec::new();
    let result = run_pixel_search_with_batches(
        &image,
        &input,
        &options,
        |_, _, _| panic!("completed auto permits bounded batch dispatch"),
        |_, _| panic!("owned front must be reused"),
        BatchEncoders {
            pilot_overlap: Some(Box::new(|request| {
                Ok(Some(handoff(
                    request,
                    &[
                        (4, [100; 3]),
                        (0, [100, 100, 101]),
                        (1, [101; 3]),
                        (0, [103; 3]),
                        (3, [104; 3]),
                        (2, [1000; 3]),
                    ],
                )))
            })),
            full: Some(Box::new(|_, tasks| {
                let ids: Vec<_> = tasks.iter().map(|t| family(t.params)).collect();
                assert_eq!(ids, [4, 1]);
                assert!(
                    tasks
                        .iter()
                        .all(|t| t.cooperative && t.budget.beat_bytes == Some(100))
                );
                waves.push(ids);
                Ok(vec![
                    complete(tasks[0].params, 110),
                    TrialOutput::Bounded { lower_bound: 100 },
                ])
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(waves, [vec![4, 1]]);
    assert_retained(&result, 0, 1, 100);
    assert_jobs(&result, 3);
    assert_eq!(result.report.trials[2].lower_bound, Some(100));
}

#[test]
fn secondary_winner_reranks_pending_rejects_after_each_fitting_wave() {
    for memory in [256 << 20, 80 << 20] {
        let (image, mut options) = options();
        options.workers.scratch_bytes = memory;
        let input = CandidateParams::default().backend(9).unwrap();
        let mut waves = Vec::new();
        let result = run_pixel_search_with_batches(
            &image,
            &input,
            &options,
            |_, _, _| panic!("batch available"),
            |_, _| panic!("owned front must be reused"),
            BatchEncoders {
                pilot_overlap: Some(Box::new(|request| {
                    Ok(Some(handoff(
                        request,
                        &[
                            (4, [100; 3]),
                            (1, [102; 3]),
                            (0, [103; 3]),
                            (3, [104; 3]),
                            (2, [105; 3]),
                        ],
                    )))
                })),
                full: Some(Box::new(|_, tasks| {
                    let ids: Vec<_> = tasks.iter().map(|t| family(t.params)).collect();
                    let bound = match ids[0] {
                        4 | 1 => 100,
                        3 => 80,
                        2 => 70,
                        _ => unreachable!(),
                    };
                    assert!(
                        tasks
                            .iter()
                            .all(|t| t.cooperative && t.budget.beat_bytes == Some(bound))
                    );
                    assert!(tasks.len() <= if memory == 80 << 20 { 1 } else { 2 });
                    waves.push(ids);
                    Ok(tasks
                        .iter()
                        .map(|t| complete(t.params, [100, 80, 60, 70, 110][family(t.params)]))
                        .collect())
                })),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            waves,
            if memory == 80 << 20 {
                vec![vec![4], vec![1], vec![3], vec![2]]
            } else {
                vec![vec![4, 1], vec![3], vec![2]]
            }
        );
        assert_retained(&result, 2, 1, 60);
        assert_jobs(&result, 5);
        let frontier = result
            .report
            .events
            .iter()
            .find_map(|event| match event {
                SearchEvent::Frontier { candidates } => Some(candidates),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            frontier
                .iter()
                .map(|c| (family(&c.params), c.context_bytes))
                .collect::<Vec<_>>(),
            [
                (4, Some(300)),
                (1, Some(306)),
                (0, Some(309)),
                (3, Some(312)),
                (2, Some(315))
            ]
        );
    }
}

#[test]
fn overlap_finalists_precheck_work_and_validate_real_dispatch_bounds() {
    for failure in ["encodes", "pixels", "callback", "bound"] {
        let (image, mut options) = options();
        if failure == "encodes" {
            options.full_limits.max_full_encodes = 2;
        }
        if failure == "pixels" {
            options.full_limits.max_full_pixels = 2 * 1024 * 512;
        }
        let input = CandidateParams::default().backend(9).unwrap();
        let mut dispatched = 0;
        let error = run_pixel_search_with_batches(
            &image,
            &input,
            &options,
            |_, _, _| panic!("batch available"),
            |_, _| panic!("owned front must be reused"),
            BatchEncoders {
                pilot_overlap: Some(Box::new(|request| {
                    Ok(Some(handoff(
                        request,
                        &[(4, [100; 3]), (1, [101; 3]), (0, [103; 3])],
                    )))
                })),
                full: Some(Box::new(|_, tasks| {
                    dispatched += tasks.len();
                    assert!(tasks.iter().all(|t| t.budget.beat_bytes == Some(100)));
                    if failure == "callback" {
                        return Err(Error::Encoding("failed wave".into()));
                    }
                    Ok(vec![
                        complete(tasks[0].params, 80),
                        TrialOutput::Bounded { lower_bound: 90 },
                    ])
                })),
                ..Default::default()
            },
        )
        .unwrap_err();
        if matches!(failure, "encodes" | "pixels") {
            assert!(matches!(error, Error::Limit(_)));
            assert_eq!(dispatched, 0);
        } else {
            assert_eq!(dispatched, 2);
            assert!(error.to_string().contains(if failure == "bound" {
                "dispatch incumbent"
            } else {
                "failed wave"
            }));
        }
    }
}

#[test]
fn weak_overlap_integer_margins_preserve_strong_and_multiregional_winners() {
    assert!(strong_overlap_challenger(&[9800; 3], &[10000; 3]));
    assert!(!strong_overlap_challenger(&[9800, 9800, 9801], &[10000; 3]));
    assert!(strong_overlap_challenger(&[9500, 9500, 12000], &[10000; 3]));
    assert!(!strong_overlap_challenger(
        &[9500, 9501, 12000],
        &[10000; 3]
    ));
    assert!(strong_overlap_challenger(
        &[950, 950, 9000],
        &[1000, 1000, 8000]
    ));
    assert!(!strong_overlap_challenger(
        &[900, 900, 9000],
        &[999, 999, 8002]
    ));
    assert!(!strong_overlap_challenger(
        &[10554, 27655, 29287],
        &[14003, 26532, 27772]
    ));
    assert!(!strong_overlap_challenger(
        &[12317, 27006, 28636],
        &[14003, 26532, 27772]
    ));
    assert!(!strong_overlap_challenger(
        &[usize::MAX; 3],
        &[usize::MAX; 3]
    ));
    assert!(strong_overlap_challenger(
        &[usize::MAX / 2; 3],
        &[usize::MAX; 3]
    ));
}

#[test]
fn weak_overlap_retains_one_two_missing_or_invalid_regions() {
    for (candidate, automatic) in [
        (vec![], vec![]),
        (vec![200], vec![100]),
        (vec![200; 2], vec![100; 2]),
        (vec![200; 3], vec![100; 2]),
        (vec![0, 200, 200], vec![100; 3]),
        (vec![200; 3], vec![0, 100, 100]),
    ] {
        assert!(strong_overlap_challenger(&candidate, &automatic));
    }
}

#[test]
fn weak_overlap_keeps_auto_and_both_zero_tools_without_mutating_scores() {
    let automatic = CandidateParams::default().backend(9).unwrap();
    let mut ranked = vec![
        (Candidate::new("automatic", automatic.clone()), vec![100; 3]),
        (
            Candidate::new("neighborhood", winner_neighborhood(&automatic, 3)),
            vec![101; 3],
        ),
        (
            Candidate::new("literal", literal(&automatic, 9, 0.01).unwrap()),
            vec![1000; 3],
        ),
        (
            Candidate::new("literal-static", literal(&automatic, 9, 0.0).unwrap()),
            vec![1000; 3],
        ),
    ];
    for (c, scores) in &mut ranked {
        c.context_bytes = Some(scores.iter().sum());
    }
    let before = ranked.clone();
    assert_eq!(prune_weak_overlap_challengers(&mut ranked, &automatic), 1);
    for (actual, expected) in ranked.iter().zip([&before[0], &before[2], &before[3]]) {
        assert_eq!(actual.0.params, expected.0.params);
        assert_eq!(actual.0.context_bytes, expected.0.context_bytes);
        assert_eq!(actual.1, expected.1);
    }
    let mut missing_auto = before[1..].to_vec();
    assert_eq!(
        prune_weak_overlap_challengers(&mut missing_auto, &automatic),
        0
    );
    assert_eq!(missing_auto.len(), 3);
}

#[test]
fn weak_overlap_screen_requires_completed_handoff_flag_e9_and_scored_auto() {
    for (enabled, effort, scored_auto) in [
        (true, 9, true),
        (false, 9, true),
        (true, 10, true),
        (true, 9, false),
    ] {
        let (image, mut options) = options();
        options.scout_near_tie_consensus = enabled;
        let input = CandidateParams::default().backend(effort).unwrap();
        let mut waves = Vec::new();
        let result = run_pixel_search_with_batches(
            &image,
            &input,
            &options,
            |p, _, budget| {
                assert_eq!(p.speed_tier, 0);
                assert_eq!(budget.beat_bytes, Some(100));
                Ok(complete(p, 110))
            },
            |_, _| panic!("owned front must be reused"),
            BatchEncoders {
                pilot_overlap: Some(Box::new(|request| {
                    let scores = [
                        (4, [10554, 27655, 29287]),
                        (1, [12317, 27006, 28636]),
                        (0, [14003, 26532, 27772]),
                    ];
                    Ok(Some(handoff(
                        request,
                        &scores[..if scored_auto { 3 } else { 2 }],
                    )))
                })),
                full: Some(Box::new(|_, tasks| {
                    waves.push(tasks.iter().map(|t| family(t.params)).collect::<Vec<_>>());
                    assert!(tasks.iter().all(|t| t.budget.beat_bytes == Some(100)));
                    Ok(tasks.iter().map(|t| complete(t.params, 110)).collect())
                })),
                ..Default::default()
            },
        )
        .unwrap();
        let filtered = enabled && effort == 9 && scored_auto;
        assert_eq!(waves, if filtered { vec![] } else { vec![vec![4, 1]] });
        assert_retained(&result, 0, 1, 100);
        assert_jobs(
            &result,
            if filtered {
                1
            } else if effort == 10 {
                4
            } else {
                3
            },
        );
        assert_eq!(
            result.report.events.iter().any(|event| matches!(
                event,
                SearchEvent::Route {
                    name: "adaptive-scout-weak-overlap-challengers",
                    candidates: 2
                }
            )),
            filtered
        );
        let frontier = result
            .report
            .events
            .iter()
            .find_map(|event| match event {
                SearchEvent::Frontier { candidates } => Some(candidates),
                _ => None,
            })
            .unwrap();
        assert_eq!(frontier.len(), if scored_auto { 3 } else { 2 });
        assert_eq!(frontier[0].context_bytes, Some(67496));
    }
}

#[test]
fn declined_overlap_does_not_apply_weak_screen_to_serial_front() {
    let (image, mut options) = options();
    options.scout_near_tie_consensus = true;
    options.scout_initial_wave = false;
    let input = CandidateParams::default().backend(9).unwrap();
    let mut calls = Vec::new();
    let result = run_pixel_search_with_batches(
        &image,
        &input,
        &options,
        |p, _, budget| {
            if calls.is_empty() {
                assert_eq!(family(p), 4);
                assert_eq!(budget.beat_bytes, None);
            }
            calls.push(family(p));
            Ok(complete(p, 100))
        },
        |p, _| Ok([101, 101, 1000, 1000, 100][family(p)]),
        BatchEncoders {
            pilot_overlap: Some(Box::new(|_| Ok(None))),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(calls, [4, 0, 1]);
    assert_jobs(&result, 3);
    assert!(!result.report.events.iter().any(|event| matches!(
        event,
        SearchEvent::Route {
            name: "adaptive-scout-weak-overlap-challengers",
            ..
        }
    )));
}
