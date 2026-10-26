use super::tests::{assert_retained, complete, family, fixture};
use super::*;
use crate::scheduler::batch::pilot_handoff::{PilotFront, PilotHandoff, PilotUsage, run_front};

#[test]
fn overlap_flag_is_opt_in_requires_pilot_and_missing_executor_falls_back() {
    assert!(!DriverOptions::default().scout_overlap_automatic);
    assert!(!DriverOptions::scout().scout_overlap_automatic);
    let (image, mut options) = fixture();
    let input = CandidateParams::default().backend(9).unwrap();
    options.scout_overlap_automatic = true;
    options.scout_pilot_first = false;
    assert!(
        run_pixel_search::<_, _, ()>(
            &image,
            &input,
            &options,
            |_, _, _| panic!("invalid overlap must not encode"),
            |_, _| panic!("invalid overlap must not probe"),
        )
        .is_err()
    );
    options.scout_pilot_first = true;
    for decline in [false, true] {
        let mut batches = BatchEncoders::default();
        if decline {
            batches.pilot_overlap = Some(Box::new(|_| Ok(None)));
        }
        let result = run_pixel_search_with_batches(
            &image,
            &input,
            &options,
            |p, _, b| {
                assert_eq!(b.beat_bytes, (family(p) != 0).then_some(100));
                Ok(complete(p, 100))
            },
            |_, _| Ok(100),
            batches,
        )
        .unwrap();
        assert_eq!(result.report.full_encodes, 5);
        assert_retained(&result, 0, 1, 100);
    }
}

#[test]
fn automatic_is_encoded_once_and_e10_replay_keeps_real_output_bounds() {
    let (image, mut options) = fixture();
    options.scout_overlap_automatic = true;
    options.scout_initial_wave = true;
    options.scout_parallel_finalists = true;
    for effort in [9, 10] {
        let input = CandidateParams::default().backend(effort).unwrap();
        let mut calls = Vec::new();
        let mut probe_count = 0;
        let mut probe_pixels = 0;
        let result = run_pixel_search_with_batches(
            &image,
            &input,
            &options,
            |p, _, budget| {
                let id = family(p);
                assert!(
                    p.speed_tier != 1 || id != 0,
                    "automatic must not be re-encoded"
                );
                calls.push((id, p.speed_tier, budget.beat_bytes));
                Ok(complete(
                    p,
                    if p.speed_tier == 0 {
                        600
                    } else {
                        [500, 300, 200, 600, 600][id]
                    },
                ))
            },
            |_, _| panic!("front must be reused"),
            BatchEncoders {
                pilot_overlap: Some(Box::new(|request| {
                    assert!(request.budget.beat_bytes.is_none());
                    let front = run_front(
                        request,
                        Box::new(|tasks, _| {
                            probe_count += tasks.len();
                            probe_pixels += tasks
                                .iter()
                                .map(|t| u64::from(t.image.width()) * u64::from(t.image.height()))
                                .sum::<u64>();
                            Ok(tasks
                                .iter()
                                .map(|t| [100, 102, 104, 300, 300][family(t.params)])
                                .collect())
                        }),
                    );
                    Ok(Some(PilotHandoff {
                        automatic: Ok(complete(request.automatic, 500)),
                        front,
                    }))
                })),
                ..Default::default()
            },
        )
        .unwrap();
        assert_retained(&result, 2, 1, 200);
        assert_eq!(result.report.full_encodes, if effort == 9 { 3 } else { 7 });
        assert_eq!(&calls[..2], &[(1, 1, Some(500)), (2, 1, Some(300))]);
        assert_eq!(result.report.probe_encodes, probe_count);
        assert_eq!(result.report.probe_pixels, probe_pixels);
        assert_eq!(result.report.screen_batches, 1);
        assert_eq!(
            result.report.trials.iter().filter(|t| t.baseline).count(),
            1
        );
        if effort == 10 {
            assert_eq!(&calls[4..], &[(2, 0, Some(200)), (0, 0, Some(200))]);
        }
    }
}

#[test]
fn auto_absent_from_rank_is_retained_without_inventing_a_reference_score() {
    let (image, mut options) = fixture();
    options.scout_overlap_automatic = true;
    let input = CandidateParams::default().backend(9).unwrap();
    let mut calls = Vec::new();
    let result = run_pixel_search_with_batches(
        &image,
        &input,
        &options,
        |p, _, budget| {
            let id = family(p);
            calls.push(id);
            assert_eq!(budget.beat_bytes, Some(50));
            Ok(if id == 1 {
                TrialOutput::Bounded { lower_bound: 50 }
            } else {
                complete(p, 40)
            })
        },
        |_, _| panic!("owned front is complete"),
        BatchEncoders {
            pilot_overlap: Some(Box::new(|request| {
                let ranked = request
                    .plan
                    .iter()
                    .filter(|c| matches!(family(&c.params), 1 | 2))
                    .cloned()
                    .map(|mut c| {
                        let score = if family(&c.params) == 1 { 100 } else { 101 };
                        c.context_bytes = Some(score * 3);
                        (c, vec![score; 3])
                    })
                    .collect();
                Ok(Some(PilotHandoff {
                    automatic: Ok(complete(request.automatic, 50)),
                    front: PilotFront {
                        ranked: Ok(ranked),
                        events: Vec::new(),
                        usage: PilotUsage::default(),
                    },
                }))
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(calls, [1, 2]);
    assert_eq!(result.report.full_encodes, 3);
    assert_retained(&result, 2, 1, 40);
}

#[test]
fn overlap_full_budget_is_checked_before_dispatch() {
    let (image, mut options) = fixture();
    options.scout_overlap_automatic = true;
    let input = CandidateParams::default().backend(9).unwrap();
    for limit in 0..3 {
        let mut options = options.clone();
        match limit {
            0 => options.full_limits.max_full_encodes = 0,
            1 => options.full_limits.max_full_pixels = 1,
            _ => options.full_limits.max_output_bytes = 0,
        }
        let result = run_pixel_search_with_batches::<_, _, ()>(
            &image,
            &input,
            &options,
            |_, _, _| panic!("no full after resource failure"),
            |_, _| panic!("no probe after resource failure"),
            BatchEncoders {
                pilot_overlap: Some(Box::new(|_| panic!("no overlap after resource failure"))),
                ..Default::default()
            },
        );
        assert!(matches!(result, Err(Error::Limit(_))));
    }
}
