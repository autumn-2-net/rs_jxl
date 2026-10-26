use super::tests::{assert_retained, complete, family, fixture};
use super::*;

fn enabled_options() -> (Image, DriverOptions) {
    let (image, mut options) = fixture();
    options.scout_parallel_finalists = true;
    options.scout_initial_wave = true;
    options.workers.tasks = 16;
    (image, options)
}

#[test]
fn initial_pilot_wave_updates_scores_bounds_and_preserves_e10_replay() {
    for effort in [9, 10] {
        let (image, mut options) = enabled_options();
        let full_count = if effort == 9 { 5 } else { 7 };
        options.full_limits.max_full_encodes = full_count;
        options.full_limits.max_full_pixels = full_count as u64 * (1 << 19);
        let mut waves = Vec::new();
        let mut replay = Vec::new();
        let result = run_pixel_search_with_batches(
            &image,
            &CandidateParams::default().backend(effort).unwrap(),
            &options,
            |params, _, budget| {
                assert_eq!(params.speed_tier, 0);
                replay.push((family(params), budget.beat_bytes));
                Ok(complete(params, if family(params) == 3 { 60 } else { 75 }))
            },
            |params, _| Ok(100 + family(params)),
            BatchEncoders {
                full: Some(Box::new(|_, tasks| {
                    let initial = waves.is_empty();
                    assert_eq!(tasks.len(), if initial { 3 } else { 2 });
                    assert!(tasks.iter().all(|task| task.cooperative
                        && task.budget.beat_bytes == if initial { None } else { Some(80) }));
                    waves.push(
                        tasks
                            .iter()
                            .map(|task| family(task.params))
                            .collect::<Vec<_>>(),
                    );
                    Ok(tasks
                        .iter()
                        .map(|task| {
                            complete(task.params, [120, 90, 80, 70, 75][family(task.params)])
                        })
                        .collect())
                })),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(waves, [vec![0, 1, 2], vec![3, 4]]);
        assert_eq!(result.report.full_encodes, full_count);
        assert_eq!(result.report.full_pixels, full_count as u64 * (1 << 19));
        assert_eq!(
            result
                .report
                .trials
                .iter()
                .filter(|trial| trial.baseline)
                .count(),
            1
        );
        assert!(result.report.trials[0].baseline);
        if effort == 9 {
            assert!(replay.is_empty());
            assert_retained(&result, 3, 1, 70);
        } else {
            assert_eq!(replay, [(3, Some(70)), (0, Some(60))]);
            assert_retained(&result, 3, 0, 60);
        }
    }
}

#[test]
fn initial_wave_stops_at_reject_then_reconsiders_it_after_palette_witness() {
    let (image, mut options) = enabled_options();
    options.coverage_scan = true;
    options.full_limits.max_full_encodes = 6;
    options.full_limits.max_full_pixels = 6 * (1 << 19);
    let mut waves = Vec::new();
    let result =
        run_pixel_search_with_batches(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |_, _, _| panic!("all selected candidates should use the supplied batch"),
            |params, _| {
                Ok(if params.predictor == Some(Predictor::Zero) {
                    if params.learning_fraction == 0.0 {
                        100
                    } else {
                        200
                    }
                } else {
                    201
                })
            },
            BatchEncoders {
                full: Some(Box::new(|_, tasks| {
                    let initial = waves.is_empty();
                    assert_eq!(tasks.len(), 2);
                    assert!(tasks.iter().all(
                        |task| task.budget.beat_bytes == if initial { None } else { Some(98) }
                    ));
                    if initial {
                        assert!(
                            tasks
                                .iter()
                                .all(|task| task.params.predictor == Some(Predictor::Zero))
                        );
                        assert_eq!(tasks[0].params.learning_fraction, 0.0);
                        assert_eq!(tasks[1].params.learning_fraction, 0.01);
                    }
                    waves.push(
                        tasks
                            .iter()
                            .map(|task| family(task.params))
                            .collect::<Vec<_>>(),
                    );
                    Ok(tasks
                        .iter()
                        .map(|task| {
                            complete(
                                task.params,
                                match task.params.predictor {
                                    Some(Predictor::Zero)
                                        if task.params.learning_fraction == 0.0 =>
                                    {
                                        99
                                    }
                                    Some(Predictor::Zero) => 98,
                                    Some(Predictor::Variable) if task.params.rct == Some(6) => 80,
                                    Some(Predictor::Variable) => 81,
                                    _ => 100,
                                },
                            )
                        })
                        .collect())
                })),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(waves, [vec![2, 2], vec![0, 1], vec![3, 4]]);
    assert_eq!(result.report.full_encodes, 6);
    assert_retained(&result, 4, 1, 80);
    assert!(result.report.events.iter().any(|event| matches!(
        event,
        SearchEvent::Route {
            name: "adaptive-scout-initial-wave",
            candidates: 2
        }
    )));
}

#[test]
fn initial_wave_does_not_skip_a_rejected_front_to_fill_its_slots() {
    let (image, options) = enabled_options();
    let mut probe_counts = [0; 5];
    let mut serial = Vec::new();
    let mut waves = Vec::new();
    let result = run_pixel_search_with_batches(
        &image,
        &CandidateParams::default().backend(9).unwrap(),
        &options,
        |params, _, budget| {
            serial.push((family(params), budget.beat_bytes));
            Ok(complete(params, 100))
        },
        |params, _| {
            let id = family(params);
            let region = probe_counts[id] % 3;
            probe_counts[id] += 1;
            Ok(match id {
                0 => 100,
                1 => 104,
                2 => [90, 120, 120][region],
                _ => 500,
            })
        },
        BatchEncoders {
            full: Some(Box::new(|_, tasks| {
                assert_eq!(tasks.len(), 1);
                assert_eq!(tasks[0].budget.beat_bytes, Some(100));
                waves.push(family(tasks[0].params));
                Ok(vec![complete(tasks[0].params, 80)])
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(serial, [(0, None)]);
    assert_eq!(waves, [2]);
    assert_retained(&result, 2, 1, 80);
    assert!(!result.report.events.iter().any(|event| matches!(
        event,
        SearchEvent::Route {
            name: "adaptive-scout-initial-wave",
            ..
        }
    )));
}

#[test]
fn initial_wave_without_batch_keeps_serial_bounds_and_outputs() {
    for enabled in [false, true] {
        let (image, mut options) = enabled_options();
        options.scout_initial_wave = enabled;
        let mut calls = Vec::new();
        let result = run_pixel_search(
            &image,
            &CandidateParams::default().backend(9).unwrap(),
            &options,
            |params, _, budget| {
                calls.push((family(params), budget.beat_bytes));
                Ok(complete(params, 100 - family(params) * 10))
            },
            |_, _| Ok(100),
        )
        .unwrap();
        assert_eq!(
            calls,
            [
                (0, None),
                (1, Some(100)),
                (2, Some(90)),
                (3, Some(80)),
                (4, Some(70))
            ]
        );
        assert_retained(&result, 4, 1, 60);
    }
}
