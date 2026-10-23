use super::{
    batch::{BatchEncoders, initial_concurrency, run_ordered},
    driver::{DriverOptions, run_pixel_search, run_pixel_search_with_batches},
    params::{CandidateParams, PlanOptions, lossless_probe_plan},
    probe::{CandidateScreen, ProbeLimits, ScreenOptions},
    search::{EncodedOutput, SearchLimits, TrialOutput},
    small_probe::{
        confirm_lossless_candidate, confirm_lossless_candidate_batched, probe_lossless_candidates,
        probe_lossless_candidates_batched,
    },
};
use crate::{Error, Image, PixelData, PixelLayout};
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

fn image(w: u32, h: u32) -> Image {
    Image::new(
        w,
        h,
        PixelLayout::Gray,
        PixelData::U8(vec![1; w as usize * h as usize]),
    )
    .unwrap()
}

fn size(params: &CandidateParams, image: &Image) -> usize {
    100 + usize::try_from(params.speed_tier).unwrap()
        + params.palette_colors as usize / 1000
        + image.width() as usize
        + image.height() as usize
}

#[test]
fn ordered_dispatch_visits_each_task_once_in_the_supplied_pool() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for slots in [1, 2, 8, 64] {
        let visits: Vec<_> = (0..37).map(|_| AtomicUsize::new(0)).collect();
        let values = run_ordered(&pool, visits.len(), slots, |i| {
            assert_eq!(rayon::current_num_threads(), 3);
            assert_eq!(visits[i].fetch_add(1, Ordering::Relaxed), 0);
            std::thread::yield_now();
            Ok(i * 7)
        })
        .unwrap();
        assert_eq!(values, (0..37).map(|i| i * 7).collect::<Vec<_>>());
        assert!(visits.iter().all(|n| n.load(Ordering::Relaxed) == 1));
    }
    assert!(run_ordered(&pool, 1, 0, |_| Ok(1)).is_err());
    assert!(run_ordered(&pool, 1, 65, |_| Ok(1)).is_err());
    assert!(run_ordered(&pool, 0, 8, |_| Ok(1)).unwrap().is_empty());
}

#[test]
fn ordered_dispatch_finishes_workers_before_returning_first_input_error() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let completed = AtomicUsize::new(0);
    let result = run_ordered(&pool, 12, 8, |i| {
        completed.fetch_add(1, Ordering::Relaxed);
        if i == 2 || i == 8 {
            Err(Error::Encoding(format!("job {i}")))
        } else {
            Ok(i)
        }
    });
    assert!(matches!(result, Err(Error::Encoding(message)) if message == "job 2"));
    assert_eq!(completed.load(Ordering::Relaxed), 12);
}

#[test]
fn source_initial_concurrency_retains_size_and_extra_plane_scratch_rules() {
    use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
    assert_eq!(initial_concurrency(&image(512, 512)), 4);
    assert_eq!(initial_concurrency(&image(513, 512)), 1);
    let original = image(512, 512);
    let mut metadata = original.metadata().clone();
    for _ in 0..4 {
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap());
    }
    let extras = (0..4).map(|_| PixelData::U8(vec![0; 512 * 512])).collect();
    assert_eq!(
        initial_concurrency(&original.with_metadata(metadata, extras).unwrap()),
        3
    );
}

#[test]
fn batch_screen_preserves_ranking_cache_and_attempt_counts() {
    let image = image(1024, 512);
    let params = CandidateParams::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for fast in [false, true] {
        let mut serial = CandidateScreen::new(&image, None, ScreenOptions::default(), |p, i| {
            Ok(size(p, i))
        })
        .unwrap();
        let mut batch = CandidateScreen::new(&image, None, ScreenOptions::default(), |_, _| {
            panic!("serial fallback")
        })
        .unwrap();
        batch.set_batch_encoder(Box::new(|tasks| {
            run_ordered(&pool, tasks.len(), 8, |i| {
                Ok(size(tasks[i].params, tasks[i].image))
            })
        }));
        let plan = lossless_probe_plan(&params, PlanOptions::default());
        let expected = serial.rank(plan.clone(), &params, fast).unwrap();
        let actual = batch.rank(plan, &params, fast).unwrap();
        assert_eq!(expected.len(), actual.len());
        for (a, b) in expected.iter().zip(&actual) {
            assert_eq!(a.params, b.params);
            assert_eq!(a.context_bytes, b.context_bytes);
        }
        assert_eq!(batch.encodes(), serial.encodes());
        assert_eq!(batch.sample_pixels(), serial.sample_pixels());
        let before = batch.encodes();
        assert_eq!(
            batch.score(&params).unwrap(),
            serial.score(&params).unwrap()
        );
        assert_eq!(batch.encodes(), before);
    }
}

#[test]
fn failed_probe_batch_commits_nothing_and_work_limits_precede_dispatch() {
    let image = image(1024, 512);
    let params = CandidateParams::default();
    for malformed in [false, true] {
        let mut screen = CandidateScreen::new(&image, None, ScreenOptions::default(), |_, _| {
            panic!("serial fallback")
        })
        .unwrap();
        screen.set_batch_encoder(Box::new(move |tasks| {
            if malformed {
                Ok(vec![0; tasks.len()])
            } else {
                Err(Error::Encoding("batch failed".into()))
            }
        }));
        assert!(screen.score(&params).is_err());
        let attempts = screen.encodes();
        assert!(attempts > 0);
        screen.set_batch_encoder(Box::new(move |tasks| {
            assert_eq!(tasks.len(), attempts);
            Ok(vec![25; tasks.len()])
        }));
        assert_eq!(screen.score(&params).unwrap(), vec![25; attempts]);
        assert_eq!(screen.encodes(), attempts * 2);
    }
    let mut screen = CandidateScreen::new(
        &image,
        None,
        ScreenOptions {
            limits: ProbeLimits {
                max_encodes: 0,
                ..Default::default()
            },
            ..Default::default()
        },
        |_, _| panic!("serial fallback"),
    )
    .unwrap();
    screen.set_batch_encoder(Box::new(|_| panic!("budget exceeded before dispatch")));
    assert!(matches!(screen.score(&params), Err(Error::Limit(_))));
    assert_eq!(screen.encodes(), 0);
}

fn full(params: &CandidateParams) -> TrialOutput<CandidateParams> {
    TrialOutput::Complete(EncodedOutput {
        bytes: vec![u8::try_from(params.speed_tier).unwrap(); 32],
        artifact: params.clone(),
    })
}

#[test]
fn initial_batches_commit_plan_order_and_retain_the_exact_winner() {
    let image = image(8, 8);
    let params = CandidateParams {
        speed_tier: 8,
        ..Default::default()
    };
    let options = DriverOptions::default();
    let serial = run_pixel_search(
        &image,
        &params,
        &options,
        |p, _, _| Ok(full(p)),
        |_, _| panic!("tiny image probe"),
    )
    .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let batches = Cell::new(0);
    let actual = run_pixel_search_with_batches(
        &image,
        &params,
        &options,
        |p, _, _| Ok(full(p)),
        |_, _| panic!("tiny image probe"),
        BatchEncoders {
            full: Some(Box::new(|_, tasks| {
                batches.set(batches.get() + 1);
                assert_eq!(tasks.len(), 4);
                assert!(tasks.iter().all(|t| t.budget.beat_bytes.is_none()));
                run_ordered(&pool, tasks.len(), 4, |i| Ok(full(tasks[i].params)))
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(batches.get(), 2);
    assert_eq!(actual.params, serial.params);
    assert_eq!(actual.bytes, serial.bytes);
    assert_eq!(actual.artifact, serial.artifact);
    assert_eq!(actual.report.full_encodes, serial.report.full_encodes);
    assert_eq!(actual.report.full_pixels, serial.report.full_pixels);
    assert_eq!(actual.report.trials.len(), serial.report.trials.len());
    for (a, b) in actual.report.trials.iter().zip(serial.report.trials) {
        assert_eq!((a.stage, &a.params, a.bytes), (b.stage, &b.params, b.bytes));
    }
}

#[test]
fn initial_batch_budget_failure_never_calls_backend() {
    let options = DriverOptions {
        full_limits: SearchLimits {
            max_full_encodes: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    let result: crate::Result<_> = run_pixel_search_with_batches(
        &image(8, 8),
        &CandidateParams {
            speed_tier: 8,
            ..Default::default()
        },
        &options,
        |_, _, _| -> crate::Result<TrialOutput<()>> { panic!("single callback") },
        |_, _| panic!("probe callback"),
        BatchEncoders {
            full: Some(Box::new(|_, _| panic!("batch callback"))),
            ..Default::default()
        },
    );
    assert!(matches!(result, Err(Error::Limit(_))));
}

#[test]
fn small_probe_batches_preserve_finalists_and_exploratory_scores() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let input = CandidateParams::default();
    for (w, h) in [(256, 256), (1024, 768)] {
        let image = image(w, h);
        for extended_rct in [false, true] {
            let options = PlanOptions { extended_rct };
            let plan = lossless_probe_plan(&input, options);
            for seed in 0..12 {
                let score = |p: &CandidateParams, image: &Image| {
                    let index = plan.iter().position(|c| &c.params == p);
                    let score = match index {
                        Some(i) if seed == 0 => {
                            if i < 6 {
                                100
                            } else {
                                10000
                            }
                        }
                        Some(i) => 1 + (i * 1297 + seed * 71) % 250,
                        None => 1000,
                    };
                    score * image.width() as usize
                };
                // Compare to exhaustive scoring, not merely another pruned path.
                let serial = probe_lossless_candidates(
                    &image,
                    &input,
                    Some(&input),
                    options,
                    false,
                    ProbeLimits::default(),
                    |p, i| Ok(score(p, i)),
                )
                .unwrap();
                for feedback in [false, true] {
                    let calls = Cell::new(0);
                    let actual = probe_lossless_candidates_batched(
                        &image,
                        &input,
                        Some(&input),
                        options,
                        feedback,
                        ProbeLimits::default(),
                        |tasks| {
                            calls.set(calls.get() + 1);
                            run_ordered(&pool, tasks.len(), 8, |i| {
                                Ok(score(tasks[i].params, tasks[i].image))
                            })
                        },
                    )
                    .unwrap();
                    assert!(calls.get() > 1);
                    assert_eq!(actual.ranked.len(), serial.ranked.len());
                    for (a, b) in actual.ranked.iter().zip(&serial.ranked) {
                        assert_eq!(a.params, b.params);
                        assert_eq!((a.bytes, a.small_bytes), (b.bytes, b.small_bytes));
                    }
                    for (a, b) in actual.exploratory.iter().zip(&serial.exploratory) {
                        assert_eq!(a.params, b.params);
                        assert_eq!(a.bytes, b.bytes);
                    }
                    assert_eq!(actual.exploratory.len(), serial.exploratory.len());
                    assert_eq!(actual.control_bytes, serial.control_bytes);
                    assert_eq!(actual.validation_encodes, serial.validation_encodes);
                    if seed == 0 && feedback {
                        assert!(actual.screen_saved_encodes > 0);
                    }
                    if !feedback {
                        assert_eq!(actual.encodes, serial.encodes);
                        assert_eq!(actual.sample_pixels, serial.sample_pixels);
                    }
                }
            }
        }
    }
}

#[test]
fn confirmation_batches_reuse_only_the_matching_witness_total() {
    let image = image(1024, 768);
    let incumbent = CandidateParams::default();
    let candidate = incumbent.backend(8).unwrap();
    for known in [None, Some(0), Some(2345)] {
        let serial = confirm_lossless_candidate(
            &image,
            &candidate,
            &incumbent,
            known,
            ProbeLimits::default(),
            |p, i| Ok(size(p, i)),
        )
        .unwrap();
        let calls = Cell::new(0);
        let result = confirm_lossless_candidate_batched(
            &image,
            &candidate,
            &incumbent,
            known,
            ProbeLimits::default(),
            |tasks| {
                assert!(tasks.len() <= 4);
                calls.set(calls.get() + tasks.len());
                Ok(tasks.iter().map(|t| size(t.params, t.image)).collect())
            },
        )
        .unwrap();
        assert_eq!(result.samples, serial.samples);
        assert_eq!(
            (
                result.candidate_bytes,
                result.incumbent_bytes,
                result.accepted
            ),
            (
                serial.candidate_bytes,
                serial.incumbent_bytes,
                serial.accepted
            )
        );
        assert_eq!(result.encodes, serial.encodes);
        assert_eq!(result.encodes, calls.get());
        assert_eq!(
            result.encodes,
            if known.is_some_and(|n| n != 0) { 3 } else { 6 }
        );
    }
}

#[test]
fn small_probe_batches_reject_bad_results_and_bound_work_before_dispatch() {
    let image = image(1024, 768);
    let params = CandidateParams::default();
    for kind in 0..4 {
        let result = probe_lossless_candidates_batched(
            &image,
            &params,
            None,
            PlanOptions::default(),
            false,
            ProbeLimits::default(),
            |tasks| match kind {
                0 => Ok(vec![0; tasks.len()]),
                1 => Ok(vec![1; tasks.len() - 1]),
                2 => Ok(vec![usize::MAX; tasks.len()]),
                _ => Err(Error::Encoding("failed task".into())),
            },
        );
        assert!(result.is_err());
    }
    for limits in [
        ProbeLimits {
            max_encodes: 0,
            ..Default::default()
        },
        ProbeLimits {
            max_encoded_pixels: 0,
            ..Default::default()
        },
        ProbeLimits {
            max_resident_sample_pixels: 0,
            ..Default::default()
        },
        ProbeLimits {
            max_candidates: 0,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            probe_lossless_candidates_batched(
                &image,
                &params,
                None,
                PlanOptions::default(),
                false,
                limits,
                |_| panic!("budget must precede callback")
            ),
            Err(Error::Limit(_))
        ));
    }
    assert!(matches!(
        confirm_lossless_candidate_batched(
            &image,
            &params,
            &params,
            None,
            ProbeLimits {
                max_resident_sample_pixels: 512 * 512,
                ..Default::default()
            },
            |_| panic!("all resident witnesses must be counted")
        ),
        Err(Error::Limit(_))
    ));
}

#[test]
fn default_driver_routes_legacy_probes_through_shared_batch_budget() {
    let image = image(1024, 512);
    let params = CandidateParams::default();
    let options = DriverOptions {
        confirmation: true,
        ..Default::default()
    };
    let cells = Cell::new(0);
    let small_cells = Cell::new(0);
    let serial = run_pixel_search(
        &image,
        &params,
        &options,
        |p, _, _| Ok(full(p)),
        |p, i| Ok(size(p, i)),
    )
    .unwrap();
    let result = run_pixel_search_with_batches(
        &image,
        &params,
        &options,
        |p, _, _| Ok(full(p)),
        |_, _| panic!("unexpected serial probe"),
        BatchEncoders {
            probe: Some(Box::new(|tasks| {
                cells.set(cells.get() + tasks.len());
                small_cells.set(
                    small_cells.get() + tasks.iter().filter(|t| t.image.width() <= 64).count(),
                );
                Ok(tasks.iter().map(|t| size(t.params, t.image)).collect())
            })),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(small_cells.get() > 0);
    assert_eq!(result.report.probe_encodes, cells.get());
    assert_eq!(result.report.probe_encodes, serial.report.probe_encodes);
    assert_eq!(result.report.probe_pixels, serial.report.probe_pixels);
    assert_eq!(result.params, serial.params);
    assert_eq!(result.report.trials.len(), serial.report.trials.len());
    for (a, b) in result.report.trials.iter().zip(&serial.report.trials) {
        assert_eq!((&a.params, a.stage, a.bytes), (&b.params, b.stage, b.bytes));
    }
    let limits = super::probe::ProbeLimits {
        max_encodes: cells.get() - 1,
        ..Default::default()
    };
    let counted = Cell::new(0);
    let limited = run_pixel_search_with_batches(
        &image,
        &params,
        &DriverOptions {
            screen: ScreenOptions {
                limits,
                ..Default::default()
            },
            ..options
        },
        |p, _, _| Ok(full(p)),
        |_, _| panic!("unexpected serial probe"),
        BatchEncoders {
            probe: Some(Box::new(|tasks| {
                counted.set(counted.get() + tasks.len());
                Ok(tasks.iter().map(|t| size(t.params, t.image)).collect())
            })),
            ..Default::default()
        },
    );
    assert!(matches!(limited, Err(Error::Limit(_))));
    assert!(counted.get() <= limits.max_encodes);
}
