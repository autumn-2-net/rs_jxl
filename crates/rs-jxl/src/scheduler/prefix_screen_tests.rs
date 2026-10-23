//! Scheduler size oracles, not codec compatibility or byte-identity claims.
use super::*;
use crate::scheduler::{
    batch::{BatchEncoders, PrefixBatchEncoder},
    driver::{DriverOptions, SearchResult, run_pixel_search_with_batches},
    params::{CandidateParams, PlanOptions, lossless_probe_plan},
    probe::{ProbeLimits, ProbeResult, ScreenOptions},
    search::{EncodedOutput, TrialOutput},
    small_probe::{
        probe_lossless_candidates, probe_lossless_candidates_batched,
        probe_lossless_candidates_with_prefix,
    },
};
use crate::{PixelData, PixelLayout};
use std::{
    cell::Cell,
    sync::{Condvar, atomic::AtomicU64},
    time::Duration,
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

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn candidates(count: usize) -> Vec<Candidate> {
    (0..count)
        .map(|i| {
            Candidate::new(
                "oracle",
                CandidateParams {
                    predictor: Some(if i % 7 == 6 {
                        Predictor::Variable
                    } else {
                        Predictor::Zero
                    }),
                    ..Default::default()
                },
            )
        })
        .collect()
}

fn top_six(scores: &[Option<usize>]) -> Vec<usize> {
    let mut order: Vec<_> = (0..scores.len()).collect();
    order.sort_by_key(|&i| scores[i].unwrap_or(usize::MAX));
    order.truncate(6);
    order
}

#[test]
fn prefix_finalists_match_exhaustive_and_sourceoff_across_pool_widths() {
    let candidates = candidates(32);
    let samples = [image(2, 3), image(3, 4), image(5, 7)];
    let order = [2, 0, 1];
    for threads in [1, 2, 4] {
        let pool = pool(threads);
        for slots in [1, 2, 8] {
            for seed in 0..12 {
                let cell = |i: usize, tile: usize| match seed {
                    0 => 1, // Exact ties preserve original plan order.
                    1 if i < 6 => 10,
                    1 => 1000,
                    _ => 1 + (i * 1297 + tile * 631 + seed * 71) % 250,
                };
                let exhaustive: Vec<_> = (0..candidates.len())
                    .map(|i| Some(order.iter().map(|&t| cell(i, t)).sum::<usize>()))
                    .collect();
                for feedback_bounds in [false, true] {
                    let work = PrefixWork::new(usize::MAX, u64::MAX);
                    let screen = PrefixScreen {
                        candidates: &candidates,
                        samples: &samples,
                        order: &order,
                        pass: PrefixPass::Screen { feedback_bounds },
                        work: &work,
                    };
                    let visits: Vec<_> = candidates.iter().map(|_| AtomicUsize::new(0)).collect();
                    let output = run_prefix_screen(Some(&pool), slots, &screen, |i, tile| {
                        assert!(crate::execution::is_serial());
                        assert!(pool.current_thread_index().is_some());
                        let next = visits[i].fetch_add(1, Ordering::Relaxed);
                        assert_eq!(tile, order[next]);
                        std::thread::yield_now();
                        Ok(cell(i, tile))
                    })
                    .unwrap();
                    let scores: Vec<_> = output.scores().iter().map(|s| s.bytes()).collect();
                    assert_eq!(top_six(&scores), top_six(&exhaustive));
                    let mut usage = PrefixUsage::default();
                    for (i, score) in output.scores().iter().enumerate() {
                        let count = visits[i].load(Ordering::Relaxed);
                        assert_eq!(score.encodes(), count);
                        usage.encodes += count;
                        usage.pixels += order[..count]
                            .iter()
                            .map(|&t| {
                                u64::from(samples[t].width()) * u64::from(samples[t].height())
                            })
                            .sum::<u64>();
                        if !feedback_bounds
                            || candidates[i].params.predictor == Some(Predictor::Variable)
                        {
                            assert_eq!(score.bytes(), exhaustive[i]);
                            assert_eq!(count, order.len());
                        }
                        if let Some(bytes) = score.bytes() {
                            assert_eq!(Some(bytes), exhaustive[i]);
                        } else {
                            assert!(count < order.len());
                            assert!(exhaustive[i] > exhaustive[top_six(&exhaustive)[5]]);
                        }
                    }
                    assert_eq!(work.usage(), usage);
                }
            }
        }
    }
}

#[test]
fn live_cutoff_reaches_slow_candidate_without_a_tile_or_wave_barrier() {
    let pool = pool(2);
    let candidates = candidates(9);
    let samples = [image(8, 8), image(8, 8), image(8, 8)];
    let order = [2, 0, 1];
    let work = PrefixWork::new(100, 100 * 64);
    let screen = PrefixScreen {
        candidates: &candidates,
        samples: &samples,
        order: &order,
        pass: PrefixPass::Screen {
            feedback_bounds: true,
        },
        work: &work,
    };
    let released = (Mutex::new(false), Condvar::new());
    let output = run_prefix_screen(Some(&pool), 2, &screen, |i, tile| {
        if i == 0 && tile == 2 {
            let (released, timeout) = released
                .1
                .wait_timeout_while(
                    released.0.lock().unwrap(),
                    Duration::from_secs(10),
                    |done| !*done,
                )
                .unwrap();
            assert!(
                *released && !timeout.timed_out(),
                "candidate queue did not progress"
            );
        }
        if i == 7 && tile == 2 {
            // Candidates 1..=6 have completed while candidate 0's first tile waits.
            *released.0.lock().unwrap() = true;
            released.1.notify_all();
        }
        Ok(if (1..=6).contains(&i) { 1 } else { 1000 })
    })
    .unwrap();
    assert_eq!(output.scores()[0].bytes(), None);
    assert_eq!(output.scores()[0].encodes(), 1);
    assert_eq!(
        top_six(
            &output
                .scores()
                .iter()
                .map(|s| s.bytes())
                .collect::<Vec<_>>()
        ),
        [1, 2, 3, 4, 5, 6]
    );
}

#[test]
fn equality_at_cutoff_completes_and_variable_never_bounds() {
    let mut candidates = candidates(8);
    candidates[6].params.predictor = Some(Predictor::Zero);
    candidates[7].params.predictor = Some(Predictor::Variable);
    let samples = [image(8, 8), image(8, 8), image(8, 8)];
    let work = PrefixWork::new(24, 24 * 64);
    let screen = PrefixScreen {
        candidates: &candidates,
        samples: &samples,
        order: &[0, 1, 2],
        pass: PrefixPass::Screen {
            feedback_bounds: true,
        },
        work: &work,
    };
    let output = run_prefix_screen(None, 8, &screen, |i, tile| {
        Ok(match i {
            6 => [1, 2, 1][tile], // Prefix equals the sixth-best total before last tile.
            7 => 1000,
            _ => 1,
        })
    })
    .unwrap();
    assert_eq!(output.scores()[6].bytes(), Some(4));
    assert_eq!(output.scores()[7].bytes(), Some(3000));
    assert_eq!(work.usage().encodes, 24);
}

#[test]
fn completed_cutoff_is_monotone_and_counts_ties_as_completed_candidates() {
    let completed = Completed::new();
    let mut previous = usize::MAX;
    for (i, bytes) in [20, 10, 10, 40, 30, 20, 200, 9, 1, 1, 1]
        .into_iter()
        .enumerate()
    {
        completed.insert(bytes);
        let next = completed.cutoff.load(Ordering::Relaxed);
        assert!(next <= previous);
        if i < 5 {
            assert_eq!(next, usize::MAX);
        }
        previous = next;
    }
    assert_eq!(*completed.best.lock().unwrap(), [1, 1, 1, 9, 10, 10]);
}

#[test]
fn hard_limits_charge_only_attempts_and_never_overshoot() {
    let candidates = candidates(20);
    let samples = [image(8, 8), image(8, 8), image(8, 8)];
    let pool = pool(4);
    for (max_encodes, max_pixels, expected) in [
        (0, u64::MAX, 0),
        (13, u64::MAX, 13),
        (usize::MAX, 13 * 64 + 63, 13),
        (usize::MAX, 63, 0),
    ] {
        let work = PrefixWork::new(max_encodes, max_pixels);
        let screen = PrefixScreen {
            candidates: &candidates,
            samples: &samples,
            order: &[2, 0, 1],
            pass: PrefixPass::Screen {
                feedback_bounds: false,
            },
            work: &work,
        };
        let attempted = AtomicUsize::new(0);
        let result = run_prefix_screen(Some(&pool), 8, &screen, |_, _| {
            attempted.fetch_add(1, Ordering::Relaxed);
            std::thread::yield_now();
            Ok(1)
        });
        assert!(matches!(
            result,
            Err(Error::Limit("cumulative scheduler probe work"))
        ));
        assert_eq!(attempted.load(Ordering::Relaxed), expected);
        assert_eq!(
            work.usage(),
            PrefixUsage {
                encodes: expected,
                pixels: expected as u64 * 64
            }
        );
    }
}

#[test]
fn candidate_roots_are_bounded_and_outer_null_scope_does_not_enter_pool() {
    let pool = pool(4);
    let candidates = candidates(32);
    let samples = [image(8, 8), image(8, 8), image(8, 8)];
    for slots in [1, 2, 8] {
        let work = PrefixWork::new(96, 96 * 64);
        let screen = PrefixScreen {
            candidates: &candidates,
            samples: &samples,
            order: &[0, 1, 2],
            pass: PrefixPass::Screen {
                feedback_bounds: false,
            },
            work: &work,
        };
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        run_prefix_screen(Some(&pool), slots, &screen, |_, tile| {
            if tile == 0 {
                let live = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(live, Ordering::Relaxed);
                assert!(live <= slots.min(4));
            }
            assert!(crate::execution::is_serial());
            assert_eq!(crate::execution::worker_count(), 1);
            std::thread::yield_now();
            if tile == 2 {
                active.fetch_sub(1, Ordering::SeqCst);
            }
            Ok(1)
        })
        .unwrap();
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert!((1..=slots.min(4)).contains(&peak.load(Ordering::Relaxed)));
    }
    let caller = std::thread::current().id();
    crate::execution::serial(|| {
        let work = PrefixWork::new(96, 96 * 64);
        let screen = PrefixScreen {
            candidates: &candidates,
            samples: &samples,
            order: &[0, 1, 2],
            pass: PrefixPass::Screen {
                feedback_bounds: true,
            },
            work: &work,
        };
        run_prefix_screen(Some(&pool), 8, &screen, |_, _| {
            assert_eq!(std::thread::current().id(), caller);
            assert!(pool.current_thread_index().is_none());
            assert!(crate::execution::is_serial());
            Ok(1)
        })
        .unwrap();
    });
}

#[test]
fn error_cancels_new_cells_preserves_original_error_and_allows_reuse() {
    let pool = pool(4);
    let candidates = candidates(20);
    let samples = [image(8, 8), image(8, 8), image(8, 8)];
    for mode in 0..3 {
        let work = PrefixWork::new(100, 100 * 64);
        let screen = PrefixScreen {
            candidates: &candidates,
            samples: &samples,
            order: &[0, 1, 2],
            pass: PrefixPass::Screen {
                feedback_bounds: false,
            },
            work: &work,
        };
        let attempted = AtomicUsize::new(0);
        let result = run_prefix_screen(Some(&pool), 1, &screen, |i, tile| {
            attempted.fetch_add(1, Ordering::Relaxed);
            assert_eq!(i, 0);
            match mode {
                0 => Err(Error::Limit("backend failure")),
                1 => Ok(0),
                _ => Ok(if tile == 0 { usize::MAX } else { 1 }),
            }
        });
        match mode {
            0 => assert!(matches!(result, Err(Error::Limit("backend failure")))),
            1 => assert!(matches!(result, Err(Error::Encoding(_)))),
            _ => assert!(matches!(result, Err(Error::Limit("probe bytes")))),
        }
        let expected = if mode == 2 { 2 } else { 1 };
        assert_eq!(attempted.load(Ordering::Relaxed), expected);
        assert_eq!(work.usage().encodes, expected);
        let output = run_prefix_screen(Some(&pool), 8, &screen, |_, _| Ok(1)).unwrap();
        assert!(output.scores().iter().all(|s| s.bytes() == Some(3)));
        assert_eq!(work.usage().encodes, expected + 60);
    }
}

#[test]
fn empty_candidates_and_invalid_orders_do_not_encode() {
    let samples = [image(8, 8)];
    let candidates = candidates(1);
    let work = PrefixWork::new(100, 10000);
    let mut screen = PrefixScreen {
        candidates: &[],
        samples: &samples,
        order: &[0],
        pass: PrefixPass::Screen {
            feedback_bounds: true,
        },
        work: &work,
    };
    assert!(
        run_prefix_screen(None, 8, &screen, |_, _| panic!("empty plan"))
            .unwrap()
            .scores()
            .is_empty()
    );
    screen.candidates = &candidates;
    for order in [&[][..], &[0, 0][..], &[1][..]] {
        screen.order = order;
        assert!(matches!(
            run_prefix_screen(None, 8, &screen, |_, _| panic!("invalid order")),
            Err(Error::InvalidInput(_))
        ));
    }
    assert_eq!(work.usage(), PrefixUsage::default());
}

fn same_finalists(actual: &ProbeResult, expected: &ProbeResult) {
    assert_eq!(actual.shortlisted, expected.shortlisted);
    for (a, b) in [
        (&actual.ranked, &expected.ranked),
        (&actual.exploratory, &expected.exploratory),
    ] {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.params, b.params);
            assert_eq!((a.bytes, a.small_bytes), (b.bytes, b.small_bytes));
        }
    }
}

#[test]
fn small_probe_oracle_validation_and_actual_savings_over_chunks8() {
    let input = CandidateParams::default();
    let pool = pool(1);
    for (w, h) in [(256, 256), (1024, 768)] {
        let image = image(w, h);
        for extended_rct in [false, true] {
            let options = PlanOptions { extended_rct };
            let plan = lossless_probe_plan(&input, options);
            let size = |params: &CandidateParams, _: &Image| match plan
                .iter()
                .position(|c| &c.params == params)
            {
                Some(i) if i < 6 => 10,
                Some(_) => 1000,
                None => 100,
            };
            let exhaustive = probe_lossless_candidates(
                &image,
                &input,
                Some(&input),
                options,
                false,
                ProbeLimits::default(),
                |p, i| Ok(size(p, i)),
            )
            .unwrap();
            let batched = probe_lossless_candidates_batched(
                &image,
                &input,
                Some(&input),
                options,
                true,
                ProbeLimits::default(),
                |tasks| Ok(tasks.iter().map(|t| size(t.params, t.image)).collect()),
            )
            .unwrap();
            for feedback in [false, true] {
                let calls = AtomicUsize::new(0);
                let mut prefix: PrefixBatchEncoder<'_> = Box::new(|screen| {
                    run_prefix_screen(Some(&pool), 8, screen, |i, tile| {
                        calls.fetch_add(1, Ordering::Relaxed);
                        Ok(size(&screen.candidates[i].params, &screen.samples[tile]))
                    })
                });
                let actual = probe_lossless_candidates_with_prefix(
                    &image,
                    &input,
                    Some(&input),
                    options,
                    feedback,
                    ProbeLimits::default(),
                    |tasks| {
                        calls.fetch_add(tasks.len(), Ordering::Relaxed);
                        Ok(tasks.iter().map(|t| size(t.params, t.image)).collect())
                    },
                    &mut prefix,
                )
                .unwrap();
                same_finalists(&actual, &exhaustive);
                assert_eq!(actual.encodes, calls.load(Ordering::Relaxed));
                if feedback {
                    assert!(actual.encodes < batched.encodes);
                    assert!(actual.screen_saved_encodes > batched.screen_saved_encodes);
                } else {
                    assert_eq!(actual.encodes, exhaustive.encodes);
                    assert_eq!(actual.screen_saved_encodes, 0);
                }
                // A ceiling fitting actual work succeeds, even when it cannot
                // accommodate the unpruned maximum or old wave implementation.
                let exact = probe_lossless_candidates_with_prefix(
                    &image,
                    &input,
                    Some(&input),
                    options,
                    feedback,
                    ProbeLimits {
                        max_encodes: actual.encodes,
                        max_encoded_pixels: actual.sample_pixels,
                        ..Default::default()
                    },
                    |tasks| Ok(tasks.iter().map(|t| size(t.params, t.image)).collect()),
                    &mut prefix,
                )
                .unwrap();
                same_finalists(&exact, &actual);
            }
        }
    }
}

fn driver_run(
    limits: ProbeLimits,
    batched: bool,
) -> (Result<SearchResult<()>>, PrefixUsage, Option<PrefixUsage>) {
    let image = image(1024, 512);
    let pool = pool(3);
    let encodes = AtomicUsize::new(0);
    let pixels = AtomicU64::new(0);
    let before_prefix = Cell::new(None);
    let size = |p: &CandidateParams, sample: &Image| {
        encodes.fetch_add(1, Ordering::Relaxed);
        pixels.fetch_add(
            u64::from(sample.width()) * u64::from(sample.height()),
            Ordering::Relaxed,
        );
        100 + p.speed_tier as usize
            + p.palette_colors as usize / 1000
            + sample.width() as usize
            + sample.height() as usize
    };
    let mut batches = BatchEncoders::default();
    if batched {
        batches.probe = Some(Box::new(|tasks| {
            Ok(tasks.iter().map(|t| size(t.params, t.image)).collect())
        }));
    }
    batches.prefix = Some(Box::new(|screen| {
        if matches!(screen.pass, PrefixPass::Screen { .. }) {
            before_prefix.set(Some(PrefixUsage {
                encodes: encodes.load(Ordering::Relaxed),
                pixels: pixels.load(Ordering::Relaxed),
            }));
        }
        run_prefix_screen(Some(&pool), 8, screen, |i, t| {
            Ok(size(&screen.candidates[i].params, &screen.samples[t]))
        })
    }));
    let result = run_pixel_search_with_batches(
        &image,
        &CandidateParams::default(),
        &DriverOptions {
            confirmation: true,
            screen: ScreenOptions {
                limits,
                ..Default::default()
            },
            ..Default::default()
        },
        |_, _, _| {
            Ok(TrialOutput::Complete(EncodedOutput {
                bytes: vec![42; 32],
                artifact: (),
            }))
        },
        |p, i| Ok(size(p, i)),
        batches,
    );
    (
        result,
        PrefixUsage {
            encodes: encodes.load(Ordering::Relaxed),
            pixels: pixels.load(Ordering::Relaxed),
        },
        before_prefix.get(),
    )
}

#[test]
fn driver_prefix_shares_cumulative_budget_with_prior_and_later_probe_routes() {
    for batched in [false, true] {
        let (result, usage, before) = driver_run(ProbeLimits::default(), batched);
        let result = result.unwrap();
        let before = before.expect("driver must invoke prefix callback");
        assert!(before.encodes >= 9);
        assert!(usage.encodes > before.encodes + 45);
        assert_eq!(result.report.probe_encodes, usage.encodes);
        assert_eq!(result.report.probe_pixels, usage.pixels);
        assert!(
            !result
                .report
                .unported
                .iter()
                .any(|s| s.contains("within-wave"))
        );
        for limits in [
            ProbeLimits {
                max_encodes: before.encodes + 4,
                ..Default::default()
            },
            ProbeLimits {
                max_encoded_pixels: before.pixels + 4 * 4096,
                ..Default::default()
            },
        ] {
            let (result, attempted, _) = driver_run(limits, batched);
            assert!(matches!(
                result,
                Err(Error::Limit("cumulative scheduler probe work"))
            ));
            assert_eq!(attempted.encodes, before.encodes + 4);
            assert_eq!(attempted.pixels, before.pixels + 4 * 4096);
        }
        let (exact, _, _) = driver_run(
            ProbeLimits {
                max_encodes: usage.encodes,
                max_encoded_pixels: usage.pixels,
                ..Default::default()
            },
            batched,
        );
        assert_eq!(exact.unwrap().params, result.params);
        let (short, attempted, _) = driver_run(
            ProbeLimits {
                max_encodes: usage.encodes - 1,
                ..Default::default()
            },
            batched,
        );
        assert!(matches!(short, Err(Error::Limit(_))));
        assert!(attempted.encodes < usage.encodes);
    }
}
