//! Frozen ProbeLosslessCandidates shortlist task and budget oracles.
use super::*;
use crate::scheduler::{
    batch::PrefixBatchEncoder,
    params::{CandidateParams, PlanOptions, lossless_probe_plan},
    probe::ProbeLimits,
    small_probe::{probe_lossless_candidates, probe_lossless_candidates_with_prefix},
};
use crate::{PixelData, PixelLayout};
use std::{cell::Cell, sync::Condvar, time::Duration};

fn image(width: u32, height: u32) -> Image {
    Image::new(
        width,
        height,
        PixelLayout::Gray,
        PixelData::U8(vec![1; width as usize * height as usize]),
    )
    .unwrap()
}

fn candidates(scores: &[usize]) -> Vec<Candidate> {
    scores
        .iter()
        .map(|&bytes| {
            let mut candidate = Candidate::new(
                "shortlist",
                CandidateParams {
                    predictor: Some(Predictor::Variable),
                    ..Default::default()
                },
            );
            candidate.bytes = Some(bytes);
            candidate
        })
        .collect()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

#[test]
fn shortlist_keeps_completed_scores_distinct_from_bounded_variable_prefixes() {
    let candidates = candidates(&[9, 10, 11, 1, 10, 3]);
    let samples = [image(2, 3), image(3, 4), image(5, 7)];
    for max_bytes in [10.0, 10.5] {
        for threads in [1, 3] {
            let pool = pool(threads);
            let work = PrefixWork::new(18, 1000);
            let screen = PrefixScreen {
                candidates: &candidates,
                samples: &samples,
                order: &[2, 0, 1],
                pass: PrefixPass::Shortlist { max_bytes },
                work: &work,
            };
            let output = run_prefix_screen(Some(&pool), 8, &screen, |i, tile| {
                Ok(if i == 5 && tile == 1 { 100 } else { 1 })
            })
            .unwrap();
            assert_eq!(
                output
                    .scores()
                    .iter()
                    .map(|s| (s.bytes(), s.encodes()))
                    .collect::<Vec<_>>(),
                [
                    (None, 2),
                    (None, 1),
                    (None, 0),
                    (Some(4), 3),
                    (None, 1),
                    (Some(105), 3)
                ]
            );
            // An over-limit LAST tile is complete, not a prefix bound. Ranking
            // rejects it afterward. Prior screen bytes cost no new work here.
            assert_eq!(
                work.usage(),
                PrefixUsage {
                    encodes: 10,
                    pixels: 217
                }
            );
        }
    }
}

#[test]
fn shortlist_candidates_advance_independently_without_tile_barriers() {
    let candidates = candidates(&[3; 6]);
    let samples = [image(8, 8), image(8, 8), image(8, 8)];
    let pool = pool(2);
    let work = PrefixWork::new(18, 18 * 64);
    let screen = PrefixScreen {
        candidates: &candidates,
        samples: &samples,
        order: &[2, 0, 1],
        pass: PrefixPass::Shortlist { max_bytes: 100.0 },
        work: &work,
    };
    let finished = (Mutex::new(false), Condvar::new());
    let output = run_prefix_screen(Some(&pool), 2, &screen, |i, tile| {
        assert!(crate::execution::is_serial());
        if i == 0 && tile == 2 {
            let (done, timeout) = finished
                .1
                .wait_timeout_while(
                    finished.0.lock().unwrap(),
                    Duration::from_secs(10),
                    |done| !*done,
                )
                .unwrap();
            assert!(
                *done && !timeout.timed_out(),
                "shortlist still has a tile barrier"
            );
        }
        if i == 1 && tile == 1 {
            *finished.0.lock().unwrap() = true;
            finished.1.notify_all();
        }
        Ok(1)
    })
    .unwrap();
    assert!(
        output
            .scores()
            .iter()
            .all(|s| s.bytes() == Some(6) && s.encodes() == 3)
    );
    assert_eq!(work.usage().encodes, 18);
}

#[test]
fn empty_shortlist_remainder_reuses_complete_screen_totals_without_work() {
    let candidates = candidates(&[3, 4, 5]);
    let work = PrefixWork::new(0, 0);
    let screen = PrefixScreen {
        candidates: &candidates,
        samples: &[],
        order: &[],
        pass: PrefixPass::Shortlist { max_bytes: 3.5 },
        work: &work,
    };
    let output = run_prefix_screen(None, 8, &screen, |_, _| panic!("no remaining tiles")).unwrap();
    assert_eq!(
        output
            .scores()
            .iter()
            .map(|s| s.bytes())
            .collect::<Vec<_>>(),
        [Some(3), Some(4), Some(5)]
    );
    assert_eq!(work.usage(), PrefixUsage::default());
}

#[test]
fn shortlist_rejects_unmeasured_initial_scores_and_invalid_float_bounds() {
    let mut candidates = candidates(&[3]);
    let samples = [image(8, 8)];
    let work = PrefixWork::new(100, 10000);
    for bytes in [None, Some(0), Some(3)] {
        candidates[0].bytes = bytes;
        for max_bytes in [-1.0, f64::INFINITY, f64::NAN, 10.0] {
            if bytes == Some(3) && max_bytes == 10.0 {
                continue;
            }
            let screen = PrefixScreen {
                candidates: &candidates,
                samples: &samples,
                order: &[0],
                pass: PrefixPass::Shortlist { max_bytes },
                work: &work,
            };
            assert!(matches!(
                run_prefix_screen(None, 8, &screen, |_, _| panic!("invalid input")),
                Err(Error::InvalidInput(_))
            ));
        }
    }
    assert_eq!(work.usage(), PrefixUsage::default());
}

#[test]
fn shortlist_error_and_limit_abort_without_exposing_the_screen_score_as_full() {
    let candidates = candidates(&[3; 6]);
    let samples = [image(8, 8), image(8, 8), image(8, 8)];
    for failure in 0..4 {
        let work = PrefixWork::new(
            if failure == 0 { 2 } else { 100 },
            if failure == 1 { 2 * 64 } else { 10000 },
        );
        let screen = PrefixScreen {
            candidates: &candidates,
            samples: &samples,
            order: &[2, 0, 1],
            pass: PrefixPass::Shortlist { max_bytes: 100.0 },
            work: &work,
        };
        let attempts = AtomicUsize::new(0);
        let result = run_prefix_screen(None, 1, &screen, |_, _| {
            attempts.fetch_add(1, Ordering::Relaxed);
            match failure {
                2 => Err(Error::Limit("shortlist backend error")),
                3 => Ok(0),
                _ => Ok(1),
            }
        });
        assert!(result.is_err());
        let expected = if failure < 2 { 2 } else { 1 };
        assert_eq!(attempts.load(Ordering::Relaxed), expected);
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
fn small_probe_reuses_prefix_callback_for_shortlist_and_shares_actual_budget() {
    let image = image(256, 256);
    let params = CandidateParams::default();
    let screen_encodes = lossless_probe_plan(&params, PlanOptions::default()).len() * 3;
    let max_encodes = 9 + screen_encodes + 4;
    for limits in [
        ProbeLimits {
            max_encodes,
            ..Default::default()
        },
        ProbeLimits {
            max_encoded_pixels: max_encodes as u64 * 4096,
            ..Default::default()
        },
    ] {
        let calls = AtomicUsize::new(0);
        let passes = Cell::new(0);
        let mut prefix: PrefixBatchEncoder<'_> = Box::new(|screen| {
            let bit = match screen.pass {
                PrefixPass::Screen { .. } => 1,
                PrefixPass::Shortlist { .. } => 2,
            };
            passes.set(passes.get() | bit);
            run_prefix_screen(None, 8, screen, |_, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(10)
            })
        });
        let result = probe_lossless_candidates_with_prefix(
            &image,
            &params,
            Some(&params),
            PlanOptions::default(),
            false,
            limits,
            |tasks| {
                calls.fetch_add(tasks.len(), Ordering::Relaxed);
                Ok(vec![10; tasks.len()])
            },
            &mut prefix,
        );
        assert!(matches!(
            result,
            Err(Error::Limit("cumulative scheduler probe work"))
        ));
        assert_eq!(passes.get(), 3);
        assert_eq!(calls.load(Ordering::Relaxed), max_encodes);
    }
}

#[test]
fn thin_frame_with_all_tiles_screened_keeps_exhaustive_finalists() {
    let image = image(8192, 1);
    let params = CandidateParams::default();
    let expected = probe_lossless_candidates(
        &image,
        &params,
        Some(&params),
        PlanOptions::default(),
        false,
        ProbeLimits::default(),
        |_, _| Ok(10),
    )
    .unwrap();
    let saw_empty = Cell::new(false);
    let mut prefix: PrefixBatchEncoder<'_> = Box::new(|screen| {
        if matches!(screen.pass, PrefixPass::Shortlist { .. }) {
            assert!(screen.order.is_empty());
            saw_empty.set(true);
        }
        run_prefix_screen(None, 8, screen, |_, _| Ok(10))
    });
    let actual = probe_lossless_candidates_with_prefix(
        &image,
        &params,
        Some(&params),
        PlanOptions::default(),
        true,
        ProbeLimits::default(),
        |tasks| Ok(vec![10; tasks.len()]),
        &mut prefix,
    )
    .unwrap();
    assert!(saw_empty.get());
    assert_eq!(actual.samples, actual.screen_samples);
    assert_eq!((actual.encodes, actual.bounded), (expected.encodes, 0));
    assert_eq!(actual.ranked.len(), expected.ranked.len());
    for (a, b) in actual.ranked.iter().zip(&expected.ranked) {
        assert_eq!(
            (&a.params, a.bytes, a.small_bytes),
            (&b.params, b.bytes, b.small_bytes)
        );
    }
}

#[test]
fn already_over_limit_shortlist_needs_no_budget_and_never_ranks_partial_scores() {
    let image = image(256, 256);
    let params = CandidateParams::default();
    let encodes = 9 + lossless_probe_plan(&params, PlanOptions::default()).len() * 3;
    let limits = ProbeLimits {
        max_encodes: encodes,
        max_encoded_pixels: encodes as u64 * 4096,
        ..Default::default()
    };
    let size = |p: &CandidateParams| if *p == params { 100 } else { 1000 };
    let mut prefix: PrefixBatchEncoder<'_> = Box::new(|screen| {
        let output = run_prefix_screen(None, 8, screen, |i, _| {
            Ok(size(&screen.candidates[i].params))
        })?;
        if matches!(screen.pass, PrefixPass::Shortlist { .. }) {
            assert_eq!(screen.work.usage(), PrefixUsage::default());
            assert!(output.scores().iter().all(|score| score.bytes().is_none()));
        }
        Ok(output)
    });
    let actual = probe_lossless_candidates_with_prefix(
        &image,
        &params,
        Some(&params),
        PlanOptions::default(),
        false,
        limits,
        |tasks| Ok(tasks.iter().map(|task| size(task.params)).collect()),
        &mut prefix,
    )
    .unwrap();
    let expected = probe_lossless_candidates(
        &image,
        &params,
        Some(&params),
        PlanOptions::default(),
        false,
        limits,
        |p, _| Ok(size(p)),
    )
    .unwrap();
    assert!(actual.ranked.is_empty());
    assert_eq!(actual.bounded, 6);
    assert_eq!(
        (actual.encodes, actual.bounded),
        (expected.encodes, expected.bounded)
    );
    assert!(
        actual
            .exploratory
            .iter()
            .all(|candidate| candidate.bytes == Some(3000))
    );
}
