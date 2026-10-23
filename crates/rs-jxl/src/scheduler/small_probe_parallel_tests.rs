//! Scheduler score oracles, not compressed-byte identity or codec acceptance.

use super::*;
use crate::{PixelData, PixelLayout, execution};
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::Duration;

fn image(width: u32, height: u32) -> Image {
    Image::new(
        width,
        height,
        PixelLayout::Gray,
        PixelData::U8(vec![1; width as usize * height as usize]),
    )
    .unwrap()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn score(plan: &[Candidate], params: &CandidateParams) -> usize {
    plan.iter()
        .position(|c| &c.params == params)
        .map_or(100, |i| 10 + i * 3)
}

fn same_winners(actual: &ProbeResult, expected: &ProbeResult) {
    assert_eq!(actual.shortlisted, expected.shortlisted);
    assert_eq!(actual.control_bytes, expected.control_bytes);
    assert_eq!(actual.small_control_bytes, expected.small_control_bytes);
    for (actual, expected) in [
        (&actual.ranked, &expected.ranked),
        (&actual.exploratory, &expected.exploratory),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.params, b.params);
            assert_eq!(a.name, b.name);
            assert_eq!(
                (a.bytes, a.small_bytes, a.context_bytes),
                (b.bytes, b.small_bytes, b.context_bytes)
            );
            assert_eq!(a.exploratory, b.exploratory);
        }
    }
}

struct Active<'a>(&'a AtomicUsize);

impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[test]
fn parallel_helper_matches_exhaustive_without_feedback_and_bounds_null_workers() {
    let params = CandidateParams::default();
    for (width, height) in [(256, 256), (1024, 768)] {
        let image = image(width, height);
        for extended_rct in [false, true] {
            let options = PlanOptions { extended_rct };
            let plan = lossless_probe_plan(&params, options);
            let expected = probe_lossless_candidates(
                &image,
                &params,
                Some(&params),
                options,
                false,
                ProbeLimits::default(),
                |p, _| Ok(score(&plan, p)),
            )
            .unwrap();
            for threads in [1, 3] {
                let pool = pool(threads);
                for slots in [1, 2, 8] {
                    let active = AtomicUsize::new(0);
                    let peak = AtomicUsize::new(0);
                    let calls = AtomicUsize::new(0);
                    let pixels = AtomicU64::new(0);
                    let actual = probe_lossless_candidates_parallel(
                        &image,
                        &params,
                        Some(&params),
                        options,
                        false,
                        ProbeLimits::default(),
                        &pool,
                        slots,
                        |p, sample| {
                            assert!(pool.current_thread_index().is_some());
                            assert!(execution::is_serial());
                            assert_eq!(execution::worker_count(), 1);
                            let live = active.fetch_add(1, Ordering::SeqCst) + 1;
                            let _active = Active(&active);
                            peak.fetch_max(live, Ordering::SeqCst);
                            assert!(live <= slots.min(threads));
                            calls.fetch_add(1, Ordering::Relaxed);
                            pixels.fetch_add(
                                u64::from(sample.width()) * u64::from(sample.height()),
                                Ordering::Relaxed,
                            );
                            std::thread::yield_now();
                            Ok(score(&plan, p))
                        },
                    )
                    .unwrap();
                    same_winners(&actual, &expected);
                    assert_eq!(actual.encodes, expected.encodes);
                    assert_eq!(actual.sample_pixels, expected.sample_pixels);
                    assert_eq!(actual.encodes, calls.load(Ordering::Relaxed));
                    assert_eq!(actual.sample_pixels, pixels.load(Ordering::Relaxed));
                    assert_eq!(actual.screen_saved_encodes, 0);
                    assert_eq!(actual.validation_encodes, expected.validation_encodes);
                    assert_eq!(active.load(Ordering::SeqCst), 0);
                    assert!((1..=slots.min(threads)).contains(&peak.load(Ordering::SeqCst)));
                }
            }
        }
    }
}

#[test]
fn parallel_helper_reuses_live_cutoff_without_a_batch_barrier() {
    let image = image(256, 256);
    let params = CandidateParams::default();
    let options = PlanOptions::default();
    let plan = lossless_probe_plan(&params, options);
    let size = |p: &CandidateParams| match plan.iter().position(|c| &c.params == p) {
        Some(i) if (1..=6).contains(&i) => 1,
        Some(_) => 1000,
        None => 100,
    };
    let expected = probe_lossless_candidates(
        &image,
        &params,
        Some(&params),
        options,
        false,
        ProbeLimits::default(),
        |p, _| Ok(size(p)),
    )
    .unwrap();
    let calls: Vec<_> = plan.iter().map(|_| AtomicUsize::new(0)).collect();
    let released = (Mutex::new(false), Condvar::new());
    let pool = pool(2);
    let actual = probe_lossless_candidates_parallel(
        &image,
        &params,
        Some(&params),
        options,
        true,
        ProbeLimits::default(),
        &pool,
        2,
        |p, _| {
            assert!(execution::is_serial());
            if let Some(i) = plan.iter().position(|c| &c.params == p) {
                let previous = calls[i].fetch_add(1, Ordering::Relaxed);
                if i == 0 && previous == 0 {
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
                if i == 7 && previous == 0 {
                    // Six other candidates have completed while candidate 0
                    // still waits inside its first encode, as in prefix tests.
                    *released.0.lock().unwrap() = true;
                    released.1.notify_all();
                }
            }
            Ok(size(p))
        },
    )
    .unwrap();
    same_winners(&actual, &expected);
    assert_eq!(calls[0].load(Ordering::Relaxed), 1);
    assert!(actual.screen_saved_encodes > 0);
    assert!(actual.encodes < expected.encodes);
    for (i, candidate) in plan.iter().enumerate() {
        if candidate.params.predictor == Some(Predictor::Variable) {
            assert_eq!(calls[i].load(Ordering::Relaxed), actual.screen_samples);
        }
    }
}

#[test]
fn parallel_helper_budget_covers_controls_prefixes_and_validation() {
    let params = CandidateParams::default();
    let options = PlanOptions::default();
    let plan = lossless_probe_plan(&params, options);
    let pool = pool(3);
    let small = image(256, 256);
    let control = small_probe_regions(&small).len();
    assert_eq!(control, 9);
    for (limits, expected) in [
        (
            ProbeLimits {
                max_encodes: control - 1,
                ..Default::default()
            },
            0,
        ),
        (
            ProbeLimits {
                max_encoded_pixels: (control * 4096 - 1) as u64,
                ..Default::default()
            },
            0,
        ),
        (
            ProbeLimits {
                max_encodes: control + 7,
                ..Default::default()
            },
            control + 7,
        ),
        (
            ProbeLimits {
                max_encoded_pixels: ((control + 7) * 4096 + 4095) as u64,
                ..Default::default()
            },
            control + 7,
        ),
        (
            ProbeLimits {
                max_candidates: 0,
                ..Default::default()
            },
            0,
        ),
        (
            ProbeLimits {
                max_resident_sample_pixels: 0,
                ..Default::default()
            },
            0,
        ),
    ] {
        let calls = AtomicUsize::new(0);
        let result = probe_lossless_candidates_parallel(
            &small,
            &params,
            Some(&params),
            options,
            false,
            limits,
            &pool,
            8,
            |p, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(score(&plan, p))
            },
        );
        assert!(matches!(result, Err(Error::Limit(_))));
        assert_eq!(calls.load(Ordering::Relaxed), expected);
    }
    let large = image(1024, 768);
    let complete = probe_lossless_candidates_parallel(
        &large,
        &params,
        Some(&params),
        options,
        false,
        ProbeLimits::default(),
        &pool,
        8,
        |p, _| Ok(score(&plan, p)),
    )
    .unwrap();
    assert_eq!(complete.validation_encodes, 4);
    let limits = ProbeLimits {
        max_encodes: complete.encodes,
        max_encoded_pixels: complete.sample_pixels,
        ..Default::default()
    };
    let exact = probe_lossless_candidates_parallel(
        &large,
        &params,
        Some(&params),
        options,
        false,
        limits,
        &pool,
        8,
        |p, _| Ok(score(&plan, p)),
    )
    .unwrap();
    same_winners(&exact, &complete);
    assert_eq!(exact.encodes, complete.encodes);
    for limits in [
        ProbeLimits {
            max_encodes: complete.encodes - 1,
            ..limits
        },
        ProbeLimits {
            max_encoded_pixels: complete.sample_pixels - 1,
            ..limits
        },
    ] {
        let calls = AtomicUsize::new(0);
        let validation = AtomicUsize::new(0);
        let result = probe_lossless_candidates_parallel(
            &large,
            &params,
            Some(&params),
            options,
            false,
            limits,
            &pool,
            8,
            |p, sample| {
                calls.fetch_add(1, Ordering::Relaxed);
                if sample.width() > 64 {
                    validation.fetch_add(1, Ordering::Relaxed);
                }
                Ok(score(&plan, p))
            },
        );
        assert!(matches!(result, Err(Error::Limit("probe work budget"))));
        assert_eq!(
            calls.load(Ordering::Relaxed),
            complete.encodes - complete.validation_encodes
        );
        assert_eq!(validation.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn parallel_helper_errors_drain_controls_and_cancel_future_prefix_cells() {
    let image = image(256, 256);
    let params = CandidateParams::default();
    let options = PlanOptions::default();
    let plan = lossless_probe_plan(&params, options);
    let pool = pool(2);
    let controls = small_probe_regions(&image).len();
    for mode in 0..5 {
        let calls = AtomicUsize::new(0);
        let first = AtomicUsize::new(0);
        let result = probe_lossless_candidates_parallel(
            &image,
            &params,
            Some(&params),
            options,
            false,
            ProbeLimits::default(),
            &pool,
            1,
            |p, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                if *p == params {
                    return match mode {
                        0 => Err(Error::Limit("control backend failure")),
                        1 => Ok(0),
                        _ => Ok(100),
                    };
                }
                assert_eq!(*p, plan[0].params);
                let tile = first.fetch_add(1, Ordering::Relaxed);
                match mode {
                    2 => Err(Error::Limit("prefix backend failure")),
                    3 => Ok(0),
                    _ => Ok(if tile == 0 { usize::MAX } else { 1 }),
                }
            },
        );
        match mode {
            0 => assert!(matches!(
                result,
                Err(Error::Limit("control backend failure"))
            )),
            1 | 3 => assert!(matches!(result, Err(Error::Encoding(_)))),
            2 => assert!(matches!(
                result,
                Err(Error::Limit("prefix backend failure"))
            )),
            _ => assert!(matches!(result, Err(Error::Limit("probe bytes")))),
        }
        let prefix = match mode {
            0 | 1 => 0,
            2 | 3 => 1,
            _ => 2,
        };
        assert_eq!(calls.load(Ordering::Relaxed), controls + prefix);
    }
    // All failure paths leave the same borrowed pool usable.
    assert!(
        probe_lossless_candidates_parallel(
            &image,
            &params,
            Some(&params),
            options,
            false,
            ProbeLimits::default(),
            &pool,
            2,
            |p, _| Ok(score(&plan, p)),
        )
        .is_ok()
    );
}

#[test]
fn parallel_helper_validates_slots_and_outer_null_scope_never_enters_pool() {
    let small = image(64, 64);
    let params = CandidateParams::default();
    let pool = pool(3);
    for slots in [0, 65, usize::MAX] {
        assert!(matches!(
            probe_lossless_candidates_parallel(
                &small,
                &params,
                None,
                PlanOptions::default(),
                true,
                ProbeLimits::default(),
                &pool,
                slots,
                |_, _| panic!("invalid worker slots"),
            ),
            Err(Error::InvalidInput(_))
        ));
    }
    let empty = probe_lossless_candidates_parallel(
        &small,
        &params,
        None,
        PlanOptions::default(),
        true,
        ProbeLimits::default(),
        &pool,
        8,
        |_, _| panic!("small-input gate"),
    )
    .unwrap();
    assert_eq!(empty.encodes, 0);
    let image = image(1024, 768);
    let options = PlanOptions::default();
    let plan = lossless_probe_plan(&params, options);
    let expected = probe_lossless_candidates(
        &image,
        &params,
        Some(&params),
        options,
        false,
        ProbeLimits::default(),
        |p, _| Ok(score(&plan, p)),
    )
    .unwrap();
    let caller = std::thread::current().id();
    let actual = execution::serial(|| {
        probe_lossless_candidates_parallel(
            &image,
            &params,
            Some(&params),
            options,
            false,
            ProbeLimits::default(),
            &pool,
            8,
            |p, _| {
                assert!(execution::is_serial());
                assert_eq!(execution::worker_count(), 1);
                assert!(pool.current_thread_index().is_none());
                assert_eq!(std::thread::current().id(), caller);
                Ok(score(&plan, p))
            },
        )
    })
    .unwrap();
    same_winners(&actual, &expected);
    assert_eq!(actual.encodes, expected.encodes);
    assert!(!execution::is_serial());
}

#[test]
fn legacy_mutable_callbacks_keep_their_serial_and_wave_contracts() {
    let image = image(256, 256);
    let params = CandidateParams::default();
    let options = PlanOptions::default();
    let plan = lossless_probe_plan(&params, options);
    let mut serial_calls = 0;
    let serial = probe_lossless_candidates(
        &image,
        &params,
        Some(&params),
        options,
        false,
        ProbeLimits::default(),
        |p, _| {
            serial_calls += 1;
            Ok(score(&plan, p))
        },
    )
    .unwrap();
    let mut batches = Vec::new();
    let batched = probe_lossless_candidates_batched(
        &image,
        &params,
        Some(&params),
        options,
        false,
        ProbeLimits::default(),
        |tasks| {
            batches.push(tasks.len());
            Ok(tasks.iter().map(|task| score(&plan, task.params)).collect())
        },
    )
    .unwrap();
    same_winners(&batched, &serial);
    assert_eq!(serial_calls, serial.encodes);
    assert_eq!(batches.iter().sum::<usize>(), batched.encodes);
    assert!(batches.contains(&8));
    assert_eq!(serial.encodes, batched.encodes);
}
