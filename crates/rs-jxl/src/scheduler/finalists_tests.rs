use super::*;
use crate::scheduler::search::EncodedOutput;
use crate::{PixelData, PixelLayout};
use std::cell::Cell;

type Probe = fn(&CandidateParams, &Image) -> Result<usize>;

fn image() -> Image {
    Image::new(16, 8, PixelLayout::Gray, PixelData::U8(vec![7; 128])).unwrap()
}

fn params(id: u32) -> CandidateParams {
    CandidateParams {
        speed_tier: 1,
        rct: Some(id),
        ..Default::default()
    }
}

fn complete(bytes: usize, artifact: usize) -> TrialOutput<usize> {
    TrialOutput::Complete(EncodedOutput {
        bytes: vec![artifact as u8; bytes],
        artifact,
    })
}

fn options() -> DriverOptions {
    let mut options = DriverOptions::scout();
    options.scout_pilot_first = true;
    options.scout_parallel_finalists = true;
    options
}

fn fixture<'a, F>(
    image: &'a Image,
    input: &'a CandidateParams,
    options: &'a DriverOptions,
    baseline: bool,
    encode: F,
    full_batch: Option<FullBatchEncoder<'a, usize>>,
) -> Driver<'a, F, Probe, usize>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<usize>>,
{
    let no_probe: Probe = |_, _| panic!("finalists must not encode probes");
    let screen = |callback: ProbeCallback<'a>| {
        CandidateScreen::new(image, None, options.screen.clone(), callback).unwrap()
    };
    let mut search = Search::new(options.full_limits);
    if baseline {
        search
            .refine(image, 9, input.clone(), true, false, &mut |_, _, budget| {
                assert!(budget.beat_bytes.is_none());
                Ok(complete(100, 0))
            })
            .unwrap();
    }
    Driver {
        image,
        input,
        options,
        scan: None,
        level: 9,
        search,
        encode,
        full_batch,
        pilot_overlap: None,
        probe: Rc::new(RefCell::new(ProbeWork {
            callback: no_probe,
            limits: options.screen.limits,
            encodes: 0,
            pixels: 0,
        })),
        probe_batch: None,
        prefix_batch: None,
        screen: screen(Box::new(no_probe)),
        context_screen: screen(Box::new(no_probe)),
        auxiliary_screen_batches: 0,
        routes: ActiveRoutes::new(options, 128, None),
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

#[test]
fn parallel_finalists_are_opt_in_and_require_pilot_first() {
    assert!(!DriverOptions::default().scout_parallel_finalists);
    let image = image();
    let mut options = DriverOptions::default();
    options.scout_parallel_finalists = true;
    let result = run_pixel_search::<_, _, ()>(
        &image,
        &CandidateParams::default().backend(9).unwrap(),
        &options,
        |_, _, _| panic!("invalid settings must not encode"),
        |_, _| panic!("invalid settings must not probe"),
    );
    assert!(matches!(result, Err(Error::InvalidInput(message))
        if message == "pilot refinements require pilot-first"));
}

#[test]
fn serial_fallback_keeps_sequential_bounds_and_reuses_duplicates() {
    let image = image();
    let input = CandidateParams::default().backend(9).unwrap();
    for (enabled, batch_present) in [(false, false), (true, false), (false, true)] {
        let mut options = options();
        options.scout_parallel_finalists = enabled;
        let mut calls = Vec::new();
        let batch: Option<FullBatchEncoder<'_, usize>> = batch_present.then(|| {
            let callback: FullBatchEncoder<'_, usize> = Box::new(|_, _| panic!("disabled batch"));
            callback
        });
        let mut driver = fixture(
            &image,
            &input,
            &options,
            true,
            |p, _, budget| {
                calls.push((p.rct, budget.beat_bytes));
                Ok(match p.rct {
                    Some(1) => complete(80, 1),
                    Some(2) => TrialOutput::Bounded { lower_bound: 80 },
                    _ => panic!("unexpected uncached candidate"),
                })
            },
            batch,
        );
        assert_eq!(
            driver
                .full_finalists(9, &[params(1), params(2), params(1), input.clone()])
                .unwrap(),
            [Some(80), None, Some(80), Some(100)]
        );
        let result = driver.finish().unwrap();
        assert_eq!(calls, [(Some(1), Some(100)), (Some(2), Some(80))]);
        assert_eq!(result.bytes, vec![1; 80]);
        assert_eq!(result.artifact, 1);
        assert_eq!(result.report.full_encodes, 3);
        assert_eq!(result.report.full_pixels, 3 * 128);
    }
}

#[test]
fn batches_use_dispatch_bounds_and_commit_stable_plan_order() {
    let image = image();
    let input = CandidateParams::default().backend(9).unwrap();
    let mut options = options();
    options.full_limits.max_full_encodes = 5;
    options.full_limits.max_full_pixels = 5 * 128;
    let mut waves = Vec::new();
    let batch = Box::new(|_: &Image, tasks: &[FullTask<'_>]| {
        assert!(tasks.iter().all(|task| task.cooperative));
        assert!(tasks.len() <= 2);
        let ids: Vec<_> = tasks.iter().map(|task| task.params.rct.unwrap()).collect();
        let bound = if waves.is_empty() { 100 } else { 80 };
        assert!(
            tasks
                .iter()
                .all(|task| task.budget.beat_bytes == Some(bound))
        );
        assert!(
            tasks
                .iter()
                .all(|task| task.budget.max_output_bytes == options.full_limits.max_output_bytes)
        );
        waves.push(ids);
        Ok(tasks
            .iter()
            .map(|task| match task.params.rct.unwrap() {
                1 => complete(80, 1),
                2 => complete(80, 2),
                3 => complete(60, 3),
                4 => TrialOutput::Bounded { lower_bound: 80 },
                _ => unreachable!(),
            })
            .collect())
    });
    let mut driver = fixture(
        &image,
        &input,
        &options,
        true,
        |_, _, _| panic!("batch available"),
        Some(batch),
    );
    assert_eq!(
        driver
            .full_finalists(
                9,
                &[
                    params(1),
                    params(2),
                    params(1),
                    input.clone(),
                    params(3),
                    params(4),
                ]
            )
            .unwrap(),
        [Some(80), Some(80), Some(80), Some(100), Some(60), None]
    );
    assert_eq!(
        driver.full_finalists(9, &[params(4), params(1)]).unwrap(),
        [None, Some(80)]
    );
    let result = driver.finish().unwrap();
    assert_eq!(waves, [vec![1, 2], vec![3, 4]]);
    assert_eq!(result.bytes, vec![3; 60]);
    assert_eq!(result.artifact, 3);
    assert_eq!(result.report.full_encodes, 5);
    assert_eq!(result.report.full_pixels, 5 * 128);
    assert_eq!(result.report.winner_index, 3);
    assert_eq!(result.report.trials[4].bytes, None);
    assert_eq!(result.report.trials[4].lower_bound, Some(60));
}

#[test]
fn worker_and_scratch_limits_cap_root_count_without_new_pool() {
    let image = image();
    let input = CandidateParams::default().backend(9).unwrap();
    let scratch = 128 * 3 * 32 + (32 << 20);
    for (workers, memory, expected) in [
        (8, 1 << 30, vec![2, 1]),
        (1, 1 << 30, vec![1, 1, 1]),
        (8, scratch, vec![1, 1, 1]),
        (8, 1, vec![1, 1, 1]),
    ] {
        let mut options = options();
        options.workers = WorkerLimits {
            tasks: workers,
            scratch_bytes: memory,
        };
        let mut waves = Vec::new();
        let batch = Box::new(|_: &Image, tasks: &[FullTask<'_>]| {
            waves.push(tasks.len());
            assert!(
                tasks
                    .iter()
                    .all(|task| task.cooperative && task.budget.beat_bytes == Some(100))
            );
            Ok(tasks
                .iter()
                .map(|task| complete(110, task.params.rct.unwrap() as usize))
                .collect())
        });
        let mut driver = fixture(
            &image,
            &input,
            &options,
            true,
            |_, _, _| panic!("batch available"),
            Some(batch),
        );
        assert_eq!(
            driver
                .full_finalists(9, &[params(1), params(2), params(3)])
                .unwrap(),
            [Some(110); 3]
        );
        let result = driver.finish().unwrap();
        assert_eq!(waves, expected);
        assert_eq!(result.report.full_encodes, 4);
        assert_eq!(result.artifact, 0);
    }
}

#[test]
fn invalid_batch_outputs_never_commit_and_cannot_reset_work_budget() {
    let image = image();
    let input = CandidateParams::default().backend(9).unwrap();
    for failure in ["count", "empty", "oversize", "bound", "callback"] {
        let mut options = options();
        options.full_limits = SearchLimits {
            max_full_encodes: 3,
            max_full_pixels: 3 * 128,
            max_output_bytes: 100,
        };
        let calls = Cell::new(0);
        let batch = Box::new(|_: &Image, tasks: &[FullTask<'_>]| {
            calls.set(calls.get() + 1);
            assert_eq!(tasks.len(), 2);
            assert!(tasks.iter().all(|task| task.budget.beat_bytes == Some(100)));
            match failure {
                "count" => Ok(vec![complete(80, 1)]),
                "empty" => Ok(vec![complete(80, 1), complete(0, 2)]),
                "oversize" => Ok(vec![complete(80, 1), complete(101, 2)]),
                // Invalid against dispatch=100 even though the first result
                // could lower the incumbent to 80 during ordered commitment.
                "bound" => Ok(vec![
                    complete(80, 1),
                    TrialOutput::Bounded { lower_bound: 90 },
                ]),
                "callback" => Err(Error::Encoding("callback failed".into())),
                _ => unreachable!(),
            }
        });
        let mut driver = fixture(
            &image,
            &input,
            &options,
            true,
            |_, _, _| panic!("batch available"),
            Some(batch),
        );
        let candidates = [params(1), params(2)];
        let error = driver.full_finalists(9, &candidates).unwrap_err();
        if failure == "bound" {
            assert!(error.to_string().contains("dispatch incumbent"));
        }
        assert_eq!(driver.search.trials().len(), 1);
        assert!(matches!(
            driver.full_finalists(9, &candidates),
            Err(Error::Limit("whole-frame search work"))
        ));
        let result = driver.finish().unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(result.bytes, vec![0; 100]);
        assert_eq!(result.report.full_encodes, 3);
        assert_eq!(result.report.full_pixels, 3 * 128);
    }
}

#[test]
fn work_limits_and_invalid_candidates_reject_before_batch_dispatch() {
    let image = image();
    let input = CandidateParams::default().backend(9).unwrap();
    for limit in ["encodes", "pixels", "stage", "params", "workers"] {
        let mut options = options();
        let mut candidates = vec![params(1), params(2)];
        let mut stage = 9;
        match limit {
            "encodes" => options.full_limits.max_full_encodes = 2,
            "pixels" => options.full_limits.max_full_pixels = 2 * 128,
            "stage" => stage = 0,
            "params" => candidates[1].rct = Some(42),
            "workers" => options.workers.tasks = 0,
            _ => unreachable!(),
        }
        let mut driver = fixture(
            &image,
            &input,
            &options,
            true,
            |_, _, _| panic!("must reject before encoding"),
            Some(Box::new(|_, _| panic!("must reject before dispatch"))),
        );
        assert!(driver.full_finalists(stage, &candidates).is_err());
        let result = driver.finish().unwrap();
        assert_eq!(result.report.full_encodes, 1);
        assert_eq!(result.report.full_pixels, 128);
    }
}

#[test]
fn finalists_need_a_completed_baseline_even_with_batch_callback() {
    let image = image();
    let input = CandidateParams::default().backend(9).unwrap();
    let options = options();
    let mut driver = fixture(
        &image,
        &input,
        &options,
        false,
        |_, _, _| panic!("missing baseline"),
        Some(Box::new(|_, _| panic!("missing baseline"))),
    );
    assert!(driver.full_finalists(9, &[]).unwrap().is_empty());
    assert!(
        driver
            .full_finalists(9, &[params(1)])
            .unwrap_err()
            .to_string()
            .contains("completed baseline")
    );
    driver
        .search
        .refine(&image, 9, input.clone(), false, false, &mut |_, _, _| {
            Ok(complete(100, 0))
        })
        .unwrap();
    assert!(
        driver
            .full_finalists(9, &[params(1)])
            .unwrap_err()
            .to_string()
            .contains("completed baseline")
    );
}
