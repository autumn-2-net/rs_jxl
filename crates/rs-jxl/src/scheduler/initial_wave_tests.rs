use super::*;
use crate::scheduler::search::EncodedOutput;
use crate::{PixelData, PixelLayout};
use std::cell::Cell;

type Full = fn(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<usize>>;
type Probe = fn(&CandidateParams, &Image) -> Result<usize>;

fn image(width: u32, height: u32, alpha: bool) -> Image {
    Image::new(
        width,
        height,
        if alpha {
            PixelLayout::GrayAlpha
        } else {
            PixelLayout::Gray
        },
        PixelData::U8(vec![
            7;
            width as usize
                * height as usize
                * (1 + usize::from(alpha))
        ]),
    )
    .unwrap()
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
    options.scout_initial_wave = true;
    options.workers.tasks = 16;
    options
}

fn fixture<'a>(
    image: &'a Image,
    input: &'a CandidateParams,
    options: &'a DriverOptions,
    full_batch: Option<FullBatchEncoder<'a, usize>>,
) -> Driver<'a, Full, Probe, usize> {
    let no_probe: Probe = |_, _| panic!("initial helper must not probe");
    let no_full: Full = |_, _, _| panic!("helper fallback must not submit work");
    let screen = |callback: ProbeCallback<'a>| {
        CandidateScreen::new(image, None, options.screen.clone(), callback).unwrap()
    };
    Driver {
        image,
        input,
        options,
        scan: None,
        level: 9,
        search: Search::new(options.full_limits),
        encode: no_full,
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
        routes: ActiveRoutes::new(
            options,
            u64::from(image.width()) * u64::from(image.height()),
            None,
        ),
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
fn initial_wave_is_independent_opt_in_and_requires_both_pilot_flags() {
    assert!(!DriverOptions::default().scout_initial_wave);
    assert!(!DriverOptions::scout().scout_initial_wave);
    let image = image(16, 8, false);
    for (pilot, parallel) in [(false, false), (false, true), (true, false)] {
        let mut options = options();
        options.scout_pilot_first = pilot;
        options.scout_parallel_finalists = parallel;
        let error = run_pixel_search::<_, _, ()>(
            &image,
            &params(0),
            &options,
            |_, _, _| panic!("invalid options must not encode"),
            |_, _| panic!("invalid options must not probe"),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput(message)
            if message == "initial wave requires pilot-first and parallel finalists"));
    }
}

#[test]
fn initial_wave_commits_canonical_baseline_ties_and_next_dispatch_bound() {
    let image = image(16, 8, false);
    let input = params(0);
    let mut options = options();
    options.full_limits.max_full_encodes = 4;
    options.full_limits.max_full_pixels = 4 * 128;
    let waves = Cell::new(0);
    let batch = Box::new(|_: &Image, tasks: &[FullTask<'_>]| {
        assert!(tasks.iter().all(|task| task.cooperative));
        let initial = waves.get() == 0;
        waves.set(waves.get() + 1);
        assert_eq!(tasks.len(), if initial { 3 } else { 1 });
        assert!(
            tasks
                .iter()
                .all(|task| task.budget.beat_bytes == if initial { None } else { Some(80) })
        );
        Ok(tasks
            .iter()
            .map(|task| {
                let id = task.params.rct.unwrap() as usize;
                if initial {
                    complete(if id == 1 { 100 } else { 80 }, id)
                } else {
                    TrialOutput::Bounded { lower_bound: 80 }
                }
            })
            .collect())
    });
    let mut driver = fixture(&image, &input, &options, Some(batch));
    assert!(driver.full_finalists(9, &[params(1)]).is_err());
    assert_eq!(
        driver
            .full_initial_wave(9, &[params(1), params(2), params(3)])
            .unwrap(),
        Some(vec![100, 80, 80])
    );
    assert_eq!(driver.full_finalists(9, &[params(4)]).unwrap(), [None]);
    let result = driver.finish().unwrap();
    assert_eq!(waves.get(), 2);
    assert_eq!(result.artifact, 2);
    assert_eq!(result.bytes, vec![2; 80]);
    assert_eq!(result.report.winner_index, 1);
    assert_eq!(result.report.full_encodes, 4);
    assert_eq!(result.report.full_pixels, 4 * 128);
    assert_eq!(
        result
            .report
            .trials
            .iter()
            .map(|trial| trial.baseline)
            .collect::<Vec<_>>(),
        [true, false, false, false]
    );
    assert_eq!(result.report.trials[3].lower_bound, Some(80));
    assert!(result.report.events.iter().any(|event| matches!(
        event,
        SearchEvent::Route {
            name: "adaptive-scout-initial-wave",
            candidates: 3
        }
    )));
}

#[test]
fn initial_wave_caps_pixels_total_scratch_workers_and_submitted_prefix() {
    for (width, height, alpha, workers, memory, expected) in [
        (1024, 512, false, 16, 1 << 30, 3),
        (1024, 513, false, 16, 1 << 30, 1),
        (1024, 512, true, 16, 1 << 30, 2),
        (1024, 512, false, 16, 160 << 20, 2),
        (1024, 512, false, 16, (160 << 20) - 1, 1),
        (16, 8, false, 1, 1 << 30, 1),
        (16, 8, false, 16, 1, 1),
    ] {
        let image = image(width, height, alpha);
        let input = params(0);
        let mut options = options();
        options.workers = WorkerLimits {
            tasks: workers,
            scratch_bytes: memory,
        };
        let calls = Cell::new(0);
        let batch = Box::new(|_: &Image, tasks: &[FullTask<'_>]| {
            calls.set(calls.get() + 1);
            assert_eq!(tasks.len(), expected);
            assert!(
                tasks
                    .iter()
                    .all(|task| task.cooperative && task.budget.beat_bytes.is_none())
            );
            Ok(tasks
                .iter()
                .map(|task| complete(100, task.params.rct.unwrap() as usize))
                .collect())
        });
        let mut driver = fixture(&image, &input, &options, Some(batch));
        assert_eq!(driver.initial_wave_capacity().unwrap(), expected);
        let sizes = driver
            .full_initial_wave(9, &[params(1), params(2), params(3), params(4)])
            .unwrap();
        if expected == 1 {
            assert!(sizes.is_none());
            assert!(driver.search.trials().is_empty());
            assert_eq!(calls.get(), 0);
        } else {
            assert_eq!(sizes, Some(vec![100; expected]));
            let result = driver.finish().unwrap();
            assert_eq!(result.report.full_encodes, expected);
            assert_eq!(
                result.report.full_pixels,
                expected as u64 * u64::from(width) * u64::from(height)
            );
            assert_eq!(result.artifact, 1);
            assert_eq!(calls.get(), 1);
        }
    }
}

#[test]
fn unsuitable_initial_waves_leave_all_work_for_the_original_path() {
    let image = image(16, 8, false);
    let input = params(0);
    for case in [
        "disabled",
        "no-callback",
        "empty",
        "single",
        "duplicate",
        "baseline",
        "nonbaseline",
    ] {
        let mut options = options();
        options.scout_initial_wave = case != "disabled";
        let batch: Option<FullBatchEncoder<'_, usize>> = if case == "no-callback" {
            None
        } else {
            Some(Box::new(|_, _| panic!("ineligible wave must not dispatch")))
        };
        let mut driver = fixture(&image, &input, &options, batch);
        let existing = matches!(case, "baseline" | "nonbaseline");
        if existing {
            driver
                .search
                .refine(
                    &image,
                    9,
                    input.clone(),
                    case == "baseline",
                    false,
                    &mut |_, _, _| Ok(complete(100, 0)),
                )
                .unwrap();
        }
        let candidates = match case {
            "empty" => vec![],
            "single" => vec![params(1)],
            "duplicate" => vec![params(1), params(1), params(2)],
            _ => vec![params(1), params(2), params(3)],
        };
        assert!(driver.full_initial_wave(9, &candidates).unwrap().is_none());
        assert_eq!(driver.search.trials().len(), usize::from(existing));
        if existing {
            assert_eq!(driver.finish().unwrap().report.full_encodes, 1);
        }
    }
}

#[test]
fn initial_wave_prechecks_all_resources_before_dispatch() {
    let image = image(16, 8, false);
    let input = params(0);
    for case in ["encodes", "pixels", "output", "stage", "params", "workers"] {
        let mut options = options();
        let mut stage = 9;
        let mut candidates = [params(1), params(2), params(3)];
        match case {
            "encodes" => options.full_limits.max_full_encodes = 2,
            "pixels" => options.full_limits.max_full_pixels = 3 * 128 - 1,
            "output" => options.full_limits.max_output_bytes = 0,
            "stage" => stage = 0,
            "params" => candidates[2].rct = Some(42),
            "workers" => options.workers.tasks = 0,
            _ => unreachable!(),
        }
        let mut driver = fixture(
            &image,
            &input,
            &options,
            Some(Box::new(|_, _| panic!("failed precheck must not dispatch"))),
        );
        assert!(
            driver.full_initial_wave(stage, &candidates).is_err(),
            "{case}"
        );
        assert!(driver.search.trials().is_empty());
    }
}

#[test]
fn every_failed_initial_wave_attempt_is_charged_without_partial_commit() {
    let image = image(16, 8, false);
    let input = params(0);
    for failure in [
        "count",
        "empty",
        "oversize",
        "leader-bound",
        "challenger-bound",
        "callback",
    ] {
        let mut options = options();
        options.full_limits = SearchLimits {
            max_full_encodes: 4,
            max_full_pixels: 4 * 128,
            max_output_bytes: 100,
        };
        let calls = Cell::new(0);
        let batch = Box::new(|_: &Image, tasks: &[FullTask<'_>]| {
            calls.set(calls.get() + 1);
            assert_eq!(tasks.len(), 3);
            assert!(tasks.iter().all(|task| task.budget.beat_bytes.is_none()));
            let mut outputs = vec![complete(90, 1), complete(80, 2), complete(70, 3)];
            match failure {
                "count" => {
                    outputs.pop();
                }
                "empty" => outputs[2] = complete(0, 3),
                "oversize" => outputs[2] = complete(101, 3),
                "leader-bound" => {
                    outputs[0] = TrialOutput::Bounded {
                        lower_bound: usize::MAX,
                    }
                }
                "challenger-bound" => {
                    outputs[2] = TrialOutput::Bounded {
                        lower_bound: usize::MAX,
                    }
                }
                "callback" => return Err(Error::Limit("injected initial failure")),
                _ => unreachable!(),
            }
            Ok(outputs)
        });
        let mut driver = fixture(&image, &input, &options, Some(batch));
        let candidates = [params(1), params(2), params(3)];
        let error = driver.full_initial_wave(9, &candidates).unwrap_err();
        if failure == "callback" {
            assert!(matches!(error, Error::Limit("injected initial failure")));
        }
        assert!(driver.search.trials().is_empty());
        assert!(driver.events.is_empty());
        assert!(driver.search.check_batch_work(&image, 1).is_ok());
        assert!(driver.search.check_batch_work(&image, 2).is_err());
        assert!(matches!(
            driver.full_initial_wave(9, &candidates),
            Err(Error::Limit("whole-frame search work"))
        ));
        driver
            .search
            .refine(&image, 9, input.clone(), true, false, &mut |_, _, _| {
                Ok(complete(50, 0))
            })
            .unwrap();
        let result = driver.finish().unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(result.report.full_encodes, 4);
        assert_eq!(result.report.full_pixels, 4 * 128);
        assert_eq!(result.report.trials.len(), 1);
        assert_eq!(result.artifact, 0);
    }
}
