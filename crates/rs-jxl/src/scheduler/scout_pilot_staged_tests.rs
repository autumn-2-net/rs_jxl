use super::tests::{assert_retained, complete, family};
use super::*;
use crate::{PixelData, PixelLayout};
use std::cell::Cell;

const WIDTH: usize = 1024;
const COARSE_PIXELS: usize = 3 * WIDTH * 8;
const FINE_PIXELS: usize = 3 * WIDTH * 32;
const TWO_STAGE_PIXELS: u64 = (5 * COARSE_PIXELS + 2 * FINE_PIXELS) as u64;
const COARSE_SCORES: [usize; 5] = [1000, 1100, 2000, 3000, 4000];
const FINE_SCORES: [usize; 5] = [1000, 1005, 2000, 3000, 4000];

fn natural_fixture() -> (Image, DriverOptions) {
    let image = Image::new(
        WIDTH as u32,
        512,
        PixelLayout::Rgb,
        PixelData::U8(
            (0..WIDTH * 512)
                .flat_map(|i| [i as u8, (i >> 8) as u8, (i >> 16) as u8])
                .collect(),
        ),
    )
    .unwrap();
    let mut options = DriverOptions::scout();
    options.scout_pilot_first = true;
    options.scout_staged_probes = true;
    options.scout_pilot_consensus = true;
    options.screen.regions.full_width_strips = true;
    options.screen.regions.strip_rows = Some(32);
    // Each stage fits, but retaining both sample sets would exceed this cap.
    options.screen.limits.max_resident_sample_pixels = FINE_PIXELS;
    let scan = scan_lossless_input(&image, options.scan);
    assert!(scan.available() && scan.likely_continuous_tone() && !scan.literal_candidate);
    for rows in [8, 32] {
        let mut regions = options.screen.regions.clone();
        regions.strip_rows = Some(rows);
        let specs =
            crate::scheduler::regions::screen_regions(&image, Some(&scan), &regions).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(
            specs.iter().map(|s| s.pixels().unwrap()).sum::<usize>(),
            3 * WIDTH * rows
        );
    }
    (image, options)
}

#[derive(Default)]
struct ProbeLog {
    coarse: Cell<usize>,
    fine: Cell<usize>,
    pixels: Cell<u64>,
}

impl ProbeLog {
    fn encodes(&self) -> usize {
        self.coarse.get() + self.fine.get()
    }
}

fn run_fixture(
    image: &Image,
    options: &DriverOptions,
    effort: u8,
    batched: bool,
    log: &ProbeLog,
    scores: [[usize; 5]; 2],
    full: impl FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<(usize, i8)>>,
) -> Result<SearchResult<(usize, i8)>> {
    let score = |params: &CandidateParams, sample: &Image| {
        assert_eq!(params.speed_tier, 1);
        assert_eq!(sample.width() as usize, WIDTH);
        let stage = match sample.height() {
            8 => {
                log.coarse.set(log.coarse.get() + 1);
                0
            }
            32 => {
                log.fine.set(log.fine.get() + 1);
                1
            }
            other => panic!("unexpected pilot sample height: {other}"),
        };
        log.pixels
            .set(log.pixels.get() + u64::from(sample.width()) * u64::from(sample.height()));
        Ok(scores[stage][family(params)])
    };
    let mut batches = BatchEncoders::default();
    if batched {
        batches.planned_probe = Some(Box::new(|tasks, _| {
            tasks
                .iter()
                .map(|task| score(task.params, task.image))
                .collect()
        }));
    }
    run_pixel_search_with_batches(
        image,
        &CandidateParams::default().backend(effort)?,
        options,
        full,
        |params, sample| {
            assert!(!batched, "planned batch must not fall back to serial");
            score(params, sample)
        },
        batches,
    )
}

#[test]
fn decisive_coarse_regions_skip_fine_but_require_complete_output() {
    let (image, mut options) = natural_fixture();
    options.scout_coarse_consensus = true;
    options.screen.limits.max_encodes = 15;
    options.screen.limits.max_encoded_pixels = (5 * COARSE_PIXELS) as u64;
    for batched in [false, true] {
        let log = ProbeLog::default();
        let result = run_fixture(
            &image,
            &options,
            9,
            batched,
            &log,
            [COARSE_SCORES, FINE_SCORES],
            |params, _, budget| {
                assert_eq!(family(params), 0);
                assert!(budget.beat_bytes.is_none());
                Ok(complete(params, 123))
            },
        )
        .unwrap();
        assert_eq!((log.coarse.get(), log.fine.get()), (15, 0));
        assert_eq!(result.report.full_encodes, 1);
        assert_retained(&result, 0, 1, 123);
        let log = ProbeLog::default();
        assert!(
            run_fixture(
                &image,
                &options,
                9,
                batched,
                &log,
                [COARSE_SCORES, FINE_SCORES],
                |_, _, _| { Ok(TrialOutput::Bounded { lower_bound: 123 }) }
            )
            .is_err()
        );
    }
}

#[test]
fn coarse_confidence_requires_all_regions_to_clear_one_percent() {
    assert!(coarse_dominated(&[1010, 1010, 1010], &[1000; 3]));
    assert!(!coarse_dominated(&[1010, 1009, 1010], &[1000; 3]));
    assert!(!coarse_dominated(&[1100, 1000, 1100], &[1000; 3]));
    assert!(!coarse_dominated(&[1010; 2], &[1000; 2]));
    assert!(!coarse_dominated(&[1010; 3], &[0, 1000, 1000]));
    let (image, mut options) = natural_fixture();
    options.scout_coarse_consensus = true;
    let log = ProbeLog::default();
    run_fixture(
        &image,
        &options,
        9,
        false,
        &log,
        [[1000, 1009, 2000, 3000, 4000], FINE_SCORES],
        |params, _, _| Ok(complete(params, 100)),
    )
    .unwrap();
    assert_eq!(log.fine.get(), 6);
}

#[test]
fn coarse_consensus_filters_clear_cross_rivals_but_refines_ambiguous_natural_routes() {
    let (image, mut options) = natural_fixture();
    options.scout_coarse_consensus = true;
    let log = ProbeLog::default();
    run_fixture(
        &image,
        &options,
        9,
        false,
        &log,
        [[1000, 1009, 2000, 3000, 1015], FINE_SCORES],
        |params, _, _| Ok(complete(params, 100)),
    )
    .unwrap();
    assert_eq!(log.coarse.get(), 15);
    assert_eq!(log.fine.get(), 6);
}

#[test]
fn staged_consensus_preserves_both_rounds_in_cumulative_work() {
    let (image, mut options) = natural_fixture();
    options.screen.limits.max_encodes = 21;
    options.screen.limits.max_encoded_pixels = TWO_STAGE_PIXELS;
    for batched in [false, true] {
        let log = ProbeLog::default();
        let result = run_fixture(
            &image,
            &options,
            9,
            batched,
            &log,
            [COARSE_SCORES, FINE_SCORES],
            |params, _, budget| {
                assert_eq!(family(params), 0);
                assert_eq!(budget.beat_bytes, None);
                Ok(complete(params, 100))
            },
        )
        .unwrap();
        assert_eq!((log.coarse.get(), log.fine.get()), (15, 6));
        assert_eq!(result.report.probe_encodes, log.encodes());
        assert_eq!(result.report.probe_pixels, log.pixels.get());
        assert_eq!(result.report.probe_pixels, TWO_STAGE_PIXELS);
        assert_eq!(result.report.screen_batches, 2);
        assert_eq!(result.report.full_encodes, 1);
        assert_retained(&result, 0, 1, 100);
    }
}

#[test]
fn natural_consensus_rejects_consistently_losing_cross_channel_full_work() {
    let (image, options) = natural_fixture();
    for batched in [false, true] {
        let log = ProbeLog::default();
        let result = run_fixture(
            &image,
            &options,
            9,
            batched,
            &log,
            [
                [1000, 1009, 2000, 3000, 1008],
                [1000, 1005, 2000, 3000, 1010],
            ],
            |params, _, _| {
                assert_eq!(family(params), 0);
                Ok(complete(params, 100))
            },
        )
        .unwrap();
        assert_eq!(result.report.full_encodes, 1);
        assert_eq!(log.fine.get(), 9);
        assert_retained(&result, 0, 1, 100);
    }
}

#[test]
fn staged_consensus_enforces_cumulative_and_resident_limits_before_full() {
    let (image, options) = natural_fixture();
    for batched in [false, true] {
        for limit in ["encodes", "pixels", "resident"] {
            let mut options = options.clone();
            match limit {
                "encodes" => options.screen.limits.max_encodes = 20,
                "pixels" => options.screen.limits.max_encoded_pixels = TWO_STAGE_PIXELS - 1,
                "resident" => options.screen.limits.max_resident_sample_pixels = FINE_PIXELS - 1,
                _ => unreachable!(),
            }
            let log = ProbeLog::default();
            let error = run_fixture(
                &image,
                &options,
                9,
                batched,
                &log,
                [COARSE_SCORES, FINE_SCORES],
                |_, _, _| panic!("failed pilot must not dispatch full candidates"),
            )
            .unwrap_err();
            assert!(
                matches!(error, Error::Limit(message) if message == if limit == "resident" {
                    "resident probe pixels"
                } else {
                    "cumulative scheduler probe work"
                })
            );
            assert_eq!(log.coarse.get(), 15);
            let fine = if batched || limit == "resident" { 0 } else { 5 };
            assert_eq!(log.fine.get(), fine);
            assert_eq!(
                log.pixels.get(),
                (5 * COARSE_PIXELS + fine * WIDTH * 32) as u64
            );
        }
    }
}

#[test]
fn staged_consensus_bounded_challenger_does_not_advance_probe_reference() {
    let (image, options) = natural_fixture();
    for batched in [false, true] {
        let log = ProbeLog::default();
        let mut calls = Vec::new();
        let result = run_fixture(
            &image,
            &options,
            9,
            batched,
            &log,
            [
                [1000, 1000, 1000, 1000, 4000],
                [1000, 1002, 1026, 1027, 4000],
            ],
            |params, _, budget| {
                let id = family(params);
                calls.push(id);
                match id {
                    0 => {
                        assert_eq!(budget.beat_bytes, None);
                        Ok(complete(params, 100))
                    }
                    1 => {
                        assert_eq!(budget.beat_bytes, Some(100));
                        Ok(TrialOutput::Bounded { lower_bound: 137 })
                    }
                    _ => panic!("bounded challenger must not advance the reference"),
                }
            },
        )
        .unwrap();
        assert_eq!(calls, [0, 1]);
        assert_eq!((log.coarse.get(), log.fine.get()), (15, 12));
        assert_eq!(result.report.full_encodes, 2);
        assert_retained(&result, 0, 1, 100);
        assert_eq!(result.report.trials[1].bytes, None);
        assert_eq!(result.report.trials[1].lower_bound, Some(100));
    }
}

#[test]
fn staged_consensus_rejects_incomplete_baseline_and_insufficient_bound() {
    let (image, options) = natural_fixture();
    for batched in [false, true] {
        for incomplete_baseline in [false, true] {
            let log = ProbeLog::default();
            let mut calls = 0;
            let error = run_fixture(
                &image,
                &options,
                9,
                batched,
                &log,
                [COARSE_SCORES, [1000, 1002, 2000, 3000, 4000]],
                |params, _, budget| {
                    calls += 1;
                    if family(params) == 0 {
                        assert_eq!(budget.beat_bytes, None);
                        return Ok(if incomplete_baseline {
                            TrialOutput::Bounded { lower_bound: 1000 }
                        } else {
                            complete(params, 100)
                        });
                    }
                    assert_eq!(family(params), 1);
                    assert_eq!(budget.beat_bytes, Some(100));
                    Ok(TrialOutput::Bounded { lower_bound: 99 })
                },
            )
            .unwrap_err();
            assert_eq!(calls, if incomplete_baseline { 1 } else { 2 });
            assert!(error.to_string().contains(if incomplete_baseline {
                "baseline/calibration callback returned an incomplete output"
            } else {
                "callback byte bound does not reach the incumbent"
            }));
        }
    }
}

#[test]
fn e10_with_staged_consensus_keeps_full_families_despite_poor_probe_scores() {
    let (image, options) = natural_fixture();
    for batched in [false, true] {
        for effort in [9, 10] {
            let log = ProbeLog::default();
            let mut families = Vec::new();
            let result = run_fixture(
                &image,
                &options,
                effort,
                batched,
                &log,
                [COARSE_SCORES, FINE_SCORES],
                |params, _, _| {
                    let size = if params.speed_tier == 0 {
                        200
                    } else {
                        families.push(family(params));
                        [100, 110, 120, 130, 80][family(params)]
                    };
                    Ok(complete(params, size))
                },
            )
            .unwrap();
            if effort == 9 {
                assert_eq!(families, [0]);
                assert_eq!((log.coarse.get(), log.fine.get()), (15, 6));
                assert_retained(&result, 0, 1, 100);
            } else {
                assert_eq!(families, [0, 1, 2, 3, 4]);
                assert_eq!((log.coarse.get(), log.fine.get()), (0, 15));
                assert_eq!(result.report.screen_batches, 1);
                assert_eq!(result.report.probe_encodes, 15);
                assert_eq!(result.report.probe_pixels, (5 * FINE_PIXELS) as u64);
                assert_eq!(result.report.full_encodes, 7);
                assert_retained(&result, 4, 1, 80);
            }
        }
    }
}
