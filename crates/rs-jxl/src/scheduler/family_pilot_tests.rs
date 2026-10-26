//! Synthetic scheduling checks, not codec compatibility or performance tests.
use super::tests::fixture;
use super::*;
use crate::scheduler::search::EncodedOutput;
use crate::{PixelData, PixelLayout};

fn complete(params: &CandidateParams, size: usize) -> TrialOutput<CandidateParams> {
    TrialOutput::Complete(EncodedOutput {
        bytes: vec![size as u8; size],
        artifact: params.clone(),
    })
}

fn assert_winner(result: &SearchResult<CandidateParams>, size: usize) {
    assert_eq!(result.bytes, vec![size as u8; size]);
    assert_eq!(result.artifact, result.params);
    let trial = &result.report.trials[result.report.winner_index];
    assert_eq!(trial.params, result.params);
    assert_eq!(trial.bytes, Some(size));
    assert_eq!(trial.lower_bound, None);
}

#[test]
fn family_tournament_keeps_original_rct_anchor_and_distinct_families() {
    for enabled in [false, true] {
        let (image, mut options) = fixture();
        options.scout_family_tournament = enabled;
        options.scout_representation_witness = true;
        options.plan.extended_rct = true;
        let input = CandidateParams::default().backend(9).unwrap();
        let learned = literal(&input, 9, 0.01).unwrap();
        let neighborhood = winner_neighborhood(&input, image.channel_count());
        let mut calls = Vec::new();
        let mut probes = Vec::new();
        let result = run_pixel_search(
            &image,
            &input,
            &options,
            |params, _, budget| {
                params.validate().unwrap();
                assert_eq!(params.speed_tier, 1);
                assert!(!calls.contains(params), "candidate was encoded twice");
                if calls.is_empty() {
                    assert_eq!(params.rct, Some(17));
                    assert_eq!(budget.beat_bytes, None);
                } else {
                    assert!(budget.beat_bytes.is_some());
                }
                calls.push(params.clone());
                Ok(complete(
                    params,
                    if params.rct == Some(6) { 40 } else { 100 },
                ))
            },
            |params, _| {
                probes.push(params.clone());
                Ok(match params.predictor {
                    Some(Predictor::Variable) => match params.rct {
                        Some(17) => 1000,
                        Some(10) => 1001,
                        Some(26) => 1002,
                        Some(6) => 1003,
                        Some(0) => 1004,
                        _ => panic!("unexpected RCT"),
                    },
                    None if params.group_size_shift.is_none() => 1005,
                    None => 1006,
                    Some(Predictor::Zero) => 1007,
                    _ => panic!("unexpected family"),
                })
            },
        )
        .unwrap();
        assert_eq!(result.report.full_encodes, if enabled { 5 } else { 8 });
        assert_eq!(result.report.trials.len(), calls.len());
        assert!(calls.contains(&input));
        assert!(calls.contains(&neighborhood));
        assert!(calls.contains(&learned));
        for rct in [0, 6, 10, 17, 26] {
            assert!(probes.iter().any(|p| p.rct == Some(rct)));
            assert_eq!(
                calls.iter().any(|p| p.rct == Some(rct)),
                !enabled || matches!(rct, 6 | 17)
            );
        }
        let frontier = result
            .report
            .events
            .iter()
            .find_map(|event| match event {
                SearchEvent::Frontier { candidates } => Some(candidates),
                _ => None,
            })
            .unwrap();
        assert_eq!(frontier.len(), calls.len());
        assert!(frontier.iter().all(|c| calls.contains(&c.params)));
        assert_eq!(result.params.predictor, Some(Predictor::Variable));
        assert_eq!(result.params.rct, Some(6));
        assert_winner(&result, 40);
    }
}

#[test]
fn wide_natural_without_competitive_zero_skips_representation_witnesses() {
    let image = Image::new(
        1024,
        512,
        PixelLayout::Rgb,
        PixelData::U16(vec![65535; 1024 * 512 * 3]),
    )
    .unwrap();
    for enabled in [false, true] {
        let (_, mut options) = fixture();
        options.coverage_scan = true;
        options.scout_representation_witness = true;
        options.scout_family_tournament = enabled;
        let scan = scan_lossless_input(&image, options.scan);
        assert!(scan.available() && scan.literal_candidate);
        assert!(image.bits() > 8);
        let input = CandidateParams::default().backend(9).unwrap();
        let learned = literal(&input, 9, 0.01).unwrap();
        let mut calls = Vec::new();
        let result = run_pixel_search(
            &image,
            &input,
            &options,
            |params, full, budget| {
                assert_eq!(full.bits(), 16);
                assert_eq!(params.speed_tier, 1);
                assert!(!calls.contains(params));
                if calls.is_empty() {
                    assert_eq!(params, &input);
                    assert_eq!(budget.beat_bytes, None);
                } else {
                    assert!(budget.beat_bytes.is_some());
                }
                calls.push(params.clone());
                Ok(complete(params, if *params == learned { 40 } else { 100 }))
            },
            |params, _| Ok(if *params == input { 100 } else { 300 }),
        )
        .unwrap();
        assert_eq!(result.report.full_encodes, if enabled { 1 } else { 3 });
        assert_eq!(
            calls
                .iter()
                .filter(|p| p.predictor == Some(Predictor::Zero))
                .count(),
            if enabled { 0 } else { 2 }
        );
        assert_eq!(
            result.report.events.iter().any(|event| matches!(
                event,
                SearchEvent::Route {
                    name: "adaptive-scout-representation-witness",
                    ..
                }
            )),
            !enabled
        );
        assert_eq!(result.params, if enabled { input } else { learned });
        assert_winner(&result, if enabled { 100 } else { 40 });
    }
}

#[test]
fn mixed_flat_eight_bit_confirms_learned_literal_despite_poor_probes() {
    let image = Image::new(
        1024,
        512,
        PixelLayout::Gray,
        PixelData::U8(
            (0..512)
                .flat_map(|y| {
                    (0..1024).map(move |x| {
                        if x < 512 {
                            0
                        } else {
                            ((x + y) % 251 + 1) as u8
                        }
                    })
                })
                .collect(),
        ),
    )
    .unwrap();
    for enabled in [false, true] {
        for bounded in [false, true] {
            let (_, mut options) = fixture();
            options.coverage_scan = true;
            options.scout_family_tournament = enabled;
            let scan = scan_lossless_input(&image, options.scan);
            assert!(scan.available() && scan.sampled_pixels != 0);
            assert!(scan.sampled_flat * 2 >= scan.sampled_pixels);
            assert!(scan.sampled_flat * 4 < scan.sampled_pixels * 3);
            let input = CandidateParams::default().backend(9).unwrap();
            let learned = literal(&input, 9, 0.01).unwrap();
            let mut calls = Vec::new();
            let result = run_pixel_search(
                &image,
                &input,
                &options,
                |params, _, budget| {
                    assert_eq!(params.speed_tier, 1);
                    assert!(!calls.contains(params));
                    calls.push(params.clone());
                    if *params == input {
                        assert_eq!(budget.beat_bytes, None);
                        Ok(complete(params, 100))
                    } else {
                        assert_eq!(params, &learned);
                        assert_eq!(params.learning_fraction, 0.01);
                        assert_eq!(budget.beat_bytes, Some(100));
                        Ok(if bounded {
                            TrialOutput::Bounded { lower_bound: 100 }
                        } else {
                            complete(params, 40)
                        })
                    }
                },
                |params, _| Ok(if *params == input { 100 } else { 1000 }),
            )
            .unwrap();
            assert_eq!(result.report.full_encodes, if enabled { 2 } else { 1 });
            assert_eq!(calls.contains(&learned), enabled);
            let literal_wins = enabled && !bounded;
            assert_eq!(result.params, if literal_wins { learned } else { input });
            assert_winner(&result, if literal_wins { 40 } else { 100 });
            if enabled && bounded {
                assert_eq!(result.report.trials[1].bytes, None);
                assert_eq!(result.report.trials[1].lower_bound, Some(100));
            }
        }
    }
}
