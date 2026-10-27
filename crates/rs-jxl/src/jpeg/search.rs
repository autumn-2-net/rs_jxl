// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! JPEG branch of enc_lossless_search.cc::EncodeLosslessSearch at b866b3d.
//! JPEG has no pixel probes/literal family. Only complete coefficient encodes
//! can win; parsed coefficients, image metadata and reconstruction are reused.

use super::{TranscodeOptions, archival, transcode};
use crate::{
    Error, Result,
    entropy::options::SpeedTier,
    scheduler::{
        params::CandidateParams,
        search::{EncodedOutput, RetainedOutput, Search, TrialBudget, TrialOutput},
    },
};

pub use crate::scheduler::search::SearchLimits;

#[derive(Clone, Debug)]
pub struct Trial {
    pub stage: u8,
    pub backend_effort: u8,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct SearchResult {
    pub data: Vec<u8>,
    pub backend_effort: u8,
    pub winner_index: usize,
    pub trials: Vec<Trial>,
    /// Sum of full image pixels processed across completed candidates.
    pub full_pixels: u64,
}

fn run<F>(
    effort: u8,
    pixels: u64,
    limits: SearchLimits,
    mut encode: F,
) -> Result<RetainedOutput<u8>>
where
    F: FnMut(u8, TrialBudget) -> Result<Vec<u8>>,
{
    if !(1..=11).contains(&effort) {
        return Err(Error::InvalidInput(
            "JPEG search effort must be 1..=11".into(),
        ));
    }
    let mut search = Search::new(limits);
    let mut encoder = |params: &CandidateParams, budget| {
        let backend = (10 - params.speed_tier) as u8;
        Ok(TrialOutput::Complete(EncodedOutput {
            bytes: encode(backend, budget)?,
            artifact: backend,
        }))
    };
    let mut regular = None;
    for (stage, backend) in [(1, 3), (3, 5), (5, 7), (7, 8), (9, 9), (10, 10)] {
        if stage > effort.min(10) {
            break;
        }
        if stage > 3 && !search.competitive(regular, stage, 1.35) {
            continue;
        }
        let size = search.refine_pixels(
            pixels,
            stage,
            CandidateParams::default().backend(backend)?,
            true,
            false,
            &mut encoder,
        )?;
        regular = regular.into_iter().chain(size).min();
    }
    search.finish()
}

/// Search the frozen optimized encoder's JPEG effort prefix. This is separate
/// from the direct `jpeg::transcode` backend. Effort 11 follows source JPEG
/// frame normalization to level 10, not the pixel-only Tectonic tournament.
/// Decoding-speed overrides are ineligible for source search and are rejected.
/// No encoded-file hashes, timeout fallbacks or sampled scores select a winner.
pub fn transcode_search(
    input: &[u8],
    options: &TranscodeOptions,
    limits: SearchLimits,
) -> Result<SearchResult> {
    let (jpeg, settings, limits) = prepare_search(input, options, limits)?;
    transcode::with_threads(&settings, |pool| {
        encode_search(&jpeg, options, &settings, limits, pool)
    })
}

pub(crate) fn transcode_search_in_pool(
    input: &[u8],
    options: &TranscodeOptions,
    limits: SearchLimits,
    pool: Option<&rayon::ThreadPool>,
) -> Result<SearchResult> {
    let (jpeg, settings, limits) = prepare_search(input, options, limits)?;
    transcode::with_pool(&settings, pool, |pool| {
        encode_search(&jpeg, options, &settings, limits, pool)
    })
}

fn prepare_search(
    input: &[u8],
    options: &TranscodeOptions,
    mut limits: SearchLimits,
) -> Result<(super::data::JpegData, transcode::Options, SearchLimits)> {
    let mut settings = options.resolve()?;
    if options.decoding_speed != 0 {
        return Err(Error::InvalidInput(
            "source JPEG search requires decoding_speed=0; use transcode for an explicit backend"
                .into(),
        ));
    }
    if limits.max_full_encodes == 0 || limits.max_full_pixels == 0 || limits.max_output_bytes == 0 {
        return Err(Error::Limit("JPEG search budgets must be nonzero"));
    }
    limits.max_output_bytes = limits.max_output_bytes.min(settings.max_output_bytes);
    settings.max_output_bytes = limits.max_output_bytes;
    let jpeg = archival::read_input(input, &options.limits.input)?;
    let pixels = u64::from(jpeg.width) * u64::from(jpeg.height);
    if pixels > limits.max_full_pixels {
        return Err(Error::Limit("whole-frame search work"));
    }
    Ok((jpeg, settings, limits))
}

fn encode_search(
    jpeg: &super::data::JpegData,
    options: &TranscodeOptions,
    settings: &transcode::Options,
    limits: SearchLimits,
    pool: Option<&rayon::ThreadPool>,
) -> Result<SearchResult> {
    let pixels = u64::from(jpeg.width) * u64::from(jpeg.height);
    let prepared = transcode::Prepared::new(jpeg, settings)?;
    let mut ac_cache = transcode::SearchCache::new(jpeg);
    let retained = run(options.effort, pixels, limits, |effort, budget| {
        let candidate = transcode::Options {
            tier: SpeedTier::try_from(10 - effort as i8)?,
            max_output_bytes: budget.max_output_bytes,
            ..settings.clone()
        };
        // e1/e2 have only one trial: no AC reuse can repay serialization.
        if options.effort < 3 {
            prepared.encode(jpeg, &candidate, pool)
        } else {
            prepared.encode_with_cache(jpeg, &candidate, pool, &mut ac_cache)
        }
    })?;
    Ok(SearchResult {
        data: retained.bytes,
        backend_effort: retained.artifact,
        winner_index: retained.winner_index,
        trials: retained
            .trials
            .into_iter()
            .map(|trial| Trial {
                stage: trial.stage,
                backend_effort: (10 - trial.params.speed_tier) as u8,
                bytes: trial.bytes.expect("JPEG candidates must complete"),
            })
            .collect(),
        full_pixels: retained.full_pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_trial_search_keeps_backend_three_and_original_jpeg() {
        let input = crate::jpeg::transcode_reference_tests::fixtures::fixtures()
            .into_iter()
            .find(|(name, _)| name == "rgb_seq_33x25")
            .unwrap()
            .1;
        for effort in [1, 2] {
            let options = TranscodeOptions {
                effort,
                ..Default::default()
            };
            let result = transcode_search(&input, &options, SearchLimits::default()).unwrap();
            assert_eq!(result.backend_effort, 3);
            assert_eq!(result.trials.len(), 1);
            assert_eq!(result.trials[0].stage, 1);
            assert_eq!(
                crate::decode::reconstruct_jpeg(&result.data, &Default::default()).unwrap(),
                input,
            );
        }
    }

    #[test]
    fn search_cache_preserves_complete_trial_ledger_and_winner() {
        let input = crate::jpeg::transcode_reference_tests::fixtures::fixtures()
            .into_iter()
            .find(|(name, _)| name == "rgb_seq_33x25")
            .unwrap()
            .1;
        let options = TranscodeOptions {
            effort: 11,
            threads: 2,
            ..Default::default()
        };
        let settings = options.resolve().unwrap();
        let jpeg = archival::read_input(&input, &options.limits.input).unwrap();
        let pixels = u64::from(jpeg.width) * u64::from(jpeg.height);
        let expected = transcode::with_threads(&settings, |pool| {
            let prepared = transcode::Prepared::new(&jpeg, &settings)?;
            run(
                options.effort,
                pixels,
                SearchLimits::default(),
                |effort, budget| {
                    let candidate = transcode::Options {
                        tier: SpeedTier::try_from(10 - effort as i8)?,
                        max_output_bytes: budget.max_output_bytes.min(settings.max_output_bytes),
                        ..settings.clone()
                    };
                    prepared.encode(&jpeg, &candidate, pool)
                },
            )
        })
        .unwrap();
        let actual = transcode_search(&input, &options, SearchLimits::default()).unwrap();
        assert_eq!(actual.data, expected.bytes);
        assert_eq!(actual.backend_effort, expected.artifact);
        assert_eq!(actual.winner_index, expected.winner_index);
        assert_eq!(actual.full_pixels, expected.full_pixels);
        assert_eq!(actual.trials.len(), expected.full_encodes);
        assert_eq!(actual.trials.len(), expected.trials.len());
        for (actual, expected) in actual.trials.iter().zip(&expected.trials) {
            assert_eq!(actual.stage, expected.stage);
            assert_eq!(
                actual.backend_effort,
                (10 - expected.params.speed_tier) as u8
            );
            assert_eq!(Some(actual.bytes), expected.bytes);
        }
        assert_eq!(
            crate::decode::reconstruct_jpeg(&actual.data, &Default::default()).unwrap(),
            input
        );
    }

    #[test]
    fn every_effort_is_the_source_prefix_and_retains_the_smallest_output() {
        let schedule = [(1, 3), (3, 5), (5, 7), (7, 8), (9, 9), (10, 10)];
        for effort in 1..=11 {
            let mut calls = Vec::new();
            let result = run(effort, 123, SearchLimits::default(), |backend, budget| {
                assert!(budget.beat_bytes.is_none()); // source JPEG baselines never abort
                calls.push(backend);
                // Regressions and ties must not displace an earlier smaller result.
                let sizes = [0, 0, 0, 100, 0, 90, 0, 94, 90, 88, 99];
                Ok(vec![backend; sizes[backend as usize]])
            })
            .unwrap();
            let expected: Vec<_> = schedule
                .into_iter()
                .filter(|(s, _)| *s <= effort.min(10))
                .collect();
            assert_eq!(calls, expected.iter().map(|(_, e)| *e).collect::<Vec<_>>());
            assert_eq!(result.full_encodes, calls.len());
            assert_eq!(result.full_pixels, 123 * calls.len() as u64);
            assert_eq!(
                result.trials.iter().map(|t| t.stage).collect::<Vec<_>>(),
                expected.iter().map(|(s, _)| *s).collect::<Vec<_>>()
            );
            let winner = if effort >= 9 {
                9
            } else if effort >= 3 {
                5
            } else {
                3
            };
            assert_eq!(result.artifact, winner);
            assert!(result.bytes.iter().all(|&byte| byte == winner));
        }
    }

    #[test]
    fn limits_fail_instead_of_returning_a_partial_search_winner() {
        for limits in [
            SearchLimits {
                max_full_encodes: 1,
                ..Default::default()
            },
            SearchLimits {
                max_full_pixels: 123,
                ..Default::default()
            },
            SearchLimits {
                max_output_bytes: 1,
                ..Default::default()
            },
        ] {
            assert!(matches!(
                run(9, 123, limits, |_, _| Ok(vec![1; 10])),
                Err(Error::Limit(_))
            ));
        }
        let mut calls = 0;
        let error = run(9, 123, SearchLimits::default(), |_, _| {
            calls += 1;
            if calls == 2 {
                Err(Error::Limit("sample cap"))
            } else {
                Ok(vec![1; 10])
            }
        })
        .unwrap_err();
        assert!(matches!(error, Error::Limit("sample cap")));
        assert_eq!(calls, 2);
    }

    #[test]
    fn invalid_options_or_empty_geometry_never_call_the_encoder() {
        for (effort, pixels) in [(0, 10), (12, 10), (1, 0)] {
            assert!(
                run(effort, pixels, SearchLimits::default(), |_, _| panic!(
                    "must not encode"
                ))
                .is_err()
            );
        }
        let options = TranscodeOptions {
            decoding_speed: 1,
            ..Default::default()
        };
        assert!(matches!(
            transcode_search(&[], &options, SearchLimits::default()),
            Err(Error::InvalidInput(_))
        ));
    }
}
