// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! `enc_lossless_search.cc::Search`: completed sizes, real-byte bounds,
//! stable output retention and related-family lookup. No encoder dependencies.

use super::params::CandidateParams;
use crate::{Error, Image, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrialBudget {
    /// A challenger only matters if it is strictly smaller than this incumbent.
    /// None for baselines/calibrations: their complete sizes are required.
    pub beat_bytes: Option<usize>,
    /// Resource ceiling, independent of the incumbent and search policy.
    pub max_output_bytes: usize,
}

#[derive(Debug)]
pub struct EncodedOutput<A = ()> {
    pub bytes: Vec<u8>,
    /// Resolved backend options, transcript, statistics, etc. from THIS encode.
    pub artifact: A,
}

#[derive(Debug)]
pub enum TrialOutput<A = ()> {
    Complete(EncodedOutput<A>),
    /// Must follow from emitted bytes or a proven encoder byte lower bound.
    /// A sample score, estimate or cancellation is never a valid bound here.
    Bounded {
        lower_bound: usize,
    },
}

#[derive(Clone, Debug)]
pub struct TrialRecord {
    pub stage: u8,
    pub params: CandidateParams,
    pub baseline: bool,
    pub calibration: bool,
    pub bytes: Option<usize>,
    pub lower_bound: Option<usize>,
}
impl TrialRecord {
    pub fn outcome_size(&self) -> Option<usize> {
        self.bytes.into_iter().chain(self.lower_bound).min()
    }
}

/// Caller-selected resource ceilings. Defaults impose no policy cap;
/// checked accounting and addressable output sizes still apply.
#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    pub max_full_encodes: usize,
    pub max_full_pixels: u64,
    pub max_output_bytes: usize,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_full_encodes: usize::MAX,
            max_full_pixels: u64::MAX,
            max_output_bytes: usize::MAX,
        }
    }
}

#[derive(Debug)]
pub struct RetainedOutput<A = ()> {
    pub bytes: Vec<u8>,
    pub params: CandidateParams,
    pub artifact: A,
    pub winner_index: usize,
    pub trials: Vec<TrialRecord>,
    pub full_encodes: usize,
    pub full_pixels: u64,
}

/// Only the incumbent buffer and current encoder output are retained. Losing
/// outputs/artifacts are dropped immediately. Invocation is serial and stable.
pub struct Search<A = ()> {
    trials: Vec<TrialRecord>,
    best: Option<(usize, EncodedOutput<A>)>,
    limits: SearchLimits,
    full_encodes: usize,
    full_pixels: u64,
}
impl<A> Search<A> {
    pub fn new(limits: SearchLimits) -> Self {
        Self {
            trials: Vec::new(),
            best: None,
            limits,
            full_encodes: 0,
            full_pixels: 0,
        }
    }
    pub fn trials(&self) -> &[TrialRecord] {
        &self.trials
    }
    pub(super) fn check_batch_work(&self, image: &Image, count: usize) -> Result<()> {
        let pixels = (u64::from(image.width()) * u64::from(image.height()))
            .checked_mul(count as u64)
            .ok_or(Error::Limit("whole-frame search work"))?;
        if self
            .full_encodes
            .checked_add(count)
            .is_none_or(|n| n > self.limits.max_full_encodes)
            || self
                .full_pixels
                .checked_add(pixels)
                .is_none_or(|n| n > self.limits.max_full_pixels)
            || self.limits.max_output_bytes == 0
        {
            return Err(Error::Limit("whole-frame search work"));
        }
        Ok(())
    }
    pub fn best_index_before(&self, stage: u8, baseline_only: bool) -> Option<usize> {
        self.trials
            .iter()
            .enumerate()
            .filter(|(_, t)| t.stage < stage && (!baseline_only || t.baseline) && t.bytes.is_some())
            .min_by_key(|(_, t)| t.bytes)
            .map(|(i, _)| i)
    }
    pub fn best_before(&self, stage: u8, baseline_only: bool) -> Option<usize> {
        self.best_index_before(stage, baseline_only)
            .and_then(|i| self.trials[i].bytes)
    }
    pub fn best_params_before(&self, stage: u8) -> Option<&CandidateParams> {
        self.best_index_before(stage, false)
            .map(|i| &self.trials[i].params)
    }
    pub fn competitive(&self, size: Option<usize>, stage: u8, tolerance: f64) -> bool {
        match (size, self.best_before(stage, true)) {
            (Some(size), Some(best)) => size as f64 <= best as f64 * tolerance,
            _ => false,
        }
    }
    pub fn find(&self, params: &CandidateParams) -> Option<usize> {
        self.trials.iter().position(|t| t.params == *params)
    }
    pub fn related(&self, params: &CandidateParams) -> Option<usize> {
        self.trials
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                !t.calibration
                    && t.params.predictor == params.predictor
                    && (t.params.palette_colors == 0) == (params.palette_colors == 0)
                    && (t.params.learning_fraction == 0.0) == (params.learning_fraction == 0.0)
                    && t.outcome_size().is_some()
            })
            .min_by_key(|(_, t)| t.outcome_size())
            .map(|(i, _)| i)
    }
    pub fn lower_backend(&self, params: &CandidateParams) -> Option<usize> {
        self.trials
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                if t.calibration || t.bytes.is_none() || t.params.speed_tier <= params.speed_tier {
                    return false;
                }
                let mut comparable = t.params.clone();
                comparable.speed_tier = params.speed_tier;
                comparable == *params
            })
            .min_by_key(|(_, t)| t.params.speed_tier)
            .map(|(i, _)| i)
    }
    pub fn palette_calibration_peer(&self, params: &CandidateParams) -> Option<usize> {
        if params.palette_colors == 0 {
            return None;
        }
        self.trials
            .iter()
            .enumerate()
            .rev()
            .find(|(_, t)| {
                if !t.calibration
                    || t.bytes.is_none()
                    || t.params.palette_colors == 0
                    || t.params.palette_colors == params.palette_colors
                {
                    return false;
                }
                let mut comparable = t.params.clone();
                comparable.palette_colors = params.palette_colors;
                comparable == *params
            })
            .map(|(i, _)| i)
    }
    /// Port of Refine/RunPending. Baseline repeats remain exact; non-baselines
    /// reuse prior complete or bounded outcomes because the incumbent shrinks.
    pub fn refine<F>(
        &mut self,
        image: &Image,
        stage: u8,
        params: CandidateParams,
        baseline: bool,
        calibration: bool,
        encode: &mut F,
    ) -> Result<Option<usize>>
    where
        F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    {
        self.refine_pixels(
            u64::from(image.width()) * u64::from(image.height()),
            stage,
            params,
            baseline,
            calibration,
            &mut |params, budget| encode(params, image, budget),
        )
    }
    /// Coefficient-only inputs share the ledger without constructing pixels.
    pub(crate) fn refine_pixels<F>(
        &mut self,
        pixels: u64,
        stage: u8,
        params: CandidateParams,
        baseline: bool,
        calibration: bool,
        encode: &mut F,
    ) -> Result<Option<usize>>
    where
        F: FnMut(&CandidateParams, TrialBudget) -> Result<TrialOutput<A>>,
    {
        params.validate()?;
        if !(1..=10).contains(&stage) {
            return Err(Error::InvalidInput("search stage must be 1..=10".into()));
        }
        if !baseline && let Some(i) = self.find(&params) {
            return Ok(self.trials[i].bytes);
        }
        if pixels == 0 {
            return Err(Error::InvalidInput("empty search image".into()));
        }
        if self.full_encodes >= self.limits.max_full_encodes
            || self
                .full_pixels
                .checked_add(pixels)
                .is_none_or(|n| n > self.limits.max_full_pixels)
        {
            return Err(Error::Limit("whole-frame search work"));
        }
        if self.limits.max_output_bytes == 0 {
            return Err(Error::Limit("whole-frame output bytes"));
        }
        let beat_bytes = if !baseline && !calibration {
            self.best.as_ref().map(|(_, output)| output.bytes.len())
        } else {
            None
        };
        let budget = TrialBudget {
            beat_bytes,
            max_output_bytes: self.limits.max_output_bytes,
        };
        self.full_encodes += 1;
        self.full_pixels += pixels;
        let output = encode(&params, budget)?;
        let mut record = TrialRecord {
            stage,
            params,
            baseline,
            calibration,
            bytes: None,
            lower_bound: None,
        };
        let index = self.trials.len();
        match output {
            TrialOutput::Bounded { lower_bound } => {
                let Some(limit) = beat_bytes else {
                    return Err(Error::Encoding(
                        "baseline/calibration callback returned an incomplete output".into(),
                    ));
                };
                if lower_bound < limit {
                    return Err(Error::Encoding(
                        "callback byte bound does not reach the incumbent".into(),
                    ));
                }
                // Match source: store the cutoff, not an arbitrary stronger bound.
                record.lower_bound = Some(limit);
            }
            TrialOutput::Complete(output) => {
                if output.bytes.is_empty() {
                    return Err(Error::Encoding(
                        "full-frame callback returned an empty output".into(),
                    ));
                }
                if output.bytes.len() > self.limits.max_output_bytes {
                    return Err(Error::Limit("whole-frame output bytes"));
                }
                record.bytes = Some(output.bytes.len());
                let wins = self.best.as_ref().is_none_or(|(best_index, best)| {
                    output.bytes.len() < best.bytes.len()
                        || (output.bytes.len() == best.bytes.len()
                            && stage < self.trials[*best_index].stage)
                });
                if wins {
                    self.best = Some((index, output));
                }
            }
        }
        let bytes = record.bytes;
        self.trials.push(record);
        Ok(bytes)
    }
    pub fn finish(self) -> Result<RetainedOutput<A>> {
        let (winner_index, output) = self
            .best
            .ok_or_else(|| Error::Encoding("search has no completed output".into()))?;
        let params = self.trials[winner_index].params.clone();
        Ok(RetainedOutput {
            bytes: output.bytes,
            artifact: output.artifact,
            params,
            winner_index,
            trials: self.trials,
            full_encodes: self.full_encodes,
            full_pixels: self.full_pixels,
        })
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    #[test]
    fn defaults_allow_work_past_the_former_encode_and_pixel_caps() {
        let limits = SearchLimits::default();
        assert_eq!(limits.max_full_encodes, usize::MAX);
        assert_eq!(limits.max_full_pixels, u64::MAX);
        assert_eq!(limits.max_output_bytes, usize::MAX);
        let mut search = Search::new(limits);
        for _ in 0..129 {
            search
                .refine_pixels(
                    8_192_000_001,
                    1,
                    CandidateParams::default(),
                    true,
                    false,
                    &mut |_, budget| {
                        assert_eq!(budget.max_output_bytes, usize::MAX);
                        Ok(TrialOutput::Complete(EncodedOutput {
                            bytes: vec![1],
                            artifact: (),
                        }))
                    },
                )
                .unwrap();
        }
        let output = search.finish().unwrap();
        assert_eq!(output.full_encodes, 129);
        assert_eq!(output.full_pixels, 129 * 8_192_000_001);
    }

    #[test]
    fn explicit_limits_reject_work_before_the_callback() {
        for limits in [
            SearchLimits {
                max_full_encodes: 0,
                ..Default::default()
            },
            SearchLimits {
                max_full_pixels: 0,
                ..Default::default()
            },
            SearchLimits {
                max_output_bytes: 0,
                ..Default::default()
            },
        ] {
            let mut search = Search::<()>::new(limits);
            assert!(matches!(
                search.refine_pixels(
                    1,
                    1,
                    CandidateParams::default(),
                    true,
                    false,
                    &mut |_, _| panic!("over-budget encode was invoked"),
                ),
                Err(Error::Limit(_))
            ));
            assert!(search.trials().is_empty());
        }
    }

    #[test]
    fn explicit_output_limit_and_counter_overflow_are_still_checked() {
        let mut search = Search::new(SearchLimits {
            max_output_bytes: 1,
            ..Default::default()
        });
        let mut encode = |_: &CandidateParams, budget: TrialBudget| {
            assert_eq!(budget.max_output_bytes, 1);
            Ok(TrialOutput::Complete(EncodedOutput {
                bytes: vec![1],
                artifact: (),
            }))
        };
        search
            .refine_pixels(
                u64::MAX,
                1,
                CandidateParams::default(),
                true,
                false,
                &mut encode,
            )
            .unwrap();
        assert!(matches!(
            search.refine_pixels(1, 1, CandidateParams::default(), true, false, &mut encode),
            Err(Error::Limit("whole-frame search work"))
        ));
        let mut search = Search::new(SearchLimits {
            max_output_bytes: 1,
            ..Default::default()
        });
        assert!(matches!(
            search.refine_pixels(
                1,
                1,
                CandidateParams::default(),
                true,
                false,
                &mut |_, _| Ok(TrialOutput::Complete(EncodedOutput {
                    bytes: vec![1, 2],
                    artifact: (),
                })),
            ),
            Err(Error::Limit("whole-frame output bytes"))
        ));
    }
}
