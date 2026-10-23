// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! b866b3d enc_lossless_probe.cc::ProbeLosslessCandidates candidate task bodies.
//! Each bounded root owns one candidate's ordered prefix. Only completed totals
//! tighten the screen cutoff; shortlisted candidates use the fixed control bound.

use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use super::{
    params::Candidate,
    worker_pool::{WorkerPlan, run_plan},
};
use crate::{Error, Image, Result, modular::predictor::Predictor};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixUsage {
    pub encodes: usize,
    pub pixels: u64,
}

struct Budget {
    max_encodes: usize,
    max_pixels: u64,
    used: PrefixUsage,
}

/// Shared hard ceiling, charged immediately before each encoder invocation,
/// including failed invocations. Pruned and cancelled cells cost no work.
pub struct PrefixWork(Mutex<Budget>);

impl PrefixWork {
    pub fn new(max_encodes: usize, max_pixels: u64) -> Self {
        Self(Mutex::new(Budget {
            max_encodes,
            max_pixels,
            used: PrefixUsage::default(),
        }))
    }

    pub fn usage(&self) -> PrefixUsage {
        self.0.lock().expect("prefix budget poisoned").used
    }

    pub(crate) fn limit_to(&self, max_encodes: usize, max_pixels: u64) {
        let mut budget = self.0.lock().expect("prefix budget poisoned");
        budget.max_encodes = budget.max_encodes.min(max_encodes);
        budget.max_pixels = budget.max_pixels.min(max_pixels);
    }

    fn reserve(&self, image: &Image) -> Result<()> {
        let mut budget = self.0.lock().expect("prefix budget poisoned");
        let pixels = u64::from(image.width()) * u64::from(image.height());
        let encodes = budget.used.encodes.checked_add(1);
        let total_pixels = budget.used.pixels.checked_add(pixels);
        if encodes.is_none_or(|n| n > budget.max_encodes)
            || total_pixels.is_none_or(|n| n > budget.max_pixels)
        {
            return Err(Error::Limit("cumulative scheduler probe work"));
        }
        budget.used = PrefixUsage {
            encodes: encodes.expect("checked encode count"),
            pixels: total_pixels.expect("checked pixel count"),
        };
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PrefixPass {
    /// Begin at zero; Variable candidates must finish their exploratory score.
    Screen { feedback_bounds: bool },
    /// Begin at Candidate.bytes, a completed screen score. All predictors may
    /// bound; the fixed source double-precision threshold is checked before each
    /// remaining tile, not rounded to an integer or tightened on completions.
    Shortlist { max_bytes: f64 },
}

/// The backend must use `run_prefix_screen` with these exact inputs and budget.
/// Tile indices in `order` are visited sequentially within each candidate.
pub struct PrefixScreen<'a> {
    pub candidates: &'a [Candidate],
    pub samples: &'a [Image],
    pub order: &'a [usize],
    pub pass: PrefixPass,
    pub work: &'a PrefixWork,
}

#[derive(Clone, Copy, Debug)]
pub struct PrefixScore {
    /// None means safely bounded, never zero or a partial completed score.
    pub(crate) bytes: Option<usize>,
    pub(crate) encodes: usize,
}

impl PrefixScore {
    pub fn bytes(self) -> Option<usize> {
        self.bytes
    }

    pub fn encodes(self) -> usize {
        self.encodes
    }
}

/// Sealed result construction keeps partial/cancelled totals out of ranking.
pub struct PrefixOutput {
    pub(crate) scores: Vec<PrefixScore>,
}

impl PrefixOutput {
    pub fn scores(&self) -> &[PrefixScore] {
        &self.scores
    }
}

struct Completed {
    best: Mutex<[usize; 6]>,
    cutoff: AtomicUsize,
}

impl Completed {
    fn new() -> Self {
        Self {
            best: Mutex::new([usize::MAX; 6]),
            cutoff: AtomicUsize::new(usize::MAX),
        }
    }

    fn insert(&self, bytes: usize) {
        let mut best = self.best.lock().expect("prefix completions poisoned");
        let position = best.partition_point(|&n| n < bytes);
        if position < best.len() {
            best.copy_within(position..5, position + 1);
            best[position] = bytes;
            // Publish under the mutex: a slower publisher must never loosen it.
            self.cutoff.store(best[5], Ordering::Relaxed);
        }
    }
}

/// Source RunProbeTasks with candidate roots, not tile rounds. No pool is
/// created. The callback receives (candidate index, actual sample index) and
/// runs with a null runner, including when the supplied pool has one thread.
/// Errors stop subsequent cell reservations and active calls drain before the
/// first observed error is returned. No partial output escapes a failed run.
pub fn run_prefix_screen(
    pool: Option<&rayon::ThreadPool>,
    worker_slots: usize,
    screen: &PrefixScreen<'_>,
    encode: impl Fn(usize, usize) -> Result<usize> + Sync,
) -> Result<PrefixOutput> {
    if (matches!(screen.pass, PrefixPass::Screen { .. }) && screen.order.is_empty())
        || screen
            .order
            .iter()
            .any(|&tile| tile >= screen.samples.len())
        || screen
            .order
            .iter()
            .enumerate()
            .any(|(i, tile)| screen.order[..i].contains(tile))
    {
        return Err(Error::InvalidInput(
            "invalid prefix screen tile order".into(),
        ));
    }
    if let PrefixPass::Shortlist { max_bytes } = screen.pass
        && (!max_bytes.is_finite()
            || max_bytes < 0.0
            || screen
                .candidates
                .iter()
                .any(|candidate| candidate.bytes.is_none_or(|bytes| bytes == 0)))
    {
        return Err(Error::InvalidInput(
            "invalid shortlist prefix score/bound".into(),
        ));
    }
    let plan = WorkerPlan::independent(screen.candidates.len(), worker_slots)?;
    let completed = Completed::new();
    // Holding this lock only across reservation makes stop/charge atomic. It is
    // never held during encoding or pool dispatch. Already reserved calls drain.
    let failure = Mutex::new(None);
    let scores = run_plan(pool, &plan, |candidate, _runner| {
        let feedback_bounds = matches!(
            screen.pass,
            PrefixPass::Screen {
                feedback_bounds: true
            }
        );
        let can_bound = feedback_bounds
            && screen.candidates[candidate].params.predictor != Some(Predictor::Variable);
        let mut score = PrefixScore {
            bytes: None,
            encodes: 0,
        };
        let mut total = match screen.pass {
            PrefixPass::Screen { .. } => 0,
            PrefixPass::Shortlist { .. } => screen.candidates[candidate]
                .bytes
                .expect("validated complete screen score"),
        };
        for &tile in screen.order {
            {
                let mut failed = failure.lock().expect("prefix failure poisoned");
                if failed.is_some() {
                    return Ok(score);
                }
                let bounded = match screen.pass {
                    PrefixPass::Screen { .. } => {
                        can_bound && total > completed.cutoff.load(Ordering::Relaxed)
                    }
                    PrefixPass::Shortlist { max_bytes } => total as f64 > max_bytes,
                };
                if bounded {
                    return Ok(score);
                }
                if let Err(error) = screen.work.reserve(&screen.samples[tile]) {
                    *failed = Some(error);
                    return Ok(score);
                }
            }
            score.encodes += 1;
            let bytes = encode(candidate, tile).and_then(|bytes| {
                if bytes == 0 {
                    return Err(Error::Encoding("empty prefix probe frame size".into()));
                }
                total.checked_add(bytes).ok_or(Error::Limit("probe bytes"))
            });
            match bytes {
                Ok(bytes) => total = bytes,
                Err(error) => {
                    failure
                        .lock()
                        .expect("prefix failure poisoned")
                        .get_or_insert(error);
                    return Ok(score);
                }
            }
        }
        score.bytes = Some(total);
        if feedback_bounds {
            completed.insert(total);
        }
        Ok(score)
    })?;
    if let Some(error) = failure.into_inner().expect("prefix failure poisoned") {
        return Err(error);
    }
    Ok(PrefixOutput { scores })
}

#[cfg(test)]
#[path = "prefix_screen_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "prefix_shortlist_tests.rs"]
mod shortlist_tests;
