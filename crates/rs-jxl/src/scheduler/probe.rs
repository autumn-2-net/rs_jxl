// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! `enc_lossless_probe.cc` cached screen, Rank, SmallerGroups and Analyze.
//! Callbacks are serial unless a caller-owned ordered batch executor is supplied.
//! Serial completed cells survive errors; failed batches commit no cells.
//! Incomplete totals are never ranked, and no worker pool is created here.

use super::worker_pool::ProbeBatchKind;
use super::{
    batch::{PlannedProbeBatchEncoder, ProbeBatchEncoder, ProbeTask},
    params::{Candidate, CandidateParams, PlanOptions, lossless_probe_plan},
    regions::{RegionOptions, copy_sample, screen_regions},
    scan::{IMPLICIT_PALETTE_SIZE, LosslessScan},
};
use crate::{Error, Image, Result, modular::predictor::Predictor};

#[derive(Clone, Copy, Debug)]
pub struct ProbeLimits {
    /// Rust resource ceilings, not C++ search-policy constants.
    pub max_encodes: usize,
    pub max_encoded_pixels: u64,
    pub max_resident_sample_pixels: usize,
    pub max_candidates: usize,
}
impl Default for ProbeLimits {
    fn default() -> Self {
        Self {
            max_encodes: 512,
            max_encoded_pixels: 128 << 20,
            max_resident_sample_pixels: 8 << 20,
            max_candidates: 128,
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct ScreenOptions {
    pub regions: RegionOptions,
    /// JXL_LOSSLESS_PREFLIGHT_YIELD=1: availability threshold 2^17, else 2^19.
    pub adaptive_coverage: bool,
    /// JXL_LOSSLESS_PROBE_TRAINING: None, .01, .03, .1.
    pub training_cap: Option<f32>,
    /// JXL_LOSSLESS_CHEAP_PROBE_CONTROL: 0=off, 1=on, 2=compact only.
    pub cheap_control: u8,
    pub limits: ProbeLimits,
}

#[derive(Clone, Debug, Default)]
pub struct ProbeResult {
    pub ranked: Vec<Candidate>,
    pub exploratory: Vec<Candidate>,
    pub samples: usize,
    pub encodes: usize,
    pub control_bytes: usize,
    pub small_control_bytes: usize,
    pub candidates: usize,
    pub shortlisted: usize,
    pub screen_samples: usize,
    pub screen_bounded: usize,
    pub screen_saved_encodes: usize,
    pub bounded: usize,
    pub validation_encodes: usize,
    pub sample_pixels: u64,
}
#[derive(Clone, Debug)]
struct Cached {
    params: CandidateParams,
    bytes: Vec<Option<usize>>,
}

pub struct CandidateScreen<'a, F> {
    image: &'a Image,
    scan: Option<&'a LosslessScan>,
    options: ScreenOptions,
    encode: F,
    batch_encode: Option<PlannedProbeBatchEncoder<'a>>,
    samples: Vec<Image>,
    cache: Vec<Cached>,
    encodes: usize,
    sample_pixels: u64,
    batches: usize,
}

pub(crate) fn total(scores: &[usize]) -> Result<usize> {
    scores.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b).ok_or(Error::Limit("probe byte total"))
    })
}
impl<'a, F> CandidateScreen<'a, F>
where
    F: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    /// Callback encodes an independent frame, without invoking this scheduler.
    /// For a serial parent encoder this can be a closure over encode().len().
    pub fn new(
        image: &'a Image,
        scan: Option<&'a LosslessScan>,
        options: ScreenOptions,
        encode: F,
    ) -> Result<Self> {
        super::scan::validate_source_input(image)?;
        if options.cheap_control > 2
            || options
                .training_cap
                .is_some_and(|v| v != 0.01 && v != 0.03 && v != 0.1)
        {
            return Err(Error::InvalidInput("invalid source probe setting".into()));
        }
        let scan = scan.filter(|s| s.available());
        if scan.is_some_and(|s| {
            s.width != image.width() as usize || s.height != image.height() as usize
        }) {
            return Err(Error::InvalidInput(
                "probe scan belongs to different dimensions".into(),
            ));
        }
        Ok(Self {
            image,
            scan,
            options,
            encode,
            batch_encode: None,
            samples: Vec::new(),
            cache: Vec::new(),
            encodes: 0,
            sample_pixels: 0,
            batches: 0,
        })
    }
    pub fn available(&self) -> bool {
        self.image.metadata().extra_channels.len() <= 8
            && self
                .image
                .metadata()
                .extra_channels
                .iter()
                .all(|ec| ec.dim_shift == 0)
            && u64::from(self.image.width()) * u64::from(self.image.height())
                >= if self.options.adaptive_coverage {
                    1 << 17
                } else {
                    1 << 19
                }
    }
    pub fn encodes(&self) -> usize {
        self.encodes
    }
    pub fn sample_pixels(&self) -> u64 {
        self.sample_pixels
    }
    pub fn batches(&self) -> usize {
        self.batches
    }
    /// Release resident inputs and their cached scores, not cumulative work.
    /// A later score rebuilds samples and charges fresh encoding attempts.
    pub(crate) fn release_samples_and_cache(&mut self) {
        self.samples = Vec::new();
        self.cache = Vec::new();
    }
    /// Executes the source default deduplicated batch through a caller-owned
    /// executor. The existing FnMut callback remains the serial fallback.
    pub fn set_batch_encoder(&mut self, mut encode: ProbeBatchEncoder<'a>) {
        self.batch_encode = Some(Box::new(move |tasks, _| encode(tasks)));
    }
    pub fn set_planned_batch_encoder(&mut self, encode: PlannedProbeBatchEncoder<'a>) {
        self.batch_encode = Some(encode);
    }
    fn init(&mut self) -> Result<()> {
        if !self.samples.is_empty() {
            return Ok(());
        }
        let specs = screen_regions(self.image, self.scan, &self.options.regions)?;
        let mut pixels = 0usize;
        for s in &specs {
            pixels = pixels
                .checked_add(s.pixels()?)
                .ok_or(Error::Limit("resident probe pixels"))?;
        }
        if pixels > self.options.limits.max_resident_sample_pixels {
            return Err(Error::Limit("resident probe pixels"));
        }
        let samples = specs
            .iter()
            .map(|s| {
                copy_sample(
                    self.image,
                    s,
                    self.options.limits.max_resident_sample_pixels,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.samples = samples;
        Ok(())
    }
    fn index(&mut self, params: &CandidateParams) -> Result<usize> {
        params.validate()?;
        let mut candidate = params.clone();
        if candidate.palette_colors > 0
            && self.scan.is_some_and(|s| {
                s.palette_color_lower_bound as u64
                    > u64::from(candidate.palette_colors) + IMPLICIT_PALETTE_SIZE as u64
            })
        {
            candidate.palette_colors = 0;
        }
        if let Some(cap) = self.options.training_cap
            && (candidate.predictor.is_none() || candidate.predictor == Some(Predictor::Variable))
        {
            candidate.learning_fraction = candidate.learning_fraction.min(cap);
        }
        if let Some(index) = self.cache.iter().position(|c| c.params == candidate) {
            return Ok(index);
        }
        if self.cache.len() >= self.options.limits.max_candidates {
            return Err(Error::Limit("probe cache candidates"));
        }
        let index = self.cache.len();
        self.cache.push(Cached {
            params: candidate,
            bytes: vec![None; self.samples.len()],
        });
        Ok(index)
    }
    fn rounds(&mut self, candidates: &[usize], tiles: &[usize]) -> Result<()> {
        self.rounds_with_kind(candidates, tiles, ProbeBatchKind::Rounds)
    }
    fn rounds_with_kind(
        &mut self,
        candidates: &[usize],
        tiles: &[usize],
        kind: ProbeBatchKind,
    ) -> Result<()> {
        let mut pending = Vec::new();
        for &tile in tiles {
            for &index in candidates {
                if self.cache[index].bytes[tile].is_none() {
                    pending.push((index, tile));
                }
            }
        }
        pending.sort_unstable();
        pending.dedup();
        if pending.is_empty() {
            return Ok(());
        }
        let work = pending.iter().try_fold(0u64, |sum, &(_, tile)| {
            sum.checked_add(
                u64::from(self.samples[tile].width()) * u64::from(self.samples[tile].height()),
            )
            .ok_or(Error::Limit("probe work pixels"))
        })?;
        if self
            .encodes
            .checked_add(pending.len())
            .is_none_or(|n| n > self.options.limits.max_encodes)
            || self
                .sample_pixels
                .checked_add(work)
                .is_none_or(|n| n > self.options.limits.max_encoded_pixels)
        {
            return Err(Error::Limit("probe work budget"));
        }
        self.batches += 1;
        if let Some(encode) = &mut self.batch_encode {
            let tasks: Vec<_> = pending
                .iter()
                .map(|&(index, tile)| ProbeTask {
                    params: &self.cache[index].params,
                    image: &self.samples[tile],
                })
                .collect();
            self.encodes += pending.len();
            self.sample_pixels += work;
            let values = encode(&tasks, kind)?;
            if values.len() != pending.len() || values.contains(&0) {
                return Err(Error::Encoding(
                    "invalid probe batch result count/size".into(),
                ));
            }
            for ((index, tile), bytes) in pending.into_iter().zip(values) {
                self.cache[index].bytes[tile] = Some(bytes);
            }
            return Ok(());
        }
        for (index, tile) in pending {
            // Count attempted work even when the callback fails; never cache failure.
            self.encodes += 1;
            self.sample_pixels +=
                u64::from(self.samples[tile].width()) * u64::from(self.samples[tile].height());
            let bytes = (self.encode)(&self.cache[index].params, &self.samples[tile])?;
            self.cache[index].bytes[tile] = Some(bytes);
        }
        Ok(())
    }
    fn completed(&self, index: usize, tiles: &[usize]) -> Result<Vec<usize>> {
        tiles
            .iter()
            .map(|&tile| {
                self.cache[index].bytes[tile]
                    .ok_or_else(|| Error::Encoding("attempted to rank an incomplete probe".into()))
            })
            .collect()
    }
    pub fn score(&mut self, params: &CandidateParams) -> Result<Vec<usize>> {
        if !self.available() {
            return Ok(Vec::new());
        }
        self.init()?;
        let index = self.index(params)?;
        let tiles: Vec<_> = (0..self.samples.len()).collect();
        self.rounds_with_kind(&[index], &tiles, ProbeBatchKind::Score)?;
        self.completed(index, &tiles)
    }
    pub fn rank(
        &mut self,
        candidates: Vec<Candidate>,
        incumbent: &CandidateParams,
        fast_first_round: bool,
    ) -> Result<Vec<Candidate>> {
        if !self.available() || candidates.is_empty() {
            return Ok(candidates);
        }
        if candidates.len() > self.options.limits.max_candidates {
            return Err(Error::Limit("probe plan candidates"));
        }
        let mut unique: Vec<Candidate> = Vec::new();
        for c in candidates {
            if !unique.iter().any(|p| p.params == c.params) {
                unique.push(c);
            }
        }
        let candidates = unique;
        let mut ordering_control = incumbent.clone();
        let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
        let compact = (1 << 19..2 << 20).contains(&pixels);
        if fast_first_round
            && (self.options.cheap_control == 1 || (self.options.cheap_control == 2 && compact))
        {
            ordering_control.speed_tier =
                ordering_control
                    .speed_tier
                    .max(if self.scan.is_some() { 6 } else { 3 });
            ordering_control.learning_fraction = ordering_control.learning_fraction.min(0.01);
        }
        let control = self.score(&ordering_control)?;
        let mut tiles: Vec<_> = (0..control.len()).collect();
        tiles.sort_by(|&a, &b| {
            let density = |i: usize| {
                control[i] as f64
                    / (u64::from(self.samples[i].width()) * u64::from(self.samples[i].height()))
                        as f64
            };
            density(b).total_cmp(&density(a))
        });
        if tiles.len() > 2 {
            let middle = tiles.len() / 2;
            tiles.swap(1, middle);
        }
        let mut indices = Vec::new();
        for candidate in &candidates {
            let mut screening = candidate.params.clone();
            if fast_first_round {
                screening.speed_tier =
                    screening
                        .speed_tier
                        .max(if self.scan.is_some() { 6 } else { 3 });
                screening.learning_fraction = screening.learning_fraction.min(0.01);
            }
            indices.push(self.index(&screening)?);
        }
        let initial = tiles.len().min(2);
        self.rounds(&indices, &tiles[..initial])?;
        let partial: Vec<_> = indices
            .iter()
            .map(|&i| total(&self.completed(i, &tiles[..initial])?))
            .collect::<Result<_>>()?;
        let mut order: Vec<_> = (0..candidates.len()).collect();
        order.sort_by_key(|&i| partial[i]);
        let shortlist = if fast_first_round { 4 } else { 6 };
        let cutoff = partial[order[shortlist.min(order.len()) - 1]];
        let mut survivors = Vec::new();
        let mut conventional = false;
        let mut literal = false;
        for i in order {
            let p = &candidates[i].params;
            let regular = p.predictor.is_none();
            let palette = p.predictor == Some(Predictor::Zero) && p.palette_colors > 0;
            let cross = p.predictor == Some(Predictor::Variable);
            let close = fast_first_round && partial[i] as f64 <= cutoff as f64 * 1.03;
            if survivors.len() < shortlist
                || close
                || (regular && !conventional)
                || (palette && !literal)
                || cross
            {
                survivors.push(i);
                conventional |= regular;
                literal |= palette;
            }
        }
        let mut remaining = Vec::new();
        for &i in &survivors {
            indices[i] = self.index(&candidates[i].params)?;
            remaining.push(indices[i]);
        }
        self.rounds(
            &remaining,
            &tiles[if fast_first_round { 0 } else { initial }..],
        )?;
        let mut ranked = Vec::new();
        for i in survivors {
            let mut c = candidates[i].clone();
            c.context_bytes = Some(total(&self.completed(indices[i], &tiles)?)?);
            ranked.push(c);
        }
        ranked.sort_by_key(|c| c.context_bytes);
        Ok(ranked)
    }
    /// Full-fidelity regional scores in one caller-owned batch. Unlike rank's
    /// cheap first round this never changes the candidate's effort or training.
    pub(crate) fn score_plan(
        &mut self,
        candidates: Vec<Candidate>,
    ) -> Result<Vec<(Candidate, Vec<usize>)>> {
        if !self.available() || candidates.is_empty() {
            return Err(Error::InvalidInput("unavailable pilot screen".into()));
        }
        if candidates.len() > self.options.limits.max_candidates {
            return Err(Error::Limit("pilot candidate count"));
        }
        self.init()?;
        let mut unique: Vec<Candidate> = Vec::new();
        for candidate in candidates {
            if !unique.iter().any(|c| c.params == candidate.params) {
                unique.push(candidate);
            }
        }
        let indices = unique
            .iter()
            .map(|c| self.index(&c.params))
            .collect::<Result<Vec<_>>>()?;
        let tiles: Vec<_> = (0..self.samples.len()).collect();
        self.rounds(&indices, &tiles)?;
        unique
            .into_iter()
            .zip(indices)
            .map(|(mut candidate, index)| {
                let scores = self.completed(index, &tiles)?;
                candidate.context_bytes = Some(total(&scores)?);
                Ok((candidate, scores))
            })
            .collect()
    }
    pub fn smaller_groups(&mut self, ranked: &[Candidate]) -> Result<Vec<Candidate>> {
        let mut result = Vec::new();
        let Some(best) = ranked
            .first()
            .and_then(|c| c.context_bytes)
            .filter(|&n| n != 0)
        else {
            return Ok(result);
        };
        if !self.available() {
            return Ok(result);
        }
        let mut tested = 0;
        for c in ranked {
            let Some(bytes) = c.context_bytes else {
                continue;
            };
            if bytes as f64 > best as f64 * 1.02 {
                break;
            }
            if c.params.palette_colors != 0
                || c.params.predictor.is_none()
                || c.params.group_size_shift != Some(3)
            {
                continue;
            }
            if tested == 2 {
                break;
            }
            tested += 1;
            let mut smaller = c.clone();
            smaller.params.group_size_shift = Some(2);
            let score = total(&self.score(&smaller.params)?)?;
            smaller.context_bytes = Some(score);
            if (score as f64) < bytes as f64 * 0.995 {
                result.push(smaller);
            }
        }
        result.sort_by_key(|c| c.context_bytes);
        Ok(result)
    }
    pub fn analyze(
        &mut self,
        params: &CandidateParams,
        incumbent: &CandidateParams,
        plan_options: PlanOptions,
    ) -> Result<ProbeResult> {
        if !self.available() {
            return Ok(ProbeResult::default());
        }
        let before = self.encodes;
        let pixels_before = self.sample_pixels;
        let plan = lossless_probe_plan(params, plan_options);
        let mut result = ProbeResult {
            candidates: plan.len(),
            ..Default::default()
        };
        for c in &plan {
            if c.params.predictor == Some(Predictor::Variable) {
                let mut c = c.clone();
                c.exploratory = true;
                result.exploratory.push(c);
            }
        }
        let ranked = self.rank(plan, incumbent, true)?;
        let control = self.score(incumbent)?;
        result.samples = control.len();
        result.control_bytes = total(&control)?;
        result.small_control_bytes = result.control_bytes;
        result.shortlisted = ranked.len();
        for mut c in ranked {
            let scores = self.score(&c.params)?;
            if promising(&scores, &control)
                || c.context_bytes
                    .is_some_and(|n| n as f64 <= result.control_bytes as f64 * 1.01)
            {
                c.bytes = c.context_bytes;
                c.small_bytes = c.context_bytes;
                result.ranked.push(c);
            }
        }
        result.encodes = self.encodes - before;
        result.sample_pixels = self.sample_pixels - pixels_before;
        Ok(result)
    }
}

pub fn promising(candidate: &[usize], incumbent: &[usize]) -> bool {
    if candidate.is_empty() || candidate.len() != incumbent.len() {
        return true;
    }
    let sum: f64 = candidate.iter().map(|&n| n as f64).sum();
    let control: f64 = incumbent.iter().map(|&n| n as f64).sum();
    sum < control * 0.998
        || candidate
            .iter()
            .zip(incumbent)
            .any(|(&a, &b)| (a as f64) < b as f64 * 0.98)
}

#[cfg(test)]
#[path = "probe_release_tests.rs"]
mod release_tests;
