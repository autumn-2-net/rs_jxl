// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! `enc_lossless_probe.cc::ProbeLosslessCandidates/ConfirmLosslessCandidate`.
//! Bounded prefixes may eliminate a crop total, never a whole-frame encoding.

use super::{
    batch::{PrefixBatchEncoder, ProbeTask},
    params::{Candidate, CandidateParams, Override, PlanOptions, lossless_probe_plan},
    prefix_screen::{PrefixOutput, PrefixPass, PrefixScreen, PrefixWork, run_prefix_screen},
    probe::{ProbeLimits, ProbeResult, total},
    regions::{SampleSpec, copy_sample, small_probe_regions},
    scan::{ScanRegion, centered_region},
    worker_pool::{WorkerPlan, run_plan},
};
use crate::{Error, Image, Result, modular::predictor::Predictor};

struct Work<F> {
    encode: F,
    limits: ProbeLimits,
    encodes: usize,
    pixels: u64,
}
impl<F: FnMut(&[ProbeTask<'_>]) -> Result<Vec<usize>>> Work<F> {
    fn run(&mut self, tasks: &[ProbeTask<'_>]) -> Result<Vec<usize>> {
        if tasks.is_empty() {
            return Ok(Vec::new());
        }
        let pixels = tasks.iter().try_fold(0u64, |n, task| {
            n.checked_add(u64::from(task.image.width()) * u64::from(task.image.height()))
                .ok_or(Error::Limit("probe work budget"))
        })?;
        if self
            .encodes
            .checked_add(tasks.len())
            .is_none_or(|n| n > self.limits.max_encodes)
            || self
                .pixels
                .checked_add(pixels)
                .is_none_or(|n| n > self.limits.max_encoded_pixels)
        {
            return Err(Error::Limit("probe work budget"));
        }
        self.encodes += tasks.len();
        self.pixels += pixels;
        let sizes = (self.encode)(tasks)?;
        if sizes.len() != tasks.len() || sizes.contains(&0) {
            return Err(Error::Encoding(
                "invalid small-probe batch result count/size".into(),
            ));
        }
        Ok(sizes)
    }

    fn run_prefix(
        &mut self,
        candidates: &[Candidate],
        samples: &[Image],
        order: &[usize],
        pass: PrefixPass,
        encode: &mut PrefixBatchEncoder<'_>,
    ) -> Result<PrefixOutput> {
        let budget = PrefixWork::new(
            self.limits.max_encodes - self.encodes,
            self.limits.max_encoded_pixels - self.pixels,
        );
        let output = encode(&PrefixScreen {
            candidates,
            samples,
            order,
            pass,
            work: &budget,
        });
        let usage = budget.usage();
        self.encodes += usage.encodes;
        self.pixels += usage.pixels;
        let output = output?;
        if output.scores.len() != candidates.len() {
            return Err(Error::Encoding("invalid prefix result count".into()));
        }
        let mut encodes = 0usize;
        let mut pixels = 0u64;
        for (candidate, score) in candidates.iter().zip(&output.scores) {
            let may_bound = match pass {
                PrefixPass::Screen { feedback_bounds } => {
                    feedback_bounds && candidate.params.predictor != Some(Predictor::Variable)
                }
                PrefixPass::Shortlist { .. } => true,
            };
            let valid = match score.bytes {
                Some(bytes) => bytes != 0 && score.encodes == order.len(),
                None => may_bound && score.encodes < order.len(),
            };
            if !valid {
                return Err(Error::Encoding("invalid prefix result score".into()));
            }
            encodes = encodes
                .checked_add(score.encodes)
                .ok_or(Error::Limit("probe work budget"))?;
            for &tile in &order[..score.encodes] {
                pixels = pixels
                    .checked_add(
                        u64::from(samples[tile].width()) * u64::from(samples[tile].height()),
                    )
                    .ok_or(Error::Limit("probe work budget"))?;
            }
        }
        if encodes != usage.encodes || pixels != usage.pixels {
            return Err(Error::Encoding("invalid prefix work accounting".into()));
        }
        Ok(output)
    }
}
fn supported(image: &Image) -> bool {
    image.metadata().extra_channels.len() <= 8
        && image
            .metadata()
            .extra_channels
            .iter()
            .all(|ec| ec.dim_shift == 0)
}

pub fn probe_lossless_candidates<F>(
    image: &Image,
    params: &CandidateParams,
    incumbent: Option<&CandidateParams>,
    options: PlanOptions,
    feedback_bounds: bool,
    limits: ProbeLimits,
    mut encode: F,
) -> Result<ProbeResult>
where
    F: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    probe_impl(
        image,
        params,
        incumbent,
        options,
        feedback_bounds,
        limits,
        1,
        None,
        |tasks| {
            tasks
                .iter()
                .map(|task| encode(task.params, task.image))
                .collect()
        },
    )
}

/// Ordered crop batches, with up to eight candidates advanced per screen round.
/// Completed rounds tighten feedback bounds. Unlike the C++ asynchronous atomic
/// cutoff, this can do extra work within a wave, but never drops a possible top-six
/// candidate. Variable-predictor exploratory candidates are always completed.
/// The legacy FnMut batch contract cannot publish completions during a call;
/// use `probe_lossless_candidates_parallel` for live asynchronous feedback.
pub fn probe_lossless_candidates_batched<F>(
    image: &Image,
    params: &CandidateParams,
    incumbent: Option<&CandidateParams>,
    options: PlanOptions,
    feedback_bounds: bool,
    limits: ProbeLimits,
    encode: F,
) -> Result<ProbeResult>
where
    F: FnMut(&[ProbeTask<'_>]) -> Result<Vec<usize>>,
{
    probe_impl(
        image,
        params,
        incumbent,
        options,
        feedback_bounds,
        limits,
        8,
        None,
        encode,
    )
}

/// Source candidate-prefix workers on a borrowed caller pool. The shared Fn
/// callback may run concurrently, always with the null-runner policy, and must
/// encode an independent sample without recursively invoking this scheduler.
/// No pool or encoder mutex is created. Worker slots must be 1..=64; an outer
/// serial scope overrides pool entry, including for controls and validation.
///
/// Reuses the same candidate plan, sample order, live completed-score cutoff,
/// shortlist and validation as `probe_lossless_candidates_with_prefix`. Its
/// remaining-work budget is shared with controls/validation, not reset for
/// each prefix pass. Feedback changes work counts, not completed-score ranking.
// Keep the existing probe arguments explicit, followed by executor and callback.
#[allow(clippy::too_many_arguments)]
pub fn probe_lossless_candidates_parallel<F>(
    image: &Image,
    params: &CandidateParams,
    incumbent: Option<&CandidateParams>,
    options: PlanOptions,
    feedback_bounds: bool,
    limits: ProbeLimits,
    pool: &rayon::ThreadPool,
    worker_slots: usize,
    encode: F,
) -> Result<ProbeResult>
where
    F: Fn(&CandidateParams, &Image) -> Result<usize> + Sync,
{
    // Validate even when the small-input gate would otherwise skip all work.
    WorkerPlan::independent(0, worker_slots)?;
    let mut prefix: PrefixBatchEncoder<'_> = Box::new(|screen| {
        run_prefix_screen(Some(pool), worker_slots, screen, |candidate, tile| {
            encode(&screen.candidates[candidate].params, &screen.samples[tile])
        })
    });
    probe_lossless_candidates_with_prefix(
        image,
        params,
        incumbent,
        options,
        feedback_bounds,
        limits,
        |tasks| {
            let plan = WorkerPlan::independent(tasks.len(), worker_slots)?;
            run_plan(Some(pool), &plan, |index, _runner| {
                encode(tasks[index].params, tasks[index].image)
            })
        },
        &mut prefix,
    )
}

/// Source asynchronous candidate screen and shortlist tasks, with the existing
/// ordered callback retained for controls and validation witnesses.
#[allow(clippy::too_many_arguments)]
pub fn probe_lossless_candidates_with_prefix<F>(
    image: &Image,
    params: &CandidateParams,
    incumbent: Option<&CandidateParams>,
    options: PlanOptions,
    feedback_bounds: bool,
    limits: ProbeLimits,
    encode: F,
    prefix: &mut PrefixBatchEncoder<'_>,
) -> Result<ProbeResult>
where
    F: FnMut(&[ProbeTask<'_>]) -> Result<Vec<usize>>,
{
    probe_impl(
        image,
        params,
        incumbent,
        options,
        feedback_bounds,
        limits,
        8,
        Some(prefix),
        encode,
    )
}

#[allow(clippy::too_many_arguments)]
fn probe_impl<F>(
    image: &Image,
    params: &CandidateParams,
    incumbent: Option<&CandidateParams>,
    options: PlanOptions,
    feedback_bounds: bool,
    limits: ProbeLimits,
    width: usize,
    mut prefix: Option<&mut PrefixBatchEncoder<'_>>,
    encode: F,
) -> Result<ProbeResult>
where
    F: FnMut(&[ProbeTask<'_>]) -> Result<Vec<usize>>,
{
    params.validate()?;
    super::scan::validate_source_input(image)?;
    if let Some(p) = incumbent {
        p.validate()?;
    }
    let pixels = u64::from(image.width()) * u64::from(image.height());
    if pixels <= 4096 || !supported(image) {
        return Ok(ProbeResult::default());
    }
    let specs = small_probe_regions(image);
    let resident = specs.iter().try_fold(0usize, |a, s| {
        a.checked_add(s.pixels()?)
            .ok_or(Error::Limit("resident probes"))
    })?;
    if resident > limits.max_resident_sample_pixels {
        return Err(Error::Limit("resident probes"));
    }
    let samples = specs
        .iter()
        .map(|s| copy_sample(image, s, limits.max_resident_sample_pixels))
        .collect::<Result<Vec<_>>>()?;
    let mut plan = lossless_probe_plan(params, options);
    if plan.len() > limits.max_candidates {
        return Err(Error::Limit("probe plan candidates"));
    }
    let mut control = incumbent.unwrap_or(params).clone();
    if incumbent.is_none() {
        control.speed_tier = 1;
        control.learning_fraction = 0.1;
        control.patches = Override::Off;
    }
    let mut work = Work {
        encode,
        limits,
        encodes: 0,
        pixels: 0,
    };
    let control_tasks: Vec<_> = samples
        .iter()
        .map(|sample| ProbeTask {
            params: &control,
            image: sample,
        })
        .collect();
    let control_sizes = work.run(&control_tasks)?;
    let mut result = ProbeResult {
        samples: samples.len(),
        candidates: plan.len(),
        control_bytes: total(&control_sizes)?,
        ..Default::default()
    };
    result.small_control_bytes = result.control_bytes;
    let mut order: Vec<_> = (0..samples.len()).collect();
    order.sort_by_key(|&i| control_sizes[i]);
    result.screen_samples = samples.len().min(3);
    let mut screened = vec![false; samples.len()];
    for i in 0..result.screen_samples {
        screened[order[(2 * i + 1) * samples.len() / (2 * result.screen_samples)]] = true;
    }
    let screen_order: Vec<_> = (0..samples.len())
        .map(|i| {
            if feedback_bounds {
                order[samples.len() - 1 - i]
            } else {
                i
            }
        })
        .filter(|&i| screened[i])
        .collect();
    let mut best_complete: Vec<usize> = Vec::new();
    let mut screen_encodes = 0;
    if let Some(prefix) = prefix.as_mut() {
        let output = work.run_prefix(
            &plan,
            &samples,
            &screen_order,
            PrefixPass::Screen { feedback_bounds },
            prefix,
        )?;
        for (candidate, score) in plan.iter_mut().zip(output.scores) {
            candidate.bytes = score.bytes;
            result.screen_bounded += usize::from(score.bytes.is_none());
            screen_encodes += score.encodes;
        }
    } else {
        for candidates in plan.chunks_mut(width) {
            let mut sizes = vec![Some(0usize); candidates.len()];
            for &tile in &screen_order {
                let cutoff = if best_complete.len() == 6 {
                    best_complete[5]
                } else {
                    usize::MAX
                };
                let mut indices = Vec::new();
                for (i, size) in sizes.iter_mut().enumerate() {
                    let Some(bytes) = *size else { continue };
                    if feedback_bounds
                        && candidates[i].params.predictor != Some(Predictor::Variable)
                        && bytes > cutoff
                    {
                        *size = None;
                        result.screen_bounded += 1;
                    } else {
                        indices.push(i);
                    }
                }
                let tasks: Vec<_> = indices
                    .iter()
                    .map(|&i| ProbeTask {
                        params: &candidates[i].params,
                        image: &samples[tile],
                    })
                    .collect();
                for (i, bytes) in indices.into_iter().zip(work.run(&tasks)?) {
                    sizes[i] = Some(
                        sizes[i]
                            .unwrap()
                            .checked_add(bytes)
                            .ok_or(Error::Limit("probe bytes"))?,
                    );
                    screen_encodes += 1;
                }
            }
            for (candidate, bytes) in candidates.iter_mut().zip(sizes) {
                candidate.bytes = bytes;
                if let Some(bytes) = bytes.filter(|_| feedback_bounds) {
                    best_complete.push(bytes);
                    best_complete.sort_unstable();
                    best_complete.truncate(6);
                }
            }
        }
    }
    for candidate in &plan {
        if candidate.params.predictor == Some(Predictor::Variable) {
            let mut exploratory = candidate.clone();
            exploratory.exploratory = true;
            result.exploratory.push(exploratory);
        }
    }
    result.screen_saved_encodes = plan.len() * result.screen_samples - screen_encodes;
    let mut candidate_order: Vec<_> = (0..plan.len()).collect();
    candidate_order.sort_by_key(|&i| plan[i].bytes.unwrap_or(usize::MAX));
    result.shortlisted = plan.len().min(6);
    let limit = result.control_bytes as f64 * 1.10;
    if let Some(prefix) = prefix.as_mut() {
        let candidates: Vec<_> = candidate_order[..result.shortlisted]
            .iter()
            .map(|&i| plan[i].clone())
            .collect();
        let remaining: Vec<_> = order
            .iter()
            .rev()
            .copied()
            .filter(|&i| !screened[i])
            .collect();
        let output = work.run_prefix(
            &candidates,
            &samples,
            &remaining,
            PrefixPass::Shortlist { max_bytes: limit },
            prefix,
        )?;
        for (mut candidate, score) in candidates.into_iter().zip(output.scores) {
            result.bounded += usize::from(score.bytes.is_none());
            if let Some(bytes) = score.bytes.filter(|&n| n as f64 <= limit) {
                candidate.bytes = Some(bytes);
                candidate.small_bytes = Some(bytes);
                result.ranked.push(candidate);
            }
        }
    } else {
        for indices in candidate_order[..result.shortlisted].chunks(width) {
            let mut sizes: Vec<_> = indices.iter().map(|&i| plan[i].bytes).collect();
            for &tile in order.iter().rev() {
                if screened[tile] {
                    continue;
                }
                let mut active = Vec::new();
                for (i, size) in sizes.iter_mut().enumerate() {
                    let Some(bytes) = *size else { continue };
                    if bytes as f64 > limit {
                        *size = None;
                        result.bounded += 1;
                    } else {
                        active.push(i);
                    }
                }
                let tasks: Vec<_> = active
                    .iter()
                    .map(|&i| ProbeTask {
                        params: &plan[indices[i]].params,
                        image: &samples[tile],
                    })
                    .collect();
                for (i, bytes) in active.into_iter().zip(work.run(&tasks)?) {
                    sizes[i] = Some(
                        sizes[i]
                            .unwrap()
                            .checked_add(bytes)
                            .ok_or(Error::Limit("probe bytes"))?,
                    );
                }
            }
            for (&index, bytes) in indices.iter().zip(sizes) {
                if let Some(bytes) = bytes.filter(|&n| n as f64 <= limit) {
                    let candidate = &mut plan[index];
                    candidate.bytes = Some(bytes);
                    candidate.small_bytes = Some(bytes);
                    result.ranked.push(candidate.clone());
                }
            }
        }
    }
    result.ranked.sort_by_key(|c| c.bytes);
    const SIDE: usize = 384;
    if !result.ranked.is_empty() && pixels >= (4 * SIDE * SIDE) as u64 {
        let anchor = specs[order[2 * order.len() / 3]].regions[0];
        let w = image.width() as usize;
        let h = image.height() as usize;
        let r = centered_region(
            w,
            h,
            anchor.x + anchor.width / 2,
            anchor.y + anchor.height / 2,
            w.min(SIDE),
            h.min(SIDE),
        );
        let spec = SampleSpec { regions: vec![r] };
        if resident
            .checked_add(spec.pixels()?)
            .is_none_or(|n| n > limits.max_resident_sample_pixels)
        {
            return Err(Error::Limit("resident validation pixels"));
        }
        let sample = copy_sample(image, &spec, limits.max_resident_sample_pixels)?;
        result.ranked.truncate(3);
        let tasks: Vec<_> = std::iter::once(&control)
            .chain(result.ranked.iter().map(|c| &c.params))
            .map(|params| ProbeTask {
                params,
                image: &sample,
            })
            .collect();
        let sizes = work.run(&tasks)?;
        result.control_bytes = sizes[0];
        result.validation_encodes = 1 + result.ranked.len();
        for (c, &bytes) in result.ranked.iter_mut().zip(&sizes[1..]) {
            c.bytes = Some(bytes);
        }
        result.ranked.retain(|c| {
            c.bytes.is_some_and(|n| {
                n as f64 <= result.control_bytes as f64 * 1.05
                    && !(n >= result.control_bytes
                        && c.small_bytes
                            .is_some_and(|s| s >= result.small_control_bytes))
            })
        });
        result.ranked.sort_by_key(|c| c.bytes);
    }
    result.encodes = work.encodes;
    result.sample_pixels = work.pixels;
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct ProbeConfirmation {
    pub samples: usize,
    pub candidate_bytes: usize,
    pub incumbent_bytes: usize,
    pub accepted: bool,
    pub encodes: usize,
}
/// `known_incumbent_sample_total` must be the sum of these same independent
/// 512-side witnesses for this exact frame/metadata/incumbent, NOT full bytes.
pub fn confirm_lossless_candidate<F>(
    image: &Image,
    candidate: &CandidateParams,
    incumbent: &CandidateParams,
    known_incumbent_sample_total: Option<usize>,
    limits: ProbeLimits,
    mut encode: F,
) -> Result<ProbeConfirmation>
where
    F: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    confirm_lossless_candidate_batched(
        image,
        candidate,
        incumbent,
        known_incumbent_sample_total,
        limits,
        |tasks| {
            tasks
                .iter()
                .map(|task| encode(task.params, task.image))
                .collect()
        },
    )
}

/// Independent confirmation witnesses, dispatched in ordered batches of at most
/// four encodes. All resident crops are accounted for before materializing them.
pub fn confirm_lossless_candidate_batched<F>(
    image: &Image,
    candidate: &CandidateParams,
    incumbent: &CandidateParams,
    known_incumbent_sample_total: Option<usize>,
    limits: ProbeLimits,
    encode: F,
) -> Result<ProbeConfirmation>
where
    F: FnMut(&[ProbeTask<'_>]) -> Result<Vec<usize>>,
{
    candidate.validate()?;
    incumbent.validate()?;
    super::scan::validate_source_input(image)?;
    let known_incumbent_sample_total = known_incumbent_sample_total.filter(|&n| n != 0);
    let mut result = ProbeConfirmation {
        samples: 0,
        candidate_bytes: 0,
        incumbent_bytes: 0,
        accepted: true,
        encodes: 0,
    };
    if u64::from(image.width()) * u64::from(image.height()) <= 262144 || !supported(image) {
        return Ok(result);
    }
    let w = image.width() as usize;
    let h = image.height() as usize;
    let width = w.min(512);
    let height = h.min(512);
    let mut specs = Vec::new();
    for i in 0..3 {
        let spec = SampleSpec {
            regions: vec![ScanRegion {
                x: i * (w - width) / 2,
                y: i * (h - height) / 2,
                width,
                height,
            }],
        };
        if !specs.contains(&spec) {
            specs.push(spec);
        }
    }
    let mut work = Work {
        encode,
        limits,
        encodes: 0,
        pixels: 0,
    };
    result.samples = specs.len();
    result.incumbent_bytes = known_incumbent_sample_total.unwrap_or(0);
    let resident = specs.iter().try_fold(0usize, |n, s| {
        n.checked_add(s.pixels()?)
            .ok_or(Error::Limit("resident confirmation pixels"))
    })?;
    if resident > limits.max_resident_sample_pixels {
        return Err(Error::Limit("resident confirmation pixels"));
    }
    let samples = specs
        .iter()
        .map(|spec| copy_sample(image, spec, limits.max_resident_sample_pixels))
        .collect::<Result<Vec<_>>>()?;
    let mut tasks = Vec::new();
    if known_incumbent_sample_total.is_none() {
        tasks.extend(samples.iter().map(|sample| ProbeTask {
            params: incumbent,
            image: sample,
        }));
    }
    tasks.extend(samples.iter().map(|sample| ProbeTask {
        params: candidate,
        image: sample,
    }));
    let mut sizes = Vec::with_capacity(tasks.len());
    for chunk in tasks.chunks(4) {
        sizes.extend(work.run(chunk)?);
    }
    let candidate_start = if known_incumbent_sample_total.is_some() {
        0
    } else {
        samples.len()
    };
    if known_incumbent_sample_total.is_none() {
        result.incumbent_bytes = total(&sizes[..candidate_start])?;
    }
    result.candidate_bytes = total(&sizes[candidate_start..])?;
    result.accepted = result.candidate_bytes < result.incumbent_bytes;
    result.encodes = work.encodes;
    Ok(result)
}

#[cfg(test)]
#[path = "small_probe_parallel_tests.rs"]
mod parallel_tests;
