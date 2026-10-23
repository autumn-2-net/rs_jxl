// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Optional direct/global/strip tournaments from enc_lossless_search.cc.
//! Sample projections only nominate full encodes; they never replace an output.

use super::super::{
    calibration::{calibration_agreement, shrink_calibration},
    scan::IMPLICIT_PALETTE_SIZE,
};
use super::*;
use crate::modular::predictor::Predictor;

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    fn tournament_full(
        &mut self,
        plans: &[CandidateParams],
        stage: u8,
        baseline: bool,
        calibration: bool,
        parallel: bool,
        cooperative: bool,
    ) -> Result<Vec<Option<usize>>> {
        let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
        let scratch = pixels
            .saturating_mul(3 + self.image.metadata().extra_channels.len() as u64)
            .saturating_mul(32)
            .saturating_add(32 << 20);
        let width = if parallel && self.full_batch.is_some() {
            self.options
                .workers
                .concurrency(plans.len(), scratch)
                .max(1)
        } else {
            1
        };
        let mut sizes = Vec::with_capacity(plans.len());
        for chunk in plans.chunks(width) {
            if chunk.len() == 1 || chunk.iter().any(|p| self.search.find(p).is_some()) {
                for params in chunk {
                    sizes.push(self.full(stage, params.clone(), baseline, calibration)?);
                }
                continue;
            }
            self.search.check_batch_work(self.image, chunk.len())?;
            let tasks: Vec<_> = chunk
                .iter()
                .map(|params| FullTask {
                    params,
                    cooperative,
                    budget: TrialBudget {
                        beat_bytes: None,
                        max_output_bytes: self.options.full_limits.max_output_bytes,
                    },
                })
                .collect();
            let outputs = self.full_batch.as_mut().expect("batch selected")(self.image, &tasks)?;
            if outputs.len() != chunk.len() {
                return Err(Error::Encoding(
                    "invalid tournament batch result count".into(),
                ));
            }
            for (params, output) in chunk.iter().zip(outputs) {
                let mut output = Some(output);
                sizes.push(self.search.refine(
                    self.image,
                    stage,
                    params.clone(),
                    baseline,
                    calibration,
                    &mut |_, _, _| Ok(output.take().expect("one invocation")),
                )?);
            }
        }
        Ok(sizes)
    }

    pub(super) fn run_tournament(&mut self) -> Result<bool> {
        if self.level != 9 || !refinable(self.image) {
            return Ok(false);
        }
        let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
        let options = self.options.frontier;
        let compact = options.compact && (1 << 19..2 << 20).contains(&pixels);
        if options.direct != DirectFrontier::Disabled
            && (pixels >= 2 << 20 || compact)
            && let Some(scan) = self.scan
        {
            let sparse = options.direct.extended()
                && !scan.likely_continuous_tone()
                && scan.likely_sparse_continuous_tone()
                && scan.palette_color_lower_bound > 70000 + IMPLICIT_PALETTE_SIZE;
            let dense = options.direct == DirectFrontier::Dense
                && scan.likely_dense_texture()
                && scan.palette_color_lower_bound > 10000 + IMPLICIT_PALETTE_SIZE;
            if (scan.likely_continuous_tone() || sparse || dense || compact)
                && self.screen.available()
            {
                let mut plans = lossless_probe_plan(self.input, self.options.plan);
                plans.push(Candidate::new("automatic-e9", self.input.backend(9)?));
                if options.direct.extended() {
                    plans.push(Candidate::new(
                        "automatic-neighborhood",
                        winner_neighborhood(self.input, self.image.channel_count()),
                    ));
                }
                plans.push(Candidate::new("literal-i1", literal(self.input, 9, 0.01)?));
                plans.push(Candidate::new("literal-i10", literal(self.input, 9, 0.1)?));
                let ranked = self.screen.rank(plans, &self.input.backend(9)?, true)?;
                let leader = ranked
                    .first()
                    .ok_or_else(|| Error::Encoding("empty direct frontier".into()))?;
                let best = leader
                    .context_bytes
                    .ok_or_else(|| Error::Encoding("unscored direct frontier".into()))?;
                let cap = if compact {
                    2
                } else if options.direct.extended() {
                    4
                } else {
                    2
                };
                let tolerance = if compact {
                    1.03
                } else if sparse {
                    1.025
                } else {
                    1.01
                };
                let mut finalists = 1;
                while finalists < cap.min(ranked.len())
                    && ranked[finalists]
                        .context_bytes
                        .is_some_and(|n| n as f64 <= best as f64 * tolerance)
                {
                    finalists += 1;
                }
                let mut selected: Vec<_> = (0..finalists).collect();
                if !compact && options.direct.calibrated() && finalists > 1 {
                    let medium = !sparse && options.medium_calibration;
                    let cheap: Vec<_> = ranked[..finalists]
                        .iter()
                        .map(|c| {
                            let mut p = c.params.backend(if medium { 7 } else { 4 })?;
                            if !medium {
                                p.learning_fraction = p.learning_fraction.min(0.01);
                            }
                            Ok(p)
                        })
                        .collect::<Result<_>>()?;
                    let full = self
                        .tournament_full(
                            &cheap,
                            8,
                            true,
                            true,
                            options.parallel_finalists,
                            options.parallel_finalists,
                        )?
                        .into_iter()
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| Error::Encoding("incomplete direct calibration".into()))?;
                    let mut projected = Vec::new();
                    let mut regions = Vec::new();
                    for i in 0..finalists {
                        let scores = self.screen.score(&cheap[i])?;
                        regions.push(scores.len());
                        let denominator = total(&scores)?;
                        projected.push(if denominator == 0 {
                            0.0
                        } else {
                            full[i] as f64 * ranked[i].context_bytes.unwrap() as f64
                                / denominator as f64
                        });
                    }
                    if matches!(
                        options.direct,
                        DirectFrontier::Shrink | DirectFrontier::Dense
                    ) {
                        projected = shrink_calibration(&cheap, &full, &projected, &regions);
                    }
                    if projected.iter().all(|&v| v > 0.0 && v.is_finite()) {
                        selected.sort_by(|&a, &b| projected[a].total_cmp(&projected[b]));
                        if sparse {
                            selected.truncate(2);
                        } else if projected[selected[0]] < projected[selected[1]] * 0.995
                            || (options.direct == DirectFrontier::Dense
                                && calibration_agreement(&projected, &full, selected[0]))
                        {
                            selected.truncate(1);
                        }
                        if options.direct.extended()
                            && !sparse
                            && selected.len() == 1
                            && ranked[selected[0]].params.predictor == Some(Predictor::Variable)
                        {
                            let leader = selected[0];
                            for i in 0..finalists {
                                if ranked[i].params.predictor.is_none()
                                    && ranked[i].params.group_size_shift == Some(2)
                                    && full[i] as f64 <= full[leader] as f64 * 1.01
                                    && projected[i] <= projected[leader] * 1.05
                                {
                                    selected.insert(0, i);
                                    break;
                                }
                            }
                        }
                    }
                }
                let plans: Vec<_> = selected.iter().map(|&i| ranked[i].params.clone()).collect();
                self.tournament_full(
                    &plans,
                    9,
                    false,
                    false,
                    options.parallel_finalists,
                    options.parallel_finalists,
                )?;
                self.events.push(SearchEvent::Route {
                    name: "direct-frontier",
                    candidates: plans.len(),
                });
                return Ok(true);
            }
        }
        if options.global_tournament {
            let mut plans = vec![
                self.input.backend(9)?,
                literal(self.input, 9, 0.01)?,
                literal(self.input, 9, 0.1)?,
            ];
            for rct in [0, 6, 10, 17, 26] {
                plans.push(self.cross_plan(rct)?);
            }
            let cheap: Vec<_> = plans.iter().map(|p| p.backend(7)).collect::<Result<_>>()?;
            let sizes = self
                .tournament_full(&cheap, 3, false, true, true, false)?
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::Encoding("incomplete global calibration".into()))?;
            let best = (0..sizes.len()).min_by_key(|&i| sizes[i]).unwrap();
            self.full(9, plans[best].clone(), false, false)?;
            self.events.push(SearchEvent::Route {
                name: "global-tournament",
                candidates: plans.len(),
            });
            return Ok(true);
        }
        if options.strip_tournament {
            // The source strip scout owns a distinct cache and coverage scan.
            let scan = scan_lossless_input(self.image, self.options.scan);
            let probe = Rc::clone(&self.probe);
            let callback: ProbeCallback<'_> =
                Box::new(move |p, sample| probe.borrow_mut().run(p, sample));
            let mut settings = self.options.screen.clone();
            settings.adaptive_coverage |= self.routes.preflight_yield;
            settings.regions.full_width_strips = true;
            let mut strips = CandidateScreen::new(self.image, Some(&scan), settings, callback)?;
            if let Some(batch) = &self.probe_batch {
                let batch = Rc::clone(batch);
                strips.set_planned_batch_encoder(Box::new(move |tasks, kind| {
                    batch.borrow_mut()(tasks, kind)
                }));
            }
            if strips.available() {
                let mut plans = lossless_probe_plan(self.input, self.options.plan);
                plans.push(Candidate::new("automatic", self.input.backend(9)?));
                plans.push(Candidate::new("literal-i1", literal(self.input, 9, 0.01)?));
                plans.push(Candidate::new("literal-i10", literal(self.input, 9, 0.1)?));
                for rct in [10, 17, 26] {
                    plans.push(Candidate::new("cross-rct", self.cross_plan(rct)?));
                }
                let ranked = strips.rank(plans, &self.input.backend(9)?, true)?;
                self.auxiliary_screen_batches += strips.batches();
                let leader = ranked
                    .first()
                    .ok_or_else(|| Error::Encoding("empty strip frontier".into()))?;
                let best = leader
                    .context_bytes
                    .ok_or_else(|| Error::Encoding("unscored strip frontier".into()))?;
                let finalists = if ranked.len() > 1
                    && ranked[1]
                        .context_bytes
                        .is_some_and(|n| n as f64 <= best as f64 * 1.01)
                {
                    2
                } else {
                    1
                };
                for candidate in &ranked[..finalists] {
                    self.full(9, candidate.params.clone(), false, false)?;
                }
                self.events.push(SearchEvent::Route {
                    name: "strip-tournament",
                    candidates: finalists,
                });
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn cross_plan(&self, rct: u32) -> Result<CandidateParams> {
        let mut p = self.input.backend(9)?;
        p.group_size_shift = Some(3);
        p.palette_colors = 0;
        p.channel_colors_percent = 0.0;
        p.channel_colors_pre_transform_percent = 0.0;
        p.patches = super::super::params::Override::Off;
        p.predictor = Some(Predictor::Variable);
        p.max_properties = 2;
        p.learning_fraction = 0.1;
        p.rct = Some(rct);
        Ok(p)
    }
}
