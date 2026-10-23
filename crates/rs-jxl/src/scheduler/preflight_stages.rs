// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Source preflight-yield promotion, geometry and anchor ordering.

use super::*;
use crate::modular::predictor::Predictor;

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    pub(super) fn plan_first_stage(
        &mut self,
        regular: Option<usize>,
        literal: Option<usize>,
    ) -> Result<()> {
        let (Some(regular), Some(literal)) = (regular, literal) else {
            return Ok(());
        };
        if !self.routes.plan_first
            || !self.screen.available()
            || !(literal as f64 >= regular as f64 * 0.8
                || self.routes.capacity_mismatch
                || self.routes.broad_scope)
            || literal as f64 > regular as f64 * 1.35
        {
            return Ok(());
        }
        self.analyze(&self.best_params(3)?, false)?;
        let Some(mut candidate) = self.analysis.ranked.first().map(|c| c.params.clone()) else {
            return Ok(());
        };
        if candidate.palette_colors != 0 || candidate.predictor != Some(Predictor::Variable) {
            return Ok(());
        }
        let proposed = total(&self.screen.score(&candidate)?)?;
        let automatic = total(&self.screen.score(&self.input.backend(7)?)?)?;
        let before = self.best_size(3)?;
        if automatic == 0
            || proposed as f64 >= automatic as f64 * if self.routes.broad_scope { 0.9 } else { 0.8 }
        {
            return Ok(());
        }
        candidate.bucket_matcher = self.routes.plan_buckets && !self.routes.scope_routing;
        // Source mutates the retained leader; only its deferred chain replay
        // drops buckets. Stage 8 can reuse this ranking without reanalysis.
        self.analysis.ranked[0].params = candidate.clone();
        let size = self.refine_promoted(3, candidate.clone(), false, false, false, true)?;
        self.early_family_winner = size.is_some_and(|n| (n as f64) < before as f64 * 0.8)
            && self.best_params(4)? == candidate;
        let mut cheap = candidate.backend(4)?;
        cheap.learning_fraction = cheap.learning_fraction.min(0.01);
        if self.early_family_winner {
            self.defer(3, cheap, false);
            self.defer(3, self.input.backend(7)?, false);
            self.defer(3, self.input.backend(9)?, false);
            if candidate.bucket_matcher {
                candidate.bucket_matcher = false;
                self.defer(3, candidate, false);
            }
        } else {
            self.calibration_full(3, cheap)?;
        }
        self.events.push(SearchEvent::Route {
            name: "plan-first",
            candidates: usize::from(self.early_family_winner),
        });
        Ok(())
    }

    pub(super) fn prepare_racing_frontier(
        &mut self,
        literal_present: bool,
        regular: Option<usize>,
        literal: Option<usize>,
    ) -> Result<()> {
        if self.routes.broad_scope && self.early_family_winner {
            self.frontier = self
                .analysis
                .ranked
                .iter()
                .filter(|c| self.search.find(&c.params).is_none())
                .cloned()
                .collect();
            self.events.push(SearchEvent::Route {
                name: "reused-frontier",
                candidates: self.frontier.len(),
            });
            return Ok(());
        }
        let regular = regular.unwrap_or(usize::MAX);
        let literal = literal.unwrap_or(usize::MAX);
        let mut geometry_winner = false;
        if self.routes.geometry_first && regular <= literal {
            let anchor = self.input.backend(7)?;
            let ranked = self.screen.rank(
                lossless_probe_plan(self.input, self.options.plan),
                &anchor,
                true,
            )?;
            let smaller = self.screen.smaller_groups(&ranked)?;
            let control = total(&self.screen.score(&anchor)?)?;
            if let Some(candidate) = smaller.first()
                && candidate
                    .context_bytes
                    .is_some_and(|n| (n as f64) < control as f64 * 0.95)
            {
                let medium = candidate.params.backend(7)?;
                let bytes = self.calibration_full(4, medium.clone())?;
                let denominator = total(&self.screen.score(&medium)?)?;
                let prediction = if denominator == 0 {
                    0.0
                } else {
                    bytes as f64 * candidate.context_bytes.unwrap() as f64 / denominator as f64
                };
                if denominator != 0 && prediction < regular as f64 * 0.85 {
                    let size = self.full(4, candidate.params.clone(), false, false)?;
                    geometry_winner = size.is_some_and(|n| (n as f64) < regular as f64 * 0.85)
                        && self.best_params(5)? == candidate.params;
                }
                self.events.push(SearchEvent::Calibration {
                    stage: 4,
                    params: candidate.params.clone(),
                    kind: "geometry-first",
                    predicted_bytes: Some(prediction),
                    accepted: geometry_winner,
                });
            }
            if geometry_winner {
                self.pending_automatic = true;
                self.defer(4, self.input.backend(8)?, false);
            }
        }
        if self.routes.preflight_yield && regular <= literal && !geometry_winner {
            let automatic = self.full(4, self.input.backend(8)?, true, false)?;
            let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
            let promote =
                pixels < 1 << 19 || automatic.is_some_and(|n| (n as f64) < regular as f64 * 0.995);
            if promote && self.routes.calibration_reuse && pixels >= 2 << 20 {
                let anchor = self.input.backend(9)?;
                let automatic_total = total(&self.screen.score(&anchor)?)?;
                let rivals = self.screen.rank(
                    lossless_probe_plan(self.input, self.options.plan),
                    &anchor,
                    true,
                )?;
                self.pending_automatic = automatic_total != 0
                    && rivals.iter().any(|c| {
                        c.context_bytes
                            .is_some_and(|n| (n as f64) < automatic_total as f64 * 0.85)
                    });
            }
            if promote && !self.pending_automatic {
                self.full(4, self.input.backend(9)?, true, false)?;
            }
        }
        self.build_frontier(literal_present)
    }

    /// Returns true when stage 5 is consumed by an automatic anchor.
    pub(super) fn middle_preflight(&mut self, regular: &mut Option<usize>) -> Result<bool> {
        let default7 = self.input.backend(7)?;
        if self.routes.early_family && self.screen.available() && self.best_params(5)? == default7 {
            self.analyze(&self.best_params(5)?, false)?;
            if let Some(candidate) = self.analysis.ranked.first().map(|c| c.params.clone())
                && candidate.palette_colors == 0
                && candidate.predictor == Some(Predictor::Variable)
            {
                let before = self.best_size(6)?;
                let size = self.refine_promoted(5, candidate.clone(), false, false, false, true)?;
                self.early_family_winner =
                    size.is_some_and(|n| n < before) && self.best_params(6)? == candidate;
            }
        }
        let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
        if !self.early_family_winner
            && self.routes.anchor_first
            && pixels >= 2 << 20
            && self.screen.available()
            && self.best_params(5)? == default7
        {
            let auto = self.input.backend(9)?;
            let neighbor = winner_neighborhood(&self.best_params(5)?, self.image.channel_count());
            let automatic = total(&self.screen.score(&auto)?)?;
            let neighborhood = total(&self.screen.score(&neighbor)?)?;
            if (automatic as f64) < neighborhood as f64 * 0.95 {
                let mut control = self.input.backend(4)?;
                control.learning_fraction = 0.01;
                let mut cheap =
                    winner_neighborhood(self.input, self.image.channel_count()).backend(4)?;
                cheap.learning_fraction = 0.01;
                let control_bytes = self.calibration_full(5, control)?;
                let neighbor_bytes = self.calibration_full(5, cheap)?;
                if control_bytes <= neighbor_bytes {
                    let size = self.refine(5, auto, true, false, false)?;
                    *regular = regular.iter().copied().chain(size).min();
                    if size.is_some() {
                        return Ok(true);
                    }
                }
            }
        }
        if self.routes.preflight_yield && pixels < 2 << 20 && self.best_params(5)? == default7 {
            let size = self.full(5, self.input.backend(9)?, true, false)?;
            *regular = regular.iter().copied().chain(size).min();
            return Ok(true);
        }
        Ok(false)
    }
}
