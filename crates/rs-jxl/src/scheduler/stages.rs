// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Pixel stage loop, early racing, optional fast-literal route and replay from
//! `enc_lossless_search.cc` (default/PREFLIGHT_YIELD=off branches).

use super::super::policy::TrialFamily;
use super::*;

fn minimum(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    a.into_iter().chain(b).min()
}

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    fn frontier_add(&mut self, name: &'static str, params: CandidateParams) {
        if self.search.find(&params).is_some() || self.frontier.iter().any(|c| c.params == params) {
            return;
        }
        self.frontier.push(Candidate::new(name, params));
    }
    pub(super) fn build_frontier(&mut self, literal_present: bool) -> Result<()> {
        let incumbent = self.best_params(5)?;
        if self.scan.is_some() {
            // Scan-guided cached ranking already visits all representations.
            self.analysis.ranked = lossless_probe_plan(self.input, self.options.plan);
        } else {
            self.analyze(&incumbent, false)?;
        }
        self.frontier_add("regular-e8", self.input.backend(8)?);
        self.frontier_add("regular-e9", self.input.backend(9)?);
        self.frontier_add(
            "anchor-neighborhood",
            winner_neighborhood(&incumbent, self.image.channel_count()),
        );
        if literal_present {
            self.frontier_add("literal-fixed", literal(self.input, 8, 0.0)?);
            self.frontier_add("literal-i1", literal(self.input, 9, 0.01)?);
            self.frontier_add("literal-i10", literal(self.input, 9, 0.1)?);
            self.frontier_add("literal-i100", literal(self.input, 9, 1.0)?);
        }
        for candidate in self
            .analysis
            .ranked
            .clone()
            .into_iter()
            .chain(self.analysis.exploratory.clone())
        {
            self.frontier_add(candidate.name, candidate.params);
        }
        let count = self.frontier.len();
        let plan = std::mem::take(&mut self.frontier);
        self.frontier = self.screen.rank(
            plan,
            &incumbent,
            self.options.fast_preflight || self.scan.is_some(),
        )?;
        if self.options.family_beam {
            self.frontier = diverse_frontier(std::mem::take(&mut self.frontier));
        }
        self.events.push(SearchEvent::EarlyRacing {
            stage: 4,
            candidates: count,
        });
        self.events.push(SearchEvent::Frontier {
            candidates: self.frontier.clone(),
        });
        Ok(())
    }
    fn fast_literal_stage(&mut self, stage: u8) -> Result<()> {
        if stage == 4 || stage == 6 || stage == 9 {
            let fraction = match stage {
                4 => 0.01,
                6 => 0.03,
                _ => 1.0,
            };
            let mut candidate = literal(self.input, 9, fraction)?;
            if self.options.palette_preselect && stage == 4 {
                let mut best_score = None;
                for fraction in [0.01, 0.03, 1.0] {
                    let choice = literal(self.input, 9, fraction)?;
                    let score = total(&self.screen.score(&choice)?)?;
                    if score != 0 && best_score.is_none_or(|old| score < old) {
                        best_score = Some(score);
                        candidate = choice;
                    }
                }
                for fraction in [0.01, 0.03, 1.0] {
                    let choice = literal(self.input, 9, fraction)?;
                    if choice != candidate {
                        self.defer(stage, choice, false);
                    }
                }
            }
            if (self.options.palette_race || self.options.palette_preselect) && stage > 4 {
                let proposed = total(&self.screen.score(&candidate)?)?;
                let incumbent = self.best_params(stage)?;
                let current = total(&self.screen.score(&incumbent)?)?;
                let accepted =
                    current == 0 || proposed == 0 || (proposed as f64) < current as f64 * 0.999;
                self.events.push(SearchEvent::Calibration {
                    stage,
                    params: candidate.clone(),
                    kind: "palette-race",
                    predicted_bytes: None,
                    accepted,
                });
                if !accepted {
                    self.defer(stage, candidate, false);
                    return Ok(());
                }
            }
            self.full(stage, candidate, false, false)?;
        }
        if stage == 10 {
            let deeper = self.deep_winner(stage, false)?;
            self.full(stage, deeper, false, false)?;
        }
        Ok(())
    }
    fn racing_stage(&mut self, stage: u8) -> Result<()> {
        if stage == 5 && self.pending_automatic {
            self.pending_automatic = false;
            let candidate = self.input.backend(9)?;
            if self
                .refine(stage, candidate.clone(), false, false, false)?
                .is_none()
            {
                self.defer(stage, candidate, false);
            }
        }
        let index = usize::from(stage - 4);
        if let Some(candidate) = self.frontier.get(index).cloned()
            && self.confirm(stage, &candidate.params)?
        {
            let neighborhood = self.routes.preflight_yield
                && candidate.params
                    == winner_neighborhood(
                        &self.best_params(stage + 1)?,
                        self.image.channel_count(),
                    );
            self.refine(
                stage,
                candidate.params,
                false,
                stage == 4 && !self.early_family_winner,
                neighborhood,
            )?;
        }
        if stage == 9 {
            let incumbent = self.best_params(stage + 1)?;
            let geometry = winner_neighborhood(&incumbent, self.image.channel_count());
            if self.routes.broad_scope && self.early_family_winner {
                self.defer(stage, geometry, false);
            } else {
                self.refine(stage, geometry, false, false, true)?;
            }
        }
        if stage == 10 {
            self.refine(stage, self.input.backend(10)?, false, false, false)?;
            let deeper = self.deep_winner(stage, true)?;
            self.refine(stage, deeper, false, false, false)?;
        }
        Ok(())
    }
    pub(super) fn run(mut self) -> Result<SearchResult<A>> {
        if self.run_scout()? {
            return self.finish();
        }
        if self.run_tournament()? {
            return self.finish();
        }
        let initial =
            PixelSearch::new(self.image, self.input.clone(), self.scan)?.initial_plan()?;
        let mut regular_size = None;
        let mut literal_size = None;
        let mut literal_present = false;
        let sizes = self.initial_full(&initial)?;
        for (request, size) in initial.into_iter().zip(sizes) {
            if request.stage == 1 {
                regular_size = size;
            }
            if request.family == TrialFamily::Literal {
                literal_size = size;
                literal_present = true;
            }
        }
        let refine_winner = refinable(self.image);
        let mut learned_size = None;
        let mut refined_size = None;
        let mut fast_literal = false;
        for stage in 3..=self.level {
            if stage == 10 {
                // Preserve rejected refinements at the deepest effort; measured
                // or byte-bounded trials never need to run again.
                for deferred in std::mem::take(&mut self.deferred) {
                    if self.search.find(&deferred.params).is_some() {
                        continue;
                    }
                    if deferred.screen {
                        self.refine(stage, deferred.params, false, false, true)?;
                    } else {
                        self.full(stage, deferred.params, false, false)?;
                    }
                }
            }
            if stage == 4 {
                let fast_family = self.options.fast_family
                    || self.options.palette_race
                    || self.options.palette_preselect;
                fast_literal = fast_family
                    && !self.early_family_winner
                    && self.screen.available()
                    && matches!((literal_size, regular_size), (Some(l), Some(r)) if l >= 1024 && (l as f64) < r as f64 * 0.8);
                if fast_family {
                    self.events.push(SearchEvent::FastLiteral {
                        enabled: fast_literal,
                    });
                }
                self.early_racing =
                    self.screen.available() && self.search.competitive(literal_size, stage, 1.35);
            }
            if fast_literal {
                self.fast_literal_stage(stage)?;
                continue;
            }
            if stage == 3 {
                self.plan_first_stage(regular_size, literal_size)?;
            }
            if self.early_racing && stage >= 4 {
                if stage == 4 {
                    self.prepare_racing_frontier(literal_present, regular_size, literal_size)?;
                }
                self.racing_stage(stage)?;
                continue;
            }
            if stage == 8 && !self.early_family_winner {
                let best_literal = minimum(literal_size, learned_size);
                let ambiguous = matches!((regular_size, best_literal), (Some(a), Some(b)) if a.max(b) as f64 <= a.min(b) as f64 * 1.35);
                let incumbent = self.best_params(stage)?;
                self.analyze(&incumbent, ambiguous)?;
            }
            let backend = match stage {
                3 => Some(7),
                5 if !refine_winner => Some(8),
                7 => Some(9),
                10 => Some(10),
                _ => None,
            };
            if let Some(effort) = backend
                && !(stage == 3 && self.early_family_winner)
                && (stage == 3 || self.search.competitive(regular_size, stage, 1.35))
            {
                let size = self.refine(stage, self.input.backend(effort)?, true, false, false)?;
                regular_size = minimum(regular_size, size);
            }
            if stage == 4 && self.search.competitive(literal_size, stage, 1.35) {
                let size = self.refine(stage, literal(self.input, 8, 0.0)?, true, false, false)?;
                literal_size = minimum(literal_size, size);
            }
            if stage == 5 && refine_winner {
                if self.middle_preflight(&mut regular_size)? {
                    continue;
                }
                let incumbent = self.best_params(stage)?;
                self.refine(
                    stage,
                    winner_neighborhood(&incumbent, self.image.channel_count()),
                    false,
                    false,
                    true,
                )?;
            }
            if stage == 6 && self.search.competitive(literal_size, stage, 1.35) {
                learned_size =
                    self.refine(stage, literal(self.input, 9, 0.01)?, true, false, false)?;
            }
            if stage == 8 && self.search.competitive(learned_size, stage, 1.02) {
                refined_size =
                    self.refine(stage, literal(self.input, 9, 0.1)?, true, false, false)?;
            }
            if stage == 9
                && matches!((refined_size, learned_size), (Some(r), Some(l)) if (r as f64) < l as f64 * 0.99)
            {
                self.refine(stage, literal(self.input, 9, 1.0)?, true, false, false)?;
            }
            if stage >= 8
                && let Some(candidate) = self.analysis.ranked.get(usize::from(stage - 8)).cloned()
                && self.confirm(stage, &candidate.params)?
            {
                self.refine(
                    stage,
                    candidate.params,
                    false,
                    stage == 8 && candidate.exploratory,
                    false,
                )?;
            }
            if stage == 9 && refine_winner {
                let incumbent = self.best_params(stage + 1)?;
                self.refine(
                    stage,
                    winner_neighborhood(&incumbent, self.image.channel_count()),
                    false,
                    false,
                    true,
                )?;
            }
            if stage == 10 && self.analysis.samples != 0 {
                let deeper = self.deep_winner(stage, true)?;
                if self.confirm(stage, &deeper)? {
                    self.refine(stage, deeper, false, false, false)?;
                }
            }
        }
        self.finish()
    }
}
