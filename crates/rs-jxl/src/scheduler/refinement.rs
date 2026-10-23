// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! b866b3d enc_lossless_search.cc:879..1534, the complete refine lambda.
//! Driver resolves source environment/geometry gates into ActiveRoutes; this
//! module retains the source branch order and exact deferred-replay flags.
//! Context scoring must share the driver's cumulative probe budget. Estimates
//! only defer/reject heuristic proposals; they never become encoded-byte bounds.

use super::super::calibration::CalibrationFeedback;
use super::*;

#[path = "refinement_evidence.rs"]
mod evidence;

#[cfg(test)]
#[path = "refinement_tests.rs"]
mod tests;

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    fn medium_check(
        &mut self,
        stage: u8,
        candidate: &CandidateParams,
        scores: &[usize],
    ) -> Result<bool> {
        let medium = candidate.backend(7)?;
        let bytes = self.calibration_full(stage, medium.clone())?;
        let medium_scores = self.screen.score(&medium)?;
        let medium_total = total(&medium_scores)?;
        if self.options.sibling_transfer {
            self.calibration_feedback.push(CalibrationFeedback {
                params: candidate.clone(),
                deep: scores.to_vec(),
                cheap: medium_scores,
                full_bytes: bytes,
            });
        }
        let score_total = total(scores)?;
        let prediction =
            (medium_total != 0).then(|| bytes as f64 * score_total as f64 / medium_total as f64);
        let best = self.best_size(stage + 1)?;
        let keep = prediction.is_none_or(|p| p <= best as f64 * 1.01);
        self.events.push(SearchEvent::Calibration {
            stage,
            params: candidate.clone(),
            kind: "medium-e7",
            predicted_bytes: prediction,
            accepted: keep,
        });
        Ok(keep)
    }

    pub(super) fn refine(
        &mut self,
        stage: u8,
        candidate: CandidateParams,
        baseline: bool,
        exploration: bool,
        neighborhood: bool,
    ) -> Result<Option<usize>> {
        self.refine_promoted(stage, candidate, baseline, exploration, neighborhood, false)
    }

    // The six source lambda arguments are separate routing decisions, not a
    // replacement policy object; keep the existing wrapper for ordinary calls.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn refine_promoted(
        &mut self,
        stage: u8,
        candidate: CandidateParams,
        baseline: bool,
        exploration: bool,
        neighborhood: bool,
        early_promotion: bool,
    ) -> Result<Option<usize>> {
        if let Some(index) = self.search.find(&candidate) {
            self.events.push(SearchEvent::Reused {
                stage,
                trial: index,
            });
            return Ok(self.search.trials()[index].bytes);
        }
        if self.options.sibling_race && (4..10).contains(&stage) {
            let incumbent = self.best_params(stage + 1)?;
            let mut sibling = candidate.clone();
            sibling.rct = incumbent.rct;
            sibling.group_size_shift = incumbent.group_size_shift;
            if incumbent.speed_tier <= 1
                && incumbent.predictor == Some(crate::modular::predictor::Predictor::Variable)
                && incumbent.palette_colors == 0
                && sibling == incumbent
            {
                let mut control = incumbent.backend(4)?;
                let mut proposal = candidate.backend(4)?;
                control.learning_fraction = control.learning_fraction.min(0.01);
                proposal.learning_fraction = proposal.learning_fraction.min(0.01);
                let control_bytes = self.calibration_full(stage, control)?;
                let proposal_bytes = self.calibration_full(stage, proposal)?;
                let deferred =
                    control_bytes != 0 && proposal_bytes as f64 > control_bytes as f64 * 1.0025;
                self.events.push(SearchEvent::SiblingRace {
                    stage,
                    params: candidate.clone(),
                    control_bytes,
                    proposal_bytes,
                    deferred,
                });
                if deferred {
                    self.defer(stage, candidate, false);
                    return Ok(None);
                }
            }
        }
        let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
        let preflight_yield = self.routes.preflight_yield;
        if preflight_yield && pixels < 1 << 19 && candidate == self.input.backend(9)? {
            return self.full(stage, candidate, true, false);
        }
        if (stage > 3 || early_promotion) && self.screen.available() {
            let incumbent_params = self.best_params(stage + 1)?;
            let incumbent = self.screen.score(&incumbent_params)?;
            let scores = self.screen.score(&candidate)?;
            let incumbent_total = total(&incumbent)?;
            let score_total = total(&scores)?;
            let mut promising = super::super::probe::promising(&scores, &incumbent);
            let mut same_rct_family = candidate.clone();
            same_rct_family.rct = incumbent_params.rct;
            let paired_rct = (self.routes.capacity_mismatch || self.routes.broad_scope)
                && stage < 10
                && incumbent_params.speed_tier <= 1
                && candidate.palette_colors == 0
                && candidate.predictor == Some(crate::modular::predictor::Predictor::Variable)
                && same_rct_family == incumbent_params;
            let mut paired_rct_keep = !paired_rct;
            if paired_rct && (!self.routes.broad_scope || promising || exploration) {
                let control = self.context_screen.score(&incumbent_params)?;
                let proposed = self.context_screen.score(&candidate)?;
                let control_total = total(&control)?;
                let proposed_total = total(&proposed)?;
                paired_rct_keep = (proposed_total as f64) < control_total as f64 * 0.998
                    || (score_total as f64) < incumbent_total as f64 * 0.99;
                let mut relative = None;
                if self.routes.multiscale_rank && control_total != 0 && incumbent_total != 0 {
                    let ratio = proposed_total as f64 / control_total as f64 * score_total as f64
                        / incumbent_total as f64;
                    paired_rct_keep = ratio < 0.995;
                    relative = Some(ratio);
                }
                if self.routes.broad_scope {
                    promising &= paired_rct_keep;
                } else {
                    promising = paired_rct_keep;
                }
                self.refinement_event(
                    stage,
                    &candidate,
                    "paired-rct-context",
                    relative,
                    paired_rct_keep,
                );
            }
            if neighborhood && score_total as f64 <= incumbent_total as f64 * 1.05 {
                promising = true;
            }
            let related = self.search.related(&candidate);
            let mut predicted = None;
            if let Some(index) = related {
                let reference_params = self.search.trials()[index].params.clone();
                let reference_total = total(&self.screen.score(&reference_params)?)?;
                if reference_total != 0 {
                    let measured = self.search.trials()[index].outcome_size().ok_or_else(|| {
                        Error::Encoding("related trial has no measured outcome".into())
                    })?;
                    let estimate = measured as f64 * score_total as f64 / reference_total as f64;
                    let best = self.best_size(stage + 1)? as f64;
                    promising |= score_total as f64 <= incumbent_total as f64 * 1.005
                        && estimate <= best * 1.005;
                    if estimate > best * 1.05 {
                        promising = false;
                    }
                    predicted = Some(estimate);
                }
            }
            let accepted = (promising && (!paired_rct || paired_rct_keep)) || exploration;
            let calibration_context = self.best_size(stage + 1)? > incumbent_total;
            self.events.push(SearchEvent::Screened {
                stage,
                params: candidate.clone(),
                candidate_bytes: score_total,
                incumbent_bytes: incumbent_total,
                related,
                predicted_bytes: predicted,
                accepted,
                exploration,
                neighborhood,
            });
            if !accepted {
                if paired_rct {
                    self.defer(stage, candidate, false);
                }
                return Ok(None);
            }

            if self.options.sibling_transfer
                && stage < 10
                && !exploration
                && !neighborhood
                && incumbent_params.speed_tier <= 1
                && incumbent_params.palette_colors > 0
            {
                for feedback in &self.calibration_feedback {
                    let Some(predicted) = feedback.sibling_estimate(&candidate, &scores) else {
                        continue;
                    };
                    if predicted <= self.best_size(stage + 1)? as f64 * 1.20 {
                        continue;
                    }
                    self.refinement_event(
                        stage,
                        &candidate,
                        "sibling-transfer",
                        Some(predicted),
                        false,
                    );
                    self.defer(stage, candidate, false);
                    return Ok(None);
                }
            }

            let context_validated = self.routes.context_feedback
                && self.early_racing
                && stage < 10
                && pixels < 1 << 19
                && calibration_context
                && candidate.palette_colors == 0
                && candidate.predictor.is_some()
                && !neighborhood
                && incumbent_params == self.input.backend(9)?;
            if context_validated {
                let control = self.context_screen.score(&incumbent_params)?;
                let proposed = self.context_screen.score(&candidate)?;
                let defer = evidence::context_defers(&scores, &incumbent, &proposed, &control)?;
                self.refinement_event(stage, &candidate, "context-validation", None, !defer);
                if defer {
                    self.defer(stage, candidate, false);
                    return Ok(None);
                }
            }

            if preflight_yield
                && stage < 10
                && neighborhood
                && (1 << 19..2 << 20).contains(&pixels)
                && incumbent_params == self.input.backend(9)?
                && score_total as f64 >= incumbent_total as f64 * 0.98
            {
                self.refinement_event(stage, &candidate, "compact-neighborhood", None, false);
                self.defer(stage, candidate, true);
                return Ok(None);
            }

            let settings = self.options.calibration;
            let lower_backend = self.search.lower_backend(&candidate);
            if !context_validated
                && settings.enabled
                && settings.routes
                && stage < 10
                && candidate.speed_tier <= 1
                && let Some(index) = lower_backend
            {
                let params = self.search.trials()[index].params.clone();
                let reference_total = total(&self.screen.score(&params)?)?;
                if reference_total != 0 {
                    let bytes = self.search.trials()[index]
                        .bytes
                        .ok_or_else(|| Error::Encoding("lower backend is incomplete".into()))?;
                    let prediction = bytes as f64 * score_total as f64 / reference_total as f64;
                    let best = self.best_size(stage + 1)?;
                    let comparable_peer = preflight_yield
                        && incumbent_params.speed_tier == candidate.speed_tier
                        && candidate.predictor.is_none()
                        && incumbent_params.predictor.is_none()
                        && candidate.rct == incumbent_params.rct
                        && candidate.palette_colors == incumbent_params.palette_colors;
                    let deep_peer_gain =
                        comparable_peer && (score_total as f64) < incumbent_total as f64 * 0.995;
                    if self.routes.automatic_yield
                        && pixels >= 2 << 20
                        && incumbent_params.speed_tier <= 1
                        && candidate == self.input.backend(9)?
                        && prediction > best as f64 * 1.0025
                    {
                        // Compare against the SAME completed automatic parent,
                        // not the current winner's unrelated representation.
                        let parent = self.context_screen.score(&params)?;
                        let proposed = self.context_screen.score(&candidate)?;
                        let denominator = total(&parent)?;
                        let context_prediction = if denominator == 0 {
                            0.0
                        } else {
                            bytes as f64 * total(&proposed)? as f64 / denominator as f64
                        };
                        let defer = parent.len() >= 3
                            && proposed.len() == parent.len()
                            && context_prediction > best as f64 * 1.0025;
                        self.refinement_event(
                            stage,
                            &candidate,
                            "automatic-yield",
                            Some(context_prediction),
                            !defer,
                        );
                        if defer {
                            self.defer(stage, candidate, false);
                            return Ok(None);
                        }
                    }
                    let peer_loss_proposal = self.routes.context_feedback
                        && comparable_peer
                        && pixels >= 2 << 20
                        && candidate == self.input.backend(9)?
                        && score_total as f64 > incumbent_total as f64 * 1.01;
                    let mut deep_peer_loss = false;
                    if peer_loss_proposal {
                        let control = self.context_screen.score(&incumbent_params)?;
                        let proposed = self.context_screen.score(&candidate)?;
                        deep_peer_loss = evidence::peer_context_loses(&proposed, &control)?;
                        self.refinement_event(
                            stage,
                            &candidate,
                            "peer-context",
                            None,
                            !deep_peer_loss,
                        );
                    }
                    let keep =
                        (prediction <= best as f64 * 1.02 || deep_peer_gain) && !deep_peer_loss;
                    self.events.push(SearchEvent::Calibration {
                        stage,
                        params: candidate.clone(),
                        kind: "lower-backend",
                        predicted_bytes: Some(prediction),
                        accepted: keep,
                    });
                    if !keep {
                        self.defer(stage, candidate, !deep_peer_loss);
                        return Ok(None);
                    }
                }
            }

            let mut same_geometry = candidate.clone();
            same_geometry.group_size_shift = incumbent_params.group_size_shift;
            let mut same_neighborhood = same_geometry.clone();
            same_neighborhood.speed_tier = incumbent_params.speed_tier;
            same_neighborhood.learning_fraction = incumbent_params.learning_fraction;
            same_neighborhood.max_properties = incumbent_params.max_properties;
            let paired_neighborhood = preflight_yield && same_neighborhood == incumbent_params;
            if self.routes.plan_first
                && stage < 10
                && neighborhood
                && incumbent_params.speed_tier <= 1
                && same_geometry == incumbent_params
                && score_total >= incumbent_total
            {
                if !self.deferred.iter().any(|trial| trial.params == candidate) {
                    self.defer(stage, candidate.clone(), false);
                }
                self.refinement_event(stage, &candidate, "marginal-geometry", None, false);
                return Ok(None);
            }

            let raw = related.is_none()
                && candidate.speed_tier <= 1
                && candidate.palette_colors == 0
                && candidate.predictor.is_some()
                && candidate.learning_fraction > 0.0;
            let palette = settings.routes
                && stage < 10
                && related.is_none()
                && candidate.speed_tier <= 1
                && incumbent_params.speed_tier <= 1
                && (incumbent_params.palette_colors == 0 || stage >= 8)
                && candidate.palette_colors > 0
                && candidate.predictor.is_some()
                && candidate.learning_fraction > 0.0;
            let mut cheap_control = incumbent_params.backend(4)?;
            cheap_control.learning_fraction = cheap_control.learning_fraction.min(0.01);
            let reuse_cheap_control = self.routes.calibration_reuse
                && paired_neighborhood
                && self.search.find(&cheap_control).is_some();
            let cheap_palette =
                palette && candidate.predictor == Some(crate::modular::predictor::Predictor::Zero);
            let closer_first = self.routes.calibration_order
                && stage == 9
                && neighborhood
                && paired_neighborhood
                && reuse_cheap_control
                && same_geometry == incumbent_params
                && score_total as f64 <= incumbent_total as f64 * 1.01;
            let medium_first = !context_validated
                && preflight_yield
                && settings.enabled
                && settings.routes
                && (closer_first
                    || !(self.routes.calibration_reuse && (cheap_palette || reuse_cheap_control)))
                && calibration_context
                && stage < 10
                && candidate.speed_tier <= 1
                && incumbent_params.speed_tier <= 1
                && score_total as f64 >= incumbent_total as f64 * 0.98
                && ((neighborhood && paired_neighborhood) || raw || palette);
            if medium_first && !self.medium_check(stage, &candidate, &scores)? {
                self.defer(stage, candidate, true);
                return Ok(None);
            }
            if !context_validated
                && settings.enabled
                && settings.routes
                && calibration_context
                && (!medium_first || closer_first)
                && neighborhood
                && stage < 10
                && candidate.speed_tier <= 1
                && (paired_neighborhood
                    || (candidate.group_size_shift != incumbent_params.group_size_shift
                        && same_geometry == incumbent_params))
            {
                let mut cheap_candidate = candidate.backend(4)?;
                cheap_candidate.learning_fraction = cheap_candidate.learning_fraction.min(0.01);
                let control_bytes = self.calibration_full(stage, cheap_control.clone())?;
                let candidate_bytes = self.calibration_full(stage, cheap_candidate.clone())?;
                let strong_gain = (score_total as f64) < incumbent_total as f64 * 0.95;
                let mut keep = candidate_bytes as f64 <= control_bytes as f64 * 1.01 || strong_gain;
                let mut relative = None;
                if keep && !strong_gain {
                    let control_total = total(&self.screen.score(&cheap_control)?)?;
                    let candidate_total = total(&self.screen.score(&cheap_candidate)?)?;
                    if control_bytes != 0
                        && candidate_total != 0
                        && incumbent_total != 0
                        && control_total != 0
                    {
                        let ratio = candidate_bytes as f64 / control_bytes as f64
                            * (score_total as f64 / incumbent_total as f64)
                            * (control_total as f64 / candidate_total as f64);
                        keep = ratio <= 1.01
                            || (self.routes.calibration_reuse
                                && candidate_bytes < control_bytes
                                && ratio <= 1.05);
                        relative = Some(ratio);
                    }
                }
                self.events.push(SearchEvent::Calibration {
                    stage,
                    params: candidate.clone(),
                    kind: "paired-geometry-ratio",
                    predicted_bytes: relative,
                    accepted: keep,
                });
                if !keep {
                    self.defer(stage, candidate, true);
                    return Ok(None);
                }
                if !medium_first
                    && paired_neighborhood
                    && !strong_gain
                    && incumbent_params.speed_tier <= 1
                    && !self.medium_check(stage, &candidate, &scores)?
                {
                    self.defer(stage, candidate, true);
                    return Ok(None);
                }
            }

            let refinement = settings.refinements
                && stage < 10
                && !(preflight_yield && lower_backend.is_some())
                && candidate.speed_tier <= 1
                && incumbent_params.speed_tier <= 1
                && candidate.predictor.is_none()
                && incumbent_params.predictor.is_none()
                && candidate.palette_colors == incumbent_params.palette_colors
                && candidate.rct == incumbent_params.rct
                && incumbent_params.learning_fraction > 0.0
                && candidate.learning_fraction >= 3.0 * incumbent_params.learning_fraction;
            let strong_small_candidate = preflight_yield
                && pixels < 1 << 19
                && (score_total as f64) < incumbent_total as f64 * 0.95;
            let committed_early = self.routes.plan_first && stage == 3 && early_promotion;
            if !context_validated
                && settings.enabled
                && calibration_context
                && !medium_first
                && !strong_small_candidate
                && !committed_early
                && (raw || refinement || palette)
            {
                let mut cheap = candidate.backend(if settings.fast { 4 } else { 6 })?;
                cheap.learning_fraction = cheap.learning_fraction.min(0.01);
                if self.options.statistical_feedback
                    && palette
                    && let Some(peer) = self.search.palette_calibration_peer(&cheap)
                {
                    let peer_params = self.search.trials()[peer].params.clone();
                    let peer_scores = self.screen.score(&peer_params)?;
                    let peer_bytes = self.search.trials()[peer].bytes.ok_or_else(|| {
                        Error::Encoding("palette calibration peer is incomplete".into())
                    })?;
                    let prediction =
                        evidence::conservative_projection(peer_bytes, &scores, &peer_scores);
                    let defer = prediction > self.best_size(stage + 1)? as f64 * 1.15;
                    self.refinement_event(
                        stage,
                        &candidate,
                        "statistical-peer",
                        Some(prediction),
                        !defer,
                    );
                    if defer {
                        self.defer(stage, candidate, true);
                        return Ok(None);
                    }
                }
                let bytes = self.calibration_full(stage, cheap.clone())?;
                let cheap_scores = self.screen.score(&cheap)?;
                let cheap_total = total(&cheap_scores)?;
                let mut ratio = 1.0f64;
                if cheap_total != 0 {
                    ratio = score_total as f64 / cheap_total as f64;
                    for (&score, &cheap) in scores.iter().zip(&cheap_scores) {
                        if cheap != 0 && cheap >= cheap_total / 16 {
                            ratio = ratio.min(score as f64 / cheap as f64);
                        }
                    }
                }
                let prediction = bytes as f64 * ratio;
                let best = self.best_size(stage + 1)?;
                let keep = cheap_total == 0
                    || prediction <= best as f64 * if early_promotion { 0.97 } else { 1.15 };
                self.events.push(SearchEvent::Calibration {
                    stage,
                    params: candidate.clone(),
                    kind: "cheap-family",
                    predicted_bytes: Some(prediction),
                    accepted: keep,
                });
                if !keep {
                    if refinement || palette {
                        self.defer(stage, candidate, palette);
                    }
                    return Ok(None);
                }
                if settings.routes
                    && stage < 10
                    && incumbent_params.speed_tier <= 1
                    && (prediction > best as f64 * 1.01
                        || (preflight_yield && score_total as f64 >= incumbent_total as f64 * 0.98))
                    && (raw || palette)
                    && !self.medium_check(stage, &candidate, &scores)?
                {
                    self.defer(stage, candidate, true);
                    return Ok(None);
                }
            }
        }
        // A fixed-tree outcome cannot anchor a learned representation. Finish
        // its first admitted trial even when it already appears to lose.
        let anchor = self.screen.available() && self.search.related(&candidate).is_none();
        self.full(stage, candidate, baseline || anchor, false)
    }

    fn refinement_event(
        &mut self,
        stage: u8,
        candidate: &CandidateParams,
        kind: &'static str,
        predicted_bytes: Option<f64>,
        accepted: bool,
    ) {
        self.events.push(SearchEvent::Calibration {
            stage,
            params: candidate.clone(),
            kind,
            predicted_bytes,
            accepted,
        });
    }
}
