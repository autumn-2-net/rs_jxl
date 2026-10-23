// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! `enc_lossless_probe.cc` calibration and diverse-frontier helpers.
//! All returned estimates are rankings, never encoded-byte lower bounds.

use super::params::{Candidate, CandidateParams};
use crate::modular::predictor::Predictor;

pub fn calibration_agreement(projected: &[f64], full_bytes: &[usize], leader: usize) -> bool {
    if projected.len() < 2 || projected.len() != full_bytes.len() || leader >= projected.len() {
        return false;
    }
    for i in 0..projected.len() {
        if projected[i] <= 0.0 || !projected[i].is_finite() || full_bytes[i] == 0 {
            return false;
        }
        if i != leader
            && !(projected[leader] < projected[i] * 0.998
                && (full_bytes[leader] as f64) < full_bytes[i] as f64 * 0.99)
        {
            return false;
        }
    }
    true
}

pub fn shrink_calibration(
    plans: &[CandidateParams],
    full_bytes: &[usize],
    projected: &[f64],
    independent_regions: &[usize],
) -> Vec<f64> {
    let mut result = projected.to_vec();
    if plans.len() != full_bytes.len()
        || plans.len() != projected.len()
        || plans.len() != independent_regions.len()
    {
        return result;
    }
    for i in 0..plans.len() {
        if full_bytes[i] == 0
            || projected[i] <= 0.0
            || !projected[i].is_finite()
            || plans[i].palette_colors != 0
            || plans[i].rct.is_none()
            || plans[i].predictor != Some(Predictor::Variable)
        {
            continue;
        }
        let mut ratios = Vec::new();
        for j in 0..plans.len() {
            let mut sibling = plans[j].clone();
            sibling.rct = plans[i].rct;
            if plans[j].rct.is_none()
                || full_bytes[j] == 0
                || projected[j] <= 0.0
                || !projected[j].is_finite()
                || plans[i] != sibling
            {
                continue;
            }
            ratios.push(projected[j] / full_bytes[j] as f64);
        }
        if ratios.len() < 2 {
            continue;
        }
        ratios.sort_by(f64::total_cmp);
        let median = (ratios[(ratios.len() - 1) / 2] + ratios[ratios.len() / 2]) * 0.5;
        let confidence = independent_regions[i] as f64 / (independent_regions[i] as f64 + 4.0);
        result[i] = full_bytes[i] as f64
            * (median + confidence * (projected[i] / full_bytes[i] as f64 - median));
    }
    result
}

#[derive(Clone, Debug)]
pub struct CalibrationFeedback {
    pub params: CandidateParams,
    pub deep: Vec<usize>,
    pub cheap: Vec<usize>,
    pub full_bytes: usize,
}
impl CalibrationFeedback {
    /// None replaces C++ zero's "insufficient evidence" sentinel.
    pub fn sibling_estimate(&self, candidate: &CandidateParams, scores: &[usize]) -> Option<f64> {
        if self.full_bytes == 0
            || self.deep.len() < 3
            || self.cheap.len() != self.deep.len()
            || scores.len() != self.deep.len()
            || candidate.palette_colors != 0
            || candidate.rct.is_none()
            || self.params.rct.is_none()
            || candidate.predictor != Some(Predictor::Variable)
        {
            return None;
        }
        let mut sibling = candidate.clone();
        sibling.rct = self.params.rct;
        if sibling != self.params {
            return None;
        }
        if scores
            .iter()
            .zip(&self.deep)
            .any(|(&s, &d)| (s as f64) < d as f64 * 0.95)
        {
            return None;
        }
        let denominator: f64 = self.cheap.iter().map(|&v| v as f64).sum();
        if denominator == 0.0 {
            return None;
        }
        Some(self.full_bytes as f64 * scores.iter().map(|&v| v as f64).sum::<f64>() / denominator)
    }
}

fn same_family(a: &CandidateParams, b: &CandidateParams) -> bool {
    a.buffering == b.buffering
        && a.responsive == b.responsive
        && a.group_size_shift == b.group_size_shift
        && a.rct == b.rct
        && (a.palette_colors > 0) == (b.palette_colors > 0)
        && a.channel_colors_percent == b.channel_colors_percent
        && a.channel_colors_pre_transform_percent == b.channel_colors_pre_transform_percent
        && a.predictor == b.predictor
        && a.max_properties == b.max_properties
        && a.patches == b.patches
        && a.delta_palette == b.delta_palette
        && a.squeeze == b.squeeze
}
pub fn diverse_frontier(ranked: Vec<Candidate>) -> Vec<Candidate> {
    let mut representatives: Vec<usize> = Vec::new();
    for i in 0..ranked.len() {
        if let Some(found) = representatives
            .iter()
            .position(|&j| same_family(&ranked[i].params, &ranked[j].params))
        {
            let previous = &ranked[representatives[found]];
            if let (Some(current), Some(old)) = (ranked[i].context_bytes, previous.context_bytes)
                && current as f64 <= old as f64 * 1.0025
                && ranked[i].params.learning_fraction < previous.params.learning_fraction
            {
                representatives[found] = i;
            }
        } else {
            representatives.push(i);
        }
    }
    let mut selected = vec![false; ranked.len()];
    let mut result = Vec::with_capacity(ranked.len());
    for i in representatives {
        selected[i] = true;
        result.push(ranked[i].clone());
    }
    for (i, c) in ranked.into_iter().enumerate() {
        if !selected[i] {
            result.push(c);
        }
    }
    result
}
