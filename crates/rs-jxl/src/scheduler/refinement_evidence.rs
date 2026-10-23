// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! b866b3d enc_lossless_racing.h::PairedLosslessEvidence /
//! ConservativeLosslessProjection and the refine lambda's paired context gates.
//! These are heuristic projections, not confidence guarantees or byte bounds.

use super::super::super::probe::total;
use crate::{Error, Result};

pub(super) fn conservative_projection(
    reference_bytes: usize,
    candidate: &[usize],
    reference: &[usize],
) -> f64 {
    if candidate.len() != reference.len() || candidate.len() < 3 {
        return 0.0;
    }
    let mut count = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut minimum = f64::INFINITY;
    for (&candidate, &reference) in candidate.iter().zip(reference) {
        // usize inputs are finite and nonnegative; source rejects zero peers.
        if reference == 0 {
            return 0.0;
        }
        let delta = candidate as f64 / reference as f64 - 1.0;
        if !delta.is_finite() {
            return 0.0;
        }
        count += 1;
        let old = mean;
        mean += (delta - old) / count as f64;
        m2 += (delta - old) * (delta - mean);
        minimum = minimum.min(delta);
    }
    let radius = 0.10 + 2.5 * (m2.max(0.0) / (count as f64 * (count - 1) as f64)).sqrt();
    if !radius.is_finite() {
        return 0.0;
    }
    reference_bytes as f64 * (1.0 + minimum.min(mean - radius)).max(0.0)
}

pub(super) fn context_defers(
    tiny: &[usize],
    tiny_control: &[usize],
    proposed: &[usize],
    control: &[usize],
) -> Result<bool> {
    if proposed.len() != control.len() || tiny.len() != tiny_control.len() {
        return Err(Error::Encoding("paired refinement regions differ".into()));
    }
    let tiny_total = total(tiny_control)?;
    let control_total = total(control)?;
    let tiny_ratio = if tiny_total == 0 {
        0.0
    } else {
        total(tiny)? as f64 / tiny_total as f64
    };
    let context_ratio = if control_total == 0 {
        0.0
    } else {
        total(proposed)? as f64 / control_total as f64
    };
    let local_win = proposed
        .iter()
        .zip(control)
        .any(|(&a, &b)| (a as f64) < b as f64 * 0.97);
    let disappearing_gain =
        context_ratio >= 0.97 && context_ratio - tiny_ratio > 3.0 * (1.0 - context_ratio).max(0.0);
    let weak_at_both_scales = tiny_ratio >= 0.98 && context_ratio >= 0.98;
    Ok(!local_win && (disappearing_gain || weak_at_both_scales || context_ratio >= 1.01))
}

pub(super) fn peer_context_loses(proposed: &[usize], control: &[usize]) -> Result<bool> {
    if proposed.len() != control.len() {
        return Err(Error::Encoding("paired refinement regions differ".into()));
    }
    Ok(total(proposed)? as f64 > total(control)? as f64 * 1.01
        && proposed
            .iter()
            .zip(control)
            .all(|(&a, &b)| a as f64 >= b as f64 * 0.995))
}
