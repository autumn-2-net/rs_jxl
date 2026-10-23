// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Explicit source experiment switches and their dependent activation gates.

use super::{DriverOptions, LosslessScan};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScopeRouting {
    #[default]
    Disabled,
    Narrow,
    Broad,
}

#[derive(Clone, Copy, Debug)]
pub struct PreflightOptions {
    pub enabled: bool,
    pub anchor_first: bool,
    pub scope: ScopeRouting,
    pub multiscale_rank: bool,
    pub calibration_reuse: bool,
    pub early_family: bool,
    pub calibration_order: bool,
    pub plan_first: bool,
    pub plan_buckets: bool,
    pub geometry_first: bool,
    pub automatic_yield: bool,
    pub context_feedback: bool,
}
impl Default for PreflightOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            anchor_first: false,
            scope: ScopeRouting::Disabled,
            multiscale_rank: false,
            calibration_reuse: false,
            early_family: false,
            calibration_order: false,
            plan_first: false,
            plan_buckets: false,
            geometry_first: true,
            automatic_yield: true,
            context_feedback: true,
        }
    }
}

/// JXL_LOSSLESS_DIRECT_FRONTIER values 0..5. All are opt-in heuristics, not
/// encoded-byte bounds or a promise of effort-prefix dominance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirectFrontier {
    #[default]
    Disabled,
    Sampled,
    Calibrated,
    Extended,
    Shrink,
    Dense,
}
impl DirectFrontier {
    pub(super) fn extended(self) -> bool {
        matches!(self, Self::Extended | Self::Shrink | Self::Dense)
    }
    pub(super) fn calibrated(self) -> bool {
        self == Self::Calibrated || self.extended()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrontierOptions {
    pub direct: DirectFrontier,
    pub compact: bool,
    pub medium_calibration: bool,
    pub global_tournament: bool,
    pub strip_tournament: bool,
    pub parallel_finalists: bool,
}

/// Source enc_lossless_workers.h. Scratch is a concurrency estimate, not a
/// hard RSS limit; a single larger job is still allowed to execute.
#[derive(Clone, Copy, Debug)]
pub struct WorkerLimits {
    pub tasks: usize,
    pub scratch_bytes: u64,
}
impl Default for WorkerLimits {
    fn default() -> Self {
        Self {
            tasks: 8,
            scratch_bytes: 1 << 30,
        }
    }
}
impl WorkerLimits {
    pub fn validate(self) -> Result<()> {
        if !(1..=64).contains(&self.tasks) || self.scratch_bytes == 0 {
            return Err(Error::InvalidInput(
                "worker tasks must be 1..64 and scratch estimate nonzero".into(),
            ));
        }
        Ok(())
    }
    pub fn concurrency(self, jobs: usize, scratch_per_task: u64) -> usize {
        let memory = self
            .scratch_bytes
            .checked_div(scratch_per_task)
            .map_or(self.tasks as u64, |n| n.max(1));
        jobs.min(self.tasks)
            .min(memory.min(usize::MAX as u64) as usize)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ActiveRoutes {
    pub preflight_yield: bool,
    pub anchor_first: bool,
    pub broad_scope: bool,
    pub multiscale_rank: bool,
    pub calibration_reuse: bool,
    pub early_family: bool,
    pub calibration_order: bool,
    pub plan_first: bool,
    pub plan_buckets: bool,
    pub geometry_first: bool,
    pub automatic_yield: bool,
    pub context_feedback: bool,
    pub scope_routing: bool,
    pub capacity_mismatch: bool,
}
impl ActiveRoutes {
    pub fn new(options: &DriverOptions, pixels: u64, scan: Option<&LosslessScan>) -> Self {
        let p = options.preflight;
        let broad_scope = p.scope == ScopeRouting::Broad;
        let reuse = p.enabled
            && pixels >= if broad_scope { 1 << 19 } else { 2 << 20 }
            && p.calibration_reuse;
        let early = reuse
            && options.calibration.enabled
            && options.calibration.fast
            && options.calibration.routes
            && p.early_family;
        let plan = early && p.plan_first;
        Self {
            preflight_yield: p.enabled,
            anchor_first: p.enabled && p.anchor_first,
            broad_scope,
            multiscale_rank: broad_scope && p.multiscale_rank,
            calibration_reuse: reuse,
            early_family: early,
            calibration_order: reuse && p.calibration_order,
            plan_first: plan,
            plan_buckets: plan && p.plan_buckets,
            geometry_first: plan && p.geometry_first,
            automatic_yield: plan && p.automatic_yield,
            context_feedback: p.enabled && p.context_feedback,
            scope_routing: p.scope != ScopeRouting::Disabled,
            capacity_mismatch: p.scope != ScopeRouting::Disabled
                && scan.is_some_and(|s| {
                    s.palette_color_lower_bound > 10000 + super::super::scan::IMPLICIT_PALETTE_SIZE
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dependent_routes_do_not_enable_their_outer_gates() {
        let mut o = DriverOptions {
            preflight: PreflightOptions {
                anchor_first: true,
                calibration_reuse: true,
                early_family: true,
                calibration_order: true,
                plan_first: true,
                plan_buckets: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(!ActiveRoutes::new(&o, 8 << 20, None).plan_first);
        o.preflight.enabled = true;
        o.calibration = super::super::CalibrationOptions {
            enabled: true,
            fast: true,
            routes: true,
            ..Default::default()
        };
        assert!(!ActiveRoutes::new(&o, 1 << 20, None).early_family);
        o.preflight.scope = ScopeRouting::Broad;
        let active = ActiveRoutes::new(&o, 1 << 20, None);
        assert!(active.early_family && active.plan_first && active.plan_buckets);
        assert!(active.anchor_first && active.geometry_first && active.context_feedback);
    }
    #[test]
    fn worker_scratch_caps_concurrency_but_does_not_reject_one_large_job() {
        let limits = WorkerLimits {
            tasks: 8,
            scratch_bytes: 1024,
        };
        assert_eq!(limits.concurrency(30, 256), 4);
        assert_eq!(limits.concurrency(30, 4096), 1);
        assert_eq!(limits.concurrency(0, 4096), 0);
        assert_eq!(limits.concurrency(30, 0), 8);
        assert!(WorkerLimits { tasks: 0, ..limits }.validate().is_err());
    }
}
