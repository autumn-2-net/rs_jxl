//! Invocation-owned palette preparation telemetry; never a production policy.

use super::{PaletteMode, PaletteStage};
use crate::modular::predictor::Predictor;
use std::sync::Mutex;

/// Opt-in preparation evidence. Install a fresh Arc AFTER native projection for
/// each complete image encode. Cloned worker options share this invocation only.
/// Read after encode returns; event order across workers is not meaningful.
/// Accepted means policy-accepted preparation, not a byte-size win. A failed
/// preparation can roll back channels after recording earlier accepted attempts.
#[derive(Debug, Default)]
pub struct PaletteReceipt {
    events: Mutex<Vec<PaletteAttempt>>,
}

// Diagnostics must not make unrelated invocations share cached preparation.
impl PartialEq for PaletteReceipt {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl PaletteReceipt {
    pub fn snapshot(&self) -> Vec<PaletteAttempt> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub(super) fn record(&self, event: PaletteAttempt) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaletteOutcome {
    KernelDeclined,
    CostRejected,
    Accepted,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PaletteKernelEvidence {
    /// The kernel was called; it may immediately decline its eligibility gate.
    pub exact_delta_attempted: bool,
    pub exact_delta_produced: bool,
    pub exact_zero_attempted: bool,
    pub zero_mask_eligible: bool,
    pub zero_mask_matches: usize,
    pub absolute_attempted: bool,
    pub absolute_produced: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PaletteAttempt {
    /// PreRct = global; PostRct = local, including full-frame route AC groups.
    pub stage: PaletteStage,
    pub mode: PaletteMode,
    pub width: usize,
    pub height: usize,
    /// Physical channel offset at this attempt, NOT original RGB coverage.
    pub begin: usize,
    pub channels: usize,
    pub limit: u32,
    pub kernel: PaletteKernelEvidence,
    /// Candidate table counts, also retained for a cost-rejected candidate.
    pub colors: usize,
    pub deltas: usize,
    pub predictor: Predictor,
    /// Number of candidate indices referencing an explicit delta column.
    /// ExactZero may insert a column even when this is zero.
    pub delta_pixels: usize,
    pub cost_before: f32,
    pub cost_after: Option<f32>,
    pub cost_bypassed: bool,
    pub outcome: PaletteOutcome,
}

#[cfg(test)]
mod tests {
    use super::super::{PaletteOptions, PaletteState, PaletteTools, apply_palettes, attempt};
    use super::*;
    use crate::modular::group::Channel;
    use std::sync::Arc;

    fn channel(pattern: &[i32], pixels: usize, width: usize) -> Channel {
        Channel {
            width,
            height: pixels / width,
            hshift: 0,
            vshift: 0,
            samples: pattern.iter().copied().cycle().take(pixels).collect(),
        }
    }

    #[test]
    fn invocation_receipt_distinguishes_production_fallback_and_local_work() {
        let receipt = Arc::new(PaletteReceipt::default());
        let original = vec![channel(&[10, 20, 30], 96, 96); 3];
        for (stage, mode, count) in [
            (
                PaletteStage::PreRct,
                PaletteMode::ExactDelta(Predictor::Left),
                3,
            ),
            (
                PaletteStage::PostRct,
                PaletteMode::ExactZero(Predictor::Left),
                3,
            ),
            (
                PaletteStage::PostRct,
                PaletteMode::ExactDelta(Predictor::Left),
                1,
            ),
        ] {
            let mut channels = original[..count].to_vec();
            let mut state = PaletteState::new(8).unwrap();
            let options = PaletteOptions {
                mode,
                direct: true,
                palette_colors: 8,
                receipt: Some(Arc::clone(&receipt)),
                ..Default::default()
            };
            let script = apply_palettes(&mut channels, &mut state, 8, stage, &options).unwrap();
            assert_eq!(script.len(), 1);
        }
        let events = receipt.snapshot();
        assert_eq!(events.len(), 3);
        assert!(events[0].kernel.exact_delta_attempted);
        assert!(events[0].kernel.exact_delta_produced);
        assert!(!events[0].kernel.absolute_attempted);
        assert_eq!(events[0].delta_pixels, 96);
        assert_eq!(events[0].stage, PaletteStage::PreRct);
        assert_eq!(events[1].stage, PaletteStage::PostRct);
        assert!(events[1].kernel.zero_mask_eligible);
        assert_eq!(events[1].kernel.zero_mask_matches, 0);
        assert_eq!((events[1].deltas, events[1].delta_pixels), (1, 0));
        assert!(events[2].kernel.exact_delta_attempted);
        assert!(!events[2].kernel.exact_delta_produced);
        assert!(events[2].kernel.absolute_produced);
        assert_eq!(events[2].deltas, 0);
        assert!(
            events
                .iter()
                .all(|e| e.outcome == PaletteOutcome::Accepted && e.cost_bypassed)
        );
        let untouched = PaletteReceipt::default();
        assert!(untouched.snapshot().is_empty());
        assert_ne!(receipt.as_ref(), &untouched);
        assert_eq!(receipt, Arc::clone(&receipt));
    }

    #[test]
    fn invocation_receipt_keeps_declines_and_cost_rejections_without_mutation() {
        let receipt = PaletteReceipt::default();
        let original = vec![channel(&[0, 1], 32, 8)];
        for (limit, bypass, outcome) in [
            (1, true, PaletteOutcome::KernelDeclined),
            (2, false, PaletteOutcome::CostRejected),
        ] {
            let mut channels = original.clone();
            let mut state = PaletteState::new(8).unwrap();
            assert!(
                !attempt(
                    &mut channels,
                    &mut state,
                    &mut Vec::new(),
                    8,
                    0,
                    1,
                    limit,
                    true,
                    PaletteMode::ExactDelta(Predictor::Left),
                    PaletteTools::default(),
                    0.0,
                    bypass,
                    Some((&receipt, PaletteStage::PostRct)),
                    None,
                )
                .unwrap()
            );
            assert_eq!(channels, original);
            assert_eq!(state, PaletteState::new(8).unwrap());
            assert_eq!(receipt.snapshot().last().unwrap().outcome, outcome);
        }
        let events = receipt.snapshot();
        assert!(!events[0].kernel.absolute_produced);
        assert!(events[1].kernel.absolute_produced);
        assert!(events[1].cost_after.unwrap() > events[1].cost_before);
        assert_eq!(events[1].deltas, 0);
    }

    #[test]
    fn invocation_receipt_worker_clones_share_only_their_sink() {
        let receipt = Arc::new(PaletteReceipt::default());
        let options = PaletteOptions {
            direct: true,
            palette_colors: 8,
            mode: PaletteMode::ExactDelta(Predictor::Top),
            receipt: Some(Arc::clone(&receipt)),
            ..Default::default()
        };
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let options = options.clone();
                scope.spawn(move || {
                    let mut channels = vec![channel(&[10, 20], 32, 8); 3];
                    let mut state = PaletteState::new(8).unwrap();
                    apply_palettes(
                        &mut channels,
                        &mut state,
                        8,
                        PaletteStage::PostRct,
                        &options,
                    )
                    .unwrap();
                });
            }
        });
        let events = receipt.snapshot();
        assert_eq!(events.len(), 4);
        assert!(
            events
                .iter()
                .all(|event| event.stage == PaletteStage::PostRct
                    && event.kernel.exact_delta_produced
                    && event.outcome == PaletteOutcome::Accepted)
        );
        assert!(PaletteReceipt::default().snapshot().is_empty());
    }
}
