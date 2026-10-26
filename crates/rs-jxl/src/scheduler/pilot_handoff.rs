//! Owned results for the optional automatic-full/pilot-front overlap.
//! Screen caches and Rc callbacks are constructed and destroyed on one thread.

use std::{cell::RefCell, rc::Rc};

use super::{PlannedProbeBatchEncoder, ProbeTask};
use crate::scheduler::{
    driver::{DriverOptions, SearchEvent, scout_pilot::rank_front},
    params::{Candidate, CandidateParams},
    probe::{CandidateScreen, ProbeLimits, ScreenOptions},
    scan::LosslessScan,
    search::{TrialBudget, TrialOutput},
};
use crate::{Error, Image, Result};

pub type PilotRank = Vec<(Candidate, Vec<usize>)>;

pub struct PilotRequest<'a> {
    pub image: &'a Image,
    pub automatic: &'a CandidateParams,
    pub plan: &'a [Candidate],
    pub options: &'a DriverOptions,
    pub scan: Option<&'a LosslessScan>,
    pub level: u8,
    pub budget: TrialBudget,
    pub probe_limits: ProbeLimits,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PilotUsage {
    pub encodes: usize,
    pub pixels: u64,
    pub batches: usize,
}

pub struct PilotFront {
    pub ranked: Result<PilotRank>,
    pub events: Vec<SearchEvent>,
    /// Attempts remain available on failure; no partial ranking is returned.
    pub usage: PilotUsage,
}

pub struct PilotHandoff<A> {
    /// Exactly one unbounded automatic attempt, including preparation failures.
    pub automatic: Result<TrialOutput<A>>,
    pub front: PilotFront,
}

pub(crate) fn screen_options(request: &PilotRequest<'_>) -> (ScreenOptions, ScreenOptions) {
    let mut fine = request.options.screen.clone();
    fine.adaptive_coverage |= request.options.preflight.enabled;
    fine.limits = request.probe_limits;
    let mut coarse = fine.clone();
    coarse.regions.context_side = 128;
    if request.options.scout_staged_probes {
        coarse.regions.context_side = 0;
        coarse.regions.full_width_strips = true;
        coarse.regions.strip_rows = Some(8);
    }
    (fine, coarse)
}

/// The callback must drain and attempt every submitted cell, including errors.
/// Construct this function's thread-local Rc graph INSIDE the executor root.
pub(crate) fn run_front(
    request: &PilotRequest<'_>,
    mut encode: PlannedProbeBatchEncoder<'_>,
) -> PilotFront {
    let usage = Rc::new(RefCell::new(PilotUsage::default()));
    let mut events = Vec::new();
    let shared = Rc::clone(&usage);
    let limits = request.probe_limits;
    let callback: PlannedProbeBatchEncoder<'_> = Box::new(move |tasks, kind| {
        let pixels = tasks.iter().try_fold(0u64, |sum, task| {
            sum.checked_add(u64::from(task.image.width()) * u64::from(task.image.height()))
                .ok_or(Error::Limit("cumulative scheduler probe work"))
        })?;
        let mut work = shared.borrow_mut();
        if work
            .encodes
            .checked_add(tasks.len())
            .is_none_or(|n| n > limits.max_encodes)
            || work
                .pixels
                .checked_add(pixels)
                .is_none_or(|n| n > limits.max_encoded_pixels)
        {
            return Err(Error::Limit("cumulative scheduler probe work"));
        }
        work.encodes += tasks.len();
        work.pixels += pixels;
        work.batches += 1;
        drop(work);
        encode(tasks, kind)
    });
    let callback = Rc::new(RefCell::new(callback));
    let ranked = (|| {
        let (fine, coarse) = screen_options(request);
        let no_serial = |_: &CandidateParams, _: &Image| -> Result<usize> {
            Err(Error::Encoding(
                "overlap front requires batch dispatch".into(),
            ))
        };
        let mut screen = CandidateScreen::new(request.image, request.scan, fine, no_serial)?;
        let mut context = CandidateScreen::new(
            request.image,
            if request.options.scout_staged_probes {
                request.scan
            } else {
                None
            },
            coarse,
            no_serial,
        )?;
        let shared = Rc::clone(&callback);
        screen.set_planned_batch_encoder(Box::new(move |tasks: &[ProbeTask<'_>], kind| {
            shared.borrow_mut()(tasks, kind)
        }));
        let shared = Rc::clone(&callback);
        context.set_planned_batch_encoder(Box::new(move |tasks, kind| {
            shared.borrow_mut()(tasks, kind)
        }));
        rank_front(
            request.plan.to_vec(),
            request.level,
            request.options,
            request.scan,
            &mut screen,
            &mut context,
            &mut events,
        )
    })();
    let usage = *usage.borrow();
    PilotFront {
        ranked,
        events,
        usage,
    }
}

#[cfg(test)]
#[path = "pilot_handoff_tests.rs"]
mod tests;
