//! Opt-in small-frame tournament. Only completed full outputs can win.
use super::*;
use crate::modular::predictor::Predictor;
use crate::scheduler::{regions::small_probe_regions, scan::literal_evidence};

pub(super) fn compact_geometry(image: &Image) -> Result<bool> {
    if !refinable(image) {
        return Ok(false);
    }
    // Compare against the existing first screening round, without copying or
    // encoding its crops. Distinct crop count matters for narrow/tiny frames.
    let screening_pixels = small_probe_regions(image)
        .iter()
        .take(3)
        .try_fold(0u64, |sum, spec| -> Result<u64> {
            Ok(sum + spec.pixels()? as u64)
        })?;
    Ok(u64::from(image.width()) * u64::from(image.height()) <= screening_pixels)
}

fn compact_plan(
    image: &Image,
    input: &CandidateParams,
    options: &DriverOptions,
) -> Result<Vec<CandidateParams>> {
    let automatic = input.backend(9)?;
    let neighborhood = winner_neighborhood(&automatic, image.channel_count());
    let learned = literal(input, 9, 0.01)?;
    // Content evidence orders work, never removes a competing representation.
    let mut plan = if literal_evidence(image) {
        vec![learned, automatic]
    } else {
        vec![automatic, learned]
    };
    plan.push(neighborhood);
    plan.push(literal(input, 9, 0.0)?);
    // Explicit extended transforms all get full-frame trials here; the larger
    // pilot's probe-rank gate does not apply to this compact tournament.
    for candidate in lossless_probe_plan(input, options.plan)
        .into_iter()
        .filter(|candidate| candidate.params.predictor == Some(Predictor::Variable))
    {
        let params = candidate.params;
        if options.scout_native_rct_aliases && image.channel_count() < 3 && params.rct == Some(6) {
            let mut raw = params.clone();
            raw.rct = Some(0);
            if plan.contains(&raw) {
                continue;
            }
        }
        if !plan.contains(&params) {
            plan.push(params);
        }
    }
    Ok(plan)
}

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    pub(super) fn run_compact_pilot(&mut self) -> Result<bool> {
        if !self.options.scout_small_full
            || !self.options.scout_pilot_first
            || !compact_geometry(self.image)?
        {
            return Ok(false);
        }
        let _scope = crate::profile::scope("scheduler.compact_pilot");
        let plan = compact_plan(self.image, self.input, self.options)?;
        let mut candidates = plan.len();
        // Tiny full encodes need no probe/front overlap or calibration warmup.
        // The first output completes; all later bounds come from real bytes.
        self.full(9, plan[0].clone(), true, false)?;
        self.full_finalists(9, &plan[1..])?;
        if self.level == 10 {
            let winner = self.best_params(10)?;
            for params in [winner.backend(10)?, self.input.backend(10)?] {
                if self.search.find(&params).is_none() {
                    self.full(10, params, false, false)?;
                    candidates += 1;
                }
            }
        }
        self.events.push(SearchEvent::Route {
            name: "adaptive-scout-small-full",
            candidates,
        });
        Ok(true)
    }
}

#[cfg(test)]
#[path = "compact_pilot_tests.rs"]
mod tests;
