//! Opt-in cooperative full finalists; only completed outputs establish bounds.
use super::*;

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    /// Number of new finalists that can share one committed-incumbent wave.
    pub(super) fn finalist_wave_capacity(&self) -> Result<usize> {
        if !self.options.scout_parallel_finalists || self.full_batch.is_none() {
            return Ok(1);
        }
        self.options.workers.validate()?;
        let scratch = u64::from(self.image.width())
            .saturating_mul(u64::from(self.image.height()))
            .saturating_mul(3 + self.image.metadata().extra_channels.len() as u64)
            .saturating_mul(32)
            .saturating_add(32 << 20);
        let requested = if self.compact_finalists()? { 8 } else { 2 };
        Ok(self
            .options
            .workers
            .concurrency(requested, scratch)
            .clamp(1, requested))
    }

    fn compact_finalists(&self) -> Result<bool> {
        Ok(self.options.scout_small_full
            && (super::compact_pilot::compact_geometry(self.image)?
                || (self.options.scout_family_tournament
                    && super::compact_pilot::family_geometry(self.image))))
    }

    /// One means keep the original serial baseline path. The estimate matches
    /// native full-batch scratch accounting, not a hard resident-memory ceiling.
    pub(super) fn initial_wave_capacity(&self) -> Result<usize> {
        if !self.options.scout_initial_wave
            || !self.options.scout_pilot_first
            || !self.options.scout_parallel_finalists
            || self.full_batch.is_none()
            || !self.search.trials().is_empty()
        {
            return Ok(1);
        }
        let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
        if pixels > 1 << 19 {
            return Ok(1);
        }
        self.options.workers.validate()?;
        let scratch = pixels
            .saturating_mul(3 + self.image.metadata().extra_channels.len() as u64)
            .saturating_mul(32)
            .saturating_add(32 << 20);
        let limits = WorkerLimits {
            scratch_bytes: self.options.workers.scratch_bytes.min(256 << 20),
            ..self.options.workers
        };
        Ok(limits.concurrency(3, scratch))
    }

    /// Submit only a fitting prefix, with the first candidate as the baseline.
    /// None submits nothing: the caller must retain the original serial path.
    pub(super) fn full_initial_wave(
        &mut self,
        stage: u8,
        params: &[CandidateParams],
    ) -> Result<Option<Vec<usize>>> {
        if !(1..=10).contains(&stage) {
            return Err(Error::InvalidInput("search stage must be 1..=10".into()));
        }
        let width = self.initial_wave_capacity()?.min(params.len());
        if width < 2 {
            return Ok(None);
        }
        let params = &params[..width];
        for (index, candidate) in params.iter().enumerate() {
            candidate.validate()?;
            if params[..index].contains(candidate) || self.search.find(candidate).is_some() {
                return Ok(None);
            }
        }
        self.search.check_batch_work(self.image, width)?;
        let budget = TrialBudget {
            beat_bytes: None,
            max_output_bytes: self.options.full_limits.max_output_bytes,
        };
        let tasks: Vec<_> = params
            .iter()
            .map(|params| FullTask {
                params,
                budget,
                cooperative: true,
            })
            .collect();
        let outputs = self.full_batch.as_mut().expect("initial batch selected")(self.image, &tasks)
            .and_then(|outputs| {
                // Even a challenger must complete: no dispatch incumbent exists.
                validate_outputs(&outputs, width, budget)?;
                Ok(outputs)
            });
        let outputs = match outputs {
            Ok(outputs) => outputs,
            Err(error) => {
                // No partial-attempt receipt exists. Charge every submitted root
                // without retaining a partial wave or replacing its error.
                for (index, candidate) in params.iter().enumerate() {
                    let charged = self.search.refine(
                        self.image,
                        stage,
                        candidate.clone(),
                        index == 0,
                        false,
                        &mut |_, _, _| Err(Error::Encoding("aborted initial wave".into())),
                    );
                    debug_assert!(matches!(charged, Err(Error::Encoding(_))));
                }
                return Err(error);
            }
        };
        let mut sizes = Vec::with_capacity(width);
        for (index, (candidate, output)) in params.iter().zip(outputs).enumerate() {
            let mut output = Some(output);
            let size = self.search.refine(
                self.image,
                stage,
                candidate.clone(),
                index == 0,
                false,
                &mut |_, _, _| Ok(output.take().expect("one initial result")),
            )?;
            sizes.push(size.ok_or_else(|| Error::Encoding("incomplete initial wave".into()))?);
        }
        self.events.push(SearchEvent::Route {
            name: "adaptive-scout-initial-wave",
            candidates: width,
        });
        Ok(Some(sizes))
    }

    /// Return one outcome per input, including reused and duplicate candidates.
    /// Each wave uses the completed incumbent at dispatch, never probe scores.
    pub(super) fn full_finalists(
        &mut self,
        stage: u8,
        params: &[CandidateParams],
    ) -> Result<Vec<Option<usize>>> {
        if params.is_empty() {
            return Ok(Vec::new());
        }
        if !(1..=10).contains(&stage) {
            return Err(Error::InvalidInput("search stage must be 1..=10".into()));
        }
        if self.search.best_before(11, true).is_none() {
            return Err(Error::Encoding(
                "full finalists require a completed baseline".into(),
            ));
        }
        if !self.options.scout_parallel_finalists || self.full_batch.is_none() {
            return params
                .iter()
                .map(|params| self.full(stage, params.clone(), false, false))
                .collect();
        }
        self.options.workers.validate()?;
        let mut pending: Vec<&CandidateParams> = Vec::new();
        for candidate in params {
            candidate.validate()?;
            if self.search.find(candidate).is_none() && !pending.contains(&candidate) {
                pending.push(candidate);
            }
        }
        if !pending.is_empty() {
            self.search.check_batch_work(self.image, pending.len())?;
            // Match native full-batch scratch accounting. One oversized root
            // remains allowed by WorkerLimits; this is not a hard RSS ceiling.
            let width = self.finalist_wave_capacity()?.min(pending.len());
            // Tiny full images have enough independent candidates already;
            // nested cooperative kernels cost more than their useful work.
            let cooperative = !self.compact_finalists()?;
            for chunk in pending.chunks(width) {
                let budget = TrialBudget {
                    beat_bytes: Some(self.best_size(11)?),
                    max_output_bytes: self.options.full_limits.max_output_bytes,
                };
                let tasks: Vec<_> = chunk
                    .iter()
                    .map(|&params| FullTask {
                        params,
                        budget,
                        cooperative,
                    })
                    .collect();
                let outputs = self.full_batch.as_mut().expect("batch selected")(self.image, &tasks)
                    .and_then(|outputs| {
                        validate_outputs(&outputs, chunk.len(), budget)?;
                        Ok(outputs)
                    });
                let outputs = match outputs {
                    Ok(outputs) => outputs,
                    Err(error) => {
                        // The legacy callback has no partial-attempt receipt.
                        // Charge every submitted job on failure, but commit no
                        // output from this wave and propagate the original error.
                        for &candidate in chunk {
                            let charged = self.search.refine(
                                self.image,
                                stage,
                                candidate.clone(),
                                false,
                                false,
                                &mut |_, _, _| {
                                    Err(Error::Encoding("aborted finalist batch".into()))
                                },
                            );
                            debug_assert!(matches!(charged, Err(Error::Encoding(_))));
                        }
                        return Err(error);
                    }
                };
                for (&candidate, output) in chunk.iter().zip(outputs) {
                    let mut output = Some(output);
                    self.search.refine(
                        self.image,
                        stage,
                        candidate.clone(),
                        false,
                        false,
                        &mut |_, _, _| Ok(output.take().expect("one finalist result")),
                    )?;
                }
            }
        }
        params
            .iter()
            .map(|params| {
                self.search
                    .find(params)
                    .map(|index| self.search.trials()[index].bytes)
                    .ok_or_else(|| Error::Encoding("missing finalist outcome".into()))
            })
            .collect()
    }
}

fn validate_outputs<A>(
    outputs: &[TrialOutput<A>],
    count: usize,
    budget: TrialBudget,
) -> Result<()> {
    if outputs.len() != count {
        return Err(Error::Encoding(
            "invalid finalist batch result count".into(),
        ));
    }
    for output in outputs {
        match output {
            TrialOutput::Bounded { lower_bound } => {
                if budget.beat_bytes.is_none_or(|bound| *lower_bound < bound) {
                    return Err(Error::Encoding(
                        "finalist byte bound does not reach dispatch incumbent".into(),
                    ));
                }
            }
            TrialOutput::Complete(output) => {
                if output.bytes.is_empty() {
                    return Err(Error::Encoding(
                        "full-frame callback returned an empty output".into(),
                    ));
                }
                if output.bytes.len() > budget.max_output_bytes {
                    return Err(Error::Limit("whole-frame output bytes"));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "finalists_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "initial_wave_tests.rs"]
mod initial_wave_tests;
