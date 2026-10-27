//! Native bridge for the separate source effort-11 frame tournament.
use super::{
    CandidateParams, NativeBackendOptions, WorkerLimits,
    native::{NativeAdaptation, NativeArtifact, NativeTrialKind, prepare},
    search::{EncodedOutput, SearchLimits, TrialBudget},
    tectonic::{self, TectonicOracle, TectonicOutput},
    worker_pool::{WorkerPlan, run_plan},
};
use crate::{EncodeOptions, Error, Image, Result};

#[derive(Debug)]
pub struct NativeTectonicResult {
    pub output: TectonicOutput<NativeArtifact>,
    pub adaptations: Vec<NativeAdaptation>,
}

struct Oracle<'a> {
    pool: Option<&'a rayon::ThreadPool>,
    base: &'a EncodeOptions,
    settings: NativeBackendOptions,
    workers: WorkerLimits,
    adaptations: Vec<NativeAdaptation>,
}

impl TectonicOracle<NativeArtifact> for Oracle<'_> {
    fn encode(
        &mut self,
        image: &Image,
        params: &CandidateParams,
        budget: TrialBudget,
    ) -> Result<EncodedOutput<NativeArtifact>> {
        let (mut artifact, record) = prepare(
            params,
            image,
            self.base,
            self.settings,
            NativeTrialKind::Full,
        )?;
        artifact.options.max_output_bytes = artifact
            .options
            .max_output_bytes
            .min(budget.max_output_bytes);
        self.adaptations.push(record);
        let bytes = crate::encode::encode_frames_in_pool(
            image,
            &[crate::frame::Frame::new(image)],
            &artifact.options,
            self.pool,
        )?;
        Ok(EncodedOutput { bytes, artifact })
    }

    fn measure_round(
        &mut self,
        image: &Image,
        candidates: &[CandidateParams],
        budget: TrialBudget,
    ) -> Result<Vec<usize>> {
        let mut artifacts = Vec::with_capacity(candidates.len());
        for params in candidates {
            let (mut artifact, record) = prepare(
                params,
                image,
                self.base,
                self.settings,
                NativeTrialKind::Full,
            )?;
            artifact.options.max_output_bytes = artifact
                .options
                .max_output_bytes
                .min(budget.max_output_bytes);
            self.adaptations.push(record);
            artifacts.push(artifact);
        }
        let scratch = u64::from(image.width())
            .saturating_mul(u64::from(image.height()))
            .saturating_mul(3 + image.metadata().extra_channels.len() as u64)
            .saturating_mul(32)
            .saturating_add(32 << 20);
        let slots = self.workers.concurrency(candidates.len(), scratch).max(1);
        let plan = WorkerPlan::independent(candidates.len(), slots)?;
        run_plan(self.pool, &plan, |i, _| {
            Ok(crate::encode::encode_serial(image, &artifacts[i].options)?.len())
        })
    }
}

/// Full source tournament, not the ordinary effort 1..10 pixel-search driver.
/// Each round uses null-runner inner jobs on one bounded caller-owned pool.
/// The winner is re-encoded as in C++, including its actual final artifact.
/// Whole codestream overhead is common to candidates for this one Image.
pub fn run_native_tectonic(
    image: &Image,
    params: &CandidateParams,
    limits: SearchLimits,
    workers: WorkerLimits,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
) -> Result<NativeTectonicResult> {
    validate(params, workers, base)?;
    if base.threads == 1 || crate::execution::is_serial() {
        return run_native_tectonic_in_pool(image, params, limits, workers, base, settings, None);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(base.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    run_native_tectonic_in_pool(image, params, limits, workers, base, settings, Some(&pool))
}

fn validate(params: &CandidateParams, workers: WorkerLimits, base: &EncodeOptions) -> Result<()> {
    if !(1..=256).contains(&base.threads) {
        return Err(Error::InvalidInput("threads must be 1..=256".into()));
    }
    workers.validate()?;
    params.validate()?;
    if params.speed_tier != -1 {
        return Err(Error::InvalidInput(
            "Tectonic tournament requires effort 11".into(),
        ));
    }
    Ok(())
}

pub(crate) fn run_native_tectonic_in_pool(
    image: &Image,
    params: &CandidateParams,
    limits: SearchLimits,
    workers: WorkerLimits,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<NativeTectonicResult> {
    validate(params, workers, base)?;
    let mut oracle = Oracle {
        pool,
        base,
        settings,
        workers,
        adaptations: Vec::new(),
    };
    let mut work = || tectonic::encode(image, params, limits, &mut oracle);
    let output = if pool.is_none() {
        crate::execution::serial(work)?
    } else {
        work()?
    };
    Ok(NativeTectonicResult {
        output,
        adaptations: oracle.adaptations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    #[test]
    fn native_tournament_runs_all_source_trials_without_recursive_effort_dispatch() {
        let image = Image::new(
            8,
            8,
            PixelLayout::Gray,
            PixelData::U8((0..64).map(|i| (i % 3) as u8).collect()),
        )
        .unwrap();
        let result = run_native_tectonic(
            &image,
            &CandidateParams::default().backend(11).unwrap(),
            SearchLimits::default(),
            WorkerLimits {
                tasks: 2,
                ..Default::default()
            },
            &EncodeOptions {
                threads: 2,
                ..Default::default()
            },
            NativeBackendOptions::default(),
        )
        .unwrap();
        assert!([23, 27].contains(&result.output.report.full_encodes));
        assert_eq!(result.adaptations.len(), result.output.report.full_encodes);
        assert!(
            result
                .adaptations
                .iter()
                .all(|a| a.effective.speed_tier == 0 && a.gaps.is_empty())
        );
        assert_eq!(
            result.output.params,
            result.output.artifact.effective_params
        );
        assert!(!result.output.bytes.is_empty());
    }
}
