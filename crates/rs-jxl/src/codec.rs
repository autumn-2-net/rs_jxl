//! Application-facing encoding policy and reusable execution resources.
//!
//! Adapters select a typed job here rather than reproducing effort dispatch.
//! Contexts retain workers only: images, frame references and candidate caches
//! remain invocation-local, including after an error. No implicit search occurs.

use std::time::Instant;

use crate::{EncodeOptions, Error, Image, Result, jpeg, multiframe, scheduler};
use scheduler::{
    CandidateParams, NativeBackendOptions, NativeImplementation,
    native::{NativeAdaptation, NativeArtifact},
    progressive::ProgressiveReport,
};

#[derive(Clone, Copy, Debug, Default)]
pub enum SourceSearch {
    #[default]
    Disabled,
    Auto,
    Scout,
    Pilot(scheduler::pilot::PilotOptions),
}

#[derive(Clone, Debug)]
pub struct SourceRequest {
    pub params: CandidateParams,
    pub search: SourceSearch,
    pub backend: NativeBackendOptions,
}

#[derive(Clone, Debug, Default)]
pub struct EncodeJob {
    pub options: EncodeOptions,
    /// None executes the explicit Modular tools in options without source policy.
    pub source: Option<SourceRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeRoute {
    Manual,
    Candidate,
    Search,
    Scout,
    Pilot,
    Tectonic,
}

#[derive(Debug)]
pub enum SearchDetails {
    None,
    Pixel(Box<scheduler::driver::SearchReport>),
    Tectonic(Box<scheduler::tectonic::TectonicReport>),
}

#[derive(Debug)]
pub struct EncodeReport {
    pub route: EncodeRoute,
    pub pixels: u64,
    pub encoded_bytes: usize,
    /// Codec work only; excludes context construction, transport and image input.
    pub codec_seconds: f64,
    pub full_encodes: usize,
    pub probe_encodes: usize,
    pub winner_index: Option<usize>,
    pub effective_params: Option<CandidateParams>,
    pub implementation: NativeImplementation,
    /// Winning source artifact, including resolved options, gaps and resource limits.
    /// Manual jobs have no source artifact.
    pub artifact: Option<NativeArtifact>,
    /// Every adaptation returned by search, including probes and losing trials.
    /// Direct candidates expose their diagnostics through `artifact` instead.
    pub adaptations: Vec<NativeAdaptation>,
    /// Original-group diagnostics when the pilot completes a real-group race.
    pub progressive: Option<ProgressiveReport>,
    pub preparation_reuse: Option<crate::modular::candidate_cache::ReuseStats>,
    pub details: SearchDetails,
}

impl EncodeReport {
    pub fn bpp(&self) -> f64 {
        self.encoded_bytes as f64 * 8.0 / self.pixels as f64
    }

    pub fn mpix_per_second(&self) -> Option<f64> {
        (self.codec_seconds > 0.0).then(|| self.pixels as f64 / (self.codec_seconds * 1_000_000.0))
    }
}

#[derive(Debug)]
pub struct EncodeResult {
    pub bytes: Vec<u8>,
    pub report: EncodeReport,
}

#[derive(Debug)]
pub struct JpegEncodeResult {
    pub bytes: Vec<u8>,
    pub codec_seconds: f64,
    pub backend_effort: u8,
    pub winner_index: usize,
    pub trials: Vec<jpeg::search::Trial>,
}

/// Caller-owned worker resources. Methods need exclusive access; independent
/// contexts may execute concurrently. One thread never creates a Rayon pool.
pub struct Encoder {
    threads: usize,
    pool: Option<rayon::ThreadPool>,
}

impl Encoder {
    pub fn new(threads: usize) -> Result<Self> {
        if !(1..=256).contains(&threads) {
            return Err(Error::InvalidInput("threads must be 1..=256".into()));
        }
        let pool = (threads > 1)
            .then(|| rayon::ThreadPoolBuilder::new().num_threads(threads).build())
            .transpose()
            .map_err(|error| Error::Encoding(error.to_string()))?;
        Ok(Self { threads, pool })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    fn check_threads(&self, threads: usize) -> Result<()> {
        if threads != self.threads {
            return Err(Error::InvalidInput(
                "job threads must match the encoder context".into(),
            ));
        }
        Ok(())
    }

    pub fn encode(&mut self, image: &Image, job: &EncodeJob) -> Result<EncodeResult> {
        self.check_threads(job.options.threads)?;
        let start = Instant::now();
        let mut report = EncodeReport {
            route: EncodeRoute::Manual,
            pixels: u64::from(image.width()) * u64::from(image.height()),
            encoded_bytes: 0,
            codec_seconds: 0.0,
            full_encodes: 1,
            probe_encodes: 0,
            winner_index: None,
            effective_params: None,
            implementation: NativeImplementation::Modular,
            artifact: None,
            adaptations: Vec::new(),
            progressive: None,
            preparation_reuse: None,
            details: SearchDetails::None,
        };
        let pool = self.pool.as_ref();
        let bytes = if let Some(source) = &job.source {
            source.params.validate()?;
            if matches!(source.search, SourceSearch::Pilot(_))
                && source.backend != NativeBackendOptions::default()
            {
                return Err(Error::InvalidInput(
                    "pilot backend controls belong in PilotOptions".into(),
                ));
            }
            match source.search {
                SourceSearch::Disabled => {
                    report.route = EncodeRoute::Candidate;
                    let output = scheduler::native::encode_native_candidate_in_pool(
                        &source.params,
                        image,
                        &job.options,
                        source.backend,
                        pool,
                    )?;
                    report.effective_params = Some(output.artifact.effective_params.clone());
                    report.implementation = output.artifact.implementation;
                    report.artifact = Some(output.artifact);
                    output.bytes
                }
                SourceSearch::Auto if source.params.speed_tier == -1 => {
                    report.route = EncodeRoute::Tectonic;
                    let result = scheduler::run_native_tectonic_in_pool(
                        image,
                        &source.params,
                        Default::default(),
                        Default::default(),
                        &job.options,
                        source.backend,
                        pool,
                    )?;
                    report.full_encodes = result.output.report.full_encodes;
                    report.winner_index = Some(result.output.report.winner_index);
                    report.effective_params = Some(result.output.artifact.effective_params.clone());
                    report.implementation = result.output.artifact.implementation;
                    report.artifact = Some(result.output.artifact);
                    report.adaptations = result.adaptations;
                    report.details = SearchDetails::Tectonic(Box::new(result.output.report));
                    result.output.bytes
                }
                search => {
                    let result = match search {
                        SourceSearch::Pilot(options) => {
                            report.route = EncodeRoute::Pilot;
                            scheduler::pilot::encode_in_pool(
                                image,
                                &source.params,
                                &job.options,
                                options,
                                pool,
                            )?
                        }
                        SourceSearch::Auto | SourceSearch::Scout => {
                            let driver = if matches!(search, SourceSearch::Scout) {
                                if !matches!(source.params.speed_tier, 0 | 1) {
                                    return Err(Error::InvalidInput(
                                        "source scout requires effort 9 or 10".into(),
                                    ));
                                }
                                report.route = EncodeRoute::Scout;
                                scheduler::DriverOptions::scout()
                            } else {
                                report.route = EncodeRoute::Search;
                                scheduler::DriverOptions::default()
                            };
                            scheduler::native::run_native_pixel_search_in_pool(
                                image,
                                &source.params,
                                &driver,
                                &job.options,
                                source.backend,
                                pool,
                            )?
                        }
                        SourceSearch::Disabled => unreachable!(),
                    };
                    report.full_encodes = result.output.report.full_encodes;
                    report.probe_encodes = result.output.report.probe_encodes;
                    report.winner_index = Some(result.output.report.winner_index);
                    report.effective_params = Some(result.output.artifact.effective_params.clone());
                    report.implementation = result.output.artifact.implementation;
                    report.artifact = Some(result.output.artifact);
                    report.adaptations = result.adaptations;
                    report.progressive = result.progressive;
                    report.preparation_reuse = result.preparation_reuse;
                    report.details = SearchDetails::Pixel(Box::new(result.output.report));
                    result.output.bytes
                }
            }
        } else {
            crate::encode::encode_frames_in_pool(
                image,
                &[crate::frame::Frame::new(image)],
                &job.options,
                pool,
            )?
        };
        report.codec_seconds = start.elapsed().as_secs_f64();
        report.encoded_bytes = bytes.len();
        Ok(EncodeResult { bytes, report })
    }

    pub fn encode_frames(
        &mut self,
        canvas: &Image,
        frames: &[crate::frame::Frame<'_>],
        options: &EncodeOptions,
    ) -> Result<Vec<u8>> {
        self.check_threads(options.threads)?;
        crate::encode::encode_frames_in_pool(canvas, frames, options, self.pool.as_ref())
    }

    pub fn encode_auto(
        &mut self,
        images: &[Image],
        durations: &[u32],
        options: &EncodeOptions,
        automatic: &multiframe::AutoOptions,
    ) -> Result<multiframe::AutoEncodeResult> {
        self.check_threads(options.threads)?;
        multiframe::encode_auto_in_pool(images, durations, options, automatic, self.pool.as_ref())
    }

    pub fn transcode_jpeg(
        &mut self,
        input: &[u8],
        options: &jpeg::TranscodeOptions,
        search: Option<jpeg::search::SearchLimits>,
    ) -> Result<JpegEncodeResult> {
        self.check_threads(options.threads)?;
        let start = Instant::now();
        if let Some(limits) = search {
            let result =
                jpeg::transcode_search_in_pool(input, options, limits, self.pool.as_ref())?;
            Ok(JpegEncodeResult {
                bytes: result.data,
                codec_seconds: start.elapsed().as_secs_f64(),
                backend_effort: result.backend_effort,
                winner_index: result.winner_index,
                trials: result.trials,
            })
        } else {
            let bytes = jpeg::transcode_in_pool(input, options, self.pool.as_ref())?;
            let backend_effort = options.effort.clamp(2, 10);
            let trials = vec![jpeg::search::Trial {
                stage: options.effort,
                backend_effort,
                bytes: bytes.len(),
            }];
            Ok(JpegEncodeResult {
                bytes,
                codec_seconds: start.elapsed().as_secs_f64(),
                backend_effort,
                winner_index: 0,
                trials,
            })
        }
    }
}

/// One-shot equivalent of Encoder::encode; context construction is explicit.
pub fn encode_job(image: &Image, job: &EncodeJob) -> Result<EncodeResult> {
    Encoder::new(job.options.threads)?.encode(image, job)
}

pub fn transcode_jpeg(
    input: &[u8],
    options: &jpeg::TranscodeOptions,
    search: Option<jpeg::search::SearchLimits>,
) -> Result<JpegEncodeResult> {
    Encoder::new(options.threads)?.transcode_jpeg(input, options, search)
}
