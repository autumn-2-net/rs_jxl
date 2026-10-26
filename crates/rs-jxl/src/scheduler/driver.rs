// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Executable pixel path from `enc_lossless_search.cc::EncodeLosslessSearch`.
//! Full-frame and crop callbacks must encode the supplied source parameters or
//! reject them. No native-option projection is performed inside this driver.

use super::{
    batch::{
        BatchEncoders, FullBatchEncoder, FullTask, PilotOverlapEncoder, PlannedProbeBatchEncoder,
        PrefixBatchEncoder, ProbeTask, initial_concurrency,
    },
    calibration::{CalibrationFeedback, diverse_frontier},
    params::{Candidate, CandidateParams, PlanOptions, lossless_probe_plan},
    policy::{PixelSearch, literal, winner_neighborhood},
    probe::{CandidateScreen, ProbeResult, ScreenOptions, total},
    scan::{LosslessScan, ScanOptions, scan_lossless_input, scan_lossless_input_with_pool},
    search::{Search, SearchLimits, TrialBudget, TrialOutput, TrialRecord},
    small_probe::{
        confirm_lossless_candidate, confirm_lossless_candidate_batched, probe_lossless_candidates,
        probe_lossless_candidates_batched, probe_lossless_candidates_with_prefix,
    },
};
use crate::{Error, Image, Result};
use std::{cell::RefCell, rc::Rc};

#[path = "compact_pilot.rs"]
mod compact_pilot;
#[path = "finalists.rs"]
mod finalists;
#[path = "frontiers.rs"]
mod frontiers;
#[path = "pilot_overlap.rs"]
mod pilot_overlap;
#[path = "pilot_witness.rs"]
pub(crate) mod pilot_witness;
#[path = "preflight_stages.rs"]
mod preflight_stages;
#[path = "refinement.rs"]
mod refinement;
#[path = "routes.rs"]
mod routes;
#[path = "scout.rs"]
mod scout;
#[path = "scout_pilot.rs"]
pub(crate) mod scout_pilot;
#[path = "stages.rs"]
mod stages;
use super::worker_pool::{ProbeBatchKind, WorkerExecution};
use routes::ActiveRoutes;
pub use routes::{DirectFrontier, FrontierOptions, PreflightOptions, ScopeRouting, WorkerLimits};

/// Caller-supplied backend declaration, NOT a compatibility certification by
/// the scheduler. The same declaration applies to full and sample callbacks.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum BackendSemantics {
    #[default]
    Unverified,
    SourceContract,
    DeclaredProxy {
        differences: Vec<String>,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CalibrationOptions {
    /// JXL_LOSSLESS_CALIBRATE=1.
    pub enabled: bool,
    /// JXL_LOSSLESS_CALIBRATE_FAST=1: e4 instead of e6 full witness.
    pub fast: bool,
    /// JXL_LOSSLESS_CALIBRATE_REFINEMENTS=1.
    pub refinements: bool,
    /// JXL_LOSSLESS_CALIBRATE_ROUTES=1, without PREFLIGHT_YIELD extensions.
    pub routes: bool,
}

#[derive(Clone, Debug, Default)]
pub struct DriverOptions {
    /// Experimental region-agreement frontier. Full candidates retain their
    /// backend strength; probe estimates are never encoded-byte bounds.
    pub adaptive_scout: bool,
    /// Experimental full-strength family probes before any full-frame encode.
    /// This is a search heuristic, not a compressed-size bound.
    pub scout_pilot_first: bool,
    /// Opt-in probe-free tournament when a full frame fits within three legacy
    /// screening crops. Requires pilot-first; candidates retain backend effort.
    pub scout_small_full: bool,
    /// Experimental family-level confirmation and probe-free small-frame plan.
    /// Keeps e9 tools; avoids the fallback's learning-slope admission gates.
    pub scout_family_tournament: bool,
    /// Native-only RCT0/6 aliases for Variable candidates with <3 total channels.
    /// Generic callbacks must leave this off unless they guarantee equivalence.
    pub scout_native_rct_aliases: bool,
    /// Opt-in cooperative finalist batches after a completed pilot baseline.
    pub scout_parallel_finalists: bool,
    /// Opt-in small-input initial wave of at most three complete encodes.
    /// Requires pilot-first and parallel finalists; never uses probe byte bounds.
    pub scout_initial_wave: bool,
    /// Experimental automatic full encode overlapping the complete pilot front.
    /// Requires pilot-first and an overlap callback; default remains serial.
    pub scout_overlap_automatic: bool,
    /// Opt-in consensus after explicit model-cost-adjusted pilot ranking.
    pub scout_pilot_consensus: bool,
    /// Independent opt-in near-tie consensus for e9 automatic pilot leaders.
    /// Allows regional wins of at most 1%; never supplies an encoded-byte bound.
    pub scout_near_tie_consensus: bool,
    /// Confirm representations whose global palette/patch capacity is absent
    /// from crop scores. Opt-in for the native content-pilot preset.
    pub scout_representation_witness: bool,
    /// Opt-in small regional first round for continuous-tone inputs only.
    pub scout_staged_probes: bool,
    /// Experimental: three coarse regions may unanimously select a clear
    /// natural-image leader without the second probe scale. Never an output bound.
    pub scout_coarse_consensus: bool,
    /// Source scan is opt-in, and only runs at level>=2 and >=2^19 pixels.
    pub coverage_scan: bool,
    pub scan: ScanOptions,
    pub plan: PlanOptions,
    pub screen: ScreenOptions,
    pub full_limits: SearchLimits,
    pub backend_semantics: BackendSemantics,
    pub fast_preflight: bool,
    pub family_beam: bool,
    pub confirmation: bool,
    pub probe_feedback_bounds: bool,
    pub calibration: CalibrationOptions,
    /// JXL_LOSSLESS_FAST_FAMILY=1.
    pub fast_family: bool,
    /// JXL_LOSSLESS_PALETTE_RACE=1 or 2. Implies fast_family.
    pub palette_race: bool,
    /// JXL_LOSSLESS_PALETTE_RACE=2. Implies palette_race.
    pub palette_preselect: bool,
    /// JXL_LOSSLESS_SIBLING_RACE=1: whole-frame e4 witnesses for raw-variable
    /// siblings differing only in RCT/group geometry, with unconditional e10 replay.
    pub sibling_race: bool,
    pub sibling_transfer: bool,
    pub statistical_feedback: bool,
    pub preflight: PreflightOptions,
    pub frontier: FrontierOptions,
    pub workers: WorkerLimits,
    pub cooperative_probes: bool,
    /// Explicit JXL_LOSSLESS_ARENA_QUIET for cooperative probes/finalists.
    pub arena_quiet: bool,
    pub data_parallel_probes: bool,
    pub longest_first_probes: bool,
    pub probe_workers: Option<usize>,
}

impl DriverOptions {
    /// Bounded full-frame search with coverage-selected, full-fidelity witnesses.
    /// Source effort 9/10 only; small/unsupported frame geometry keeps the source route.
    pub fn scout() -> Self {
        Self {
            adaptive_scout: true,
            coverage_scan: true,
            scan: ScanOptions {
                palette_scope: true,
            },
            screen: ScreenOptions {
                cheap_control: 1,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub enum SearchEvent {
    Route {
        name: &'static str,
        candidates: usize,
    },
    SiblingRace {
        stage: u8,
        params: CandidateParams,
        control_bytes: usize,
        proposal_bytes: usize,
        deferred: bool,
    },
    EarlyRacing {
        stage: u8,
        candidates: usize,
    },
    FastLiteral {
        enabled: bool,
    },
    Frontier {
        candidates: Vec<Candidate>,
    },
    Reused {
        stage: u8,
        trial: usize,
    },
    Screened {
        stage: u8,
        params: CandidateParams,
        candidate_bytes: usize,
        incumbent_bytes: usize,
        related: Option<usize>,
        predicted_bytes: Option<f64>,
        accepted: bool,
        exploration: bool,
        neighborhood: bool,
    },
    Calibration {
        stage: u8,
        params: CandidateParams,
        kind: &'static str,
        predicted_bytes: Option<f64>,
        accepted: bool,
    },
    Deferred {
        stage: u8,
        params: CandidateParams,
        screened_on_replay: bool,
    },
    Confirmation {
        stage: u8,
        samples: usize,
        candidate_bytes: usize,
        incumbent_bytes: usize,
        accepted: bool,
    },
}

#[derive(Clone, Debug)]
pub struct SearchReport {
    pub source_commit: &'static str,
    pub backend_semantics: BackendSemantics,
    pub trials: Vec<TrialRecord>,
    pub winner_index: usize,
    pub full_encodes: usize,
    pub full_pixels: u64,
    pub probe_encodes: usize,
    pub probe_pixels: u64,
    pub screen_batches: usize,
    pub events: Vec<SearchEvent>,
    /// Source execution/diagnostic details not reproduced by this workflow.
    pub unported: &'static [&'static str],
}

#[derive(Debug)]
pub struct SearchResult<A = ()> {
    pub bytes: Vec<u8>,
    pub params: CandidateParams,
    pub artifact: A,
    pub report: SearchReport,
}

const UNPORTED: &[&str] = &["automatic environment parsing and diagnostic final-plan files"];
const LEGACY_UNPORTED: &[&str] = &[
    "within-wave asynchronous legacy-probe cutoff updates (completed-wave updates instead)",
    "automatic environment parsing and diagnostic final-plan files",
];

type ProbeCallback<'a> = Box<dyn FnMut(&CandidateParams, &Image) -> Result<usize> + 'a>;

struct ProbeWork<P> {
    callback: P,
    limits: super::probe::ProbeLimits,
    encodes: usize,
    pixels: u64,
}
impl<P: FnMut(&CandidateParams, &Image) -> Result<usize>> ProbeWork<P> {
    fn run(&mut self, params: &CandidateParams, sample: &Image) -> Result<usize> {
        let pixels = u64::from(sample.width()) * u64::from(sample.height());
        if self.encodes >= self.limits.max_encodes
            || self
                .pixels
                .checked_add(pixels)
                .is_none_or(|n| n > self.limits.max_encoded_pixels)
        {
            return Err(Error::Limit("cumulative scheduler probe work"));
        }
        self.encodes += 1;
        self.pixels += pixels;
        let bytes = (self.callback)(params, sample)?;
        if bytes == 0 {
            return Err(Error::Encoding(
                "sample callback returned an empty frame size".into(),
            ));
        }
        Ok(bytes)
    }
}

#[derive(Clone)]
struct Deferred {
    params: CandidateParams,
    screen: bool,
}

fn refinable(image: &Image) -> bool {
    u64::from(image.width()) * u64::from(image.height()) > 4096
        && image.metadata().extra_channels.len() <= 8
        && image
            .metadata()
            .extra_channels
            .iter()
            .all(|ec| ec.dim_shift == 0)
}

struct Driver<'a, F, P, A> {
    image: &'a Image,
    input: &'a CandidateParams,
    options: &'a DriverOptions,
    scan: Option<&'a LosslessScan>,
    level: u8,
    search: Search<A>,
    encode: F,
    full_batch: Option<FullBatchEncoder<'a, A>>,
    pilot_overlap: Option<PilotOverlapEncoder<'a, A>>,
    probe: Rc<RefCell<ProbeWork<P>>>,
    probe_batch: Option<Rc<RefCell<PlannedProbeBatchEncoder<'a>>>>,
    prefix_batch: Option<PrefixBatchEncoder<'a>>,
    screen: CandidateScreen<'a, ProbeCallback<'a>>,
    context_screen: CandidateScreen<'a, ProbeCallback<'a>>,
    auxiliary_screen_batches: usize,
    routes: ActiveRoutes,
    calibration_feedback: Vec<CalibrationFeedback>,
    early_racing: bool,
    early_family_winner: bool,
    pending_automatic: bool,
    analysis: ProbeResult,
    frontier: Vec<Candidate>,
    deferred: Vec<Deferred>,
    confirmation_cache: Option<(usize, usize)>,
    events: Vec<SearchEvent>,
}

/// Runs one source-backed finite search and returns the exact retained winning
/// output/artifact, never a re-encode. Callback errors and resource exhaustion
/// propagate; no partial output is committed. Baseline/calibration callbacks
/// must complete; challenger callbacks may use only TrialBudget's real bound.
pub fn run_pixel_search<F, P, A>(
    image: &Image,
    params: &CandidateParams,
    options: &DriverOptions,
    encode_full: F,
    encode_probe: P,
) -> Result<SearchResult<A>>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    run_pixel_search_with_batches(
        image,
        params,
        options,
        encode_full,
        encode_probe,
        BatchEncoders::default(),
    )
}

/// Source default batch dispatch, with serial callbacks kept for individual
/// refinements and legacy callers. No sampling/selection option is changed.
pub fn run_pixel_search_with_batches<F, P, A>(
    image: &Image,
    params: &CandidateParams,
    options: &DriverOptions,
    encode_full: F,
    encode_probe: P,
    mut batches: BatchEncoders<'_, A>,
) -> Result<SearchResult<A>>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    params.validate()?;
    if options.scout_initial_wave
        && !(options.scout_pilot_first && options.scout_parallel_finalists)
    {
        return Err(Error::InvalidInput(
            "initial wave requires pilot-first and parallel finalists".into(),
        ));
    }
    if options.scout_coarse_consensus
        && !(options.scout_pilot_consensus && options.scout_staged_probes)
    {
        return Err(Error::InvalidInput(
            "coarse consensus requires staged probes and model consensus".into(),
        ));
    }
    if (options.scout_pilot_consensus
        || options.scout_small_full
        || options.scout_family_tournament
        || options.scout_near_tie_consensus
        || options.scout_representation_witness
        || options.scout_staged_probes
        || options.scout_parallel_finalists
        || options.scout_overlap_automatic
        || options.scout_native_rct_aliases)
        && !options.scout_pilot_first
    {
        return Err(Error::InvalidInput(
            "pilot refinements require pilot-first".into(),
        ));
    }
    if options.scout_pilot_first && !options.adaptive_scout {
        return Err(Error::InvalidInput(
            "pilot-first requires adaptive scout".into(),
        ));
    }
    if options.scout_pilot_first && options.screen.training_cap.is_some() {
        return Err(Error::InvalidInput(
            "pilot-first requires full-strength probe training".into(),
        ));
    }
    if options.adaptive_scout && !matches!(params.speed_tier, 0 | 1) {
        return Err(Error::InvalidInput(
            "adaptive scout requires source effort 9 or 10".into(),
        ));
    }
    options.workers.validate()?;
    WorkerExecution {
        limits: options.workers,
        probe_workers: options.probe_workers,
        ..Default::default()
    }
    .validate()?;
    if batches.probe.is_some() && batches.planned_probe.is_some() {
        return Err(Error::InvalidInput(
            "choose legacy or planned probe batching, not both".into(),
        ));
    }
    super::scan::validate_source_input(image)?;
    if !super::policy::default_pixel_options(params) {
        return Err(Error::InvalidInput(
            "source pixel search requires default source options".into(),
        ));
    }
    // These switches change additional control flow, not merely crop geometry.
    if (options.screen.adaptive_coverage && !options.preflight.enabled)
        || (options.screen.regions.context_side != 0 && !options.scout_pilot_first)
        || (options.screen.regions.full_width_strips && !options.adaptive_scout)
    {
        return Err(Error::InvalidInput("driver probe overrides require preflight coverage or explicit probe APIs; context/full-width routes are selected through DriverOptions".into()));
    }
    let pixels = u64::from(image.width()) * u64::from(image.height());
    let direct = options.frontier.direct != DirectFrontier::Disabled
        && params.speed_tier == 1
        && refinable(image)
        && (pixels >= 2 << 20 || (options.frontier.compact && pixels >= 1 << 19));
    let adaptive_coverage = options.screen.adaptive_coverage || options.preflight.enabled;
    let coverage_min = if options.scout_pilot_first && adaptive_coverage {
        1 << 17
    } else {
        1 << 19
    };
    let scan =
        if direct || (options.coverage_scan && params.speed_tier <= 8 && pixels >= coverage_min) {
            Some(match batches.scan_pool {
                Some(pool) => scan_lossless_input_with_pool(image, options.scan, pool),
                None => scan_lossless_input(image, options.scan),
            })
        } else {
            None
        };
    let scan = scan.as_ref().filter(|s| s.available());
    let probe = Rc::new(RefCell::new(ProbeWork {
        callback: encode_probe,
        limits: options.screen.limits,
        encodes: 0,
        pixels: 0,
    }));
    let shared = Rc::clone(&probe);
    let callback: ProbeCallback<'_> = Box::new(move |p, sample| shared.borrow_mut().run(p, sample));
    let mut screen_options = options.screen.clone();
    screen_options.adaptive_coverage = adaptive_coverage;
    let mut screen = CandidateScreen::new(image, scan, screen_options.clone(), callback)?;
    let shared = Rc::clone(&probe);
    let callback: ProbeCallback<'_> = Box::new(move |p, sample| shared.borrow_mut().run(p, sample));
    screen_options.regions.context_side = 128;
    if options.scout_staged_probes {
        screen_options.regions.context_side = 0;
        screen_options.regions.full_width_strips = true;
        screen_options.regions.strip_rows = Some(8);
    }
    let mut context_screen = CandidateScreen::new(
        image,
        if options.scout_staged_probes {
            scan
        } else {
            None
        },
        screen_options,
        callback,
    )?;
    let planned = batches.planned_probe.take().or_else(|| {
        batches.probe.take().map(|mut legacy| {
            let callback: PlannedProbeBatchEncoder<'_> = Box::new(move |tasks, _| legacy(tasks));
            callback
        })
    });
    let probe_batch = planned.map(|mut batch| {
        let shared = Rc::clone(&probe);
        let callback: PlannedProbeBatchEncoder<'_> =
            Box::new(move |tasks: &[ProbeTask<'_>], kind| {
                let mut work = shared.borrow_mut();
                let pixels = tasks.iter().try_fold(0u64, |n, task| {
                    n.checked_add(u64::from(task.image.width()) * u64::from(task.image.height()))
                        .ok_or(Error::Limit("cumulative scheduler probe work"))
                })?;
                if work
                    .encodes
                    .checked_add(tasks.len())
                    .is_none_or(|n| n > work.limits.max_encodes)
                    || work
                        .pixels
                        .checked_add(pixels)
                        .is_none_or(|n| n > work.limits.max_encoded_pixels)
                {
                    return Err(Error::Limit("cumulative scheduler probe work"));
                }
                work.encodes += tasks.len();
                work.pixels += pixels;
                let sizes = batch(tasks, kind)?;
                if sizes.len() != tasks.len() || sizes.contains(&0) {
                    return Err(Error::Encoding(
                        "invalid probe batch result count/size".into(),
                    ));
                }
                Ok(sizes)
            });
        Rc::new(RefCell::new(callback))
    });
    if let Some(batch) = &probe_batch {
        let shared = Rc::clone(batch);
        screen.set_planned_batch_encoder(Box::new(move |tasks, kind| {
            shared.borrow_mut()(tasks, kind)
        }));
        let shared = Rc::clone(batch);
        context_screen.set_planned_batch_encoder(Box::new(move |tasks, kind| {
            shared.borrow_mut()(tasks, kind)
        }));
    }
    let prefix_batch = batches.prefix.take().map(|mut batch| {
        let shared = Rc::clone(&probe);
        let callback: PrefixBatchEncoder<'_> = Box::new(move |screen| {
            {
                let work = shared.borrow();
                screen.work.limit_to(
                    work.limits.max_encodes - work.encodes,
                    work.limits.max_encoded_pixels - work.pixels,
                );
            }
            let result = batch(screen);
            // The synchronous callback has drained all roots, even on error.
            // Only actual attempts, not the maximum possible prefix, are charged.
            let usage = screen.work.usage();
            let mut work = shared.borrow_mut();
            work.encodes += usage.encodes;
            work.pixels += usage.pixels;
            result
        });
        callback
    });
    let driver = Driver {
        image,
        input: params,
        options,
        scan,
        level: (10 - params.speed_tier) as u8,
        search: Search::new(options.full_limits),
        encode: encode_full,
        full_batch: batches.full,
        pilot_overlap: batches.pilot_overlap,
        probe,
        probe_batch,
        prefix_batch,
        screen,
        context_screen,
        auxiliary_screen_batches: 0,
        routes: ActiveRoutes::new(options, pixels, scan),
        calibration_feedback: Vec::new(),
        early_racing: false,
        early_family_winner: false,
        pending_automatic: false,
        analysis: ProbeResult::default(),
        frontier: Vec::new(),
        deferred: Vec::new(),
        confirmation_cache: None,
        events: Vec::new(),
    };
    driver.run()
}

impl<'a, F, P, A> Driver<'a, F, P, A>
where
    F: FnMut(&CandidateParams, &Image, TrialBudget) -> Result<TrialOutput<A>>,
    P: FnMut(&CandidateParams, &Image) -> Result<usize>,
{
    fn best_params(&self, before: u8) -> Result<CandidateParams> {
        self.search
            .best_params_before(before)
            .cloned()
            .ok_or_else(|| Error::Encoding("missing completed search anchor".into()))
    }
    fn best_size(&self, before: u8) -> Result<usize> {
        self.search
            .best_before(before, false)
            .ok_or_else(|| Error::Encoding("missing completed search size".into()))
    }
    fn full(
        &mut self,
        stage: u8,
        params: CandidateParams,
        baseline: bool,
        calibration: bool,
    ) -> Result<Option<usize>> {
        self.search.refine(
            self.image,
            stage,
            params,
            baseline,
            calibration,
            &mut self.encode,
        )
    }
    fn initial_full(
        &mut self,
        requests: &[super::policy::TrialRequest],
    ) -> Result<Vec<Option<usize>>> {
        let concurrency = if self.full_batch.is_some() {
            initial_concurrency(self.image)
        } else {
            1
        };
        let mut sizes = Vec::with_capacity(requests.len());
        for chunk in requests.chunks(concurrency) {
            if chunk.len() == 1 {
                sizes.push(self.full(chunk[0].stage, chunk[0].params.clone(), true, false)?);
                continue;
            }
            self.search.check_batch_work(self.image, chunk.len())?;
            let tasks: Vec<_> = chunk
                .iter()
                .map(|request| FullTask {
                    params: &request.params,
                    cooperative: false,
                    budget: TrialBudget {
                        beat_bytes: None,
                        max_output_bytes: self.options.full_limits.max_output_bytes,
                    },
                })
                .collect();
            let outputs = self.full_batch.as_mut().expect("batch selected")(self.image, &tasks)?;
            if outputs.len() != chunk.len() {
                return Err(Error::Encoding(
                    "invalid full-frame batch result count".into(),
                ));
            }
            // Search::RunPending commits in plan order, never completion order.
            for (request, output) in chunk.iter().zip(outputs) {
                let mut output = Some(output);
                sizes.push(self.search.refine(
                    self.image,
                    request.stage,
                    request.params.clone(),
                    true,
                    false,
                    &mut |_, _, _| Ok(output.take().expect("one invocation")),
                )?);
            }
        }
        Ok(sizes)
    }
    fn calibration_full(&mut self, stage: u8, params: CandidateParams) -> Result<usize> {
        self.full(stage, params, false, true)?.ok_or_else(|| {
            Error::Encoding(
                "calibration requires a completed measurement, not a prior bound".into(),
            )
        })
    }
    fn defer(&mut self, stage: u8, params: CandidateParams, screen: bool) {
        self.events.push(SearchEvent::Deferred {
            stage,
            params: params.clone(),
            screened_on_replay: screen,
        });
        self.deferred.push(Deferred { params, screen });
    }
    fn analyze(&mut self, incumbent: &CandidateParams, ambiguous: bool) -> Result<()> {
        let mut analysis =
            if (self.scan.is_some() || self.routes.preflight_yield) && self.screen.available() {
                self.screen
                    .analyze(self.input, incumbent, self.options.plan)?
            } else if let Some(prefix) = &mut self.prefix_batch {
                let probe = Rc::clone(&self.probe);
                let batch = self.probe_batch.as_ref();
                probe_lossless_candidates_with_prefix(
                    self.image,
                    self.input,
                    Some(incumbent),
                    self.options.plan,
                    self.options.probe_feedback_bounds,
                    self.options.screen.limits,
                    |tasks| match batch {
                        Some(batch) => batch.borrow_mut()(tasks, ProbeBatchKind::Round),
                        None => tasks
                            .iter()
                            .map(|task| probe.borrow_mut().run(task.params, task.image))
                            .collect(),
                    },
                    prefix,
                )?
            } else if let Some(batch) = &self.probe_batch {
                probe_lossless_candidates_batched(
                    self.image,
                    self.input,
                    Some(incumbent),
                    self.options.plan,
                    self.options.probe_feedback_bounds,
                    self.options.screen.limits,
                    |tasks| batch.borrow_mut()(tasks, ProbeBatchKind::Round),
                )?
            } else {
                let probe = Rc::clone(&self.probe);
                probe_lossless_candidates(
                    self.image,
                    self.input,
                    Some(incumbent),
                    self.options.plan,
                    self.options.probe_feedback_bounds,
                    self.options.screen.limits,
                    move |p, sample| probe.borrow_mut().run(p, sample),
                )?
            };
        if analysis.ranked.is_empty() && ambiguous && !analysis.exploratory.is_empty() {
            analysis.ranked = std::mem::take(&mut analysis.exploratory);
        }
        if self.routes.multiscale_rank
            && let Some(leader) = analysis.ranked.first().cloned()
            && leader.params.palette_colors == 0
            && leader.params.predictor == Some(crate::modular::predictor::Predictor::Variable)
        {
            let mut siblings = Vec::new();
            for (i, candidate) in analysis.ranked.iter().enumerate() {
                let mut family = candidate.params.clone();
                family.rct = leader.params.rct;
                if family != leader.params
                    || candidate.context_bytes.unwrap_or(0) as f64
                        > leader.context_bytes.unwrap_or(0) as f64 * 1.06
                {
                    continue;
                }
                let context = self.context_screen.score(&candidate.params)?;
                siblings.push((
                    i,
                    candidate.context_bytes.unwrap_or(0) as f64 * total(&context)? as f64,
                ));
            }
            let positions = siblings.clone();
            siblings.sort_by(|a, b| a.1.total_cmp(&b.1));
            let ranked = analysis.ranked.clone();
            for ((position, _), (source, _)) in positions.into_iter().zip(siblings) {
                analysis.ranked[position] = ranked[source].clone();
            }
        }
        self.analysis = analysis;
        Ok(())
    }
    fn confirm(&mut self, stage: u8, candidate: &CandidateParams) -> Result<bool> {
        if !self.options.confirmation {
            return Ok(true);
        }
        let incumbent = self
            .search
            .best_index_before(stage, false)
            .ok_or_else(|| Error::Encoding("confirmation without incumbent".into()))?;
        let params = self.search.trials()[incumbent].params.clone();
        let cached = self
            .confirmation_cache
            .filter(|&(i, _)| i == incumbent)
            .map(|(_, bytes)| bytes);
        let probe = Rc::clone(&self.probe);
        let result = if let Some(batch) = &self.probe_batch {
            confirm_lossless_candidate_batched(
                self.image,
                candidate,
                &params,
                cached,
                self.options.screen.limits,
                |tasks| batch.borrow_mut()(tasks, ProbeBatchKind::Round),
            )?
        } else {
            confirm_lossless_candidate(
                self.image,
                candidate,
                &params,
                cached,
                self.options.screen.limits,
                move |p, sample| probe.borrow_mut().run(p, sample),
            )?
        };
        self.confirmation_cache = Some((incumbent, result.incumbent_bytes));
        self.events.push(SearchEvent::Confirmation {
            stage,
            samples: result.samples,
            candidate_bytes: result.candidate_bytes,
            incumbent_bytes: result.incumbent_bytes,
            accepted: result.accepted,
        });
        Ok(result.accepted)
    }
    fn deep_winner(&self, before: u8, geometry: bool) -> Result<CandidateParams> {
        let mut deeper = self.best_params(before)?;
        deeper.speed_tier = 0;
        deeper.learning_fraction = deeper.learning_fraction.max(0.5);
        deeper.lz77_careful_first_pass = true;
        if geometry {
            deeper.max_properties = self.image.channel_count().saturating_sub(1).min(4) as u8;
            deeper.group_size_shift = Some(3);
        }
        Ok(deeper)
    }
    fn finish(self) -> Result<SearchResult<A>> {
        let probes = self.probe.borrow();
        let probe_encodes = probes.encodes;
        let probe_pixels = probes.pixels;
        drop(probes);
        let retained = self.search.finish()?;
        Ok(SearchResult {
            bytes: retained.bytes,
            params: retained.params,
            artifact: retained.artifact,
            report: SearchReport {
                source_commit: super::SOURCE_COMMIT,
                backend_semantics: self.options.backend_semantics.clone(),
                trials: retained.trials,
                winner_index: retained.winner_index,
                full_encodes: retained.full_encodes,
                full_pixels: retained.full_pixels,
                probe_encodes,
                probe_pixels,
                screen_batches: self.screen.batches()
                    + self.context_screen.batches()
                    + self.auxiliary_screen_batches,
                events: self.events,
                unported: if self.prefix_batch.is_some() {
                    UNPORTED
                } else {
                    LEGACY_UNPORTED
                },
            },
        })
    }
}
