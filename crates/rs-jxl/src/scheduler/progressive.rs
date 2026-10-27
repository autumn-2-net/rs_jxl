//! Progressive original-group racing. Estimates choose work; only completed
//! native encodings are outputs and only retained wire bytes are lower bounds.
use std::time::Instant;

use super::{
    Candidate, CandidateParams, DriverOptions, NativeBackendOptions, NativeSearchResult,
    batch::pilot_handoff::{PilotRequest, PilotUsage},
    driver::{BackendSemantics, SearchEvent, SearchReport, SearchResult},
    native::{NativeAdaptation, NativeArtifact, NativeTrialKind, prepare, scout_execution_options},
    params::lossless_probe_plan,
    policy::{literal, winner_neighborhood},
    scan::{LosslessScan, scan_lossless_input, scan_lossless_input_with_pool},
    search::{EncodedOutput, Search, TrialBudget, TrialOutput},
};
use crate::{
    EncodeOptions, Error, Image, Result, encode::session::PixelSession,
    modular::predictor::Predictor,
};

#[path = "group_admission.rs"]
mod group_admission;
#[path = "group_regions.rs"]
mod regions;

#[cfg(test)]
#[path = "progressive_resume_tests.rs"]
mod resume_tests;

/// Work/memory ceilings for real-group probes, not timeouts or effort changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRaceOptions {
    /// Use the cheap pilot for sparse/low-color content, or when initial real
    /// probes exceed two full images of input work. Wide integers retain the
    /// old policy. A decisive cheap front skips real-group probing entirely.
    /// Disable only for experiments.
    pub content_route: bool,
    /// Preserve cheap-front admission for ambiguous fronts. Natural and Zero
    /// representations remain protected. Disable only for paired experiments.
    pub shortlist: bool,
    /// At least three activity strata are required before heuristic rejection.
    pub probe_regions: usize,
    /// Maximum regions for refining close challengers against a finished
    /// reference's group lengths, without re-encoding that reference.
    pub max_probe_regions: usize,
    /// Required weighted disadvantage in basis points, with no more than
    /// 0.25% total weighted regional gains. Extra rounds reduce
    /// this margin to a floor of 25 basis points. Not a final-size guarantee.
    pub discard_margin_basis_points: u16,
    /// Retained encoded payload/output bytes at checkpoints; not a process RSS
    /// ceiling (worker scratch and temporary framing copies are separate).
    /// Exceeding this is an error, not a fallback; unlimited by default.
    pub max_retained_bytes: usize,
}
impl Default for GroupRaceOptions {
    fn default() -> Self {
        Self {
            content_route: true,
            shortlist: true,
            probe_regions: 3,
            max_probe_regions: 9,
            discard_margin_basis_points: 100,
            max_retained_bytes: usize::MAX,
        }
    }
}
impl GroupRaceOptions {
    pub(crate) fn validate(self) -> Result<()> {
        if !(3..=12).contains(&self.probe_regions)
            || !(self.probe_regions..=12).contains(&self.max_probe_regions)
            || self.discard_margin_basis_points < 25
            || self.discard_margin_basis_points > 10000
            || self.max_retained_bytes == 0
        {
            return Err(Error::InvalidInput(
                "invalid progressive group race limits".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateState {
    Sampled,
    /// Cheap-front heuristic rejection, before encoding any original groups.
    FrontScreened,
    Screened,
    Complete,
    ByteBounded,
}

#[derive(Clone, Debug)]
pub struct GroupCandidateReport {
    pub name: &'static str,
    pub params: CandidateParams,
    pub state: CandidateState,
    pub total_groups: usize,
    pub encoded_groups: usize,
    pub reused_groups: usize,
    pub encoded_pixels: u64,
    /// Cheap crop scores, not comparable to original-group byte lengths.
    pub pilot_scores: Vec<usize>,
    pub sampled_region_bytes: Vec<usize>,
    pub estimated_bytes: f64,
    pub seconds: f64,
    pub final_bytes: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct ProgressiveReport {
    pub regions: Vec<crate::modular::layout::Rect>,
    pub candidates: Vec<GroupCandidateReport>,
    pub peak_retained_bytes: usize,
    pub group_pixels: u64,
}

struct Contender<'a> {
    candidate: Candidate,
    artifact: NativeArtifact,
    session: Option<PixelSession<'a>>,
    report: GroupCandidateReport,
    completed_group_bytes: Option<Vec<usize>>,
}

fn candidate_plan(
    image: &Image,
    input: &CandidateParams,
    driver: &DriverOptions,
    scan: &LosslessScan,
) -> Result<Vec<Candidate>> {
    let auto = input.backend(9)?;
    let mut plan = vec![
        Candidate::new("automatic", auto.clone()),
        Candidate::new(
            "neighborhood",
            winner_neighborhood(&auto, image.channel_count()),
        ),
        Candidate::new("literal", literal(input, 9, 0.01)?),
    ];
    plan.extend(
        lossless_probe_plan(input, driver.plan)
            .into_iter()
            .filter(|c| c.params.predictor == Some(Predictor::Variable))
            .filter(|c| !(image.channel_count() < 3 && c.params.rct == Some(6))),
    );
    if scan.sampled_pixels != 0
        && scan.sampled_colors <= 256
        && scan.sampled_flat * 4 >= scan.sampled_pixels * 3
    {
        plan.push(Candidate::new("literal-static", literal(input, 9, 0.0)?));
    }
    let mut unique: Vec<Candidate> = Vec::new();
    for candidate in plan {
        if !unique.iter().any(|c| c.params == candidate.params) {
            unique.push(candidate);
        }
    }
    Ok(unique)
}

fn estimate(scores: &[usize], regions: &[regions::Region]) -> f64 {
    let mut bytes = [0.0; 3];
    let mut pixels = [0.0; 3];
    let mut weights = [0.0; 3];
    for (&score, region) in scores.iter().zip(regions) {
        bytes[region.stratum] += score as f64;
        pixels[region.stratum] += (region.rect.width * region.rect.height) as f64;
        weights[region.stratum] = region.weight;
    }
    (0..3)
        .filter(|&i| pixels[i] != 0.0)
        .map(|i| bytes[i] / pixels[i] * weights[i])
        .sum()
}

fn dominated(
    candidate: &[usize],
    leader: &[usize],
    regions: &[regions::Region],
    margin: u16,
) -> bool {
    if candidate.len() < 3 || candidate.len() != leader.len() || candidate.len() != regions.len() {
        return false;
    }
    let leader_size = estimate(leader, regions);
    let mut pixels = [0usize; 3];
    for r in regions {
        pixels[r.stratum] += r.rect.width * r.rect.height;
    }
    // Sum positive regional gains instead of letting a single tiny local win
    // veto strong aggregate evidence. This is a sampled policy, never a bound.
    let mut gain = 0.0;
    for ((&a, &b), r) in candidate.iter().zip(leader).zip(regions) {
        gain += b.saturating_sub(a) as f64 * r.weight / pixels[r.stratum] as f64;
    }
    if leader_size <= 0.0 || gain > leader_size * 0.0025 {
        return false;
    }
    estimate(candidate, regions) >= estimate(leader, regions) * (1.0 + f64::from(margin) / 10000.0)
}

fn retained(
    contenders: &[Contender<'_>],
    incumbent_bytes: usize,
    options: GroupRaceOptions,
) -> Result<usize> {
    let size = contenders.iter().try_fold(incumbent_bytes, |total, c| {
        total
            .checked_add(c.session.as_ref().map_or(0, |s| s.groups.retained_bytes()))
            .ok_or(Error::Limit("progressive retained bytes"))
    })?;
    if size > options.max_retained_bytes {
        return Err(Error::Limit("progressive retained bytes"));
    }
    Ok(size)
}

fn with_pool<T: Send>(pool: Option<&rayon::ThreadPool>, work: impl FnOnce() -> T + Send) -> T {
    match pool {
        Some(pool) => crate::execution::with_pool(pool, work),
        None => crate::execution::serial(work),
    }
}

fn group_batch(contender: &Contender<'_>, pool: Option<&rayon::ThreadPool>) -> usize {
    let area = (128usize << contender.artifact.options.group_size_shift).pow(2);
    pool.map_or(1, rayon::ThreadPool::current_num_threads)
        .min((4usize << 20).div_ceil(area))
        .max(1)
}

fn sample(
    contender: &mut Contender<'_>,
    regions: &[regions::Region],
    image: &Image,
    pool: Option<&rayon::ThreadPool>,
) -> Result<()> {
    let batch = group_batch(contender, pool);
    let groups = regions
        .iter()
        .map(|r| {
            regions::groups_for_rect(
                r.rect,
                image.width() as usize,
                image.height() as usize,
                contender.artifact.options.group_size_shift,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let pending: Vec<_> = groups.iter().flatten().copied().collect();
    let session = contender.session.as_mut().expect("active group session");
    let start = Instant::now();
    for ids in pending.chunks(batch) {
        with_pool(pool, || session.groups.encode_groups(ids))?;
    }
    contender.report.seconds += start.elapsed().as_secs_f64();
    for (ids, region) in groups.iter().zip(regions) {
        let bytes = ids.iter().try_fold(0usize, |n, &id| {
            n.checked_add(
                session
                    .groups
                    .group_bytes(id)
                    .expect("completed sample group"),
            )
            .ok_or(Error::Limit("group sample bytes"))
        })?;
        contender.report.sampled_region_bytes.push(bytes);
        contender.report.encoded_groups = session.groups.completed_groups();
        contender.report.encoded_pixels += (region.rect.width * region.rect.height) as u64;
    }
    Ok(())
}

fn completed_scores(
    contender: &Contender<'_>,
    regions: &[regions::Region],
    image: &Image,
) -> Result<Vec<usize>> {
    let bytes = contender
        .completed_group_bytes
        .as_ref()
        .expect("completed reference");
    regions
        .iter()
        .map(|r| {
            regions::groups_for_rect(
                r.rect,
                image.width() as usize,
                image.height() as usize,
                contender.artifact.options.group_size_shift,
            )?
            .iter()
            .try_fold(0usize, |sum, &id| {
                sum.checked_add(bytes[id])
                    .ok_or(Error::Limit("reference group bytes"))
            })
        })
        .collect()
}

fn best_group(contenders: &[Contender<'_>]) -> Option<usize> {
    contenders
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.report.final_bytes.map(|n| (i, n)))
        .min_by_key(|&(_, n)| n)
        .map(|(i, _)| i)
}

fn initial_work_fits(pixels: u64, sampled: u64, candidates: usize) -> bool {
    pixels != 0
        && sampled != 0
        && candidates != 0
        && pixels
            .checked_mul(2)
            .zip(sampled.checked_mul(candidates as u64))
            .is_some_and(|(budget, work)| work <= budget)
}

fn reserve_full_work(pixels: u64, trials: usize, driver: &DriverOptions) -> Result<()> {
    if trials > driver.full_limits.max_full_encodes
        || pixels
            .checked_mul(trials as u64)
            .is_none_or(|n| n > driver.full_limits.max_full_pixels)
    {
        return Err(Error::Limit("group race work"));
    }
    Ok(())
}

fn decisive_front(
    ranked: &[(Candidate, Vec<usize>)],
    driver: &DriverOptions,
    scan: &LosslessScan,
) -> Option<CandidateParams> {
    let (leader, scores) = ranked.first()?;
    (!ranked[1..].iter().any(|(candidate, candidate_scores)| {
        super::driver::scout_pilot::pilot_eligible(
            9,
            driver.scout_pilot_consensus,
            driver.scout_near_tie_consensus && scan.likely_continuous_tone(),
            &leader.params,
            scores,
            &candidate.params,
            candidate_scores,
        )
    }))
    .then(|| leader.params.clone())
}

fn finish(
    search: Search<NativeArtifact>,
    adaptations: Vec<NativeAdaptation>,
    events: Vec<SearchEvent>,
    usage: PilotUsage,
    progressive: Option<ProgressiveReport>,
) -> Result<NativeSearchResult> {
    let retained = search.finish()?;
    let report = SearchReport {
        source_commit: super::SOURCE_COMMIT,
        backend_semantics: BackendSemantics::DeclaredProxy {
            differences: vec![
                "progressive pilot selection; inspect adaptations, events and group report".into(),
            ],
        },
        trials: retained.trials,
        winner_index: retained.winner_index,
        full_encodes: retained.full_encodes,
        full_pixels: retained.full_pixels,
        probe_encodes: usage.encodes,
        probe_pixels: usage.pixels,
        screen_batches: usage.batches,
        events,
        unported: &["sample-based elimination is not a full-frame size guarantee"],
    };
    Ok(NativeSearchResult {
        output: SearchResult {
            bytes: retained.bytes,
            artifact: retained.artifact,
            params: retained.params,
            report,
        },
        adaptations,
        preparation_reuse: None,
        progressive,
    })
}

#[cfg(test)]
fn resume(
    contender: &mut Contender<'_>,
    cutoff: Option<usize>,
    pool: &rayon::ThreadPool,
    other_retained: usize,
    options: GroupRaceOptions,
    peak: &mut usize,
) -> Result<TrialOutput<NativeArtifact>> {
    resume_in_pool(contender, cutoff, Some(pool), other_retained, options, peak)
}

fn resume_in_pool(
    contender: &mut Contender<'_>,
    cutoff: Option<usize>,
    pool: Option<&rayon::ThreadPool>,
    other_retained: usize,
    options: GroupRaceOptions,
    peak: &mut usize,
) -> Result<TrialOutput<NativeArtifact>> {
    let mut session = contender.session.take().expect("resumable finalist");
    contender.report.reused_groups = session.groups.completed_groups();
    let missing: Vec<_> = (0..session.groups.group_count())
        .filter(|&i| session.groups.group_bytes(i).is_none())
        .collect();
    // Limit live group scratch, while each group may still use the same pool
    // for its MA subtree/property work. No nested worker pools are created.
    let batch = group_batch(contender, pool);
    for ids in missing.chunks(batch) {
        if let Some(bound) = cutoff.filter(|&b| session.groups.retained_bytes() >= b) {
            contender.report.state = CandidateState::ByteBounded;
            return Ok(TrialOutput::Bounded { lower_bound: bound });
        }
        with_pool(pool, || session.groups.encode_groups(ids))?;
        contender.report.encoded_groups = session.groups.completed_groups();
        for &id in ids {
            let r = session.groups.group_rect(id)?;
            contender.report.encoded_pixels += (r.width * r.height) as u64;
        }
        let retained = other_retained
            .checked_add(session.groups.retained_bytes())
            .ok_or(Error::Limit("progressive retained bytes"))?;
        *peak = (*peak).max(retained);
        if retained > options.max_retained_bytes {
            return Err(Error::Limit("progressive retained bytes"));
        }
    }
    // The last batch (or initial samples) can already prove this candidate
    // cannot win. Do not allocate framing or hit an output cap for that loser.
    if cutoff.is_some_and(|b| session.groups.retained_bytes() >= b) {
        contender.report.state = CandidateState::ByteBounded;
        return Ok(TrialOutput::Bounded {
            lower_bound: session.groups.retained_bytes(),
        });
    }
    contender.completed_group_bytes = Some(
        (0..session.groups.group_count())
            .map(|i| session.groups.group_bytes(i).expect("completed group"))
            .collect(),
    );
    let bytes = session.finish()?;
    let size = other_retained
        .checked_add(bytes.len())
        .ok_or(Error::Limit("progressive retained bytes"))?;
    *peak = (*peak).max(size);
    if size > options.max_retained_bytes {
        return Err(Error::Limit("progressive retained bytes"));
    }
    contender.report.state = CandidateState::Complete;
    contender.report.final_bytes = Some(bytes.len());
    Ok(TrialOutput::Complete(EncodedOutput {
        bytes,
        artifact: contender.artifact.clone(),
    }))
}

/// None means no real-group encoding was started and the existing pilot must
/// run with its original arguments. Errors are never hidden as ineligibility.
pub(crate) fn try_encode(
    image: &Image,
    input: &CandidateParams,
    driver: &DriverOptions,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    options: GroupRaceOptions,
) -> Result<Option<NativeSearchResult>> {
    if !validate_input(image, input, base, settings, options)? {
        return Ok(None);
    }
    if base.threads == 1 || crate::execution::is_serial() {
        return try_encode_in_pool(image, input, driver, base, settings, options, None);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(base.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    try_encode_in_pool(image, input, driver, base, settings, options, Some(&pool))
}

pub(crate) fn try_encode_in_pool(
    image: &Image,
    input: &CandidateParams,
    driver: &DriverOptions,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    options: GroupRaceOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Option<NativeSearchResult>> {
    if !validate_input(image, input, base, settings, options)? {
        return Ok(None);
    }
    let work = || try_encode_inner(image, input, driver, base, settings, options, pool);
    if pool.is_none() {
        crate::execution::serial(work)
    } else {
        work()
    }
}

fn validate_input(
    image: &Image,
    input: &CandidateParams,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    options: GroupRaceOptions,
) -> Result<bool> {
    options.validate()?;
    input.validate()?;
    let pixels = u64::from(image.width()) * u64::from(image.height());
    if input.speed_tier != 1
        || pixels < 2 << 20
        || (options.content_route
            && (image.bits() > 8
                || image
                    .metadata()
                    .extra_channels
                    .iter()
                    .any(|c| c.bit_depth.bits_per_sample > 8)))
        || !super::policy::default_pixel_options(input)
        || (base.modular.representation_receipt.is_some() && !settings.record_representations)
    {
        return Ok(false);
    }
    if !(1..=256).contains(&base.threads) {
        return Err(Error::InvalidInput("threads must be 1..=256".into()));
    }
    if pixels > base.max_pixels {
        return Err(Error::Limit("pixel count"));
    }
    if base.max_output_bytes == 0 {
        return Err(Error::Limit("encoded output bytes"));
    }
    Ok(true)
}

fn try_encode_inner(
    image: &Image,
    input: &CandidateParams,
    driver: &DriverOptions,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    options: GroupRaceOptions,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Option<NativeSearchResult>> {
    let pixels = u64::from(image.width()) * u64::from(image.height());
    let scan = match pool {
        Some(pool) => scan_lossless_input_with_pool(image, driver.scan, pool),
        None => scan_lossless_input(image, driver.scan),
    };
    if options.content_route && !scan.likely_continuous_tone() {
        return Ok(None);
    }
    let mut base = scout_execution_options(base, true);
    base.max_output_bytes = base
        .max_output_bytes
        .min(driver.full_limits.max_output_bytes);
    let plan = candidate_plan(image, input, driver, &scan)?;
    reserve_full_work(pixels, 1, driver)?;
    if base.max_output_bytes == 0 {
        return Err(Error::Limit("encoded output bytes"));
    }
    let mut contenders = Vec::new();
    let mut adaptations = Vec::new();
    for candidate in plan {
        let (artifact, adaptation) = prepare(
            &candidate.params,
            image,
            &base,
            settings,
            NativeTrialKind::Full,
        )?;
        let start = Instant::now();
        let Some(session) = with_pool(pool, || PixelSession::new(image, &artifact.options))? else {
            return Ok(None);
        };
        let report = GroupCandidateReport {
            name: candidate.name,
            params: candidate.params.clone(),
            state: CandidateState::Sampled,
            total_groups: session.groups.group_count(),
            encoded_groups: 0,
            reused_groups: 0,
            encoded_pixels: 0,
            pilot_scores: Vec::new(),
            sampled_region_bytes: Vec::new(),
            estimated_bytes: 0.0,
            seconds: start.elapsed().as_secs_f64(),
            final_bytes: None,
        };
        contenders.push(Contender {
            candidate,
            artifact,
            session: Some(session),
            report,
            completed_group_bytes: None,
        });
        adaptations.push(adaptation);
    }
    let side = contenders
        .iter()
        .map(|c| 128usize << c.artifact.options.group_size_shift)
        .max()
        .unwrap();
    let regions = regions::plan(image.width() as usize, image.height() as usize, side, &scan)?;
    let sample_count = options.probe_regions.min(regions.len());
    if sample_count < 3 {
        return Ok(None);
    }
    let sampled = &regions[..sample_count];
    let initial_pixels = sampled
        .iter()
        .map(|r| (r.rect.width * r.rect.height) as u64)
        .sum();
    if options.content_route && !initial_work_fits(pixels, initial_pixels, contenders.len()) {
        return Ok(None);
    }
    let mut usage = PilotUsage::default();
    let mut events = Vec::new();
    if options.content_route {
        let plan: Vec<_> = contenders.iter().map(|c| c.candidate.clone()).collect();
        let automatic = input.backend(9)?;
        let front = super::native::front::run_in_pool(
            &PilotRequest {
                image,
                automatic: &automatic,
                plan: &plan,
                options: driver,
                scan: Some(&scan),
                level: 9,
                budget: TrialBudget {
                    beat_bytes: None,
                    max_output_bytes: base.max_output_bytes,
                },
                probe_limits: driver.screen.limits,
            },
            &base,
            settings,
            pool,
            &mut adaptations,
        )?;
        usage = front.usage;
        events = front.events;
        let ranked = front.ranked?;
        for contender in &mut contenders {
            if let Some((_, scores)) = ranked
                .iter()
                .find(|(c, _)| c.params == contender.candidate.params)
            {
                contender.report.pilot_scores = scores.clone();
            }
        }
        if let Some(params) = decisive_front(&ranked, driver, &scan) {
            let witnesses = if driver.scout_representation_witness {
                super::driver::pilot_witness::candidates(image, input, &params, Some(&scan))?
            } else {
                Vec::new()
            };
            {
                // No partial groups have been encoded. Complete the witnesses
                // and this finalist once, without probing unrelated groups.
                let mut finalists = witnesses;
                if !finalists.contains(&params) {
                    finalists.push(params);
                }
                reserve_full_work(pixels, finalists.len(), driver)?;
                drop(contenders);
                adaptations.retain(|a| a.kind == NativeTrialKind::Probe);
                let mut search = Search::new(driver.full_limits);
                for params in &finalists {
                    // Session construction starts its diagnostic collector.
                    // Standalone invocations require fresh collectors.
                    let (artifact, adaptation) =
                        prepare(params, image, &base, settings, NativeTrialKind::Full)?;
                    adaptations.push(adaptation);
                    let retained = search.best_before(10, false).unwrap_or(0);
                    let baseline = search.trials().is_empty();
                    search.refine(
                        image,
                        9,
                        params.clone(),
                        baseline,
                        false,
                        &mut |_, _, budget| {
                            Ok(
                                match super::native::encode_full_trial(
                                    image,
                                    &artifact.options,
                                    budget.beat_bytes,
                                    pool,
                                    None,
                                )? {
                                    crate::encode::budget::TrialEncode::Complete(bytes) => {
                                        if retained
                                            .checked_add(bytes.len())
                                            .is_none_or(|n| n > options.max_retained_bytes)
                                        {
                                            return Err(Error::Limit("progressive retained bytes"));
                                        }
                                        TrialOutput::Complete(EncodedOutput {
                                            bytes,
                                            artifact: artifact.clone(),
                                        })
                                    }
                                    crate::encode::budget::TrialEncode::Bounded { lower_bound } => {
                                        TrialOutput::Bounded { lower_bound }
                                    }
                                },
                            )
                        },
                    )?;
                }
                events.push(SearchEvent::Route {
                    name: "progressive-cheap-decisive",
                    candidates: finalists.len(),
                });
                return finish(search, adaptations, events, usage, None).map(Some);
            }
        }
        if options.shortlist {
            let admitted = group_admission::admission(&plan, &ranked, driver, &scan);
            for (contender, keep) in contenders.iter_mut().zip(admitted) {
                if !keep {
                    contender.session = None;
                    contender.report.state = CandidateState::FrontScreened;
                }
            }
        }
    }
    // Only ambiguous fronts need this reservation. A decisive front above
    // legitimately fits a one-full-encode caller budget. No real groups have
    // run yet; two extra full trials cover representation witnesses.
    let active = contenders.iter().filter(|c| c.session.is_some()).count();
    reserve_full_work(pixels, active + 2, driver)?;
    let mut peak = 0;
    for i in 0..contenders.len() {
        if contenders[i].session.is_none() {
            continue;
        }
        sample(&mut contenders[i], sampled, image, pool)?;
        peak = peak.max(retained(&contenders, 0, options)?);
    }
    for contender in &mut contenders {
        contender.report.estimated_bytes =
            estimate(&contender.report.sampled_region_bytes, sampled);
    }
    let mut order: Vec<_> = (0..contenders.len())
        .filter(|&i| contenders[i].session.is_some())
        .collect();
    order.sort_by(|&a, &b| {
        contenders[a]
            .report
            .estimated_bytes
            .total_cmp(&contenders[b].report.estimated_bytes)
    });
    let leader = order[0];
    let leader_scores = &contenders[leader].report.sampled_region_bytes;
    let witness_leader = order
        .iter()
        .copied()
        .find(|&i| {
            contenders[i].candidate.params.predictor == Some(Predictor::Zero)
                && super::driver::scout_pilot::competitive(
                    &contenders[i].report.sampled_region_bytes,
                    leader_scores,
                )
        })
        .unwrap_or(leader);
    let witnesses = if driver.scout_representation_witness {
        super::driver::pilot_witness::candidates(
            image,
            input,
            &contenders[witness_leader].candidate.params,
            Some(&scan),
        )?
    } else {
        Vec::new()
    };
    let mut search = Search::new(driver.full_limits);
    // Global palette/patch witnesses remain actual complete native encodes.
    // Existing group candidates reuse their own samples instead of restarting.
    for witness in witnesses {
        let before = search.best_before(10, false);
        if let Some(index) = contenders
            .iter()
            .position(|c| c.session.is_some() && c.candidate.params == witness)
        {
            let baseline = search.trials().is_empty();
            let other = retained(
                &contenders,
                search.best_before(10, false).unwrap_or(0),
                options,
            )? - contenders[index]
                .session
                .as_ref()
                .unwrap()
                .groups
                .retained_bytes();
            let start = Instant::now();
            search.refine(image, 9, witness, baseline, false, &mut |_, _, budget| {
                resume_in_pool(
                    &mut contenders[index],
                    budget.beat_bytes,
                    pool,
                    other,
                    options,
                    &mut peak,
                )
            })?;
            contenders[index].report.seconds += start.elapsed().as_secs_f64();
        } else {
            let (artifact, adaptation) =
                prepare(&witness, image, &base, settings, NativeTrialKind::Full)?;
            adaptations.push(adaptation);
            let baseline = search.trials().is_empty();
            let other = retained(&contenders, before.unwrap_or(0), options)?;
            search.refine(image, 9, witness, baseline, false, &mut |_, _, budget| {
                Ok(
                    match super::native::encode_full_trial(
                        image,
                        &artifact.options,
                        budget.beat_bytes,
                        pool,
                        None,
                    )? {
                        crate::encode::budget::TrialEncode::Complete(bytes) => {
                            let retained = other
                                .checked_add(bytes.len())
                                .ok_or(Error::Limit("progressive retained bytes"))?;
                            peak = peak.max(retained);
                            if retained > options.max_retained_bytes {
                                return Err(Error::Limit("progressive retained bytes"));
                            }
                            TrialOutput::Complete(EncodedOutput {
                                bytes,
                                artifact: artifact.clone(),
                            })
                        }
                        crate::encode::budget::TrialEncode::Bounded { lower_bound } => {
                            TrialOutput::Bounded { lower_bound }
                        }
                    },
                )
            })?;
        }
        peak = peak.max(retained(
            &contenders,
            search.best_before(10, false).unwrap_or(0),
            options,
        )?);
        events.push(SearchEvent::Route {
            name: "progressive-representation-witness",
            candidates: 1,
        });
    }
    for index in order {
        if contenders[index].session.is_none() {
            continue;
        }
        let reference = best_group(&contenders).unwrap_or(leader);
        let learned_witness = contenders[reference].candidate.params.predictor
            == Some(Predictor::Zero)
            && contenders[reference].candidate.params.learning_fraction == 0.0
            && contenders[index].candidate.params.predictor == Some(Predictor::Zero)
            && contenders[index].candidate.params.learning_fraction > 0.0;
        let can_refine = contenders[reference].completed_group_bytes.is_some();
        if index != reference && !learned_witness {
            loop {
                let session = contenders[index].session.as_ref().unwrap();
                if session.groups.completed_groups() == session.groups.group_count()
                    || search
                        .best_before(10, false)
                        .is_some_and(|b| session.groups.retained_bytes() >= b)
                {
                    break;
                }
                let count = contenders[index].report.sampled_region_bytes.len();
                let current = &regions[..count];
                let scores = if can_refine {
                    completed_scores(&contenders[reference], current, image)?
                } else {
                    contenders[reference].report.sampled_region_bytes.clone()
                };
                let margin = (options.discard_margin_basis_points
                    / (count / sample_count).max(1) as u16)
                    .max(25);
                if dominated(
                    &contenders[index].report.sampled_region_bytes,
                    &scores,
                    current,
                    margin,
                ) {
                    contenders[index].report.state = CandidateState::Screened;
                    contenders[index].session = None;
                    break;
                }
                let end = (count + sample_count)
                    .min(options.max_probe_regions)
                    .min(regions.len());
                if !can_refine
                    || end <= count
                    || contenders[index].report.estimated_bytes <= estimate(&scores, current)
                {
                    break;
                }
                sample(&mut contenders[index], &regions[count..end], image, pool)?;
                contenders[index].report.estimated_bytes = estimate(
                    &contenders[index].report.sampled_region_bytes,
                    &regions[..end],
                );
                peak = peak.max(retained(
                    &contenders,
                    search.best_before(10, false).unwrap_or(0),
                    options,
                )?);
            }
        }
        if contenders[index].session.is_none() {
            continue;
        }
        let baseline = search.trials().is_empty();
        let before = search.best_before(10, false);
        let other = retained(&contenders, before.unwrap_or(0), options)?
            - contenders[index]
                .session
                .as_ref()
                .unwrap()
                .groups
                .retained_bytes();
        let params = contenders[index].candidate.params.clone();
        let start = Instant::now();
        search.refine(image, 9, params, baseline, false, &mut |_, _, budget| {
            resume_in_pool(
                &mut contenders[index],
                budget.beat_bytes,
                pool,
                other,
                options,
                &mut peak,
            )
        })?;
        contenders[index].report.seconds += start.elapsed().as_secs_f64();
    }
    let progressive = ProgressiveReport {
        regions: regions
            .iter()
            .take(
                contenders
                    .iter()
                    .map(|c| c.report.sampled_region_bytes.len())
                    .max()
                    .unwrap_or(0),
            )
            .map(|r| r.rect)
            .collect(),
        peak_retained_bytes: peak,
        group_pixels: contenders.iter().map(|c| c.report.encoded_pixels).sum(),
        candidates: contenders.into_iter().map(|c| c.report).collect(),
    };
    events.push(SearchEvent::Route {
        name: "progressive-real-groups",
        candidates: progressive.candidates.len(),
    });
    finish(search, adaptations, events, usage, Some(progressive)).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_group_executor_uses_supplied_workers_and_null_stays_local() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let caller = std::thread::current().id();
        with_pool(None, || {
            assert!(crate::execution::is_serial());
            assert_eq!(crate::execution::worker_count(), 1);
            assert_eq!(std::thread::current().id(), caller);
        });
        with_pool(Some(&pool), || {
            assert!(pool.current_thread_index().is_some());
            assert_eq!(crate::execution::worker_count(), 2);
            let caller = std::thread::current().id();
            with_pool(None, || {
                assert!(crate::execution::is_serial());
                assert_eq!(std::thread::current().id(), caller);
            });
            assert!(!crate::execution::is_serial());
        });
        assert!(!crate::execution::is_serial());
    }

    #[test]
    fn retained_byte_defaults_are_unlimited_but_explicit_budgets_still_apply() {
        let options = GroupRaceOptions::default();
        options.validate().unwrap();
        assert_eq!(options.max_retained_bytes, usize::MAX);
        let beyond_old_cap = (256 << 20) + 1;
        assert_eq!(
            retained(&[], beyond_old_cap, options).unwrap(),
            beyond_old_cap
        );
        assert_eq!(retained(&[], usize::MAX, options).unwrap(), usize::MAX);
        let limited = GroupRaceOptions {
            max_retained_bytes: 7,
            ..options
        };
        assert_eq!(retained(&[], 7, limited).unwrap(), 7);
        assert!(matches!(
            retained(&[], 8, limited),
            Err(Error::Limit("progressive retained bytes"))
        ));
    }

    #[test]
    fn cheap_admission_keeps_the_old_single_finalist_decision() {
        let driver = DriverOptions::scout();
        let scan = LosslessScan {
            sampled_pixels: 1024,
            sampled_colors: 1000,
            ..Default::default()
        };
        let auto = CandidateParams::default().backend(9).unwrap();
        let cross = CandidateParams {
            predictor: Some(Predictor::Variable),
            ..auto.clone()
        };
        let mut ranked = vec![(Candidate::new("automatic", auto.clone()), vec![100; 3])];
        assert_eq!(decisive_front(&ranked, &driver, &scan), Some(auto.clone()));
        ranked.push((Candidate::new("cross", cross), vec![100; 3]));
        assert!(decisive_front(&ranked, &driver, &scan).is_none());
        ranked[1].1 = vec![200; 3];
        assert_eq!(decisive_front(&ranked, &driver, &scan), Some(auto));
        let learned = literal(&CandidateParams::default(), 9, 0.01).unwrap();
        ranked = vec![
            (
                Candidate::new(
                    "static",
                    literal(&CandidateParams::default(), 9, 0.0).unwrap(),
                ),
                vec![100; 3],
            ),
            (Candidate::new("learned", learned), vec![200; 3]),
        ];
        assert!(decisive_front(&ranked, &driver, &scan).is_none());
        assert!(decisive_front(&[], &driver, &scan).is_none());
    }

    #[test]
    fn wide_samples_keep_the_existing_policy_before_any_group_work() {
        let image = Image::new(
            2048,
            1024,
            crate::PixelLayout::Gray,
            crate::PixelData::U16(vec![0; 2048 * 1024]),
        )
        .unwrap();
        assert!(
            try_encode(
                &image,
                &CandidateParams::default().backend(9).unwrap(),
                &DriverOptions::scout(),
                &EncodeOptions::default(),
                NativeBackendOptions::default(),
                GroupRaceOptions::default()
            )
            .unwrap()
            .is_none()
        );
        let image = Image::new(
            2048,
            1024,
            crate::PixelLayout::Gray,
            crate::PixelData::U8(vec![0; 2048 * 1024]),
        )
        .unwrap();
        let mut metadata = image.metadata().clone();
        metadata.extra_channels.push(
            crate::metadata::ExtraChannelInfo::new(crate::metadata::ExtraChannelType::Depth, 16)
                .unwrap(),
        );
        let image = image
            .with_metadata(metadata, vec![crate::PixelData::U16(vec![0; 2048 * 1024])])
            .unwrap();
        assert!(
            try_encode(
                &image,
                &CandidateParams::default().backend(9).unwrap(),
                &DriverOptions::scout(),
                &EncodeOptions::default(),
                NativeBackendOptions::default(),
                GroupRaceOptions::default()
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn decisive_front_reserves_only_its_actual_finalists() {
        let mut driver = DriverOptions::scout();
        driver.full_limits.max_full_encodes = 2;
        driver.full_limits.max_full_pixels = 200;
        assert!(reserve_full_work(100, 1, &driver).is_ok());
        assert!(reserve_full_work(100, 2, &driver).is_ok());
        assert!(reserve_full_work(100, 3, &driver).is_err());
        assert!(reserve_full_work(101, 2, &driver).is_err());
        assert!(reserve_full_work(u64::MAX, 2, &driver).is_err());
    }

    #[test]
    fn initial_race_work_is_bounded_relative_to_the_original_image() {
        assert!(initial_work_fits(10, 3, 5));
        assert!(initial_work_fits(10, 4, 5));
        assert!(!initial_work_fits(10, 3, 7));
        assert!(!initial_work_fits(0, 1, 1));
        assert!(!initial_work_fits(10, 0, 1));
        assert!(!initial_work_fits(10, 1, 0));
        assert!(!initial_work_fits(u64::MAX, u64::MAX, 9));
    }

    fn regions() -> Vec<regions::Region> {
        (0..3)
            .map(|stratum| regions::Region {
                stratum,
                weight: 100.0,
                rect: crate::modular::layout::Rect {
                    x: stratum * 10,
                    y: 0,
                    width: 10,
                    height: 10,
                },
            })
            .collect()
    }
    #[test]
    fn heuristic_rejection_requires_three_losses_and_an_aggregate_margin() {
        let regions = regions();
        assert!(dominated(&[1020; 3], &[1000; 3], &regions, 100));
        for scores in [[1000; 3], [1005; 3], [990, 1100, 1100], [900, 1100, 1200]] {
            assert!(!dominated(&scores, &[1000; 3], &regions, 100));
        }
        assert!(!dominated(&[1100; 2], &[1000; 2], &regions[..2], 100));
        assert!(dominated(&[1100; 3], &[0, 1000, 1000], &regions, 100));
        assert!(!dominated(&[1100; 3], &[0; 3], &regions, 100));
        assert!(!dominated(&[], &[], &[], 100));
    }

    #[test]
    fn negligible_blank_headers_do_not_protect_large_content_losses() {
        let regions = regions();
        assert!(dominated(
            &[22, 65000, 667000],
            &[201, 46500, 460000],
            &regions,
            100
        ));
        assert!(!dominated(
            &[22, 45000, 667000],
            &[201, 46500, 460000],
            &regions,
            100
        ));
        // Header differences are not negligible in genuinely tiny images.
        assert!(!dominated(&[1, 101, 101], &[20, 100, 100], &regions, 100));
        assert!(dominated(&[999, 1040, 1040], &[1000; 3], &regions, 100));
        assert!(!dominated(&[990, 1040, 1040], &[1000; 3], &regions, 100));
    }
    #[test]
    fn score_amortizes_by_stratum_area_not_group_count() {
        let mut regions = regions();
        regions[0].weight = 900.0;
        assert_eq!(estimate(&[100, 200, 300], &regions), 1400.0);
        regions.push(regions[0]);
        assert_eq!(estimate(&[100, 200, 300, 100], &regions), 1400.0);
    }

    #[test]
    fn candidate_plan_keeps_automatic_literal_and_extended_cross_families() {
        let image = Image::new(
            1,
            1,
            crate::PixelLayout::Rgb,
            crate::PixelData::U8(vec![0; 3]),
        )
        .unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        let mut driver = DriverOptions::scout();
        driver.plan.extended_rct = true;
        let scan = LosslessScan {
            sampled_pixels: 1024,
            sampled_flat: 900,
            sampled_colors: 64,
            ..Default::default()
        };
        let plan = candidate_plan(&image, &input, &driver, &scan).unwrap();
        assert!(plan.iter().any(|c| c.name == "automatic"));
        assert!(plan.iter().any(|c| c.name == "literal-static"));
        for rct in [0, 6, 26] {
            assert!(
                plan.iter()
                    .any(|c| c.params.predictor == Some(Predictor::Variable)
                        && c.params.rct == Some(rct))
            );
        }
        assert!(plan.iter().all(|c| c.params.speed_tier == 1));
        for (i, a) in plan.iter().enumerate() {
            assert!(!plan[..i].iter().any(|b| b.params == a.params));
        }
    }

    #[test]
    fn unsupported_inputs_and_shared_receipts_fall_back_before_encoding() {
        use crate::modular::encode::receipt::ReceiptCollector;
        let small = Image::new(
            8,
            8,
            crate::PixelLayout::Gray,
            crate::PixelData::U8(vec![0; 64]),
        )
        .unwrap();
        let input = CandidateParams::default().backend(9).unwrap();
        let driver = DriverOptions::scout();
        let settings = NativeBackendOptions::default();
        let mut base = EncodeOptions::default();
        assert!(
            try_encode(
                &small,
                &input,
                &driver,
                &base,
                settings,
                GroupRaceOptions::default()
            )
            .unwrap()
            .is_none()
        );
        let image = Image::new(
            2048,
            1024,
            crate::PixelLayout::Gray,
            crate::PixelData::U8(vec![0; 2048 * 1024]),
        )
        .unwrap();
        let receipt = std::sync::Arc::new(ReceiptCollector::default());
        base.modular.representation_receipt = Some(receipt.clone());
        assert!(
            try_encode(
                &image,
                &input,
                &driver,
                &base,
                settings,
                GroupRaceOptions::default()
            )
            .unwrap()
            .is_none()
        );
        receipt.begin().unwrap();
        base.modular.representation_receipt = None;
        assert!(
            try_encode(
                &image,
                &input.backend(10).unwrap(),
                &driver,
                &base,
                settings,
                GroupRaceOptions::default()
            )
            .unwrap()
            .is_none()
        );
        let mut limited = driver;
        limited.full_limits.max_full_encodes = 0;
        assert!(
            try_encode(
                &image,
                &input,
                &limited,
                &base,
                settings,
                GroupRaceOptions::default()
            )
            .unwrap()
            .is_none()
        );
        assert!(matches!(
            try_encode(
                &image,
                &input,
                &limited,
                &base,
                settings,
                GroupRaceOptions {
                    content_route: false,
                    ..Default::default()
                }
            ),
            Err(Error::Limit("group race work"))
        ));
    }
}
