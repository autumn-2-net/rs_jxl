//! Opt-in concrete native encoder bridge. No default encode entry point is
//! changed. Strict and explicitly reported experimental adaptations are distinct.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::encode::budget::{SectionBudget, TrialEncode};
use crate::{EncodeOptions, Error, Image, Result};

use super::{
    adapter::{
        AdaptationState, CapabilityGap, NativeResourceLimits, NativeResourcePolicy,
        project_native_with_resources,
    },
    batch::BatchEncoders,
    driver::{BackendSemantics, DriverOptions, SearchResult, run_pixel_search_with_batches},
    params::{CandidateParams, Override},
    prefix_screen::run_prefix_screen,
    search::{EncodedOutput, TrialBudget, TrialOutput},
    worker_pool::{TaskRunner, WorkerExecution, WorkerPlan, run_plan},
};

#[path = "native_overlap.rs"]
mod native_overlap;

#[path = "native_tools.rs"]
mod native_tools;
#[cfg(test)]
#[path = "native_probe_sampling_tests.rs"]
mod probe_sampling_tests;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeAcceptance {
    #[default]
    RequireExact,
    /// Opt-in source-policy experiments; every projection difference is retained.
    AllowReportedGaps,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeBackendOptions {
    pub acceptance: NativeAcceptance,
    /// Explicit override for full native test runs. Does not pretend the source
    /// automatic patch decision was performed. Strict mode rejects a rewrite.
    pub patches_off: bool,
    /// Diagnostic only: bypass small-image tuple-palette rejection in Zero
    /// predictor probes. Requires AllowReportedGaps; full encodes are unchanged.
    pub probe_palette: bool,
    /// Opt-in literal-family palette tools for both probes and full trials.
    /// Explicit caller palettes and automatic/cross-channel families are retained.
    pub literal_palette_tools: bool,
    /// Experimental ranking estimate using actual emitted tree/model bits.
    /// Requires AllowReportedGaps; full encoded bytes are unaffected.
    pub model_adjusted_probes: bool,
    /// Experimental quarter-rate MA sampling in probes only; complete trials
    /// retain their requested policy. Requires AllowReportedGaps.
    pub probe_sampling_quarter: bool,
    /// Experimental family-specific backend tools; not the ordinary source preset.
    pub routed_tools: bool,
    /// Collect actual accepted representations independently for every trial.
    pub record_representations: bool,
    pub resources: NativeResourcePolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeTrialKind {
    Full,
    Probe,
}

#[derive(Clone, Debug)]
pub struct NativeAdaptation {
    pub kind: NativeTrialKind,
    pub width: u32,
    pub height: u32,
    pub requested: CandidateParams,
    pub effective: CandidateParams,
    pub gaps: Vec<CapabilityGap>,
    pub state: AdaptationState,
    pub resources: NativeResourceLimits,
    pub representation: Option<std::sync::Arc<crate::modular::encode::receipt::ReceiptCollector>>,
    /// Invocation-local emitted header costs; bounded/failed trials may be partial.
    pub model_costs: Option<std::sync::Arc<crate::modular::encode::receipt_cost::ModelCostReceipt>>,
}

#[derive(Clone, Debug)]
pub struct NativeArtifact {
    /// Resolved options from this invocation. Modular fields apply only to the
    /// Modular implementation; FJXL uses fixed source effort 2 and the recorded
    /// thread/resource limits, preserving the original outer image metadata.
    pub options: EncodeOptions,
    pub implementation: NativeImplementation,
    pub effective_params: CandidateParams,
    pub gaps: Vec<CapabilityGap>,
    pub resources: NativeResourceLimits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeImplementation {
    Modular,
    FastLossless,
}

#[derive(Debug)]
pub struct NativeSearchResult {
    pub output: SearchResult<NativeArtifact>,
    /// One entry per callback, including all independent probe encodes.
    pub adaptations: Vec<NativeAdaptation>,
}

#[derive(Clone, Copy)]
pub(super) struct FrameContext<'a> {
    pub canvas: &'a Image,
    pub options: &'a crate::frame::FrameOptions,
    pub last: bool,
}

fn encode_full_trial(
    image: &Image,
    options: &EncodeOptions,
    beat_bytes: Option<usize>,
    pool: Option<&rayon::ThreadPool>,
    context: Option<FrameContext<'_>>,
) -> Result<TrialEncode<Vec<u8>>> {
    if let Some(context) = context {
        let frame = crate::frame::Frame {
            image,
            options: context.options.clone(),
        };
        let budget = SectionBudget::new(beat_bytes, options.max_output_bytes);
        let encode = || {
            crate::encode::encode_logical_frame_trial(
                context.canvas,
                &frame,
                options,
                context.last,
                budget.as_ref(),
            )
        };
        return match pool {
            Some(pool) => crate::execution::with_pool(pool, encode),
            None => crate::execution::serial(encode),
        };
    }
    match pool {
        Some(pool) => crate::encode::encode_trial_with_pool(image, options, beat_bytes, pool),
        None => crate::encode::encode_trial_serial(image, options, beat_bytes),
    }
}

pub(super) fn prepare(
    params: &CandidateParams,
    image: &Image,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    kind: NativeTrialKind,
) -> Result<(NativeArtifact, NativeAdaptation)> {
    let mut effective = params.clone();
    let reduced_probe_sampling = settings.probe_sampling_quarter
        && kind == NativeTrialKind::Probe
        && effective.learning_fraction >= 0.01;
    if reduced_probe_sampling {
        effective.learning_fraction *= 0.25;
    }
    let rewritten = settings.patches_off && effective.patches != Override::Off;
    if settings.patches_off {
        effective.patches = Override::Off;
    }
    let mut projection =
        project_native_with_resources(&effective, image, base, settings.resources)?;
    if reduced_probe_sampling {
        projection
            .gaps
            .push(CapabilityGap::ReducedProbeTrainingSamples);
    }
    if settings.routed_tools
        && let Some(options) = &mut projection.options
    {
        native_tools::apply(options, &effective);
    }
    if settings.literal_palette_tools
        && base.modular.palette_policy.is_none()
        && effective.predictor == Some(crate::modular::predictor::Predictor::Zero)
        && (1..=65536).contains(&effective.palette_colors)
        && let Some(policy) = projection
            .options
            .as_mut()
            .and_then(|options| options.modular.palette_policy.as_mut())
    {
        policy.force_palette = true;
        policy.tools.order = crate::modular::palette::PaletteOrder::FrequencyBandsMorton;
        policy.tools.mapping = crate::modular::palette::PaletteMapping::Flat;
        projection.gaps.push(CapabilityGap::LiteralPaletteTools);
    }
    if rewritten {
        projection.gaps.push(CapabilityGap::ForcedPatchesOff);
    }
    let model_adjusted = settings.model_adjusted_probes && kind == NativeTrialKind::Probe;
    if model_adjusted {
        projection.gaps.push(CapabilityGap::ModelScaledProbeScore);
    }
    let collect_model_costs = model_adjusted
        || base.modular.model_cost_receipt.is_some()
        || base
            .source_patch_reference
            .as_ref()
            .is_some_and(|reference| reference.modular.model_cost_receipt.is_some());
    let model_costs = projection.options.as_mut().and_then(|options| {
        let receipt = collect_model_costs.then(|| {
            std::sync::Arc::new(crate::modular::encode::receipt_cost::ModelCostReceipt::default())
        });
        options.modular.model_cost_receipt = receipt.clone();
        // Atlas resolution replaces ModularOptions, so bind its emitted costs
        // explicitly. The single-use representation collector stays separate.
        if let Some(reference) = &mut options.source_patch_reference {
            reference.modular.model_cost_receipt = receipt.clone();
        }
        receipt
    });
    if settings.probe_palette
        && kind == NativeTrialKind::Probe
        && effective.predictor == Some(crate::modular::predictor::Predictor::Zero)
        && (1..=65536).contains(&effective.palette_colors)
        && let Some(policy) = projection
            .options
            .as_mut()
            .and_then(|options| options.modular.palette_policy.as_mut())
    {
        policy.force_palette = true;
        projection.gaps.push(CapabilityGap::ForcedProbePalette);
    }
    let representation = projection.options.as_mut().and_then(|options| {
        if settings.record_representations {
            options.modular.representation_receipt = Some(std::sync::Arc::new(
                crate::modular::encode::receipt::ReceiptCollector::default(),
            ));
        }
        options.modular.representation_receipt.clone()
    });
    let adaptation = NativeAdaptation {
        kind,
        width: image.width(),
        height: image.height(),
        requested: params.clone(),
        effective: effective.clone(),
        gaps: projection.gaps.clone(),
        state: projection.state(),
        resources: projection.resources,
        representation,
        model_costs,
    };
    let gaps = projection.gaps.clone();
    let resources = projection.resources;
    let mut options = match settings.acceptance {
        NativeAcceptance::RequireExact => projection.exact_options()?,
        NativeAcceptance::AllowReportedGaps => projection
            .options
            .ok_or_else(|| Error::Encoding(format!("unavailable native candidate: {gaps:?}")))?,
    };
    if kind == NativeTrialKind::Probe && options.modular.source_channel_tree_final_only {
        options.modular.source_channel_tree = crate::modular::ma_train::ChannelTreeMode::Disabled;
        if let Some(reference) = &mut options.source_patch_reference {
            reference.modular.source_channel_tree =
                crate::modular::ma_train::ChannelTreeMode::Disabled;
        }
    }
    if crate::profile::enabled() {
        let modular = &options.modular;
        let dim = 128usize << options.group_size_shift;
        let groups =
            (image.width() as usize).div_ceil(dim) * (image.height() as usize).div_ceil(dim);
        let dc_groups = (image.width() as usize).div_ceil(dim * 8)
            * (image.height() as usize).div_ceil(dim * 8);
        let groupwise = modular.palette_policy.as_ref().is_some_and(|p| p.groupwise);
        crate::profile::note(
            "scheduler.native.policy",
            format_args!(
                "kind={kind:?} size={}x{} tier={} buffering={:?} predictor={:?} fixed_tree={:?} rct={:?} rct_candidates={} wp_candidates={} group_dim={dim} groups={groups} num_streams={} shared={} global_palette={} groupwise={} responsive={} patches={} threads={}",
                image.width(),
                image.height(),
                effective.speed_tier,
                effective.buffering,
                modular.predictor,
                modular.fixed_tree,
                modular.rct,
                modular.rct_candidates.unwrap_or(0),
                modular.weighted_candidates.unwrap_or(1),
                groups + 3 * dc_groups + 18,
                options.shared_tree && !groupwise,
                options.global_palette && !groupwise,
                groupwise,
                options.responsive,
                options.source_patches.is_some(),
                options.threads,
            ),
        );
        if let Some(sampling) = &modular.sampling {
            crate::profile::note(
                "scheduler.native.sampling",
                format_args!(
                    "raw_fraction={} effective_fraction={} properties_count={} properties_first16={:?} previous_channels={} quantile_bins={} node_threshold={} tree_mode={:?} predictor_override={:?} max_channel_size={} sample_limit={} max_tree_leaves={}",
                    effective.learning_fraction,
                    sampling.nb_repeats,
                    sampling.properties.len(),
                    &sampling.properties[..sampling.properties.len().min(16)],
                    modular.previous_channels,
                    sampling.max_property_values,
                    sampling.node_threshold,
                    sampling.tree_mode,
                    sampling.predictor_override,
                    sampling.max_channel_size,
                    modular.sample_limit,
                    modular.max_tree_leaves,
                ),
            );
        }
        crate::profile::note(
            "scheduler.native.entropy",
            format_args!(
                "lz_enabled={} source_lz={:?} careful={} bucket={} uint={:?} length_uint={:?} min_symbol={} clustering={:?} ans={:?} streaming={} source_tree={}",
                modular.lz77,
                modular.source_lz77.as_ref().map(|lz| lz.method),
                modular
                    .source_lz77
                    .as_ref()
                    .is_some_and(|lz| lz.careful_first_pass),
                modular
                    .source_lz77
                    .as_ref()
                    .is_some_and(|lz| lz.bucket_matcher),
                modular.entropy_options.uint_config,
                modular.entropy_options.lz77_length_config,
                modular.entropy_options.lz77_min_symbol,
                modular.entropy_options.clustering,
                modular.entropy_options.ans_histogram,
                modular.entropy_options.streaming_mode,
                modular.source_tree_entropy,
            ),
        );
    }
    Ok((
        NativeArtifact {
            options,
            implementation: NativeImplementation::Modular,
            effective_params: effective,
            gaps,
            resources,
        },
        adaptation,
    ))
}

/// Encode one explicit candidate with the native encoder. Does not run search.
/// Caller can request patches Off in CandidateParams without a rewrite gap.
pub fn encode_native_candidate(
    params: &CandidateParams,
    image: &Image,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
) -> Result<EncodedOutput<NativeArtifact>> {
    let (mut artifact, _) = prepare(params, image, base, settings, NativeTrialKind::Full)?;
    let bytes =
        if super::native_fast::eligible(image, &artifact.effective_params, &artifact.options) {
            let bytes = super::native_fast::encode(image, &artifact.options)?;
            artifact.implementation = NativeImplementation::FastLossless;
            bytes
        } else {
            crate::encode(image, &artifact.options)?
        };
    Ok(EncodedOutput { bytes, artifact })
}

/// Concrete full workflow for native integration/round-trip tests. It uses the
/// generic source driver; only this bridge knows EncodeOptions. Resource errors
/// propagate and are NEVER converted into TrialOutput::Bounded. The native
/// encoder can stop challengers on finalized-section lower bounds; baselines
/// and calibrations stay exact. The incumbent is retained without re-encoding.
/// All work shares one bounded pool;
/// independent candidate prefixes and batch internals use the null runner.
pub fn run_native_pixel_search(
    image: &Image,
    params: &CandidateParams,
    driver_options: &DriverOptions,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
) -> Result<NativeSearchResult> {
    if !(1..=256).contains(&base.threads) {
        return Err(Error::InvalidInput("threads must be 1..=256".into()));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(base.threads)
        .build()
        .map_err(|e| Error::Encoding(e.to_string()))?;
    run_native_search_with_pool(image, params, driver_options, base, settings, &pool, None)
}

pub(super) fn run_native_search_with_pool(
    image: &Image,
    params: &CandidateParams,
    driver_options: &DriverOptions,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    pool: &rayon::ThreadPool,
    context: Option<FrameContext<'_>>,
) -> Result<NativeSearchResult> {
    if driver_options.scout_pilot_consensus && !settings.model_adjusted_probes {
        return Err(Error::InvalidInput(
            "native scout pilot consensus requires model-adjusted probe ranking".into(),
        ));
    }
    let base = scout_execution_options(base, driver_options.adaptive_scout);
    let base = &base;
    let mut driver_options = driver_options.clone();
    let execution = WorkerExecution {
        limits: driver_options.workers,
        probe_workers: driver_options.probe_workers,
        cooperative: driver_options.cooperative_probes,
        arena_quiet: driver_options.arena_quiet,
        data_parallel: driver_options.data_parallel_probes,
        longest_first: driver_options.longest_first_probes,
    };
    execution.validate()?;
    // A no-gap projection is policy coverage, not an independently validated
    // source-equivalence certificate. Keep strict native runs Unverified.
    driver_options.backend_semantics = match settings.acceptance {
        NativeAcceptance::RequireExact => BackendSemantics::Unverified,
        NativeAcceptance::AllowReportedGaps => BackendSemantics::DeclaredProxy {
            differences: vec!["explicit native projection; inspect adaptations".into()],
        },
    };
    let adaptations = Rc::new(RefCell::new(Vec::new()));
    let full_records = Rc::clone(&adaptations);
    let probe_records = Rc::clone(&adaptations);
    let full_batch_records = Rc::clone(&adaptations);
    let probe_batch_records = Rc::clone(&adaptations);
    let prefix_records = Rc::clone(&adaptations);
    let overlap_records = Rc::clone(&adaptations);
    let parent_pixels = u64::from(image.width()) * u64::from(image.height());
    let mut output = run_pixel_search_with_batches(
        image,
        params,
        &driver_options,
        |params, image, budget: TrialBudget| {
            let _scope = crate::profile::scope("scheduler.full_encode");
            let start = crate::profile::enabled().then(std::time::Instant::now);
            let (mut artifact, record) =
                prepare(params, image, base, settings, NativeTrialKind::Full)?;
            artifact.options.max_output_bytes = artifact
                .options
                .max_output_bytes
                .min(budget.max_output_bytes);
            full_records.borrow_mut().push(record);
            let trial = encode_full_trial(
                image,
                &artifact.options,
                budget.beat_bytes,
                Some(pool),
                context,
            )?;
            if let Some(start) = start {
                crate::profile::note(
                    "scheduler.full_result",
                    format_args!(
                        "seconds={:.6} bytes={:?} params={params:?}",
                        start.elapsed().as_secs_f64(),
                        match &trial {
                            TrialEncode::Complete(bytes) => Some(bytes.len()),
                            TrialEncode::Bounded { .. } => None,
                        },
                    ),
                );
            }
            Ok(match trial {
                TrialEncode::Complete(bytes) => {
                    TrialOutput::Complete(EncodedOutput { bytes, artifact })
                }
                TrialEncode::Bounded { lower_bound } => TrialOutput::Bounded { lower_bound },
            })
        },
        |params, image| {
            let (artifact, record) =
                prepare(params, image, base, settings, NativeTrialKind::Probe)?;
            probe_records.borrow_mut().push(record);
            let bytes = crate::encode::encode_with_pool(image, &artifact.options, pool)?;
            native_tools::probe_score(
                bytes.len(),
                image,
                parent_pixels,
                &artifact.options,
                settings.model_adjusted_probes,
            )
        },
        BatchEncoders {
            pilot_overlap: Some(Box::new(|request| {
                native_overlap::run(
                    request,
                    base,
                    settings,
                    pool,
                    context,
                    execution,
                    &mut overlap_records.borrow_mut(),
                )
            })),
            full: Some(Box::new(|image, tasks| {
                let mut artifacts = Vec::with_capacity(tasks.len());
                for task in tasks {
                    let (mut artifact, record) =
                        prepare(task.params, image, base, settings, NativeTrialKind::Full)?;
                    artifact.options.max_output_bytes = artifact
                        .options
                        .max_output_bytes
                        .min(task.budget.max_output_bytes);
                    full_batch_records.borrow_mut().push(record);
                    artifacts.push(artifact);
                }
                // Source batches commit in plan order. Each independent job
                // uses the null runner; parallelism belongs to this outer pool.
                let cooperative = tasks.first().is_some_and(|t| t.cooperative);
                if tasks.iter().any(|t| t.cooperative != cooperative) {
                    return Err(Error::InvalidInput("mixed finalist worker modes".into()));
                }
                let scratch = u64::from(image.width())
                    .saturating_mul(u64::from(image.height()))
                    .saturating_mul(3 + image.metadata().extra_channels.len() as u64)
                    .saturating_mul(32)
                    .saturating_add(32 << 20);
                let plan = if cooperative
                    && tasks.len() == 1
                    && execution.limits.tasks >= pool.current_num_threads()
                {
                    // One finalist already owns the available pool. Avoid an
                    // extra arena/queue layer around every inner kernel task.
                    WorkerPlan::data_parallel(1)
                } else if cooperative {
                    WorkerPlan::cooperative(tasks.len(), execution.limits, scratch)?
                        .with_quiet(execution.arena_quiet)
                } else {
                    WorkerPlan::independent(tasks.len(), 64)?
                };
                let outcomes = run_plan(Some(pool), &plan, |i, runner| {
                    let _scope = crate::profile::scope("scheduler.full_encode");
                    let start = crate::profile::enabled().then(std::time::Instant::now);
                    let outcome = match runner {
                        TaskRunner::Null => encode_full_trial(
                            image,
                            &artifacts[i].options,
                            tasks[i].budget.beat_bytes,
                            None,
                            context,
                        ),
                        TaskRunner::Pool(pool) => encode_full_trial(
                            image,
                            &artifacts[i].options,
                            tasks[i].budget.beat_bytes,
                            Some(pool),
                            context,
                        ),
                    }?;
                    if let Some(start) = start {
                        crate::profile::note(
                            "scheduler.full_result",
                            format_args!(
                                "seconds={:.6} bytes={:?} batched=true params={:?}",
                                start.elapsed().as_secs_f64(),
                                match &outcome {
                                    TrialEncode::Complete(bytes) => Some(bytes.len()),
                                    TrialEncode::Bounded { .. } => None,
                                },
                                tasks[i].params,
                            ),
                        );
                    }
                    Ok(outcome)
                })?;
                Ok(outcomes
                    .into_iter()
                    .zip(artifacts)
                    .map(|(outcome, artifact)| match outcome {
                        TrialEncode::Complete(bytes) => {
                            TrialOutput::Complete(EncodedOutput { bytes, artifact })
                        }
                        TrialEncode::Bounded { lower_bound } => {
                            TrialOutput::Bounded { lower_bound }
                        }
                    })
                    .collect())
            })),
            probe: None,
            planned_probe: Some(Box::new(|tasks, kind| {
                let mut artifacts = Vec::with_capacity(tasks.len());
                for task in tasks {
                    let (artifact, record) = prepare(
                        task.params,
                        task.image,
                        base,
                        settings,
                        NativeTrialKind::Probe,
                    )?;
                    probe_batch_records.borrow_mut().push(record);
                    artifacts.push(artifact);
                }
                let plan = execution.plan_probes(tasks, kind, true)?;
                run_plan(Some(pool), &plan, |i, runner| {
                    let bytes = match runner {
                        TaskRunner::Null => {
                            crate::encode::encode_serial(tasks[i].image, &artifacts[i].options)?
                        }
                        TaskRunner::Pool(pool) => crate::encode::encode_with_pool(
                            tasks[i].image,
                            &artifacts[i].options,
                            pool,
                        )?,
                    };
                    native_tools::probe_score(
                        bytes.len(),
                        tasks[i].image,
                        parent_pixels,
                        &artifacts[i].options,
                        settings.model_adjusted_probes,
                    )
                })
            })),
            prefix: Some(Box::new(|screen| {
                // Prepare in canonical candidate/tile order on the caller.
                // Workers borrow only immutable artifacts and attempt flags;
                // the Rc records collection never crosses into the pool.
                let prepared = screen
                    .candidates
                    .iter()
                    .map(|candidate| {
                        screen
                            .order
                            .iter()
                            .map(|&tile| {
                                let (artifact, record) = prepare(
                                    &candidate.params,
                                    &screen.samples[tile],
                                    base,
                                    settings,
                                    NativeTrialKind::Probe,
                                )?;
                                Ok((artifact, record, AtomicBool::new(false)))
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let result = run_prefix_screen(
                    Some(pool),
                    execution.probe_workers.unwrap_or(execution.limits.tasks),
                    screen,
                    |candidate, tile| {
                        let position = screen
                            .order
                            .iter()
                            .position(|&t| t == tile)
                            .expect("prefix tile belongs to screen order");
                        let (artifact, _, attempted) = &prepared[candidate][position];
                        attempted.store(true, Ordering::Relaxed);
                        let bytes =
                            crate::encode::encode_serial(&screen.samples[tile], &artifact.options)?;
                        native_tools::probe_score(
                            bytes.len(),
                            &screen.samples[tile],
                            parent_pixels,
                            &artifact.options,
                            settings.model_adjusted_probes,
                        )
                    },
                );
                let mut records = prefix_records.borrow_mut();
                for (_, record, attempted) in prepared.into_iter().flatten() {
                    if attempted.into_inner() {
                        records.push(record);
                    }
                }
                result
            })),
            scan_pool: Some(pool),
        },
    )?;
    let adaptations = std::mem::take(&mut *adaptations.borrow_mut());
    if settings.acceptance == NativeAcceptance::AllowReportedGaps {
        let mut differences = Vec::new();
        for gap in adaptations.iter().flat_map(|a| &a.gaps) {
            let name = format!("{gap:?}");
            if !differences.contains(&name) {
                differences.push(name);
            }
        }
        output.report.backend_semantics = BackendSemantics::DeclaredProxy { differences };
    }
    Ok(NativeSearchResult {
        output,
        adaptations,
    })
}

fn scout_execution_options(base: &EncodeOptions, enabled: bool) -> EncodeOptions {
    let mut options = base.clone();
    if enabled {
        // These tools preserve the selected sample sites and predictor state;
        // row striping, which approximates learning history, stays disabled.
        options.modular.parallel_tokens = true;
        options.modular.source_gather.get_or_insert_with(|| {
            crate::modular::sample::GatherOptions {
                workers: base.threads.clamp(1, 64),
                sparse: true,
                ..Default::default()
            }
        });
        let lz =
            options
                .modular
                .source_lz77
                .get_or_insert_with(|| crate::tools::lz77::SourceConfig {
                    compact_steps: true,
                    ..Default::default()
                });
        lz.parallelism
            .get_or_insert(crate::tools::lz77::SourceParallelism {
                max_workers: base.threads.clamp(1, 64),
                fast_streams: true,
                ..Default::default()
            });
    }
    options
}

#[cfg(test)]
#[path = "native_route_tests.rs"]
mod route_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn channel_tree_final_only_distinguishes_probe_and_full_candidates() {
        use crate::modular::ma_train::ChannelTreeMode::{ChannelsAndRows, Disabled};
        let image = image();
        for final_only in [false, true] {
            let base = EncodeOptions {
                modular: crate::ModularOptions {
                    source_channel_tree: ChannelsAndRows,
                    source_channel_tree_final_only: final_only,
                    ..Default::default()
                },
                ..Default::default()
            };
            for speed_tier in [0, 1, 2, 7, 9] {
                for kind in [NativeTrialKind::Full, NativeTrialKind::Probe] {
                    let params = CandidateParams {
                        speed_tier,
                        ..Default::default()
                    };
                    let (artifact, _) = prepare(
                        &params,
                        &image,
                        &base,
                        NativeBackendOptions::default(),
                        kind,
                    )
                    .unwrap();
                    let expected =
                        if final_only && (speed_tier > 1 || kind == NativeTrialKind::Probe) {
                            Disabled
                        } else {
                            ChannelsAndRows
                        };
                    assert_eq!(artifact.options.modular.source_channel_tree, expected);
                    if let Some(reference) = &artifact.options.source_patch_reference {
                        assert_eq!(reference.modular.source_channel_tree, expected);
                    }
                }
            }
        }
    }
    use crate::{PixelData, PixelLayout};

    fn image() -> Image {
        Image::new(
            8,
            8,
            PixelLayout::Gray,
            PixelData::U8((0..64).map(|i| (i % 4) as u8).collect()),
        )
        .unwrap()
    }

    #[test]
    fn strict_bridge_rejects_an_explicit_source_policy_rewrite() {
        let result = prepare(
            &CandidateParams::default(),
            &image(),
            &EncodeOptions::default(),
            NativeBackendOptions {
                patches_off: true,
                ..Default::default()
            },
            NativeTrialKind::Full,
        );
        assert!(
            matches!(result, Err(Error::Encoding(message)) if message.contains("ForcedPatchesOff"))
        );
    }

    #[test]
    fn invalid_threads_are_rejected_before_pool_creation() {
        for threads in [0, 257] {
            let result = run_native_pixel_search(
                &image(),
                &CandidateParams::default(),
                &DriverOptions::default(),
                &EncodeOptions {
                    threads,
                    ..Default::default()
                },
                NativeBackendOptions::default(),
            );
            assert!(
                matches!(result, Err(Error::InvalidInput(message)) if message == "threads must be 1..=256")
            );
        }
    }

    #[test]
    fn native_initial_batches_preserve_output_across_pool_widths() {
        let image = image();
        let params = CandidateParams {
            speed_tier: 8,
            ..Default::default()
        };
        let options = DriverOptions::default();
        let run = |threads| {
            run_native_pixel_search(
                &image,
                &params,
                &options,
                &EncodeOptions {
                    threads,
                    ..Default::default()
                },
                NativeBackendOptions::default(),
            )
            .unwrap()
        };
        let serial = run(1);
        let parallel = run(3);
        assert_eq!(serial.output.bytes, parallel.output.bytes);
        assert_eq!(serial.output.params, parallel.output.params);
        assert_eq!(
            serial.output.report.full_encodes,
            parallel.output.report.full_encodes
        );
        assert_eq!(serial.adaptations.len(), parallel.adaptations.len());
        assert!(parallel.adaptations.iter().all(|a| a.gaps.is_empty()));
    }

    #[test]
    fn bridge_records_actual_resource_bounds_separately_from_capability_gaps() {
        let params = CandidateParams {
            patches: Override::Off,
            ..Default::default()
        };
        let settings = NativeBackendOptions {
            acceptance: NativeAcceptance::AllowReportedGaps,
            resources: NativeResourcePolicy {
                max_tree_leaves: Some(64),
            },
            ..Default::default()
        };
        let (artifact, record) = prepare(
            &params,
            &image(),
            &EncodeOptions::default(),
            settings,
            NativeTrialKind::Full,
        )
        .unwrap();
        assert_eq!(artifact.resources, record.resources);
        assert_eq!(artifact.resources.max_tree_leaves, 64);
        assert_eq!(artifact.options.modular.max_tree_leaves, 64);
        assert_eq!(
            artifact.resources.sample_limit,
            artifact.options.modular.sample_limit
        );
        assert!(!artifact.gaps.contains(&CapabilityGap::NativeTreeLeafLimit));
        assert!(!artifact.gaps.contains(&CapabilityGap::TreeEntropyPolicy));
        assert!(artifact.options.modular.source_tree_entropy);
        assert!(artifact.gaps.is_empty());
    }

    #[test]
    fn native_level_one_retains_options_and_reports_patch_rewrite() {
        let image = image();
        let result = run_native_pixel_search(
            &image,
            &CandidateParams {
                speed_tier: 9,
                ..Default::default()
            },
            &DriverOptions::default(),
            &EncodeOptions::default(),
            NativeBackendOptions {
                acceptance: NativeAcceptance::AllowReportedGaps,
                patches_off: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.adaptations.len(), 1);
        assert_eq!(result.output.report.full_encodes, 1);
        assert_eq!(result.output.report.probe_encodes, 0);
        assert_eq!(result.output.params.patches, Override::Default);
        assert_eq!(
            result.output.artifact.effective_params.patches,
            Override::Off
        );
        assert!(result.output.artifact.options.patches.is_none());
        assert!(result.output.artifact.options.source_patches.is_none());
        assert!(
            result
                .output
                .artifact
                .gaps
                .contains(&CapabilityGap::ForcedPatchesOff)
        );
        assert!(matches!(
            result.output.report.backend_semantics,
            BackendSemantics::DeclaredProxy { .. }
        ));
        assert_eq!(
            result.output.bytes,
            crate::encode(&image, &result.output.artifact.options).unwrap()
        );
    }

    #[test]
    fn strict_level_one_source_path_does_not_claim_external_validation() {
        let result = run_native_pixel_search(
            &image(),
            &CandidateParams {
                speed_tier: 9,
                ..Default::default()
            },
            &DriverOptions::default(),
            &EncodeOptions::default(),
            NativeBackendOptions::default(),
        )
        .unwrap();
        assert!(result.output.artifact.gaps.is_empty());
        assert!(result.output.artifact.options.modular.source_tree_entropy);
        assert!(result.output.artifact.options.source_patches.is_none());
        assert_eq!(
            result.output.report.backend_semantics,
            BackendSemantics::Unverified
        );
        assert_eq!(result.output.report.full_encodes, 1);
    }

    #[test]
    fn strict_source_search_keeps_automatic_patches_and_retains_the_actual_output() {
        let image = image();
        let result = run_native_pixel_search(
            &image,
            &CandidateParams {
                speed_tier: 7,
                ..Default::default()
            },
            &DriverOptions::default(),
            &EncodeOptions::default(),
            NativeBackendOptions::default(),
        )
        .unwrap();
        assert!(
            result
                .adaptations
                .iter()
                .all(|record| record.gaps.is_empty())
        );
        assert!(
            result
                .adaptations
                .iter()
                .any(|record| record.effective.patches == Override::Default
                    && record.effective.speed_tier < 6)
        );
        assert!(result.output.artifact.gaps.is_empty());
        assert_eq!(
            result.output.report.backend_semantics,
            BackendSemantics::Unverified
        );
        assert_eq!(
            result.output.bytes,
            crate::encode(&image, &result.output.artifact.options).unwrap()
        );
    }
}
