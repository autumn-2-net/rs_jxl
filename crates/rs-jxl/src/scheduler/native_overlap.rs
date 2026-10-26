//! Small-input overlap on the caller's arena, with a full scratch reservation.
use super::*;
use crate::scheduler::{
    batch::pilot_handoff::{PilotFront, PilotHandoff, PilotRequest, run_front, screen_options},
    regions::screen_regions,
    worker_pool::{PROBE_SCRATCH_BYTES, WorkerLimits},
};

struct Reservation {
    outer: WorkerLimits,
    probes: WorkerExecution,
}

fn reservation(
    request: &PilotRequest<'_>,
    execution: WorkerExecution,
    pool_threads: usize,
) -> Result<Option<Reservation>> {
    if let Some(scan) = request.scan {
        let continuous = scan.likely_continuous_tone();
        let content_allowed = continuous;
        if crate::profile::enabled() {
            crate::profile::note(
                "scheduler.overlap_route",
                format_args!(
                    "colors={} flat={} sampled={} continuous={continuous} literal={} content_allowed={content_allowed}",
                    scan.sampled_colors,
                    scan.sampled_flat,
                    scan.sampled_pixels,
                    scan.literal_candidate,
                ),
            );
        }
        if !content_allowed {
            return Ok(None);
        }
    }
    let pixels = u64::from(request.image.width()) * u64::from(request.image.height());
    let workers = execution.limits.tasks.min(pool_threads).min(16);
    if pixels > 1 << 19 || workers < 2 || crate::execution::is_serial() {
        return Ok(None);
    }
    let scratch = execution.limits.scratch_bytes.min(256 << 20);
    let full = pixels
        .saturating_mul(3 + request.image.metadata().extra_channels.len() as u64)
        .saturating_mul(32)
        .saturating_add(32 << 20);
    let (fine, coarse) = screen_options(request);
    let sample_pixels = |options: &crate::scheduler::probe::ScreenOptions| -> Result<u64> {
        screen_regions(request.image, request.scan, &options.regions)?
            .iter()
            .try_fold(0u64, |sum, region| {
                sum.checked_add(region.pixels()? as u64)
                    .ok_or(Error::Limit("resident probe pixels"))
            })
    };
    let mut resident = sample_pixels(&fine)?;
    if request.level == 9 && request.options.scout_staged_probes {
        resident = resident.max(sample_pixels(&coarse)?);
    }
    if resident > request.probe_limits.max_resident_sample_pixels as u64 {
        return Err(Error::Limit("resident probe pixels"));
    }
    // Samples use at most four bytes/channel. Coarse samples are freed before
    // fine allocation; budget the larger live set, not both cumulative sets.
    let samples = resident
        .saturating_mul(request.image.channel_count() as u64)
        .saturating_mul(4);
    let Some(remaining) = scratch.checked_sub(full.saturating_add(samples)) else {
        return Ok(None);
    };
    let roots = (remaining / PROBE_SCRATCH_BYTES).min((workers - 1) as u64) as usize;
    if roots == 0 {
        return Ok(None);
    }
    let mut probes = execution;
    probes.limits = WorkerLimits {
        tasks: roots,
        scratch_bytes: remaining,
    };
    probes.probe_workers = Some(execution.probe_workers.unwrap_or(workers).min(roots));
    Ok(Some(Reservation {
        outer: WorkerLimits {
            tasks: workers,
            scratch_bytes: scratch,
        },
        probes,
    }))
}

enum RootOutput {
    Full(Result<TrialOutput<NativeArtifact>>, Vec<NativeAdaptation>),
    Front(PilotFront, Vec<NativeAdaptation>),
}

pub(super) fn run(
    request: &PilotRequest<'_>,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    pool: &rayon::ThreadPool,
    context: Option<FrameContext<'_>>,
    execution: WorkerExecution,
    records: &mut Vec<NativeAdaptation>,
) -> Result<Option<PilotHandoff<NativeArtifact>>> {
    let Some(reserved) = reservation(request, execution, pool.current_num_threads())? else {
        return Ok(None);
    };
    if request.budget.beat_bytes.is_some() {
        return Err(Error::InvalidInput(
            "overlap automatic must complete".into(),
        ));
    }
    // Memory is reserved above for heterogeneous roots. The outer plan only
    // caps total physical participation; child probe roots use the remainder.
    let plan = WorkerPlan::cooperative(2, reserved.outer, 1)?.with_quiet(execution.arena_quiet);
    let roots = run_plan(Some(pool), &plan, |index, _| {
        let mut records = Vec::new();
        if index == 0 {
            let _scope = crate::profile::scope("scheduler.full_encode");
            let start = crate::profile::enabled().then(std::time::Instant::now);
            let result = (|| {
                let (mut artifact, record) = prepare(
                    request.automatic,
                    request.image,
                    base,
                    settings,
                    NativeTrialKind::Full,
                )?;
                records.push(record);
                artifact.options.max_output_bytes = artifact
                    .options
                    .max_output_bytes
                    .min(request.budget.max_output_bytes);
                let trial =
                    encode_full_trial(request.image, &artifact.options, None, Some(pool), context)?;
                Ok(match trial {
                    TrialEncode::Complete(bytes) => {
                        TrialOutput::Complete(EncodedOutput { bytes, artifact })
                    }
                    TrialEncode::Bounded { lower_bound } => TrialOutput::Bounded { lower_bound },
                })
            })();
            if let Some(start) = start {
                crate::profile::note(
                    "scheduler.full_result",
                    format_args!(
                        "seconds={:.6} bytes={:?} overlap=true params={:?}",
                        start.elapsed().as_secs_f64(),
                        match &result {
                            Ok(TrialOutput::Complete(output)) => Some(output.bytes.len()),
                            _ => None,
                        },
                        request.automatic,
                    ),
                );
            }
            // Encode errors are payloads, so neither outer root is cancelled.
            Ok(RootOutput::Full(result, records))
        } else {
            let front = run_front(
                request,
                Box::new(|tasks, kind| {
                    let plan = reserved.probes.plan_probes(tasks, kind, true)?;
                    let results = run_plan(Some(pool), &plan, |i, runner| {
                        let mut record = None;
                        let result = (|| {
                            let (artifact, adaptation) = prepare(
                                tasks[i].params,
                                tasks[i].image,
                                base,
                                settings,
                                NativeTrialKind::Probe,
                            )?;
                            record = Some(adaptation);
                            let bytes = match runner {
                                TaskRunner::Null => {
                                    crate::encode::encode_serial(tasks[i].image, &artifact.options)?
                                }
                                TaskRunner::Pool(pool) => crate::encode::encode_with_pool(
                                    tasks[i].image,
                                    &artifact.options,
                                    pool,
                                )?,
                            };
                            native_tools::probe_score(
                                bytes.len(),
                                tasks[i].image,
                                u64::from(request.image.width())
                                    * u64::from(request.image.height()),
                                &artifact.options,
                                settings.model_adjusted_probes,
                            )
                        })();
                        // Every submitted cell is attempted and drained even if an
                        // earlier cell failed; run_front charges precisely this set.
                        Ok((result, record))
                    })?;
                    let mut scores = Vec::with_capacity(results.len());
                    for (result, record) in results {
                        records.extend(record);
                        scores.push(result);
                    }
                    scores.into_iter().collect()
                }),
            );
            Ok(RootOutput::Front(front, records))
        }
    })?;
    let mut roots = roots.into_iter();
    let Some(RootOutput::Full(automatic, full_records)) = roots.next() else {
        return Err(Error::Encoding("missing overlap automatic handoff".into()));
    };
    let Some(RootOutput::Front(front, front_records)) = roots.next() else {
        return Err(Error::Encoding("missing overlap front handoff".into()));
    };
    records.extend(full_records);
    records.extend(front_records);
    Ok(Some(PilotHandoff { automatic, front }))
}

#[cfg(test)]
#[path = "native_overlap_tests.rs"]
mod tests;
