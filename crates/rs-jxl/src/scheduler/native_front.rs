//! Cheap pilot admission for original-group racing, using the same native
//! projection, cost adjustment and bounded worker dispatch as the old pilot.
use super::*;
use crate::scheduler::batch::pilot_handoff::{PilotFront, PilotRequest, run_front};

pub(crate) fn run_in_pool(
    request: &PilotRequest<'_>,
    base: &EncodeOptions,
    settings: NativeBackendOptions,
    pool: Option<&rayon::ThreadPool>,
    records: &mut Vec<NativeAdaptation>,
) -> Result<PilotFront> {
    let driver = request.options;
    let execution = WorkerExecution {
        limits: driver.workers,
        probe_workers: driver.probe_workers,
        cooperative: driver.cooperative_probes,
        arena_quiet: driver.arena_quiet,
        data_parallel: driver.data_parallel_probes,
        longest_first: driver.longest_first_probes,
    };
    execution.validate()?;
    Ok(run_front(
        request,
        Box::new(|tasks, kind| {
            let plan = execution.plan_probes(tasks, kind, true)?;
            let results = run_plan(pool, &plan, |i, runner| {
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
                        u64::from(request.image.width()) * u64::from(request.image.height()),
                        &artifact.options,
                        settings.model_adjusted_probes,
                    )
                })();
                // Drain all submitted cells, including a failing cell's peers.
                Ok((result, record))
            })?;
            let mut scores = Vec::with_capacity(results.len());
            let mut error = None;
            for (result, record) in results {
                records.extend(record);
                match result {
                    Ok(score) => scores.push(score),
                    Err(e) if error.is_none() => error = Some(e),
                    Err(_) => {}
                }
            }
            match error {
                Some(error) => Err(error),
                None => Ok(scores),
            }
        }),
    ))
}
