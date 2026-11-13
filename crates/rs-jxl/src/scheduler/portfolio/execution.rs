//! Portfolio-owned caches and bounded task dispatch, separate from ranking.
use std::{sync::Arc, time::Instant};

use super::{Candidate, Options, Report};
use crate::modular::candidate_cache::{CacheScope, ReuseStats};
use crate::scheduler::worker_pool::{PROBE_SCRATCH_BYTES, TaskRunner, WorkerPlan, run_plan};
use crate::{EncodeOptions, Image, Result};

pub(super) fn base(options: &Options, cache: Option<&CacheScope<'_>>) -> EncodeOptions {
    let mut base = options.base();
    base.modular.candidate_cache = cache.map(|c| Arc::clone(&c.cache));
    base
}

pub(super) fn add_stats(total: &mut ReuseStats, part: ReuseStats) {
    total.hits += part.hits;
    total.misses += part.misses;
    total.duplicate_builds += part.duplicate_builds;
    total.bypasses += part.bypasses;
    total.entries += part.entries;
    total.retained_bytes += part.retained_bytes;
    // Sum of scope peaks is a conservative retained-buffer tally, not RSS.
    total.peak_bytes += part.peak_bytes;
}

struct Timed {
    region: usize,
    trial: usize,
    pixels: u64,
    seconds: f64,
    result: Result<usize>,
}

pub(super) fn probes(
    samples: &[Image],
    caches: &[CacheScope<'_>],
    tasks: &[(usize, usize)],
    options: &Options,
    report: &mut Report,
    pool: Option<&rayon::ThreadPool>,
) -> Result<()> {
    if tasks.is_empty() {
        return Ok(());
    }
    let slots = options
        .threads
        .min(64)
        .min((1 << 30) / PROBE_SCRATCH_BYTES as usize);
    let plan = if options.parallel_probes {
        WorkerPlan::independent(tasks.len(), slots)?
    } else {
        WorkerPlan::data_parallel(tasks.len())
    };
    report.probe_workers = report
        .probe_workers
        .max(if pool.is_some() && options.parallel_probes {
            plan.worker_slots()
        } else {
            1
        });
    let candidates: Vec<Candidate> = report.candidates.iter().map(|t| t.candidate).collect();
    let results = run_plan(pool, &plan, |i, runner| {
        let (region, trial) = tasks[i];
        let image = &samples[region];
        let start = Instant::now();
        let result = candidates[trial]
            .build(image, &base(options, caches.get(region)), options.effort)
            .and_then(|configured| match runner {
                TaskRunner::Null => crate::encode::encode_serial(image, &configured),
                TaskRunner::Pool(pool) => crate::encode::encode_with_pool(image, &configured, pool),
            })
            .map(|bytes| bytes.len());
        // Drain the batch even if an individual candidate fails; record its
        // failure instead of losing the remaining scheduled measurements.
        Ok(Timed {
            region,
            trial,
            pixels: u64::from(image.width()) * u64::from(image.height()),
            seconds: start.elapsed().as_secs_f64(),
            result,
        })
    })?;
    for result in results {
        report.probe_encodes += 1;
        report.probe_pixels += result.pixels;
        let record = &mut report.candidates[result.trial];
        record.probe_seconds += result.seconds;
        match result.result {
            Ok(bytes) => record.probe_bytes[result.region] = Some(bytes),
            Err(error) => {
                record.decision = "probe-error";
                record.reason = "crop failure is recorded, not interpreted as a size score".into();
                record.error = Some(error.to_string());
            }
        }
    }
    Ok(())
}
