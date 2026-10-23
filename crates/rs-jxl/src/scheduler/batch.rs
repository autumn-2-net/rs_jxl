//! Default source worker dispatch. Candidate/region order determines results,
//! never worker completion order. Only the supplied Rayon pool is entered.

use super::worker_pool::ProbeBatchKind;
use super::worker_pool::{WorkerPlan, run_plan};
use super::{
    params::CandidateParams,
    search::{TrialBudget, TrialOutput},
};
use crate::{Image, Result};

#[derive(Clone, Copy)]
pub struct ProbeTask<'a> {
    pub params: &'a CandidateParams,
    pub image: &'a Image,
}

#[derive(Clone, Copy)]
pub struct FullTask<'a> {
    pub params: &'a CandidateParams,
    pub budget: TrialBudget,
    /// Optional finalist arena: roots may borrow the same pool for inner work.
    pub cooperative: bool,
}

/// A callback completes every requested cell, returning values in input order.
/// Any failure aborts this batch: no partial score is a completed candidate.
pub type ProbeBatchEncoder<'a> = Box<dyn FnMut(&[ProbeTask<'_>]) -> Result<Vec<usize>> + 'a>;
pub type PlannedProbeBatchEncoder<'a> =
    Box<dyn FnMut(&[ProbeTask<'_>], ProbeBatchKind) -> Result<Vec<usize>> + 'a>;
/// Whole candidate screen/shortlist tasks, with actual-attempt accounting.
/// Implement using `prefix_screen::run_prefix_screen`; no partial scores escape.
pub type PrefixBatchEncoder<'a> = Box<
    dyn FnMut(&super::prefix_screen::PrefixScreen<'_>) -> Result<super::prefix_screen::PrefixOutput>
        + 'a,
>;
pub type FullBatchEncoder<'a, A> =
    Box<dyn FnMut(&Image, &[FullTask<'_>]) -> Result<Vec<TrialOutput<A>>> + 'a>;

pub struct BatchEncoders<'a, A> {
    pub full: Option<FullBatchEncoder<'a, A>>,
    pub probe: Option<ProbeBatchEncoder<'a>>,
    /// Tagged cache work; mutually exclusive with the legacy probe callback.
    pub planned_probe: Option<PlannedProbeBatchEncoder<'a>>,
    /// Optional asynchronous screen/shortlist; legacy FnMut callbacks stay valid.
    pub prefix: Option<PrefixBatchEncoder<'a>>,
    /// Coverage remains opt-in; this only supplies its ScanPlane runner.
    pub scan_pool: Option<&'a rayon::ThreadPool>,
}

impl<A> Default for BatchEncoders<'_, A> {
    fn default() -> Self {
        Self {
            full: None,
            probe: None,
            planned_probe: None,
            prefix: None,
            scan_pool: None,
        }
    }
}

/// enc_lossless_search.cc: small full-frame candidates share the caller runner.
/// This is the source concurrency heuristic, not a per-encode allocation cap.
pub fn initial_concurrency(image: &Image) -> usize {
    let pixels = u64::from(image.width()) * u64::from(image.height());
    let scratch = pixels
        .saturating_mul(5 + image.metadata().extra_channels.len() as u64)
        .saturating_mul(32);
    if pixels <= 262144 && scratch != 0 {
        ((256u64 << 20) / scratch).clamp(1, 4) as usize
    } else {
        1
    }
}

/// enc_lossless_probe.cc::RunProbeTasks, default worker slots = 8. A shared
/// root queue prevents a slow job from fixing an entire worker's tail. Every
/// callback retains the null-runner policy, including nested core algorithms.
pub fn run_ordered<T: Send + Sync>(
    pool: &rayon::ThreadPool,
    count: usize,
    worker_slots: usize,
    task: impl Fn(usize) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    run_plan(
        Some(pool),
        &WorkerPlan::independent(count, worker_slots)?,
        |index, _runner| task(index),
    )
}
