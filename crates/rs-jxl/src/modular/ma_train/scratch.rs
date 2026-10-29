// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Invocation-owned property workspaces and exclusive reusable scratch leases.
//! Scratch retention does not impose a histogram-cache or tree-search budget.

use super::{
    AtomicUsize, BinHistogram, Cost, EventScratch, HistogramShards, HistogramSplit, Mutex,
    Ordering, ScoreScratch,
};

#[derive(Default)]
pub(super) struct PropertyScratch {
    pub(super) dominated_predictors: usize,
    pub(super) splits: Vec<HistogramSplit>,
    pub(super) score_mask: Vec<u8>,
    pub(super) saved_histograms: Vec<BinHistogram>,
    pub(super) property_bins: Vec<usize>,
    pub(super) shards: HistogramShards,
    pub(super) scoring: ScoreScratch,
    pub(super) events: EventScratch,
    #[cfg(test)]
    pub(super) sharded_predictors: Vec<usize>,
    #[cfg(test)]
    pub(super) sparse_predictors: Vec<usize>,
    #[cfg(test)]
    pub(super) packed_bins: usize,
    pub(super) used: Vec<usize>,
    pub(super) weights: Vec<u32>,
    pub(super) below: Vec<Cost>,
    pub(super) above: Vec<Cost>,
    pub(super) increase: Vec<u32>,
    pub(super) extra: Vec<u64>,
    pub(super) paired_increase: Vec<u32>,
    pub(super) paired_extra: Vec<u64>,
    pub(super) counts_above: Vec<u32>,
    pub(super) counts_below: Vec<u32>,
    #[cfg(test)]
    pub(super) scored_predictors: Vec<usize>,
}

/// Source PropertyWorkspace lifetime, without indexing scratch by Rayon thread
/// ID: nested ready-child jobs can suspend a worker while another job runs on it.
/// A lease owns its buffers exclusively and holds no lock during the sweep.
pub(super) struct ScratchPool {
    pub(super) idle: Mutex<Vec<PropertyScratch>>,
    pub(super) retained_limit: usize,
    pub(super) created: AtomicUsize,
    pub(super) reused: AtomicUsize,
    // Reset at learner entry, set only by actual parent/property dispatches.
    pub(super) dispatch_slots: AtomicUsize,
}

impl ScratchPool {
    pub(super) fn new(retained_limit: usize) -> Self {
        Self {
            idle: Mutex::new(Vec::with_capacity(retained_limit)),
            retained_limit,
            created: AtomicUsize::new(0),
            reused: AtomicUsize::new(0),
            dispatch_slots: AtomicUsize::new(0),
        }
    }

    pub(super) fn acquire(&self) -> ScratchLease<'_> {
        let available = self.idle.lock().expect("MA scratch pool poisoned").pop();
        let scratch = if let Some(scratch) = available {
            self.reused.fetch_add(1, Ordering::Relaxed);
            scratch
        } else {
            self.created.fetch_add(1, Ordering::Relaxed);
            PropertyScratch::default()
        };
        ScratchLease {
            pool: self,
            scratch,
        }
    }
}

pub(super) struct ScratchLease<'a> {
    pool: &'a ScratchPool,
    pub(super) scratch: PropertyScratch,
}

impl Drop for ScratchLease<'_> {
    fn drop(&mut self) {
        let mut idle = self.pool.idle.lock().expect("MA scratch pool poisoned");
        if idle.len() < self.pool.retained_limit {
            idle.push(std::mem::take(&mut self.scratch));
        }
    }
}
