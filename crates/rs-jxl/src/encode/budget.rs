// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! enc_lossless_search.h::LosslessTrialBudget and enc_frame.cc::EncodeGroups.
//! Only finalized, disjoint output sections enter this candidate-local tally.

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};

use crate::{Error, Result, execution};

#[derive(Debug)]
pub(crate) enum TrialEncode<T> {
    Complete(T),
    Bounded { lower_bound: usize },
}

impl<T> TrialEncode<T> {
    pub(crate) fn into_complete(self) -> Result<T> {
        match self {
            Self::Complete(value) => Ok(value),
            Self::Bounded { .. } => Err(Error::Encoding(
                "unbounded encoder returned an incomplete output".into(),
            )),
        }
    }
}

macro_rules! complete_or_return {
    ($result:expr) => {
        match $result? {
            $crate::encode::budget::TrialEncode::Complete(value) => value,
            $crate::encode::budget::TrialEncode::Bounded { lower_bound } => {
                return Ok($crate::encode::budget::TrialEncode::Bounded { lower_bound });
            }
        }
    };
}
pub(crate) use complete_or_return;

pub(crate) struct SectionBudget {
    beat_bytes: usize,
    hard_limit: usize,
    bytes: Mutex<usize>,
    overhead: Mutex<usize>,
    reached: AtomicBool,
    #[cfg(test)]
    pub(crate) sections_started: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    pub(crate) areas_started: std::sync::atomic::AtomicUsize,
}

impl SectionBudget {
    pub(crate) fn new(beat_bytes: Option<usize>, hard_limit: usize) -> Option<Self> {
        let beat_bytes = beat_bytes.filter(|&n| n != 0 && n < hard_limit)?;
        Some(Self {
            beat_bytes,
            hard_limit,
            bytes: Mutex::new(0),
            overhead: Mutex::new(0),
            reached: AtomicBool::new(false),
            #[cfg(test)]
            sections_started: Default::default(),
            #[cfg(test)]
            areas_started: Default::default(),
        })
    }

    pub(crate) fn reached(&self) -> Option<usize> {
        self.reached
            .load(Ordering::Relaxed)
            .then_some(self.beat_bytes)
    }

    pub(crate) fn section_bytes(&self) -> usize {
        *self.bytes.lock().expect("section budget poisoned")
    }

    // Called only between frame dispatches, never concurrently with credit.
    // Header bytes and uninstrumented/collapsed frames constrain resources but
    // do not turn source-excluded frames into bounded trials.
    pub(crate) fn retain_overhead(&self, bytes: usize) -> Result<()> {
        let sections = self.bytes.lock().expect("section budget poisoned");
        let mut overhead = self.overhead.lock().expect("section overhead poisoned");
        let next = overhead
            .checked_add(bytes)
            .ok_or(Error::Limit("output size overflow"))?;
        if sections
            .checked_add(next)
            .is_none_or(|n| n > self.hard_limit)
        {
            return Err(Error::Limit("encoded output bytes"));
        }
        *overhead = next;
        Ok(())
    }

    fn credit(&self, bytes: usize) -> Result<()> {
        let mut total = self.bytes.lock().expect("section budget poisoned");
        // Do not saturate at the cutoff: in-flight sections can still expose a
        // genuine hard-limit failure, which must outrank cancellation.
        let next = total
            .checked_add(bytes)
            .filter(|&n| n <= self.hard_limit)
            .ok_or(Error::Limit("frame section bytes"))?;
        let overhead = self.overhead.lock().expect("section overhead poisoned");
        if next
            .checked_add(*overhead)
            .is_none_or(|n| n > self.hard_limit)
        {
            return Err(Error::Limit("encoded output bytes"));
        }
        *total = next;
        if *total >= self.beat_bytes {
            self.reached.store(true, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Called exactly once per final global/DC/AC stream, never for cost searches
/// or scratch writers. The complete payload would be retained unchanged in the
/// TOC; neither later concatenation nor container assembly can shorten it.
pub(crate) fn section(
    budget: Option<&SectionBudget>,
    encode: impl FnOnce() -> Result<Vec<u8>>,
) -> Result<TrialEncode<Vec<u8>>> {
    if let Some(lower_bound) = budget.and_then(SectionBudget::reached) {
        return Ok(TrialEncode::Bounded { lower_bound });
    }
    #[cfg(test)]
    if let Some(budget) = budget {
        budget.sections_started.fetch_add(1, Ordering::Relaxed);
    }
    let bytes = encode()?;
    if let Some(budget) = budget {
        budget.credit(bytes.len())?;
        if let Some(lower_bound) = budget.reached() {
            return Ok(TrialEncode::Bounded { lower_bound });
        }
    }
    Ok(TrialEncode::Complete(bytes))
}

pub(crate) fn sections<T: Sync>(
    budget: Option<&SectionBudget>,
    items: &[T],
    encode: impl Fn(usize, &T) -> Result<Vec<u8>> + Send + Sync,
) -> Result<TrialEncode<Vec<Vec<u8>>>> {
    if budget.is_none() {
        return execution::try_map(items, encode).map(TrialEncode::Complete);
    }
    // Cancellation is a successful, typed outcome, never an Error sentinel.
    // try_map drains active tasks and propagates their errors before this fold.
    let encoded = execution::try_map(items, |i, item| section(budget, || encode(i, item)))?;
    let mut complete = Vec::with_capacity(encoded.len());
    for output in encoded {
        match output {
            TrialEncode::Complete(bytes) => complete.push(bytes),
            TrialEncode::Bounded { lower_bound } => {
                return Ok(TrialEncode::Bounded { lower_bound });
            }
        }
    }
    Ok(TrialEncode::Complete(complete))
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
