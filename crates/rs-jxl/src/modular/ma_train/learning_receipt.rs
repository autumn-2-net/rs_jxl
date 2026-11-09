//! Optional invocation evidence for the PR5007 experiment, outside pixel loops.

use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Default)]
pub struct TreeLearningReceipt {
    pub(super) learners: AtomicUsize,
    pub(super) records: AtomicUsize,
    pub(super) cost_splits: AtomicUsize,
    pub(super) symmetric_splits: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TreeLearningSnapshot {
    pub learners: usize,
    pub records: usize,
    pub cost_splits: usize,
    pub symmetric_splits: usize,
}

impl TreeLearningReceipt {
    /// Read after all encoder workers have completed.
    pub fn snapshot(&self) -> TreeLearningSnapshot {
        TreeLearningSnapshot {
            learners: self.learners.load(Ordering::Relaxed),
            records: self.records.load(Ordering::Relaxed),
            cost_splits: self.cost_splits.load(Ordering::Relaxed),
            symmetric_splits: self.symmetric_splits.load(Ordering::Relaxed),
        }
    }
}
