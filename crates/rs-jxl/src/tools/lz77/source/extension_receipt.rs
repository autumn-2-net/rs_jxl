//! Opt-in experiment counters. Flush once per matcher, not once per match.

#[cfg(any(feature = "experimental-lz77", test))]
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct MatchExtensionReceipt {
    fast_matchers: AtomicU64,
    chain_matchers: AtomicU64,
    extensions: AtomicU64,
    batch_groups: AtomicU64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MatchExtensionSnapshot {
    pub fast_matchers: u64,
    pub chain_matchers: u64,
    pub extensions: u64,
    pub batch_groups: u64,
}

impl MatchExtensionReceipt {
    pub fn snapshot(&self) -> MatchExtensionSnapshot {
        MatchExtensionSnapshot {
            fast_matchers: self.fast_matchers.load(Ordering::Relaxed),
            chain_matchers: self.chain_matchers.load(Ordering::Relaxed),
            extensions: self.extensions.load(Ordering::Relaxed),
            batch_groups: self.batch_groups.load(Ordering::Relaxed),
        }
    }
}

#[cfg(any(feature = "experimental-lz77", test))]
#[derive(Default)]
pub(crate) struct LocalReceipt {
    target: Option<Arc<MatchExtensionReceipt>>,
    chain: bool,
    pub(crate) extensions: u64,
    pub(crate) batch_groups: u64,
}

#[cfg(any(feature = "experimental-lz77", test))]
impl LocalReceipt {
    #[inline]
    pub(crate) fn extend(&mut self, count: u64, batched: bool) {
        if self.target.is_some() {
            self.extensions += count;
            self.batch_groups += u64::from(batched);
        }
    }

    pub(crate) fn new(target: Option<Arc<MatchExtensionReceipt>>, chain: bool) -> Self {
        Self {
            target,
            chain,
            ..Self::default()
        }
    }
}

#[cfg(any(feature = "experimental-lz77", test))]
impl Drop for LocalReceipt {
    fn drop(&mut self) {
        if let Some(target) = &self.target {
            let matchers = if self.chain {
                &target.chain_matchers
            } else {
                &target.fast_matchers
            };
            matchers.fetch_add(1, Ordering::Relaxed);
            target
                .extensions
                .fetch_add(self.extensions, Ordering::Relaxed);
            target
                .batch_groups
                .fetch_add(self.batch_groups, Ordering::Relaxed);
        }
    }
}
