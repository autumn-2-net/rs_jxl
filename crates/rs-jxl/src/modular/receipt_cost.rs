//! Emitted MA tree/model costs for one caller-owned encode invocation.
//!
//! These are measured wire bits, not an entropy estimate or an encoded-byte
//! bound. They exclude transform/WP headers, palette and pixel tokens, patch
//! dictionaries, framing and padding. A tree's own entropy coding is part of
//! tree_bits, not counted again as an image entropy-model header.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Install a fresh Arc in ModularOptions AFTER cloning base options for each
/// invocation. Cloned options within that invocation aggregate shared/local
/// writers, including groupwise streams and any emitted reference frames.
/// Read only after successful completion and joining all encoding work. A
/// bounded/failed encode may have partial counts for discarded sections; reuse
/// accumulates rather than resets. Counts do not imply fixed whole-image costs.
#[derive(Debug, Default)]
pub struct ModelCostReceipt {
    tree_bits: AtomicUsize,
    entropy_header_bits: AtomicUsize,
}

impl ModelCostReceipt {
    pub fn bits(&self) -> usize {
        self.tree_bits() + self.entropy_header_bits()
    }

    pub fn tree_bits(&self) -> usize {
        self.tree_bits.load(Ordering::Relaxed)
    }

    pub fn entropy_header_bits(&self) -> usize {
        self.entropy_header_bits.load(Ordering::Relaxed)
    }

    pub(super) fn record(&self, tree_bits: usize, entropy_header_bits: usize) {
        self.tree_bits.fetch_add(tree_bits, Ordering::Relaxed);
        self.entropy_header_bits
            .fetch_add(entropy_header_bits, Ordering::Relaxed);
    }
}
