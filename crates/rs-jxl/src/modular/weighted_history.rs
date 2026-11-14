//! Invocation-local reuse of the exact whole-channel gather history.
//! The borrow keeps transformed planes immutable until tokenization finishes.

use std::sync::OnceLock;

use super::{encode::ModularOptions, group::Channel, predictor::Predictor, weighted};

pub(super) const CACHE_BYTES: usize = 8 << 20;

pub(super) struct History<'a> {
    planes: &'a [Channel],
    header: weighted::Header,
    channels: Vec<OnceLock<Vec<[i32; 2]>>>,
}

impl<'a> History<'a> {
    pub(super) fn new(
        planes: &'a [Channel],
        header: weighted::Header,
        options: &ModularOptions,
    ) -> Option<Self> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var("RS_JXL_WP_REUSE").as_deref() != Ok("0")) {
            return None;
        }
        let policy = options.sampling.as_ref()?;
        if policy.nb_repeats == 0.0
            || options.fixed_tree.is_some()
            || options.custom_tree.is_some()
            || options.tree_groups.is_some()
            || options.block_context.is_some()
            || options.fast_context.is_some()
            || options.refinement.enabled()
            || options
                .source_gather
                .as_ref()
                .is_some_and(|g| g.stripes != 1)
        {
            return None;
        }
        let predictors = super::training::source_predictors(options.predictor, policy).ok()?;
        if !predictors.contains(&Predictor::Weighted) && !policy.properties.contains(&15) {
            return None;
        }
        let samples = planes
            .iter()
            .try_fold(0usize, |sum, c| sum.checked_add(c.samples.len()))?;
        // Cache budget only: larger groups retain the uncached encoder unchanged.
        if samples == 0 || samples > CACHE_BYTES / std::mem::size_of::<[i32; 2]>() {
            return None;
        }
        let mut channels = Vec::new();
        channels.try_reserve_exact(planes.len()).ok()?;
        channels.resize_with(planes.len(), OnceLock::new);
        Some(Self {
            planes,
            header,
            channels,
        })
    }

    pub(super) fn matches(&self, planes: &[Channel], header: &weighted::Header) -> bool {
        std::ptr::eq(self.planes, planes) && self.header == *header
    }

    pub(super) fn bytes(&self) -> usize {
        self.planes.iter().map(|c| c.samples.len()).sum::<usize>() * std::mem::size_of::<[i32; 2]>()
    }

    pub(super) fn start(&self, channel: usize) -> Option<Vec<[i32; 2]>> {
        let plane = self.planes.get(channel)?;
        if self.channels[channel].get().is_some() {
            return None;
        }
        let mut values = Vec::new();
        values.try_reserve_exact(plane.samples.len()).ok()?;
        Some(values)
    }

    pub(super) fn finish(&self, channel: usize, values: Vec<[i32; 2]>) {
        if values.len() == self.planes[channel].samples.len() {
            let _ = self.channels[channel].set(values);
        }
    }

    pub(super) fn channel(&self, channel: usize) -> Option<&[[i32; 2]]> {
        self.channels.get(channel)?.get().map(Vec::as_slice)
    }
}
