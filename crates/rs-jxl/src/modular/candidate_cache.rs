//! Bounded, invocation-local sharing of immutable preparation between candidates.
//! No compressed results, learned trees or candidate-specific sample sets live here.
#![forbid(unsafe_code)]

use std::{marker::PhantomData, sync::{Arc, Mutex}};
use super::{
    encode::{ModularOptions, Prepared},
    group::Transform,
    layout::{Passes, Rect},
    orchestration::PreparedLayout,
    palette::PaletteOptions,
    predictor::Predictor,
};
use crate::{Image, Result};

const MAX_ENTRIES: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparationKey {
    area: Option<Rect>,
    group_size_shift: u8,
    passes: Passes,
    responsive: bool,
    rct: Option<u32>,
    rct_candidates: Option<usize>,
    weighted: bool,
    weighted_preset: u8,
    weighted_candidates: Option<usize>,
    parallel_weighted: bool,
    palette_colors: Option<u32>,
    palette_predictor: Option<Predictor>,
    palette_policy: Option<PaletteOptions>,
    squeeze: Vec<bool>,
}

impl PreparationKey {
    pub(super) fn full(options: &ModularOptions, shift: u8, passes: &Passes, responsive: bool) -> Self {
        Self {
            area: None, group_size_shift: shift, passes: passes.clone(), responsive,
            rct: options.rct, rct_candidates: options.rct_candidates,
            weighted: options.predictor.has_weighted(), weighted_preset: options.weighted_preset,
            weighted_candidates: options.weighted_candidates, parallel_weighted: options.parallel_weighted,
            palette_colors: options.palette_colors, palette_predictor: options.palette_predictor,
            palette_policy: options.palette_policy.clone(), squeeze: options.squeeze.clone(),
        }
    }

    pub(super) fn area(options: &ModularOptions, shift: u8, area: Rect) -> Self {
        Self { area: Some(area), ..Self::full(options, shift, &Passes::default(), false) }
    }

    fn heap_bytes(&self) -> usize {
        self.squeeze.capacity()
            .saturating_add(self.passes.downsample.capacity().saturating_mul(size_of::<super::layout::Downsample>()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReuseStats {
    pub hits: usize,
    pub misses: usize,
    pub duplicate_builds: usize,
    pub bypasses: usize,
    pub entries: usize,
    /// Retained prepared buffers/metadata, not process RSS or all encoder scratch.
    pub retained_bytes: usize,
    pub peak_bytes: usize,
}

struct Entry {
    key: PreparationKey,
    prepared: Arc<PreparedLayout>,
}

struct State {
    active: bool,
    entries: [Option<Entry>; MAX_ENTRIES],
    stats: ReuseStats,
}

/// Opaque handle installed by native candidate search. It cannot be constructed
/// publicly or used after its root-image borrow scope ends.
pub struct CandidateCache {
    root: usize,
    limit: usize,
    state: Mutex<State>,
}

impl std::fmt::Debug for CandidateCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandidateCache").field("limit", &self.limit).finish_non_exhaustive()
    }
}

pub(crate) struct CacheScope<'a> {
    pub(crate) cache: Arc<CandidateCache>,
    _root: PhantomData<&'a Image>,
}

impl Drop for CacheScope<'_> {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().unwrap_or_else(|e| e.into_inner());
        state.active = false;
        state.entries.fill_with(|| None);
        state.stats.entries = 0;
        state.stats.retained_bytes = 0;
    }
}

impl CandidateCache {
    pub(crate) fn scope(image: &Image, limit: usize) -> CacheScope<'_> {
        CacheScope {
            cache: Arc::new(Self {
                root: image as *const Image as usize, limit,
                state: Mutex::new(State {
                    active: true, entries: std::array::from_fn(|_| None), stats: ReuseStats::default(),
                }),
            }),
            _root: PhantomData,
        }
    }

    pub fn stats(&self) -> ReuseStats {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).stats
    }

    pub(super) fn matches(&self, image: &Image) -> bool {
        // Only identity is compared, never dereferenced. CacheScope's borrow
        // keeps this immutable Image alive; Drop disables retained handles.
        self.root == image as *const Image as usize && self.limit != 0
            && self.state.lock().unwrap_or_else(|e| e.into_inner()).active
    }

    pub(super) fn get_or_build(
        &self,
        key: PreparationKey,
        build: impl FnOnce() -> Result<PreparedLayout>,
        size: impl FnOnce(&PreparedLayout) -> usize,
    ) -> Result<Arc<PreparedLayout>> {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.active {
                if let Some(entry) = state.entries.iter().flatten().find(|e| e.key == key) {
                    let prepared = Arc::clone(&entry.prepared);
                    state.stats.hits += 1;
                    return Ok(prepared);
                }
                state.stats.misses += 1;
            }
        }
        // Never wait for a builder while occupying cooperative worker slots.
        // Simultaneous cold misses may compute twice, but never serialize Rayon.
        let prepared = Arc::new(build()?);
        let bytes = size(&prepared).saturating_add(key.heap_bytes()).saturating_add(size_of::<Entry>());
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.active {
            return Ok(prepared);
        }
        if state.entries.iter().flatten().any(|e| e.key == key) {
            state.stats.duplicate_builds += 1;
            return Ok(prepared);
        }
        if bytes <= self.limit.saturating_sub(state.stats.retained_bytes)
            && let Some(slot) = state.entries.iter_mut().find(|slot| slot.is_none())
        {
            *slot = Some(Entry { key, prepared: Arc::clone(&prepared) });
            state.stats.entries += 1;
            state.stats.retained_bytes += bytes;
            state.stats.peak_bytes = state.stats.peak_bytes.max(state.stats.retained_bytes);
        } else {
            state.stats.bypasses += 1;
        }
        Ok(prepared)
    }
}

pub(super) fn prepared_bytes(layout: &PreparedLayout) -> usize {
    fn payload(prepared: &Prepared) -> usize {
        let channels = prepared.planes.iter().fold(
            prepared.planes.capacity().saturating_mul(size_of::<super::group::Channel>()),
            |n, channel| n.saturating_add(channel.samples.capacity().saturating_mul(size_of::<i32>())),
        );
        prepared.transforms.iter().fold(
            channels.saturating_add(prepared.transforms.capacity().saturating_mul(size_of::<Transform>())),
            |n, transform| n.saturating_add(match transform {
                Transform::SqueezeScript(parameters) => parameters.capacity().saturating_mul(size_of::<super::layout::SqueezeParams>()),
                _ => 0,
            }),
        )
    }
    let mut bytes = size_of::<PreparedLayout>().saturating_add(2 * size_of::<usize>())
        .saturating_add(payload(&layout.global))
        .saturating_add(layout.streams.capacity().saturating_mul(size_of::<Prepared>()));
    for prepared in &layout.streams {
        bytes = bytes.saturating_add(payload(prepared));
    }
    bytes = bytes.saturating_add(layout.partition.streams.capacity().saturating_mul(size_of::<super::layout::StreamPlan>()));
    for stream in &layout.partition.streams {
        bytes = bytes.saturating_add(stream.channels.capacity().saturating_mul(size_of::<super::layout::ChannelTile>()));
    }
    bytes
}
