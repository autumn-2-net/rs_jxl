// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_index.h::LZ77StaticIndex's baseline stable counting-sort index at
//! b866b3defec13494bfbaf705db93581cda118e37. Short32 and long8/32 slots
//! include the current origin, so only 31 and 7/31 earlier positions survive.
//! Run-key sharing/alignment and deep16/deep64 proposals retain source caps.
//! Parallel build and match batches keep the source stable candidate order.
//! Reused by match batches and seed-boundary block parsing. Suffix search has
//! a separate index; neither matcher substitutes for the other's nominations.

use super::super::{MIN_MATCH, Match, special_distance};
use super::distances::Codes;
use super::matchers::{filled, hash, window};
use super::search::{MatchCache, Query, physical};
use super::{
    FilterPolicy, Literal, SourceBucketMode, SourceConfig, SourceLz77Method, SourceMatchCost,
};
use crate::tools::lz77_prefix::fingerprint_mask;
use crate::{Error, Result, execution};

#[path = "lz77_source_deep.rs"]
mod deep;
use deep::DeepChain;
#[path = "lz77_source_index_batch.rs"]
mod batch;
#[path = "lz77_source_index_parallel.rs"]
mod parallel;
pub(super) use batch::Batch;
#[path = "lz77_source_values.rs"]
mod values;
use values::Values;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceRunKeys {
    #[default]
    Disabled,
    /// RUN_KEYS=1: hash zero-run length and the following three values.
    LengthAndSuffix,
    /// RUN_KEYS=2: share a run's slot and align older run ends when querying.
    SharedSuffix,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceDeepPrefix {
    #[default]
    Disabled,
    /// DEEP_PREFIX=1: four visits per 64/16-token chain.
    Four,
    /// DEEP_PREFIX=2: eight visits per chain; long ring narrowed to nine slots.
    Eight,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceIndexMode {
    #[default]
    Disabled,
    /// INDEXED=1: the largest stream must contain at least 75% of all tokens.
    Dominant,
    /// INDEXED=2: every sufficiently large stream may use its own index.
    All,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceIndexTags {
    /// Preserve the existing prefix_filter switch (scalar precomputed tags).
    #[default]
    Default,
    Disabled,
    /// INDEX_TAGS=1, mandatory-three-token tags in stable slot order.
    Scalar,
    /// INDEX_TAGS=2, runtime SIMD mask with the portable scalar fallback.
    Simd,
}

pub(super) fn enabled(inputs: &[&[Literal]], config: &SourceConfig) -> bool {
    let total: usize = inputs.iter().map(|s| s.len()).sum();
    let largest = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
    config.index_mode != SourceIndexMode::Disabled
        && config.method == SourceLz77Method::Optc256
        && config.bucket_route().indexed
        && config.match_cost == SourceMatchCost::Frontier
        && (config.small_index
            || (config.parallelism.is_some()
                && !execution::is_serial()
                && rayon::current_thread_index().is_some()))
        && largest >= minimum(config)
        && largest <= 1 << 24
        && (config.index_mode == SourceIndexMode::All || largest >= total - total / 4)
}

pub(super) fn minimum(config: &SourceConfig) -> usize {
    if config.small_index { 8192 } else { 65536 }
}

struct Slots {
    offsets: Vec<u32>,
    slots: Vec<u32>,
    positions: Vec<u32>,
}

impl Slots {
    fn build(size: usize, count: usize, key: impl Fn(usize) -> u32) -> Result<Self> {
        Self::build_shared(size, count, key, |_| false)
    }

    fn build_shared(
        size: usize,
        count: usize,
        key: impl Fn(usize) -> u32,
        shared: impl Fn(usize) -> bool,
    ) -> Result<Self> {
        let mut offsets = filled((1 << 15) + 1, 0u32)?;
        let mut slots = filled(size, 0u32)?;
        for (position, slot) in slots.iter_mut().take(count).enumerate() {
            let hash = key(position);
            if !shared(position) {
                offsets[(hash as usize & 32767) + 1] += 1;
            }
            *slot = hash;
        }
        for bucket in 1..offsets.len() {
            offsets[bucket] += offsets[bucket - 1];
        }
        let mut next = offsets.clone();
        let mut positions = filled(*offsets.last().unwrap() as usize, 0u32)?;
        let mut previous_slot = 0;
        for (position, entry) in slots.iter_mut().take(count).enumerate() {
            if shared(position) {
                *entry = previous_slot;
                continue;
            }
            let bucket = *entry as usize & 32767;
            let slot = next[bucket] as usize;
            next[bucket] += 1;
            positions[slot] = position as u32;
            *entry = slot as u32;
            previous_slot = *entry;
        }
        Ok(Self {
            offsets,
            slots,
            positions,
        })
    }

    fn recent(&self, position: usize, key: u32, capacity: usize) -> &[u32] {
        let end = self.slots[position] as usize + 1;
        let begin = (self.offsets[key as usize & 32767] as usize).max(end.saturating_sub(capacity));
        &self.positions[begin..end]
    }
}

fn zero_suffix(zeros: &[u32], position: usize, window: usize) -> bool {
    zeros[position] >= 8
        && (zeros[position] as usize) < window
        && position + zeros[position] as usize + 3 <= zeros.len()
}

fn shared_run(zeros: &[u32], position: usize, window: usize, mode: SourceRunKeys) -> bool {
    mode == SourceRunKeys::SharedSuffix
        && position != 0
        && zero_suffix(zeros, position, window)
        && zero_suffix(zeros, position - 1, window)
        && zeros[position - 1] == zeros[position] + 1
}

fn run_hash(
    values: &[u32],
    zeros: &[u32],
    position: usize,
    window: usize,
    full: bool,
    mode: SourceRunKeys,
) -> u32 {
    run_hash_count(values, zeros[position], position, window, full, mode)
}

fn run_hash_count(
    values: &[u32],
    zeros: u32,
    position: usize,
    window: usize,
    full: bool,
    mode: SourceRunKeys,
) -> u32 {
    values::long_hash(values, zeros, position, window, full, mode)
}

#[derive(Clone, Copy, Default)]
pub(super) struct Entry {
    pub(super) length: usize,
    pub(super) symbol: u32,
    pub(super) cost: f32,
}

// The source's 16-entry pruning certificate is deliberately bounded. Overflow
// stops retaining new certificates, never drops the candidate being emitted.
#[derive(Clone)]
pub(super) struct Dominance {
    pub(super) entries: [Entry; 16],
    pub(super) size: usize,
    pub(super) overflowed: bool,
}
impl Dominance {
    pub(super) fn new() -> Self {
        Self {
            entries: [Entry::default(); 16],
            size: 0,
            overflowed: false,
        }
    }
    pub(super) fn required(&self, cost: f32, symbol: u32) -> usize {
        for entry in self.entries[..self.size].iter().rev() {
            if entry.cost < cost || (entry.cost == cost && entry.symbol <= symbol) {
                return MIN_MATCH.max(entry.length + 1);
            }
        }
        MIN_MATCH
    }
    pub(super) fn add(&mut self, length: usize, symbol: u32, cost: f32) {
        let mut begin = 0;
        while begin < self.size
            && (self.entries[begin].cost < cost
                || (self.entries[begin].cost == cost && self.entries[begin].symbol < symbol))
        {
            if self.entries[begin].length >= length {
                return;
            }
            begin += 1;
        }
        if begin < self.size
            && self.entries[begin].cost == cost
            && self.entries[begin].symbol == symbol
            && self.entries[begin].length >= length
        {
            return;
        }
        let mut end = begin;
        while end < self.size && self.entries[end].length <= length {
            end += 1;
        }
        let remaining = self.size - (end - begin);
        if remaining == self.entries.len() {
            self.overflowed = true;
            return;
        }
        self.entries.copy_within(end..self.size, begin);
        self.entries.copy_within(begin..remaining, begin + 1);
        self.entries[begin] = Entry {
            length,
            symbol,
            cost,
        };
        self.size = remaining + 1;
    }
}

pub(super) struct StaticIndex<'a> {
    values: Values<'a>,
    short: Slots,
    long: Slots,
    zeros: Vec<u32>,
    window: usize,
    maximum: usize,
    full: bool,
    long_capacity: usize,
    short_tags: Vec<u8>,
    long_tags: Vec<u8>,
    counts: Vec<u16>,
    simd_tags: bool,
    anchors_first: bool,
    word_match: bool,
    width: usize,
    codes: Codes,
    anchors: Vec<usize>,
    run_keys: SourceRunKeys,
    deep_prefix: SourceDeepPrefix,
    deep16: DeepChain,
    deep64: DeepChain,
    #[cfg(test)]
    build_tasks: usize,
}

impl<'a> StaticIndex<'a> {
    #[cfg(test)]
    pub(super) fn layout(&self) -> (bool, bool) {
        (
            matches!(self.values, Values::Narrow(_)),
            !self.counts.is_empty(),
        )
    }

    pub(super) fn new(values: &'a [u32], width: usize, config: &SourceConfig) -> Result<Self> {
        let _profile = crate::profile::scope("lz77.source.static_index_build");
        // All source dispatches cap indexed streams at 2^24. Keep every
        // u32 slot conversion and optional 2-byte/token allocation bounded.
        if values.len() > 1 << 24 {
            return Err(Error::InvalidInput(
                "source LZ77 index exceeds token limit".into(),
            ));
        }
        let full = config.bucket_route().mode == SourceBucketMode::WideRing;
        let long_capacity = if full { 32 } else { 8 };
        let window = window(values.len());
        let build_tasks = parallel::tasks(values.len(), config);
        let (short, long, zeros) = if build_tasks > 1 {
            parallel::build(values, window, full, config.index_run_keys, build_tasks)?
        } else {
            let short = Slots::build(values.len(), values.len(), |p| hash(values, p))?;
            let mut zeros = filled(values.len(), 0u32)?;
            let mut count = 0;
            for p in (0..values.len()).rev() {
                count = if values[p] == 0 {
                    (count + 1).min(window as u32)
                } else {
                    0
                };
                zeros[p] = count;
            }
            let long = Slots::build_shared(
                values.len(),
                values.len().saturating_sub(7),
                |p| run_hash(values, &zeros, p, window, full, config.index_run_keys),
                |p| shared_run(&zeros, p, window, config.index_run_keys),
            )?;
            (short, long, zeros)
        };
        let (deep16, deep64) = if config.index_deep_prefix == SourceDeepPrefix::Disabled {
            (DeepChain::default(), DeepChain::default())
        } else {
            (DeepChain::build(values, 16)?, DeepChain::build(values, 64)?)
        };
        let tags = match config.index_tags {
            SourceIndexTags::Default => config.prefix_filter != FilterPolicy::Disabled,
            SourceIndexTags::Disabled => false,
            SourceIndexTags::Scalar | SourceIndexTags::Simd => true,
        };
        let make_tags = |slots: &Slots| -> Result<Vec<u8>> {
            let mut result = super::reserved(if tags { slots.positions.len() } else { 0 })?;
            if tags {
                result.extend(
                    slots
                        .positions
                        .iter()
                        .map(|&p| (hash(values, p as usize) >> 24) as u8),
                );
            }
            Ok(result)
        };
        let short_tags = make_tags(&short)?;
        let long_tags = make_tags(&long)?;
        // Source coordinates retain the offsets/slots and add two saturated
        // byte counts. Shared run positions reuse the same long slot/count.
        let mut counts = filled(
            if config.index_coordinates {
                values.len()
            } else {
                0
            },
            0u16,
        )?;
        for (position, count) in counts.iter_mut().enumerate() {
            *count = short.recent(position, hash(values, position), 32).len() as u16;
            if position + 8 <= values.len() {
                let key = run_hash(
                    values,
                    &zeros,
                    position,
                    window,
                    full,
                    config.index_run_keys,
                );
                *count |= (long.recent(position, key, long_capacity).len() as u16) << 8;
            }
        }
        let mut anchors = Vec::new();
        if width != 0 {
            for symbol in 0..24 {
                let distance = special_distance(symbol, width);
                if distance > 0 && distance < window as i128 {
                    anchors.push(distance as usize);
                }
            }
        }
        Ok(Self {
            values: Values::new(values, config.index_compact_values)?,
            short,
            long,
            zeros,
            window,
            maximum: config.max_match_length,
            full,
            long_capacity,
            short_tags,
            long_tags,
            counts,
            simd_tags: config.index_tags == SourceIndexTags::Simd,
            anchors_first: config.index_anchors_first,
            word_match: config.index_word_match,
            width,
            codes: Codes::new(width, true)?,
            anchors,
            run_keys: config.index_run_keys,
            deep_prefix: config.index_deep_prefix,
            deep16,
            deep64,
            #[cfg(test)]
            build_tasks,
        })
    }

    pub(super) fn find_matches(
        &self,
        position: usize,
        cost: impl Fn(u32) -> f32,
        visit: impl FnMut(Match),
    ) {
        self.find_cached_matches(position, cost, visit, &mut MatchCache::uncached());
    }

    fn find_cached_matches(
        &self,
        position: usize,
        cost: impl Fn(u32) -> f32,
        visit: impl FnMut(Match),
        cache: &mut MatchCache,
    ) {
        self.find_bounded_matches(position, usize::MAX, cost, visit, cache);
    }

    pub(super) fn find_bounded_matches(
        &self,
        position: usize,
        maximum: usize,
        cost: impl Fn(u32) -> f32,
        visit: impl FnMut(Match),
        cache: &mut MatchCache,
    ) {
        self.find_query(
            position,
            Query {
                maximum,
                ..Query::default()
            },
            cost,
            visit,
            cache,
        );
    }

    pub(super) fn find_query(
        &self,
        position: usize,
        query: Query<'_>,
        cost: impl Fn(u32) -> f32,
        mut visit: impl FnMut(Match),
        cache: &mut MatchCache,
    ) {
        let mut local = Dominance::new();
        let frontier = query.result.unwrap_or(&mut local);
        *frontier = Dominance::new();
        let limit = self
            .maximum
            .min(query.maximum)
            .min(self.values.len().saturating_sub(position));
        if limit < MIN_MATCH {
            return;
        }
        cache.advance(position);
        let complete = std::cell::Cell::new(false);
        if let Some(carry) = query.carry {
            carry.visit(position, MIN_MATCH, limit, |length, symbol| {
                let bits = cost(symbol);
                visit(Match {
                    length,
                    symbol,
                    distance: physical(symbol, self.width),
                });
                frontier.add(length, symbol, bits);
                complete.set(complete.get() || query.bound.reached(length, limit, bits, symbol));
            });
            if complete.get() {
                return;
            }
        }
        let mut candidate = |previous: usize| {
            if previous >= position
                || position - previous >= self.window
                || self.values.get(position) != self.values.get(previous)
                || self.values.get(position + MIN_MATCH - 1)
                    != self.values.get(previous + MIN_MATCH - 1)
            {
                return;
            }
            let distance = position - previous;
            let symbol = self.codes.symbol(distance);
            let bits = cost(symbol);
            let required = frontier.required(bits, symbol);
            if required > limit
                || self.values.get(position + required - 1)
                    != self.values.get(previous + required - 1)
            {
                return;
            }
            let prefix = limit
                .min(self.zeros[position].min(self.zeros[previous]) as usize)
                .max(cache.prefix(distance, limit));
            let length = self
                .values
                .length(position, previous, prefix, limit, self.word_match);
            cache.remember(distance, length);
            if length < required {
                return;
            }
            visit(Match {
                length,
                distance,
                symbol,
            });
            frontier.add(length, symbol, bits);
            complete.set(query.bound.reached(length, limit, bits, symbol));
        };
        if query.hint_distance != 0 && query.hint_distance <= position {
            candidate(position - query.hint_distance);
            if complete.get() {
                return;
            }
        }
        let short_end = self.short.slots[position] as usize + 1;
        let short_count = if self.counts.is_empty() {
            self.short
                .recent(position, self.values.hash(position), 32)
                .len()
        } else {
            (self.counts[position] & 255) as usize
        };
        let oldest = if short_count == 32 {
            self.short.positions[short_end - short_count] as usize
        } else {
            0
        };
        let tag = self.short_tags.get(short_end - 1).copied().unwrap_or(0);
        let anchors = |candidate: &mut dyn FnMut(usize)| {
            for &distance in &self.anchors {
                if distance <= position && position - distance < oldest {
                    candidate(position - distance);
                }
                if complete.get() {
                    break;
                }
            }
        };
        if self.anchors_first {
            anchors(&mut candidate);
        }
        if complete.get() {
            return;
        }
        if self.deep_prefix != SourceDeepPrefix::Disabled {
            let visits = if self.deep_prefix == SourceDeepPrefix::Four {
                4
            } else {
                8
            };
            self.deep64
                .find(position, self.window, visits, &mut candidate);
            self.deep16
                .find(position, self.window, visits, &mut candidate);
        }
        if visit_tagged_slots(
            &self.short,
            &self.short_tags,
            position,
            short_count,
            tag,
            self.simd_tags,
            true,
            || complete.get(),
            &mut candidate,
        ) {
            return;
        }
        if self.values.len() - position >= 8 {
            let available = if self.counts.is_empty() {
                let key = self.values.long_hash(
                    self.zeros[position],
                    position,
                    self.window,
                    self.full,
                    self.run_keys,
                );
                self.long.recent(position, key, self.long_capacity).len()
            } else {
                (self.counts[position] >> 8) as usize
            };
            let capacity = if self.deep_prefix == SourceDeepPrefix::Eight {
                self.long_capacity.min(9)
            } else {
                self.long_capacity
            };
            let count = available.min(capacity);
            if visit_tagged_slots(
                &self.long,
                &self.long_tags,
                position,
                count,
                tag,
                self.simd_tags,
                false,
                || complete.get(),
                |mut previous| {
                    if previous < oldest {
                        if self.run_keys == SourceRunKeys::SharedSuffix
                            && zero_suffix(&self.zeros, position, self.window)
                            && zero_suffix(&self.zeros, previous, self.window)
                            && self.zeros[previous] >= self.zeros[position]
                        {
                            previous += (self.zeros[previous] - self.zeros[position]) as usize;
                        }
                        candidate(previous);
                    }
                },
            ) {
                return;
            }
        }
        if !self.anchors_first {
            anchors(&mut candidate);
        }
    }
}

// Include the current slot in a full lane, then mask it out. Descending bits
// preserve newest-first visitation, including run-shared and partial buckets.
// Keep source bucket bounds, tag policy and early-stop callbacks explicit.
#[allow(clippy::too_many_arguments)]
fn visit_tagged_slots(
    slots: &Slots,
    tags: &[u8],
    position: usize,
    count: usize,
    wanted: u8,
    simd: bool,
    stop_after_tag_miss: bool,
    stop: impl Fn() -> bool,
    mut visit: impl FnMut(usize),
) -> bool {
    let end = slots.slots[position] as usize + 1;
    let begin = end - count;
    if simd {
        let mut mask = fingerprint_mask(&tags[begin..end], wanted).expect("index capacities <=32");
        mask &= !(1u64 << (count - 1));
        while mask != 0 {
            let bit = 63 - mask.leading_zeros() as usize;
            mask &= !(1u64 << bit);
            visit(slots.positions[begin + bit] as usize);
            if stop() {
                return true;
            }
        }
    } else {
        for slot in (begin..end - 1).rev() {
            if tags.is_empty() || tags[slot] == wanted {
                visit(slots.positions[slot] as usize);
            } else if !stop_after_tag_miss {
                continue;
            }
            if stop() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::dp::Frontier;
    use super::super::matchers::{Buckets, long_hash};
    use super::*;

    #[test]
    fn shared_query_resets_results_and_bound_stops_at_cheapest_full_length() {
        use super::super::search::DistanceBound;
        let values = vec![7; 200];
        let index = StaticIndex::new(&values, 0, &SourceConfig::default()).unwrap();
        let calls = std::cell::Cell::new(0);
        let mut frontier = Dominance::new();
        let query = Query {
            maximum: 30,
            result: Some(&mut frontier),
            bound: DistanceBound {
                cost: 0.0,
                symbol: 0,
            },
            ..Query::default()
        };
        index.find_query(
            100,
            query,
            |_| {
                calls.set(calls.get() + 1);
                0.0
            },
            |_| {},
            &mut MatchCache::uncached(),
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(frontier.size, 1);
        assert_eq!(
            (frontier.entries[0].length, frontier.entries[0].symbol),
            (30, 0)
        );
        calls.set(0);
        index.find_bounded_matches(
            100,
            30,
            |_| {
                calls.set(calls.get() + 1);
                0.0
            },
            |_| {},
            &mut MatchCache::uncached(),
        );
        assert!(calls.get() > 1);
        index.find_query(
            values.len(),
            Query {
                result: Some(&mut frontier),
                ..Query::default()
            },
            |_| 0.0,
            |_| {},
            &mut MatchCache::uncached(),
        );
        assert_eq!(frontier.size, 0);
        assert!(!frontier.overflowed);
    }

    #[test]
    fn word_queries_verify_hints_and_sparse_cache_matches_uncached_candidates() {
        let values: Vec<u32> = (0..9000)
            .map(|i| if i % 127 < 49 { 0 } else { 1 + i % 19 })
            .collect();
        let config = SourceConfig {
            index_word_match: true,
            block_sparse_cache: true,
            index_tags: SourceIndexTags::Simd,
            ..SourceConfig::default()
        };
        let word = StaticIndex::new(&values, 37, &config).unwrap();
        let scalar = StaticIndex::new(
            &values,
            37,
            &SourceConfig {
                index_word_match: false,
                ..config.clone()
            },
        )
        .unwrap();
        let mut cache = MatchCache::for_blocks(word.window, &config).unwrap();
        for p in (0..values.len()).step_by(7) {
            let mut expected = Vec::new();
            let mut actual = Vec::new();
            let cost = |d| (d % 13) as f32;
            let query = || Query {
                maximum: 131,
                hint_distance: if p % 3 == 0 { p + 1 } else { 19 },
                ..Query::default()
            };
            scalar.find_query(
                p,
                query(),
                cost,
                |m| expected.push((m.length, m.distance, m.symbol)),
                &mut MatchCache::uncached(),
            );
            word.find_query(
                p,
                query(),
                cost,
                |m| {
                    assert!(m.distance > 0 && m.distance <= p && m.distance < word.window);
                    assert_eq!(
                        &values[p..p + m.length],
                        &values[p - m.distance..p - m.distance + m.length]
                    );
                    actual.push((m.length, m.distance, m.symbol));
                },
                &mut cache,
            );
            assert_eq!(actual, expected, "position={p}");
        }
    }

    #[test]
    fn precomputed_simd_tags_preserve_scalar_candidates_with_shared_runs_and_tails() {
        let mut values = Vec::new();
        for run in 0..50 {
            values.extend(std::iter::repeat_n(0, 17 + run % 29));
            values.extend([1000, 1001, 1002, (run % 3) as u32, u32::MAX]);
        }
        values.extend([1, 2]);
        for run_keys in [
            SourceRunKeys::Disabled,
            SourceRunKeys::LengthAndSuffix,
            SourceRunKeys::SharedSuffix,
        ] {
            let mut config = SourceConfig {
                index_run_keys: run_keys,
                index_tags: SourceIndexTags::Scalar,
                ..SourceConfig::default()
            };
            let scalar = StaticIndex::new(&values, 37, &config).unwrap();
            for (tags, slots) in [
                (&scalar.short_tags, &scalar.short),
                (&scalar.long_tags, &scalar.long),
            ] {
                assert_eq!(tags.len(), slots.positions.len());
                for (tag, &p) in tags.iter().zip(&slots.positions) {
                    assert_eq!(*tag, (hash(&values, p as usize) >> 24) as u8);
                }
            }
            config.index_tags = SourceIndexTags::Simd;
            let simd = StaticIndex::new(&values, 37, &config).unwrap();
            for p in (0..values.len()).rev() {
                let mut expected = Vec::new();
                let mut actual = Vec::new();
                scalar.find_matches(
                    p,
                    |d| (d % 11) as f32,
                    |m| expected.push((m.length, m.distance, m.symbol)),
                );
                simd.find_matches(
                    p,
                    |d| (d % 11) as f32,
                    |m| actual.push((m.length, m.distance, m.symbol)),
                );
                assert_eq!(actual, expected, "p={p}, runs={run_keys:?}");
            }
        }
    }

    #[test]
    fn stable_slots_include_current_but_never_future_positions() {
        let slots = Slots::build(200, 193, |p| (p % 3) as u32).unwrap();
        for p in 0..193 {
            for capacity in [8, 32] {
                let expected: Vec<_> = (0..=p)
                    .rev()
                    .filter(|i| i % 3 == p % 3)
                    .take(capacity)
                    .map(|i| i as u32)
                    .collect();
                assert_eq!(
                    slots
                        .recent(p, (p % 3) as u32, capacity)
                        .iter()
                        .rev()
                        .copied()
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }

    fn check_index<const LONG: usize>(values: &[u32], config: &SourceConfig, width: usize) {
        let index = StaticIndex::new(values, width, config).unwrap();
        let mut buckets = Buckets::<LONG>::new(values, width, config).unwrap();
        let cost = |symbol| ((symbol * 17) % 31) as f32;
        let mut expected = Vec::new();
        for position in 0..values.len() {
            buckets.update(position);
            let mut frontier = Frontier::new(100).unwrap();
            buckets.find_matches(position, true, |m| {
                frontier.add(m, f64::from(cost(m.symbol)))
            });
            expected.push(
                frontier
                    .entries
                    .iter()
                    .map(|e| (e.matched.length, e.matched.symbol))
                    .collect::<Vec<_>>(),
            );
        }
        // The immutable index must work in reverse order as well as forwards.
        for position in (0..values.len()).rev() {
            let mut frontier = Frontier::new(100).unwrap();
            index.find_matches(position, cost, |m| {
                frontier.add(m, f64::from(cost(m.symbol)))
            });
            let actual: Vec<_> = frontier
                .entries
                .iter()
                .map(|e| (e.matched.length, e.matched.symbol))
                .collect();
            assert_eq!(actual, expected[position], "position={position}");
        }
    }

    #[test]
    fn indexed_frontiers_equal_mutable_rings_with_tags_tails_and_zero_runs() {
        let mut values: Vec<_> = (0..1200)
            .map(|i| {
                if i % 29 < 8 {
                    u32::MAX - (i % 29)
                } else {
                    i % 37
                }
            })
            .collect();
        values.extend([0; 513]);
        values.extend([u32::MAX, 0, 0, 0, 1, 0, 0]);
        for width in [0, 1, 37] {
            for prefix_filter in [FilterPolicy::Disabled, FilterPolicy::Eager] {
                let mut config = SourceConfig {
                    prefix_filter,
                    max_match_length: 301,
                    ..SourceConfig::default()
                };
                check_index::<8>(&values, &config, width);
                config.bucket_mode = SourceBucketMode::WideRing;
                check_index::<32>(&values, &config, width);
            }
        }
    }

    #[test]
    fn run_keys_share_slots_and_align_old_run_ends_to_recover_evicted_matches() {
        let mut values = vec![0; 120];
        values.extend([11, 13, 17]);
        let position = values.len();
        values.extend([0; 40]);
        values.extend([11, 13, 17]);
        for mode in [
            SourceRunKeys::Disabled,
            SourceRunKeys::LengthAndSuffix,
            SourceRunKeys::SharedSuffix,
        ] {
            let config = SourceConfig {
                index_run_keys: mode,
                prefix_filter: FilterPolicy::Eager,
                ..SourceConfig::default()
            };
            let index = StaticIndex::new(&values, 0, &config).unwrap();
            let mut found = false;
            index.find_matches(
                position,
                |_| 1.0,
                |m| {
                    found |= m.length == 43 && m.distance == position - 80;
                },
            );
            assert_eq!(found, mode != SourceRunKeys::Disabled, "mode={mode:?}");
            let valid = values.len() - 7;
            let shared = (0..valid)
                .filter(|&p| shared_run(&index.zeros, p, index.window, mode))
                .count();
            assert_eq!(index.long.positions.len(), valid - shared);
            if mode == SourceRunKeys::SharedSuffix {
                for p in 1..=112 {
                    assert_eq!(index.long.slots[p], index.long.slots[0]);
                }
                assert_ne!(index.long.slots[113], index.long.slots[0]);
                assert_eq!(
                    run_hash(&values, &index.zeros, 0, index.window, false, mode),
                    run_hash(&values, &index.zeros, 80, index.window, false, mode)
                );
            }
            if mode == SourceRunKeys::LengthAndSuffix {
                assert_ne!(
                    run_hash(&values, &index.zeros, 0, index.window, false, mode),
                    run_hash(&values, &index.zeros, 80, index.window, false, mode)
                );
            }
        }
    }

    #[test]
    fn run_key_suffix_bounds_require_three_values_and_exclude_window_saturation() {
        let zeros = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0];
        assert!(!zero_suffix(&zeros, 0, 16));
        let zeros = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0];
        assert!(!zero_suffix(&zeros, 0, 8));
        assert!(zero_suffix(&zeros, 0, 16));
        let values = [0, 0, 0, 0, 0, 0, 0, 0, 11, 13, 17];
        for full in [false, true] {
            assert_eq!(
                run_hash(&values, &zeros, 0, 8, full, SourceRunKeys::SharedSuffix),
                long_hash(&values, 0, hash(&values, 0), full)
            );
        }
    }

    #[test]
    fn deep_chains_recover_long_prefixes_evicted_from_both_recent_rings() {
        let mut values = Vec::new();
        for block in 0..60u32 {
            values.extend([10, 11, 12, 13, 14, 15, 16, 17]);
            values.extend((8..64).map(|i| 10000 + block * 64 + i));
        }
        let position = values.len();
        values.extend_from_within(0..64);
        for deep in [
            SourceDeepPrefix::Disabled,
            SourceDeepPrefix::Four,
            SourceDeepPrefix::Eight,
        ] {
            let config = SourceConfig {
                index_deep_prefix: deep,
                ..SourceConfig::default()
            };
            let index = StaticIndex::new(&values, 0, &config).unwrap();
            let mut found = false;
            index.find_matches(
                position,
                |_| 1.0,
                |m| {
                    found |= m.distance == position && m.length == 64;
                },
            );
            assert_eq!(found, deep != SourceDeepPrefix::Disabled);
        }
    }

    #[test]
    fn deep_eight_retains_source_nine_slot_long_query_cap_not_the_full_wide_ring() {
        let prefix = [10, 11, 12, 13, 14, 15, 16, 17];
        let mut values = Vec::new();
        for block in 0..16u32 {
            values.extend(prefix);
            values.push(1000 + block);
            for filler in 0..3 {
                let mut changed = prefix;
                changed[3] = 10000 + block * 3 + filler;
                values.extend(changed);
                values.push(2000 + block);
            }
        }
        let position = values.len();
        values.extend(prefix);
        values.push(1000);
        for deep in [SourceDeepPrefix::Four, SourceDeepPrefix::Eight] {
            let config = SourceConfig {
                index_deep_prefix: deep,
                bucket_mode: SourceBucketMode::WideRing,
                ..SourceConfig::default()
            };
            let index = StaticIndex::new(&values, 0, &config).unwrap();
            let mut found = false;
            index.find_matches(
                position,
                |_| 1.0,
                |m| {
                    found |= m.distance == position && m.length == 9;
                },
            );
            assert_eq!(found, deep == SourceDeepPrefix::Four);
        }
    }

    #[test]
    fn dominance_overflow_only_loses_pruning_certificates() {
        let mut frontier = Dominance::new();
        for i in 0..32 {
            frontier.add(3 + i, i as u32, i as f32);
        }
        assert_eq!(frontier.size, 16);
        assert!(frontier.overflowed);
        assert_eq!(frontier.required(40.0, 40), 19);
        frontier.add(100, 0, 0.0);
        assert_eq!(frontier.size, 1);
        assert_eq!(frontier.required(0.0, 0), 101);
        assert!(frontier.overflowed);
    }

    // Called by the existing worker-pool test. This port creates no executor;
    // the test supplies the same already installed pool as the public caller.
    pub(in super::super) fn check_installed_pool() {
        use super::super::{SourceParallelism, Work, run};
        assert_eq!(rayon::current_num_threads(), 2);
        let mut values: Vec<u32> = (0..65559).map(|i| (i % 97) + 1).collect();
        values[16000..51000].fill(0);
        for index_run_keys in [SourceRunKeys::Disabled, SourceRunKeys::LengthAndSuffix] {
            let config = SourceConfig {
                method: SourceLz77Method::Optc256,
                bucket_mode: SourceBucketMode::WideRing,
                match_cost: SourceMatchCost::Frontier,
                index_mode: SourceIndexMode::All,
                small_index: true,
                index_run_keys,
                parallel_index_build: true,
                index_compact_values: true,
                index_coordinates: true,
                index_match_cache: true,
                max_match_length: 127,
                parallelism: Some(SourceParallelism::default()),
                require_global_savings: false,
                ..SourceConfig::default()
            };
            // Logical chunks/caches retain source counts on a two-thread pool.
            assert_eq!(parallel::tasks(values.len(), &config), 4);
            let index = StaticIndex::new(&values, 37, &config).unwrap();
            let serial = execution::serial(|| StaticIndex::new(&values, 37, &config).unwrap());
            assert_eq!(index.build_tasks, 4);
            assert_eq!(serial.build_tasks, 1);
            assert_eq!(index.layout(), (true, true));
            assert_eq!(serial.layout(), (true, true));
            assert_eq!(index.counts, serial.counts);
            assert_eq!(index.zeros, serial.zeros);
            assert_eq!(index.short.offsets, serial.short.offsets);
            assert_eq!(index.short.slots, serial.short.slots);
            assert_eq!(index.short.positions, serial.short.positions);
            assert_eq!(index.long.offsets, serial.long.offsets);
            assert_eq!(index.long.slots, serial.long.slots);
            assert_eq!(index.long.positions, serial.long.positions);
            let mut batch = Batch::new(&index, 37, &config).unwrap();
            let mut serial_batch = execution::serial(|| Batch::new(&serial, 37, &config).unwrap());
            assert_eq!(batch.tasks(), 8);
            assert_eq!(serial_batch.tasks(), 1);
            for position in (0..values.len()).step_by(509).chain([values.len() - 1]) {
                let mut actual = Vec::new();
                let mut expected = Vec::new();
                let cost = |symbol| ((symbol * 17) % 31) as f32;
                index.find_matches(position, cost, |m| {
                    actual.push((m.length, m.distance, m.symbol))
                });
                serial.find_matches(position, cost, |m| {
                    expected.push((m.length, m.distance, m.symbol))
                });
                assert_eq!(actual, expected);
                actual.clear();
                expected.clear();
                batch.find_matches(&index, position, cost, |m| {
                    actual.push((m.length, m.distance, m.symbol))
                });
                execution::serial(|| {
                    serial_batch.find_matches(&serial, position, cost, |m| {
                        expected.push((m.length, m.distance, m.symbol))
                    })
                });
                assert_eq!(actual, expected);
            }
            let input: Vec<_> = values
                .iter()
                .map(|&value| Literal { value, context: 0 })
                .collect();
            let mut parallel_work = Work::default();
            let actual = run(&[&input], &[37], &config, &mut parallel_work).unwrap();
            let mut serial_work = Work::default();
            let expected =
                execution::serial(|| run(&[&input], &[37], &config, &mut serial_work).unwrap());
            assert_eq!(actual, expected);
            assert_eq!(parallel_work.indexed_streams, 1);
            assert_eq!(serial_work.indexed_streams, 1);
            for index_deep_prefix in [SourceDeepPrefix::Four, SourceDeepPrefix::Eight] {
                let mut restricted = config.clone();
                restricted.index_deep_prefix = index_deep_prefix;
                assert_eq!(parallel::tasks(values.len(), &restricted), 1);
            }
            let mut restricted = config.clone();
            restricted.index_run_keys = SourceRunKeys::SharedSuffix;
            assert_eq!(parallel::tasks(values.len(), &restricted), 1);
            restricted = config.clone();
            restricted.parallelism.as_mut().unwrap().scratch_bytes = 1;
            assert_eq!(parallel::tasks(values.len(), &restricted), 1);
            assert_eq!(Batch::new(&index, 37, &restricted).unwrap().tasks(), 1);
        }
    }

    #[test]
    fn index_options_never_start_a_pool_without_an_installed_caller() {
        let config = SourceConfig {
            parallel_index_build: true,
            index_match_cache: true,
            parallelism: Some(super::super::SourceParallelism::default()),
            ..SourceConfig::default()
        };
        assert!(rayon::current_thread_index().is_none());
        assert_eq!(parallel::tasks(65536, &config), 1);
        let index = StaticIndex::new(&[0; 127], 0, &config).unwrap();
        assert_eq!(index.build_tasks, 1);
        assert_eq!(Batch::new(&index, 0, &config).unwrap().tasks(), 1);
    }
}

#[cfg(test)]
#[path = "lz77_source_compact_tests.rs"]
mod compact_tests;
