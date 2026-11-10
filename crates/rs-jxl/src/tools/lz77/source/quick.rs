// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Source seed pricing/control flow derives from BSD-3-Clause libjxl
// enc_lz77.cc at b866b3defec13494bfbaf705db93581cda118e37.

//! Opt-in candidate experiments, not alternate costs or decoder semantics.
//! Sparse probing is Brotli-inspired; the double hash is a bounded experiment,
//! not a claim of equivalence to either Brotli or the source chain matcher.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    Command, DistanceCodes, Fast, Literal, MIN_MATCH, Match, SourceConfig, WINDOW_SIZE,
    fast_stream, prices, reserved, static_cost_with_penalty,
};
use crate::{Error, Result};

mod cost;

/// Independent fast-seed candidate selectors; disabled by default.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceQuickMode {
    #[default]
    Off,
    /// Existing Fast<7>, with escalating probes after misses.
    Sparse,
    /// Fixed two-entry short3/long8 buckets and at most five spatial distances.
    DoubleFast,
    /// DoubleFast with the same sparse miss schedule as Sparse.
    DoubleFastSparse,
    /// Sparse two-key matching ranked by source seed gain, preserving careful seeds.
    DoubleFastCost,
}

impl SourceQuickMode {
    fn sparse(self) -> bool {
        matches!(
            self,
            Self::Sparse | Self::DoubleFastSparse | Self::DoubleFastCost
        )
    }

    fn double(self) -> bool {
        matches!(
            self,
            Self::DoubleFast | Self::DoubleFastSparse | Self::DoubleFastCost
        )
    }
}

/// Per-call counters, or a receipt snapshot summed over completed calls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SourceQuickStats {
    pub calls: u64,
    pub sparse_calls: u64,
    pub double_fast_calls: u64,
    pub cost_ranked_calls: u64,
    /// Original careful seeds retained by DoubleFastCost; not quick-parser calls.
    pub preserved_careful_calls: u64,
    pub queries: u64,
    /// Queries with no match yielding strictly positive source-model savings.
    pub query_misses: u64,
    /// Queries finding a legal match whose source cost is >= its literal cost.
    pub unprofitable_queries: u64,
    /// Emitted matches, including non-profitable matches in ungated methods.
    pub matches: u64,
    pub useful_matches: u64,
    /// Useful emitted matches of at least eight values; these reset sparsity.
    pub long_matches: u64,
    pub matched_values: u64,
    /// Extra literals emitted by sparse scheduling, not source rejection skips.
    pub skipped_positions: u64,
    /// Whole-parser wall seconds (allocation, queries and output), not query-only CPU time.
    pub parse_seconds: f64,
    /// Sum of strictly positive emitted source-model f32 gains, promoted to f64.
    pub useful_saving_bits: f64,
}

/// Optional shared receipt. Parsing merges local counters once, on success.
/// Concurrent snapshots are diagnostic, not a transactional set of counters.
#[derive(Debug, Default)]
pub struct SourceQuickReceipt {
    counters: [AtomicU64; 14],
    useful_saving_bits: AtomicU64,
}

impl SourceQuickReceipt {
    pub fn snapshot(&self) -> SourceQuickStats {
        let [
            calls,
            sparse_calls,
            double_fast_calls,
            cost_ranked_calls,
            preserved_careful_calls,
            queries,
            query_misses,
            unprofitable_queries,
            matches,
            useful_matches,
            long_matches,
            matched_values,
            skipped_positions,
            parse_nanos,
        ] = self
            .counters
            .each_ref()
            .map(|counter| counter.load(Ordering::Relaxed));
        SourceQuickStats {
            calls,
            sparse_calls,
            double_fast_calls,
            cost_ranked_calls,
            preserved_careful_calls,
            queries,
            query_misses,
            unprofitable_queries,
            matches,
            useful_matches,
            long_matches,
            matched_values,
            skipped_positions,
            parse_seconds: parse_nanos as f64 * 1e-9,
            useful_saving_bits: f64::from_bits(self.useful_saving_bits.load(Ordering::Relaxed)),
        }
    }

    fn record(&self, stats: SourceQuickStats) {
        let values = [
            stats.calls,
            stats.sparse_calls,
            stats.double_fast_calls,
            stats.cost_ranked_calls,
            stats.preserved_careful_calls,
            stats.queries,
            stats.query_misses,
            stats.unprofitable_queries,
            stats.matches,
            stats.useful_matches,
            stats.long_matches,
            stats.matched_values,
            stats.skipped_positions,
            (stats.parse_seconds * 1e9) as u64,
        ];
        for (counter, value) in self.counters.iter().zip(values) {
            counter.fetch_add(value, Ordering::Relaxed);
        }
        let _ =
            self.useful_saving_bits
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                    Some((f64::from_bits(bits) + stats.useful_saving_bits).to_bits())
                });
    }

    pub(super) fn record_careful(&self) {
        // This counter records bypass, not activation of the quick matcher.
        self.record(SourceQuickStats {
            preserved_careful_calls: 1,
            ..Default::default()
        });
    }
}

const HASH_BITS: usize = 12;
const BUCKETS: usize = 1 << HASH_BITS;
const EMPTY: u32 = u32::MAX;
const LONG_MATCH: usize = 8;
const MAX_PROBE_STEP: usize = 64;

// Both full-width keys are verified against original values before extension.
fn key<const N: usize>(values: &[u32], position: usize) -> Option<usize> {
    let prefix = values.get(position..)?.get(..N)?;
    let mut hash = 0x9e37_79b9u32 ^ N as u32;
    for &value in prefix {
        hash = (hash ^ value).rotate_left(5).wrapping_mul(0x85eb_ca6b);
    }
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0xc2b2_ae35);
    hash ^= hash >> 13;
    Some((hash >> (32 - HASH_BITS)) as usize)
}

struct DoubleHash<'a> {
    values: &'a [u32],
    maximum: usize,
    short: Vec<[u32; 2]>,
    long: Vec<[u32; 2]>,
    codes: DistanceCodes,
    spatial: [usize; 29],
    spatial_count: usize,
}

impl<'a> DoubleHash<'a> {
    fn new(values: &'a [u32], width: usize, config: &SourceConfig) -> Result<Self> {
        if u32::try_from(values.len()).is_err() {
            return Err(Error::Limit("source quick LZ77 index domain"));
        }
        let mut short = reserved(BUCKETS)?;
        short.resize(BUCKETS, [EMPTY; 2]);
        let mut long = reserved(BUCKETS)?;
        long.resize(BUCKETS, [EMPTY; 2]);
        let mut spatial = [0; 29];
        let mut spatial_count = 0;
        for distance in [
            Some(1),
            width.checked_sub(1),
            Some(width),
            width.checked_add(1),
            width.checked_mul(2),
        ]
        .into_iter()
        .flatten()
        {
            if (1..=WINDOW_SIZE).contains(&distance)
                && !spatial[..spatial_count].contains(&distance)
            {
                spatial[spatial_count] = distance;
                spatial_count += 1;
            }
        }
        if config.quick == SourceQuickMode::DoubleFastCost && width != 0 {
            for code in 0..24 {
                let distance = super::super::special_distance(code, width);
                if (1..=WINDOW_SIZE as i128).contains(&distance) {
                    let distance = distance as usize;
                    if !spatial[..spatial_count].contains(&distance) {
                        spatial[spatial_count] = distance;
                        spatial_count += 1;
                    }
                }
            }
        }
        Ok(Self {
            values,
            maximum: config.max_match_length,
            short,
            long,
            // At most 120 canonical distances; no width-sized auxiliary table.
            codes: DistanceCodes::new(width, WINDOW_SIZE),
            spatial,
            spatial_count,
        })
    }

    fn consider(&self, position: usize, previous: usize, best: &mut Match) {
        if previous >= position || position - previous > WINDOW_SIZE {
            return;
        }
        let limit = self.maximum.min(self.values.len() - position);
        if limit < MIN_MATCH
            || self.values[position..position + MIN_MATCH]
                != self.values[previous..previous + MIN_MATCH]
        {
            return;
        }
        let distance = position - previous;
        let symbol = self.codes.symbol(distance);
        let required = if best.length == 0 {
            MIN_MATCH
        } else {
            best.length + usize::from(symbol >= best.symbol)
        };
        if required > limit
            || self.values[position + required - 1] != self.values[previous + required - 1]
        {
            return;
        }
        // Original-value comparison proves overlapping copies too. No hash
        // equality, fingerprint, or last-value check is accepted as a match.
        let length = MIN_MATCH
            + super::search::fast_match_length(
                &self.values[position + MIN_MATCH..position + limit],
                &self.values[previous + MIN_MATCH..previous + limit],
            );
        if length >= required {
            *best = Match {
                length,
                distance,
                symbol,
            };
        }
    }

    fn find(&mut self, position: usize) -> Match {
        let mut best = Match::default();
        self.origins(position, |previous| {
            self.consider(position, previous, &mut best)
        });
        self.update_range(position, position + 1);
        best
    }

    fn origins(&self, position: usize, mut visit: impl FnMut(usize)) {
        if let Some(hash) = key::<8>(self.values, position) {
            for previous in self.long[hash] {
                if previous != EMPTY {
                    visit(previous as usize);
                }
            }
        }
        if let Some(hash) = key::<3>(self.values, position) {
            for previous in self.short[hash] {
                if previous != EMPTY {
                    visit(previous as usize);
                }
            }
        }
        // Tiny spatial set uses this stream's actual width, even with empty
        // buckets. DistanceCodes is only the canonical symbol mapping.
        for &distance in &self.spatial[..self.spatial_count] {
            if distance <= position {
                visit(position - distance);
            }
        }
    }

    fn update_range(&mut self, begin: usize, end: usize) {
        for position in begin..end {
            if let Some(hash) = key::<3>(self.values, position) {
                self.short[hash] = [position as u32, self.short[hash][0]];
            }
            if let Some(hash) = key::<8>(self.values, position) {
                self.long[hash] = [position as u32, self.long[hash][0]];
            }
        }
    }
}

enum Selector<'a> {
    Fast(Fast<'a, 7>),
    Double(DoubleHash<'a>),
}

impl Selector<'_> {
    fn find(&mut self, position: usize) -> Match {
        match self {
            Self::Fast(matcher) => matcher.find(position),
            Self::Double(matcher) => matcher.find(position),
        }
    }

    fn find_priced(
        &mut self,
        position: usize,
        sums: &[f32],
        penalty: f32,
        require_gain: bool,
    ) -> (Match, bool) {
        match self {
            Self::Double(matcher) => matcher.find_priced(position, sums, penalty, require_gain),
            Self::Fast(_) => unreachable!("cost ranking requires the two-key matcher"),
        }
    }

    fn update_range(&mut self, begin: usize, end: usize) {
        match self {
            Self::Fast(matcher) => matcher.update_range(begin, end),
            Self::Double(matcher) => matcher.update_range(begin, end),
        }
    }
}

#[derive(Default)]
struct SparseProbes {
    misses: usize,
}

impl SparseProbes {
    fn miss(&mut self) -> usize {
        self.misses = self.misses.saturating_add(1);
        (1 + (self.misses >> 5)).min(MAX_PROBE_STEP)
    }

    fn useful_long_match(&mut self) {
        self.misses = 0;
    }
}

pub(super) fn parse(
    input: &[Literal],
    values: &[u32],
    width: usize,
    config: &SourceConfig,
    penalties: &prices::Penalties<'_>,
    sums: &[f32],
    saving: &mut f32,
) -> Result<Vec<Command>> {
    if config.quick == SourceQuickMode::Off {
        return match config.method.fast_capacity() {
            Some(1) => fast_stream::<1>(input, values, width, config, penalties, sums, saving),
            Some(3) => fast_stream::<3>(input, values, width, config, penalties, sums, saving),
            Some(7) | None => {
                fast_stream::<7>(input, values, width, config, penalties, sums, saving)
            }
            Some(15) => fast_stream::<15>(input, values, width, config, penalties, sums, saving),
            Some(31) => fast_stream::<31>(input, values, width, config, penalties, sums, saving),
            _ => unreachable!("source fast bucket capacity"),
        };
    }
    let started = std::time::Instant::now();
    let _profile = crate::profile::scope("lz77.source.quick_seed");
    let mut matcher = if config.quick.double() {
        Selector::Double(DoubleHash::new(values, width, config)?)
    } else {
        Selector::Fast(Fast::<7>::new(values, width, config)?)
    };
    let mut output = reserved(input.len())?;
    let mut probes = SparseProbes::default();
    let mut stats = SourceQuickStats {
        calls: 1,
        sparse_calls: u64::from(config.quick.sparse()),
        double_fast_calls: u64::from(config.quick.double()),
        cost_ranked_calls: u64::from(config.quick == SourceQuickMode::DoubleFastCost),
        ..Default::default()
    };
    let mut position = 0;
    while position < input.len() {
        if input.len() - position < MIN_MATCH {
            output.push(Command::Literal(input[position]));
            position += 1;
            continue;
        }
        let (matched, priced_legal) = if config.quick == SourceQuickMode::DoubleFastCost {
            matcher.find_priced(
                position,
                sums,
                penalties.get(input[position].context),
                config.method.runtime_cost_comparison(),
            )
        } else {
            (matcher.find(position), false)
        };
        stats.queries += 1;
        let mut useful = false;
        let consumed = if matched.length < MIN_MATCH {
            stats.unprofitable_queries += u64::from(priced_legal);
            output.push(Command::Literal(input[position]));
            1
        } else {
            let raw = sums[position + matched.length] - sums[position];
            let cost = static_cost_with_penalty(matched, penalties.get(input[position].context));
            useful = raw > cost;
            stats.unprofitable_queries += u64::from(!useful);
            // Identical source seed gate, rejection skip and f32 savings order.
            if config.method.runtime_cost_comparison() && cost > raw {
                let count = if raw < matched.length as f32 && matched.length > MIN_MATCH {
                    matched.length - MIN_MATCH + 1
                } else {
                    1
                };
                output.extend(
                    input[position..position + count]
                        .iter()
                        .copied()
                        .map(Command::Literal),
                );
                count
            } else {
                *saving += raw - cost;
                output.push(matched.command(input[position].context));
                stats.matches += 1;
                stats.matched_values += matched.length as u64;
                stats.useful_matches += u64::from(useful);
                if useful {
                    stats.useful_saving_bits += f64::from(raw - cost);
                }
                if useful && matched.length >= LONG_MATCH {
                    probes.useful_long_match();
                    stats.long_matches += 1;
                }
                matched.length
            }
        };
        stats.query_misses += u64::from(!useful);
        let mut end = position + consumed;
        if config.quick.sparse() && !useful {
            let extra = (probes.miss() - 1).min(input.len() - end);
            output.extend(
                input[end..end + extra]
                    .iter()
                    .copied()
                    .map(Command::Literal),
            );
            stats.skipped_positions += extra as u64;
            end += extra;
        }
        // find inserts its own origin. Insert every consumed/skipped position
        // exactly once, in order, so sparse queries never see future history.
        matcher.update_range(position + 1, end);
        position = end;
    }
    stats.parse_seconds = started.elapsed().as_secs_f64();
    if let Some(receipt) = &config.quick_receipt {
        receipt.record(stats);
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
