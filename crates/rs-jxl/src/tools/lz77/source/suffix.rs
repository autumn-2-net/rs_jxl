// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Search, active ranks and LCP derived from BSD-3-Clause libjxl;
// see docs/PORTING_KERNELS.md. The external Rust utility is Apache-2.0.

//! enc_lz77_suffix.h / enc_lz77_lcp.h at b866b3d. The source's external
//! libsais_int uses the bounded serial adapter in sais.rs and the
//! vendored pure-Rust translation, not a substitute suffix-order algorithm.
//! Source neighbor nominations and strict sliding-window semantics are kept.

use super::super::{MIN_MATCH, Match, special_distance};
use super::distances::Codes;
use super::index::Dominance;
use super::matchers::{filled, window};
use super::sais::suffix_array;
use super::search::{MatchCache, Query, match_length, physical};
use super::{SourceConfig, reserved};
use crate::Result;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceSuffixMode {
    #[default]
    Disabled,
    /// SUFFIX=1: nearest lexicographic earlier positions, filtered by window.
    Adjacent,
    /// SUFFIX=2: nearest ranks still active in the strict sliding window.
    Sliding,
    /// SUFFIX=3: sliding neighbors with Kasai/blocked-RMQ prefix lengths.
    SlidingLcp,
    /// SUFFIX=4: earlier-position neighbors with precomputed prefix lengths.
    AdjacentLcp,
}

impl SourceSuffixMode {
    fn sliding(self) -> bool {
        matches!(self, Self::Sliding | Self::SlidingLcp)
    }
    fn lcp(self) -> bool {
        matches!(self, Self::SlidingLcp | Self::AdjacentLcp)
    }
}

struct ActiveRanks {
    levels: Vec<Vec<u64>>,
}

impl ActiveRanks {
    fn new(mut size: usize) -> Result<Self> {
        let mut levels = Vec::new();
        loop {
            size = size.div_ceil(64);
            levels.push(filled(size.max(1), 0)?);
            if size <= 1 {
                break;
            }
        }
        Ok(Self { levels })
    }

    fn set(&mut self, mut rank: usize, active: bool) {
        for level in &mut self.levels {
            let word = rank / 64;
            let mask = 1u64 << (rank % 64);
            let was_empty = level[word] == 0;
            if active {
                level[word] |= mask;
            } else {
                level[word] &= !mask;
            }
            if was_empty == (level[word] == 0) {
                break;
            }
            rank = word;
        }
    }

    fn before(&self, level: usize, rank: usize) -> Option<usize> {
        if rank == 0 || level == self.levels.len() {
            return None;
        }
        let word = (rank - 1) / 64;
        let present = self.levels[level][word] & (u64::MAX >> (63 - (rank - 1) % 64));
        if present != 0 {
            return Some(word * 64 + 63 - present.leading_zeros() as usize);
        }
        let previous = self.before(level + 1, word)?;
        Some(previous * 64 + 63 - self.levels[level][previous].leading_zeros() as usize)
    }

    fn at_or_after(&self, level: usize, rank: usize) -> Option<usize> {
        let words = self.levels.get(level)?;
        let word = rank / 64;
        let present = words.get(word)? & (u64::MAX << (rank % 64));
        if present != 0 {
            return Some(word * 64 + present.trailing_zeros() as usize);
        }
        let next = self.at_or_after(level + 1, word + 1)?;
        Some(next * 64 + words[next].trailing_zeros() as usize)
    }
}

struct Lcp {
    lcp: Vec<u32>,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    minima: Vec<Vec<u32>>,
}

impl Lcp {
    const BLOCK: usize = 32;

    fn new(values: &[u32], suffixes: &[u32], ranks: &[u32]) -> Result<Self> {
        let size = values.len();
        let mut lcp = filled(size, 0u32)?;
        let mut common = 0;
        for position in 0..size {
            let rank = ranks[position] as usize;
            if rank == 0 {
                common = 0;
                continue;
            }
            let previous = suffixes[rank - 1] as usize;
            while position + common < size
                && previous + common < size
                && values[position + common] == values[previous + common]
            {
                common += 1;
            }
            lcp[rank] = common as u32;
            common = common.saturating_sub(1);
        }
        let mut prefix = filled(size, 0u32)?;
        let mut suffix = filled(size, 0u32)?;
        let blocks = size.div_ceil(Self::BLOCK);
        let mut minima = vec![filled(blocks, u32::MAX)?];
        for (block, minimum) in minima[0].iter_mut().enumerate() {
            let begin = block * Self::BLOCK;
            let end = size.min(begin + Self::BLOCK);
            let mut low = u32::MAX;
            for i in begin..end {
                low = low.min(lcp[i]);
                prefix[i] = low;
            }
            *minimum = low;
            low = u32::MAX;
            for i in (begin..end).rev() {
                low = low.min(lcp[i]);
                suffix[i] = low;
            }
        }
        let mut step = 1;
        while step * 2 <= blocks {
            let previous = minima.last().unwrap();
            let mut next = reserved(blocks - step * 2 + 1)?;
            for i in 0..blocks - step * 2 + 1 {
                next.push(previous[i].min(previous[i + step]));
            }
            minima.push(next);
            step *= 2;
        }
        Ok(Self {
            lcp,
            prefix,
            suffix,
            minima,
        })
    }

    fn between(&self, a: usize, b: usize) -> u32 {
        if a == b {
            return u32::MAX;
        }
        let (a, b) = (a.min(b) + 1, a.max(b));
        if a / Self::BLOCK == b / Self::BLOCK {
            return *self.lcp[a..=b].iter().min().unwrap();
        }
        let mut low = self.suffix[a].min(self.prefix[b]);
        let (begin, end) = (a / Self::BLOCK + 1, b / Self::BLOCK);
        if end > begin {
            let level = (end - begin).ilog2() as usize;
            let width = 1 << level;
            low = low
                .min(self.minima[level][begin])
                .min(self.minima[level][end - width]);
        }
        low
    }
}

pub(super) struct SuffixIndex<'a> {
    values: &'a [u32],
    zeros: Vec<u32>,
    before: Vec<u32>,
    after: Vec<u32>,
    recent: Vec<u32>,
    before_length: Vec<u32>,
    after_length: Vec<u32>,
    anchors: Vec<usize>,
    codes: Codes,
    window: usize,
    maximum: usize,
    width: usize,
}

impl<'a> SuffixIndex<'a> {
    pub(super) fn new(
        values: &'a [u32],
        width: usize,
        config: &SourceConfig,
    ) -> Result<Option<Self>> {
        Self::build(values, width, window(values.len()), config)
    }

    fn build(
        values: &'a [u32],
        width: usize,
        window: usize,
        config: &SourceConfig,
    ) -> Result<Option<Self>> {
        let size = values.len();
        if config.suffix_mode == SourceSuffixMode::Disabled
            || size == 0
            || size > 1 << 24
            || window == 0
            || window > u32::MAX as usize
        {
            return Ok(None);
        }
        let Some(suffixes) = suffix_array(values, config.suffix_scratch_bytes)? else {
            return Ok(None);
        };
        let mut ranks = filled(size, 0u32)?;
        for (rank, &p) in suffixes.iter().enumerate() {
            ranks[p as usize] = rank as u32;
        }
        let mut before = filled(size, u32::MAX)?;
        let mut after = filled(size, u32::MAX)?;
        if config.suffix_mode.sliding() {
            let mut active = ActiveRanks::new(size)?;
            for position in 0..size {
                if position != 0 {
                    active.set(ranks[position - 1] as usize, true);
                }
                if position >= window {
                    active.set(ranks[position - window] as usize, false);
                }
                let rank = ranks[position] as usize;
                if let Some(previous) = active.before(0, rank) {
                    before[position] = suffixes[previous];
                }
                if let Some(next) = active.at_or_after(0, rank + 1) {
                    after[position] = suffixes[next];
                }
            }
        } else {
            let mut stack: Vec<u32> = reserved(size)?;
            for &position in &suffixes {
                while stack.last().is_some_and(|&p| p > position) {
                    after[stack.pop().unwrap() as usize] = position;
                }
                if let Some(&previous) = stack.last() {
                    before[position as usize] = previous;
                }
                stack.push(position);
            }
        }
        let (mut before_length, mut after_length) = (Vec::new(), Vec::new());
        if config.suffix_mode.lcp() {
            let lcp = Lcp::new(values, &suffixes, &ranks)?;
            before_length = filled(size, 0)?;
            after_length = filled(size, 0)?;
            for p in 0..size {
                if before[p] != u32::MAX {
                    before_length[p] =
                        lcp.between(ranks[p] as usize, ranks[before[p] as usize] as usize);
                }
                if after[p] != u32::MAX {
                    after_length[p] =
                        lcp.between(ranks[p] as usize, ranks[after[p] as usize] as usize);
                }
            }
        }
        let mut recent = filled(size, u32::MAX)?;
        let mut heads = filled(1 << 16, u32::MAX)?;
        for p in 0..size.saturating_sub(2) {
            let key = values[p].wrapping_mul(0x85eb_ca6b)
                ^ values[p + 1].wrapping_mul(0xc2b2_ae35)
                ^ values[p + 2].wrapping_mul(0x9e37_79b9);
            let key = (key ^ (key >> 16)) as usize & 65535;
            recent[p] = heads[key];
            heads[key] = p as u32;
        }
        let mut zeros = filled(size, 0u32)?;
        let mut zero = 0;
        for p in (0..size).rev() {
            zero = if values[p] == 0 {
                (zero + 1).min(window as u32)
            } else {
                0
            };
            zeros[p] = zero;
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
        Ok(Some(Self {
            values,
            zeros,
            before,
            after,
            recent,
            before_length,
            after_length,
            anchors,
            codes: Codes::new(width, true)?,
            window,
            maximum: config.max_match_length,
            width,
        }))
    }

    #[cfg(test)]
    pub(super) fn find(
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
        let limit = query
            .maximum
            .min(self.maximum)
            .min(self.values.len().saturating_sub(position));
        if limit < MIN_MATCH {
            return;
        }
        cache.advance(position);
        // Source suffix queries deliberately ignore the distance bound.
        if let Some(carry) = query.carry {
            carry.visit(position, MIN_MATCH, limit, |length, symbol| {
                let bits = cost(symbol);
                visit(Match {
                    length,
                    symbol,
                    distance: physical(symbol, self.width),
                });
                frontier.add(length, symbol, bits);
            });
        }
        let mut candidate = |previous: u32, known: Option<u32>| {
            let previous = previous as usize;
            if previous >= position
                || position - previous >= self.window
                || self.values[previous] != self.values[position]
                || self.values[previous + MIN_MATCH - 1] != self.values[position + MIN_MATCH - 1]
            {
                return;
            }
            let distance = position - previous;
            let symbol = self.codes.symbol(distance);
            let bits = cost(symbol);
            let required = frontier.required(bits, symbol);
            if required > limit
                || self.values[position + required - 1] != self.values[previous + required - 1]
            {
                return;
            }
            let length = if let Some(known) = known {
                limit.min(known as usize)
            } else {
                let prefix = limit
                    .min(self.zeros[position].min(self.zeros[previous]) as usize)
                    .max(cache.prefix(distance, limit));
                prefix
                    + match_length(
                        &self.values[position + prefix..position + limit],
                        &self.values[previous + prefix..previous + limit],
                    )
            };
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
        };
        if query.hint_distance != 0 && query.hint_distance <= position {
            candidate((position - query.hint_distance) as u32, None);
        }
        candidate(
            self.before[position],
            self.before_length.get(position).copied(),
        );
        candidate(
            self.after[position],
            self.after_length.get(position).copied(),
        );
        let mut previous = self.recent[position];
        for _ in 0..8 {
            if previous as usize >= position || position - previous as usize >= self.window {
                break;
            }
            candidate(previous, None);
            previous = self.recent[previous as usize];
        }
        for &distance in &self.anchors {
            if distance <= position {
                candidate((position - distance) as u32, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_carry_emits_first_but_source_distance_bound_is_ignored() {
        use super::super::search::{DistanceBound, FrontierCarry};
        let values = vec![7; 200];
        let config = SourceConfig {
            suffix_mode: SourceSuffixMode::SlidingLcp,
            ..SourceConfig::default()
        };
        let index = SuffixIndex::new(&values, 0, &config).unwrap().unwrap();
        let mut frontier = Dominance::new();
        frontier.add(40, 0, 0.0);
        let mut carry = FrontierCarry::new();
        carry.remember(99, &frontier);
        let calls = std::cell::Cell::new(0);
        let mut matches = Vec::new();
        index.find_query(
            100,
            Query {
                maximum: 20,
                result: Some(&mut frontier),
                bound: DistanceBound {
                    cost: 0.0,
                    symbol: 0,
                },
                carry: Some(&carry),
                ..Query::default()
            },
            |_| {
                calls.set(calls.get() + 1);
                0.0
            },
            |m| matches.push((m.length, m.distance)),
            &mut MatchCache::uncached(),
        );
        assert_eq!(matches, [(20, 1)]);
        assert!(calls.get() > 1);
        assert_eq!(frontier.size, 1);
    }

    #[test]
    fn sais_order_equals_lexicographic_suffix_oracle_exhaustively() {
        for size in 0..=7 {
            for mut code in 0..3usize.pow(size) {
                let values: Vec<_> = (0..size)
                    .map(|_| {
                        let value = (code % 3) as u32;
                        code /= 3;
                        value
                    })
                    .collect();
                let mut expected: Vec<_> = (0..values.len() as u32).collect();
                expected.sort_by(|&a, &b| values[a as usize..].cmp(&values[b as usize..]));
                assert_eq!(
                    suffix_array(&values, usize::MAX).unwrap().unwrap(),
                    expected
                );
            }
        }
        assert_eq!(
            suffix_array(&[0xfffff, 0, 0xfffff], usize::MAX)
                .unwrap()
                .unwrap(),
            [1, 2, 0]
        );
    }

    #[test]
    fn active_rank_hierarchy_crosses_word_and_level_boundaries() {
        let size = 8201;
        let mut active = ActiveRanks::new(size).unwrap();
        let mut expected = std::collections::BTreeSet::new();
        for rank in [0, 1, 63, 64, 65, 4095, 4096, 8199, 8200] {
            active.set(rank, true);
            expected.insert(rank);
        }
        for remove in [None, Some(63), Some(4096), Some(0), Some(8200)] {
            if let Some(rank) = remove {
                active.set(rank, false);
                expected.remove(&rank);
            }
            for rank in 0..size {
                assert_eq!(
                    active.before(0, rank),
                    expected.range(..rank).next_back().copied()
                );
                assert_eq!(
                    active.at_or_after(0, rank + 1),
                    expected.range(rank + 1..).next().copied()
                );
            }
        }
    }

    #[test]
    fn blocked_lcp_equals_direct_comparison_at_every_rank_pair() {
        let values: Vec<_> = (0..151)
            .map(|i| if i % 53 < 37 { 0 } else { i % 7 })
            .collect();
        let suffixes = suffix_array(&values, usize::MAX).unwrap().unwrap();
        let mut ranks = vec![0; values.len()];
        for (rank, &p) in suffixes.iter().enumerate() {
            ranks[p as usize] = rank as u32;
        }
        let lcp = Lcp::new(&values, &suffixes, &ranks).unwrap();
        for a in 0..values.len() {
            for b in 0..values.len() {
                let expected = if a == b {
                    u32::MAX
                } else {
                    values[a..]
                        .iter()
                        .zip(&values[b..])
                        .take_while(|(x, y)| x == y)
                        .count() as u32
                };
                assert_eq!(lcp.between(ranks[a] as usize, ranks[b] as usize), expected);
            }
        }
    }

    #[test]
    fn neighbors_match_rank_oracle_and_lcp_preserves_candidate_order() {
        let values: Vec<_> = (0..257)
            .map(|i| if i % 61 < 17 { 0 } else { i % 13 })
            .collect();
        let suffixes = suffix_array(&values, usize::MAX).unwrap().unwrap();
        let mut ranks = vec![0; values.len()];
        for (rank, &p) in suffixes.iter().enumerate() {
            ranks[p as usize] = rank;
        }
        for mode in [SourceSuffixMode::Adjacent, SourceSuffixMode::Sliding] {
            let mut config = SourceConfig {
                suffix_mode: mode,
                ..SourceConfig::default()
            };
            let index = SuffixIndex::build(&values, 13, 64, &config)
                .unwrap()
                .unwrap();
            config.suffix_mode = if mode.sliding() {
                SourceSuffixMode::SlidingLcp
            } else {
                SourceSuffixMode::AdjacentLcp
            };
            let known = SuffixIndex::build(&values, 13, 64, &config)
                .unwrap()
                .unwrap();
            let mut cache = MatchCache::new(64).unwrap();
            for p in 0..values.len() {
                let allowed =
                    |q: &&u32| (**q as usize) < p && (!mode.sliding() || p - (**q as usize) < 64);
                let before = suffixes[..ranks[p]]
                    .iter()
                    .rev()
                    .find(allowed)
                    .copied()
                    .unwrap_or(u32::MAX);
                let after = suffixes[ranks[p] + 1..]
                    .iter()
                    .find(allowed)
                    .copied()
                    .unwrap_or(u32::MAX);
                assert_eq!((index.before[p], index.after[p]), (before, after));
                let mut a = Vec::new();
                let mut b = Vec::new();
                index.find(
                    p,
                    75,
                    |d| (d % 7) as f32,
                    |m| a.push((m.length, m.distance, m.symbol)),
                    &mut MatchCache::uncached(),
                );
                known.find(
                    p,
                    75,
                    |d| (d % 7) as f32,
                    |m| b.push((m.length, m.distance, m.symbol)),
                    &mut cache,
                );
                assert_eq!(a, b, "position={p} mode={mode:?}");
            }
        }
    }

    #[test]
    fn unsupported_alphabets_and_scratch_limits_fall_back_before_building() {
        let mut config = SourceConfig {
            suffix_mode: SourceSuffixMode::SlidingLcp,
            ..SourceConfig::default()
        };
        assert!(SuffixIndex::new(&[1 << 20], 0, &config).unwrap().is_none());
        assert!(SuffixIndex::new(&[], 0, &config).unwrap().is_none());
        config.suffix_scratch_bytes = 0;
        assert!(SuffixIndex::new(&[0; 100], 0, &config).unwrap().is_none());
    }
}
