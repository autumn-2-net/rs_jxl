// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77.cc at b866b3defec13494bfbaf705db93581cda118e37:
//! LZ77HashMap (exact 1/3/7/15/31 rings), HashChain (including zero-run chains),
//! and LZ77BucketMatcher short32/128 plus long8/32, full-prefix and history modes.
//! No proxy capacity substitutions.
//! FAST_RANGES selects repeated insertion and eight-value extension separately;
//! explicit prefix modes select scalar tags or native SIMD. Optimal chains
//! alone enable the verified cache with the source seed
//! compression gate; fast/careful seeds and short32/long8 buckets are uncached.

use super::super::dp::MatchCache;
use super::super::{
    DistanceCodes, FilterPolicy, MIN_MATCH, Match, PrefixFilter, RecentBucket, SpecialPrefixGroup,
    WINDOW_SIZE, group_distances, prefix_hash, reserved, special_distance,
};
use super::{
    SourceBucketMode, SourceChainCache, SourceConfig, SourceFastPrefix, SourceFastRanges,
    SourceLz77Method,
};
use crate::Result;

pub(super) fn hash(values: &[u32], position: usize) -> u32 {
    // C++ GetHash returns zero for an incomplete tail, which still participates
    // in HashChain::Update and skipped-position updates in the fast matcher.
    prefix_hash(values, position).unwrap_or(0)
}

pub(super) fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut values = reserved(count)?;
    values.resize(count, value);
    Ok(values)
}

pub(super) fn window(size: usize) -> usize {
    size.clamp(1, WINDOW_SIZE).next_power_of_two()
}

// enc_lz77_parse.h::PreferLZ77MatchCaching. Seed tokens are wire tokens, so a
// match contributes both its length token and its distance token.
fn prefer_match_caching(tokens: usize, seed_tokens: usize) -> bool {
    tokens >= 4096 && seed_tokens != 0 && seed_tokens <= tokens / 16
}

#[derive(Clone)]
struct Ring<const N: usize> {
    positions: [usize; N],
    next: usize,
    size: usize,
}

impl<const N: usize> Ring<N> {
    fn new() -> Self {
        Self {
            positions: [0; N],
            next: 0,
            size: 0,
        }
    }
    fn insert(&mut self, position: usize) {
        self.positions[self.next] = position;
        self.next = (self.next + 1) % N;
        self.size = (self.size + 1).min(N);
    }
    // LZ77InsertRepeatedBucket: preserve physical ring rotation as well as the
    // retained entries because LZ77HashMap searches physical slots in order.
    fn repeated(&mut self, begin: usize, end: usize) {
        let count = end - begin;
        let retained = count.min(N);
        let first = (self.next + count - retained) % N;
        for i in 0..retained {
            self.positions[(first + i) % N] = end - retained + i;
        }
        self.next = (self.next + count) % N;
        self.size = (self.size + count).min(N);
    }
}

pub(super) struct Fast<'a, const N: usize> {
    values: &'a [u32],
    maximum: usize,
    buckets: Vec<Ring<N>>,
    codes: super::distances::FastCodes,
    spatial: Vec<(usize, u32)>,
    groups: Vec<SpecialPrefixGroup>,
    prefix: PrefixFilter<'a>,
    prefix_simd: bool,
    ranges: bool,
    fast_ranges: SourceFastRanges,
}

impl<'a, const N: usize> Fast<'a, N> {
    pub(super) fn new(values: &'a [u32], width: usize, config: &SourceConfig) -> Result<Self> {
        let codes = super::distances::FastCodes::new(width)?;
        let mut spatial = Vec::new();
        if width != 0 {
            for symbol in 0..if N <= 3 { 24 } else { 120 } {
                let distance = special_distance(symbol, width);
                if distance <= WINDOW_SIZE as i128 {
                    spatial.push((distance as usize, symbol as u32));
                }
            }
        }
        if config.range_search {
            spatial.sort_by_key(|&(distance, _)| distance);
        }
        let mut distances: Vec<_> = spatial.iter().map(|&(distance, _)| distance).collect();
        distances.sort_unstable();
        distances.dedup();
        let groups = group_distances(&distances).expect("positive sorted source distances");
        let explicit = config.fast_prefix.or_else(|| {
            (config.prefix_filter == FilterPolicy::Disabled && config.fast_ranges.extension())
                .then_some(SourceFastPrefix::Simd)
        });
        let (policy, prefix_simd) = match explicit {
            Some(SourceFastPrefix::Disabled) => (FilterPolicy::Disabled, false),
            Some(mode) => (
                if config.range_search {
                    FilterPolicy::Lazy
                } else {
                    FilterPolicy::Eager
                },
                mode == SourceFastPrefix::Simd,
            ),
            None => (config.prefix_filter, true),
        };
        let policy = if spatial.is_empty() {
            FilterPolicy::Disabled
        } else {
            policy
        };
        Ok(Self {
            values,
            maximum: config.max_match_length,
            buckets: filled(1 << 14, Ring::new())?,
            codes,
            spatial,
            groups,
            prefix: PrefixFilter::new(values, policy),
            prefix_simd,
            ranges: config.range_search,
            fast_ranges: config.fast_ranges,
        })
    }

    pub(super) fn find(&mut self, position: usize) -> Match {
        let key = hash(self.values, position) as usize & ((1 << 14) - 1);
        self.prefix
            .begin_query(position)
            .expect("source query in bounds");
        let ring = &self.buckets[key];
        let mut best = Match::default();
        let mut candidates = [(0usize, 0u32); 120];
        let mut count = 0;
        if ring.size == N {
            let safe = position - ring.positions[ring.next];
            if self.ranges {
                if self
                    .spatial
                    .last()
                    .is_some_and(|&(distance, _)| distance > safe)
                {
                    if self.prefix.cached_bytes() != 0 && self.prefix_simd {
                        self.prefix
                            .visit_special(position, safe, MIN_MATCH, &self.groups, |distance| {
                                candidates[count] = (distance, self.codes.symbol(distance));
                                count += 1;
                            })
                            .expect("source spatial groups");
                    } else {
                        // enc_lz77.cc::LZ77HashMap::FindMatch upper_bound path.
                        // Pending-prefix work counts entries, including narrow
                        // width aliases, exactly as source last - first does.
                        let first = self
                            .spatial
                            .partition_point(|&(distance, _)| distance <= safe);
                        let eligible = self.spatial[first..]
                            .partition_point(|&(distance, _)| distance <= position);
                        self.prefix.record_candidate_work(eligible as u64);
                        for &(distance, symbol) in &self.spatial[first..first + eligible] {
                            if self
                                .prefix
                                .may_match(position, position - distance, MIN_MATCH)
                                .expect("source spatial prefix")
                            {
                                candidates[count] = (distance, symbol);
                                count += 1;
                            }
                        }
                    }
                }
            } else {
                for &(distance, symbol) in &self.spatial {
                    if distance <= safe || distance > position {
                        continue;
                    }
                    self.prefix.record_candidate_work(1);
                    if self
                        .prefix
                        .may_match(position, position - distance, MIN_MATCH)
                        .expect("previous candidate")
                    {
                        candidates[count] = (distance, symbol);
                        count += 1;
                    }
                }
            }
        }
        let limit = self.maximum.min(self.values.len() - position);
        let mut consider = |distance: usize, symbol: u32| {
            let previous = position - distance;
            let required = if best.length == 0 {
                MIN_MATCH
            } else {
                best.length + usize::from(symbol >= best.symbol)
            };
            if required > limit
                || self.values[position] != self.values[previous]
                || self.values[position + required - 1] != self.values[previous + required - 1]
            {
                return;
            }
            let length = if self.fast_ranges.extension() {
                super::search::match_length(
                    &self.values[position..position + limit],
                    &self.values[previous..previous + limit],
                )
            } else {
                let mut length = 0;
                while length < limit
                    && self.values[position + length] == self.values[previous + length]
                {
                    length += 1;
                }
                length
            };
            if length >= required {
                best = Match {
                    length,
                    distance,
                    symbol,
                };
            }
        };
        for &(distance, symbol) in &candidates[..count] {
            consider(distance, symbol);
        }
        for &previous in &ring.positions[..ring.size] {
            let distance = position - previous;
            if distance <= WINDOW_SIZE {
                consider(distance, self.codes.symbol(distance));
            }
        }
        self.buckets[key].insert(position);
        best
    }

    pub(super) fn update_range(&mut self, mut begin: usize, end: usize) {
        if !self.fast_ranges.repeated() {
            for position in begin..end {
                let key = hash(self.values, position) as usize & ((1 << 14) - 1);
                self.buckets[key].insert(position);
            }
            return;
        }
        while begin < end {
            let mut run_end = begin + 1;
            let limit = end + (self.values.len() - end).min(2);
            while run_end < limit && self.values[run_end] == self.values[begin] {
                run_end += 1;
            }
            let count = (run_end - begin).saturating_sub(2);
            let key = hash(self.values, begin) as usize & ((1 << 14) - 1);
            if count >= N {
                self.buckets[key].repeated(begin, begin + count);
                begin += count;
            } else {
                self.buckets[key].insert(begin);
                begin += 1;
            }
        }
    }
}

pub(super) struct Chain<'a> {
    values: &'a [u32],
    window: usize,
    maximum: usize,
    limit: usize,
    head: Vec<u32>,
    chain: Vec<u32>,
    val: Vec<u32>,
    zeros: Vec<u32>,
    headz: Vec<u32>,
    chainz: Vec<u32>,
    numzeros: usize,
    codes: DistanceCodes,
    cache: MatchCache,
    #[cfg(test)]
    pub(super) last_visits: usize,
}

impl<'a> Chain<'a> {
    pub(super) fn new(
        values: &'a [u32],
        width: usize,
        limit: usize,
        maximum: usize,
    ) -> Result<Self> {
        let window = window(values.len());
        let mut chain = reserved(window)?;
        chain.extend((0..window).map(|i| i as u32));
        Ok(Self {
            values,
            window,
            maximum,
            limit,
            head: filled(1 << 15, u32::MAX)?,
            chainz: chain.clone(),
            chain,
            val: filled(window, u32::MAX)?,
            zeros: filled(window, 0)?,
            headz: filled(window + 1, u32::MAX)?,
            numzeros: 0,
            codes: DistanceCodes::new(width, WINDOW_SIZE),
            cache: MatchCache::uncached(),
            #[cfg(test)]
            last_visits: 0,
        })
    }

    pub(super) fn update(&mut self, position: usize) {
        let hash = hash(self.values, position) as usize & ((1 << 15) - 1);
        let slot = position & (self.window - 1);
        self.val[slot] = hash as u32;
        if self.head[hash] != u32::MAX {
            self.chain[slot] = self.head[hash];
        }
        self.head[hash] = slot as u32;
        if position > 0 && self.values[position] != self.values[position - 1] {
            self.numzeros = 0;
        }
        let end = (position + self.window).min(self.values.len());
        if self.numzeros > 0 {
            if !(self.numzeros >= self.window
                && self.values[end - 1] == 0
                && end == position + self.window)
            {
                self.numzeros -= 1;
            }
        } else {
            while position + self.numzeros < end && self.values[position + self.numzeros] == 0 {
                self.numzeros += 1;
            }
        }
        self.zeros[slot] = self.numzeros as u32;
        if self.headz[self.numzeros] != u32::MAX {
            self.chainz[slot] = self.headz[self.numzeros];
        }
        self.headz[self.numzeros] = slot as u32;
    }

    pub(super) fn update_range(&mut self, begin: usize, end: usize) {
        for position in begin..end {
            self.update(position);
        }
    }

    pub(super) fn find_matches(
        &mut self,
        position: usize,
        keep_short: bool,
        mut visit: impl FnMut(Match),
    ) {
        let slot = position & (self.window - 1);
        let hash = self.val[slot];
        let mut previous_slot = self.chain[slot] as usize;
        let (mut previous_distance, mut best, mut visits) = (0usize, 0usize, 0usize);
        let limit = self.maximum.min(self.values.len() - position);
        loop {
            let distance = slot.wrapping_sub(previous_slot) & (self.window - 1);
            if distance < previous_distance || distance > position.min(WINDOW_SIZE) {
                break;
            }
            previous_distance = distance;
            let mut length = 0;
            if distance > 0 {
                let prefix = if self.numzeros > 3 {
                    (self.numzeros - 1)
                        .min(self.zeros[previous_slot] as usize)
                        .min(limit.saturating_sub(1))
                } else {
                    0
                };
                length =
                    self.cache
                        .length_with_prefix(self.values, position, distance, limit, prefix);
                if length >= MIN_MATCH && (keep_short || length >= best.saturating_sub(2)) {
                    visit(Match {
                        length,
                        distance,
                        symbol: self.codes.symbol(distance),
                    });
                    best = best.max(length);
                }
            }
            visits += 1;
            if visits >= self.limit {
                break;
            }
            if self.numzeros >= 3 && length > self.numzeros {
                let next = self.chainz[previous_slot] as usize;
                if previous_slot == next {
                    break;
                }
                previous_slot = next;
                if self.zeros[previous_slot] as usize != self.numzeros {
                    break;
                }
            } else {
                let next = self.chain[previous_slot] as usize;
                if previous_slot == next {
                    break;
                }
                previous_slot = next;
                if self.val[previous_slot] != hash {
                    break;
                }
            }
        }
        #[cfg(test)]
        {
            self.last_visits = visits;
        }
    }

    pub(super) fn find(&mut self, position: usize) -> Match {
        let mut best = Match {
            length: 1,
            ..Match::default()
        };
        self.find_matches(position, false, |candidate| {
            if candidate.length > best.length
                || (candidate.length == best.length && candidate.symbol < best.symbol)
            {
                best = candidate;
            }
        });
        best
    }
}

// enc_lz77_bucket.h::LZ77BucketHistory. Sampling is by eviction count, not
// absolute token position; duplicate representatives are visited only once.
#[derive(Clone, Default)]
struct HistoryBucket {
    positions: [usize; 12],
    count: u32,
}

impl HistoryBucket {
    fn insert(&mut self, position: usize) {
        self.count = self.count.wrapping_add(1);
        for level in 0..3 {
            let shift = 2 + 2 * level;
            if self.count & ((1 << shift) - 1) == 0 {
                self.positions[level * 4 + ((self.count >> shift) & 3) as usize] = position;
            }
        }
    }

    fn visit(&self, mut visit: impl FnMut(usize)) {
        let mut seen = [0; 12];
        let mut visited = 0;
        for level in 0..3 {
            let count = self.count >> (2 + 2 * level);
            for i in 0..count.min(4) {
                let position = self.positions[level * 4 + ((count - i) & 3) as usize];
                if !seen[..visited].contains(&position) {
                    seen[visited] = position;
                    visited += 1;
                    visit(position);
                }
            }
        }
    }
}

pub(super) fn long_hash(values: &[u32], position: usize, short: u32, full: bool) -> u32 {
    let mut key = short
        ^ values[position + 3].wrapping_mul(0x85eb_ca6b)
        ^ values[position + 7].wrapping_mul(0xc2b2_ae35);
    if full {
        key ^= values[position + 4].wrapping_mul(0x9e37_79b9);
        key ^= values[position + 5].wrapping_mul(0x27d4_eb2d);
        key ^= values[position + 6].wrapping_mul(0x1656_67b1);
    }
    key ^ (key >> 16)
}

// Keep the public <=64 fingerprint bucket contract unchanged. Source disables
// fingerprints for the 128-entry short ring but still filters its long ring.
enum ShortBuckets {
    Normal(Vec<RecentBucket<32>>),
    Large(Vec<Ring<128>>),
}

impl ShortBuckets {
    fn new(large: bool) -> Result<Self> {
        if large {
            return Ok(Self::Large(filled(1 << 15, Ring::new())?));
        }
        let mut buckets = reserved(1 << 15)?;
        for _ in 0..1 << 15 {
            buckets.push(RecentBucket::new().expect("source short32 capacity"));
        }
        Ok(Self::Normal(buckets))
    }

    fn insert(&mut self, position: usize, hash: u32) {
        let slot = hash as usize & 32767;
        match self {
            Self::Normal(buckets) => buckets[slot].insert(position, hash),
            Self::Large(buckets) => buckets[slot].insert(position),
        }
    }

    fn visit(&mut self, hash: u32, filter: bool, mut visit: impl FnMut(usize)) -> usize {
        let slot = hash as usize & 32767;
        match self {
            Self::Normal(buckets) => {
                let bucket = &mut buckets[slot];
                bucket.visit(filter.then_some(hash), MIN_MATCH, visit);
                bucket.oldest_if_full().unwrap_or(0)
            }
            Self::Large(buckets) => {
                let bucket = &buckets[slot];
                for i in 0..bucket.size {
                    visit(bucket.positions[bucket.next.wrapping_sub(i + 1) & 127]);
                }
                if bucket.size == 128 {
                    bucket.positions[bucket.next]
                } else {
                    0
                }
            }
        }
    }
}

pub(super) struct Buckets<'a, const LONG: usize = 8> {
    values: &'a [u32],
    window: usize,
    maximum: usize,
    short: ShortBuckets,
    long: Vec<RecentBucket<LONG>>,
    history: Vec<HistoryBucket>,
    full_prefix: bool,
    hash: u32,
    long_key: u32,
    zero_run: usize,
    zeros: Vec<usize>,
    anchors: Vec<usize>,
    codes: super::distances::Codes,
    fingerprints: bool,
    cache: MatchCache,
}

impl<'a, const LONG: usize> Buckets<'a, LONG> {
    pub(super) fn new(values: &'a [u32], width: usize, config: &SourceConfig) -> Result<Self> {
        let window = window(values.len());
        let mode = config.bucket_route().mode;
        let short = ShortBuckets::new(mode == SourceBucketMode::Short128)?;
        let mut long = reserved(1 << 15)?;
        for _ in 0..1 << 15 {
            long.push(RecentBucket::new().expect("source long capacity"));
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
            values,
            window,
            maximum: config.max_match_length,
            short,
            long,
            history: if matches!(mode, SourceBucketMode::History | SourceBucketMode::WideKey) {
                filled(1 << 15, HistoryBucket::default())?
            } else {
                Vec::new()
            },
            full_prefix: matches!(
                mode,
                SourceBucketMode::WideKey
                    | SourceBucketMode::WideRing
                    | SourceBucketMode::PreciseRing
            ),
            hash: 0,
            long_key: 0,
            zero_run: 0,
            zeros: filled(window, 0)?,
            anchors,
            codes: super::distances::Codes::new(width, config.bucket_distances)?,
            fingerprints: config.prefix_filter != FilterPolicy::Disabled,
            cache: MatchCache::uncached(),
        })
    }

    pub(super) fn update(&mut self, position: usize) {
        self.hash = hash(self.values, position);
        self.short.insert(position, self.hash);
        self.long_key = self.hash;
        if self.values.len() - position >= 8 {
            self.long_key = long_hash(self.values, position, self.hash, self.full_prefix);
            let slot = self.long_key as usize & 32767;
            if !self.history.is_empty()
                && let Some(evicted) = self.long[slot].oldest_if_full()
            {
                self.history[slot].insert(evicted);
            }
            // The long-key ring's filter is the MANDATORY three-token hash.
            self.long[self.long_key as usize & 32767].insert(position, self.hash);
        }
        let end = (position + self.window).min(self.values.len());
        if self.zero_run != 0 {
            if self.zero_run < self.window
                || end != position + self.window
                || self.values[end - 1] != 0
            {
                self.zero_run -= 1;
            }
        } else {
            while position + self.zero_run < end && self.values[position + self.zero_run] == 0 {
                self.zero_run += 1;
            }
        }
        self.zeros[position & (self.window - 1)] = self.zero_run;
    }

    fn candidates(&mut self, position: usize) -> ([usize; 196], usize) {
        // Bound all source modes: short128 + long32 + history12 + anchors24.
        let mut candidates = [0; 196];
        let mut count = 0;
        let tag = self.fingerprints.then_some(self.hash);
        let oldest = self.short.visit(self.hash, self.fingerprints, |p| {
            candidates[count] = p;
            count += 1;
        });
        if self.values.len() - position >= 8 {
            self.long[self.long_key as usize & 32767].visit(tag, MIN_MATCH, |p| {
                if p < oldest {
                    candidates[count] = p;
                    count += 1;
                }
            });
            if !self.history.is_empty() {
                self.history[self.long_key as usize & 32767].visit(|p| {
                    if p < oldest {
                        candidates[count] = p;
                        count += 1;
                    }
                });
            }
        }
        for &distance in &self.anchors {
            if distance <= position && position - distance < oldest {
                candidates[count] = position - distance;
                count += 1;
            }
        }
        (candidates, count)
    }

    pub(super) fn find_matches(
        &mut self,
        position: usize,
        keep_short: bool,
        mut visit: impl FnMut(Match),
    ) {
        let (candidates, count) = self.candidates(position);
        let limit = self.maximum.min(self.values.len() - position);
        let mut best: usize = 0;
        for &previous in &candidates[..count] {
            if previous >= position || position - previous >= self.window {
                continue;
            }
            let required = if keep_short {
                MIN_MATCH
            } else {
                MIN_MATCH.max(best.saturating_sub(2))
            };
            if required > limit
                || self.values[position] != self.values[previous]
                || self.values[position + required - 1] != self.values[previous + required - 1]
            {
                continue;
            }
            let distance = position - previous;
            let prefix = self
                .zero_run
                .min(self.zeros[previous & (self.window - 1)])
                .min(limit);
            let length =
                self.cache
                    .length_with_prefix(self.values, position, distance, limit, prefix);
            if length < required {
                continue;
            }
            visit(Match {
                length,
                distance,
                symbol: self.codes.symbol(distance),
            });
            best = best.max(length);
        }
    }

    pub(super) fn find_cost_matches(
        &mut self,
        position: usize,
        cost: impl Fn(u32) -> f32,
        mut visit: impl FnMut(Match),
    ) {
        let limit = self.maximum.min(self.values.len() - position);
        if limit < MIN_MATCH {
            return;
        }
        let (candidates, count) = self.candidates(position);
        let mut frontier = super::index::Dominance::new();
        for &previous in &candidates[..count] {
            if previous >= position
                || position - previous >= self.window
                || self.values[position] != self.values[previous]
                || self.values[position + MIN_MATCH - 1] != self.values[previous + MIN_MATCH - 1]
            {
                continue;
            }
            let distance = position - previous;
            let symbol = self.codes.symbol(distance);
            let bits = cost(symbol);
            let required = frontier.required(bits, symbol);
            if required > limit
                || self.values[position + required - 1] != self.values[previous + required - 1]
            {
                continue;
            }
            let prefix = self
                .zero_run
                .min(self.zeros[previous & (self.window - 1)])
                .min(limit);
            let length =
                self.cache
                    .length_with_prefix(self.values, position, distance, limit, prefix);
            if length < required {
                continue;
            }
            visit(Match {
                length,
                distance,
                symbol,
            });
            frontier.add(length, symbol, bits);
        }
    }
}

pub(super) enum Optimal<'a> {
    Chain(Chain<'a>),
    Buckets(Buckets<'a>),
    WideBuckets(Buckets<'a, 32>),
}

impl<'a> Optimal<'a> {
    pub(super) fn new(
        values: &'a [u32],
        width: usize,
        config: &SourceConfig,
        seed_tokens: usize,
    ) -> Result<Self> {
        if !config.match_cost.range()
            && config.method == SourceLz77Method::Optc256
            && config.bucket_route().enabled
        {
            if matches!(
                config.bucket_route().mode,
                SourceBucketMode::WideRing | SourceBucketMode::PreciseRing
            ) {
                Ok(Self::WideBuckets(Buckets::new(values, width, config)?))
            } else {
                Ok(Self::Buckets(Buckets::new(values, width, config)?))
            }
        } else {
            let limit = config.method.chain_maximum().unwrap_or(256);
            let limit = if config.match_cost.range() {
                limit
            } else {
                // enc_lz77_parse.h::LosslessLZ77ChainLimit: invalid values do
                // not disable search or expand the method's compiled maximum.
                config
                    .chain_limit
                    .filter(|n| (1..=256).contains(n))
                    .map_or(limit, |n| limit.min(n as usize))
            };
            let mut chain = Chain::new(values, width, limit, config.max_match_length)?;
            let cached = match config.chain_match_cache {
                SourceChainCache::Auto => prefer_match_caching(values.len(), seed_tokens),
                SourceChainCache::Disabled => false,
                SourceChainCache::Enabled => true,
            };
            if !config.match_cost.range() && cached {
                chain.cache = MatchCache::new(chain.window)?;
            }
            Ok(Self::Chain(chain))
        }
    }
    #[cfg(test)]
    pub(super) fn match_cache_enabled(&self) -> bool {
        match self {
            Self::Chain(m) => m.cache.is_enabled(),
            Self::Buckets(m) => m.cache.is_enabled(),
            Self::WideBuckets(m) => m.cache.is_enabled(),
        }
    }
    pub(super) fn update(&mut self, position: usize) {
        match self {
            Self::Chain(m) => m.update(position),
            Self::Buckets(m) => m.update(position),
            Self::WideBuckets(m) => m.update(position),
        }
    }
    pub(super) fn find_matches(
        &mut self,
        position: usize,
        keep_short: bool,
        visit: impl FnMut(Match),
    ) {
        match self {
            Self::Chain(m) => m.find_matches(position, keep_short, visit),
            Self::Buckets(m) => m.find_matches(position, keep_short, visit),
            Self::WideBuckets(m) => m.find_matches(position, keep_short, visit),
        }
    }

    pub(super) fn find_cost_matches(
        &mut self,
        position: usize,
        cost: impl Fn(u32) -> f32,
        visit: impl FnMut(Match),
    ) {
        match self {
            Self::Chain(m) => m.find_matches(position, true, visit),
            Self::Buckets(m) => m.find_cost_matches(position, cost, visit),
            Self::WideBuckets(m) => m.find_cost_matches(position, cost, visit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_chain_limit_and_cache_overrides_keep_source_method_and_route_gates() {
        let values = [7; 8192];
        for method in [
            SourceLz77Method::Optc1,
            SourceLz77Method::Optc3,
            SourceLz77Method::Optc8,
            SourceLz77Method::Optc256,
        ] {
            let maximum = method.chain_maximum().unwrap();
            for requested in [
                None,
                Some(0),
                Some(1),
                Some(3),
                Some(8),
                Some(255),
                Some(256),
                Some(257),
                Some(u32::MAX),
            ] {
                let config = SourceConfig {
                    method,
                    chain_limit: requested,
                    ..SourceConfig::default()
                };
                let Optimal::Chain(chain) = Optimal::new(&values, 0, &config, 512).unwrap() else {
                    unreachable!()
                };
                let expected = requested
                    .filter(|&n| n > 0 && n <= 256)
                    .map_or(maximum, |n| maximum.min(n as usize));
                assert_eq!(chain.limit, expected);
                assert!(chain.cache.is_enabled());
            }
        }
        for (size, seed, policy, enabled) in [
            (63, 3, SourceChainCache::Auto, false),
            (63, 3, SourceChainCache::Enabled, true),
            (8192, 512, SourceChainCache::Auto, true),
            (8192, 513, SourceChainCache::Auto, false),
            (8192, 0, SourceChainCache::Auto, false),
            (8192, 512, SourceChainCache::Disabled, false),
        ] {
            let config = SourceConfig {
                method: SourceLz77Method::Optc256,
                chain_match_cache: policy,
                ..SourceConfig::default()
            };
            assert_eq!(
                Optimal::new(&values[..size], 0, &config, seed)
                    .unwrap()
                    .match_cache_enabled(),
                enabled
            );
        }
        for match_cost in [
            super::super::SourceMatchCost::RangeReplay,
            super::super::SourceMatchCost::Range,
            super::super::SourceMatchCost::RangeRefit,
        ] {
            let config = SourceConfig {
                method: SourceLz77Method::Optc8,
                match_cost,
                chain_limit: Some(1),
                chain_match_cache: SourceChainCache::Enabled,
                ..SourceConfig::default()
            };
            let Optimal::Chain(chain) = Optimal::new(&values, 0, &config, 1).unwrap() else {
                unreachable!()
            };
            assert_eq!(chain.limit, 8);
            assert!(!chain.cache.is_enabled());
        }
        let config = SourceConfig {
            method: SourceLz77Method::Optc256,
            bucket_matcher: true,
            chain_limit: Some(1),
            chain_match_cache: SourceChainCache::Enabled,
            ..SourceConfig::default()
        };
        let buckets = Optimal::new(&values, 0, &config, 1).unwrap();
        assert!(matches!(buckets, Optimal::Buckets(_)));
        assert!(!buckets.match_cache_enabled());
    }

    #[test]
    fn cost_bucket_dominance_preserves_scalar_best_distance_for_every_length() {
        fn compare<const LONG: usize>(mode: SourceBucketMode) {
            let values: Vec<u32> = (0..513)
                .map(|i| if i < 110 { 0 } else { (i % 17) as u32 })
                .collect();
            let config = SourceConfig {
                bucket_mode: mode,
                bucket_distances: true,
                prefix_filter: FilterPolicy::Eager,
                max_match_length: 97,
                ..SourceConfig::default()
            };
            let mut plain = Buckets::<LONG>::new(&values, 37, &config).unwrap();
            let mut pruned = Buckets::<LONG>::new(&values, 37, &config).unwrap();
            let cost = |symbol| ((symbol * 7) % 13) as f32;
            let mut removed = 0;
            for pos in 0..values.len() {
                plain.update(pos);
                pruned.update(pos);
                let (mut all, mut actual) = (Vec::new(), Vec::new());
                plain.find_matches(pos, true, |m| all.push(tuple(m)));
                pruned.find_cost_matches(pos, cost, |m| actual.push(tuple(m)));
                let mut cursor = 0;
                for proposal in &actual {
                    while all[cursor] != *proposal {
                        cursor += 1;
                    }
                    cursor += 1;
                }
                removed += all.len() - actual.len();
                for length in 3..=97.min(values.len() - pos) {
                    let best = |matches: &[(usize, usize, u32)]| {
                        matches
                            .iter()
                            .filter(|m| m.0 >= length)
                            .min_by(|a, b| cost(a.2).total_cmp(&cost(b.2)).then(a.2.cmp(&b.2)))
                            .map(|m| (cost(m.2), m.2))
                    };
                    assert_eq!(best(&actual), best(&all));
                }
            }
            assert!(removed > 0);
        }
        for mode in [
            SourceBucketMode::Default,
            SourceBucketMode::History,
            SourceBucketMode::WideKey,
            SourceBucketMode::Short128,
        ] {
            compare::<8>(mode);
        }
        compare::<32>(SourceBucketMode::WideRing);
        compare::<32>(SourceBucketMode::PreciseRing);
    }

    #[test]
    fn dense_bucket_distance_symbols_preserve_every_candidate_in_source_order() {
        fn compare<const LONG: usize>(mode: SourceBucketMode, width: usize) {
            let values: Vec<u32> = (0..513)
                .map(|i| if i < 100 { 0 } else { (i % 17) as u32 })
                .collect();
            let config = SourceConfig {
                bucket_mode: mode,
                max_match_length: 129,
                ..SourceConfig::default()
            };
            let mut sparse = Buckets::<LONG>::new(&values, width, &config).unwrap();
            let mut dense = Buckets::<LONG>::new(
                &values,
                width,
                &SourceConfig {
                    bucket_distances: true,
                    ..config
                },
            )
            .unwrap();
            for pos in 0..values.len() {
                sparse.update(pos);
                dense.update(pos);
                let (mut expected, mut actual) = (Vec::new(), Vec::new());
                sparse.find_matches(pos, true, |m| expected.push(tuple(m)));
                dense.find_matches(pos, true, |m| actual.push(tuple(m)));
                assert_eq!(actual, expected);
                for (length, distance, _) in actual {
                    for i in 0..length {
                        assert_eq!(values[pos + i], values[pos + i - distance]);
                    }
                }
            }
        }
        for width in [0, 1, 37, 9362] {
            for mode in [
                SourceBucketMode::Default,
                SourceBucketMode::History,
                SourceBucketMode::WideKey,
                SourceBucketMode::Short128,
            ] {
                compare::<8>(mode, width);
            }
            compare::<32>(SourceBucketMode::WideRing, width);
            compare::<32>(SourceBucketMode::PreciseRing, width);
        }
    }

    #[test]
    fn short128_retains_source_capacity_rotation_and_ignores_short_tags() {
        let mut buckets = ShortBuckets::new(true).unwrap();
        for position in 0..300 {
            let key = if position % 2 == 0 {
                0x1100_0017
            } else {
                0x2200_0017
            };
            buckets.insert(position, key);
            let mut actual = Vec::new();
            let oldest = buckets.visit(0xff00_0017, true, |p| actual.push(p));
            let expected: Vec<_> = (0..=position).rev().take(128).collect();
            assert_eq!(actual, expected);
            assert_eq!(oldest, position.saturating_sub(127));
        }
    }

    #[test]
    fn short128_recovers_three_token_copy_evicted_by_short32() {
        let mut values = vec![7, 8, 9, 1000];
        for i in 0..80 {
            values.extend([7, 8, 10 + i, 2000 + i]);
        }
        // Use exactly matching mandatory prefixes with unique fourth values,
        // so neither a long-prefix ring nor a spatial anchor rescues short32.
        for i in (4..values.len()).step_by(4) {
            values[i + 2] = 9;
        }
        let position = values.len();
        values.extend([7, 8, 9]);
        for mode in [SourceBucketMode::Default, SourceBucketMode::Short128] {
            let config = SourceConfig {
                bucket_mode: mode,
                prefix_filter: FilterPolicy::Eager,
                ..SourceConfig::default()
            };
            let mut buckets = Buckets::<8>::new(&values, 0, &config).unwrap();
            for p in 0..=position {
                buckets.update(p);
            }
            let mut distances = Vec::new();
            buckets.find_matches(position, true, |m| distances.push(m.distance));
            assert_eq!(
                distances.contains(&position),
                mode == SourceBucketMode::Short128
            );
        }
    }

    fn tuple(m: Match) -> (usize, usize, u32) {
        (m.length, m.distance, m.symbol)
    }

    #[test]
    fn history_samples_eviction_ages_and_deduplicates_levels_in_source_order() {
        let mut history = HistoryBucket::default();
        let position = |n: usize| n * 11 + n % 3;
        for evictions in 1..=1024 {
            history.insert(position(evictions));
            let mut expected = Vec::new();
            for stride in [4, 16, 64] {
                let count = evictions / stride;
                for offset in 0..count.min(4) {
                    let p = position((count - offset) * stride);
                    if !expected.contains(&p) {
                        expected.push(p);
                    }
                }
            }
            let mut actual = Vec::new();
            history.visit(|p| actual.push(p));
            assert_eq!(actual, expected, "evictions={evictions}");
            assert!(actual.len() <= 12);
        }
    }

    #[test]
    fn history_recovers_a_long_evicted_match_not_present_in_either_recent_ring() {
        let prefix = [10, 11, 12, 13, 14, 15, 16, 17];
        let mut values = Vec::new();
        for block in 0..200u32 {
            values.extend(prefix);
            values.push(1000 + block);
        }
        let position = values.len();
        values.extend(prefix);
        values.push(1063);
        for bucket_mode in [
            SourceBucketMode::Default,
            SourceBucketMode::History,
            SourceBucketMode::WideKey,
        ] {
            for prefix_filter in [FilterPolicy::Disabled, FilterPolicy::Eager] {
                let config = SourceConfig {
                    bucket_mode,
                    prefix_filter,
                    ..SourceConfig::default()
                };
                let mut matcher = Buckets::<8>::new(&values, 0, &config).unwrap();
                for p in 0..=position {
                    matcher.update(p);
                }
                let mut matches = Vec::new();
                matcher.find_matches(position, true, |m| matches.push(tuple(m)));
                assert_eq!(
                    matches
                        .iter()
                        .any(|&(len, distance, _)| len == 9 && distance == position - 63 * 9),
                    bucket_mode != SourceBucketMode::Default
                );
                assert_eq!(
                    matches.iter().map(|m| m.0).max(),
                    Some(if bucket_mode == SourceBucketMode::Default {
                        8
                    } else {
                        9
                    })
                );
            }
        }
    }

    #[test]
    fn wide_ring_really_retains_32_long_prefix_positions_and_hashes_middle_values() {
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
        let config = SourceConfig {
            bucket_mode: SourceBucketMode::WideRing,
            prefix_filter: FilterPolicy::Eager,
            ..SourceConfig::default()
        };
        let mut narrow = Buckets::<8>::new(&values, 0, &config).unwrap();
        let mut wide = Buckets::<32>::new(&values, 0, &config).unwrap();
        for p in 0..=position {
            narrow.update(p);
            wide.update(p);
        }
        let mut narrow_found = false;
        let mut wide_found = false;
        narrow.find_matches(position, true, |m| {
            narrow_found |= m.distance == position && m.length == 9
        });
        wide.find_matches(position, true, |m| {
            wide_found |= m.distance == position && m.length == 9
        });
        assert!(!narrow_found);
        assert!(wide_found);
        let hash = hash(&prefix, 0);
        for middle in 4..=6 {
            let mut changed = prefix;
            changed[middle] ^= 1234567;
            assert_eq!(
                long_hash(&prefix, 0, hash, false),
                long_hash(&changed, 0, hash, false)
            );
            assert_ne!(
                long_hash(&prefix, 0, hash, true),
                long_hash(&changed, 0, hash, true)
            );
        }
    }

    fn check_ring<const N: usize>() {
        for filled in 0..2 * N + 1 {
            for count in 0..4 * N + 1 {
                let mut scalar = Ring::<N>::new();
                for p in 0..filled {
                    scalar.insert(p);
                }
                let mut bulk = scalar.clone();
                bulk.repeated(filled, filled + count);
                for p in filled..filled + count {
                    scalar.insert(p);
                }
                assert_eq!(bulk.positions, scalar.positions);
                assert_eq!((bulk.next, bulk.size), (scalar.next, scalar.size));
            }
        }
    }

    #[test]
    fn repeated_insertion_preserves_physical_ring_rotation() {
        assert!((1 << 14) * std::mem::size_of::<Ring<31>>() + WINDOW_SIZE + 1 < 8 << 20);
        check_ring::<1>();
        check_ring::<3>();
        check_ring::<7>();
        check_ring::<15>();
        check_ring::<31>();
    }

    fn check_range<const N: usize>(values: &[u32]) {
        let mut scalar = vec![Ring::<N>::new(); 1 << 14];
        for position in 0..values.len() {
            scalar[hash(values, position) as usize & 16383].insert(position);
        }
        for fast_ranges in [
            SourceFastRanges::Disabled,
            SourceFastRanges::RepeatedInsert,
            SourceFastRanges::MatchLength,
            SourceFastRanges::Both,
        ] {
            let config = SourceConfig {
                fast_ranges,
                ..SourceConfig::default()
            };
            let mut matcher = Fast::<N>::new(values, 0, &config).unwrap();
            // Split inside runs, including one- and two-value tail prefixes.
            let mut begin = 0;
            for end in [1, 2, 7, 31, 127, 130, values.len()] {
                let end = end.min(values.len());
                if end > begin {
                    matcher.update_range(begin, end);
                    begin = end;
                }
            }
            for (bulk, scalar) in matcher.buckets.iter().zip(&scalar) {
                assert_eq!(bulk.positions, scalar.positions);
                assert_eq!((bulk.next, bulk.size), (scalar.next, scalar.size));
            }
        }
    }

    #[test]
    fn repeated_range_update_includes_hash_zero_tails() {
        let mut values = vec![0; 130];
        values.extend((0..130).map(|i| i % 7));
        values.extend([u32::MAX; 71]);
        for tail in 0..3 {
            check_range::<1>(&values[..values.len() - tail]);
            check_range::<3>(&values[..values.len() - tail]);
            check_range::<7>(&values[..values.len() - tail]);
            check_range::<15>(&values[..values.len() - tail]);
            check_range::<31>(&values[..values.len() - tail]);
        }
    }

    fn check_fast<const N: usize>(values: &[u32], width: usize, config: &SourceConfig) {
        let mut matcher = Fast::<N>::new(values, width, config).unwrap();
        if config.fast_prefix.is_some() || config.fast_ranges.extension() {
            // Exercise activated source scalar/SIMD tags even on short oracle
            // fixtures. Separate tests check the real feedback threshold.
            matcher
                .prefix
                .record_candidate_work(matcher.prefix.activation_threshold());
        }
        let codes = DistanceCodes::new(width, WINDOW_SIZE);
        let mut history = vec![Vec::<usize>::new(); 1 << 14];
        for position in 0..values.len().saturating_sub(2) {
            let key = hash(values, position) as usize & 16383;
            let bucket = &mut history[key];
            let mut previous = bucket.clone();
            if width != 0 && bucket.len() == N {
                for symbol in 0..if N <= 3 { 24 } else { 120 } {
                    let distance = special_distance(symbol, width);
                    if distance <= position.min(WINDOW_SIZE) as i128
                        && position - (distance as usize) < bucket[0]
                    {
                        previous.push(position - distance as usize);
                    }
                }
            }
            let mut expected = Match::default();
            for p in previous {
                let distance = position - p;
                if distance > WINDOW_SIZE {
                    continue;
                }
                let length = values[position..]
                    .iter()
                    .zip(&values[p..])
                    .take(config.max_match_length)
                    .take_while(|(a, b)| a == b)
                    .count();
                let symbol = codes.symbol(distance);
                if length >= MIN_MATCH
                    && (length > expected.length
                        || (length == expected.length && symbol < expected.symbol))
                {
                    expected = Match {
                        length,
                        distance,
                        symbol,
                    };
                }
            }
            assert_eq!(
                tuple(matcher.find(position)),
                tuple(expected),
                "N={N} width={width} pos={position}"
            );
            if bucket.len() == N {
                bucket.remove(0);
            }
            bucket.push(position);
        }
    }

    #[test]
    fn fast_routes_equal_scalar_source_search_with_narrow_aliases() {
        let mut values: Vec<u32> = (0..311)
            .map(|i| if i % 53 < 24 { i % 6 } else { i * 131 })
            .collect();
        values.extend_from_within(17..117);
        values.extend([0; 101]);
        for width in [0, 1, 2, 37, usize::MAX] {
            for range_search in [false, true] {
                for prefix_filter in [
                    FilterPolicy::Disabled,
                    FilterPolicy::Eager,
                    FilterPolicy::Lazy,
                ] {
                    let config = SourceConfig {
                        range_search,
                        prefix_filter,
                        max_match_length: 79,
                        ..SourceConfig::default()
                    };
                    check_fast::<1>(&values, width, &config);
                    check_fast::<3>(&values, width, &config);
                    check_fast::<7>(&values, width, &config);
                    check_fast::<15>(&values, width, &config);
                    check_fast::<31>(&values, width, &config);
                }
            }
        }
    }

    #[test]
    fn spatial_range_counts_source_aliases_and_skips_an_unused_prefix_cache() {
        let values = [7; 128];
        for width in [1, 2, 37] {
            let config = SourceConfig {
                prefix_filter: FilterPolicy::Lazy,
                ..SourceConfig::default()
            };
            let mut matcher = Fast::<3>::new(&values, width, &config).unwrap();
            matcher.update_range(0, 7);
            let expected = matcher
                .spatial
                .iter()
                .filter(|&&(distance, _)| distance > 3 && distance <= 7)
                .count();
            matcher.find(7);
            assert_eq!(matcher.prefix.candidate_work(), expected as u64);
            assert_eq!(matcher.prefix.cached_bytes(), 0);
        }
        let config = SourceConfig {
            prefix_filter: FilterPolicy::Eager,
            ..SourceConfig::default()
        };
        let matcher = Fast::<3>::new(&values, 0, &config).unwrap();
        assert!(matcher.spatial.is_empty());
        assert_eq!(matcher.prefix.cached_bytes(), 0);
    }

    #[test]
    fn fast_range_modes_and_explicit_prefixes_equal_scalar_search() {
        let mut values: Vec<u32> = (0..193)
            .map(|i| if i % 31 < 9 { u32::MAX } else { 65530 + i % 17 })
            .collect();
        values.extend_from_within(7..104);
        values.extend([0; 67]);
        values.extend([u32::MAX, 0]);
        for fast_ranges in [
            SourceFastRanges::Disabled,
            SourceFastRanges::RepeatedInsert,
            SourceFastRanges::MatchLength,
            SourceFastRanges::Both,
        ] {
            for fast_prefix in [
                None,
                Some(SourceFastPrefix::Disabled),
                Some(SourceFastPrefix::Scalar),
                Some(SourceFastPrefix::Simd),
            ] {
                for range_search in [false, true] {
                    for width in [0, 1, 37] {
                        let config = SourceConfig {
                            fast_ranges,
                            fast_prefix,
                            range_search,
                            max_match_length: 33,
                            ..SourceConfig::default()
                        };
                        check_fast::<1>(&values, width, &config);
                        check_fast::<3>(&values, width, &config);
                        check_fast::<7>(&values, width, &config);
                        check_fast::<15>(&values, width, &config);
                        check_fast::<31>(&values, width, &config);
                    }
                }
            }
        }
    }

    #[test]
    fn source_fast_prefix_implicit_override_and_activation_gates() {
        use crate::tools::lz77_prefix::Activation;
        let values = vec![7; 8193];
        for fast_ranges in [
            SourceFastRanges::Disabled,
            SourceFastRanges::RepeatedInsert,
            SourceFastRanges::MatchLength,
            SourceFastRanges::Both,
        ] {
            for fast_prefix in [
                None,
                Some(SourceFastPrefix::Disabled),
                Some(SourceFastPrefix::Scalar),
                Some(SourceFastPrefix::Simd),
            ] {
                let enabled = match fast_prefix {
                    None => fast_ranges.extension(),
                    Some(mode) => mode != SourceFastPrefix::Disabled,
                };
                for range_search in [false, true] {
                    let config = SourceConfig {
                        fast_ranges,
                        fast_prefix,
                        range_search,
                        ..SourceConfig::default()
                    };
                    let mut matcher = Fast::<3>::new(&values, 37, &config).unwrap();
                    let initial = if !enabled {
                        Activation::Disabled
                    } else if range_search {
                        Activation::Pending
                    } else {
                        Activation::Active
                    };
                    assert_eq!(matcher.prefix.activation(), initial);
                    assert_eq!(
                        matcher.prefix.cached_bytes(),
                        if initial == Activation::Active {
                            values.len()
                        } else {
                            0
                        }
                    );
                    matcher.prefix.record_candidate_work(4095);
                    matcher.find(0);
                    assert_eq!(matcher.prefix.activation(), initial);
                    matcher.prefix.record_candidate_work(1);
                    matcher.find(1);
                    assert_eq!(
                        matcher.prefix.activation(),
                        if enabled {
                            Activation::Active
                        } else {
                            Activation::Disabled
                        }
                    );
                    if enabled {
                        assert_eq!(
                            matcher.prefix_simd,
                            fast_prefix != Some(SourceFastPrefix::Scalar)
                        );
                    }
                    let mut late = Fast::<3>::new(&values, 37, &config).unwrap();
                    late.prefix.record_candidate_work(4096);
                    late.find(values.len() / 2);
                    assert_eq!(
                        late.prefix.activation(),
                        if initial == Activation::Pending {
                            Activation::SkippedLate
                        } else {
                            initial
                        }
                    );
                    let no_spatial = Fast::<3>::new(&values, 0, &config).unwrap();
                    assert_eq!(no_spatial.prefix.activation(), Activation::Disabled);
                    assert_eq!(no_spatial.prefix.cached_bytes(), 0);
                }
            }
        }
        // Preserve the pre-existing explicit FilterPolicy API when no source
        // prefix override is provided, including eager+range-search callers.
        let config = SourceConfig {
            fast_ranges: SourceFastRanges::MatchLength,
            prefix_filter: FilterPolicy::Eager,
            ..SourceConfig::default()
        };
        assert_eq!(
            Fast::<3>::new(&values, 37, &config)
                .unwrap()
                .prefix
                .activation(),
            Activation::Active
        );
    }

    #[test]
    fn source_match_cache_gate_is_per_stream_and_chain_only() {
        for (tokens, seed_tokens, expected) in [
            (0, 0, false),
            (4095, 1, false),
            (4096, 0, false),
            (4096, 1, true),
            (4096, 256, true),
            (4096, 257, false),
            (4111, 257, false),
            (4112, 257, true),
        ] {
            assert_eq!(prefer_match_caching(tokens, seed_tokens), expected);
        }
        let values = vec![1; 4096];
        assert!(
            !Chain::new(&values, 0, 256, usize::MAX)
                .unwrap()
                .cache
                .is_enabled()
        );
        for (method, bucket_matcher) in [
            (SourceLz77Method::Optc8, false),
            (SourceLz77Method::Optc256, false),
            (SourceLz77Method::Optc256, true),
        ] {
            let config = SourceConfig {
                method,
                bucket_matcher,
                ..SourceConfig::default()
            };
            for seed_tokens in [0, 256, 257] {
                let matcher = Optimal::new(&values, 0, &config, seed_tokens).unwrap();
                assert_eq!(
                    matcher.match_cache_enabled(),
                    !bucket_matcher && seed_tokens == 256
                );
            }
        }
    }

    #[test]
    fn cached_chain_preserves_candidate_lengths_order_and_visit_caps() {
        let mut values = vec![0; 131];
        values.extend((0..263).map(|i| u32::MAX - i % 7));
        values.extend_from_within(17..213);
        values.extend([9, 0, 0, 0, 9, 0, 0]);
        for cap in [8, 256] {
            for keep_short in [false, true] {
                let mut plain = Chain::new(&values, 2, cap, 97).unwrap();
                let mut cached = Chain::new(&values, 2, cap, 97).unwrap();
                cached.cache = MatchCache::new(cached.window).unwrap();
                for position in 0..values.len() {
                    plain.update(position);
                    cached.update(position);
                    let mut expected = Vec::new();
                    let mut actual = Vec::new();
                    plain.find_matches(position, keep_short, |m| expected.push(tuple(m)));
                    cached.find_matches(position, keep_short, |m| actual.push(tuple(m)));
                    assert_eq!(
                        actual, expected,
                        "cap={cap} pos={position} short={keep_short}"
                    );
                    assert_eq!(cached.last_visits, plain.last_visits);
                }
            }
        }
    }

    #[test]
    fn fast_three_is_not_a_power_of_two_proxy() {
        let values = [
            11, 22, 33, 91, 11, 22, 33, 92, 11, 22, 33, 93, 11, 22, 33, 91,
        ];
        let config = SourceConfig::default();
        let mut one = Fast::<1>::new(&values, 0, &config).unwrap();
        let mut three = Fast::<3>::new(&values, 0, &config).unwrap();
        one.update_range(0, 12);
        three.update_range(0, 12);
        assert_eq!(one.find(12).length, 3);
        assert_eq!(three.find(12).length, 4);
        three.update_range(13, 16);
        let key = hash(&values, 0) as usize & 16383;
        assert_eq!(three.buckets[key].size, 3);
        assert!(!three.buckets[key].positions.contains(&0));
    }

    #[test]
    fn chain_caps_are_actual_visits_one_three_eight_and_256() {
        let values = vec![1; 600];
        for cap in [1, 3, 8, 256] {
            let mut chain = Chain::new(&values, 0, cap, usize::MAX).unwrap();
            chain.update_range(0, 301);
            let mut distances = Vec::new();
            chain.find_matches(300, true, |m| {
                assert_eq!(m.length, 300);
                distances.push(m.distance);
            });
            assert_eq!(chain.last_visits, cap);
            assert_eq!(distances, (1..=cap).collect::<Vec<_>>());
        }
    }

    #[test]
    fn physical_decoder_limit_is_inclusive_only_for_fast_rings() {
        assert_eq!(window(0), 1);
        assert_eq!(window(17), 32);
        assert_eq!(window(WINDOW_SIZE + 1), WINDOW_SIZE);
        let mut values = vec![0; WINDOW_SIZE + 3];
        values[..3].copy_from_slice(&[17, 19, 23]);
        values[WINDOW_SIZE..].copy_from_slice(&[17, 19, 23]);
        let mut fast = Fast::<1>::new(&values, 0, &SourceConfig::default()).unwrap();
        fast.update_range(0, WINDOW_SIZE);
        assert_eq!(
            tuple(fast.find(WINDOW_SIZE)),
            (3, WINDOW_SIZE, WINDOW_SIZE as u32 - 1)
        );
        let mut chain = Chain::new(&values, 0, 256, usize::MAX).unwrap();
        chain.update_range(0, WINDOW_SIZE + 1);
        chain.find_matches(WINDOW_SIZE, true, |_| {
            panic!("full-window chain slot must alias current")
        });
    }

    #[test]
    fn hash_collision_does_not_become_a_match() {
        let mut seen = vec![None; 1 << 14];
        let (a, b) = (0u32..=16384)
            .find_map(|i| {
                let prefix = [i, i.wrapping_mul(17), i ^ 0x1234];
                let key = hash(&prefix, 0) as usize & 16383;
                if let Some(a) = seen[key] {
                    Some((a, prefix))
                } else {
                    seen[key] = Some(prefix);
                    None
                }
            })
            .unwrap();
        let values = [a, b].concat();
        let mut fast = Fast::<1>::new(&values, 1, &SourceConfig::default()).unwrap();
        fast.find(0);
        assert_eq!(fast.find(3).length, 0);
    }

    #[test]
    fn long_eight_recovers_evicted_short_prefix_and_tags_do_not_change_matches() {
        let original = [11, 22, 33, 1111, 44, 55, 66, 7777];
        let mut values = original.to_vec();
        for i in 0..48 {
            values.extend([11, 22, 33, 2000 + i, 44, 55, 66, 8000 + i]);
        }
        let position = values.len();
        values.extend(original);
        let mut reference = None;
        for prefix_filter in [FilterPolicy::Disabled, FilterPolicy::Eager] {
            let config = SourceConfig {
                prefix_filter,
                ..SourceConfig::default()
            };
            let mut buckets = Buckets::<8>::new(&values, 0, &config).unwrap();
            for p in 0..=position {
                buckets.update(p);
            }
            assert!(buckets.short.visit(buckets.hash, false, |_| {}) > 0);
            let mut matches = Vec::new();
            buckets.find_matches(position, true, |m| matches.push(tuple(m)));
            assert!(matches.contains(&(8, position, position as u32 - 1)));
            if let Some(expected) = &reference {
                assert_eq!(&matches, expected);
            }
            reference = Some(matches);
        }
    }

    #[test]
    fn zero_run_shortcuts_never_cross_nonzero_boundaries_or_tails() {
        let mut values = vec![0; 129];
        values.push(u32::MAX);
        values.extend([0; 127]);
        values.extend([9, 0, 0, 0, 9, 0, 0]);
        let mut chain = Chain::new(&values, 2, 256, 91).unwrap();
        let config = SourceConfig {
            max_match_length: 91,
            prefix_filter: FilterPolicy::Eager,
            ..SourceConfig::default()
        };
        let mut buckets = Buckets::<8>::new(&values, 2, &config).unwrap();
        for position in 0..values.len() {
            let check = |m: Match| {
                let expected = values[position..]
                    .iter()
                    .zip(&values[position - m.distance..])
                    .take(91)
                    .take_while(|(a, b)| a == b)
                    .count();
                assert_eq!(m.length, expected, "pos={position} dist={}", m.distance);
                assert!(m.length >= MIN_MATCH);
            };
            chain.update(position);
            chain.find_matches(position, true, check);
            buckets.update(position);
            buckets.find_matches(position, true, check);
        }
    }
}
