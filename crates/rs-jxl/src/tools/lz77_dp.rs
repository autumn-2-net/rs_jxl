// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; the complete upstream notice and license
// are retained in docs/PORTING_KERNELS.md.

//! Private optimal parser for tools::lz77, requiring no parent registration.
//!
//! Frozen b866b3defec13494bfbaf705db93581cda118e37 sources:
//! * enc_lz77.cc: SymbolCostEstimator, ApplyLZ77_RangeParsed/Optimal,
//!   LZ77BucketMatcher::FindCostMatches and the eight-query repetition skip.
//! * enc_lz77_parse.h: LZ77MatchDominance, BuildLZ77MatchRanges,
//!   LZ77MatchCache, LZ77MatchLength, and LZ77RangeParser's dyadic path.
//! * enc_lz77_cost.h: VisitLZ77CostIntervals.
//! * enc_lz77_block.h: frontier-to-interval relaxation and compact backtracking.
//!
//! Adaptations: f64 costs, physical distances, sparse context IDs, existing
//! recent-prefix candidate traversal, and a dynamically sized frontier bounded
//! by bucket capacity plus 120 spatial candidates and one seed match. Unlike
//! the C++ 16-entry pruning cache it cannot overflow and silently lose a range.
//! Length intervals support any zero-LSB hybrid configuration, including the
//! caller's (4,0,0), instead of only the C++ optimized (0,0,0) specialization.
//! The exact 16-token-block dyadic parser is always used: there is no queue
//! overflow fallback that might expand every endpoint of every long match.
//! Source optimal callers can select enc_lz77.cc's 12-byte CompactStep storage
//! without changing this relaxation engine. Legacy and precise callers retain
//! f64 nodes; compact callers supply already-rounded f32 costs and recover the
//! physical distance and original-start context during backtracking.
//!
//! The optional C++ repetition skip is a search heuristic, not exact global
//! optimality. Every seed edge is retained even inside skipped origins; each
//! pass also compares against its seed under the same frozen model. Refit is
//! bounded to one additional pass, not a fixed-point iteration or a guarantee
//! about serialized size. No histogram-header cost is modeled here.

use std::collections::BTreeMap;

use super::{
    Command, DistanceCodes, HybridUintConfig, Literal, Lz77Mode, Lz77Options, MIN_MATCH, Match,
    Matcher, SymbolCosts, finish, reserved,
};
use crate::{Error, Result};

struct EntropyCosts<'a> {
    symbols: SymbolCosts,
    distances: SymbolCosts,
    codes: DistanceCodes,
    options: &'a Lz77Options,
}

impl<'a> EntropyCosts<'a> {
    fn new(seed: &[Command], width: usize, options: &'a Lz77Options) -> Self {
        let codes = DistanceCodes::new(width, options.window_size);
        let mut counts = BTreeMap::<u32, BTreeMap<u32, usize>>::new();
        let mut distance_counts = BTreeMap::<u32, usize>::new();
        for &command in seed {
            let (context, token) = match command {
                Command::Literal(literal) => (
                    literal.context,
                    options.literal_uint_config.encode(literal.value).token,
                ),
                Command::Match {
                    context,
                    length,
                    distance,
                } => {
                    let distance_token = options
                        .optimal_distance_uint_config
                        .encode(codes.symbol(distance as usize))
                        .token;
                    *distance_counts.entry(distance_token).or_default() += 1;
                    let token = options
                        .optimal_length_uint_config
                        .encode(length - MIN_MATCH as u32)
                        .token
                        + options.optimal_min_symbol;
                    (context, token)
                }
            };
            *counts.entry(context).or_default().entry(token).or_default() += 1;
        }
        // A separate model avoids reserving a sentinel context that might
        // collide with a caller's valid u32 context (including u32::MAX).
        let mut distances = BTreeMap::new();
        if !distance_counts.is_empty() {
            distances.insert(0, distance_counts);
        }
        Self {
            symbols: SymbolCosts::from_counts(counts, options.force_huffman),
            distances: SymbolCosts::from_counts(distances, options.force_huffman),
            codes,
            options,
        }
    }

    fn literal(&self, literal: Literal) -> f64 {
        let encoded = self.options.literal_uint_config.encode(literal.value);
        self.symbols.bits(literal.context, encoded.token) + f64::from(encoded.nbits)
    }

    fn length(&self, context: u32, length: usize) -> f64 {
        let encoded = self
            .options
            .optimal_length_uint_config
            .encode((length - MIN_MATCH) as u32);
        self.symbols
            .bits(context, encoded.token + self.options.optimal_min_symbol)
            + f64::from(encoded.nbits)
    }

    fn distance(&self, symbol: u32) -> f64 {
        let encoded = self.options.optimal_distance_uint_config.encode(symbol);
        self.distances.bits(0, encoded.token) + f64::from(encoded.nbits)
    }

    fn total(&self, commands: &[Command]) -> f64 {
        commands
            .iter()
            .map(|&command| match command {
                Command::Literal(literal) => self.literal(literal),
                Command::Match {
                    context,
                    length,
                    distance,
                } => {
                    self.length(context, length as usize)
                        + self.distance(self.codes.symbol(distance as usize))
                }
            })
            .sum()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PricedMatch {
    pub(super) matched: Match,
    pub(super) cost: f64,
}

pub(super) struct Frontier {
    // Increasing distance cost (symbol breaks ties) and strictly increasing
    // length. Thus each entry wins until the preceding entry's length ends.
    pub(super) entries: Vec<PricedMatch>,
}

impl Frontier {
    pub(super) fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            entries: reserved(capacity)?,
        })
    }

    fn required_length(&self, cost: f64, symbol: u32) -> Option<usize> {
        for entry in self.entries.iter().rev() {
            if entry.cost < cost || (entry.cost == cost && entry.matched.symbol <= symbol) {
                return entry.matched.length.checked_add(1);
            }
        }
        Some(MIN_MATCH)
    }

    // LZ77MatchDominance::Add. Retaining every nondominated entry also gives
    // the exact BuildLZ77MatchRanges envelope, without length-sized vectors.
    pub(super) fn add(&mut self, matched: Match, cost: f64) {
        if matched.length < MIN_MATCH {
            return;
        }
        let mut begin = 0;
        while begin < self.entries.len()
            && (self.entries[begin].cost < cost
                || (self.entries[begin].cost == cost
                    && self.entries[begin].matched.symbol < matched.symbol))
        {
            if self.entries[begin].matched.length >= matched.length {
                return;
            }
            begin += 1;
        }
        if begin < self.entries.len()
            && self.entries[begin].cost == cost
            && self.entries[begin].matched.symbol == matched.symbol
            && self.entries[begin].matched.length >= matched.length
        {
            return;
        }
        let mut end = begin;
        while end < self.entries.len() && self.entries[end].matched.length <= matched.length {
            end += 1;
        }
        self.entries.drain(begin..end);
        self.entries.insert(begin, PricedMatch { matched, cost });
    }
}

pub(super) struct MatchCache {
    // A verified interval at physical distance d stays valid at every later
    // origin inside that interval. Include the decoder's exact window limit.
    ends: Vec<usize>,
    #[cfg(test)]
    compared_values: usize,
}

impl MatchCache {
    pub(super) fn uncached() -> Self {
        Self {
            ends: Vec::new(),
            #[cfg(test)]
            compared_values: 0,
        }
    }

    pub(super) fn new(window: usize) -> Result<Self> {
        let mut ends = reserved(window + 1)?;
        ends.resize(window + 1, 0);
        Ok(Self {
            ends,
            #[cfg(test)]
            compared_values: 0,
        })
    }

    #[cfg(test)]
    pub(super) fn is_enabled(&self) -> bool {
        !self.ends.is_empty()
    }

    pub(super) fn length(
        &mut self,
        values: &[u32],
        position: usize,
        distance: usize,
        limit: usize,
    ) -> usize {
        self.length_with_prefix(values, position, distance, limit, 0)
    }

    pub(super) fn length_with_prefix(
        &mut self,
        values: &[u32],
        position: usize,
        distance: usize,
        limit: usize,
        prefix: usize,
    ) -> usize {
        let previous = position - distance;
        let mut length = self
            .ends
            .get(distance)
            .copied()
            .unwrap_or(0)
            .saturating_sub(position)
            .max(prefix)
            .min(limit);
        // LZ77MatchLength's eight-value block comparison, using safe slices.
        while limit - length >= 8 {
            #[cfg(test)]
            {
                self.compared_values += 8;
            }
            if values[position + length..position + length + 8]
                != values[previous + length..previous + length + 8]
            {
                break;
            }
            length += 8;
        }
        while length < limit {
            #[cfg(test)]
            {
                self.compared_values += 1;
            }
            if values[position + length] != values[previous + length] {
                break;
            }
            length += 1;
        }
        if let Some(end) = self.ends.get_mut(distance) {
            *end = (*end).max(position + length);
        }
        length
    }
}

fn find_frontier<const N: usize>(
    matcher: &mut Matcher<'_, N>,
    position: usize,
    costs: &EntropyCosts<'_>,
    cache: &mut MatchCache,
    frontier: &mut Frontier,
) {
    let values = matcher.values;
    let options = matcher.options;
    let limit = options.max_match_length.min(values.len() - position);
    matcher.visit_candidates(position, |previous, codes| {
        if previous >= position || position - previous > options.window_size {
            return;
        }
        let distance = position - previous;
        let symbol = codes.symbol(distance);
        let cost = costs.distance(symbol);
        let Some(required) = frontier.required_length(cost, symbol) else {
            return;
        };
        if required > limit
            || values[position] != values[previous]
            || values[position + required - 1] != values[previous + required - 1]
        {
            return;
        }
        let length = cache.length(values, position, distance, limit);
        if length >= required {
            frontier.add(
                Match {
                    length,
                    distance,
                    symbol,
                },
                cost,
            );
        }
    });
}

// VisitLZ77CostIntervals, generalized from (0,0,0) to zero-LSB hybrid
// configurations. Token and extra-bit count stay fixed across each interval;
// a different payload value does not change the modeled number of bits.
pub(super) fn visit_intervals(
    mut begin: usize,
    end: usize,
    config: HybridUintConfig,
    mut visit: impl FnMut(usize, usize),
) {
    debug_assert_eq!(config.lsb_in_token(), 0);
    while begin < end {
        let value = (begin - MIN_MATCH) as u32;
        let extra = config.encode(value).nbits;
        // u64 represents the exclusive boundary 2^32 on 32-bit targets too.
        let next = ((u64::from(value) >> extra) + 1) << extra;
        let stop = MIN_MATCH + next.min((end - MIN_MATCH) as u64) as usize;
        debug_assert!(stop > begin && stop <= end);
        visit(begin, stop);
        begin = stop;
    }
}

#[derive(Clone, Copy, Debug)]
struct Step {
    cost: f64,
    length: u32,
    distance: u32,
}

#[path = "lz77_source_compact_dp.rs"]
mod compact_steps;
use compact_steps::Steps;

impl Default for Step {
    fn default() -> Self {
        Self {
            cost: f64::INFINITY,
            length: 0,
            distance: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Offer {
    cost: f64,
    origin: usize,
    distance: u32,
}

impl Default for Offer {
    fn default() -> Self {
        Self {
            cost: f64::INFINITY,
            origin: 0,
            distance: 0,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct Work {
    queries: usize,
    compared_values: usize,
    intervals: usize,
    scalar_endpoints: usize,
    tree_updates: usize,
}

pub(super) struct RangeParser {
    steps: Steps,
    tree: Vec<Offer>,
    base: usize,
    depth: u32,
    path: Vec<Offer>,
    previous_block: Option<usize>,
    source_ties: bool,
    #[cfg(test)]
    work: Work,
}

impl RangeParser {
    const SHIFT: u32 = 4;
    const WIDTH: usize = 1 << Self::SHIFT;

    fn new(size: usize) -> Result<Self> {
        Self::with_storage(size, None)
    }

    fn with_storage(size: usize, compact_width: Option<usize>) -> Result<Self> {
        let steps = Steps::new(size, compact_width)?;
        let base = ((size >> Self::SHIFT) + 1)
            .checked_next_power_of_two()
            .ok_or(Error::Limit("LZ77 DP tree size"))?;
        let nodes = base
            .checked_mul(2)
            .ok_or(Error::Limit("LZ77 DP tree size"))?;
        let mut tree = reserved(nodes)?;
        tree.resize(nodes, Offer::default());
        let depth = base.trailing_zeros();
        let mut path = reserved(depth as usize + 1)?;
        path.resize(depth as usize + 1, Offer::default());
        Ok(Self {
            steps,
            tree,
            base,
            depth,
            path,
            previous_block: None,
            source_ties: false,
            #[cfg(test)]
            work: Work::default(),
        })
    }

    pub(super) fn new_source(size: usize) -> Result<Self> {
        let mut parser = Self::new(size)?;
        parser.source_ties = true;
        Ok(parser)
    }

    pub(super) fn new_source_compact(size: usize, width: usize) -> Result<Self> {
        let mut parser = Self::with_storage(size, Some(width))?;
        parser.source_ties = true;
        Ok(parser)
    }

    pub(super) fn cost(&self, position: usize) -> f64 {
        self.steps.get(position).cost
    }

    fn better(&self, offer: Offer, previous: Offer) -> bool {
        offer.cost < previous.cost
            || (self.source_ties && offer.cost == previous.cost && offer.origin < previous.origin)
    }

    fn relax(&mut self, position: usize, offer: Offer) {
        let step = self.steps.get(position);
        if self.better(
            offer,
            Offer {
                cost: step.cost,
                origin: position - step.length as usize,
                distance: step.distance,
            },
        ) {
            self.steps.set(
                position,
                Step {
                    cost: offer.cost,
                    length: (position - offer.origin) as u32,
                    distance: offer.distance,
                },
            );
        }
    }

    pub(super) fn advance(&mut self, position: usize) {
        let block = position >> Self::SHIFT;
        let first = match self.previous_block {
            None => 0,
            Some(previous) if previous == block => self.depth + 1,
            Some(previous) => {
                debug_assert!(block > previous);
                self.depth - (usize::BITS - 1 - (block ^ previous).leading_zeros())
            }
        };
        let mut best = if first == 0 {
            Offer::default()
        } else {
            self.path[first as usize - 1]
        };
        // Future-only full-block updates leave the shared prefix of the last
        // root-to-leaf path unchanged (LZ77RangeParser::Advance).
        for depth in first..=self.depth {
            let offer = self.tree[(self.base + block) >> (self.depth - depth)];
            if self.better(offer, best) {
                best = offer;
            }
            self.path[depth as usize] = best;
        }
        self.previous_block = Some(block);
        if best.cost.is_finite() {
            self.relax(position, best);
        }
    }

    pub(super) fn literal(&mut self, origin: usize, cost: f64) {
        self.relax(
            origin + 1,
            Offer {
                cost,
                origin,
                distance: 0,
            },
        );
    }

    pub(super) fn interval(
        &mut self,
        origin: usize,
        begin: usize,
        end: usize,
        distance: usize,
        cost: f64,
    ) {
        debug_assert!(begin >= MIN_MATCH && begin < end);
        debug_assert!(origin + end <= self.steps.len());
        let offer = Offer {
            cost,
            origin,
            distance: self.steps.encode_distance(distance),
        };
        let mut first = origin + begin;
        let last = origin + end;
        let aligned = first.saturating_add(Self::WIDTH - 1) & !(Self::WIDTH - 1);
        let aligned = aligned.min(last);
        #[cfg(test)]
        {
            self.work.intervals += 1;
        }
        while first < aligned {
            self.relax(first, offer);
            first += 1;
            #[cfg(test)]
            {
                self.work.scalar_endpoints += 1;
            }
        }
        let full_end = first.max(last & !(Self::WIDTH - 1));
        for position in full_end..last {
            self.relax(position, offer);
            #[cfg(test)]
            {
                self.work.scalar_endpoints += 1;
            }
        }
        let mut left = self.base + (first >> Self::SHIFT);
        let mut right = self.base + (full_end >> Self::SHIFT);
        debug_assert!(
            left == right || self.previous_block.is_none_or(|b| first >> Self::SHIFT > b)
        );
        while left < right {
            if left & 1 != 0 {
                self.update_node(left, offer);
                left += 1;
            }
            if right & 1 != 0 {
                right -= 1;
                self.update_node(right, offer);
            }
            left >>= 1;
            right >>= 1;
        }
    }

    fn update_node(&mut self, node: usize, offer: Offer) {
        #[cfg(test)]
        {
            self.work.tree_updates += 1;
        }
        if self.better(offer, self.tree[node]) {
            self.tree[node] = offer;
        }
    }

    pub(super) fn backtrack(&self, input: &[Literal]) -> Result<Vec<Command>> {
        let mut output = reserved(input.len())?;
        let mut position = input.len();
        while position != 0 {
            let step = self.steps.get(position);
            let length = step.length as usize;
            if length == 0 || length > position {
                return Err(Error::Encoding("invalid optimal LZ77 backtrack".into()));
            }
            let origin = position - length;
            let distance = self.steps.physical_distance(step.distance)?;
            if distance == 0 {
                debug_assert_eq!(length, 1);
                output.push(Command::Literal(input[origin]));
            } else {
                debug_assert!(length >= MIN_MATCH && distance as usize <= origin);
                output.push(Command::Match {
                    context: input[origin].context,
                    length: step.length,
                    distance,
                });
            }
            position = origin;
        }
        output.reverse();
        Ok(output)
    }
}

struct Parsed {
    commands: Vec<Command>,
    cost: f64,
    #[cfg(test)]
    work: Work,
}

fn parse_pass<const N: usize>(
    input: &[Literal],
    values: &[u32],
    width: usize,
    options: &Lz77Options,
    costs: &EntropyCosts<'_>,
    seed: &[Command],
) -> Result<Parsed> {
    let mut matcher = Matcher::<N>::new(values, width, options)?;
    let mut cache = MatchCache::new(options.window_size.min(values.len()))?;
    let mut frontier = Frontier::new(N + 121)?;
    let mut parser = RangeParser::new(input.len())?;
    let (mut seed_index, mut seed_position) = (0, 0);
    let (mut skip, mut repeats, mut previous_end, mut previous_distance) = (0, 0, 0, 0);
    for (position, &literal) in input.iter().enumerate() {
        parser.advance(position);
        let prefix_cost = parser.cost(position);
        parser.literal(position, prefix_cost + costs.literal(literal));
        frontier.entries.clear();
        let searched = skip == 0;
        if searched {
            #[cfg(test)]
            {
                parser.work.queries += 1;
            }
            find_frontier(&mut matcher, position, costs, &mut cache, &mut frontier);
        } else {
            skip -= 1;
            matcher.update_range(position + 1);
        }
        // A seeded command is already verified. Preserve its edge even when
        // the bounded matcher evicts its source or repetition search skips it.
        if position == seed_position {
            match seed[seed_index] {
                Command::Literal(_) => seed_position += 1,
                Command::Match {
                    length, distance, ..
                } => {
                    let matched = Match {
                        length: length as usize,
                        distance: distance as usize,
                        symbol: costs.codes.symbol(distance as usize),
                    };
                    frontier.add(matched, costs.distance(matched.symbol));
                    seed_position += length as usize;
                }
            }
            seed_index += 1;
        }
        let mut begin = MIN_MATCH;
        for entry in &frontier.entries {
            let matched = entry.matched;
            visit_intervals(
                begin,
                matched.length + 1,
                options.optimal_length_uint_config,
                |first, end| {
                    let cost = prefix_cost + (costs.length(literal.context, first) + entry.cost);
                    parser.interval(position, first, end, matched.distance, cost);
                },
            );
            begin = matched.length + 1;
        }
        if searched && options.optimal_skip_repetitions {
            if let Some(longest) = frontier.entries.last() {
                let matched = longest.matched;
                let end = position + matched.length;
                // Physical distance one correctly handles narrow-width aliases
                // that the C++ coded-distance test treats as RLE as well.
                if matched.distance == 1
                    || (matched.length >= 256
                        && end == previous_end
                        && matched.distance == previous_distance)
                {
                    repeats += 1;
                } else {
                    repeats = 0;
                }
                previous_end = end;
                previous_distance = matched.distance;
                if repeats >= 8 && matched.length >= 9 {
                    skip = matched.length - 9;
                    repeats = 0;
                }
            } else {
                repeats = 0;
            }
        }
    }
    debug_assert_eq!(seed_position, input.len());
    debug_assert_eq!(seed_index, seed.len());
    parser.advance(input.len());
    #[cfg(test)]
    {
        parser.work.compared_values = cache.compared_values;
    }
    Ok(Parsed {
        commands: parser.backtrack(input)?,
        cost: parser.cost(input.len()),
        #[cfg(test)]
        work: parser.work,
    })
}

fn passes<const N: usize>(
    input: &[Literal],
    width: usize,
    options: &Lz77Options,
) -> Result<Vec<Command>> {
    let mut seed_options = options.clone();
    seed_options.mode = Lz77Mode::Lazy;
    seed_options.require_global_savings = false;
    let mut seed = super::parse(input, width, &seed_options)?;
    let mut values = reserved(input.len())?;
    values.extend(input.iter().map(|literal| literal.value));
    let (mut raw_cost, mut final_cost) = (0.0, 0.0);
    for _ in 0..1 + usize::from(options.optimal_refit) {
        let costs = EntropyCosts::new(&seed, width, options);
        let parsed = parse_pass::<N>(input, &values, width, options, &costs, &seed)?;
        raw_cost = input.iter().map(|&literal| costs.literal(literal)).sum();
        final_cost = costs.total(&seed);
        if parsed.cost <= final_cost {
            final_cost = parsed.cost;
            seed = parsed.commands;
        }
    }
    finish(input, seed, raw_cost - final_cost, raw_cost, options)
}

pub(super) fn parse(
    input: &[Literal],
    width: usize,
    options: &Lz77Options,
) -> Result<Vec<Command>> {
    match options.bucket_capacity {
        1 => passes::<1>(input, width, options),
        2 => passes::<2>(input, width, options),
        4 => passes::<4>(input, width, options),
        8 => passes::<8>(input, width, options),
        16 => passes::<16>(input, width, options),
        32 => passes::<32>(input, width, options),
        64 => passes::<64>(input, width, options),
        _ => unreachable!("validated bucket capacity"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{FilterPolicy, WINDOW_SIZE, distance_symbol, literals};
    use super::*;

    fn options() -> Lz77Options {
        Lz77Options {
            mode: Lz77Mode::Optimal,
            bucket_bits: 2,
            require_global_savings: false,
            ..Lz77Options::default()
        }
    }

    fn input(values: &[u32]) -> Vec<Literal> {
        values
            .iter()
            .enumerate()
            .map(|(i, &value)| Literal {
                context: if i % 3 == 0 { u32::MAX } else { (i % 7) as u32 },
                value,
            })
            .collect()
    }

    fn expand(commands: &[Command], original: &[Literal], options: &Lz77Options) {
        let mut values = Vec::new();
        for &command in commands {
            let position = values.len();
            match command {
                Command::Literal(literal) => {
                    assert_eq!(literal, original[position]);
                    values.push(literal.value);
                }
                Command::Match {
                    context,
                    length,
                    distance,
                } => {
                    let (length, distance) = (length as usize, distance as usize);
                    assert_eq!(context, original[position].context);
                    assert!((MIN_MATCH..=options.max_match_length).contains(&length));
                    assert!((1..=options.window_size.min(position)).contains(&distance));
                    assert!(length <= original.len() - position);
                    for _ in 0..length {
                        values.push(values[values.len() - distance]);
                    }
                }
            }
        }
        assert_eq!(values, original.iter().map(|l| l.value).collect::<Vec<_>>());
    }

    fn next(random: &mut u32) -> u32 {
        *random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *random
    }

    #[test]
    fn frontier_matches_exhaustive_envelope_and_exceeds_sixteen_entries() {
        let mut frontier = Frontier::new(185).unwrap();
        for i in 0..64 {
            frontier.add(
                Match {
                    length: 3 + i,
                    distance: i + 1,
                    symbol: i as u32,
                },
                i as f64,
            );
        }
        assert_eq!(frontier.entries.len(), 64);
        let mut random = 1;
        let mut all = Vec::<PricedMatch>::new();
        frontier.entries.clear();
        for _ in 0..185 {
            let matched = Match {
                length: 3 + next(&mut random) as usize % 61,
                symbol: next(&mut random) % 35,
                distance: 1,
            };
            let cost = f64::from(next(&mut random) % 16) / 2.0;
            all.push(PricedMatch { matched, cost });
            frontier.add(matched, cost);
            for length in 3..=63 {
                let expected = all
                    .iter()
                    .filter(|e| e.matched.length >= length)
                    .min_by(|a, b| {
                        a.cost
                            .total_cmp(&b.cost)
                            .then(a.matched.symbol.cmp(&b.matched.symbol))
                    })
                    .map(|e| (e.cost, e.matched.symbol));
                let actual = frontier
                    .entries
                    .iter()
                    .find(|e| e.matched.length >= length)
                    .map(|e| (e.cost, e.matched.symbol));
                assert_eq!(actual, expected);
            }
            for entries in frontier.entries.windows(2) {
                assert!(entries[0].matched.length < entries[1].matched.length);
                assert!(
                    entries[0].cost < entries[1].cost
                        || (entries[0].cost == entries[1].cost
                            && entries[0].matched.symbol < entries[1].matched.symbol)
                );
            }
            let required = frontier.required_length(cost, matched.symbol).unwrap();
            for length in MIN_MATCH..required {
                assert!(all.iter().any(|e| e.matched.length >= length
                    && (e.cost < cost || (e.cost == cost && e.matched.symbol <= matched.symbol))));
            }
        }
    }

    #[test]
    fn hybrid_intervals_cover_every_length_and_power_boundary() {
        for (split, msb) in [(0, 0), (1, 0), (4, 0), (4, 2), (15, 0), (15, 15)] {
            let config = HybridUintConfig::new(split, msb, 0).unwrap();
            for first in [
                0usize, 1, 2, 3, 15, 16, 17, 31, 32, 63, 127, 255, 1023, 32767,
            ] {
                let begin = first + MIN_MATCH;
                let end = begin + 513;
                let mut previous = begin;
                visit_intervals(begin, end, config, |a, b| {
                    assert_eq!(a, previous);
                    assert!(b > a && b <= end);
                    let encoded = config.encode((a - MIN_MATCH) as u32);
                    for length in a..b {
                        let actual = config.encode((length - MIN_MATCH) as u32);
                        assert_eq!((actual.token, actual.nbits), (encoded.token, encoded.nbits));
                    }
                    previous = b;
                });
                assert_eq!(previous, end);
            }
            for exponent in 16..32 {
                let power = 1usize << exponent;
                let begin = power - 2;
                let end = (power + 8).min(u32::MAX as usize);
                let mut covered = 0;
                visit_intervals(begin, end, config, |a, b| {
                    covered += b - a;
                    let first = config.encode((a - MIN_MATCH) as u32);
                    for length in a..b {
                        let encoded = config.encode((length - MIN_MATCH) as u32);
                        assert_eq!((encoded.token, encoded.nbits), (first.token, first.nbits));
                    }
                });
                assert_eq!(covered, end - begin);
            }
            let end = (u32::MAX as usize).min(usize::MAX - 1);
            let mut covered = 0;
            visit_intervals(end - 31, end, config, |a, b| covered += b - a);
            assert_eq!(covered, 31);
        }
    }

    #[test]
    fn dyadic_relaxation_equals_scalar_endpoint_dp() {
        for size in [0, 1, 2, 15, 16, 17, 31, 32, 33, 255, 256, 257, 513] {
            let mut parser = RangeParser::new(size).unwrap();
            let mut reference = vec![f64::INFINITY; size + 1];
            reference[0] = 0.0;
            let mut random = 1;
            for origin in 0..size {
                parser.advance(origin);
                assert_eq!(
                    parser.cost(origin),
                    reference[origin],
                    "size={size}, origin={origin}"
                );
                let cost = reference[origin] + f64::from(next(&mut random) % 7);
                parser.literal(origin, cost);
                reference[origin + 1] = reference[origin + 1].min(cost);
                for _ in 0..7 {
                    let begin = MIN_MATCH + next(&mut random) as usize % 19;
                    let end = (begin + 1 + next(&mut random) as usize % 257).min(size - origin + 1);
                    if origin == 0 || begin >= end {
                        continue;
                    }
                    let cost = reference[origin] + f64::from(next(&mut random) % 23) / 2.0;
                    parser.interval(origin, begin, end, 1, cost);
                    for value in &mut reference[origin + begin..origin + end] {
                        *value = value.min(cost);
                    }
                }
            }
            parser.advance(size);
            assert_eq!(parser.cost(size), reference[size]);
            assert!(parser.work.scalar_endpoints <= 2 * RangeParser::WIDTH * parser.work.intervals);
            assert!(
                parser.work.tree_updates <= 2 * (parser.depth as usize + 1) * parser.work.intervals
            );
        }
    }

    fn exhaustive_cost(
        original: &[Literal],
        costs: &EntropyCosts<'_>,
        options: &Lz77Options,
    ) -> f64 {
        let mut dp = vec![f64::INFINITY; original.len() + 1];
        dp[0] = 0.0;
        for position in 0..original.len() {
            dp[position + 1] =
                dp[position + 1].min(dp[position] + costs.literal(original[position]));
            for distance in 1..=position.min(options.window_size) {
                let limit = (original.len() - position).min(options.max_match_length);
                let length = original[position..]
                    .iter()
                    .zip(&original[position - distance..])
                    .take(limit)
                    .take_while(|(a, b)| a.value == b.value)
                    .count();
                for length in MIN_MATCH..=length {
                    let cost = dp[position]
                        + costs.length(original[position].context, length)
                        + costs.distance(costs.codes.symbol(distance));
                    dp[position + length] = dp[position + length].min(cost);
                }
            }
        }
        dp[original.len()]
    }

    #[test]
    fn exact_search_matches_exhaustive_shortest_path_and_decodes() {
        let mut random = 9;
        for case in 0..96 {
            let size = 4 + case % 45;
            let period = 1 + case % 11;
            let values: Vec<_> = (0..size)
                .map(|i| {
                    if i % 17 < 11 {
                        u32::MAX - (i % period) as u32
                    } else {
                        next(&mut random) % 4
                    }
                })
                .collect();
            let original = input(&values);
            for width in [0, 1, 2, 3, 17, usize::MAX] {
                let options = Lz77Options {
                    bucket_bits: 0,
                    bucket_capacity: 64,
                    window_size: 1 + case % 32,
                    max_match_length: 3 + case % 37,
                    optimal_skip_repetitions: false,
                    optimal_length_uint_config: HybridUintConfig::new(
                        if case % 2 == 0 { 0 } else { 4 },
                        0,
                        0,
                    )
                    .unwrap(),
                    ..options()
                };
                let seed = literals(&original).unwrap();
                let costs = EntropyCosts::new(&seed, width, &options);
                let parsed =
                    parse_pass::<64>(&original, &values, width, &options, &costs, &seed).unwrap();
                let expected = exhaustive_cost(&original, &costs, &options);
                assert!(
                    (parsed.cost - expected).abs() < 1e-8,
                    "case={case}, width={width}"
                );
                assert!((costs.total(&parsed.commands) - expected).abs() < 1e-8);
                expand(&parsed.commands, &original, &options);
            }
        }
    }

    #[test]
    fn seed_model_counts_lengths_distances_and_sparse_contexts() {
        let options = options();
        let literal = Literal {
            context: u32::MAX,
            value: u32::MAX,
        };
        let seed = [
            Command::Literal(literal),
            Command::Match {
                context: u32::MAX,
                length: 31,
                distance: 1,
            },
            Command::Literal(Literal {
                context: 7,
                value: 1234,
            }),
        ];
        let costs = EntropyCosts::new(&seed, 1, &options);
        assert_eq!(
            costs
                .symbols
                .bits(u32::MAX, options.literal_uint_config.encode(u32::MAX).token),
            1.0
        );
        assert_eq!(costs.length(u32::MAX, 31), 5.0);
        assert_eq!(costs.literal(literal), 30.0);
        assert_eq!(costs.distance(distance_symbol(1, 1).unwrap()), 0.0);
        assert_eq!(costs.symbols.bits(0, 0), 12.0);
        assert_eq!(costs.distances.bits(0, 1), 12.0);
    }

    #[test]
    fn refit_is_one_bounded_additional_pass_with_seed_fallback() {
        let values: Vec<_> = (0..1031).map(|i| u32::MAX - (i % 13) as u32).collect();
        let original = input(&values);
        for refit in [false, true] {
            let options = Lz77Options {
                bucket_capacity: 4,
                optimal_refit: refit,
                ..options()
            };
            let seed_options = Lz77Options {
                mode: Lz77Mode::Lazy,
                ..options.clone()
            };
            let mut seed = super::super::parse(&original, 13, &seed_options).unwrap();
            for _ in 0..1 + usize::from(refit) {
                let costs = EntropyCosts::new(&seed, 13, &options);
                let parsed =
                    parse_pass::<4>(&original, &values, 13, &options, &costs, &seed).unwrap();
                assert!(parsed.cost <= costs.total(&seed) + 1e-8);
                if parsed.cost <= costs.total(&seed) {
                    seed = parsed.commands;
                }
                expand(&seed, &original, &options);
            }
            let actual = super::super::parse(&original, 13, &options).unwrap();
            assert_eq!(actual, seed);
        }
    }

    #[test]
    fn long_runs_and_periods_have_bounded_matching_and_relaxation_work() {
        for (size, period, skip) in [
            (65_539, 1, true),
            (65_539, 7, true),
            (4099, 1, false),
            (4099, 7, false),
        ] {
            let values: Vec<_> = (0..size).map(|i| u32::MAX - (i % period) as u32).collect();
            let original = input(&values);
            let options = Lz77Options {
                optimal_skip_repetitions: skip,
                ..options()
            };
            let seed_options = Lz77Options {
                mode: Lz77Mode::Lazy,
                ..options.clone()
            };
            let seed = super::super::parse(&original, period, &seed_options).unwrap();
            let costs = EntropyCosts::new(&seed, period, &options);
            let parsed =
                parse_pass::<32>(&original, &values, period, &options, &costs, &seed).unwrap();
            expand(&parsed.commands, &original, &options);
            assert!(parsed.cost <= costs.total(&seed) + 1e-8);
            assert!(
                parsed.work.compared_values <= 128 * size,
                "{:#?}",
                parsed.work
            );
            assert!(parsed.work.scalar_endpoints <= 32 * parsed.work.intervals);
            assert!(parsed.work.tree_updates <= 2 * usize::BITS as usize * parsed.work.intervals);
            if skip {
                assert!(parsed.work.queries < 128, "{:#?}", parsed.work);
            } else {
                assert_eq!(parsed.work.queries, size);
            }
        }
    }

    #[test]
    fn verified_cache_handles_overlap_exact_window_and_shortened_limits() {
        let values: Vec<_> = (0..200).map(|i| (i % 64) as u32).collect();
        let mut cache = MatchCache::new(64).unwrap();
        assert_eq!(cache.length(&values, 64, 64, 100), 100);
        let work = cache.compared_values;
        assert_eq!(cache.length(&values, 65, 64, 70), 70);
        assert_eq!(cache.compared_values, work);
        assert_eq!(cache.length(&values, 66, 64, 98), 98);
        assert_eq!(cache.compared_values, work);
        assert_eq!(cache.length(&values, 67, 64, 133), 133);
        assert_eq!(cache.length(&values, 199, 64, 1), 1);
    }

    #[test]
    fn uncached_match_extension_keeps_prefix_and_tail_semantics() {
        let values: Vec<_> = (0..201).map(|i| (i % 7) as u32).collect();
        let mut plain = MatchCache::uncached();
        let mut cached = MatchCache::new(7).unwrap();
        for position in 7..values.len() {
            for limit in [0, 1, 2, 3, 7, 8, 9, values.len() - position] {
                let limit = limit.min(values.len() - position);
                let prefix = limit.min(3);
                assert_eq!(
                    plain.length_with_prefix(&values, position, 7, limit, prefix),
                    limit
                );
                assert_eq!(
                    cached.length_with_prefix(&values, position, 7, limit, prefix),
                    limit
                );
            }
        }
        assert!(!plain.is_enabled());
        assert!(plain.compared_values > cached.compared_values);
    }

    #[test]
    fn public_optimal_roundtrips_tails_filters_widths_and_model_configs() {
        let mut random = 17;
        let values: Vec<_> = (0..771)
            .map(|i| match i % 97 {
                0..=30 => 0,
                31..=65 => u32::MAX - (i % 3) as u32,
                _ => next(&mut random),
            })
            .collect();
        for capacity in [1, 2, 4, 8, 16, 32, 64] {
            for width in [0, 1, 2, 3, 17, usize::MAX] {
                for end in [0, 1, 2, 3, 4, 769, 770, 771] {
                    let original = input(&values[..end]);
                    let options = Lz77Options {
                        bucket_capacity: capacity,
                        max_match_length: 97,
                        optimal_refit: capacity % 3 == 1,
                        force_huffman: width == 2,
                        optimal_length_uint_config: HybridUintConfig::new(
                            4,
                            if width == 3 { 2 } else { 0 },
                            0,
                        )
                        .unwrap(),
                        optimal_distance_uint_config: if width == 17 {
                            HybridUintConfig::default()
                        } else {
                            HybridUintConfig::new(0, 0, 0).unwrap()
                        },
                        ..options()
                    };
                    let baseline = super::super::parse(&original, width, &options).unwrap();
                    expand(&baseline, &original, &options);
                    for policy in [FilterPolicy::Disabled, FilterPolicy::Eager] {
                        let changed = Lz77Options {
                            prefix_filter: policy,
                            range_search: false,
                            ..options.clone()
                        };
                        let actual = super::super::parse(&original, width, &changed).unwrap();
                        assert_eq!(actual, baseline);
                    }
                }
            }
        }
    }

    #[test]
    fn optimal_options_reject_unsupported_intervals_and_overlapping_alphabets() {
        assert!(
            super::super::parse(
                &[],
                0,
                &Lz77Options {
                    optimal_length_uint_config: HybridUintConfig::new(4, 0, 1).unwrap(),
                    ..options()
                }
            )
            .is_err()
        );
        assert!(
            super::super::parse(
                &[],
                0,
                &Lz77Options {
                    optimal_min_symbol: u32::MAX,
                    ..options()
                }
            )
            .is_err()
        );
        let original = input(&[u32::MAX]);
        assert!(
            super::super::parse(
                &original,
                0,
                &Lz77Options {
                    optimal_min_symbol: 1,
                    ..options()
                }
            )
            .is_err()
        );
        // Unused optimal settings do not alter the established modes.
        assert!(
            super::super::parse(
                &original,
                0,
                &Lz77Options {
                    mode: Lz77Mode::Lazy,
                    optimal_min_symbol: u32::MAX,
                    optimal_length_uint_config: HybridUintConfig::new(4, 0, 1).unwrap(),
                    ..options()
                }
            )
            .is_ok()
        );
        assert_eq!(Lz77Options::default().mode, Lz77Mode::Lazy);
        assert_eq!(Lz77Options::default().window_size, WINDOW_SIZE);
    }
}
