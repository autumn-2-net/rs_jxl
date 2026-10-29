// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_block.h and enc_lz77_parse.h at b866b3d. Encoder-only cuts
//! limit destinations, never reset the dictionary or alter physical distances.

use super::super::dp::visit_intervals;
use super::index::Dominance;
use super::matchers::window;
use super::search::{DistancePrices, FrontierCarry, MatchCache, Query, physical};
use super::{
    Command, Costs, Literal, MIN_MATCH, Match, Parser, Schedule, SourceConfig, reserved, workers,
};
use crate::{Error, Result};

pub(super) fn shared_index<'a>(
    values: &'a [Vec<u32>],
    widths: &[usize],
    config: &SourceConfig,
) -> Result<Option<(usize, super::StaticIndex<'a>)>> {
    use super::{
        SourceDeepPrefix, SourceIndexMode, SourceIndexTags, SourceLz77Method, SourceRunKeys,
    };
    if !config.shared_index
        || config.index_mode != SourceIndexMode::All
        || config.block_mode != SourceBlockMode::Intervals
        || config.length_uint_config != super::HybridUintConfig::new(0, 0, 0).unwrap()
        || config.method != SourceLz77Method::Optc256
        || !config.bucket_route().indexed
        || values.is_empty()
    {
        return Ok(None);
    }
    let mut largest = 0;
    for i in 1..values.len() {
        if values[i].len() > values[largest].len() {
            largest = i;
        }
    }
    if !(65536..=1 << 24).contains(&values[largest].len()) {
        return Ok(None);
    }
    let shared_config = SourceConfig {
        index_tags: SourceIndexTags::Scalar,
        index_run_keys: SourceRunKeys::Disabled,
        index_deep_prefix: SourceDeepPrefix::Disabled,
        index_anchors_first: true,
        index_word_match: false,
        // This source builder narrows automatically, independently of the
        // ordinary block COMPACT_VALUES switch; coordinates remain disabled.
        index_compact_values: true,
        index_coordinates: false,
        parallel_index_build: true,
        ..config.clone()
    };
    let index = super::StaticIndex::new(
        &values[largest],
        widths.get(largest).copied().unwrap_or(0),
        &shared_config,
    )?;
    Ok(Some((largest, index)))
}

// enc_lz77_seed.h: per-part double sums, strict positive gains, longer copy on
// equal positive gains, then double reduction followed by one f32 conversion.
pub(super) fn indexed_seed(
    input: &[Literal],
    index: &super::StaticIndex<'_>,
    config: &SourceConfig,
    costs: &Costs,
) -> Result<(Vec<Command>, f32, f32)> {
    let tasks = workers::indexed_seed_tasks(config, input.len());
    let slots = workers::index_workers(config, tasks, 4 << 20);
    let schedule = Schedule::jobs(tasks, slots);
    let results = schedule.collect(tasks, |part| {
        let begin = input.len() * part / tasks;
        let end = input.len() * (part + 1) / tasks;
        let mut prefix = reserved(end - begin + 1)?;
        prefix.push(0.0f64);
        for literal in &input[begin..end] {
            let encoded = config.literal_uint_config.encode(literal.value);
            let price = encoded.nbits as f32 + costs.bits(literal.context, encoded.token);
            prefix.push(prefix.last().copied().unwrap() + f64::from(price));
        }
        let raw = *prefix.last().unwrap();
        let mut saving = 0.0;
        let mut output = reserved(end - begin)?;
        let mut position = begin;
        while position < end {
            let (mut best, mut gain) = (None, 0.0);
            index.find_bounded_matches(
                position,
                end - position,
                super::static_distance,
                |matched| {
                    let decrease = prefix[position - begin + matched.length]
                        - prefix[position - begin]
                        - f64::from(super::static_cost(matched, input[position].context, costs));
                    if decrease > gain
                        || (decrease == gain
                            && gain > 0.0
                            && best.is_some_and(|m: Match| matched.length > m.length))
                    {
                        gain = decrease;
                        best = Some(matched);
                    }
                },
                &mut MatchCache::uncached(),
            );
            if let Some(matched) = best {
                output.push(Command::Match {
                    context: input[position].context,
                    length: matched.length as u32,
                    distance: matched.distance as u32,
                });
                saving += gain;
                position += matched.length;
            } else {
                output.push(Command::Literal(input[position]));
                position += 1;
            }
        }
        Ok((output, saving, raw))
    })?;
    let mut output = reserved(input.len())?;
    let (mut saving, mut raw) = (0.0f64, 0.0f64);
    for part in results {
        let (commands, decrease, cost) = part.unwrap_or_default();
        output.extend(commands);
        saving += decrease;
        raw += cost;
    }
    Ok((output, saving as f32, raw as f32))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceBlockMode {
    #[default]
    Disabled,
    /// BLOCK_DP=1: explicit endpoints with strict update order.
    Endpoints,
    /// BLOCK_DP=2: source heap intervals, capped at 65536 queued offers.
    Intervals,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceSeedGuided {
    #[default]
    Disabled,
    /// SEED_GUIDED=1: nominate the containing seed copy's verified distance.
    Hints,
    /// SEED_GUIDED=2: also restrict long-copy interiors to repair origins.
    SparseOrigins,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SeedSpan {
    begin: usize,
    end: usize,
    distance: usize,
}

impl SeedSpan {
    fn search(self, position: usize) -> bool {
        debug_assert!((self.begin..self.end).contains(&position));
        self.end - self.begin < 64
            || position - self.begin < 8
            || self.end - position <= 8
            || (position - self.begin).is_multiple_of(32)
    }
}

fn seed_spans(seed: &[Command], size: usize) -> Result<Vec<SeedSpan>> {
    let copies = seed
        .iter()
        .filter(|c| matches!(c, Command::Match { .. }))
        .count();
    let mut spans = reserved(copies)?;
    let mut position = 0;
    for command in seed {
        let length = command_length(command);
        if length == 0 || length > size.saturating_sub(position) {
            return Err(Error::InvalidInput("invalid source LZ77 seed span".into()));
        }
        if let Command::Match { distance, .. } = *command {
            // Commands already contain physical distances, unlike source wire tokens.
            spans.push(SeedSpan {
                begin: position,
                end: position + length,
                distance: distance as usize,
            });
        }
        position += length;
    }
    if position != size {
        return Err(Error::InvalidInput(
            "incomplete source LZ77 seed spans".into(),
        ));
    }
    Ok(spans)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceSeedBlocks {
    #[default]
    Disabled,
    /// SEED_BLOCKS=1: separate fast dictionaries per seed block.
    Cold,
    /// SEED_BLOCKS=2: overlap prior tokens, then trim the encoded prefix.
    /// The source's default SEED_HISTORY is 32768; valid bounds are 1..=2^20.
    Warm { history: usize },
}

pub(super) struct SeedPart {
    pub(super) owner: usize,
    pub(super) begin: usize,
    pub(super) end: usize,
    pub(super) prefix: usize,
}

pub(super) fn seed_parts(
    inputs: &[&[Literal]],
    config: &SourceConfig,
    eligible: bool,
) -> Result<Vec<SeedPart>> {
    let total: usize = inputs.iter().map(|s| s.len()).sum();
    // Preserve the first largest stream on ties, as in the source loop.
    let mut largest = 0;
    for i in 1..inputs.len() {
        if inputs[i].len() > inputs[largest].len() {
            largest = i;
        }
    }
    let split = eligible
        && config.seed_blocks != SourceSeedBlocks::Disabled
        && !inputs.is_empty()
        && config.parallelism.is_some()
        && !crate::execution::is_serial()
        && rayon::current_thread_index().is_some()
        && inputs[largest].len() >= 262144
        && inputs[largest].len() >= total - total / 4;
    let parts = if split {
        workers::index_tasks(config, (inputs[largest].len() / 65536).min(16), 8 << 20)
    } else {
        1
    };
    let mut result = reserved(inputs.len() + parts - 1)?;
    for (owner, input) in inputs.iter().enumerate() {
        let count = if owner == largest { parts } else { 1 };
        for part in 0..count {
            let begin = input.len() * part / count;
            let prefix = match config.seed_blocks {
                SourceSeedBlocks::Warm { history } if split => begin.min(history),
                _ => 0,
            };
            result.push(SeedPart {
                owner,
                begin: begin - prefix,
                end: input.len() * (part + 1) / count,
                prefix,
            });
        }
    }
    Ok(result)
}

pub(super) fn merge_seed(
    inputs: &[&[Literal]],
    parts: &[SeedPart],
    seed: Vec<Vec<Command>>,
) -> Result<Vec<Vec<Command>>> {
    if parts.len() == inputs.len()
        && parts
            .iter()
            .enumerate()
            .all(|(i, part)| part.owner == i && part.prefix == 0)
    {
        return Ok(seed);
    }
    let mut result = reserved(inputs.len())?;
    result.resize_with(inputs.len(), Vec::new);
    for (part, commands) in parts.iter().zip(seed) {
        if part.prefix == 0 {
            result[part.owner].extend(commands);
        } else {
            trim_prefix(
                &inputs[part.owner][part.begin..part.end],
                part.prefix,
                &commands,
                &mut result[part.owner],
            )?;
        }
    }
    Ok(result)
}

fn command_length(command: &Command) -> usize {
    match command {
        Command::Literal(_) => 1,
        Command::Match { length, .. } => *length as usize,
    }
}

pub(super) fn cuts(
    input: &[Literal],
    seed: &[Command],
    parts: usize,
    work: bool,
) -> Result<Vec<usize>> {
    if parts == 0 {
        return Err(Error::InvalidInput("zero source LZ77 blocks".into()));
    }
    let mut boundaries = reserved(seed.len())?;
    let (mut position, mut weight) = (0usize, 0u64);
    for command in seed {
        let length = command_length(command);
        if length == 0 || length > input.len().saturating_sub(position) {
            return Err(Error::InvalidInput(
                "invalid source LZ77 seed coverage".into(),
            ));
        }
        weight += if !work {
            length as u64
        } else {
            match command {
                Command::Match { .. } => (16 * length.min(64) + length / 16 + 1) as u64,
                Command::Literal(_) => {
                    if position != 0 && input[position].value == input[position - 1].value {
                        1
                    } else {
                        16
                    }
                }
            }
        };
        position += length;
        boundaries.push((position, weight));
    }
    if position != input.len() {
        return Err(Error::InvalidInput("incomplete source LZ77 seed".into()));
    }
    let mut result = reserved(parts.min(seed.len()).saturating_add(2))?;
    result.push(0);
    let mut target = 1;
    for (position, reached) in boundaries {
        let threshold =
            |target: usize| (u128::from(weight) * target as u128 / parts as u128) as u64;
        if target < parts && reached >= threshold(target) && position < input.len() {
            result.push(position);
            loop {
                target += 1;
                if target >= parts || reached < threshold(target) {
                    break;
                }
            }
        }
    }
    result.push(input.len());
    Ok(result)
}

pub(super) fn trim_prefix(
    input: &[Literal],
    prefix: usize,
    seed: &[Command],
    output: &mut Vec<Command>,
) -> Result<()> {
    if prefix > input.len() {
        return Err(Error::InvalidInput("source LZ77 seed prefix".into()));
    }
    let mut position = 0;
    for command in seed {
        let length = command_length(command);
        if length == 0 || length > input.len().saturating_sub(position) {
            return Err(Error::InvalidInput("invalid source LZ77 warm seed".into()));
        }
        let end = position + length;
        let first = position.max(prefix);
        if first < end {
            if let Command::Match { distance, .. } = *command
                && end - first >= MIN_MATCH
            {
                output.push(Command::Match {
                    context: input[first].context,
                    length: (end - first) as u32,
                    distance,
                });
            } else {
                output.extend(input[first..end].iter().copied().map(Command::Literal));
            }
        }
        position = end;
    }
    if position != input.len() {
        return Err(Error::InvalidInput(
            "incomplete source LZ77 warm seed".into(),
        ));
    }
    Ok(())
}

pub(super) fn parse(
    input: &[Literal],
    seed: &[Command],
    width: usize,
    config: &SourceConfig,
    costs: &Costs,
    find: impl Fn(usize, Query<'_>, &mut MatchCache, &dyn Fn(u32) -> f32, &mut dyn FnMut(Match))
    + Sync
    + Send,
) -> Result<Vec<Command>> {
    let cache_bytes = MatchCache::block_bytes(window(input.len()), config);
    // Shared across all blocks. At most (2^20 + 120) floats plus 128 classes.
    let prices = DistancePrices::new(window(input.len()), width, config, |d| costs.distance(d))?;
    let logical = workers::index_tasks(
        config,
        (input.len() / 65536).clamp(1, 32),
        (8 << 20) + cache_bytes,
    );
    let parts = logical.max((logical * config.block_grain).min(input.len() / 16384));
    let cuts = cuts(input, seed, parts, config.block_work_cuts)?;
    let spans = if config.block_seed_guided == SourceSeedGuided::Disabled {
        Vec::new()
    } else {
        seed_spans(seed, input.len())?
    };
    let largest = cuts
        .windows(2)
        .map(|cut| cut[1] - cut[0])
        .max()
        .unwrap_or(0);
    let slots = workers::index_workers(
        config,
        logical,
        Parser::block_scratch(largest, config).saturating_add((8 << 20) + cache_bytes),
    );
    let schedule = Schedule::jobs(cuts.len() - 1, slots);
    let results = schedule.collect(cuts.len() - 1, |task| {
        parse_part(
            input,
            cuts[task],
            cuts[task + 1],
            width,
            config,
            costs,
            &prices,
            &spans,
            &find,
        )
    })?;
    let mut output = reserved(input.len())?;
    for result in results {
        output.extend(result.unwrap_or_default());
    }
    Ok(output)
}

fn zero_tail_end(
    input: &[Literal],
    begin: usize,
    end: usize,
    config: &SourceConfig,
    costs: &Costs,
) -> usize {
    let mut search_end = end;
    while search_end > begin {
        let token = input[search_end - 1];
        let encoded = config.literal_uint_config.encode(token.value);
        if encoded.nbits != 0 || costs.bits(token.context, encoded.token) != 0.0 {
            break;
        }
        search_end -= 1;
    }
    search_end
}

fn literal_bound_prefix(
    input: &[Literal],
    config: &SourceConfig,
    costs: &Costs,
) -> Result<Vec<f64>> {
    let mut prefix = reserved(input.len() + 1)?;
    prefix.push(0.0f64);
    for token in input {
        let encoded = config.literal_uint_config.encode(token.value);
        // Unlike LAZY, source LITERAL_BOUND adds Bits before the extra bits.
        prefix.push(
            prefix.last().copied().unwrap()
                + f64::from(costs.bits(token.context, encoded.token))
                + f64::from(encoded.nbits),
        );
    }
    Ok(prefix)
}

fn literal_bound_exceeded(
    cost: f32,
    base: f32,
    prefix_end: f64,
    prefix_begin: f64,
    end_length: usize,
) -> bool {
    let upper = f64::from(base) + prefix_end - prefix_begin;
    // C++ first multiplies size_t by float epsilon in float, then by double.
    let margin = f64::from((end_length + 2) as f32 * f32::EPSILON) * upper.max(1.0);
    f64::from(cost) > upper + margin
}

// Keep source block bounds and the distinct shared cost tables explicit.
#[allow(clippy::too_many_arguments)]
fn parse_part(
    input: &[Literal],
    begin: usize,
    end: usize,
    width: usize,
    config: &SourceConfig,
    costs: &Costs,
    prices: &DistancePrices,
    spans: &[SeedSpan],
    find: &impl Fn(usize, Query<'_>, &mut MatchCache, &dyn Fn(u32) -> f32, &mut dyn FnMut(Match)),
) -> Result<Vec<Command>> {
    let search_end = if config.block_zero_tail {
        zero_tail_end(input, begin, end, config, costs)
    } else {
        end
    };
    let mut cache = MatchCache::for_blocks(window(input.len()), config)?;
    if config.block_lazy {
        return parse_lazy(
            input, begin, end, width, config, costs, prices, &mut cache, find,
        );
    }
    let mut parser = Parser::blocks(end - begin, config)?;
    let literal_prefix = if config.block_literal_bound {
        literal_bound_prefix(&input[begin..end], config, costs)?
    } else {
        Vec::new()
    };
    let mut carry = FrontierCarry::new();
    let mut span = spans.partition_point(|copy| copy.end <= begin);
    let distance_price = |d| prices.get(d, |d| costs.distance(d));
    let (mut skip, mut repeats, mut previous_end, mut previous_symbol) = (0, 0, 0, 0);
    let mut next = 0;
    while next < end - begin {
        let local = next;
        next += 1;
        let position = begin + local;
        let token = input[position];
        parser.advance(local);
        if config.block_dp_jump && config.block_mode == SourceBlockMode::Intervals && skip != 0 {
            let target = parser.jump_literal_only(local, (end - begin).min(local + skip));
            if target > local {
                skip -= target - local;
                next = target;
                continue;
            }
        }
        let base = parser.cost(local) as f32;
        let encoded = config.literal_uint_config.encode(token.value);
        // Block parsing uses direct left-associative literal prices, not the
        // cumulative subtraction used by ApplyLZ77_Optimal's endpoint parser.
        let literal = base + costs.bits(token.context, encoded.token) + encoded.nbits as f32;
        parser.block_literal(local, token.context, literal);
        if position >= search_end {
            continue;
        }
        if skip != 0 {
            skip -= 1;
            continue;
        }
        while span < spans.len() && spans[span].end <= position {
            span += 1;
        }
        let in_copy = span < spans.len() && spans[span].begin <= position;
        if config.block_seed_guided == SourceSeedGuided::SparseOrigins
            && in_copy
            && !spans[span].search(position)
        {
            continue;
        }
        let hint_distance = if in_copy { spans[span].distance } else { 0 };
        let mut frontier = Dominance::new();
        let carried = config.block_frontier_carry.then_some(&carry);
        let mut query = Query {
            maximum: end - position,
            hint_distance,
            bound: prices.bound,
            carry: carried,
            ..Query::default()
        };
        if config.block_shared_frontier {
            query.result = Some(&mut frontier);
            find(position, query, &mut cache, &distance_price, &mut |_| {});
        } else {
            find(position, query, &mut cache, &distance_price, &mut |m| {
                frontier.add(m.length, m.symbol, distance_price(m.symbol))
            });
        }
        let (mut longest, mut symbol) = (0, 0);
        let mut relax = |first, last, distance| {
            visit_intervals(first, last + 1, config.length_uint_config, |a, b| {
                let price = costs.length(token.context, a, config) + distance_price(distance);
                let cost = base + price;
                if config.block_literal_bound
                    && literal_bound_exceeded(
                        cost,
                        base,
                        literal_prefix[local + b - 1],
                        literal_prefix[local],
                        b,
                    )
                {
                    return;
                }
                parser.block_interval(local, a, b, physical(distance, width), token.context, cost);
            });
            if last >= longest {
                longest = last;
                symbol = distance;
            }
        };
        if frontier.overflowed {
            // A full 16-entry pruning certificate must never drop proposals.
            // Source overflow replay omits both result and distance bound.
            let query = Query {
                maximum: end - position,
                hint_distance,
                carry: carried,
                ..Query::default()
            };
            find(position, query, &mut cache, &distance_price, &mut |m| {
                relax(MIN_MATCH, m.length, m.symbol)
            });
        } else {
            let mut first = MIN_MATCH;
            for entry in &frontier.entries[..frontier.size] {
                relax(first, entry.length, entry.symbol);
                first = entry.length + 1;
            }
        }
        if config.block_frontier_carry {
            carry.remember(position, &frontier);
        }
        if longest < MIN_MATCH {
            repeats = 0;
            continue;
        }
        let match_end = position + longest;
        if symbol == 0
            || (symbol == 1 && width != 0)
            || (longest >= 256 && match_end == previous_end && symbol == previous_symbol)
        {
            repeats += 1;
        } else {
            repeats = 0;
        }
        previous_end = match_end;
        previous_symbol = symbol;
        if repeats >= 8 && longest >= 9 {
            skip = longest - 9;
            repeats = 0;
        }
    }
    parser.advance(end - begin);
    parser.backtrack(&input[begin..end])
}

#[derive(Clone, Copy, Default)]
struct LazyChoice {
    length: usize,
    symbol: u32,
    gain: f64,
}

// Source passes integer 0 as VisitLZ77CostIntervals's prefix_cost, deducing
// Cost=int. Its float length+distance sum is truncated, then promoted to the
// callback's float bits before subtraction from the double literal prefix.
fn lazy_price(length: f32, distance: f32) -> f32 {
    ((length + distance) as i32) as f32
}

// Keep the source block origin, physical-distance width and shared prices explicit.
#[allow(clippy::too_many_arguments)]
fn parse_lazy(
    input: &[Literal],
    begin: usize,
    end: usize,
    width: usize,
    config: &SourceConfig,
    costs: &Costs,
    prices: &DistancePrices,
    cache: &mut MatchCache,
    find: &impl Fn(usize, Query<'_>, &mut MatchCache, &dyn Fn(u32) -> f32, &mut dyn FnMut(Match)),
) -> Result<Vec<Command>> {
    let count = end - begin;
    let mut prefix = reserved(count + 1)?;
    prefix.push(0.0f64);
    for token in &input[begin..end] {
        let encoded = config.literal_uint_config.encode(token.value);
        // Preserve the source's two separate double additions, not an f32 sum.
        prefix.push(
            prefix.last().copied().unwrap()
                + f64::from(encoded.nbits)
                + f64::from(costs.bits(token.context, encoded.token)),
        );
    }
    // Source LAZY uses the dense table or direct price, never DISTANCE_CLASSES.
    let distance_price = |d| prices.get_table(d, |d| costs.distance(d));
    let mut choose = |local: usize| {
        let mut best = LazyChoice::default();
        if count - local < MIN_MATCH || prefix[count] == prefix[local] {
            return best;
        }
        let position = begin + local;
        // Guidance, carry, shared results and the bound are not passed by source LAZY.
        let query = Query {
            maximum: count - local,
            ..Query::default()
        };
        find(position, query, cache, &distance_price, &mut |matched| {
            let distance = distance_price(matched.symbol);
            visit_intervals(
                MIN_MATCH,
                matched.length + 1,
                config.length_uint_config,
                |first, stop| {
                    let length = stop - 1;
                    let bits = lazy_price(
                        costs.length(input[position].context, first, config),
                        distance,
                    );
                    let gain = prefix[local + length] - prefix[local] - f64::from(bits);
                    if gain > best.gain || (gain == best.gain && gain > 0.0 && length > best.length)
                    {
                        best = LazyChoice {
                            length,
                            symbol: matched.symbol,
                            gain,
                        };
                    }
                },
            );
        });
        best
    };
    let mut output = reserved(count)?;
    let mut local = 0;
    let mut current = LazyChoice::default();
    let mut pending = false;
    while local < count {
        if !pending {
            current = choose(local);
        }
        pending = false;
        if current.length >= MIN_MATCH && local + 1 < count {
            let next = choose(local + 1);
            if next.gain > current.gain {
                output.push(Command::Literal(input[begin + local]));
                local += 1;
                current = next;
                pending = true;
                continue;
            }
        }
        if current.length < MIN_MATCH {
            output.push(Command::Literal(input[begin + local]));
            local += 1;
        } else {
            output.push(Command::Match {
                context: input[begin + local].context,
                length: current.length as u32,
                distance: physical(current.symbol, width) as u32,
            });
            local += current.length;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod guided_tests;

#[cfg(test)]
mod pruning_tests;

#[cfg(test)]
mod dp_tests;

#[cfg(test)]
mod count_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_block_queries_replay_across_cuts_with_both_index_types() {
        use super::super::{SourceSuffixMode, StaticIndex, suffix::SuffixIndex};
        let values: Vec<u32> = (0..511).map(|i| 10000 + i % 17).collect();
        let input: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &value)| Literal {
                value,
                context: (i % 3) as u32,
            })
            .collect();
        for width in [0, 37] {
            for suffix_mode in [SourceSuffixMode::Disabled, SourceSuffixMode::SlidingLcp] {
                for block_mode in [SourceBlockMode::Endpoints, SourceBlockMode::Intervals] {
                    for cache_mode in 0..3 {
                        let config = SourceConfig {
                            suffix_mode,
                            block_mode,
                            index_word_match: true,
                            index_match_cache: cache_mode == 1,
                            block_sparse_cache: cache_mode == 2,
                            block_shared_frontier: true,
                            block_frontier_carry: true,
                            block_distance_bound: true,
                            block_distance_table: true,
                            block_distance_classes: true,
                            max_match_length: 67,
                            ..SourceConfig::default()
                        };
                        let costs = Costs::raw(&[&input], &config);
                        let prices =
                            DistancePrices::new(window(input.len()), width, &config, |d| {
                                costs.distance(d)
                            })
                            .unwrap();
                        let index = StaticIndex::new(&values, width, &config).unwrap();
                        let suffix = SuffixIndex::new(&values, width, &config).unwrap();
                        let find =
                            |p,
                             query: Query<'_>,
                             cache: &mut MatchCache,
                             price: &dyn Fn(u32) -> f32,
                             visit: &mut dyn FnMut(Match)| {
                                if let Some(suffix) = &suffix {
                                    suffix.find_query(p, query, price, visit, cache);
                                } else {
                                    index.find_query(p, query, price, visit, cache);
                                }
                            };
                        let mut decoded = Vec::new();
                        for cut in [0, 97, 223, input.len()].windows(2) {
                            let commands = parse_part(
                                &input,
                                cut[0],
                                cut[1],
                                width,
                                &config,
                                &costs,
                                &prices,
                                &[],
                                &find,
                            )
                            .unwrap();
                            let mut copies = 0;
                            for command in commands {
                                let position = decoded.len();
                                match command {
                                    Command::Literal(literal) => {
                                        assert_eq!(literal, input[position]);
                                        decoded.push(literal.value);
                                    }
                                    Command::Match {
                                        context,
                                        length,
                                        distance,
                                    } => {
                                        assert_eq!(context, input[position].context);
                                        assert!((3..=67).contains(&length));
                                        assert!(distance != 0 && distance as usize <= position);
                                        assert!(position + length as usize <= cut[1]);
                                        copies += 1;
                                        for _ in 0..length {
                                            decoded
                                                .push(decoded[decoded.len() - distance as usize]);
                                        }
                                    }
                                }
                            }
                            assert!(copies != 0);
                            assert_eq!(decoded.len(), cut[1]);
                        }
                        assert_eq!(decoded, values);
                    }
                }
            }
        }
    }

    #[test]
    fn overflow_requery_omits_shared_result_and_distance_bound() {
        let input = vec![
            Literal {
                context: 0,
                value: 65536
            };
            100
        ];
        let config = SourceConfig {
            block_mode: SourceBlockMode::Intervals,
            block_shared_frontier: true,
            block_frontier_carry: true,
            block_distance_bound: true,
            block_seed_guided: SourceSeedGuided::Hints,
            ..SourceConfig::default()
        };
        let costs = Costs::raw(&[&input], &config);
        let prices = DistancePrices::new(128, 0, &config, |d| costs.distance(d)).unwrap();
        let spans = [SeedSpan {
            begin: 34,
            end: input.len(),
            distance: 17,
        }];
        let replays = std::cell::Cell::new(0);
        let find = |p,
                    query: Query<'_>,
                    _: &mut MatchCache,
                    _: &dyn Fn(u32) -> f32,
                    visit: &mut dyn FnMut(Match)| {
            assert_eq!(query.hint_distance, 17);
            if let Some(frontier) = query.result {
                *frontier = Dominance::new();
                if query.maximum >= 19 {
                    for i in 0..17 {
                        frontier.add(3 + i, i as u32, i as f32);
                    }
                }
                assert!(query.bound.cost.is_finite());
            } else {
                assert_eq!(query.bound, super::super::search::DistanceBound::default());
                assert!(query.carry.is_some());
                assert!(p >= 34);
                replays.set(replays.get() + 1);
                visit(Match {
                    length: 19,
                    symbol: 16,
                    distance: 17,
                });
            }
        };
        let commands = parse_part(
            &input,
            34,
            input.len(),
            0,
            &config,
            &costs,
            &prices,
            &spans,
            &find,
        )
        .unwrap();
        assert!(replays.get() > 0);
        let mut decoded = vec![65536; 34];
        for command in commands {
            match command {
                Command::Literal(t) => decoded.push(t.value),
                Command::Match {
                    length, distance, ..
                } => {
                    for _ in 0..length {
                        decoded.push(decoded[decoded.len() - distance as usize]);
                    }
                }
            }
        }
        assert_eq!(decoded, vec![65536; input.len()]);
    }

    #[test]
    fn seed_splitting_requires_dominant_large_stream_and_installed_bounded_pool() {
        let input = vec![
            Literal {
                context: 0,
                value: 1
            };
            262144
        ];
        let other = vec![
            Literal {
                context: 0,
                value: 2
            };
            100000
        ];
        let config = SourceConfig {
            seed_blocks: SourceSeedBlocks::Warm { history: 32768 },
            parallelism: Some(super::super::SourceParallelism {
                max_workers: 2,
                scratch_bytes: 16 << 20,
                fast_streams: true,
            }),
            ..SourceConfig::default()
        };
        assert_eq!(seed_parts(&[&input], &config, true).unwrap().len(), 1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let parts = seed_parts(&[&input, &[]], &config, true).unwrap();
            assert_eq!(parts.len(), 3);
            assert_eq!(
                (parts[0].begin, parts[0].end, parts[0].prefix),
                (0, 131072, 0)
            );
            assert_eq!(
                (parts[1].begin, parts[1].end, parts[1].prefix),
                (98304, 262144, 32768)
            );
            assert_eq!((parts[2].owner, parts[2].begin, parts[2].end), (1, 0, 0));
            assert_eq!(
                seed_parts(&[&input, &other], &config, true).unwrap().len(),
                2
            );
            assert_eq!(seed_parts(&[&input], &config, false).unwrap().len(), 1);
            crate::execution::serial(|| {
                assert_eq!(seed_parts(&[&input], &config, true).unwrap().len(), 1)
            });
        });
    }

    #[test]
    fn indexed_seed_uses_positive_gain_and_double_prefix_costs() {
        let values: Vec<_> = (0..513).map(|i| 65536 + i % 17).collect();
        let input: Vec<_> = values
            .iter()
            .map(|&value| Literal { context: 0, value })
            .collect();
        let config = SourceConfig::default();
        let costs = Costs::raw(&[&input], &config);
        let index = super::super::StaticIndex::new(&values, 0, &config).unwrap();
        let (commands, saving, raw) = indexed_seed(&input, &index, &config, &costs).unwrap();
        assert!(saving > 0.0);
        let expected_raw: f64 = input
            .iter()
            .map(|t| {
                let code = config.literal_uint_config.encode(t.value);
                f64::from(code.nbits as f32 + costs.bits(t.context, code.token))
            })
            .sum();
        assert_eq!(raw, expected_raw as f32);
        let mut decoded = Vec::new();
        for command in &commands {
            match *command {
                Command::Literal(t) => decoded.push(t.value),
                Command::Match {
                    length, distance, ..
                } => {
                    for _ in 0..length {
                        decoded.push(decoded[decoded.len() - distance as usize]);
                    }
                }
            }
        }
        assert_eq!(decoded, values);
        let zeros = vec![0; 83];
        let input = vec![
            Literal {
                context: 0,
                value: 0
            };
            83
        ];
        let costs = Costs::raw(&[&input], &config);
        let index = super::super::StaticIndex::new(&zeros, 0, &config).unwrap();
        let (commands, saving, raw) = indexed_seed(&input, &index, &config, &costs).unwrap();
        assert_eq!((saving, raw), (0.0, 0.0));
        assert!(
            commands
                .iter()
                .all(|command| matches!(command, Command::Literal(_)))
        );
    }

    #[test]
    fn cuts_do_not_split_copies_or_duplicate_crossed_targets() {
        let input = vec![
            Literal {
                context: 0,
                value: 9
            };
            100
        ];
        let seed = [
            Command::Literal(input[0]),
            Command::Match {
                context: 0,
                length: 90,
                distance: 1,
            },
            Command::Match {
                context: 0,
                length: 9,
                distance: 1,
            },
        ];
        assert_eq!(cuts(&input, &seed, 8, false).unwrap(), [0, 91, 100]);
        assert_eq!(cuts(&input, &seed, 8, true).unwrap(), [0, 91, 100]);
        assert_eq!(cuts(&[], &[], 1, false).unwrap(), [0, 0]);
        assert!(cuts(&input, &seed[..2], 8, false).is_err());
        assert!(cuts(&input, &seed, 0, false).is_err());
    }

    #[test]
    fn warm_prefix_rebases_context_preserves_distance_and_literalizes_short_tail() {
        let input: Vec<_> = (0..10)
            .map(|i| Literal {
                context: i,
                value: 7,
            })
            .collect();
        let seed = [
            Command::Literal(input[0]),
            Command::Match {
                context: 1,
                length: 9,
                distance: 1,
            },
        ];
        for prefix in 0..=10 {
            let mut result = Vec::new();
            trim_prefix(&input, prefix, &seed, &mut result).unwrap();
            assert_eq!(
                result.iter().map(command_length).sum::<usize>(),
                10 - prefix
            );
            if (1..=7).contains(&prefix) {
                assert_eq!(
                    result,
                    [Command::Match {
                        context: prefix as u32,
                        length: (10 - prefix) as u32,
                        distance: 1
                    }]
                );
            } else if prefix >= 8 {
                assert_eq!(
                    result,
                    input[prefix..]
                        .iter()
                        .copied()
                        .map(Command::Literal)
                        .collect::<Vec<_>>()
                );
            }
        }
        assert!(trim_prefix(&input, 11, &seed, &mut Vec::new()).is_err());
    }

    #[test]
    fn block_destination_cap_does_not_reset_dictionary() {
        let values: Vec<_> = (0..83).map(|i| 65536 + i % 17).collect();
        let input: Vec<_> = values
            .iter()
            .map(|&value| Literal { context: 0, value })
            .collect();
        let mut config = SourceConfig {
            block_mode: SourceBlockMode::Intervals,
            ..SourceConfig::default()
        };
        let costs = Costs::raw(&[&input], &config);
        let index = super::super::StaticIndex::new(&values, 0, &config).unwrap();
        let find = |p,
                    query: Query<'_>,
                    cache: &mut MatchCache,
                    price: &dyn Fn(u32) -> f32,
                    visit: &mut dyn FnMut(Match)| {
            index.find_query(p, query, price, visit, cache);
        };
        let prices =
            DistancePrices::new(window(input.len()), 0, &config, |d| costs.distance(d)).unwrap();
        let queued = parse_part(
            &input,
            34,
            input.len(),
            0,
            &config,
            &costs,
            &prices,
            &[],
            &find,
        )
        .unwrap();
        assert!(queued.iter().any(|c| matches!(c, Command::Match { .. })));
        let mut decoded = values[..34].to_vec();
        for command in &queued {
            match *command {
                Command::Literal(t) => decoded.push(t.value),
                Command::Match {
                    length, distance, ..
                } => {
                    for _ in 0..length {
                        decoded.push(decoded[decoded.len() - distance as usize]);
                    }
                }
            }
        }
        assert_eq!(decoded, values);
        config.block_mode = SourceBlockMode::Endpoints;
        let scalar = parse_part(
            &input,
            34,
            input.len(),
            0,
            &config,
            &costs,
            &prices,
            &[],
            &find,
        )
        .unwrap();
        assert_eq!(queued, scalar);
    }
}
