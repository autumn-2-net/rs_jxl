// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_parse.h::LZ77RangeParser and enc_lz77_block.h option gates at
//! b866b3defec13494bfbaf705db93581cda118e37. In particular, queued offers
//! must not replace an equal-cost explicit endpoint already written earlier.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::super::dp::RangeParser;
use super::matchers::filled;
use super::{Command, Literal, SourceBlockMode, SourceConfig, reserved};
use crate::{Error, Result};

#[path = "lz77_source_relax.rs"]
mod vector;
use vector::{Update, VectorSteps};

/// DYADIC_DP=0..3. Only interval block parsing enables the selected tree.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceBlockDyadic {
    #[default]
    Disabled,
    /// 1: one endpoint per leaf.
    Endpoints,
    /// 2: 64 endpoints per leaf; boundary fragments are explicit.
    Blocks64,
    /// 3: 16 endpoints per leaf; boundary fragments are explicit.
    Blocks16,
}

impl SourceBlockDyadic {
    fn shift(self) -> Option<u32> {
        match self {
            Self::Disabled => None,
            Self::Endpoints => Some(0),
            Self::Blocks64 => Some(6),
            Self::Blocks16 => Some(4),
        }
    }
}

/// BLOCK_ENVELOPE=0..3; source order is 64, 16, 256 endpoints per block.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceBlockEnvelope {
    #[default]
    Disabled,
    Blocks64,
    Blocks16,
    Blocks256,
}

impl SourceBlockEnvelope {
    fn shift(self) -> u32 {
        match self {
            Self::Disabled => 0,
            Self::Blocks64 => 6,
            Self::Blocks16 => 4,
            Self::Blocks256 => 8,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Options {
    calendar: bool,
    vector: bool,
    dominance: bool,
    dyadic: SourceBlockDyadic,
    envelope: SourceBlockEnvelope,
}

impl Options {
    fn blocks(config: &SourceConfig) -> Self {
        let intervals = config.block_mode == SourceBlockMode::Intervals;
        Self {
            calendar: intervals && config.block_calendar,
            vector: config.block_vector_dp,
            dominance: config.block_range_dominance,
            dyadic: if intervals {
                config.block_dyadic
            } else {
                SourceBlockDyadic::Disabled
            },
            envelope: if intervals {
                config.block_envelope
            } else {
                SourceBlockEnvelope::Disabled
            },
        }
    }
}

#[derive(Clone, Copy)]
struct Step {
    cost: f32,
    length: u32,
    // Core commands use physical distances; zero still denotes a literal.
    distance: u32,
    context: u32,
}

#[derive(Clone, Copy)]
struct Offer {
    origin: usize,
    begin: usize,
    end: usize,
    distance: usize,
    context: u32,
    cost: f32,
}

impl Default for Offer {
    fn default() -> Self {
        Self {
            origin: 0,
            begin: 0,
            end: 0,
            distance: 0,
            context: 0,
            cost: f32::INFINITY,
        }
    }
}

#[derive(Clone, Copy)]
struct DyadicOffer {
    cost: f32,
    origin: u32,
    distance: u32,
    context: u32,
}

impl Default for DyadicOffer {
    fn default() -> Self {
        Self {
            cost: f32::INFINITY,
            origin: 0,
            distance: 0,
            context: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct Event {
    offer: Offer,
    next: u32,
}

#[derive(Clone, Copy)]
struct Queued<const ACTIVE: bool>(Offer);

impl<const ACTIVE: bool> Ord for Queued<ACTIVE> {
    fn cmp(&self, other: &Self) -> Ordering {
        if ACTIVE {
            other
                .0
                .cost
                .total_cmp(&self.0.cost)
                .then_with(|| other.0.origin.cmp(&self.0.origin))
        } else {
            other.0.begin.cmp(&self.0.begin)
        }
    }
}
impl<const ACTIVE: bool> PartialOrd for Queued<ACTIVE> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const ACTIVE: bool> PartialEq for Queued<ACTIVE> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const ACTIVE: bool> Eq for Queued<ACTIVE> {}

pub(super) struct HeapParser {
    steps: Vec<Step>,
    vector: Option<VectorSteps>,
    pending: BinaryHeap<Queued<false>>,
    active: BinaryHeap<Queued<true>>,
    queue_limit: usize,
    compact_at: usize,
    count: usize,
    heads: Vec<u32>,
    events: Vec<Event>,
    next_time: usize,
    pending_events: usize,
    free_event: u32,
    last_explicit: usize,
    block_shift: u32,
    block_best: Vec<Offer>,
    cover_costs: Vec<f32>,
    dyadic: Vec<DyadicOffer>,
    dyadic_base: usize,
    dyadic_depth: u32,
    dyadic_shift: u32,
    dyadic_previous: Option<usize>,
    dyadic_path: [DyadicOffer; usize::BITS as usize],
    #[cfg(test)]
    dominated: usize,
}

impl HeapParser {
    fn new(size: usize, queue_limit: usize) -> Result<Self> {
        Self::with_options(size, queue_limit, Options::default())
    }

    fn with_options(size: usize, queue_limit: usize, options: Options) -> Result<Self> {
        if size > i32::MAX as usize || queue_limit >= u32::MAX as usize {
            return Err(Error::Limit("LZ77 range indices"));
        }
        let count = size.checked_add(1).ok_or(Error::Limit("LZ77 range size"))?;
        let steps = filled(
            if options.vector { 0 } else { count },
            Step {
                cost: f32::INFINITY,
                length: 0,
                distance: 0,
                context: 0,
            },
        )?;
        let vector = if options.vector {
            Some(VectorSteps::new(count)?)
        } else {
            None
        };
        let block_shift = options.envelope.shift();
        let (mut dyadic_base, mut dyadic_depth) = (1usize, 0);
        let dyadic_shift = options.dyadic.shift().unwrap_or(0);
        let dyadic = if options.dyadic != SourceBlockDyadic::Disabled {
            while dyadic_base <= size >> dyadic_shift {
                dyadic_base = dyadic_base
                    .checked_mul(2)
                    .ok_or(Error::Limit("LZ77 dyadic size"))?;
                dyadic_depth += 1;
            }
            filled(
                dyadic_base
                    .checked_mul(2)
                    .ok_or(Error::Limit("LZ77 dyadic size"))?,
                DyadicOffer::default(),
            )?
        } else {
            Vec::new()
        };
        let mut parser = Self {
            steps,
            vector,
            pending: BinaryHeap::new(),
            active: BinaryHeap::new(),
            queue_limit,
            compact_at: 4096,
            count,
            heads: filled(if options.calendar { count } else { 0 }, u32::MAX)?,
            events: Vec::new(),
            next_time: 0,
            pending_events: 0,
            free_event: u32::MAX,
            last_explicit: 0,
            block_shift,
            block_best: filled(
                if block_shift != 0 {
                    (size >> block_shift) + 1
                } else {
                    0
                },
                Offer::default(),
            )?,
            cover_costs: filled(
                if options.dominance {
                    (size >> 4) + 1
                } else {
                    0
                },
                f32::INFINITY,
            )?,
            dyadic,
            dyadic_base,
            dyadic_depth,
            dyadic_shift,
            dyadic_previous: None,
            dyadic_path: [DyadicOffer::default(); usize::BITS as usize],
            #[cfg(test)]
            dominated: 0,
        };
        parser.relax(0, 0, 0, 0.0);
        Ok(parser)
    }

    fn at(&self, pos: usize) -> Step {
        self.vector
            .as_ref()
            .map_or_else(|| self.steps[pos], |v| v.at(pos))
    }

    fn relax(&mut self, pos: usize, origin: usize, distance: usize, cost: f32) {
        self.relax_context(pos, origin, distance, 0, cost);
    }

    fn relax_context(
        &mut self,
        pos: usize,
        origin: usize,
        distance: usize,
        context: u32,
        cost: f32,
    ) {
        if cost < self.at(pos).cost {
            let step = Step {
                cost,
                length: (pos - origin) as u32,
                distance: distance as u32,
                context,
            };
            if let Some(vector) = &mut self.vector {
                vector.set(pos, step);
            } else {
                self.steps[pos] = step;
            }
        }
    }

    fn advance(&mut self, pos: usize) {
        if !self.dyadic.is_empty() {
            let block = pos >> self.dyadic_shift;
            let first = match self.dyadic_previous {
                None => 0,
                Some(previous) if previous == block => self.dyadic_depth + 1,
                Some(previous) => {
                    debug_assert!(block > previous);
                    self.dyadic_depth - (block ^ previous).ilog2()
                }
            };
            let mut best = if first == 0 {
                DyadicOffer::default()
            } else {
                self.dyadic_path[first as usize - 1]
            };
            for depth in first..=self.dyadic_depth {
                let offer = self.dyadic[(self.dyadic_base + block) >> (self.dyadic_depth - depth)];
                if offer.cost < best.cost {
                    best = offer;
                }
                self.dyadic_path[depth as usize] = best;
            }
            self.dyadic_previous = Some(block);
            if best.cost != f32::INFINITY {
                self.relax_context(
                    pos,
                    best.origin as usize,
                    best.distance as usize,
                    best.context,
                    best.cost,
                );
            }
            return;
        }
        if !self.block_best.is_empty() {
            let offer = self.block_best[pos >> self.block_shift];
            if offer.cost != f32::INFINITY {
                self.relax_context(pos, offer.origin, offer.distance, offer.context, offer.cost);
            }
            return;
        }
        if !self.heads.is_empty() {
            while self.next_time <= pos {
                let mut id = self.heads[self.next_time];
                self.next_time += 1;
                while id != u32::MAX {
                    let event = self.events[id as usize];
                    self.events[id as usize].next = self.free_event;
                    self.free_event = id;
                    self.pending_events -= 1;
                    if event.offer.end > pos {
                        self.active.push(Queued(event.offer));
                    }
                    id = event.next;
                }
            }
        }
        while self
            .pending
            .peek()
            .is_some_and(|entry| entry.0.begin <= pos)
        {
            let offer = self.pending.pop().unwrap().0;
            if offer.end > pos {
                self.active.push(Queued(offer));
            }
        }
        while self.active.peek().is_some_and(|entry| entry.0.end <= pos) {
            self.active.pop();
        }
        if self.active.len() >= self.compact_at {
            self.active.retain(|entry| entry.0.end > pos);
            self.compact_at = 2 * 4096.max(self.active.len());
        }
        if let Some(&Queued(offer)) = self.active.peek() {
            self.relax_context(pos, offer.origin, offer.distance, offer.context, offer.cost);
        }
    }

    fn jump_literal_only(&mut self, pos: usize, limit: usize) -> usize {
        if self.vector.is_some()
            || !self.dyadic.is_empty()
            || !self.heads.is_empty()
            || self.active.is_empty()
            || pos < self.last_explicit
            || limit <= pos
        {
            return pos;
        }
        let offer = self.active.peek().unwrap().0;
        if offer.end <= pos + 1 || offer.cost > self.at(pos).cost {
            return pos;
        }
        let mut target = limit.min(offer.end - 1);
        if let Some(pending) = self.pending.peek() {
            if pending.0.begin <= pos + 1 {
                return pos;
            }
            target = target.min(pending.0.begin - 1);
        }
        if target > pos {
            self.relax_context(
                target,
                offer.origin,
                offer.distance,
                offer.context,
                offer.cost,
            );
        }
        target
    }

    fn queued(&self) -> usize {
        self.pending.len() + self.pending_events + self.active.len()
    }

    fn interval(&mut self, origin: usize, begin: usize, end: usize, distance: usize, cost: f32) {
        self.interval_context(origin, begin, end, distance, 0, cost);
    }

    // Keep the source origin, relative endpoints, physical distance and context explicit.
    #[allow(clippy::too_many_arguments)]
    fn interval_context(
        &mut self,
        origin: usize,
        begin: usize,
        end: usize,
        distance: usize,
        context: u32,
        cost: f32,
    ) {
        debug_assert!(begin > 0 && begin < end && origin + end <= self.count);
        if !self.cover_costs.is_empty() {
            let (first, last) = (origin + begin, origin + end);
            let (first_block, last_block) = (first >> 4, (last - 1) >> 4);
            if last_block - first_block < 4
                && !self.cover_costs[first_block..=last_block]
                    .iter()
                    .any(|&covered| covered > cost)
            {
                #[cfg(test)]
                {
                    self.dominated += 1;
                }
                return;
            }
            let first_full = (first + 15) >> 4;
            let end_full = (last >> 4).min(first_full + 32);
            for b in first_full..end_full {
                self.cover_costs[b] = self.cover_costs[b].min(cost);
            }
        }
        if !self.dyadic.is_empty() {
            let offer = DyadicOffer {
                cost,
                origin: origin as u32,
                distance: distance as u32,
                context,
            };
            let width = 1usize << self.dyadic_shift;
            let (mut first, last) = (origin + begin, origin + end);
            let aligned = last.min((first + width - 1) & !(width - 1));
            while first < aligned {
                self.relax_context(first, origin, distance, context, cost);
                first += 1;
            }
            let full_end = first.max(last & !(width - 1));
            for pos in full_end..last {
                self.relax_context(pos, origin, distance, context, cost);
            }
            let mut left = self.dyadic_base + (first >> self.dyadic_shift);
            let mut right = self.dyadic_base + (full_end >> self.dyadic_shift);
            debug_assert!(
                left == right
                    || self
                        .dyadic_previous
                        .is_none_or(|previous| (first >> self.dyadic_shift) > previous)
            );
            while left < right {
                if left & 1 != 0 {
                    if cost < self.dyadic[left].cost {
                        self.dyadic[left] = offer;
                    }
                    left += 1;
                }
                if right & 1 != 0 {
                    right -= 1;
                    if cost < self.dyadic[right].cost {
                        self.dyadic[right] = offer;
                    }
                }
                left >>= 1;
                right >>= 1;
            }
            return;
        }
        let offer = Offer {
            origin,
            begin: origin + begin,
            end: origin + end,
            distance,
            context,
            cost,
        };
        if !self.block_best.is_empty() {
            let width = 1usize << self.block_shift;
            let (mut first, last) = (origin + begin, origin + end);
            let aligned = last.min((first + width - 1) & !(width - 1));
            while first < aligned {
                self.relax_context(first, origin, distance, context, cost);
                first += 1;
            }
            while last - first >= width {
                let best = &mut self.block_best[first >> self.block_shift];
                if cost < best.cost {
                    *best = offer;
                }
                first += width;
            }
            while first < last {
                self.relax_context(first, origin, distance, context, cost);
                first += 1;
            }
            return;
        }
        if end - begin <= 16 || self.queued() >= self.queue_limit {
            self.last_explicit = self.last_explicit.max(origin + end - 1);
            if let Some(vector) = &mut self.vector {
                vector.update(Update {
                    start: origin + begin,
                    end: origin + end,
                    first_length: begin as u32,
                    distance: distance as u32,
                    context,
                    cost,
                });
                return;
            }
            for length in begin..end {
                self.relax_context(origin + length, origin, distance, context, cost);
            }
        } else if self.heads.is_empty() {
            self.pending.push(Queued(offer));
        } else {
            let mut id = self.free_event;
            let event = Event {
                offer,
                next: self.heads[offer.begin],
            };
            if id == u32::MAX {
                id = self.events.len() as u32;
                self.events.push(event);
            } else {
                self.free_event = self.events[id as usize].next;
                self.events[id as usize] = event;
            }
            self.heads[offer.begin] = id;
            self.pending_events += 1;
        }
    }

    fn backtrack(&self, input: &[Literal]) -> Result<Vec<Command>> {
        let mut output = reserved(input.len())?;
        let mut end = input.len();
        while end != 0 {
            let step = self.at(end);
            if step.length == 0 || step.length as usize > end {
                return Err(Error::Encoding("invalid source range backtrack".into()));
            }
            end -= step.length as usize;
            output.push(if step.distance == 0 {
                Command::Literal(input[end])
            } else {
                Command::Match {
                    context: input[end].context,
                    length: step.length,
                    distance: step.distance,
                }
            });
        }
        output.reverse();
        Ok(output)
    }
}

pub(super) enum Parser {
    Endpoint(Box<RangeParser>),
    Heap(Box<HeapParser>),
}

impl Parser {
    pub(super) fn block_scratch(size: usize, config: &SourceConfig) -> usize {
        let options = Options::blocks(config);
        // Include retained heap/event capacities and all requested arrays, even
        // if a higher-precedence variant bypasses them. Output and literal-price
        // scratch are covered by the caller's existing 64 bytes per endpoint.
        let mut bytes = size.saturating_add(1).saturating_mul(64);
        if options.calendar {
            bytes = bytes.saturating_add(size.saturating_add(1).saturating_mul(4));
        }
        if options.dominance {
            bytes = bytes.saturating_add((size / 16 + 1).saturating_mul(4));
        }
        if options.envelope != SourceBlockEnvelope::Disabled {
            bytes = bytes.saturating_add(
                ((size >> options.envelope.shift()) + 1)
                    .saturating_mul(std::mem::size_of::<Offer>()),
            );
        }
        if let Some(shift) = options.dyadic.shift() {
            let base = (size >> shift)
                .saturating_add(1)
                .checked_next_power_of_two()
                .unwrap_or(usize::MAX);
            bytes = bytes.saturating_add(
                base.saturating_mul(2)
                    .saturating_mul(std::mem::size_of::<DyadicOffer>()),
            );
        }
        // The caller's 8 MiB fixed allowance covers two heaps plus recycled
        // events, each retaining at most 65536 slots (128 bytes combined).
        bytes
    }

    pub(super) fn blocks(size: usize, config: &SourceConfig) -> Result<Self> {
        Ok(Self::Heap(Box::new(HeapParser::with_options(
            size,
            if config.block_mode == SourceBlockMode::Intervals {
                1 << 16
            } else {
                0
            },
            Options::blocks(config),
        )?)))
    }

    pub(super) fn new(size: usize, range: bool, compact: bool, width: usize) -> Result<Self> {
        if range {
            Ok(Self::Heap(Box::new(HeapParser::new(size, 1 << 18)?)))
        } else if compact {
            Ok(Self::Endpoint(Box::new(RangeParser::new_source_compact(
                size, width,
            )?)))
        } else {
            Ok(Self::Endpoint(Box::new(RangeParser::new_source(size)?)))
        }
    }
    pub(super) fn advance(&mut self, pos: usize) {
        match self {
            Self::Endpoint(p) => p.advance(pos),
            Self::Heap(p) => p.advance(pos),
        }
    }
    pub(super) fn cost(&self, pos: usize) -> f64 {
        match self {
            Self::Endpoint(p) => p.cost(pos),
            Self::Heap(p) => f64::from(p.at(pos).cost),
        }
    }
    pub(super) fn literal(&mut self, pos: usize, cost: f64) {
        match self {
            Self::Endpoint(p) => p.literal(pos, cost),
            Self::Heap(p) => p.relax(pos + 1, pos, 0, cost as f32),
        }
    }
    pub(super) fn block_literal(&mut self, pos: usize, context: u32, cost: f32) {
        let Self::Heap(p) = self else {
            unreachable!("block parser is not an endpoint parser")
        };
        p.relax_context(pos + 1, pos, 0, context, cost);
    }

    pub(super) fn jump_literal_only(&mut self, pos: usize, limit: usize) -> usize {
        match self {
            Self::Heap(p) => p.jump_literal_only(pos, limit),
            Self::Endpoint(_) => pos,
        }
    }

    // Match the source's explicit block-origin/context/distance interface.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_interval(
        &mut self,
        origin: usize,
        begin: usize,
        end: usize,
        distance: usize,
        context: u32,
        cost: f32,
    ) {
        let Self::Heap(p) = self else {
            unreachable!("block parser is not an endpoint parser")
        };
        p.interval_context(origin, begin, end, distance, context, cost);
    }
    pub(super) fn interval(
        &mut self,
        origin: usize,
        begin: usize,
        end: usize,
        distance: usize,
        cost: f64,
    ) {
        match self {
            Self::Endpoint(p) => p.interval(origin, begin, end, distance, cost),
            Self::Heap(p) => p.interval(origin, begin, end, distance, cost as f32),
        }
    }
    pub(super) fn backtrack(&self, input: &[Literal]) -> Result<Vec<Command>> {
        match self {
            Self::Endpoint(p) => p.backtrack(input),
            Self::Heap(p) => p.backtrack(input),
        }
    }
}

#[cfg(test)]
#[path = "lz77_source_range_tests.rs"]
mod variant_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precise_endpoint_costs_are_not_narrowed_to_float() {
        let mut parser = Parser::new(100, false, false, 0).unwrap();
        parser.interval(0, 3, 90, 1, 16_777_217.0);
        parser.advance(50);
        assert_eq!(parser.cost(50), 16_777_217.0);
        parser.literal(50, 16_777_217.25);
        parser.advance(51);
        assert_eq!(parser.cost(51), 16_777_217.0);
        parser.advance(90);
        parser.literal(90, 16_777_217.25);
        parser.advance(91);
        assert_eq!(parser.cost(91), 16_777_217.25);
    }

    #[test]
    fn queued_ties_keep_explicit_endpoints_and_expire_at_exclusive_end() {
        let mut parser = HeapParser::new(64, 1 << 18).unwrap();
        parser.interval(0, 3, 40, 1, 10.0);
        parser.interval(1, 20, 22, 2, 10.0);
        parser.advance(21);
        assert_eq!(parser.steps[21].distance, 2);
        parser.advance(22);
        assert_eq!(parser.steps[22].distance, 2);
        parser.advance(23);
        assert_eq!(parser.steps[23].distance, 1);
        parser.advance(40);
        assert!(parser.steps[40].cost.is_infinite());
    }

    #[test]
    fn queue_limit_falls_back_to_exact_endpoints_and_heap_prefers_earlier_origin() {
        for limit in [0, 1, 1 << 18] {
            let mut parser = HeapParser::new(100, limit).unwrap();
            let mut expected = [f32::INFINITY; 101];
            expected[0] = 0.0;
            for origin in 0..32 {
                parser.advance(origin);
                assert_eq!(parser.steps[origin].cost, expected[origin]);
                parser.relax(origin + 1, origin, 0, 100.0 + origin as f32);
                expected[origin + 1] = expected[origin + 1].min(100.0 + origin as f32);
                let end = 61 - origin;
                let cost = (origin % 7) as f32;
                parser.interval(origin, 3, end, origin + 1, cost);
                for step in &mut expected[origin + 3..origin + end] {
                    *step = step.min(cost);
                }
            }
            for (pos, &cost) in expected.iter().enumerate().skip(32) {
                parser.advance(pos);
                assert_eq!(parser.steps[pos].cost, cost);
            }
        }
        let mut parser = HeapParser::new(100, 1 << 18).unwrap();
        parser.interval(1, 3, 90, 2, 4.0);
        parser.interval(0, 3, 90, 1, 4.0);
        parser.advance(50);
        assert_eq!(parser.steps[50].distance, 1);
    }
}
