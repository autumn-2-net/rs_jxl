// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_parse.h / enc_lz77_cost.h at b866b3d: word extension, exact
//! direct-mapped verified endpoints, carried Pareto endpoints and prices.
//! These are the source index/block helpers, not the legacy parser's cache.

use super::index::Dominance;
use super::matchers::filled;
use super::{SourceConfig, reserved};
use crate::{Error, Result};

// LZ77WordMatchLength<Value=u32>. Construct each LoadLE64 word explicitly:
// safe for any slice alignment and independent of the Rust host byte order.
pub(super) fn word_length(a: &[u32], b: &[u32]) -> usize {
    debug_assert_eq!(a.len(), b.len());
    let mut length = 0;
    while a.len() - length >= 2 {
        let left = u64::from(a[length]) | (u64::from(a[length + 1]) << 32);
        let right = u64::from(b[length]) | (u64::from(b[length + 1]) << 32);
        let diff = left ^ right;
        if diff != 0 {
            return length + diff.trailing_zeros() as usize / 32;
        }
        length += 2;
    }
    while length < a.len() && a[length] == b[length] {
        length += 1;
    }
    length
}

// LZ77WordMatchLength<Value=u16>: four whole values per unaligned LE64 word.
// Explicit assembly avoids unsafe loads and never reads beyond the slice tail.
pub(super) fn word_length_u16(a: &[u16], b: &[u16]) -> usize {
    debug_assert_eq!(a.len(), b.len());
    let mut length = 0;
    while a.len() - length >= 4 {
        let load = |values: &[u16]| {
            u64::from(values[length])
                | (u64::from(values[length + 1]) << 16)
                | (u64::from(values[length + 2]) << 32)
                | (u64::from(values[length + 3]) << 48)
        };
        let diff = load(a) ^ load(b);
        if diff != 0 {
            return length + diff.trailing_zeros() as usize / 16;
        }
        length += 4;
    }
    while length < a.len() && a[length] == b[length] {
        length += 1;
    }
    length
}

pub(super) fn match_length<Value: PartialEq>(a: &[Value], b: &[Value]) -> usize {
    debug_assert_eq!(a.len(), b.len());
    let mut length = 0;
    while a.len() - length >= 8 && a[length..length + 8] == b[length..length + 8] {
        length += 8;
    }
    while length < a.len() && a[length] == b[length] {
        length += 1;
    }
    length
}

// Fast seeds mostly stop in the first eight extension values. Word comparison
// locates that first mismatch without rescanning a failed eight-value block.
#[inline]
pub(super) fn fast_match_length(a: &[u32], b: &[u32]) -> usize {
    debug_assert_eq!(a.len(), b.len());
    let head = a.len().min(8);
    let matched = word_length(&a[..head], &b[..head]);
    if matched < head || head == a.len() {
        matched
    } else {
        head + match_length(&a[head..], &b[head..])
    }
}

pub(super) struct MatchCache {
    ends: Vec<u32>,
    distances: Vec<u32>,
    window: usize,
    position: usize,
}

impl MatchCache {
    pub(super) fn uncached() -> Self {
        Self {
            ends: Vec::new(),
            distances: Vec::new(),
            window: 0,
            position: 0,
        }
    }

    pub(super) fn new(window: usize) -> Result<Self> {
        Self::with_entries(window, 0)
    }

    fn with_entries(window: usize, entries: usize) -> Result<Self> {
        let size = if entries == 0 {
            window
        } else {
            entries.min(window)
        };
        // Source callers supply power-of-two windows and 4096 sparse slots.
        if window == 0 || window > super::WINDOW_SIZE || (entries != 0 && !size.is_power_of_two()) {
            return Err(Error::InvalidInput("invalid source LZ77 cache size".into()));
        }
        Ok(Self {
            ends: filled(size, 0)?,
            distances: filled(if entries == 0 { 0 } else { size }, 0)?,
            window,
            position: 0,
        })
    }

    pub(super) fn for_blocks(window: usize, config: &SourceConfig) -> Result<Self> {
        if config.block_sparse_cache {
            Self::with_entries(window, 4096)
        } else if config.index_match_cache {
            Self::new(window)
        } else {
            Ok(Self::uncached())
        }
    }

    pub(super) fn block_bytes(window: usize, config: &SourceConfig) -> usize {
        // Source reserves 32768 in its task bound even for a tiny window.
        if config.block_sparse_cache {
            32768
        } else if config.index_match_cache {
            4 * window
        } else {
            0
        }
    }

    pub(super) fn advance(&mut self, position: usize) {
        debug_assert!(position >= self.position);
        self.position = position;
    }

    fn slot(&self, distance: usize) -> Option<usize> {
        if distance == 0 || distance >= self.window {
            return None;
        }
        Some(if self.distances.is_empty() {
            distance
        } else {
            distance & (self.ends.len() - 1)
        })
    }

    pub(super) fn prefix(&self, distance: usize, limit: usize) -> usize {
        let Some(slot) = self.slot(distance) else {
            return 0;
        };
        if !self.distances.is_empty() && self.distances[slot] as usize != distance {
            return 0;
        }
        (self.ends[slot] as usize)
            .saturating_sub(self.position)
            .min(limit)
    }

    pub(super) fn remember(&mut self, distance: usize, length: usize) {
        let Some(slot) = self.slot(distance) else {
            return;
        };
        if !self.distances.is_empty() {
            self.distances[slot] = distance as u32;
        }
        // Overwrite even a same-distance shorter endpoint, as in source.
        self.ends[slot] = self.position.saturating_add(length).min(u32::MAX as usize) as u32;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DistanceBound {
    pub(super) cost: f32,
    pub(super) symbol: u32,
}

impl Default for DistanceBound {
    fn default() -> Self {
        Self {
            cost: f32::NEG_INFINITY,
            symbol: 0,
        }
    }
}

impl DistanceBound {
    pub(super) fn minimum(maximum: u32, price: impl Fn(u32) -> f32) -> Self {
        let mut result = Self {
            cost: f32::INFINITY,
            symbol: 0,
        };
        let mut first = 0u64;
        while first <= u64::from(maximum) {
            let cost = price(first as u32);
            if cost < result.cost {
                result = Self {
                    cost,
                    symbol: first as u32,
                };
            }
            first += if first < 16 {
                1
            } else {
                1u64 << (first.ilog2() - 2)
            };
        }
        result
    }

    pub(super) fn reached(self, length: usize, limit: usize, cost: f32, symbol: u32) -> bool {
        length == limit && (cost < self.cost || (cost == self.cost && symbol <= self.symbol))
    }
}

pub(super) struct DistanceClasses {
    prices: [f32; 128],
}

impl DistanceClasses {
    pub(super) fn new(price: impl Fn(u32) -> f32) -> Self {
        let mut prices = [0.0; 128];
        for symbol in 0..16 {
            prices[symbol as usize] = price(symbol);
        }
        for bit in 4..32 {
            for top in 0..4 {
                prices[(16 + 4 * (bit - 4) + top) as usize] = price((4 + top) << (bit - 2));
            }
        }
        Self { prices }
    }

    pub(super) fn get(&self, distance: u32) -> f32 {
        let class = if distance < 16 {
            distance
        } else {
            let bit = distance.ilog2();
            16 + 4 * (bit - 4) + ((distance >> (bit - 2)) & 3)
        };
        self.prices[class as usize]
    }
}

pub(super) struct DistancePrices {
    classes: Option<DistanceClasses>,
    table: Vec<f32>,
    pub(super) bound: DistanceBound,
}

impl DistancePrices {
    pub(super) fn new(
        window: usize,
        width: usize,
        config: &SourceConfig,
        price: impl Fn(u32) -> f32,
    ) -> Result<Self> {
        if window == 0 || window > super::WINDOW_SIZE {
            return Err(Error::InvalidInput(
                "invalid source LZ77 price window".into(),
            ));
        }
        let size = window + if width == 0 { 0 } else { 120 };
        let bound = if config.block_distance_bound {
            DistanceBound::minimum((size - 1) as u32, &price)
        } else {
            DistanceBound::default()
        };
        let classes = config
            .block_distance_classes
            .then(|| DistanceClasses::new(&price));
        let mut table = reserved(if config.block_distance_table { size } else { 0 })?;
        if config.block_distance_table {
            let mut begin = 0;
            while begin < size {
                let span = if begin < 16 {
                    1
                } else {
                    1 << (begin.ilog2() - 2)
                };
                let end = size.min(begin + span);
                table.resize(end, price(begin as u32));
                begin = end;
            }
        }
        Ok(Self {
            classes,
            table,
            bound,
        })
    }

    pub(super) fn get(&self, symbol: u32, direct: impl Fn(u32) -> f32) -> f32 {
        // Source builds both when requested, but classes take query precedence.
        if let Some(classes) = &self.classes {
            classes.get(symbol)
        } else {
            self.get_table(symbol, direct)
        }
    }

    pub(super) fn get_table(&self, symbol: u32, direct: impl Fn(u32) -> f32) -> f32 {
        if self.table.is_empty() {
            direct(symbol)
        } else {
            self.table[symbol as usize]
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Carried {
    end: usize,
    symbol: u32,
}

pub(super) struct FrontierCarry {
    entries: [Carried; 16],
    position: usize,
    size: usize,
}

impl FrontierCarry {
    pub(super) fn new() -> Self {
        Self {
            entries: [Carried::default(); 16],
            position: 0,
            size: 0,
        }
    }

    pub(super) fn remember(&mut self, position: usize, frontier: &Dominance) {
        self.position = position;
        self.size = 0;
        for entry in &frontier.entries[..frontier.size] {
            if let Some(end) = position.checked_add(entry.length) {
                self.entries[self.size] = Carried {
                    end,
                    symbol: entry.symbol,
                };
                self.size += 1;
            }
        }
    }

    pub(super) fn visit(
        &self,
        position: usize,
        minimum: usize,
        maximum: usize,
        mut visit: impl FnMut(usize, u32),
    ) {
        if position < self.position {
            return;
        }
        for entry in &self.entries[..self.size] {
            if entry.end <= position {
                continue;
            }
            let length = maximum.min(entry.end - position);
            if length >= minimum {
                visit(length, entry.symbol);
            }
        }
    }
}

pub(super) struct Query<'a> {
    pub(super) maximum: usize,
    pub(super) hint_distance: usize,
    pub(super) result: Option<&'a mut Dominance>,
    pub(super) bound: DistanceBound,
    pub(super) carry: Option<&'a FrontierCarry>,
}

impl Default for Query<'_> {
    fn default() -> Self {
        Self {
            maximum: usize::MAX,
            hint_distance: 0,
            result: None,
            bound: DistanceBound::default(),
            carry: None,
        }
    }
}

pub(super) fn physical(symbol: u32, width: usize) -> usize {
    if width != 0 && symbol < 120 {
        super::super::special_distance(symbol as usize, width) as usize
    } else {
        symbol as usize + 1 - if width == 0 { 0 } else { 120 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_extension_matches_scalar_at_every_bit_tail_and_slice_alignment() {
        let values: Vec<u32> = (0..40u32).map(|i| i.wrapping_mul(0x9e3779b9)).collect();
        for offset in 0..4 {
            for size in 0..=33 {
                let a = &values[offset..offset + size];
                assert_eq!(word_length(a, a), size);
                for mismatch in 0..size {
                    for bit in 0..32 {
                        let mut b = a.to_vec();
                        b[mismatch] ^= 1u32 << bit;
                        let expected = a.iter().zip(&b).take_while(|(a, b)| a == b).count();
                        assert_eq!(word_length(a, &b), expected);
                        assert_eq!(match_length(a, &b), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn source_cache_checks_sparse_tags_and_overwrites_shorter_endpoints() {
        for entries in [0, 4096] {
            let mut cache = MatchCache::with_entries(8192, entries).unwrap();
            cache.advance(9000);
            cache.remember(3, 100);
            assert_eq!(cache.prefix(3, 200), 100);
            cache.remember(4099, 7);
            assert_eq!(cache.prefix(3, 200), if entries == 0 { 100 } else { 0 });
            assert_eq!(cache.prefix(4099, 200), 7);
            cache.remember(3, 100);
            cache.remember(3, 5);
            cache.advance(9002);
            assert_eq!(cache.prefix(3, 2), 2);
            assert_eq!(cache.prefix(3, 200), 3);
            cache.advance(9005);
            assert_eq!(cache.prefix(3, 200), 0);
            for distance in [0, 8192, usize::MAX] {
                cache.remember(distance, 100);
                assert_eq!(cache.prefix(distance, 200), 0);
            }
            cache.advance(u32::MAX as usize - 2);
            cache.remember(3, 100);
            assert_eq!(cache.prefix(3, 100), 2);
        }
        let config = SourceConfig {
            block_sparse_cache: true,
            index_match_cache: true,
            ..SourceConfig::default()
        };
        let cache = MatchCache::for_blocks(32, &config).unwrap();
        assert_eq!((cache.ends.len(), cache.distances.len()), (32, 32));
        assert_eq!(MatchCache::block_bytes(32, &config), 32768);
    }

    fn price(symbol: u32) -> f32 {
        let encoded = crate::entropy::HybridUintConfig::new(4, 2, 0)
            .unwrap()
            .encode(symbol);
        encoded.nbits as f32 + ((encoded.token * 19) % 31) as f32 * 0.25
    }

    #[test]
    fn distance_classes_tables_and_bounds_equal_scalar_uint420_prices() {
        let classes = DistanceClasses::new(price);
        for symbol in 0..65536 {
            assert_eq!(classes.get(symbol), price(symbol));
        }
        for bit in 16..32 {
            for top in 0..4 {
                let first = (4u32 + top) << (bit - 2);
                let last = first | ((1u32 << (bit - 2)) - 1);
                for symbol in [first, first + 1, last] {
                    assert_eq!(classes.get(symbol), price(symbol));
                }
            }
        }
        for width in [0, 37] {
            for mask in 0..4 {
                let config = SourceConfig {
                    block_distance_table: mask & 1 != 0,
                    block_distance_classes: mask & 2 != 0,
                    block_distance_bound: true,
                    ..SourceConfig::default()
                };
                let prices = DistancePrices::new(8192, width, &config, price).unwrap();
                let size = 8192 + if width == 0 { 0 } else { 120 };
                let mut expected = DistanceBound {
                    cost: f32::INFINITY,
                    symbol: 0,
                };
                for symbol in 0..size {
                    assert_eq!(prices.get(symbol, price), price(symbol));
                    if price(symbol) < expected.cost {
                        expected = DistanceBound {
                            cost: price(symbol),
                            symbol,
                        };
                    }
                }
                assert_eq!(prices.bound, expected);
                assert_eq!(
                    prices.table.len(),
                    if mask & 1 != 0 { size as usize } else { 0 }
                );
            }
        }
        assert_eq!(
            DistanceBound::minimum(u32::MAX, |_| 7.0),
            DistanceBound {
                cost: 7.0,
                symbol: 0
            }
        );
        assert!(!DistanceBound::default().reached(100, 100, 0.0, 0));
    }

    #[test]
    fn distance_classes_win_when_both_source_price_caches_are_requested() {
        let calls = std::cell::Cell::new(0u32);
        let config = SourceConfig {
            block_distance_table: true,
            block_distance_classes: true,
            ..SourceConfig::default()
        };
        let prices = DistancePrices::new(32, 0, &config, |_| {
            calls.set(calls.get() + 1);
            calls.get() as f32
        })
        .unwrap();
        assert_ne!(prices.table[0], prices.classes.as_ref().unwrap().get(0));
        assert_eq!(
            prices.get(0, |_| panic!("direct price must not be used")),
            prices.classes.as_ref().unwrap().get(0)
        );
        assert_eq!(
            prices.get_table(0, |_| panic!("dense table must be used")),
            prices.table[0]
        );
        let classes_only = DistancePrices::new(
            32,
            0,
            &SourceConfig {
                block_distance_classes: true,
                ..SourceConfig::default()
            },
            |_| 7.0,
        )
        .unwrap();
        assert_eq!(classes_only.get(0, |_| 19.0), 7.0);
        assert_eq!(classes_only.get_table(0, |_| 19.0), 19.0);
    }

    #[test]
    fn carried_frontier_preserves_order_clips_expires_and_resets_on_empty() {
        let mut frontier = Dominance::new();
        for i in 0..17 {
            frontier.add(3 + i, i as u32, i as f32);
        }
        assert!(frontier.overflowed);
        let mut carry = FrontierCarry::new();
        carry.remember(100, &frontier);
        let mut actual = Vec::new();
        carry.visit(99, 3, 10, |n, d| actual.push((n, d)));
        assert!(actual.is_empty());
        carry.visit(104, 3, 10, |n, d| actual.push((n, d)));
        let expected: Vec<_> = (4..16).map(|i| ((i - 1).min(10), i as u32)).collect();
        assert_eq!(actual, expected);
        carry.remember(105, &Dominance::new());
        carry.visit(105, 3, 100, |_, _| {
            panic!("empty frontier must reset carry")
        });
        carry.remember(usize::MAX - 2, &frontier);
        assert_eq!(carry.size, 0);
    }
}
