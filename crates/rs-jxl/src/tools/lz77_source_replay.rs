// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77.cc::LZ77CachedMatches / ApplyLZ77_RangeParsed, frozen at
//! b866b3defec13494bfbaf705db93581cda118e37. The 32 MiB payload budget is
//! shared across streams; spilling one stream does not disable other caches.

use super::super::{HybridUintConfig, Match, reserved, special_distance};
use crate::{Error, Result};

pub(super) const CACHE_LIMIT: usize = 32 << 20;

/// Preserve discovery order while reducing equal uint420 distance classes.
pub(super) struct Classes {
    pub(super) entries: Vec<Match>,
    slots: [usize; 128],
}

impl Classes {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            entries: reserved(128)?,
            slots: [usize::MAX; 128],
        })
    }
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.slots.fill(usize::MAX);
    }
    pub(super) fn add(&mut self, matched: Match) {
        let class = HybridUintConfig::default().encode(matched.symbol).token as usize;
        let slot = self.slots[class];
        if slot == usize::MAX {
            self.slots[class] = self.entries.len();
            self.entries.push(matched);
        } else {
            let kept = &mut self.entries[slot];
            if matched.length > kept.length
                || (matched.length == kept.length && matched.symbol < kept.symbol)
            {
                *kept = matched;
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Position {
    position: usize,
    offset: u32,
    count: u32,
}

#[derive(Clone, Copy)]
struct CachedMatch {
    length: u32,
    symbol: u32,
}

#[derive(Default)]
pub(super) struct Cache {
    positions: Vec<Position>,
    matches: Vec<CachedMatch>,
    // None is the source SIZE_MAX sentinel, including empty full caches.
    replay_limit: Option<usize>,
}

pub(super) struct Budget {
    pub(super) used: usize,
    limit: usize,
}

impl Budget {
    pub(super) fn new(limit: usize) -> Self {
        Self { used: 0, limit }
    }
}

pub(super) struct Pass<'a> {
    cache: &'a mut Cache,
    budget: &'a mut Budget,
    second: bool,
    cursor: usize,
}

impl<'a> Pass<'a> {
    pub(super) fn new(cache: &'a mut Cache, budget: &'a mut Budget, second: bool) -> Self {
        Self {
            cache,
            budget,
            second,
            cursor: 0,
        }
    }
    pub(super) fn full_replay(&self) -> bool {
        self.second && self.cache.replay_limit.is_none()
    }
    #[cfg(test)]
    pub(super) fn partial_replay(&self) -> bool {
        self.second && self.cache.replay_limit.is_some()
    }
    pub(super) fn replaying(&self, position: usize) -> bool {
        self.second && self.cache.replay_limit.is_none_or(|limit| position < limit)
    }
    pub(super) fn load(&mut self, position: usize, width: usize, classes: &mut Classes) -> bool {
        let Some(saved) = self.cache.positions.get(self.cursor) else {
            return false;
        };
        if saved.position != position {
            return false;
        }
        self.cursor += 1;
        let begin = saved.offset as usize;
        for matched in &self.cache.matches[begin..begin + saved.count as usize] {
            let distance = if width != 0 && matched.symbol < 120 {
                special_distance(matched.symbol as usize, width) as usize
            } else {
                matched.symbol as usize + 1 - if width == 0 { 0 } else { 120 }
            };
            classes.entries.push(Match {
                length: matched.length as usize,
                distance,
                symbol: matched.symbol,
            });
        }
        true
    }
    pub(super) fn record(&mut self, position: usize, matches: &[Match]) -> Result<()> {
        if self.second
            || matches.is_empty()
            || self
                .cache
                .replay_limit
                .is_some_and(|limit| position >= limit)
        {
            return Ok(());
        }
        // These are source layouts, not Rust's larger usize-based Match. The
        // stored match payload is also exactly two u32s, avoiding a fake bound.
        let bytes =
            std::mem::size_of::<Position>() + matches.len() * std::mem::size_of::<CachedMatch>();
        if bytes > self.budget.limit - self.budget.used {
            self.cache.replay_limit = Some(position);
            return Ok(());
        }
        self.cache
            .positions
            .try_reserve(1)
            .map_err(|_| Error::Limit("LZ77 replay positions"))?;
        self.cache
            .matches
            .try_reserve(matches.len())
            .map_err(|_| Error::Limit("LZ77 replay matches"))?;
        self.cache.positions.push(Position {
            position,
            offset: self.cache.matches.len() as u32,
            count: matches.len() as u32,
        });
        self.cache
            .matches
            .extend(matches.iter().map(|m| CachedMatch {
                length: m.length as u32,
                symbol: m.symbol,
            }));
        self.budget.used += bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::DistanceCodes;
    use super::*;

    fn matched(length: usize, distance: usize, width: usize) -> Match {
        Match {
            length,
            distance,
            symbol: DistanceCodes::new(width, 1 << 20).symbol(distance),
        }
    }

    #[test]
    fn class_reduction_keeps_discovery_order_longest_and_then_smallest_symbol() {
        let mut classes = Classes::new().unwrap();
        for (length, symbol) in [(6, 31), (7, 5), (5, 30), (6, 28), (9, 29)] {
            classes.add(Match {
                length,
                symbol,
                distance: symbol as usize + 1,
            });
        }
        assert_eq!(
            classes
                .entries
                .iter()
                .map(|m| (m.length, m.symbol))
                .collect::<Vec<_>>(),
            [(9, 29), (7, 5)]
        );
        classes.clear();
        classes.add(Match {
            length: 7,
            symbol: 30,
            distance: 31,
        });
        classes.add(Match {
            length: 7,
            symbol: 28,
            distance: 29,
        });
        assert_eq!(classes.entries[0].symbol, 28);
    }

    #[test]
    fn replay_budget_is_shared_and_spill_is_per_stream_at_exact_position() {
        assert_eq!(std::mem::size_of::<CachedMatch>(), 8);
        let bytes = std::mem::size_of::<Position>() + 8;
        let mut budget = Budget::new(bytes * 2);
        let mut first = Cache::default();
        let mut second = Cache::default();
        let candidate = matched(33, 7, 0);
        Pass::new(&mut first, &mut budget, false)
            .record(7, &[candidate])
            .unwrap();
        Pass::new(&mut second, &mut budget, false)
            .record(8, &[candidate])
            .unwrap();
        assert_eq!(budget.used, bytes * 2);
        Pass::new(&mut first, &mut budget, false)
            .record(9, &[candidate])
            .unwrap();
        assert_eq!(first.replay_limit, Some(9));
        assert!(second.replay_limit.is_none());
        let mut replay = Pass::new(&mut first, &mut budget, true);
        assert!(!replay.full_replay());
        assert!(replay.replaying(8));
        assert!(!replay.replaying(9));
        let mut classes = Classes::new().unwrap();
        assert!(!replay.load(6, 0, &mut classes));
        assert!(replay.load(7, 0, &mut classes));
        assert_eq!(
            (classes.entries[0].length, classes.entries[0].distance),
            (33, 7)
        );
        assert!(!replay.load(8, 0, &mut classes));
    }

    #[test]
    fn cached_symbols_recover_physical_distances_including_spatial_aliases() {
        for width in [0, 1, 2, 37] {
            let mut cache = Cache::default();
            let mut budget = Budget::new(CACHE_LIMIT);
            let original: Vec<_> = [1, 2, 7, 37, 500]
                .into_iter()
                .map(|d| matched(91, d, width))
                .collect();
            Pass::new(&mut cache, &mut budget, false)
                .record(500, &original)
                .unwrap();
            let mut pass = Pass::new(&mut cache, &mut budget, true);
            assert!(pass.full_replay());
            let mut classes = Classes::new().unwrap();
            assert!(pass.load(500, width, &mut classes));
            assert_eq!(
                classes
                    .entries
                    .iter()
                    .map(|m| (m.length, m.distance, m.symbol))
                    .collect::<Vec<_>>(),
                original
                    .iter()
                    .map(|m| (m.length, m.distance, m.symbol))
                    .collect::<Vec<_>>()
            );
        }
    }
}
