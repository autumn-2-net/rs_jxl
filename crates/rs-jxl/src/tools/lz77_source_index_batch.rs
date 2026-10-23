// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77.cc's 16384-position indexed match batches at
//! b866b3defec13494bfbaf705db93581cda118e37. Overflow re-queries the index.

use super::super::{reserved, workers};
use super::{Dominance, Match, MatchCache, SourceConfig, StaticIndex, filled, special_distance};
use crate::Result;

const MATCH_BATCH: usize = 16384;

pub(in super::super) struct Batch {
    matches: Vec<Dominance>,
    caches: Vec<MatchCache>,
    begin: usize,
    end: usize,
    width: usize,
    #[cfg(test)]
    pub(in super::super) overflows: usize,
    #[cfg(test)]
    pub(in super::super) positions: usize,
}

impl Batch {
    pub(in super::super) fn has_bounded_frontier(&self, position: usize) -> bool {
        position >= self.begin
            && position < self.end
            && !self.matches[position - self.begin].overflowed
    }

    #[cfg(test)]
    pub(in super::super) fn tasks(&self) -> usize {
        self.caches.len()
    }

    pub(in super::super) fn new(
        index: &StaticIndex<'_>,
        width: usize,
        config: &SourceConfig,
    ) -> Result<Self> {
        // INDEX_CACHE=2 is implemented only by source block parsing.
        let cached = config.index_match_cache && !config.block_sparse_cache;
        let scratch = if cached {
            index.window * size_of::<u32>()
        } else {
            1 << 20
        };
        let tasks = workers::index_tasks(config, MATCH_BATCH / 1024, scratch);
        let mut caches = reserved(tasks)?;
        for _ in 0..tasks {
            caches.push(if cached {
                MatchCache::new(index.window)?
            } else {
                MatchCache::uncached()
            });
        }
        Ok(Self {
            matches: filled(MATCH_BATCH, Dominance::new())?,
            caches,
            begin: 0,
            end: 0,
            width,
            #[cfg(test)]
            overflows: 0,
            #[cfg(test)]
            positions: 0,
        })
    }

    pub(in super::super) fn find_matches(
        &mut self,
        index: &StaticIndex<'_>,
        position: usize,
        cost: impl Fn(u32) -> f32 + Sync,
        mut visit: impl FnMut(Match),
    ) {
        debug_assert!(position >= self.begin);
        if position >= self.end {
            self.begin = position;
            self.end = index.values.len().min(position + MATCH_BATCH);
            let count = self.end - self.begin;
            let tasks = self.caches.len().min(count.div_ceil(1024));
            let mut remaining = &mut self.matches[..count];
            let mut parts = Vec::with_capacity(tasks);
            for (task, cache) in self.caches[..tasks].iter_mut().enumerate() {
                let begin = count * task / tasks;
                let end = count * (task + 1) / tasks;
                let (frontiers, rest) = remaining.split_at_mut(end - begin);
                remaining = rest;
                parts.push((self.begin + begin, frontiers, cache));
            }
            workers::each(&mut parts, |_, (begin, frontiers, cache)| {
                for (offset, frontier) in frontiers.iter_mut().enumerate() {
                    *frontier = Dominance::new();
                    index.find_cached_matches(
                        *begin + offset,
                        &cost,
                        |matched| {
                            frontier.add(matched.length, matched.symbol, cost(matched.symbol));
                        },
                        cache,
                    );
                }
            });
            #[cfg(test)]
            {
                self.positions += count;
            }
        }
        let frontier = &self.matches[position - self.begin];
        if frontier.overflowed {
            #[cfg(test)]
            {
                self.overflows += 1;
            }
            // Do not emit a truncated 16-entry result. Source repeats this
            // query uncached and streams every survivor to the DP frontier.
            index.find_matches(position, cost, visit);
        } else {
            for entry in &frontier.entries[..frontier.size] {
                let distance = if self.width != 0 && entry.symbol < 120 {
                    special_distance(entry.symbol as usize, self.width) as usize
                } else {
                    entry.symbol as usize + 1 - if self.width == 0 { 0 } else { 120 }
                };
                visit(Match {
                    length: entry.length,
                    distance,
                    symbol: entry.symbol,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::super::dp::Frontier;
    use super::*;

    fn tuples(
        index: &StaticIndex<'_>,
        position: usize,
        batch: Option<&mut Batch>,
    ) -> Vec<(usize, usize, u32)> {
        let mut frontier = Frontier::new(100).unwrap();
        let cost = |symbol| ((symbol * 17) % 31) as f32;
        let add = |matched: Match| frontier.add(matched, f64::from(cost(matched.symbol)));
        if let Some(batch) = batch {
            batch.find_matches(index, position, cost, add);
        } else {
            index.find_matches(position, cost, add);
        }
        frontier
            .entries
            .iter()
            .map(|entry| {
                (
                    entry.matched.length,
                    entry.matched.distance,
                    entry.matched.symbol,
                )
            })
            .collect()
    }

    #[test]
    fn cached_batches_preserve_frontiers_across_skips_refills_and_spatial_symbols() {
        let values: Vec<_> = (0..17033)
            .map(|i| if i % 191 < 51 { 0 } else { (i % 97) as u32 })
            .collect();
        for width in [0, 37] {
            let mut expected = None;
            for index_match_cache in [false, true] {
                let config = SourceConfig {
                    index_match_cache,
                    max_match_length: 127,
                    ..SourceConfig::default()
                };
                let index = StaticIndex::new(&values, width, &config).unwrap();
                let mut batch = Batch::new(&index, width, &config).unwrap();
                let mut results = Vec::new();
                let (mut expected_positions, mut end) = (0, 0);
                for position in (0..values.len()).step_by(37).chain([values.len() - 1]) {
                    if position >= end {
                        end = values.len().min(position + MATCH_BATCH);
                        expected_positions += end - position;
                    }
                    let actual = tuples(&index, position, Some(&mut batch));
                    assert_eq!(
                        actual,
                        tuples(&index, position, None),
                        "position={position}"
                    );
                    results.push(actual);
                }
                assert_eq!(batch.tasks(), 1);
                assert_eq!(batch.positions, expected_positions);
                if let Some(expected) = &expected {
                    assert_eq!(&results, expected);
                }
                expected = Some(results);
            }
        }
    }

    #[test]
    fn overflow_requeries_all_twenty_two_candidates_in_source_order() {
        let prefix: Vec<u32> = (100..140).collect();
        let mut values = Vec::new();
        for length in (3..=24).rev() {
            values.extend_from_slice(&prefix[..length]);
            values.push(10000 + length as u32);
        }
        let position = values.len();
        values.extend_from_slice(&prefix);
        let config = SourceConfig {
            index_match_cache: true,
            ..SourceConfig::default()
        };
        let index = StaticIndex::new(&values, 0, &config).unwrap();
        let cost = |symbol| symbol as f32;
        let mut expected = Vec::new();
        index.find_matches(position, cost, |m| {
            expected.push((m.length, m.distance, m.symbol))
        });
        assert_eq!(expected.len(), 22);
        let mut batch = Batch::new(&index, 0, &config).unwrap();
        assert!(!batch.has_bounded_frontier(position));
        let mut actual = Vec::new();
        batch.find_matches(&index, position, cost, |m| {
            actual.push((m.length, m.distance, m.symbol))
        });
        assert_eq!(actual, expected);
        assert_eq!(batch.overflows, 1);
        assert!(batch.matches[0].overflowed);
        assert_eq!(batch.matches[0].size, 16);
        assert!(!batch.has_bounded_frontier(position));
        batch.find_matches(&index, values.len() - 1, cost, |_| {});
        assert!(batch.has_bounded_frontier(values.len() - 1));
    }
}
