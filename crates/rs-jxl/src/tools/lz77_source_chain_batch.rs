// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Hash-chain semantics derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Immutable source chains and bounded, parallel SymbolId frontiers. Unlike the
//! bucket index, this preserves the source's coarse-chain visit budget, including
//! collisions. DP remains global and consumes batches in increasing position order.

use super::distances::FastCodes;
use super::index::Dominance;
use super::matchers::{filled, hash, window};
use super::search::{match_length, physical};
use super::{MIN_MATCH, Match, SourceConfig, SourceIndexMode, SourceLz77Method, SourceMatchCost};
use crate::Result;

const NONE: u32 = u32::MAX;
const HASH_BITS: usize = 15;
const BATCH: usize = 16384;
const SPECULATIVE_MATCH: usize = 128;

#[derive(Clone, Copy)]
struct Layout {
    window: usize,
    fine_buckets: usize,
    bytes: usize,
}

impl Layout {
    fn new(size: usize, budget: usize) -> Option<Self> {
        if size > 1 << 24 {
            return None;
        }
        let window = window(size);
        // Three u32 arrays, temporary heads, bounded batch frontiers and the
        // largest possible i8 distance table. Refinement adds two u32 arrays.
        let coarse = size
            .checked_mul(12)?
            .checked_add(((1 << HASH_BITS) + window + 1) * 4)?
            .checked_add(size.min(BATCH) * size_of::<Dominance>())?
            .checked_add(super::WINDOW_SIZE + 1)?;
        if coarse > budget {
            return None;
        }
        let fine_buckets = window.max(1 << HASH_BITS);
        let refined = coarse
            .checked_add(size.checked_mul(8)?)?
            .checked_add(fine_buckets * 4)?;
        Some(if refined <= budget {
            Self {
                window,
                fine_buckets,
                bytes: refined,
            }
        } else {
            Self {
                window,
                fine_buckets: 0,
                bytes: coarse,
            }
        })
    }
}

pub(super) fn enabled(size: usize, seed_tokens: usize, config: &SourceConfig) -> bool {
    config.chain_batch
        && config.method == SourceLz77Method::Optc256
        && !config.bucket_route().enabled
        && config.index_mode == SourceIndexMode::Disabled
        && config.match_cost == SourceMatchCost::SymbolId
        && size >= 65536
        // A full batch would otherwise redo positions skipped inside long copies.
        && seed_tokens > size / 4
        && Layout::new(size, budget(config)).is_some()
}

fn budget(config: &SourceConfig) -> usize {
    config
        .chain_batch_bytes
        .min(config.parallelism.map_or(usize::MAX, |p| p.scratch_bytes))
}

struct Index<'a> {
    values: &'a [u32],
    previous: Vec<u32>,
    zero_previous: Vec<u32>,
    zeros: Vec<u32>,
    fine_previous: Vec<u32>,
    ordinal: Vec<u32>,
    codes: FastCodes,
    window: usize,
    maximum: usize,
    limit: usize,
}

impl<'a> Index<'a> {
    fn new(values: &'a [u32], width: usize, config: &SourceConfig, layout: Layout) -> Result<Self> {
        let _profile = crate::profile::scope("lz77.source.chain_index_build");
        let size = values.len();
        let mut zeros = filled(size, 0u32)?;
        let mut run = 0;
        for p in (0..size).rev() {
            run = if values[p] == 0 {
                (run + 1).min(layout.window as u32)
            } else {
                0
            };
            zeros[p] = run;
        }
        let mut previous = filled(size, NONE)?;
        let mut zero_previous = filled(size, NONE)?;
        let mut heads = filled(1 << HASH_BITS, NONE)?;
        let mut zero_heads = filled(layout.window + 1, NONE)?;
        let refined = layout.fine_buckets != 0;
        let mut fine_heads = filled(layout.fine_buckets, NONE)?;
        let mut fine_previous = filled(if refined { size } else { 0 }, NONE)?;
        let mut ordinal = filled(if refined { size } else { 0 }, 0u32)?;
        for p in 0..size {
            let key = hash(values, p) as usize;
            let coarse = key & ((1 << HASH_BITS) - 1);
            previous[p] = heads[coarse];
            heads[coarse] = p as u32;
            let count = zeros[p] as usize;
            zero_previous[p] = zero_heads[count];
            zero_heads[count] = p as u32;
            if refined {
                ordinal[p] = if previous[p] == NONE {
                    0
                } else {
                    ordinal[previous[p] as usize] + 1
                };
                let fine = key & (layout.fine_buckets - 1);
                fine_previous[p] = fine_heads[fine];
                fine_heads[fine] = p as u32;
            }
        }
        Ok(Self {
            values,
            previous,
            zero_previous,
            zeros,
            fine_previous,
            ordinal,
            codes: FastCodes::new(width)?,
            window: layout.window,
            maximum: config.max_match_length,
            limit: config
                .chain_limit
                .filter(|n| (1..=256).contains(n))
                .unwrap_or(256) as usize,
        })
    }

    fn find(&self, position: usize, visit: impl FnMut(Match)) {
        self.find_limited(position, usize::MAX, visit);
    }

    fn find_limited(&self, position: usize, maximum: usize, mut visit: impl FnMut(Match)) {
        let full_limit = self.maximum.min(self.values.len() - position);
        let limit = full_limit.min(maximum);
        if limit < MIN_MATCH {
            return;
        }
        let zeros = self.zeros[position] as usize;
        let refined = zeros < 3 && !self.fine_previous.is_empty();
        let mut previous = if refined {
            self.fine_previous[position]
        } else {
            self.previous[position]
        };
        let (mut visits, mut best) = (0, 0usize);
        while previous != NONE {
            let p = previous as usize;
            let distance = position - p;
            // The mutable ring aliases distance == window with the current slot.
            if distance >= self.window {
                break;
            }
            visits = if refined {
                (self.ordinal[position] - self.ordinal[p]) as usize
            } else {
                visits + 1
            };
            if visits > self.limit {
                break;
            }
            let required = MIN_MATCH.max(best.saturating_sub(2));
            let prefix = if zeros > 3 {
                (zeros - 1).min(self.zeros[p] as usize).min(limit - 1)
            } else {
                0
            };
            let reject = zeros < 3
                && (required > limit
                    || self.values[position + required - 1] != self.values[p + required - 1]
                    || self.values[position..position + MIN_MATCH]
                        != self.values[p..p + MIN_MATCH]);
            let length = if reject {
                0
            } else {
                let prefix = if zeros < 3 { MIN_MATCH } else { prefix };
                prefix
                    + match_length(
                        &self.values[position + prefix..position + limit],
                        &self.values[p + prefix..p + limit],
                    )
            };
            if length >= required {
                visit(Match {
                    length,
                    distance,
                    symbol: self.codes.symbol(distance),
                });
                if length == limit && limit < full_limit {
                    // The caller must requery this whole frontier. Further
                    // speculative candidates would all be discarded.
                    return;
                }
                best = best.max(length);
            }
            if visits == self.limit {
                break;
            }
            if refined {
                previous = self.fine_previous[p];
            } else if zeros >= 3 && length > zeros {
                previous = self.zero_previous[p];
                if previous != NONE && self.zeros[previous as usize] as usize != zeros {
                    break;
                }
            } else {
                previous = self.previous[p];
            }
        }
    }
}

pub(super) struct Batch<'a> {
    index: Index<'a>,
    frontiers: Vec<Dominance>,
    begin: usize,
    end: usize,
    workers: usize,
    width: usize,
    computed: usize,
    consumed: usize,
    overflows: usize,
    demand_only: bool,
}

impl<'a> Batch<'a> {
    pub(super) fn new(
        values: &'a [u32],
        width: usize,
        config: &SourceConfig,
        seed_tokens: usize,
    ) -> Result<Option<Self>> {
        if !enabled(values.len(), seed_tokens, config) {
            return Ok(None);
        }
        let layout = Layout::new(values.len(), budget(config)).expect("checked chain batch layout");
        let index = Index::new(values, width, config, layout)?;
        let workers = super::workers::index_workers(config, BATCH / 1024, 1024);
        crate::profile::note(
            "lz77.source.chain_batch",
            format_args!(
                "tokens={} workers={} refined={} scratch_bound={} batch={}",
                values.len(),
                workers,
                layout.fine_buckets != 0,
                layout.bytes,
                BATCH,
            ),
        );
        Ok(Some(Self {
            index,
            frontiers: filled(values.len().min(BATCH), Dominance::new())?,
            begin: 0,
            end: 0,
            workers,
            width,
            computed: 0,
            consumed: 0,
            overflows: 0,
            demand_only: false,
        }))
    }

    pub(super) fn find_matches(&mut self, position: usize, mut visit: impl FnMut(Match)) {
        debug_assert!(position >= self.begin);
        if position >= self.end && self.computed >= BATCH && self.consumed * 4 < self.computed {
            self.demand_only = true;
        }
        if self.demand_only {
            self.consumed += 1;
            self.index.find(position, visit);
            return;
        }
        if position >= self.end {
            let _profile = crate::profile::scope("lz77.source.chain_batch_fill");
            self.begin = position;
            self.end = self.index.values.len().min(position + BATCH);
            let count = self.end - self.begin;
            let tasks = self.workers.min(count.div_ceil(1024));
            let mut remaining = &mut self.frontiers[..count];
            let mut parts = Vec::with_capacity(tasks);
            for task in 0..tasks {
                let begin = count * task / tasks;
                let end = count * (task + 1) / tasks;
                let (frontiers, rest) = remaining.split_at_mut(end - begin);
                remaining = rest;
                parts.push((self.begin + begin, frontiers));
            }
            let index = &self.index;
            super::workers::each(&mut parts, |_, (begin, frontiers)| {
                for (offset, frontier) in frontiers.iter_mut().enumerate() {
                    *frontier = Dominance::new();
                    let position = *begin + offset;
                    let truncated =
                        index.maximum.min(index.values.len() - position) > SPECULATIVE_MATCH;
                    // Long-copy interiors are often skipped by the global DP.
                    // Bound speculative comparisons, not the consumed match set.
                    index.find_limited(position, SPECULATIVE_MATCH, |m| {
                        frontier.add(m.length, m.symbol, m.symbol as f32);
                        frontier.overflowed |= truncated && m.length == SPECULATIVE_MATCH;
                    });
                }
            });
            self.computed += count;
        }
        self.consumed += 1;
        let frontier = &self.frontiers[position - self.begin];
        if frontier.overflowed {
            self.overflows += 1;
            self.index.find(position, visit);
        } else {
            for entry in &frontier.entries[..frontier.size] {
                visit(Match {
                    length: entry.length,
                    symbol: entry.symbol,
                    distance: physical(entry.symbol, self.width),
                });
            }
        }
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        crate::profile::note(
            "lz77.source.chain_batch_work",
            format_args!(
                "tokens={} computed={} consumed={} overflow_requeries={} demand_only={}",
                self.index.values.len(),
                self.computed,
                self.consumed,
                self.overflows,
                self.demand_only,
            ),
        );
    }
}

#[cfg(test)]
#[path = "lz77_source_chain_batch_tests.rs"]
mod tests;
