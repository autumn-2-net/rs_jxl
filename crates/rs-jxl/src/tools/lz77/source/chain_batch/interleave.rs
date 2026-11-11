// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Hash-chain semantics derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Interleave independent immutable queries, not dependent links in one query.
//! Each lane preserves Index::find_limited's visit order and budget. Loading
//! all lanes' next links before extending matches exposes independent memory
//! requests; finished lanes refill from the remaining contiguous positions.
//! This is an independent Rust experiment, not a port of FXZ's radix format.

use super::{Index, MIN_MATCH, Match, NONE, match_length};

#[derive(Clone, Copy)]
struct Query {
    position: usize,
    limit: usize,
    zeros: usize,
    best: usize,
    visits: usize,
    previous: u32,
    refined: bool,
    truncated: bool,
}

impl Query {
    const EMPTY: Self = Self {
        position: 0,
        limit: 0,
        zeros: 0,
        best: 0,
        visits: 0,
        previous: NONE,
        refined: false,
        truncated: false,
    };
}

#[derive(Clone, Copy)]
struct Candidate {
    position: usize,
    next: u32,
    zero_next: u32,
    active: bool,
}

impl Candidate {
    const EMPTY: Self = Self {
        position: 0,
        next: NONE,
        zero_next: NONE,
        active: false,
    };
}

impl Index<'_> {
    fn query(&self, position: usize, maximum: usize) -> Query {
        let full_limit = self.maximum.min(self.values.len() - position);
        let limit = full_limit.min(maximum);
        let zeros = self.zeros[position] as usize;
        let refined = zeros < 3 && !self.fine_previous.is_empty();
        Query {
            position,
            limit,
            zeros,
            refined,
            truncated: limit < full_limit,
            previous: if limit < MIN_MATCH {
                NONE
            } else if refined {
                self.fine_previous[position]
            } else {
                self.previous[position]
            },
            ..Query::EMPTY
        }
    }

    #[inline]
    fn prepare(&self, q: &mut Query) -> Candidate {
        if q.previous == NONE {
            return Candidate::EMPTY;
        }
        let p = q.previous as usize;
        if q.position - p >= self.window {
            q.previous = NONE;
            return Candidate::EMPTY;
        }
        q.visits = if q.refined {
            (self.ordinal[q.position] - self.ordinal[p]) as usize
        } else {
            q.visits + 1
        };
        if q.visits > self.limit {
            q.previous = NONE;
            return Candidate::EMPTY;
        }
        Candidate {
            position: p,
            next: if q.refined {
                self.fine_previous[p]
            } else {
                self.previous[p]
            },
            zero_next: if q.zeros >= 3 {
                self.zero_previous[p]
            } else {
                NONE
            },
            active: true,
        }
    }

    #[inline]
    fn extend(&self, q: &mut Query, c: Candidate, visit: &mut impl FnMut(usize, Match)) {
        let required = MIN_MATCH.max(q.best.saturating_sub(2));
        let reject = q.zeros < 3
            && (required > q.limit
                || self.values[q.position + required - 1]
                    != self.values[c.position + required - 1]
                || self.values[q.position..q.position + MIN_MATCH]
                    != self.values[c.position..c.position + MIN_MATCH]);
        let length = if reject {
            0
        } else if q.zeros >= 3 {
            self.zero_length(q.position, c.position, q.limit, q.zeros)
        } else {
            MIN_MATCH
                + match_length(
                    &self.values[q.position + MIN_MATCH..q.position + q.limit],
                    &self.values[c.position + MIN_MATCH..c.position + q.limit],
                )
        };
        if length >= required {
            let distance = q.position - c.position;
            visit(
                q.position,
                Match {
                    length,
                    distance,
                    symbol: self.codes.symbol(distance),
                },
            );
            if length == q.limit && q.truncated {
                q.previous = NONE;
                return;
            }
            q.best = q.best.max(length);
        }
        q.previous = if q.visits == self.limit {
            NONE
        } else if !q.refined && q.zeros >= 3 && length > q.zeros {
            // This choice depends on the match just computed. Preloading the
            // link is safe; choosing the zero chain before this point is not.
            let next = c.zero_next;
            if next != NONE && self.zeros[next as usize] as usize != q.zeros {
                NONE
            } else {
                next
            }
        } else {
            c.next
        };
    }

    pub(super) fn find_interleaved<const LANES: usize>(
        &self,
        begin: usize,
        count: usize,
        maximum: usize,
        mut visit: impl FnMut(usize, Match),
    ) {
        debug_assert!(matches!(LANES, 2 | 4 | 8));
        debug_assert!(begin <= self.values.len() && count <= self.values.len() - begin);
        let end = begin + count;
        let mut next = begin;
        let mut queries = [Query::EMPTY; LANES];
        loop {
            let mut pending = false;
            let mut candidates = [Candidate::EMPTY; LANES];
            for lane in 0..LANES {
                if queries[lane].previous == NONE && next < end {
                    queries[lane] = self.query(next, maximum);
                    next += 1;
                }
                candidates[lane] = self.prepare(&mut queries[lane]);
                pending |= candidates[lane].active;
            }
            // Complete each candidate only after issuing the independent loads.
            for lane in 0..LANES {
                if candidates[lane].active {
                    self.extend(&mut queries[lane], candidates[lane], &mut visit);
                }
            }
            if !pending && next == end {
                break;
            }
        }
    }
}
