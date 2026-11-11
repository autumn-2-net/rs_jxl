//! Deterministic feedback for work ahead of the global parser. A wrong forecast
//! costs time only: every requested position still gets its complete frontier.

use super::BATCH;

const OBSERVATIONS: usize = 128;
const MAX_OBSERVATIONS: usize = 4096;

pub(super) struct Speculation {
    workers: usize,
    window: usize,
    first: usize,
    queries: usize,
    long: usize,
    observations: usize,
}

impl Speculation {
    pub(super) fn new(workers: usize) -> Self {
        Self {
            workers,
            window: 0,
            first: 0,
            queries: 0,
            long: 0,
            observations: OBSERVATIONS,
        }
    }

    pub(super) fn window(&self) -> usize {
        self.window
    }

    pub(super) fn demand(&mut self, position: usize, long: bool) {
        if self.workers <= 1 {
            return;
        }
        if self.queries == 0 {
            self.first = position;
        }
        self.queries += 1;
        self.long += usize::from(long);
        if self.queries < self.observations {
            return;
        }
        let span = position - self.first + 1;
        // Long frontiers need full requeries; sparse origins waste lookahead.
        if self.queries * 4 >= span.saturating_mul(3) && self.long * 16 <= self.queries {
            self.window = (self.workers * 512).clamp(2048, BATCH);
        }
        self.queries = 0;
        self.long = 0;
    }

    pub(super) fn finish(&mut self, computed: usize, consumed: usize, overflows: usize) {
        debug_assert!(overflows <= consumed && consumed <= computed);
        if consumed * 4 >= computed * 3 && overflows * 16 <= consumed {
            self.window = (self.window * 2).min(BATCH);
            self.observations = OBSERVATIONS;
        } else {
            self.window = 0;
            // Avoid repeatedly paying a small fill for the same sparse region.
            // This delays speculative retries, never a demanded full query.
            self.observations = (self.observations * 4).min(MAX_OBSERVATIONS);
        }
        self.queries = 0;
        self.long = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_queries_ramp_up_but_sparse_and_overflow_windows_return_to_demand() {
        let mut p = Speculation::new(8);
        for position in 0..OBSERVATIONS {
            p.demand(position, false);
        }
        assert_eq!(p.window(), 4096);
        p.finish(4096, 4096, 0);
        assert_eq!(p.window(), 8192);
        p.finish(8192, 8192, 0);
        assert_eq!(p.window(), BATCH);
        p.finish(BATCH, BATCH / 3, 0);
        assert_eq!(p.window(), 0);
        assert_eq!(p.observations, 512);
        for position in 0..512 {
            p.demand(position * 4, false);
        }
        assert_eq!(p.window(), 0);
        for position in 0..512 {
            p.demand(position + BATCH, false);
        }
        assert_eq!(p.window(), 4096);
        p.finish(4096, 4096, 4096 / 8);
        assert_eq!(p.window(), 0);
        assert_eq!(p.observations, 2048);
        p.finish(4096, 1, 0);
        assert_eq!(p.observations, MAX_OBSERVATIONS);
    }

    #[test]
    fn long_matches_and_one_worker_never_speculate() {
        for workers in [1, 8] {
            let mut p = Speculation::new(workers);
            for position in 0..OBSERVATIONS * 3 {
                p.demand(position, workers > 1);
            }
            assert_eq!(p.window(), 0);
        }
    }
}
