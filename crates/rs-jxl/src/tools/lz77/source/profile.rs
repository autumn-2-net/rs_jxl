//! Sampled inner-loop diagnostics. No clocks or logging when profiling is off.

use std::time::{Duration, Instant};

pub(super) struct Profile {
    enabled: bool,
    tokens: usize,
    samples: usize,
    index: Duration,
    matching: Duration,
    dp: Duration,
}

impl Profile {
    pub(super) fn new(tokens: usize) -> Self {
        Self {
            enabled: crate::profile::enabled(),
            tokens,
            samples: 0,
            index: Duration::ZERO,
            matching: Duration::ZERO,
            dp: Duration::ZERO,
        }
    }

    pub(super) fn sample(&mut self, position: usize) -> Option<Instant> {
        // Mix positions instead of sampling a fixed phase of periodic runs.
        let mixed = (position as u64).wrapping_mul(0x9e3779b97f4a7c15);
        if self.enabled && mixed >> 57 == 0 {
            self.samples += 1;
            Some(Instant::now())
        } else {
            None
        }
    }

    pub(super) fn index(&mut self, start: &mut Option<Instant>) {
        Self::lap(&mut self.index, start);
    }

    pub(super) fn matching(&mut self, start: &mut Option<Instant>) {
        Self::lap(&mut self.matching, start);
    }

    pub(super) fn dp(&mut self, start: &mut Option<Instant>) {
        Self::lap(&mut self.dp, start);
    }

    fn lap(total: &mut Duration, start: &mut Option<Instant>) {
        if let Some(previous) = start {
            let now = Instant::now();
            *total += now.duration_since(*previous);
            *previous = now;
        }
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        if self.enabled {
            crate::profile::note(
                "lz77.source.sampled",
                format_args!(
                    "tokens={} samples={} index_s={:.6} matching_frontier_s={:.6} dp_s={:.6} (sampled only, not wall totals)",
                    self.tokens,
                    self.samples,
                    self.index.as_secs_f64(),
                    self.matching.as_secs_f64(),
                    self.dp.as_secs_f64(),
                ),
            );
        }
    }
}
