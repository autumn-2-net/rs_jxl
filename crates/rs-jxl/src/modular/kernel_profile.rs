//! Opt-in sampled pixel-stage diagnostics. Compiled out of normal builds.
//! Times include clock overhead; never scale them into claimed wall times.

use std::time::{Duration, Instant};

fn selected(ordinal: u64, seed: u64) -> bool {
    let phase = (ordinal >> 7)
        .wrapping_add(seed)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        >> 57;
    ordinal & 127 == phase
}

pub(super) struct Profile {
    enabled: bool,
    site: &'static str,
    labels: [&'static str; 4],
    seed: u64,
    pixels: u64,
    samples: u64,
    clocks: Duration,
    times: [Duration; 4],
    calls: [u64; 4],
}

impl Profile {
    pub(super) fn new(site: &'static str, labels: [&'static str; 4], seed: u64) -> Self {
        Self {
            enabled: crate::profile::enabled(),
            site,
            labels,
            seed,
            pixels: 0,
            samples: 0,
            clocks: Duration::ZERO,
            times: [Duration::ZERO; 4],
            calls: [0; 4],
        }
    }

    pub(super) fn start(&mut self) -> Option<Instant> {
        if !self.enabled {
            return None;
        }
        let ordinal = self.pixels;
        self.pixels = self.pixels.wrapping_add(1);
        if !selected(ordinal, self.seed) {
            return None;
        }
        self.samples += 1;
        let empty = Instant::now();
        let now = Instant::now();
        self.clocks += now.duration_since(empty);
        Some(now)
    }

    pub(super) fn lap(&mut self, stage: usize, clock: &mut Option<Instant>) {
        if let Some(previous) = clock {
            let now = Instant::now();
            self.times[stage] += now.duration_since(*previous);
            self.calls[stage] += 1;
            *previous = now;
        }
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        if self.enabled && self.samples != 0 {
            crate::profile::note(
                "modular.kernel_sampled",
                format_args!(
                    "site={} pixels={} samples={} clock_ns={} labels={} calls={:?} ns={:?} (sampled, includes timing overhead; not wall totals)",
                    self.site,
                    self.pixels,
                    self.samples,
                    self.clocks.as_nanos(),
                    self.labels.join(","),
                    self.calls,
                    self.times.map(|time| time.as_nanos()),
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotating_samples_and_disabled_profile() {
        let mut phases = [false; 128];
        for block in 0..256 {
            let chosen: Vec<_> = (0..128)
                .filter(|phase| selected((block << 7) | phase, 17))
                .collect();
            assert_eq!(chosen.len(), 1);
            phases[chosen[0] as usize] = true;
        }
        assert!(phases.into_iter().filter(|v| *v).count() > 100);
        // Different channels/streams must not all time the first pixel.
        let first_samples = (0..128).filter(|seed| selected(0, *seed)).count();
        assert!((1..4).contains(&first_samples));
        let mut profile = Profile::new("test", ["a", "b", "c", "d"], 0);
        profile.enabled = false;
        for _ in 0..1024 {
            let mut clock = profile.start();
            assert!(clock.is_none());
            profile.lap(0, &mut clock);
        }
        assert_eq!(profile.pixels, 0);
        assert_eq!(profile.calls, [0; 4]);
        assert_eq!(profile.times, [Duration::ZERO; 4]);
    }
}
