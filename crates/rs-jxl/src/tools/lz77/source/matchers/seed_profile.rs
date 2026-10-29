//! Sampled Fast diagnostics; no global locks or per-query logging.

use std::time::{Duration, Instant};

use super::{Fast, MIN_MATCH, Match, WINDOW_SIZE};

fn sampled_query(ordinal: u64) -> bool {
    // One sample per 128 calls, with a different phase in each block.
    let phase = (ordinal >> 7).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 57;
    ordinal & 127 == phase
}

#[derive(Default)]
pub(super) struct Sample {
    ring_entries: usize,
    ring_window: usize,
    ring_prefix: usize,
    spatial_entries: usize,
    spatial_survivors: usize,
    grouped: usize,
    active_prefix: usize,
    length_bin: usize,
}

impl Sample {
    pub(super) fn capture<const N: usize>(
        matcher: &Fast<'_, N>,
        key: usize,
        position: usize,
        best: Match,
        spatial_survivors: usize,
    ) -> Self {
        let ring = &matcher.buckets[key];
        let limit = matcher.maximum.min(matcher.values.len() - position);
        let mut sample = Self {
            ring_entries: ring.size as usize,
            spatial_survivors,
            active_prefix: usize::from(matcher.prefix.cached_bytes() != 0),
            length_bin: match best.length {
                0..=2 => 0,
                3 => 1,
                4..=7 => 2,
                8..=31 => 3,
                32..=127 => 4,
                _ => 5,
            },
            ..Self::default()
        };
        for &previous in &ring.positions[..ring.size as usize] {
            let previous = previous as usize;
            if position - previous <= WINDOW_SIZE {
                sample.ring_window += 1;
                if limit >= MIN_MATCH
                    && matcher.values[position..position + MIN_MATCH]
                        == matcher.values[previous..previous + MIN_MATCH]
                {
                    sample.ring_prefix += 1;
                }
            }
        }
        if ring.size as usize == N {
            let safe = position - ring.positions[ring.next as usize] as usize;
            // Count source entries, including narrow-width aliases. The SIMD
            // grouped path canonicalizes those aliases before filtering.
            sample.spatial_entries = matcher
                .spatial
                .iter()
                .filter(|&&(distance, _)| distance > safe && distance <= position)
                .count();
            sample.grouped = usize::from(
                matcher.ranges
                    && matcher.prefix_simd
                    && sample.active_prefix != 0
                    && matcher.spatial.last().is_some_and(|&(d, _)| d > safe),
            );
        }
        sample
    }
}

pub(super) struct Profile {
    enabled: bool,
    tokens: usize,
    width: usize,
    capacity: usize,
    queries: u64,
    samples: usize,
    prepare: Duration,
    spatial_filter: Duration,
    spatial_matches: Duration,
    ring_matches: Duration,
    totals: Sample,
    lengths: [usize; 6],
}

impl Profile {
    pub(super) fn new(tokens: usize, width: usize, capacity: usize) -> Self {
        Self {
            enabled: crate::profile::enabled(),
            tokens,
            width,
            capacity,
            queries: 0,
            samples: 0,
            prepare: Duration::ZERO,
            spatial_filter: Duration::ZERO,
            spatial_matches: Duration::ZERO,
            ring_matches: Duration::ZERO,
            totals: Sample::default(),
            lengths: [0; 6],
        }
    }

    #[inline]
    pub(super) fn sample_next(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        let ordinal = self.queries;
        self.queries = self.queries.wrapping_add(1);
        sampled_query(ordinal)
    }

    pub(super) fn prepare(&mut self, clock: &mut Option<Instant>) {
        Self::lap(&mut self.prepare, clock);
    }

    pub(super) fn spatial_filter(&mut self, clock: &mut Option<Instant>) {
        Self::lap(&mut self.spatial_filter, clock);
    }

    pub(super) fn spatial_matches(&mut self, clock: &mut Option<Instant>) {
        Self::lap(&mut self.spatial_matches, clock);
    }

    pub(super) fn ring_matches(&mut self, clock: &mut Option<Instant>) {
        Self::lap(&mut self.ring_matches, clock);
    }

    fn lap(total: &mut Duration, clock: &mut Option<Instant>) {
        if let Some(previous) = clock {
            let now = Instant::now();
            *total += now.duration_since(*previous);
            *previous = now;
        }
    }

    pub(super) fn record(&mut self, sample: Sample) {
        self.samples += 1;
        self.totals.ring_entries += sample.ring_entries;
        self.totals.ring_window += sample.ring_window;
        self.totals.ring_prefix += sample.ring_prefix;
        self.totals.spatial_entries += sample.spatial_entries;
        self.totals.spatial_survivors += sample.spatial_survivors;
        self.totals.grouped += sample.grouped;
        self.totals.active_prefix += sample.active_prefix;
        self.lengths[sample.length_bin] += 1;
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        if self.enabled {
            crate::profile::note(
                "lz77.source.fast_sampled",
                format_args!(
                    "tokens={} width={} ring={} queries={} samples={} prepare_s={:.6} spatial_filter_s={:.6} spatial_match_s={:.6} ring_match_s={:.6} ring_entries={} ring_in_window={} ring_prefix_equal={} spatial_eligible_entries={} spatial_survivors={} grouped_queries={} active_prefix_queries={} best_lengths_0_3_4to7_8to31_32to127_128plus={:?} (one sample per 128 queries; sampled times are not wall totals)",
                    self.tokens,
                    self.width,
                    self.capacity,
                    self.queries,
                    self.samples,
                    self.prepare.as_secs_f64(),
                    self.spatial_filter.as_secs_f64(),
                    self.spatial_matches.as_secs_f64(),
                    self.ring_matches.as_secs_f64(),
                    self.totals.ring_entries,
                    self.totals.ring_window,
                    self.totals.ring_prefix,
                    self.totals.spatial_entries,
                    self.totals.spatial_survivors,
                    self.totals.grouped,
                    self.totals.active_prefix,
                    self.lengths,
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::lz77::{SourceConfig, SourceFastPrefix, SourceFastRanges};

    #[test]
    fn sampling_has_one_rotating_phase_per_128_calls() {
        let mut phases = [false; 128];
        for block in (0..256).chain([u64::MAX >> 7]) {
            let selected: Vec<_> = (0..128)
                .filter(|&phase| sampled_query((block << 7) | phase))
                .collect();
            assert_eq!(selected.len(), 1);
            phases[selected[0] as usize] = true;
        }
        assert!(phases.into_iter().filter(|&seen| seen).count() > 100);
        let mut profile = Profile::new(0, 0, 7);
        profile.enabled = false;
        for _ in 0..1024 {
            assert!(!profile.sample_next());
        }
        assert_eq!(profile.queries, 0);
        assert_eq!(profile.samples, 0);
    }

    #[test]
    fn sampled_queries_leave_matches_buckets_and_feedback_unchanged() {
        let mut values: Vec<_> = (0..193u32).map(|i| (i % 31) * 65537).collect();
        values.extend_from_within(7..104);
        values.extend([0; 97]);
        values.extend([u32::MAX, 0]);
        for fast_prefix in [
            SourceFastPrefix::Disabled,
            SourceFastPrefix::Scalar,
            SourceFastPrefix::Simd,
        ] {
            for width in [0, 1, 37] {
                let config = SourceConfig {
                    fast_prefix: Some(fast_prefix),
                    fast_ranges: SourceFastRanges::Both,
                    ..SourceConfig::default()
                };
                let mut plain = Fast::<7>::new(&values, width, &config).unwrap();
                let mut sampled = Fast::<7>::new(&values, width, &config).unwrap();
                plain.diagnostics.enabled = false;
                sampled.diagnostics.enabled = false;
                for matcher in [&mut plain, &mut sampled] {
                    matcher
                        .prefix
                        .record_candidate_work(matcher.prefix.activation_threshold());
                }
                for position in 0..values.len() {
                    assert_eq!(
                        plain.find_impl::<false>(position),
                        sampled.find_impl::<true>(position)
                    );
                    assert_eq!(plain.prefix.activation(), sampled.prefix.activation());
                    assert_eq!(
                        plain.prefix.candidate_work(),
                        sampled.prefix.candidate_work()
                    );
                }
                for (a, b) in plain.buckets.iter().zip(&sampled.buckets) {
                    assert_eq!(a.positions, b.positions);
                    assert_eq!((a.next, a.size), (b.next, b.size));
                }
                assert_eq!(plain.diagnostics.samples, 0);
                assert_eq!(sampled.diagnostics.samples, values.len());
                assert_eq!(
                    sampled.diagnostics.lengths.iter().sum::<usize>(),
                    values.len()
                );
                let totals = &sampled.diagnostics.totals;
                assert!(totals.ring_prefix <= totals.ring_window);
                assert!(totals.ring_window <= totals.ring_entries);
                assert!(totals.spatial_survivors <= totals.spatial_entries);
            }
        }
    }
}
