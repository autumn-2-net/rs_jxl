//! Sample-distribution proposals for opt-in static leaf offsets.
//! Scores exclude tree signaling and are not serialized coding costs.

use crate::entropy::HybridUintConfig;
use crate::{Error, Result};

use super::{Samples, pack_signed};

const DENSE_TOKENS: usize = 4096;
// A bounded, direct-mapped cache: collisions only repeat the original log2.
const ENTROPY_CACHE_ENTRIES: usize = 256;
// Baseline plus legacy centers, modes, quantiles and baseline neighbors.
const MAX_CANDIDATES: usize = 1 + (3 + 4 + 4 + 1) * 5;

fn reserve<T>(values: &mut Vec<T>, count: usize) -> Result<()> {
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("offset distribution allocation"))
}

fn neighborhood(candidates: &mut Vec<i32>, center: i32) {
    for delta in [0, -1, 1, -2, 2] {
        let offset = center.wrapping_add(delta);
        if !candidates.contains(&offset) {
            candidates.push(offset);
        }
    }
}

fn entropy_term(count: u64, n: f64) -> f64 {
    if count == 0 {
        0.0
    } else {
        let count = count as f64;
        count * (n / count).log2()
    }
}

struct EntropyCache {
    n: f64,
    entries: Vec<(u64, f64)>,
}

impl EntropyCache {
    fn new(n: usize) -> Result<Self> {
        // Small leaves need fewer slots; no entropy terms are precomputed.
        let slots = (n.min(ENTROPY_CACHE_ENTRIES - 1) + 1).next_power_of_two();
        let mut entries = Vec::new();
        reserve(&mut entries, slots)?;
        entries.resize(slots, (0, 0.0));
        Ok(Self {
            n: n as f64,
            entries,
        })
    }

    fn term(&mut self, count: u64) -> f64 {
        if count == 0 {
            return 0.0;
        }
        // Mix high bits too: repeated power-of-two counts must not all collide.
        let slot =
            (count.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 56) as usize & (self.entries.len() - 1);
        let entry = &mut self.entries[slot];
        if entry.0 != count {
            *entry = (count, entropy_term(count, self.n));
        }
        entry.1
    }
}

pub(super) fn scores(samples: &mut Samples, config: HybridUintConfig) -> Result<Vec<(i32, f64)>> {
    scores_impl::<true>(samples, config)
}

// The uncached specialization is used only by equivalence tests and the benchmark.
fn scores_impl<const CACHE_ENTROPY: bool>(
    samples: &mut Samples,
    config: HybridUintConfig,
) -> Result<Vec<(i32, f64)>> {
    if samples.values.is_empty() || samples.count == 0 {
        let mut result = Vec::new();
        reserve(&mut result, 1)?;
        result.push((samples.baseline, 0.0));
        return Ok(result);
    }
    // Preserve wide mean/median sources; wrapped order is only for new seeds.
    let mean = samples.mean();
    samples.values.sort_unstable();
    let median = samples.values[(samples.values.len() - 1) / 2] as i32;
    let mut wrapped = Vec::new();
    reserve(&mut wrapped, samples.values.len())?;
    wrapped.extend(samples.values.iter().map(|&value| value as i32));
    wrapped.sort_unstable();
    let mut groups: Vec<(i32, u64)> = Vec::new();
    reserve(&mut groups, wrapped.len())?;
    for value in wrapped {
        if let Some(last) = groups.last_mut().filter(|last| last.0 == value) {
            last.1 += 1;
        } else {
            groups.push((value, 1));
        }
    }

    let mut candidates = Vec::new();
    reserve(&mut candidates, MAX_CANDIDATES)?;
    candidates.push(samples.baseline);
    for center in [mean, median, 0] {
        neighborhood(&mut candidates, center);
    }
    if samples.baseline != 0 {
        neighborhood(&mut candidates, samples.baseline);
    }
    let mut modes = Vec::new();
    reserve(&mut modes, groups.len())?;
    modes.extend_from_slice(&groups);
    modes.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for &(value, _) in modes.iter().take(4) {
        neighborhood(&mut candidates, value);
    }
    for numerator in [1u128, 2, 6, 7] {
        let rank = (samples.values.len() - 1) as u128 * numerator / 8;
        let mut cumulative = 0u128;
        for &(value, count) in &groups {
            cumulative += u128::from(count);
            if cumulative > rank {
                neighborhood(&mut candidates, value);
                break;
            }
        }
    }

    // For validated configs, u32::MAX has the largest token. Wide alphabets
    // instead use scratch bounded by distinct samples, reused per candidate.
    let token_count = u64::from(config.encode(u32::MAX).token) + 1;
    let mut dense = Vec::new();
    let mut sparse: Vec<(u32, u64)> = Vec::new();
    if token_count <= DENSE_TOKENS as u64 {
        reserve(&mut dense, token_count as usize)?;
        dense.resize(token_count as usize, 0u64);
    } else {
        reserve(&mut sparse, groups.len())?;
    }
    let mut result = Vec::new();
    reserve(&mut result, candidates.len())?;
    let n = samples.values.len() as f64;
    let population_scale = samples.count as f64 / n;
    let mut entropy_cache = if CACHE_ENTROPY {
        Some(EntropyCache::new(samples.values.len())?)
    } else {
        None
    };
    let mut term = |count| match &mut entropy_cache {
        Some(cache) => cache.term(count),
        None => entropy_term(count, n),
    };
    for offset in candidates {
        dense.fill(0);
        sparse.clear();
        let mut extra = 0u128;
        for &(value, count) in &groups {
            // Equivalent modulo 2^32 subtraction, safe even for i64 extremes.
            let encoded = config.encode(pack_signed(value.wrapping_sub(offset)));
            extra += u128::from(encoded.nbits) * u128::from(count);
            if dense.is_empty() {
                sparse.push((encoded.token, count));
            } else {
                dense[encoded.token as usize] += count;
            }
        }
        let entropy = if dense.is_empty() {
            sparse.sort_unstable_by_key(|&(token, _)| token);
            let mut entropy = 0.0;
            let mut previous = None;
            let mut count = 0;
            for &(token, multiplicity) in &sparse {
                if previous != Some(token) {
                    entropy += term(count);
                    previous = Some(token);
                    count = 0;
                }
                count += multiplicity;
            }
            entropy + term(count)
        } else {
            dense.iter().map(|&count| term(count)).sum()
        };
        result.push((offset, (extra as f64 + entropy) * population_scale));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use super::*;

    fn scoring_cases() -> Vec<(&'static str, Samples, HybridUintConfig)> {
        let dense = HybridUintConfig::default();
        let sparse = HybridUintConfig::new(15, 15, 0).unwrap();
        let sparse_lsb = HybridUintConfig::new(15, 0, 15).unwrap();
        [
            ("low-cardinality", 1024, 0, dense),
            ("large-low-cardinality", 16384, 0, dense),
            ("high-cardinality", 1024, 1, dense),
            ("large-high-cardinality", 8192, 1, dense),
            ("sparse-literals", 1024, 1, sparse),
            ("sparse-lsb", 1024, 1, sparse_lsb),
            ("sparse-low-cardinality", 1024, 0, sparse),
            ("singleton", 1, 0, dense),
            ("small-dense", 7, 2, dense),
            ("small-sparse", 7, 2, sparse),
            ("small-high-cardinality", 31, 1, dense),
        ]
        .into_iter()
        .map(|(name, n, kind, config)| {
            let mut samples = Samples {
                baseline: -37,
                ..Samples::new(19)
            };
            let mut random = 0x1234_5678_9abc_def0u64;
            for i in 0..n {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                let value = match kind {
                    0 => [-1000, 0, 1000, 1000][i % 4],
                    1 => i64::from(random as i32),
                    _ => [i64::MIN, i64::MAX, -1, 0, 1, 4_294_967_296, 17][i % 7],
                };
                samples.add(value, n).unwrap();
            }
            // A larger population must not change the cache's reservoir-size n.
            samples.count *= 7;
            samples.sum *= 7;
            (name, samples, config)
        })
        .collect()
    }

    fn assert_identical_scores(actual: &[(i32, f64)], expected: &[(i32, f64)]) {
        assert_eq!(actual.len(), expected.len());
        for (&(offset, bits), &(old_offset, old_bits)) in actual.iter().zip(expected) {
            assert_eq!(offset, old_offset);
            assert_eq!(bits.to_bits(), old_bits.to_bits(), "offset={offset}");
        }
    }

    #[test]
    fn cached_scores_are_bit_identical_to_uncached_scores() {
        for (name, mut samples, config) in scoring_cases() {
            let old = scores_impl::<false>(&mut samples, config).unwrap();
            let new = scores(&mut samples, config).unwrap();
            assert_eq!(new[0].0, samples.baseline, "{name}");
            assert_identical_scores(&new, &old);
            assert_identical_scores(&scores(&mut samples, config).unwrap(), &old);
        }
    }

    #[test]
    fn entropy_cache_is_lazy_bounded_and_exact_on_collisions() {
        for (n, slots) in [
            (1, 2),
            (2, 4),
            (7, 8),
            (31, 32),
            (255, 256),
            (256, 256),
            (1024, 256),
            (16384, 256),
            (usize::MAX, 256),
        ] {
            let mut cache = EntropyCache::new(n).unwrap();
            assert!(cache.entries.len().is_power_of_two());
            assert!(cache.entries.len() <= ENTROPY_CACHE_ENTRIES);
            assert_eq!(cache.entries.len(), slots);
            assert!(cache.entries.iter().all(|&entry| entry == (0, 0.0)));
            for count in [0, 1, n as u64, 1, n as u64, 0] {
                assert_eq!(
                    cache.term(count).to_bits(),
                    entropy_term(count, n as f64).to_bits()
                );
                let entries = cache.entries.clone();
                assert_eq!(
                    cache.term(count).to_bits(),
                    entropy_term(count, n as f64).to_bits()
                );
                assert_eq!(cache.entries, entries);
            }
        }

        let n = 4096;
        let mut cache = EntropyCache::new(n).unwrap();
        // More distinct counts than slots guarantees eviction, not just hits.
        for count in (0..=n as u64).chain((0..=n as u64).rev()) {
            assert_eq!(
                cache.term(count).to_bits(),
                entropy_term(count, n as f64).to_bits()
            );
        }
        assert_eq!(cache.entries.len(), ENTROPY_CACHE_ENTRIES);
    }

    #[test]
    #[ignore = "opt-in scoring microbenchmark; run release with --ignored --nocapture"]
    fn cached_vs_uncached_distribution_scoring() {
        type Scorer = fn(&mut Samples, HybridUintConfig) -> Result<Vec<(i32, f64)>>;
        const REPEATS: usize = 5;

        println!(
            "Complete scores calls, sorted reservoirs, fresh per-call caches; median of {REPEATS}"
        );
        println!("fixture n candidates cache_bytes old_us new_us old/new");
        for (name, mut samples, config) in scoring_cases() {
            let old = scores_impl::<false>(&mut samples, config).unwrap();
            assert_identical_scores(&scores(&mut samples, config).unwrap(), &old);
            let n = samples.values.len();
            let iterations = (131_072 / n).clamp(16, 2048);
            let measure = |samples: &mut Samples, scorer: Scorer| {
                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(
                        black_box(scorer)(black_box(&mut *samples), black_box(config)).unwrap(),
                    );
                }
                start.elapsed()
            };
            for _ in 0..3 {
                black_box(scores_impl::<false>(&mut samples, config).unwrap());
                black_box(scores(&mut samples, config).unwrap());
            }
            let mut old_times = [Duration::ZERO; REPEATS];
            let mut new_times = [Duration::ZERO; REPEATS];
            for repeat in 0..REPEATS {
                if repeat % 2 == 0 {
                    old_times[repeat] = measure(&mut samples, scores_impl::<false>);
                    new_times[repeat] = measure(&mut samples, scores);
                } else {
                    new_times[repeat] = measure(&mut samples, scores);
                    old_times[repeat] = measure(&mut samples, scores_impl::<false>);
                }
            }
            old_times.sort_unstable();
            new_times.sort_unstable();
            let old_us = old_times[REPEATS / 2].as_secs_f64() * 1e6 / iterations as f64;
            let new_us = new_times[REPEATS / 2].as_secs_f64() * 1e6 / iterations as f64;
            let cache_bytes =
                EntropyCache::new(n).unwrap().entries.len() * std::mem::size_of::<(u64, f64)>();
            println!(
                "{name:<24} {n:>5} {:>3} {cache_bytes:>4} {old_us:>10.3} {new_us:>10.3} {:>7.3}x",
                old.len(),
                old_us / new_us,
            );
        }
    }

    fn reference(samples: &Samples, config: HybridUintConfig, offset: i32) -> f64 {
        let mut counts = BTreeMap::<u32, u64>::new();
        let mut extra = 0u128;
        for &value in &samples.values {
            let residual = (i128::from(value) - i128::from(offset)) as i32;
            let encoded = config.encode(pack_signed(residual));
            *counts.entry(encoded.token).or_default() += 1;
            extra += u128::from(encoded.nbits);
        }
        let n = samples.values.len() as f64;
        let entropy: f64 = counts
            .values()
            .map(|&count| {
                let count = count as f64;
                count * (n / count).log2()
            })
            .sum();
        (extra as f64 + entropy) * (samples.count as f64 / n)
    }

    #[test]
    fn grouped_scores_match_direct_wide_wrapping_reference() {
        let configs = [
            HybridUintConfig::default(),
            HybridUintConfig::new(4, 1, 2).unwrap(),
            HybridUintConfig::new(0, 0, 0).unwrap(),
            HybridUintConfig::new(8, 0, 0).unwrap(),
            HybridUintConfig::new(15, 0, 0).unwrap(),
            HybridUintConfig::new(15, 15, 0).unwrap(),
            HybridUintConfig::new(15, 0, 15).unwrap(),
        ];
        for capacity in [1, 7, 1024] {
            let mut samples = Samples::new(19);
            for value in [
                i64::MIN,
                i64::MAX,
                i64::from(i32::MIN),
                i64::from(i32::MAX),
                -4_294_967_297,
                4_294_967_296,
                0,
                0,
            ] {
                samples.add(value, capacity).unwrap();
            }
            let mut random = 0x1234_5678_9abc_def0u64;
            for i in 0..300 {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                let value = match i % 4 {
                    0 => (random % 17) as i64 - 8,
                    1 => (random as i32) as i64,
                    2 => random as i64,
                    _ => 4_294_967_296 + (random % 17) as i64 - 8,
                };
                samples.add(value, capacity).unwrap();
            }
            for config in configs {
                let result = scores(&mut samples, config).unwrap();
                let uncached = scores_impl::<false>(&mut samples, config).unwrap();
                assert_identical_scores(&result, &uncached);
                assert_eq!(result[0].0, 0);
                let unique: BTreeSet<_> = result.iter().map(|&(offset, _)| offset).collect();
                assert_eq!(unique.len(), result.len());
                for (offset, bits) in result {
                    let expected = reference(&samples, config, offset);
                    assert!(bits.is_finite() && bits >= 0.0);
                    assert!((bits - expected).abs() <= 1e-10 * expected.abs().max(1.0));
                }
            }
        }
    }

    #[test]
    fn modes_find_a_gain_outside_legacy_neighborhoods() {
        let mut samples = Samples::new(0);
        for (value, count) in [(-1000, 40), (0, 20), (1000, 40)] {
            for _ in 0..count {
                samples.add(value, 100).unwrap();
            }
        }
        assert_eq!(samples.mean(), 0);
        let config = HybridUintConfig::new(4, 1, 2).unwrap();
        let result = scores(&mut samples, config).unwrap();
        assert_eq!(samples.values[49], 0);
        let legacy_best = (-2..=2)
            .map(|offset| reference(&samples, config, offset))
            .fold(f64::INFINITY, f64::min);
        for mode in [-1000, 1000] {
            let bits = result
                .iter()
                .find(|&&(offset, _)| offset == mode)
                .unwrap()
                .1;
            assert!(bits < legacy_best - 90.0);
        }
    }

    #[test]
    fn small_reservoirs_are_deterministic_and_keep_wide_centers() {
        let config = HybridUintConfig::default();
        for capacity in [1, 2, 7] {
            let make = || {
                let mut samples = Samples::new(3);
                for i in 0..1000 {
                    samples.add(4_294_967_296 + i * 17, capacity).unwrap();
                }
                samples
            };
            let mut a = make();
            let mut b = make();
            let mean = a.mean();
            let mut wide = a.values.clone();
            wide.sort_unstable();
            let median = wide[(wide.len() - 1) / 2] as i32;
            let first = scores(&mut a, config).unwrap();
            assert_eq!(first, scores(&mut b, config).unwrap());
            assert_eq!(first, scores(&mut a, config).unwrap());
            for center in [0, mean, median] {
                for delta in [-2, -1, 0, 1, 2] {
                    assert!(
                        first
                            .iter()
                            .any(|&(offset, _)| offset == center.wrapping_add(delta))
                    );
                }
            }
            assert_eq!(a.values.len(), capacity);
            assert_eq!(a.count, 1000);
        }
    }

    #[test]
    fn boundary_neighbors_wrap_and_empty_samples_are_harmless() {
        let config = HybridUintConfig::default();
        assert_eq!(scores(&mut Samples::new(0), config).unwrap(), [(0, 0.0)]);
        let mut samples = Samples::new(0);
        samples.add(i64::from(i32::MIN), 1).unwrap();
        let result = scores(&mut samples, config).unwrap();
        assert!(
            result
                .iter()
                .any(|&(offset, bits)| offset == i32::MIN && bits == 0.0)
        );
        assert!(result.iter().any(|&(offset, _)| offset == i32::MAX));

        let mut samples = Samples::new(0);
        samples.add(2_147_483_648, 2).unwrap();
        samples.add(2_147_483_646, 2).unwrap();
        let result = scores(&mut samples, config).unwrap();
        assert!(result.iter().any(|&(offset, _)| offset == i32::MAX));
    }

    #[test]
    fn nonzero_baseline_is_first_and_scores_use_absolute_not_delta_offsets() {
        let mut samples = Samples {
            baseline: -37,
            ..Samples::new(0)
        };
        for value in [-1000, -37, 0, 18, 1000].repeat(7) {
            samples.add(value, 35).unwrap();
        }
        let config = HybridUintConfig::new(4, 1, 2).unwrap();
        let result = scores(&mut samples, config).unwrap();
        assert_eq!(result[0].0, -37);
        assert!(result.iter().any(|&(offset, _)| offset == 0));
        for (offset, bits) in result {
            assert!((bits - reference(&samples, config, offset)).abs() < 1e-8);
        }
    }

    #[test]
    fn quantiles_and_tied_modes_have_deterministic_neighborhoods() {
        let mut samples = Samples::new(0);
        for i in (0..17).rev() {
            samples.add(i * 1000, 17).unwrap();
        }
        let result = scores(&mut samples, HybridUintConfig::default()).unwrap();
        // Equal-count modes choose the four smallest signed wrapped values.
        // Quantiles use floor((n - 1) * numerator / 8).
        for center in [0i32, 1000, 2000, 3000, 4000, 12000, 14000] {
            for delta in [-2, -1, 0, 1, 2] {
                assert!(
                    result
                        .iter()
                        .any(|&(offset, _)| offset == center.wrapping_add(delta))
                );
            }
        }
    }
}
