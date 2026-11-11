//! Sample-distribution proposals for opt-in static leaf offsets.
//! Scores exclude tree signaling and are not serialized coding costs.

use crate::entropy::HybridUintConfig;
use crate::{Error, Result};

use super::{Samples, pack_signed};

const DENSE_TOKENS: usize = 4096;
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

pub(super) fn scores(samples: &mut Samples, config: HybridUintConfig) -> Result<Vec<(i32, f64)>> {
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
                    entropy += entropy_term(count, n);
                    previous = Some(token);
                    count = 0;
                }
                count += multiplicity;
            }
            entropy + entropy_term(count, n)
        } else {
            dense.iter().map(|&count| entropy_term(count, n)).sum()
        };
        result.push((offset, (extra as f64 + entropy) * population_scale));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

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
