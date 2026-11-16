//! Optional coordinate refinement of exact EMA labels. Only histogram/table
//! Fast ANS costs are optimized here, not context-map or full-file bytes.

use super::Histogram;
use crate::{Error, Result};

fn cost(histogram: &Histogram) -> Result<f64> {
    // Empty centers are removed before serialization.
    if histogram.total == 0 {
        Ok(0.0)
    } else {
        histogram.population_cost().map(f64::from)
    }
}

fn prices(histogram: &Histogram) -> Vec<f32> {
    if histogram.total == 0 {
        return Vec::new();
    }
    let denominator = histogram.total as f64 + 256e-12;
    (0..256)
        .map(|s| {
            -((histogram.counts.get(s).copied().unwrap_or(0) as f64 + 1e-12) / denominator).log2()
                as f32
        })
        .collect()
}

fn without(center: &Histogram, part: &Histogram) -> Result<Histogram> {
    let mut result = center.clone();
    result.total = result
        .total
        .checked_sub(part.total)
        .ok_or_else(|| Error::InvalidInput("EMA relocation population mismatch".into()))?;
    for (i, &count) in part.counts.iter().enumerate() {
        let value = result
            .counts
            .get_mut(i)
            .ok_or_else(|| Error::InvalidInput("EMA relocation alphabet mismatch".into()))?;
        *value = value
            .checked_sub(count)
            .ok_or_else(|| Error::InvalidInput("EMA relocation bin mismatch".into()))?;
    }
    Ok(result)
}

pub(super) fn relocate(
    input: &[Histogram],
    centers: &mut [Histogram],
    labels: &mut [usize],
    passes: u8,
) -> Result<()> {
    let _profile = crate::profile::scope("entropy.cluster.ema.polish");
    let mut costs = centers.iter().map(cost).collect::<Result<Vec<_>>>()?;
    let start = costs.iter().sum::<f64>();
    let mut tables: Vec<_> = centers.iter().map(prices).collect();
    let mut moves = 0;
    let mut completed = 0;
    for _ in 0..passes {
        completed += 1;
        let previous_moves = moves;
        for (histogram, label) in input.iter().zip(labels.iter_mut()) {
            if histogram.total == 0 {
                continue;
            }
            let source = *label;
            let mut candidates = [(f32::INFINITY, usize::MAX); 4];
            let mut empty = None;
            for (id, table) in tables.iter().enumerate() {
                if id == source {
                    continue;
                }
                if centers[id].total == 0 {
                    empty.get_or_insert(id);
                    continue;
                }
                let score = histogram
                    .counts
                    .iter()
                    .zip(table)
                    .map(|(&n, &p)| n as f32 * p)
                    .sum::<f32>();
                if let Some(pos) = candidates.iter().position(|&(c, _)| score < c) {
                    for j in (pos + 1..candidates.len()).rev() {
                        candidates[j] = candidates[j - 1];
                    }
                    candidates[pos] = (score, id);
                }
            }
            let remainder = without(&centers[source], histogram)?;
            let remaining_cost = cost(&remainder)?;
            let mut best = None;
            let mut best_delta = 0.0;
            for target in candidates
                .iter()
                .filter_map(|&(_, id)| (id != usize::MAX).then_some(id))
                .chain(empty)
            {
                let mut combined = centers[target].clone();
                combined.add(histogram)?;
                let combined_cost = cost(&combined)?;
                let delta = (remaining_cost + combined_cost) - (costs[source] + costs[target]);
                if delta < best_delta {
                    best_delta = delta;
                    best = Some((target, combined, combined_cost));
                }
            }
            if let Some((target, combined, combined_cost)) = best {
                centers[source] = remainder;
                centers[target] = combined;
                costs[source] = remaining_cost;
                costs[target] = combined_cost;
                tables[source] = prices(&centers[source]);
                tables[target] = prices(&centers[target]);
                *label = target;
                moves += 1;
            }
        }
        if moves == previous_moves {
            break;
        }
    }
    for center in centers {
        // Relocation can remove the highest occupied symbol. Restore the
        // cluster API's trimmed-alphabet invariant as well as Shannon cost.
        *center = Histogram::new(&center.counts)?;
    }
    crate::profile::note(
        "entropy.cluster.ema.polish",
        format_args!(
            "passes={completed} moves={moves} fast_ans_before={start:.3} fast_ans_after={:.3}",
            costs.iter().sum::<f64>(),
        ),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relocation_preserves_counts_and_reduces_native_population_cost() {
        let input = [vec![100, 0], vec![0, 100], vec![95, 5], vec![0, 0]]
            .iter()
            .map(|h| Histogram::new(h).unwrap())
            .collect::<Vec<_>>();
        let mut centers = [vec![100, 100], vec![95, 5], vec![0, 0]]
            .iter()
            .map(|h| Histogram::new(h).unwrap())
            .collect::<Vec<_>>();
        let mut labels = vec![0, 0, 1, 0];
        let before = centers.iter().map(|c| cost(c).unwrap()).sum::<f64>();
        relocate(&input, &mut centers, &mut labels, 3).unwrap();
        let after = centers.iter().map(|c| cost(c).unwrap()).sum::<f64>();
        assert!(after < before);
        let mut rebuilt = vec![vec![0; 2]; centers.len()];
        for (h, &id) in input.iter().zip(&labels) {
            for (n, &v) in rebuilt[id].iter_mut().zip(&h.counts) {
                *n += v;
            }
        }
        for (h, expected) in centers.iter().zip(rebuilt) {
            assert_eq!(h.counts[0], expected[0]);
            assert_eq!(h.counts.get(1).copied().unwrap_or(0), expected[1]);
        }
        let labels_before = labels.clone();
        relocate(&input, &mut centers, &mut labels, 0).unwrap();
        assert_eq!(labels, labels_before);
    }

    #[test]
    fn relocation_trims_a_surviving_sources_highest_symbol() {
        let input = [vec![0, 100], vec![100, 0], vec![0, 100]]
            .iter()
            .map(|h| Histogram::new(h).unwrap())
            .collect::<Vec<_>>();
        let mut centers = [vec![100, 100], vec![0, 100]]
            .iter()
            .map(|h| Histogram::new(h).unwrap())
            .collect::<Vec<_>>();
        let mut labels = vec![0, 0, 1];
        relocate(&input, &mut centers, &mut labels, 1).unwrap();
        assert_eq!(centers[0].counts, vec![100]);
        assert_eq!(centers[0].total, 100);
        assert_eq!(centers[1].counts, vec![0, 200]);
        assert_eq!(labels, vec![1, 0, 1]);
    }
}
