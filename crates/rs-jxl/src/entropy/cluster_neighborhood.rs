//! Optional local search after ordinary clustering. Population costs and the
//! coarse entropy bound are proposal heuristics, not final encoded-size proofs:
//! rebuilding adaptive HybridUint configs can change the symbol distributions.
//! The caller must compare complete serialized models against the baseline.

use super::{Histogram, entropy_sketch, merge_lower_bound, pair_population_cost};
use crate::bit_writer::BitWriter;
use crate::entropy::context_map::write_context_map;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub struct ContextMapOptions {
    /// Number of shortlisted pairs whose context maps are actually serialized.
    pub shortlist: usize,
    /// Maximum number of consecutive extra merges in one proposal chain.
    pub max_merges: usize,
}

impl Default for ContextMapOptions {
    fn default() -> Self {
        Self {
            shortlist: 8,
            max_merges: 4,
        }
    }
}

impl ContextMapOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.shortlist == 0 || self.max_merges == 0 {
            return Err(Error::InvalidInput(
                "context-map shortlist and merge budget must be positive".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) struct Plan {
    pub counts: Vec<Vec<u64>>,
    pub map: Vec<u8>,
    pub merges: usize,
}

#[derive(Default)]
pub(crate) struct Work {
    pub pairs: usize,
    pub map_trials: usize,
    pub merges: usize,
}

fn map_bits(map: &[u8], histograms: usize) -> Result<usize> {
    // A single raw context has no map syntax, unlike several contexts mapped
    // to one histogram (three bits).
    if map.len() == 1 && histograms == 1 {
        return Ok(0);
    }
    let capacity = map
        .len()
        .checked_mul(8)
        .and_then(|n| n.checked_add(65536))
        .ok_or(Error::Limit("context-map scratch size"))?;
    let mut writer = BitWriter::new(capacity);
    write_context_map(map, histograms, &mut writer)?;
    Ok(writer.bits_written())
}

// Keep raw-context positions (including the LZ distance context) unchanged.
// Only histogram IDs change; reorder the corresponding tables in the same way.
fn merged_map(map: &[u8], first: usize, second: usize) -> (Vec<u8>, Vec<usize>) {
    let mut ids = [usize::MAX; 256];
    let mut order = Vec::new();
    let map = map
        .iter()
        .map(|&id| {
            let id = if usize::from(id) == second {
                first
            } else {
                usize::from(id)
            };
            if ids[id] == usize::MAX {
                ids[id] = order.len();
                order.push(id);
            }
            ids[id] as u8
        })
        .collect();
    (map, order)
}

fn frequency_term(n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        n as f64 * (n as f64).log2()
    }
}

struct Pair {
    first: usize,
    second: usize,
    population_delta: f64,
    rank: f64,
}

/// At most two complete-model proposals: the first merge and the chain's end.
/// Existing greedy merges are not undone, and no new symbol semantics are used.
pub(crate) fn propose(
    raw: &[Vec<u64>],
    baseline_map: &[u8],
    histogram_count: usize,
    options: &ContextMapOptions,
) -> Result<(Vec<Plan>, Work)> {
    options.validate()?;
    if raw.len() != baseline_map.len() || histogram_count == 0 || histogram_count > 256 {
        return Err(Error::InvalidInput(
            "invalid context-map proposal shape".into(),
        ));
    }
    // Also validates dense IDs and references to every histogram.
    let mut current_map_bits = map_bits(baseline_map, histogram_count)?;
    let mut histograms = (0..histogram_count)
        .map(|_| Histogram::new(&[]))
        .collect::<Result<Vec<_>>>()?;
    for (counts, &id) in raw.iter().zip(baseline_map) {
        histograms
            .get_mut(usize::from(id))
            .ok_or_else(|| Error::InvalidInput("invalid context-map histogram ID".into()))?
            .add(&Histogram::new(counts)?)?;
    }
    // Wide prefix alphabets have no ANS population estimate in this heuristic.
    if histograms.iter().any(|h| h.counts.len() > 256) {
        return Ok((Vec::new(), Work::default()));
    }
    let mut map = baseline_map.to_vec();
    let mut plans = Vec::new();
    let mut work = Work::default();
    for _ in 0..options.max_merges.min(histogram_count - 1) {
        let costs = histograms
            .iter()
            .map(Histogram::population_cost)
            .collect::<Result<Vec<_>>>()?;
        let sketches: Vec<_> = histograms
            .iter()
            .map(|h| entropy_sketch(&h.counts))
            .collect();
        let mut frequencies = vec![0usize; histograms.len()];
        for &id in &map {
            frequencies[usize::from(id)] += 1;
        }
        let mut shortlist: Vec<Pair> = Vec::new();
        for first in 0..histograms.len() {
            for second in first + 1..histograms.len() {
                work.pairs += 1;
                let old = f64::from(costs[first]) + f64::from(costs[second]);
                // Even eliminating the entire map cannot cover this estimated
                // population loss. This only prunes proposals, not codec data.
                if merge_lower_bound(&sketches[first], &sketches[second]) - old
                    >= current_map_bits as f64
                {
                    continue;
                }
                let population_delta = f64::from(pair_population_cost(
                    &histograms[first],
                    &histograms[second],
                )?) - old;
                if population_delta >= current_map_bits as f64 {
                    continue;
                }
                let a = frequencies[first];
                let b = frequencies[second];
                let rank = population_delta + frequency_term(a) + frequency_term(b)
                    - frequency_term(a + b);
                let position = shortlist.partition_point(|p| p.rank <= rank);
                if position < options.shortlist {
                    shortlist.insert(
                        position,
                        Pair {
                            first,
                            second,
                            population_delta,
                            rank,
                        },
                    );
                    if shortlist.len() > options.shortlist {
                        shortlist.pop();
                    }
                }
            }
        }
        let mut best = None;
        let mut best_delta = 0.0;
        for pair in shortlist {
            let (proposal, order) = merged_map(&map, pair.first, pair.second);
            let bits = map_bits(&proposal, order.len())?;
            work.map_trials += 1;
            let delta = pair.population_delta + bits as f64 - current_map_bits as f64;
            if delta < best_delta {
                best_delta = delta;
                best = Some((pair, proposal, order, bits));
            }
        }
        let Some((pair, next_map, order, bits)) = best else {
            break;
        };
        let mut combined = histograms[pair.first].clone();
        combined.add(&histograms[pair.second])?;
        histograms[pair.first] = combined;
        let mut old: Vec<_> = histograms.into_iter().map(Some).collect();
        histograms = order
            .into_iter()
            .map(|i| old[i].take().expect("unique ID"))
            .collect();
        map = next_map;
        current_map_bits = bits;
        work.merges += 1;
        let plan = Plan {
            counts: histograms.iter().map(|h| h.counts.clone()).collect(),
            map: map.clone(),
            merges: work.merges,
        };
        if plans.len() == 2 {
            plans.pop();
        }
        plans.push(plan);
    }
    Ok((plans, work))
}

#[cfg(test)]
#[path = "cluster_neighborhood_tests.rs"]
mod tests;
