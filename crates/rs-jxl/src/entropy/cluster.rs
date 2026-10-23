// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Port of lib/jxl/enc_cluster.cc, baseline
//! b866b3defec13494bfbaf705db93581cda118e37: FastClusterHistograms,
//! ClusterHistograms, HistogramReindex, and the entropy/distance kernels.
//! FastLog2f's rational polynomial is from base/fast_math-inl.h.
//!
//! Implemented: largest/farthest seeding, the 48-bit distinctness threshold,
//! sequential assignment with updated centers, prefix singleton protection,
//! versioned cost-reducing ANS merges, and first-occurrence canonical indexing.
//! The merge cost is the source ANSPopulationCost (Fast normalization), including
//! serialized histogram bits, not just Shannon entropy.
//! Best refinement uses enc_cluster_bound.h's 16-bin entropy lower bound and
//! rounding margin to skip non-beneficial pairs. This source optimization is
//! active without its experimental environment gate; surviving pairs retain
//! the exact population cost, heap ordering, and version checks.
//!
//! Preexisting coding centers use the source KL divergence, remain immutable,
//! and retain their indices during incremental clustering/refit. Seed distances
//! and initial refinement rows use the caller's Rayon pool at the source work
//! thresholds, without the C++ experimental environment gates. Ordered reduction
//! preserves scalar ties; assignment and subsequent merges remain sequential.
//!
//! Native AVX2 conditioning, entropy, distance and KL dispatch lives in simd.rs,
//! with source FMA distinctions and portable scalar fallback. Source audit
//! hooks and final-plan replay are not ported. No context-map signaling cost is added:
//! the source merge heuristic does not include it either.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::ans_histogram::{AnsHistogramStrategy, NormalizedHistogram};
use crate::{Error, Result};

const MAX_ALPHABET: usize = super::prefix_code::MAX_ALPHABET_SIZE;
const MAX_CLUSTERS: usize = 256;
const MIN_DISTINCT_DISTANCE: f32 = 48.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterStrategy {
    Fastest,
    Fast,
    #[default]
    Best,
}

#[derive(Clone, Debug)]
struct Histogram {
    counts: Vec<u64>,
    total: u64,
    entropy: f32,
}

impl Histogram {
    fn new(counts: &[u64]) -> Result<Self> {
        if counts.len() > MAX_ALPHABET {
            return Err(Error::InvalidInput(
                "cluster histogram exceeds 32768 symbols".into(),
            ));
        }
        let (total, length) = super::simd::condition(counts)
            .ok_or(Error::Limit("cluster histogram weight exceeds i32::MAX"))?;
        let counts = if length == 0 {
            vec![0]
        } else {
            counts[..length].to_vec()
        };
        let entropy = shannon_entropy(&counts, total);
        Ok(Self {
            counts,
            total,
            entropy,
        })
    }

    fn add(&mut self, other: &Self) -> Result<()> {
        let total = self
            .total
            .checked_add(other.total)
            .filter(|&sum| sum <= i32::MAX as u64)
            .ok_or(Error::Limit("merged histogram weight exceeds i32::MAX"))?;
        self.counts
            .resize(self.counts.len().max(other.counts.len()), 0);
        for (destination, &source) in self.counts.iter_mut().zip(&other.counts) {
            *destination += source;
        }
        self.total = total;
        Ok(())
    }

    fn population_cost(&self) -> Result<f32> {
        // Histogram::ANSPopulationCost deliberately excludes wide prefix
        // histograms from its ANS population estimate, even in prefix mode.
        if self.counts.len() > 256 {
            return Ok(f32::MAX);
        }
        Ok(NormalizedHistogram::compute(&self.counts, AnsHistogramStrategy::Fast)?.cost)
    }
}

#[cfg(test)]
use super::simd::{entropy_term, fast_log2};

fn shannon_entropy(counts: &[u64], total: u64) -> f32 {
    super::simd::entropy(counts, total)
}

fn distance(a: &Histogram, b: &Histogram, prefix_codes: bool) -> f32 {
    if a.total == 0 || b.total == 0 {
        return 0.0;
    }
    let total = a.total + b.total;
    let combined_entropy = super::simd::combined_entropy(&a.counts, &b.counts, total);
    let result = combined_entropy - a.entropy - b.entropy;
    if !prefix_codes {
        return result;
    }
    let mut combined_symbols = 0;
    let mut a_symbols = 0;
    let mut b_symbols = 0;
    for i in 0..a.counts.len().max(b.counts.len()) {
        let ac = a.counts.get(i).copied().unwrap_or(0);
        let bc = b.counts.get(i).copied().unwrap_or(0);
        combined_symbols += usize::from(ac + bc != 0);
        a_symbols += usize::from(ac != 0);
        b_symbols += usize::from(bc != 0);
    }
    if combined_symbols <= 1 {
        return result;
    }
    let singleton_penalty = if a_symbols <= 1 { a.total as f32 } else { 0.0 }
        + if b_symbols <= 1 { b.total as f32 } else { 0.0 };
    result.max(singleton_penalty)
}

fn kl_divergence(actual: &Histogram, coding: &Histogram) -> f32 {
    if actual.total == 0 {
        return 0.0;
    }
    if coding.total == 0 {
        return f32::INFINITY;
    }
    super::simd::coding_cost(&actual.counts, &coding.counts, coding.total) - actual.entropy
}

fn pool_workers() -> usize {
    // Do not initialize a global pool when Model is used outside the encoder.
    if !crate::execution::is_serial() && rayon::current_thread_index().is_some() {
        crate::execution::worker_count()
    } else {
        1
    }
}

fn update_distances(
    input: &[Histogram],
    center: &Histogram,
    distances: &mut [f32],
    prefix_codes: bool,
    workers: usize,
) -> usize {
    let tasks = if crate::execution::is_serial() {
        1
    } else {
        super::jobs::task_count(input.len().div_ceil(2048), workers, 0).max(1)
    };
    let chunk_size = input.len().div_ceil(tasks);
    #[cfg(test)]
    let probe = super::worker_test::current();
    let update = |(chunk, distances): (usize, &mut [f32])| {
        #[cfg(test)]
        probe.record("cluster.seed");
        let begin = chunk * chunk_size;
        let mut best: Option<(usize, f32)> = None;
        for (i, current) in distances.iter_mut().enumerate() {
            if *current == 0.0 {
                continue;
            }
            *current = (*current).min(distance(&input[begin + i], center, prefix_codes));
            if best.is_none_or(|(_, previous)| *current > previous) {
                best = Some((begin + i, *current));
            }
        }
        best
    };
    let furthest: Vec<_> = if tasks > 1 {
        let mut chunks: Vec<_> = distances.chunks_mut(chunk_size).collect();
        crate::execution::map_mut(&mut chunks, |i, chunk| update((i, chunk)))
    } else {
        distances
            .chunks_mut(chunk_size)
            .enumerate()
            .map(update)
            .collect()
    };
    let mut largest = 0;
    // Each block keeps its first maximum, including rounded negative distances.
    // Ignore skipped zero entries just as the scalar source loop does.
    for (index, value) in furthest.into_iter().flatten() {
        if value > distances[largest] {
            largest = index;
        }
    }
    largest
}

fn fast_cluster(
    input: &[Histogram],
    mut output: Vec<Histogram>,
    limit: usize,
    prefix_codes: bool,
    workers: usize,
) -> Result<(Vec<Histogram>, Vec<usize>)> {
    let seed_profile = crate::profile::scope("entropy.cluster.seed");
    let previous = output.len();
    output
        .try_reserve(limit - previous)
        .map_err(|_| Error::Limit("histogram centers"))?;
    let mut symbols = vec![limit; input.len()];
    let mut distances = vec![f32::MAX; input.len()];
    let mut largest = 0;
    for (i, histogram) in input.iter().enumerate() {
        if histogram.total == 0 {
            symbols[i] = 0;
            distances[i] = 0.0;
        } else if histogram.total > input[largest].total {
            largest = i;
        }
    }
    if previous != 0 {
        for (histogram, current) in input.iter().zip(&mut distances) {
            if *current == 0.0 {
                continue;
            }
            for center in &output {
                *current = (*current).min(kl_divergence(histogram, center));
            }
        }
        let mut furthest = 0;
        for i in 1..input.len() {
            if distances[i] > distances[furthest] {
                furthest = i;
            }
        }
        if distances[furthest] > 0.0 {
            largest = furthest;
        }
    }
    while output.len() < limit {
        symbols[largest] = output.len();
        output.push(input[largest].clone());
        distances[largest] = 0.0;
        let center = output.last().expect("just inserted center");
        largest = update_distances(input, center, &mut distances, prefix_codes, workers);
        if distances[largest] < MIN_DISTINCT_DISTANCE {
            break;
        }
    }
    crate::profile::note(
        "entropy.cluster.seed",
        format_args!(
            "inputs={} centers={} previous={previous} limit={limit} workers={workers}",
            input.len(),
            output.len()
        ),
    );
    drop(seed_profile);
    let _assign_profile = crate::profile::scope("entropy.cluster.assign");
    for i in 0..input.len() {
        if symbols[i] != limit {
            continue;
        }
        let mut best = 0;
        let mut best_distance = f32::MAX;
        for (j, histogram) in output.iter().enumerate() {
            let candidate = if j < previous {
                kl_divergence(&input[i], histogram)
            } else {
                distance(&input[i], histogram, prefix_codes)
            };
            if candidate < best_distance {
                best = j;
                best_distance = candidate;
            }
        }
        if !best_distance.is_finite() || best_distance == f32::MAX {
            return Err(Error::Encoding(
                "no finite histogram cluster distance".into(),
            ));
        }
        if best >= previous {
            output[best].add(&input[i])?;
            output[best].entropy = shannon_entropy(&output[best].counts, output[best].total);
        }
        symbols[i] = best;
    }
    Ok((output, symbols))
}

#[derive(Clone, Copy, Debug)]
struct Pair {
    cost: f32,
    first: usize,
    second: usize,
    version: u32,
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pair {}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse the source's lexicographic key.
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.first.cmp(&self.first))
            .then_with(|| other.second.cmp(&self.second))
            .then_with(|| other.version.cmp(&self.version))
    }
}

fn pair_cost(
    histograms: &[Histogram],
    versions: &[u32],
    first: usize,
    second: usize,
) -> Result<Option<Pair>> {
    let mut combined = histograms[first].clone();
    combined.add(&histograms[second])?;
    let cost =
        combined.population_cost()? - (histograms[first].entropy + histograms[second].entropy);
    Ok((cost < 0.0).then_some(Pair {
        cost,
        first,
        second,
        version: versions[first].max(versions[second]),
    }))
}

fn enqueue_pair(
    histograms: &[Histogram],
    versions: &[u32],
    first: usize,
    second: usize,
    heap: &mut BinaryHeap<Pair>,
) -> Result<()> {
    if let Some(pair) = pair_cost(histograms, versions, first, second)? {
        heap.try_reserve(1)
            .map_err(|_| Error::Limit("histogram merge queue"))?;
        heap.push(pair);
    }
    Ok(())
}

type EntropySketch = [f64; 16];

fn entropy_sketch(counts: &[u64]) -> EntropySketch {
    let mut sketch = [0.0; 16];
    for (symbol, &count) in counts.iter().enumerate() {
        let mut bucket = symbol;
        if symbol >= 8 {
            bucket = 8;
            let mut value = symbol >> 4;
            while value != 0 && bucket < 15 {
                bucket += 1;
                value >>= 1;
            }
        }
        sketch[bucket] += count as f64;
    }
    sketch
}

fn merge_lower_bound(a: &EntropySketch, b: &EntropySketch) -> f64 {
    let total: f64 = a.iter().zip(b).map(|(a, b)| a + b).sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for (a, b) in a.iter().zip(b) {
        let count = a + b;
        if count != 0.0 {
            entropy += count * (total / count).log2();
        }
    }
    // Source margin covers float/fixed-point population-cost rounding.
    (entropy - 1.0 - total * 0.0001).max(0.0)
}

fn refine(histograms: &mut [Histogram], symbols: &mut [usize]) -> Result<()> {
    refine_with_bound(histograms, symbols, true).map(|_| ())
}

fn refine_with_bound(
    histograms: &mut [Histogram],
    symbols: &mut [usize],
    use_bound: bool,
) -> Result<usize> {
    refine_with_workers(histograms, symbols, use_bound, pool_workers())
}

fn refine_with_workers(
    histograms: &mut [Histogram],
    symbols: &mut [usize],
    use_bound: bool,
    workers: usize,
) -> Result<usize> {
    let _profile = crate::profile::scope("entropy.cluster.refine");
    crate::profile::note(
        "entropy.cluster.refine",
        format_args!("centers={}", histograms.len()),
    );
    for histogram in histograms.iter_mut() {
        histogram.entropy = histogram.population_cost()?;
    }
    let mut sketches: Option<Vec<_>> = use_bound.then(|| {
        histograms
            .iter()
            .map(|h| entropy_sketch(&h.counts))
            .collect()
    });
    let mut bounded_pairs = 0usize;
    let mut versions = vec![1u32; histograms.len()];
    let mut renumbering: Vec<usize> = (0..histograms.len()).collect();
    let mut next_version = 2;
    let mut pairs = BinaryHeap::new();
    #[cfg(test)]
    let probe = super::worker_test::current();
    let build_row = |first: usize| -> Result<(Vec<Pair>, usize)> {
        #[cfg(test)]
        probe.record("cluster.refine");
        let mut row = Vec::new();
        let mut bounded = 0;
        for second in first + 1..histograms.len() {
            if sketches.as_ref().is_some_and(|s| {
                merge_lower_bound(&s[first], &s[second])
                    >= f64::from(histograms[first].entropy) + f64::from(histograms[second].entropy)
            }) {
                bounded += 1;
                continue;
            }
            if let Some(pair) = pair_cost(histograms, &versions, first, second)? {
                row.try_reserve(1)
                    .map_err(|_| Error::Limit("histogram merge row"))?;
                row.push(pair);
            }
        }
        Ok((row, bounded))
    };
    let parallel = !crate::execution::is_serial() && workers > 1 && histograms.len() >= 32;
    let rows: Vec<_> = if parallel {
        #[cfg(test)]
        probe.record("cluster.refine.parallel_dispatch");
        let order: Vec<_> = (0..histograms.len()).collect();
        super::jobs::map(&order, workers, 1 << 20, build_row)
    } else {
        (0..histograms.len()).map(build_row).collect()
    };
    // Ordered rows preserve both the source heap insertion order and the first
    // error. Workers never mutate histograms, versions, or the merge queue.
    for row in rows {
        let (row, bounded) = row?;
        bounded_pairs += bounded;
        pairs
            .try_reserve(row.len())
            .map_err(|_| Error::Limit("histogram merge queue"))?;
        pairs.extend(row);
    }
    while let Some(pair) = pairs.pop() {
        let Pair {
            first,
            second,
            version,
            ..
        } = pair;
        if versions[first] == 0
            || versions[second] == 0
            || version != versions[first].max(versions[second])
        {
            continue;
        }
        let other = histograms[second].clone();
        histograms[first].add(&other)?;
        if let Some(sketches) = &mut sketches {
            let other = sketches[second];
            for (destination, source) in sketches[first].iter_mut().zip(other) {
                *destination += source;
            }
        }
        histograms[first].entropy = histograms[first].population_cost()?;
        for target in &mut renumbering {
            if *target == second {
                *target = first;
            }
        }
        versions[second] = 0;
        versions[first] = next_version;
        next_version += 1;
        for j in 0..histograms.len() {
            if j != first && versions[j] != 0 {
                if sketches.as_ref().is_some_and(|s| {
                    merge_lower_bound(&s[first], &s[j])
                        >= f64::from(histograms[first].entropy) + f64::from(histograms[j].entropy)
                }) {
                    bounded_pairs += 1;
                    continue;
                }
                enqueue_pair(
                    histograms,
                    &versions,
                    first.min(j),
                    first.max(j),
                    &mut pairs,
                )?;
            }
        }
    }
    // Canonical reindexing below also removes dead centers, so an intermediate
    // compact-and-reverse-renumber pass is unnecessary.
    for symbol in symbols {
        *symbol = renumbering[*symbol];
    }
    crate::profile::note(
        "entropy.cluster.refine.result",
        format_args!(
            "merges={} bound={use_bound} skipped_pairs={bounded_pairs} parallel_pairs={parallel}",
            next_version - 2
        ),
    );
    Ok(bounded_pairs)
}

/// Source best-mode clustering for ANS histograms. Output identifiers are dense,
/// ordered by their first input occurrence, and fit u8. Every output histogram
/// is the exact sum of the inputs assigned to it, with trailing zeros removed.
/// Empty histograms are represented by `[0]`, without synthetic observations.
///
/// The total weight across all inputs is bounded by i32::MAX so every candidate
/// merge stays inside the source normalizer's arithmetic domain. The requested
/// cluster count is capped at 256 and the input count; zero is invalid.
pub fn cluster(counts: &[Vec<u64>], max_clusters: usize) -> Result<(Vec<Vec<u64>>, Vec<u8>)> {
    cluster_with_options(counts, max_clusters, ClusterStrategy::Best, false)
}

/// Effort-policy entry point. `prefix_codes` enables the source singleton
/// distance guard; best-mode refinement still uses ANS population cost, exactly
/// as enc_cluster.cc does even when force_huffman is set.
pub fn cluster_with_options(
    counts: &[Vec<u64>],
    max_clusters: usize,
    strategy: ClusterStrategy,
    prefix_codes: bool,
) -> Result<(Vec<Vec<u64>>, Vec<u8>)> {
    if max_clusters == 0 {
        return Err(Error::InvalidInput(
            "maximum cluster count must be positive".into(),
        ));
    }
    cluster_with_preexisting(counts, &[], max_clusters, strategy, prefix_codes)
}

/// Incremental source ClusterHistograms, with immutable pre-stored coding
/// distributions. `previous` must contain coding weights, not old raw sample
/// counts: ANS frequencies, or `1 << (15 - depth)` for nonzero prefix depths.
/// Zero-depth prefix entries have zero weight, including source singletons.
/// All inputs must use the same symbol/HybridUint mapping as those old tables.
///
/// Existing IDs stay pinned even if unused. At most `max_new_clusters` centers
/// are added (at most four for Fastest), subject to the total u8 limit of 256.
/// Zero new centers performs a frozen-table refit; unsupported symbols then
/// return an error rather than silently assigning an unusable coding table.
/// Only new centers absorb assigned counts. Best refinement runs only when
/// `previous` is empty, exactly as in the source incremental path.
///
/// Old weights are preserved apart from trimming trailing zeros. New weights
/// sum only inputs assigned to new centers. Empty inputs map to ID zero.
/// Input weight is bounded by i32::MAX; each old coding center has the same
/// individual bound. The returned map describes only these new inputs. A shared
/// wire model must retain its preceding context-map entries and encoding tables,
/// including any old centers unused by this batch; do not serialize this as a
/// standalone model with unreachable trailing histogram IDs.
pub fn cluster_with_preexisting(
    counts: &[Vec<u64>],
    previous: &[Vec<u64>],
    max_new_clusters: usize,
    strategy: ClusterStrategy,
    prefix_codes: bool,
) -> Result<(Vec<Vec<u64>>, Vec<u8>)> {
    let _profile = crate::profile::scope("entropy.cluster");
    if crate::profile::enabled() {
        crate::profile::note(
            "entropy.cluster",
            format_args!(
                "inputs={} bins={} previous={} new_cap={max_new_clusters} strategy={strategy:?} prefix_guard={prefix_codes}",
                counts.len(),
                counts.iter().map(Vec::len).sum::<usize>(),
                previous.len(),
            ),
        );
    }
    if previous.len() > MAX_CLUSTERS {
        return Err(Error::InvalidInput(
            "preexisting cluster count exceeds 256".into(),
        ));
    }
    if previous.is_empty() && max_new_clusters == 0 {
        return Err(Error::InvalidInput(
            "maximum cluster count must be positive".into(),
        ));
    }
    let centers = previous
        .iter()
        .map(|counts| Histogram::new(counts))
        .collect::<Result<Vec<_>>>()?;
    if counts.is_empty() {
        return Ok((centers.into_iter().map(|h| h.counts).collect(), Vec::new()));
    }
    let mut input = Vec::new();
    input
        .try_reserve_exact(counts.len())
        .map_err(|_| Error::Limit("input histograms"))?;
    let mut total = 0u64;
    for counts in counts {
        let histogram = Histogram::new(counts)?;
        total = total
            .checked_add(histogram.total)
            .filter(|&sum| sum <= i32::MAX as u64)
            .ok_or(Error::Limit("total clustering weight exceeds i32::MAX"))?;
        input.push(histogram);
    }
    let mut additional = max_new_clusters
        .min(MAX_CLUSTERS - previous.len())
        .min(counts.len());
    if strategy == ClusterStrategy::Fastest {
        additional = additional.min(4);
    }
    let (mut histograms, mut symbols) = fast_cluster(
        &input,
        centers,
        previous.len() + additional,
        prefix_codes,
        pool_workers(),
    )?;
    if previous.is_empty() && strategy == ClusterStrategy::Best {
        refine(&mut histograms, &mut symbols)?;
    }
    let mut canonical = vec![usize::MAX; histograms.len()];
    let mut output = Vec::new();
    let mut context_map = Vec::with_capacity(symbols.len());
    for index in 0..previous.len() {
        canonical[index] = index;
        output.push(std::mem::take(&mut histograms[index].counts));
    }
    for symbol in symbols {
        if canonical[symbol] == usize::MAX {
            canonical[symbol] = output.len();
            output.push(std::mem::take(&mut histograms[symbol].counts));
        }
        context_map.push(canonical[symbol] as u8);
    }
    Ok((output, context_map))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_scope_inside_pool_keeps_seed_and_refine_on_caller_thread() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let counts: Vec<_> = (0..2049)
            .map(|i| {
                let mut counts = vec![0; 32];
                counts[i % 32] = 100;
                counts
            })
            .collect();
        pool.install(|| {
            let caller = std::thread::current().id();
            assert_eq!(pool_workers(), 4);
            crate::execution::serial(|| {
                assert_eq!(pool_workers(), 1);
                let ((centers, map), events) = super::super::worker_test::observe(|| {
                    cluster_with_options(&counts, 32, ClusterStrategy::Best, false).unwrap()
                });
                assert_eq!(centers.len(), 32);
                assert_eq!(map.len(), counts.len());
                assert_eq!(events.len(), 64);
                // Both workloads meet their parallel thresholds. One seed task
                // per center also catches dispatch even if nothing is stolen.
                assert_eq!(
                    events.iter().filter(|e| e.stage == "cluster.seed").count(),
                    32
                );
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| e.stage == "cluster.refine")
                        .count(),
                    32
                );
                assert!(
                    events.iter().all(|e| e.thread == caller && e.serial),
                    "{events:?}"
                );

                // An explicit internal worker request cannot override a null
                // runner either: check the gates at the actual dispatch sites.
                let input: Vec<_> = counts.iter().map(|h| Histogram::new(h).unwrap()).collect();
                let mut distances = vec![f32::MAX; input.len()];
                let mut centers = input[..32].to_vec();
                let mut symbols: Vec<_> = (0..32).collect();
                let (_, events) = super::super::worker_test::observe(|| {
                    update_distances(&input, &input[0], &mut distances, false, 4);
                    refine_with_workers(&mut centers, &mut symbols, true, 4).unwrap();
                });
                assert_eq!(events.len(), 33);
                assert_eq!(
                    events.iter().filter(|e| e.stage == "cluster.seed").count(),
                    1
                );
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| e.stage == "cluster.refine")
                        .count(),
                    32
                );
                assert!(
                    events.iter().all(|e| e.thread == caller && e.serial),
                    "{events:?}"
                );
            });
            assert!(!crate::execution::is_serial());
            assert_eq!(pool_workers(), 4);
        });
    }

    fn assert_same_histograms(actual: &[Histogram], expected: &[Histogram]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.counts, b.counts);
            assert_eq!(a.total, b.total);
            assert_eq!(a.entropy.to_bits(), b.entropy.to_bits());
        }
    }

    #[test]
    fn kl_matches_coding_cost_and_rejects_missing_support() {
        let empty = Histogram::new(&[]).unwrap();
        let actual = Histogram::new(&[80, 20, 0]).unwrap();
        let coding = Histogram::new(&[1024, 3072, 0, 0]).unwrap();
        let cost = -80.0f64 * 0.25f64.log2() - 20.0 * 0.75f64.log2();
        let entropy = -80.0f64 * 0.8f64.log2() - 20.0 * 0.2f64.log2();
        assert!((f64::from(kl_divergence(&actual, &coding)) - (cost - entropy)).abs() < 0.001);
        assert_eq!(kl_divergence(&empty, &empty), 0.0);
        assert_eq!(kl_divergence(&actual, &empty), f32::INFINITY);
        assert_eq!(
            kl_divergence(&actual, &Histogram::new(&[4096]).unwrap()),
            f32::INFINITY
        );
        assert_eq!(
            kl_divergence(&actual, &Histogram::new(&[0, 4096]).unwrap()),
            f32::INFINITY
        );
    }

    #[test]
    fn incremental_centers_are_pinned_and_only_new_centers_absorb_counts() {
        let previous = vec![vec![0, 4096], vec![4096], vec![2048, 2048]];
        let input = vec![
            vec![9000],
            vec![0, 7000],
            vec![],
            vec![0, 0, 5000],
            vec![0, 0, 2000],
        ];
        for strategy in [
            ClusterStrategy::Fastest,
            ClusterStrategy::Fast,
            ClusterStrategy::Best,
        ] {
            let (centers, map) =
                cluster_with_preexisting(&input, &previous, 10, strategy, false).unwrap();
            assert_eq!(&centers[..previous.len()], &previous);
            assert_eq!(centers[3..], [vec![0, 0, 7000]]);
            assert_eq!(map, [1, 0, 0, 3, 3]);
        }
        let (centers, map) = cluster_with_preexisting(
            &[vec![5], vec![4]],
            &[vec![4096]],
            1,
            ClusterStrategy::Best,
            false,
        )
        .unwrap();
        // Source seeds once before checking the 48-bit threshold, even if an
        // old table already supports everything. The unused old ID stays pinned.
        assert_eq!(centers, [vec![4096], vec![9]]);
        assert_eq!(map, [1, 1]);
    }

    #[test]
    fn incremental_seed_uses_first_kl_maximum_not_largest_new_population() {
        let (centers, map) = cluster_with_preexisting(
            &[vec![100], vec![0, 5], vec![0, 0, 500]],
            &[vec![4096]],
            1,
            ClusterStrategy::Best,
            false,
        )
        .unwrap();
        assert_eq!(map, [0, 1, 1]);
        assert_eq!(centers, [vec![4096], vec![0, 5, 500]]);
    }

    #[test]
    fn frozen_refit_retains_unused_centers_ties_and_rejects_unencodable_symbols() {
        let previous = vec![vec![2048, 2048], vec![2048, 2048], vec![0, 0, 4096]];
        let input = vec![vec![100, 100], vec![0, 0, 77], vec![]];
        let (centers, map) =
            cluster_with_preexisting(&input, &previous, 0, ClusterStrategy::Best, false).unwrap();
        assert_eq!(centers, previous);
        assert_eq!(map, [0, 2, 0]);
        assert!(
            cluster_with_preexisting(
                &[vec![0, 0, 0, 1]],
                &previous,
                0,
                ClusterStrategy::Fast,
                false,
            )
            .is_err()
        );
        // Source HistogramFromSymbolInfo represents zero-depth prefix tables
        // as all-zero coding weights, not an invented singleton probability.
        assert!(
            cluster_with_preexisting(&[vec![1]], &[vec![0]], 0, ClusterStrategy::Fast, true,)
                .is_err()
        );
        assert_eq!(
            cluster_with_preexisting(&[], &previous, 0, ClusterStrategy::Fast, false,).unwrap(),
            (previous, vec![])
        );
    }

    #[test]
    fn incremental_histogram_budget_and_weights_are_checked() {
        let previous = vec![vec![4096]; 256];
        let (centers, map) = cluster_with_preexisting(
            &[vec![1]],
            &previous,
            usize::MAX,
            ClusterStrategy::Best,
            false,
        )
        .unwrap();
        assert_eq!(centers, previous);
        assert_eq!(map, [0]);
        assert!(
            cluster_with_preexisting(&[vec![0, 1]], &previous, 1, ClusterStrategy::Best, false,)
                .is_err()
        );
        assert!(
            cluster_with_preexisting(&[], &vec![vec![1]; 257], 0, ClusterStrategy::Fast, false,)
                .is_err()
        );
        assert!(
            cluster_with_preexisting(&[], &[vec![u64::MAX]], 0, ClusterStrategy::Fast, false,)
                .is_err()
        );
        assert!(
            cluster_with_preexisting(
                &[vec![i32::MAX as u64], vec![1]],
                &[vec![4096]],
                0,
                ClusterStrategy::Fast,
                false,
            )
            .is_err()
        );
        let input: Vec<_> = (1..10)
            .map(|symbol| {
                let mut counts = vec![0; symbol + 1];
                counts[symbol] = 1000;
                counts
            })
            .collect();
        let (centers, map) =
            cluster_with_preexisting(&input, &[vec![4096]], 100, ClusterStrategy::Fastest, false)
                .unwrap();
        assert_eq!(centers.len(), 5);
        assert_eq!(centers[0], [4096]);
        assert!(map.iter().all(|&id| (1..=4).contains(&id)));
        assert_eq!(centers[1..].iter().flatten().sum::<u64>(), 9000);
    }

    #[test]
    fn incremental_mapped_tokens_decode_with_preexisting_ans_tables() {
        use super::super::ans::{AnsCode, AnsToken, tests::decoder_table, write_tokens};
        use super::super::ans_histogram::test_decode::Reader;
        use crate::bit_writer::BitWriter;

        let mut codes = vec![
            AnsCode::from_counts(&[500, 17, 0, 2], 8).unwrap(),
            AnsCode::from_counts(&[0, 0, 100], 8).unwrap(),
        ];
        let previous: Vec<Vec<u64>> = codes
            .iter()
            .map(|code| {
                code.normalized_counts()
                    .iter()
                    .map(|&count| u64::from(count))
                    .collect()
            })
            .collect();
        let input = vec![
            vec![5000, 170, 0, 20],
            vec![0, 0, 1000],
            vec![0, 0, 0, 0, 80],
        ];
        let (centers, map) =
            cluster_with_preexisting(&input, &previous, 1, ClusterStrategy::Best, false).unwrap();
        assert_eq!(&centers[..previous.len()], &previous);
        for counts in &centers[previous.len()..] {
            codes.push(AnsCode::from_counts(counts, 8).unwrap());
        }
        let mut tokens = Vec::new();
        for (context, counts) in input.iter().enumerate() {
            for (symbol, &count) in counts.iter().enumerate() {
                for i in 0..count {
                    tokens.push(AnsToken {
                        context: usize::from(map[context]),
                        symbol,
                        nbits: 3,
                        bits: (i % 8) as u32,
                    });
                }
            }
        }
        let mut writer = BitWriter::new(65536);
        for code in &codes {
            code.write_histogram(&mut writer).unwrap();
        }
        write_tokens(&codes, &tokens, &mut writer).unwrap();
        let end = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        let mut reader = Reader {
            bytes: &bytes,
            position: 0,
        };
        let counts: Vec<_> = codes.iter().map(|_| reader.histogram()).collect();
        let tables: Vec<_> = counts
            .iter()
            .map(|counts| decoder_table(counts, 8))
            .collect();
        let mut state = reader.read(32);
        for token in tokens {
            let (symbol, offset) = tables[token.context][(state & 4095) as usize];
            assert_eq!(symbol, token.symbol);
            state = u32::from(counts[token.context][symbol]) * (state >> 12) + offset;
            if state < 65536 {
                state = (state << 16) | reader.read(16);
            }
            assert_eq!(reader.read(token.nbits), token.bits);
        }
        assert_eq!(state, 0x13_0000);
        assert_eq!(reader.position, end);
    }

    #[test]
    fn seed_workers_preserve_scalar_distances_centers_and_ties() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let input: Vec<_> = (0..8193)
                .map(|i| {
                    let mut counts = vec![0; 64];
                    if i % 17 != 0 {
                        counts[i % 16] = 100 + (i % 7) as u64;
                        if i % 3 == 0 {
                            counts[32 + i % 16] = 1;
                        }
                    }
                    Histogram::new(&counts).unwrap()
                })
                .collect();
            for prefix in [false, true] {
                for previous in [vec![], vec![Histogram::new(&[2048, 2048]).unwrap()]] {
                    let (expected, map) =
                        fast_cluster(&input, previous.clone(), 12, prefix, 1).unwrap();
                    let (actual, actual_map) =
                        fast_cluster(&input, previous, 12, prefix, 4).unwrap();
                    assert_eq!(actual_map, map);
                    assert_same_histograms(&actual, &expected);
                }
            }
        });
    }

    #[test]
    fn refinement_workers_preserve_merge_versions_and_error_order() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let input: Vec<_> = (0..64)
                .map(|i| {
                    let mut counts = vec![0; 32];
                    counts[i / 2] = 100 + i as u64;
                    Histogram::new(&counts).unwrap()
                })
                .collect();
            for bound in [false, true] {
                let mut actual = input.clone();
                let mut expected = input.clone();
                let mut map: Vec<_> = (0..64).collect();
                let mut actual_map = map.clone();
                let skipped = refine_with_workers(&mut expected, &mut map, bound, 1).unwrap();
                assert_eq!(
                    refine_with_workers(&mut actual, &mut actual_map, bound, 4).unwrap(),
                    skipped
                );
                assert_eq!(actual_map, map);
                assert!(map.iter().enumerate().any(|(i, &target)| i != target));
                assert_same_histograms(&actual, &expected);
            }
            let mut invalid = vec![Histogram::new(&[1]).unwrap(); 32];
            invalid[0] = Histogram::new(&[i32::MAX as u64]).unwrap();
            let mut other = invalid.clone();
            let a = refine_with_workers(&mut invalid, &mut [], false, 1).unwrap_err();
            let b = refine_with_workers(&mut other, &mut [], false, 4).unwrap_err();
            assert_eq!(a.to_string(), b.to_string());
        });
    }

    #[test]
    fn source_entropy_sketch_bucket_boundaries_and_additivity() {
        for (symbol, bucket) in [
            (0, 0),
            (7, 7),
            (8, 8),
            (15, 8),
            (16, 9),
            (31, 9),
            (32, 10),
            (63, 10),
            (64, 11),
            (127, 11),
            (128, 12),
            (255, 12),
            (256, 13),
            (511, 13),
            (512, 14),
            (1023, 14),
            (1024, 15),
            (32767, 15),
        ] {
            let mut counts = vec![0; symbol + 1];
            counts[symbol] = 17;
            let mut expected = [0.0; 16];
            expected[bucket] = 17.0;
            assert_eq!(entropy_sketch(&counts), expected);
        }
        let a: Vec<u64> = (0..513).map(|i| (i % 71) as u64).collect();
        let b: Vec<u64> = (0..513).map(|i| ((i * 17) % 29) as u64).collect();
        let sum: Vec<u64> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
        let mut combined = entropy_sketch(&a);
        for (destination, source) in combined.iter_mut().zip(entropy_sketch(&b)) {
            *destination += source;
        }
        assert_eq!(combined, entropy_sketch(&sum));
        assert_eq!(merge_lower_bound(&[0.0; 16], &[0.0; 16]), 0.0);
    }

    #[test]
    fn source_merge_bound_never_prunes_beneficial_fixture_pairs() {
        let counts = vec![
            vec![],
            vec![0; 256],
            vec![500_000_000],
            vec![0, 500_000_000],
            vec![1; 256],
            vec![4096; 256],
            (0..256)
                .map(|i| if i == 128 { 500_000_000 } else { 1 })
                .collect(),
            (0..256)
                .map(|i| ((i * i * 71 + 17) % 1013) as u64)
                .collect(),
        ];
        for a in &counts {
            for b in &counts {
                let a = Histogram::new(a).unwrap();
                let b = Histogram::new(b).unwrap();
                let mut combined = a.clone();
                combined.add(&b).unwrap();
                let bound =
                    merge_lower_bound(&entropy_sketch(&a.counts), &entropy_sketch(&b.counts));
                let cost = combined.population_cost().unwrap();
                assert!(bound <= f64::from(cost));
                if bound
                    >= f64::from(a.population_cost().unwrap())
                        + f64::from(b.population_cost().unwrap())
                {
                    assert!(
                        cost - (a.population_cost().unwrap() + b.population_cost().unwrap()) >= 0.0
                    );
                }
            }
        }
    }

    #[test]
    fn bounded_refinement_matches_unpruned_merges_and_remapping() {
        let mut skipped = 0;
        let mut remapped = 0;
        for scale in [1u64, 100, 1_000_000] {
            for shape in 0..3 {
                let input: Vec<Histogram> = (0..16)
                    .map(|i| {
                        let mut counts = vec![0; 64];
                        let symbol = i / 2;
                        counts[symbol] = scale * (1 + i as u64 % 2);
                        if shape >= 1 {
                            counts[symbol + 16] = 2 * counts[symbol];
                        }
                        if shape == 2 {
                            counts[63] = scale;
                        }
                        Histogram::new(&counts).unwrap()
                    })
                    .collect();
                let mut bounded = input.clone();
                let mut unpruned = input.clone();
                let mut actual: Vec<usize> = (0..input.len()).collect();
                let mut expected = actual.clone();
                skipped += refine_with_bound(&mut bounded, &mut actual, true).unwrap();
                assert_eq!(
                    refine_with_bound(&mut unpruned, &mut expected, false).unwrap(),
                    0
                );
                assert_eq!(actual, expected, "scale={scale} shape={shape}");
                remapped += actual
                    .iter()
                    .enumerate()
                    .filter(|&(i, &target)| i != target)
                    .count();
                for (a, b) in bounded.iter().zip(&unpruned) {
                    assert_eq!(a.counts, b.counts);
                    assert_eq!(a.total, b.total);
                    assert_eq!(a.entropy.to_bits(), b.entropy.to_bits());
                }
            }
        }
        assert!(
            skipped > 0,
            "the source lower-bound fast path must actually run"
        );
        assert!(
            remapped > 0,
            "the fixtures must exercise sketch updates after merging"
        );
    }

    fn assert_partition(input: &[Vec<u64>], output: &[Vec<u64>], map: &[u8], limit: usize) {
        assert_eq!(input.len(), map.len());
        assert!(output.len() <= limit.min(256));
        let alphabet = input.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let mut recomputed = vec![vec![0u64; alphabet]; output.len()];
        let mut next = 0;
        let mut seen = vec![false; output.len()];
        for (counts, &cluster) in input.iter().zip(map) {
            let cluster = usize::from(cluster);
            assert!(cluster < output.len());
            if !seen[cluster] {
                assert_eq!(cluster, next);
                next += 1;
                seen[cluster] = true;
            }
            for (symbol, &count) in counts.iter().enumerate() {
                recomputed[cluster][symbol] += count;
            }
        }
        assert!(seen.into_iter().all(|value| value));
        for (counts, expected) in output.iter().zip(&recomputed) {
            for (symbol, &expected) in expected.iter().enumerate() {
                assert_eq!(counts.get(symbol).copied().unwrap_or(0), expected);
            }
        }
    }

    #[test]
    fn wide_prefix_histograms_preserve_all_bins() {
        let mut a = vec![0; 32768];
        a[32767] = 100;
        let mut b = vec![0; 513];
        b[512] = 90;
        let input = [a, b, vec![], vec![100]];
        for strategy in [
            ClusterStrategy::Fastest,
            ClusterStrategy::Fast,
            ClusterStrategy::Best,
        ] {
            let (histograms, map) = cluster_with_options(&input, 3, strategy, true).unwrap();
            assert_partition(&input, &histograms, &map, 3);
        }
    }

    #[test]
    fn source_fast_log_and_entropy_are_close_to_independent_log2() {
        for count in 1..=4096 {
            let probability = count as f32 / 4096.0;
            assert!((fast_log2(probability) - probability.log2()).abs() < 0.000006);
        }
        assert_eq!(shannon_entropy(&[0, 0], 0), 0.0);
        assert_eq!(shannon_entropy(&[0, 100], 100), 0.0);
        assert!((shannon_entropy(&[1, 1], 2) - 2.0).abs() < 0.00001);
        let total = i32::MAX as u64;
        assert_eq!(entropy_term(total - 1, 1.0 / total as f32, total), 0.0);
    }

    #[test]
    fn empty_singleton_and_canonical_known_partitions() {
        assert_eq!(cluster(&[], 1).unwrap(), (vec![], vec![]));
        assert_eq!(
            cluster(&[vec![], vec![0; 8]], 256).unwrap(),
            (vec![vec![0]], vec![0, 0])
        );
        let input = vec![vec![0, 10_000], vec![20_000], vec![0, 5_000], vec![]];
        let (output, map) = cluster(&input, 256).unwrap();
        assert_eq!(map, [0, 1, 0, 1]);
        assert_eq!(output, [vec![0, 15_000], vec![20_000]]);
        assert_partition(&input, &output, &map, 256);
        let (output, map) = cluster(&input, 1).unwrap();
        assert_eq!(map, [0; 4]);
        assert_eq!(output, [vec![20_000, 15_000]]);
    }

    #[test]
    fn threshold_and_prefix_singleton_guard_follow_source() {
        let input = vec![vec![1], vec![0, 1]];
        let (output, map) = cluster_with_options(&input, 2, ClusterStrategy::Fast, false).unwrap();
        assert_eq!(output, [vec![1, 1]]);
        assert_eq!(map, [0, 0]);
        let a = Histogram::new(&[1_000_000]).unwrap();
        let b = Histogram::new(&[1000, 1]).unwrap();
        assert!(distance(&a, &b, false) < 48.0);
        assert!(distance(&a, &b, true) >= 1_000_000.0);
    }

    #[test]
    fn merge_heap_removes_stale_pairs_and_never_raises_population_cost() {
        let mut histograms = vec![
            Histogram::new(&[100, 100]).unwrap(),
            Histogram::new(&[200, 200]).unwrap(),
            Histogram::new(&[300, 300]).unwrap(),
            Histogram::new(&[0, 0, 1000]).unwrap(),
        ];
        let original = histograms.clone();
        let before: f32 = original
            .iter()
            .map(|histogram| histogram.population_cost().unwrap())
            .sum();
        let mut map = vec![0, 1, 2, 3];
        refine(&mut histograms, &mut map).unwrap();
        assert_eq!(map, [0, 0, 0, 3]);
        assert_eq!(histograms[0].counts, [600, 600]);
        let after =
            histograms[0].population_cost().unwrap() + histograms[3].population_cost().unwrap();
        assert!(after < before);
    }

    #[test]
    fn every_strategy_preserves_weights_caps_and_determinism() {
        let input: Vec<Vec<u64>> = (0..15)
            .map(|context| {
                (0..32)
                    .map(|symbol| {
                        if symbol % 5 == context % 5 {
                            (1000 + context * 17 + symbol * 13) as u64
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();
        for strategy in [
            ClusterStrategy::Fastest,
            ClusterStrategy::Fast,
            ClusterStrategy::Best,
        ] {
            for limit in [1, 2, 4, 12, 300] {
                let result = cluster_with_options(&input, limit, strategy, false).unwrap();
                let repeat = cluster_with_options(&input, limit, strategy, false).unwrap();
                assert_eq!(result, repeat);
                assert_partition(&input, &result.0, &result.1, limit);
                if strategy == ClusterStrategy::Fastest {
                    assert!(result.0.len() <= 4);
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_alphabets_and_unbounded_weights() {
        assert!(cluster(&[], 0).is_err());
        assert!(cluster(&[vec![0; MAX_ALPHABET + 1]], 1).is_err());
        assert!(cluster(&[vec![u64::MAX]], 1).is_err());
        assert!(cluster(&[vec![i32::MAX as u64], vec![1]], 2).is_err());
        assert_eq!(
            cluster(&[vec![i32::MAX as u64]], 1).unwrap().0,
            [vec![i32::MAX as u64]]
        );
    }
}
