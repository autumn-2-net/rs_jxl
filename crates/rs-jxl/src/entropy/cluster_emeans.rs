//! Experimental categorical adaptation of mini-batch e-means (Bessa et al.,
//! arXiv:2611.09501). This is not the paper's Euclidean k-means algorithm:
//! histogram populations weight categorical cross-entropy, not vector MSE.
//! EMA, cosine steps and donor splitting only propose context assignments.
//! Final tables sum original integer counts exactly, then use ordinary ANS
//! population-cost merging. No float probabilities are written to the stream.

use super::{Histogram, seed_histograms, shannon_entropy};
use crate::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "cluster_emeans_refit.rs"]
mod refit;

static WORK: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Process-cumulative experiment diagnostics, not encoding policy. Read only
/// after all encode workers finish; concurrent snapshots are not transactional.
#[derive(Clone, Copy, Debug, Default)]
pub struct Work {
    pub calls: u64,
    pub fallbacks: u64,
    pub contexts: u64,
    pub initial_centers: u64,
    pub occupied_centers: u64,
    pub parallel_batches: u64,
}

pub fn work() -> Work {
    let n = WORK.each_ref().map(|v| v.load(Ordering::Relaxed));
    Work {
        calls: n[0],
        fallbacks: n[1],
        contexts: n[2],
        initial_centers: n[3],
        occupied_centers: n[4],
        parallel_batches: n[5],
    }
}

pub(crate) fn fallback() {
    WORK[1].fetch_add(1, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Initialization {
    #[default]
    Random,
    Farthest,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Distance {
    #[default]
    CrossEntropy,
    /// Squared Hellinger distance with population-weighted sqrt centroids.
    Hellinger,
    /// Entropy increase when combining an input with a projected EMA center.
    MergeEntropy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub passes: u8,
    pub initialization: Initialization,
    pub distance: Distance,
    /// Optional ordered relocation passes scored by native Fast ANS cost.
    pub polish_passes: u8,
    /// Reassign once against frozen final centers before integer regrouping.
    pub final_assignment: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            passes: 3,
            initialization: Initialization::Random,
            distance: Distance::CrossEntropy,
            polish_passes: 0,
            final_assignment: false,
        }
    }
}

impl Options {
    pub(crate) fn validate(self) -> Result<()> {
        if self.passes == 0 {
            return Err(Error::InvalidInput(
                "EMA clustering needs at least one pass".into(),
            ));
        }
        Ok(())
    }
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn shuffle(&mut self, order: &mut [usize]) {
        for i in (1..order.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
    }
}

// Published throughput/step laws are only starting heuristics in the weighted
// categorical domain. They are not validated scaling laws for JXL histograms.
fn recipe(n: usize, k: usize, passes: u8) -> (usize, f64) {
    let (nf, kf, tf) = (n as f64, k as f64, f64::from(passes));
    let estimate = 0.035134 * kf.powf(0.134749) * nf.powf(0.876473) * tf.powf(0.813344);
    let batch = (2f64.powf(estimate.max(1.0).log2().round()) as usize)
        .max(8)
        .min((n / 2).max(1));
    let alpha = (0.8244
        * kf.powf(-0.4617)
        * (batch as f64).powf(0.9003)
        * nf.powf(-0.4498)
        * tf.powf(-0.1057))
    .clamp(0.0001, 1.0);
    (batch, alpha)
}

struct Center {
    mass: Vec<f64>,
    prices: Vec<f32>,
    total: f64,
    roots: Vec<f32>,
    projected: Option<Histogram>,
}

impl Center {
    fn from_histogram(histogram: &Histogram, width: usize, distance: Distance) -> Self {
        let mut mass = vec![0.0; width];
        for (value, &count) in mass.iter_mut().zip(&histogram.counts) {
            let probability = count as f64 / histogram.total as f64;
            *value = if distance == Distance::Hellinger {
                probability.sqrt() * 1e-10
            } else {
                probability * 1e-10
            };
        }
        let mut center = Self {
            mass,
            prices: vec![0.0; width],
            total: 1e-10,
            roots: if distance == Distance::Hellinger {
                vec![0.0; width]
            } else {
                Vec::new()
            },
            projected: None,
        };
        center.refresh(distance);
        center
    }

    fn refresh(&mut self, distance: Distance) {
        if distance == Distance::Hellinger {
            // The Hellinger barycenter is a normalized weighted sum of roots,
            // not the square root of an arithmetic probability centroid.
            let norm = self.mass.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for (root, &mass) in self.roots.iter_mut().zip(&self.mass) {
                    *root = (mass / norm) as f32;
                }
            }
            return;
        }
        self.total = self.mass.iter().sum();
        if self.total == 0.0 {
            // Keep the last distribution available until the donor revival.
            return;
        }
        if distance == Distance::MergeEntropy {
            return;
        }
        // Full-support smoothing belongs only to assignment scoring. The
        // final integer histograms below contain no invented observations.
        let smooth = 1e-12;
        let normalizer = 1.0 + smooth * self.mass.len() as f64;
        for (price, &mass) in self.prices.iter_mut().zip(&self.mass) {
            *price = -((mass / self.total + smooth) / normalizer).log2() as f32;
        }
    }
}

fn nearest(histogram: &Histogram, roots: &[f32], centers: &[Center], distance: Distance) -> usize {
    if distance != Distance::CrossEntropy {
        let mut best = 0;
        let mut cost = f32::INFINITY;
        for (j, center) in centers.iter().enumerate() {
            let current = if distance == Distance::Hellinger {
                1.0 - roots
                    .iter()
                    .zip(&center.roots)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            } else {
                super::distance(
                    histogram,
                    center.projected.as_ref().expect("projected EMA center"),
                    false,
                )
            };
            if current < cost {
                cost = current;
                best = j;
            }
        }
        return best;
    }
    let mut best = 0;
    let mut cost = f32::INFINITY;
    for (j, center) in centers.iter().enumerate() {
        let mut lanes = [0.0f32; 4];
        let chunks = histogram.counts.len() / 4;
        for i in 0..chunks {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                *sum += histogram.counts[i * 4 + lane] as f32 * center.prices[i * 4 + lane];
            }
        }
        let mut current = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for i in chunks * 4..histogram.counts.len() {
            current += histogram.counts[i] as f32 * center.prices[i];
        }
        if current < cost {
            cost = current;
            best = j;
        }
    }
    best
}

fn revive(centers: &mut [Center], tick: usize, distance: Distance) -> usize {
    let threshold = centers.iter().map(|c| c.total).sum::<f64>() / centers.len() as f64 * 0.1;
    let mut revived = 0;
    for dead in 0..centers.len() {
        if centers[dead].total >= threshold {
            continue;
        }
        let donor = (0..centers.len())
            .filter(|&i| i != dead)
            .max_by(|&a, &b| {
                centers[a]
                    .total
                    .total_cmp(&centers[b].total)
                    .then(b.cmp(&a))
            })
            .expect("at least two centers");
        if centers[donor].total <= threshold {
            continue;
        }
        for symbol in 0..centers[donor].mass.len() {
            let fraction = if distance == Distance::Hellinger {
                // Split root statistics and population by the same scalar.
                // Per-symbol perturbation would break the weighted-root bound.
                0.5
            } else if (symbol + tick) % 2 == 0 {
                0.501
            } else {
                0.499
            };
            let value = centers[donor].mass[symbol];
            centers[dead].mass[symbol] = value * fraction;
            centers[donor].mass[symbol] = value * (1.0 - fraction);
        }
        if distance == Distance::Hellinger {
            centers[donor].total *= 0.5;
            centers[dead].total = centers[donor].total;
        }
        centers[donor].refresh(distance);
        centers[dead].refresh(distance);
        revived += 1;
    }
    revived
}

fn project_centers(centers: &mut [Center], population: u64) -> Result<()> {
    // EMA masses describe a batch, not the whole input. Restore a common
    // population scale before comparing entropy-of-merging in bits. Even tiny
    // seed weights then represent meaningful centers rather than zero cost.
    let total = centers.iter().map(|c| c.total).sum::<f64>();
    let shares: Vec<_> = centers
        .iter()
        .map(|c| c.total / total * population as f64)
        .collect();
    let mut targets: Vec<_> = shares.iter().map(|n| n.floor() as u64).collect();
    let remainder = population - targets.iter().sum::<u64>();
    let mut order: Vec<_> = (0..centers.len()).collect();
    order.sort_by(|&a, &b| {
        shares[b]
            .fract()
            .total_cmp(&shares[a].fract())
            .then(a.cmp(&b))
    });
    for &i in order.iter().take(remainder as usize) {
        targets[i] += 1;
    }
    for (center, target) in centers.iter_mut().zip(targets) {
        if target == 0 {
            center.projected = Some(Histogram::new(&[])?);
            continue;
        }
        let mut counts: Vec<_> = center
            .mass
            .iter()
            .map(|&m| (m / center.total * target as f64).floor() as u64)
            .collect();
        // Preserve target population without inventing support. Assign floor
        // remainder to the largest existing bin; this is a scoring projection,
        // never the final serialized integer histogram.
        let peak = center
            .mass
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        counts[peak] += target - counts.iter().sum::<u64>();
        center.projected = Some(Histogram::new(&counts)?);
    }
    Ok(())
}

pub(super) fn propose(
    input: &[Histogram],
    limit: usize,
    options: Options,
    workers: usize,
) -> Result<Option<(Vec<Histogram>, Vec<usize>)>> {
    options.validate()?;
    let width = input.iter().map(|h| h.counts.len()).max().unwrap_or(0);
    let mut order: Vec<_> = input
        .iter()
        .enumerate()
        .filter_map(|(i, h)| (h.total > 0).then_some(i))
        .collect();
    // This is experiment eligibility, not a codec limit: all other inputs
    // retain ordinary Best clustering, including prefix/wide alphabets.
    if width > 256 || order.len() < 32 || limit < 2 {
        fallback();
        return Ok(None);
    }
    let _profile = crate::profile::scope("entropy.cluster.ema");
    let mut random = Random(0x243f6a8885a308d3);
    random.shuffle(&mut order);
    let seed_profile = crate::profile::scope("entropy.cluster.ema.seed");
    let mut centers: Vec<_> = match options.initialization {
        Initialization::Random => order
            .iter()
            .take(limit)
            .map(|&i| Center::from_histogram(&input[i], width, options.distance))
            .collect(),
        Initialization::Farthest => {
            let (seeds, _) = seed_histograms(input, Vec::new(), limit, false, workers)?;
            seeds
                .iter()
                .map(|h| Center::from_histogram(h, width, options.distance))
                .collect()
        }
    };
    drop(seed_profile);
    if centers.len() < 2 {
        fallback();
        return Ok(None);
    }
    let (batch, alpha0) = recipe(order.len(), centers.len(), options.passes);
    let steps = order.len().div_ceil(batch) * usize::from(options.passes);
    let mut symbols = vec![0; input.len()];
    let mut batch_mass = vec![vec![0u64; width]; centers.len()];
    let roots: Vec<Vec<f32>> = if options.distance == Distance::Hellinger {
        input
            .iter()
            .map(|h| {
                h.counts
                    .iter()
                    .map(|&n| {
                        if h.total == 0 {
                            0.0
                        } else {
                            (n as f64 / h.total as f64).sqrt() as f32
                        }
                    })
                    .collect()
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut root_mass = if roots.is_empty() {
        Vec::new()
    } else {
        vec![vec![0.0f64; width]; centers.len()]
    };
    let population = input.iter().map(|h| h.total).sum();
    if options.distance == Distance::MergeEntropy {
        project_centers(&mut centers, population)?;
    }
    let mut tick = 0;
    let mut revivals = 0;
    let mut parallel_batches = 0;
    #[cfg(test)]
    let probe = super::super::worker_test::current();
    let train_profile = crate::profile::scope("entropy.cluster.ema.train");
    for _ in 0..options.passes {
        random.shuffle(&mut order);
        for indices in order.chunks(batch) {
            let parallel = super::super::jobs::task_count(indices.len(), workers, 0) > 1
                && indices.len() >= 64
                && indices.len() * centers.len() * width >= 32768;
            let assignments: Vec<_> = if parallel {
                parallel_batches += 1;
                super::super::jobs::map(indices, workers, 0, |i| {
                    #[cfg(test)]
                    probe.record("cluster.ema.assign");
                    nearest(
                        &input[i],
                        roots.get(i).map_or(&[], Vec::as_slice),
                        &centers,
                        options.distance,
                    )
                })
            } else {
                indices
                    .iter()
                    .map(|&i| {
                        nearest(
                            &input[i],
                            roots.get(i).map_or(&[], Vec::as_slice),
                            &centers,
                            options.distance,
                        )
                    })
                    .collect()
            };
            for row in &mut batch_mass {
                row.fill(0);
            }
            for row in &mut root_mass {
                row.fill(0.0);
            }
            for (&i, &cluster) in indices.iter().zip(&assignments) {
                symbols[i] = cluster;
                for (sum, &count) in batch_mass[cluster].iter_mut().zip(&input[i].counts) {
                    *sum += count;
                }
                if options.distance == Distance::Hellinger {
                    for (sum, &root) in root_mass[cluster].iter_mut().zip(&roots[i]) {
                        *sum += f64::from(root) * input[i].total as f64;
                    }
                }
            }
            let phase = tick as f64 / (steps - 1).max(1) as f64;
            let alpha = alpha0 * (0.1 + 0.45 * (1.0 + (std::f64::consts::PI * phase).cos()));
            for (id, (center, counts)) in centers.iter_mut().zip(&batch_mass).enumerate() {
                if options.distance == Distance::Hellinger {
                    center.total =
                        (1.0 - alpha) * center.total + alpha * counts.iter().sum::<u64>() as f64;
                    for (mass, &root) in center.mass.iter_mut().zip(&root_mass[id]) {
                        *mass = (1.0 - alpha) * *mass + alpha * root;
                    }
                    center.refresh(options.distance);
                    continue;
                }
                for (mass, &count) in center.mass.iter_mut().zip(counts) {
                    *mass = (1.0 - alpha) * *mass + alpha * count as f64;
                }
                center.refresh(options.distance);
            }
            tick += 1;
            if tick % 5 == 0 && tick < steps {
                revivals += revive(&mut centers, tick, options.distance);
            }
            if options.distance == Distance::MergeEntropy {
                project_centers(&mut centers, population)?;
            }
        }
    }
    drop(train_profile);
    if options.final_assignment {
        let _profile = crate::profile::scope("entropy.cluster.ema.final_assign");
        let mut changed = 0;
        let mut population_changed = 0u64;
        let assignments = super::super::jobs::map(&order, workers, 0, |i| {
            nearest(
                &input[i],
                roots.get(i).map_or(&[], Vec::as_slice),
                &centers,
                options.distance,
            )
        });
        for (&i, id) in order.iter().zip(assignments) {
            if symbols[i] != id {
                changed += 1;
                population_changed += input[i].total;
            }
            symbols[i] = id;
        }
        crate::profile::note(
            "entropy.cluster.ema.final_assign",
            format_args!("changed_contexts={changed} changed_population={population_changed}"),
        );
    }
    // Last-pass assignments are authoritative; EMA state is never serialized.
    let mut output = (0..centers.len())
        .map(|_| Histogram::new(&[]))
        .collect::<Result<Vec<_>>>()?;
    for (h, &cluster) in input.iter().zip(&symbols) {
        output[cluster].add(h)?;
    }
    if options.polish_passes != 0 {
        refit::relocate(input, &mut output, &mut symbols, options.polish_passes)?;
    }
    // Remove unused centers before pair refinement, keeping empty inputs valid.
    let mut remap = vec![0; output.len()];
    let mut used = Vec::new();
    for (id, mut h) in output.into_iter().enumerate() {
        if h.total == 0 {
            continue;
        }
        h.entropy = shannon_entropy(&h.counts, h.total);
        remap[id] = used.len();
        used.push(h);
    }
    for id in &mut symbols {
        *id = remap[*id];
    }
    for (counter, value) in WORK.iter().zip([
        1,
        0,
        order.len() as u64,
        centers.len() as u64,
        used.len() as u64,
        parallel_batches as u64,
    ]) {
        counter.fetch_add(value, Ordering::Relaxed);
    }
    crate::profile::note(
        "entropy.cluster.ema",
        format_args!(
            "inputs={} nonempty={} centers={} used={} bins={width} passes={} batch={batch} alpha={alpha0:.6} revivals={revivals} parallel_batches={parallel_batches} init={:?} distance={:?} polish={} final_assign={}",
            input.len(),
            order.len(),
            centers.len(),
            used.len(),
            options.passes,
            options.initialization,
            options.distance,
            options.polish_passes,
            options.final_assignment,
        ),
    );
    Ok(Some((used, symbols)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::cluster::{
        ClusterStrategy, cluster_with_options, cluster_with_preexisting,
    };

    fn fixture() -> Vec<Vec<u64>> {
        (0..200)
            .map(|i| {
                if i % 17 == 0 {
                    return vec![];
                }
                (0..64)
                    .map(|s| {
                        if s % 9 == i % 9 {
                            (s * 19 + i * 7 + 1) as u64
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn check_counts(raw: &[Vec<u64>], centers: &[Vec<u64>], map: &[u8]) {
        assert_eq!(raw.len(), map.len());
        assert!(centers.len() <= 256);
        let mut rebuilt = vec![vec![0u64; 256]; centers.len()];
        for (h, &id) in raw.iter().zip(map) {
            for (sum, &count) in rebuilt[id as usize].iter_mut().zip(h) {
                *sum += count;
            }
        }
        for (expected, actual) in rebuilt.iter().zip(centers) {
            assert_eq!(&expected[..actual.len()], actual);
            assert!(expected[actual.len()..].iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn ema_rebuilds_exact_counts_and_is_worker_independent() {
        let counts = fixture().into_iter().cycle().take(2048).collect::<Vec<_>>();
        let input = counts
            .iter()
            .map(|h| Histogram::new(h).unwrap())
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for initialization in [Initialization::Random, Initialization::Farthest] {
            for passes in [1, 3] {
                let options = Options {
                    passes,
                    initialization,
                    ..Options::default()
                };
                let a = propose(&input, 16, options, 1).unwrap().unwrap();
                let (b, events) = pool.install(|| {
                    super::super::super::worker_test::observe(|| {
                        propose(&input, 16, options, 4).unwrap().unwrap()
                    })
                });
                if passes == 3 && initialization == Initialization::Random {
                    assert!(!events.is_empty(), "must exercise parallel assignment");
                    assert!(
                        events
                            .iter()
                            .all(|e| e.stage == "cluster.ema.assign" && !e.serial)
                    );
                    let (_, serial_events) = pool.install(|| {
                        super::super::super::worker_test::observe(|| {
                            crate::execution::serial(|| propose(&input, 16, options, 4).unwrap())
                        })
                    });
                    assert!(serial_events.is_empty());
                }
                assert_eq!(a.1, b.1);
                assert_eq!(
                    a.0.iter().map(|h| &h.counts).collect::<Vec<_>>(),
                    b.0.iter().map(|h| &h.counts).collect::<Vec<_>>()
                );
                let (centers, map) =
                    cluster_with_options(&counts, 16, ClusterStrategy::Emeans(options), false)
                        .unwrap();
                check_counts(&counts, &centers, &map);
            }
        }
    }

    #[test]
    fn empty_singleton_wide_and_existing_tables_fall_back() {
        let ema = ClusterStrategy::Emeans(Options::default());
        for counts in [
            vec![],
            vec![vec![]; 100],
            vec![vec![10]; 100],
            vec![vec![0, 1]; 5],
            vec![vec![1; 257]; 40],
        ] {
            let expected = cluster_with_options(&counts, 16, ClusterStrategy::Best, true).unwrap();
            assert_eq!(
                cluster_with_options(&counts, 16, ema, true).unwrap(),
                expected
            );
            let (centers, map) = cluster_with_options(&counts, 16, ema, false).unwrap();
            if !counts.is_empty() && counts.iter().all(|h| h.len() <= 256) {
                check_counts(&counts, &centers, &map);
            }
        }
        let counts = fixture();
        let old = vec![vec![16; 256]];
        for limit in [0, 8] {
            assert_eq!(
                cluster_with_preexisting(&counts, &old, limit, ema, false).unwrap(),
                cluster_with_preexisting(&counts, &old, limit, ClusterStrategy::Best, false)
                    .unwrap()
            );
        }
    }

    #[test]
    fn invalid_options_and_weight_limits_still_fail() {
        let bad = ClusterStrategy::Emeans(Options {
            passes: 0,
            ..Options::default()
        });
        assert!(cluster_with_options(&[], 16, bad, false).is_err());
        let ema = ClusterStrategy::Emeans(Options::default());
        assert!(cluster_with_options(&vec![vec![u64::MAX]; 32], 16, ema, false).is_err());
        let mut counts = fixture();
        counts[0] = vec![i32::MAX as u64];
        assert!(cluster_with_options(&counts, 16, ema, false).is_err());
    }

    #[test]
    fn extreme_sparse_populations_and_revival_remain_finite() {
        let counts = (0..96)
            .map(|i| {
                let mut h = vec![0; 256];
                h[i % 256] = if i == 0 { 1_000_000_000 } else { 1 };
                h
            })
            .collect::<Vec<_>>();
        let (centers, map) = cluster_with_options(
            &counts,
            16,
            ClusterStrategy::Emeans(Options::default()),
            false,
        )
        .unwrap();
        check_counts(&counts, &centers, &map);
    }

    #[test]
    fn alternative_distances_and_polish_keep_exact_counts() {
        let counts = fixture().into_iter().cycle().take(2048).collect::<Vec<_>>();
        let input = counts
            .iter()
            .map(|h| Histogram::new(h).unwrap())
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for distance in [Distance::Hellinger, Distance::MergeEntropy] {
            for polish_passes in [0, 1] {
                let options = Options {
                    distance,
                    polish_passes,
                    final_assignment: polish_passes != 0,
                    ..Options::default()
                };
                let serial = propose(&input, 16, options, 1).unwrap().unwrap();
                let parallel = pool.install(|| propose(&input, 16, options, 4).unwrap().unwrap());
                assert_eq!(serial.1, parallel.1);
                assert_eq!(
                    serial.0.iter().map(|h| &h.counts).collect::<Vec<_>>(),
                    parallel.0.iter().map(|h| &h.counts).collect::<Vec<_>>()
                );
                check_counts(
                    &counts,
                    &serial
                        .0
                        .iter()
                        .map(|h| h.counts.clone())
                        .collect::<Vec<_>>(),
                    &serial.1.iter().map(|&i| i as u8).collect::<Vec<_>>(),
                );
            }
        }
    }

    #[test]
    fn hellinger_barycenter_and_merge_projection_have_expected_scale() {
        let a = Histogram::new(&[100, 0]).unwrap();
        let b = Histogram::new(&[0, 100]).unwrap();
        let mut center = Center::from_histogram(&a, 2, Distance::Hellinger);
        center.mass = vec![9.0, 1.0];
        center.total = 10.0;
        center.refresh(Distance::Hellinger);
        assert!((center.roots[0].powi(2) - 81.0 / 82.0).abs() < 1e-6);
        assert!((center.roots.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
        let mut centers = vec![
            Center::from_histogram(&a, 2, Distance::MergeEntropy),
            Center::from_histogram(&b, 2, Distance::MergeEntropy),
        ];
        project_centers(&mut centers, 200).unwrap();
        assert_eq!(centers[0].projected.as_ref().unwrap().total, 100);
        assert_eq!(nearest(&a, &[], &centers, Distance::MergeEntropy), 0);
        assert_eq!(nearest(&b, &[], &centers, Distance::MergeEntropy), 1);
        project_centers(&mut centers, i32::MAX as u64).unwrap();
        assert_eq!(
            centers
                .iter()
                .map(|c| c.projected.as_ref().unwrap().total)
                .sum::<u64>(),
            i32::MAX as u64
        );
        assert_eq!(nearest(&a, &[], &centers, Distance::MergeEntropy), 0);
    }

    #[test]
    fn hellinger_differs_from_smoothed_kl_for_missing_support() {
        let a = Histogram::new(&[100, 0]).unwrap();
        let b = Histogram::new(&[90, 10]).unwrap();
        let x = Histogram::new(&[99, 1]).unwrap();
        let kl = vec![
            Center::from_histogram(&a, 2, Distance::CrossEntropy),
            Center::from_histogram(&b, 2, Distance::CrossEntropy),
        ];
        let hell = vec![
            Center::from_histogram(&a, 2, Distance::Hellinger),
            Center::from_histogram(&b, 2, Distance::Hellinger),
        ];
        assert_eq!(nearest(&x, &[], &kl, Distance::CrossEntropy), 1);
        assert_eq!(
            nearest(&x, &[0.99f32.sqrt(), 0.1], &hell, Distance::Hellinger),
            0
        );
    }

    #[test]
    fn hellinger_revival_preserves_population_and_root_statistics() {
        let h = Histogram::new(&[100, 0]).unwrap();
        let mut centers = vec![
            Center::from_histogram(&h, 2, Distance::Hellinger),
            Center::from_histogram(&h, 2, Distance::Hellinger),
        ];
        centers[0].mass = vec![100.0, 0.0];
        centers[0].total = 100.0;
        centers[1].mass.fill(0.0);
        centers[1].total = 0.0;
        assert_eq!(revive(&mut centers, 0, Distance::Hellinger), 1);
        for c in &centers {
            assert_eq!(c.total, 50.0);
            assert_eq!(c.mass, vec![50.0, 0.0]);
            assert!(c.mass.iter().map(|m| m * m).sum::<f64>().sqrt() <= c.total);
        }
    }
}
