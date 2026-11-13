//! Encoder-only reference-instance proposals. No new Patch/decoder syntax.
//! Scores rank representatives, not encoded files. Actual signed residuals
//! and standard frame coding are rebuilt by the existing Add pipeline.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{DetectionSamples, DetectionView, PendingPatch};
use crate::Result;
use crate::entropy::HybridUintConfig;
use crate::modular::tree::pack_signed;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RepresentativeMode {
    #[default]
    First,
    Medoid,
    MinEntropy,
}

#[derive(Clone, Debug)]
pub struct RepresentativeOptions {
    pub mode: RepresentativeMode,
    /// Sample budgets, not input or decoder limits. Zero means exhaustive.
    pub candidates: usize,
    pub occurrences: usize,
    pub pixels: usize,
    pub receipt: Option<Arc<RepresentativeReceipt>>,
}

impl Default for RepresentativeOptions {
    fn default() -> Self {
        Self {
            mode: RepresentativeMode::First,
            candidates: 8,
            occurrences: 16,
            pixels: 128,
            receipt: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RepresentativeStats {
    pub retained_groups: usize,
    pub occurrences: usize,
    pub searched_groups: usize,
    pub variable_sample_groups: usize,
    pub changed_groups: usize,
    pub candidates: usize,
    pub sampled_occurrences: usize,
    pub sampled_pixels: usize,
    pub selection_seconds: f64,
    pub baseline_proxy: f64,
    pub selected_proxy: f64,
    /// Successfully assembled/replay-checked plans. It proves frame emission
    /// only when the enclosing single encode also completes successfully.
    pub emitted_plans: usize,
    pub atlas_pixels: usize,
    pub dictionary_positions: usize,
    pub float_fallback_groups: usize,
}

#[derive(Debug, Default)]
pub struct RepresentativeReceipt(Mutex<RepresentativeStats>);

impl RepresentativeReceipt {
    pub fn snapshot(&self) -> RepresentativeStats {
        self.0
            .lock()
            .expect("patch representative receipt poisoned")
            .clone()
    }

    pub(super) fn record(&self, work: &RepresentativeStats) {
        let mut total = self
            .0
            .lock()
            .expect("patch representative receipt poisoned");
        total.retained_groups += work.retained_groups;
        total.occurrences += work.occurrences;
        total.searched_groups += work.searched_groups;
        total.variable_sample_groups += work.variable_sample_groups;
        total.changed_groups += work.changed_groups;
        total.candidates += work.candidates;
        total.sampled_occurrences += work.sampled_occurrences;
        total.sampled_pixels += work.sampled_pixels;
        total.selection_seconds += work.selection_seconds;
        total.baseline_proxy += work.baseline_proxy;
        total.selected_proxy += work.selected_proxy;
        total.float_fallback_groups += work.float_fallback_groups;
    }

    pub(super) fn emitted(&self, atlas_pixels: usize, positions: usize) {
        let mut total = self
            .0
            .lock()
            .expect("patch representative receipt poisoned");
        total.emitted_plans += 1;
        total.atlas_pixels += atlas_pixels;
        total.dictionary_positions += positions;
    }
}

fn indices(length: usize, budget: usize) -> Vec<usize> {
    let count = if budget == 0 {
        length
    } else {
        length.min(budget)
    };
    if count <= 1 {
        return (0..count).collect();
    }
    (0..count)
        .map(|i| ((i as u128 * (length - 1) as u128) / (count - 1) as u128) as usize)
        .collect()
}

// Center, west, north and northwest. Atlas candidates are background-relative
// with zero outside their rectangle. Entropy observations retain original
// backgrounds/neighbors; Medoid compares background-relative instances.
type Stencil = [i32; 4];

fn sample(
    image: DetectionView<'_>,
    position: (u32, u32),
    origin: usize,
    width: usize,
    points: &[usize],
    atlas: bool,
) -> Vec<Stencil> {
    let DetectionSamples::Integer(view, _) = image.samples else {
        unreachable!()
    };
    let mut values = Vec::with_capacity(points.len() * image.color_channels);
    for c in 0..image.color_channels {
        let plane = &view.planes[c];
        let base = if atlas { plane[origin] } else { 0 };
        for &point in points {
            let (x, y) = (point % width, point / width);
            let index = (position.1 as usize + y) * image.width + position.0 as usize + x;
            let has_left = if atlas {
                x > 0
            } else {
                position.0 as usize + x > 0
            };
            let has_top = if atlas {
                y > 0
            } else {
                position.1 as usize + y > 0
            };
            values.push([
                plane[index] - base,
                if has_left { plane[index - 1] - base } else { 0 },
                if has_top {
                    plane[index - image.width] - base
                } else {
                    0
                },
                if has_left && has_top {
                    plane[index - image.width - 1] - base
                } else {
                    0
                },
            ]);
        }
    }
    values
}

fn medoid(reference: &[Stencil], observations: &[Vec<Stencil>]) -> f64 {
    observations
        .iter()
        .flat_map(|row| row.iter().zip(reference))
        .map(|(value, base)| u64::from(value[0].abs_diff(base[0])))
        .sum::<u64>() as f64
}

struct Costs {
    counts: [[u64; 256]; 4],
    extra: [u64; 4],
    count: u64,
}

impl Costs {
    fn new() -> Self {
        Self {
            counts: [[0; 256]; 4],
            extra: [0; 4],
            count: 0,
        }
    }

    fn add(&mut self, stencil: Stencil, config: HybridUintConfig) {
        let [v, w, n, nw] = stencil;
        let gradient = (w + n - nw).clamp(w.min(n), w.max(n));
        for (p, prediction) in [0, w, n, gradient].into_iter().enumerate() {
            let symbol = config.encode(pack_signed(v - prediction));
            self.counts[p][symbol.token as usize] += 1;
            self.extra[p] += u64::from(symbol.nbits);
        }
        self.count += 1;
    }

    fn bits(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let n = self.count as f64;
        self.counts
            .iter()
            .zip(self.extra)
            .map(|(hist, extra)| {
                extra as f64
                    + hist
                        .iter()
                        .filter(|&&v| v != 0)
                        .map(|&v| v as f64 * (n / v as f64).log2())
                        .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
    }
}

fn entropy(
    reference: &[Stencil],
    observations: &[Vec<Stencil>],
    channels: usize,
    occurrences: usize,
    area: usize,
) -> f64 {
    // Isolated-occurrence estimate: other overlapping groups, full-frame MA,
    // atlas packing neighbors, LZ77 and entropy headers are intentionally absent.
    let config = HybridUintConfig::new(4, 1, 2).expect("standard HybridUint");
    let points = reference.len() / channels;
    let mut bits = 0.0;
    for c in 0..channels {
        let range = c * points..(c + 1) * points;
        let mut atlas = Costs::new();
        let mut residual = Costs::new();
        for &value in &reference[range.clone()] {
            atlas.add(value, config);
        }
        for row in observations {
            for (value, base) in row[range.clone()].iter().zip(&reference[range.clone()]) {
                residual.add(std::array::from_fn(|i| value[i] - base[i]), config);
            }
        }
        bits += atlas.bits() + residual.bits() * occurrences as f64 / observations.len() as f64;
    }
    bits * area as f64 / points as f64
}

pub(super) fn choose(
    image: DetectionView<'_>,
    patch: &mut PendingPatch,
    width: usize,
    height: usize,
    options: &RepresentativeOptions,
    stats: &mut RepresentativeStats,
) -> Result<()> {
    stats.retained_groups += 1;
    stats.occurrences += patch.positions.len();
    if options.mode == RepresentativeMode::First {
        return Ok(());
    }
    if matches!(image.samples, DetectionSamples::Float(_)) {
        stats.float_fallback_groups += 1;
        return Ok(());
    }
    let start = Instant::now();
    let origins = patch
        .origins
        .as_ref()
        .expect("non-first integer detector retains origins");
    let mut order: Vec<_> = (0..patch.positions.len()).collect();
    order.sort_unstable_by_key(|&i| patch.positions[i]);
    let candidates = indices(order.len(), options.candidates);
    let selected = indices(order.len(), options.occurrences);
    let points = indices(width * height, options.pixels);
    let observations: Vec<_> = selected
        .iter()
        .map(|&i| {
            let i = order[i];
            sample(
                image,
                patch.positions[i],
                origins[i],
                width,
                &points,
                options.mode == RepresentativeMode::Medoid,
            )
        })
        .collect();
    stats.searched_groups += 1;
    stats.sampled_occurrences += observations.len();
    stats.sampled_pixels += points.len();
    let first = sample(
        image,
        patch.representative,
        patch.reference,
        width,
        &points,
        true,
    );
    let score = |reference: &[Stencil]| match options.mode {
        RepresentativeMode::Medoid => medoid(reference, &observations),
        RepresentativeMode::MinEntropy => entropy(
            reference,
            &observations,
            image.color_channels,
            patch.positions.len(),
            width * height,
        ),
        RepresentativeMode::First => unreachable!(),
    };
    let baseline = score(&first);
    let mut best = baseline;
    let mut best_index = order[0];
    let mut variable = false;
    stats.candidates += 1;
    for &i in candidates.iter().skip(1) {
        let index = order[i];
        let candidate = sample(
            image,
            patch.positions[index],
            origins[index],
            width,
            &points,
            true,
        );
        stats.candidates += 1;
        if candidate == first {
            continue;
        }
        variable = true;
        let cost = score(&candidate);
        if cost < best {
            best = cost;
            best_index = index;
        }
    }
    stats.variable_sample_groups += usize::from(variable);
    stats.baseline_proxy += baseline;
    stats.selected_proxy += best;
    if patch.representative != patch.positions[best_index] {
        stats.changed_groups += 1;
        patch.representative = patch.positions[best_index];
        patch.reference = origins[best_index];
    }
    stats.selection_seconds += start.elapsed().as_secs_f64();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_budgets_include_first_and_last_without_duplicates() {
        for length in 1..100 {
            for budget in 0..120 {
                let values = indices(length, budget);
                assert_eq!(values[0], 0);
                assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
                assert_eq!(
                    values.len(),
                    if budget == 0 {
                        length
                    } else {
                        budget.min(length)
                    }
                );
                if values.len() > 1 {
                    assert_eq!(values.last(), Some(&(length - 1)));
                }
            }
        }
    }

    #[test]
    fn medoid_prefers_central_instance_and_includes_zero_residual() {
        let values: Vec<_> = [100, 101, 101, 102, 110]
            .into_iter()
            .map(|v| vec![[v, 0, 0, 0]])
            .collect();
        assert!(medoid(&values[1], &values) < medoid(&values[4], &values));
        assert_eq!(medoid(&values[0], &[values[0].clone()]), 0.0);
    }

    #[test]
    fn entropy_observations_keep_background_and_external_neighbors() {
        let planes = vec![vec![
            100, 101, 102, 103, 104, 180, 181, 107, 108, 182, 183, 111,
        ]];
        let image = DetectionView::integer(super::super::ImageView {
            width: 4,
            height: 3,
            bit_depth: 8,
            color_channels: 1,
            planes: &planes,
        });
        let raw = sample(image, (1, 1), 0, 2, &[0, 3], false);
        let atlas = sample(image, (1, 1), 0, 2, &[0, 3], true);
        assert_eq!(raw, [[180, 104, 101, 100], [183, 182, 181, 180]]);
        assert_eq!(atlas, [[80, 0, 0, 0], [83, 82, 81, 80]]);
        assert_eq!(
            std::array::from_fn::<_, 4, _>(|i| raw[0][i] - atlas[0][i]),
            [100, 104, 101, 100]
        );
        assert_eq!(
            std::array::from_fn::<_, 4, _>(|i| raw[1][i] - atlas[1][i]),
            [100; 4]
        );
    }

    #[test]
    fn entropy_proxy_is_finite_for_signed_16bit_differences() {
        let values = vec![
            vec![[65535, -65535, 0, 65535]],
            vec![[-65535, 0, 65535, -65535]],
        ];
        for value in &values {
            let cost = entropy(value, &values, 1, 100, 1);
            assert!(cost.is_finite() && cost >= 0.0);
        }
    }

    #[test]
    fn alternative_representatives_keep_groups_and_reconstruct_hidden_rgb() {
        use crate::tools::patches::{
            ReferenceImage, SourcePatchConfig, compose_add, discover_source,
        };
        for depth in [8, 16] {
            for colors in [1, 3] {
                let scale = if depth == 8 { 1 } else { 257 };
                let mut image = ReferenceImage {
                    width: 129,
                    height: 83,
                    bit_depth: depth,
                    color_channels: colors,
                    planes: vec![vec![0; 129 * 83]; colors + 1],
                };
                // Scan order differs from lexicographic reference order.
                for (x0, y0, value) in [(73, 17, 82), (17, 37, 87), (49, 57, 83), (97, 37, 82)] {
                    for plane in &mut image.planes[..colors] {
                        for y in y0..y0 + 5 {
                            plane[y * image.width + x0..y * image.width + x0 + 5]
                                .fill(value * scale);
                        }
                    }
                }
                let baseline = discover_source(image.view(), &SourcePatchConfig::default())
                    .unwrap()
                    .unwrap();
                for mode in [
                    RepresentativeMode::First,
                    RepresentativeMode::Medoid,
                    RepresentativeMode::MinEntropy,
                ] {
                    let receipt = Arc::new(RepresentativeReceipt::default());
                    let mut config = SourcePatchConfig::default();
                    config.representative = RepresentativeOptions {
                        mode,
                        candidates: 0,
                        occurrences: 0,
                        pixels: 0,
                        receipt: Some(receipt.clone()),
                    };
                    let plan = discover_source(image.view(), &config).unwrap().unwrap();
                    assert_eq!(plan.dictionary, baseline.dictionary);
                    assert_eq!(
                        compose_add(&plan.residual, &plan.reference, &plan.dictionary).unwrap(),
                        image.planes
                    );
                    assert_eq!(plan.residual.planes[colors], image.planes[colors]);
                    let work = receipt.snapshot();
                    assert_eq!(work.emitted_plans, 1);
                    assert_eq!(work.retained_groups, 1);
                    assert!(work.selected_proxy <= work.baseline_proxy);
                    if mode == RepresentativeMode::First {
                        assert_eq!(plan.reference, baseline.reference);
                        assert_eq!(work.searched_groups, 0);
                    } else if mode == RepresentativeMode::Medoid {
                        assert_eq!(work.changed_groups, 1);
                        assert_ne!(plan.reference, baseline.reference);
                    }
                }
            }
        }
    }
}
