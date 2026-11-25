//! Encoder-side same-size, zero-offset Source Add experiment.
//! Fingerprints provide atomic candidate groups. A binary mask shortlists
//! pairs and a signed per-channel rate proxy scores every covered sample.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    DetectionSamples, DetectionView, ImageView, PatchInfo, PatchKey, PatchOptions, PendingPatch,
    representative_samples,
};
use crate::{entropy::HybridUintConfig, modular::tree::pack_signed};

pub(super) const MAX_UNMATCHED: usize = 512;
const MAX_TEMPLATES_PER_SIZE: usize = 16;
const MAX_RECTANGLES: usize = 4096;
const MAX_ATOM_OCCURRENCES: usize = 8;
const MAX_TEMPLATE_SAMPLES: usize = 2_000_000;
const MAX_SCORED_SAMPLES: usize = 4_000_000;
const POSITION_BITS: i64 = 12;
const REFERENCE_BITS: i64 = 40;
const NEW_ATLAS_BITS: i64 = 64;

type Size = (usize, usize);

#[derive(Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl Rect {
    fn new((x, y): (u32, u32), (width, height): Size) -> Self {
        Self {
            x: x as usize,
            y: y as usize,
            width,
            height,
        }
    }

    fn overlaps(self, other: Self) -> bool {
        self.x < other.x + other.width
            && other.x < self.x + self.width
            && self.y < other.y + other.height
            && other.y < self.y + self.height
    }
}

fn disjoint(rects: &[Rect], occupied: &[Rect]) -> bool {
    rects.iter().enumerate().all(|(i, &rect)| {
        !occupied
            .iter()
            .chain(&rects[..i])
            .any(|&other| rect.overlaps(other))
    })
}

struct Template {
    samples: Vec<i32>,
    shape: Vec<u64>,
}

impl Template {
    fn new(samples: Vec<i32>, size: Size, maximum: i32) -> Self {
        let area = size.0 * size.1;
        let mut shape = vec![0u64; area.div_ceil(64)];
        for channel in samples.chunks_exact(area) {
            for (i, &value) in channel.iter().enumerate() {
                if value.abs() * 32 > maximum {
                    shape[i / 64] |= 1 << (i % 64);
                }
            }
        }
        Self { samples, shape }
    }

    fn similar(&self, other: &Self, maximum: i32) -> bool {
        let (mut intersection, mut union) = (0, 0);
        for (&a, &b) in self.shape.iter().zip(&other.shape) {
            intersection += (a & b).count_ones();
            union += (a | b).count_ones();
        }
        union > 0
            && intersection * 4 >= union * 3
            && self
                .samples
                .iter()
                .zip(&other.samples)
                .step_by(self.samples.len().div_ceil(32))
                .all(|(&a, &b)| a.abs_diff(b) * 8 <= maximum as u32)
    }
}

struct Atom {
    positions: Vec<(u32, u32)>,
    template: Template,
    rects: Vec<Rect>,
}

struct Rate {
    model: HybridUintConfig,
    remaining: usize,
}

impl Rate {
    fn new() -> Self {
        Self {
            model: HybridUintConfig::new(4, 1, 2).expect("standard HybridUint"),
            remaining: MAX_SCORED_SAMPLES,
        }
    }

    // Approximate token cost with a zero-favoring prior plus exact hybrid
    // extra bits. Independent local predictors are optimistic proxies for MA.
    fn symbol_bits(&self, value: i32) -> i64 {
        let symbol = self.model.encode(pack_signed(value));
        i64::from(u32::BITS - (symbol.token as u32).leading_zeros()) + i64::from(symbol.nbits)
    }

    fn add(&self, costs: &mut [i64; 4], [v, w, n, nw]: [i32; 4]) {
        let gradient = (w + n - nw).clamp(w.min(n), w.max(n));
        for (cost, prediction) in costs.iter_mut().zip([0, w, n, gradient]) {
            *cost += self.symbol_bits(v - prediction);
        }
    }

    fn atlas_bits(&self, template: &Template, (width, height): Size) -> i64 {
        let area = width * height;
        template
            .samples
            .chunks_exact(area)
            .map(|plane| {
                let mut costs = [0; 4];
                for y in 0..height {
                    for x in 0..width {
                        let i = y * width + x;
                        self.add(
                            &mut costs,
                            [
                                plane[i],
                                if x > 0 { plane[i - 1] } else { 0 },
                                if y > 0 { plane[i - width] } else { 0 },
                                if x > 0 && y > 0 {
                                    plane[i - width - 1]
                                } else {
                                    0
                                },
                            ],
                        );
                    }
                }
                costs.into_iter().min().unwrap()
            })
            .sum()
    }

    fn savings(&mut self, view: ImageView<'_>, atom: &Atom, template: &Template) -> Option<i64> {
        self.remaining = self
            .remaining
            .checked_sub(template.samples.len() * atom.rects.len())?;
        let mut gain = 0;
        for rect in &atom.rects {
            gain += self.rectangle_savings(view, *rect, template) - POSITION_BITS;
        }
        Some(gain)
    }

    fn rectangle_savings(&self, view: ImageView<'_>, rect: Rect, template: &Template) -> i64 {
        let area = rect.width * rect.height;
        let mut savings = 0;
        for (image, patch) in view.planes.iter().zip(template.samples.chunks_exact(area)) {
            let (mut raw, mut residual) = ([0; 4], [0; 4]);
            // Include one bottom/right pixel for predictors affected by the
            // subtraction outside the destination rectangle. With n <= 16,
            // original - delta lies in [-65535, 131070]; i32 also holds
            // every resulting predictor difference.
            let sample = |x: usize, y: usize| {
                let original = image[y * view.width + x];
                let patch_x = x.wrapping_sub(rect.x);
                let patch_y = y.wrapping_sub(rect.y);
                let delta = if patch_x < rect.width && patch_y < rect.height {
                    patch[patch_y * rect.width + patch_x]
                } else {
                    0
                };
                (original, original - delta)
            };
            let right = (rect.x + rect.width + 1).min(view.width);
            let bottom = (rect.y + rect.height + 1).min(view.height);
            for y in rect.y..bottom {
                for x in rect.x..right {
                    let (v, rv) = sample(x, y);
                    let (w, rw) = if x > 0 { sample(x - 1, y) } else { (0, 0) };
                    let (n, rn) = if y > 0 { sample(x, y - 1) } else { (0, 0) };
                    let (nw, rnw) = if x > 0 && y > 0 {
                        sample(x - 1, y - 1)
                    } else {
                        (0, 0)
                    };
                    self.add(&mut raw, [v, w, n, nw]);
                    self.add(&mut residual, [rv, rw, rn, rnw]);
                }
            }
            savings += raw.into_iter().min().unwrap() - residual.into_iter().min().unwrap();
        }
        savings
    }
}

pub(super) fn propose(
    image: DetectionView<'_>,
    baseline: &[PatchInfo],
    unmatched: Vec<(Size, PendingPatch)>,
    limits: &PatchOptions,
) -> Option<Vec<PatchInfo>> {
    let DetectionSamples::Integer(view, _) = image.samples else {
        return None;
    };
    if unmatched.is_empty()
        || baseline
            .iter()
            .map(|group| group.positions.len())
            .sum::<usize>()
            > MAX_RECTANGLES
    {
        return None;
    }
    let maximum = (1i32 << view.bit_depth) - 1;
    let mut occupied = Vec::new();
    let mut baseline_templates = BTreeMap::<Size, Vec<(usize, Template)>>::new();
    let mut remaining_baseline_samples = MAX_TEMPLATE_SAMPLES;
    let sizes: BTreeSet<_> = unmatched.iter().map(|(size, _)| *size).collect();
    for (index, group) in baseline.iter().enumerate() {
        let size = (group.key.width, group.key.height);
        occupied.extend(
            group
                .positions
                .iter()
                .map(|&position| Rect::new(position, size)),
        );
        if !sizes.contains(&size) || group.key.samples.len() > remaining_baseline_samples {
            continue;
        }
        let templates = baseline_templates.entry(size).or_default();
        if templates.len() < MAX_TEMPLATES_PER_SIZE {
            remaining_baseline_samples -= group.key.samples.len();
            templates.push((
                index,
                Template::new(group.key.samples.clone(), size, maximum),
            ));
        }
    }
    let mut atoms = BTreeMap::<Size, Vec<Atom>>::new();
    let mut remaining_atom_samples = MAX_TEMPLATE_SAMPLES;
    for (size, pending) in unmatched {
        let samples = size.0 * size.1 * image.color_channels;
        if pending.positions.len() > MAX_ATOM_OCCURRENCES || samples > remaining_atom_samples {
            continue;
        }
        let rects: Vec<_> = pending
            .positions
            .iter()
            .map(|&position| Rect::new(position, size))
            .collect();
        if !disjoint(&rects, &occupied) {
            continue;
        }
        remaining_atom_samples -= samples;
        let samples = representative_samples(image, &pending, size.0, size.1);
        atoms.entry(size).or_default().push(Atom {
            positions: pending.positions,
            template: Template::new(samples, size, maximum),
            rects,
        });
    }

    let mut rate = Rate::new();
    let mut attachments = BTreeMap::<usize, Vec<(u32, u32)>>::new();
    let mut new_groups = Vec::new();
    let mut atlas_area: usize = baseline.iter().map(|g| g.key.width * g.key.height).sum();
    for (size, bucket) in atoms {
        let mut unused = Vec::new();
        for atom in bucket {
            if occupied.len() + atom.rects.len() > MAX_RECTANGLES
                || !disjoint(&atom.rects, &occupied)
            {
                continue;
            }
            let mut best = None;
            let mut best_gain = 0;
            for (index, template) in baseline_templates.get(&size).into_iter().flatten() {
                if template.similar(&atom.template, maximum)
                    && let Some(gain) = rate.savings(view, &atom, template)
                    && gain > best_gain
                {
                    best = Some(*index);
                    best_gain = gain;
                }
            }
            if let Some(index) = best {
                occupied.extend(atom.rects);
                attachments.entry(index).or_default().extend(atom.positions);
            } else {
                unused.push(Some(atom));
            }
        }
        let area = size.0 * size.1;
        for i in 0..unused.len() {
            if atlas_area + area > limits.max_reference_pixels
                || baseline.len() + new_groups.len() >= limits.max_unique_patches
            {
                break;
            }
            let Some(mut seed) = unused[i].take() else {
                continue;
            };
            let mut rects = seed.rects.clone();
            if occupied.len() + rects.len() > MAX_RECTANGLES || !disjoint(&rects, &occupied) {
                continue;
            }
            let Some(seed_gain) = rate.savings(view, &seed, &seed.template) else {
                continue;
            };
            let mut gain = seed_gain - rate.atlas_bits(&seed.template, size) - REFERENCE_BITS;
            if baseline.is_empty() && new_groups.is_empty() {
                gain -= NEW_ATLAS_BITS;
            }
            let mut members = Vec::new();
            let mut count = seed.positions.len();
            for (offset, other) in unused[i + 1..]
                .iter()
                .take(MAX_TEMPLATES_PER_SIZE)
                .enumerate()
            {
                let Some(other) = other else {
                    continue;
                };
                if occupied.len() + rects.len() + other.rects.len() > MAX_RECTANGLES
                    || !seed.template.similar(&other.template, maximum)
                    || !disjoint(&other.rects, &occupied)
                    || !disjoint(&other.rects, &rects)
                {
                    continue;
                }
                if let Some(savings) = rate.savings(view, other, &seed.template)
                    && savings > 0
                {
                    count += other.positions.len();
                    gain += savings;
                    rects.extend(&other.rects);
                    members.push(i + 1 + offset);
                }
            }
            if count < limits.min_occurrences || gain <= 0 {
                continue;
            }
            for member in members {
                if let Some(atom) = unused[member].take() {
                    seed.positions.extend(atom.positions);
                }
            }
            seed.positions.sort_unstable();
            occupied.extend(rects);
            atlas_area += area;
            new_groups.push(PatchInfo {
                key: PatchKey {
                    width: size.0,
                    height: size.1,
                    samples: seed.template.samples,
                },
                positions: seed.positions,
            });
        }
    }

    if attachments.is_empty() && new_groups.is_empty() {
        return None;
    }
    let mut candidate: Vec<_> = baseline
        .iter()
        .enumerate()
        .map(|(i, group)| {
            let mut positions = group.positions.clone();
            if let Some(extra) = attachments.remove(&i) {
                positions.extend(extra);
                positions.sort_unstable();
            }
            PatchInfo {
                key: group.key.clone(),
                positions,
            }
        })
        .collect();
    candidate.extend(new_groups);
    candidate.sort_by_key(|p| std::cmp::Reverse(p.key.width * p.key.height));
    Some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_match_checks_full_rectangles_against_baseline_and_other_members() {
        let baseline = Rect::new((3, 5), (10, 10));
        let first = Rect::new((13, 5), (10, 10));
        let second = Rect::new((15, 14), (8, 10));
        assert!(disjoint(&[first], &[baseline]));
        assert!(disjoint(&[second], &[baseline]));
        assert!(!disjoint(&[first, second], &[baseline]));
        assert!(!disjoint(&[Rect::new((12, 14), (8, 2))], &[baseline]));
    }

    #[test]
    fn near_match_rate_rejects_agreement_limited_to_color_probes() {
        let size = (8, 8);
        let template = Template::new(
            (0..64)
                .map(|i| if i % 2 == 0 { 100 } else { -100 })
                .collect(),
            size,
            255,
        );
        let observed = Template::new(vec![100; 64], size, 255);
        assert!(template.similar(&observed, 255));
        let planes = vec![vec![100; 64]];
        let view = ImageView {
            width: 8,
            height: 8,
            color_channels: 1,
            bit_depth: 8,
            planes: &planes,
        };
        let atom = Atom {
            positions: vec![(0, 0)],
            template: observed,
            rects: vec![Rect::new((0, 0), size)],
        };
        assert!(Rate::new().savings(view, &atom, &template).unwrap() < 0);
    }
}
