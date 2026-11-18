//! Estimate the relative size of frame candidates before allocating any
//! candidate images. The encoder still creates exact, lossless payloads.
use super::{Rect, Strategy, candidates::Candidate, partition::LAYER_OVERHEAD_BYTES};
use crate::{Image, PixelData};

pub(super) fn select(candidates: &mut Vec<Candidate>, images: &[Image], index: usize) {
    let image = &images[index];
    if candidates.len() <= 1 {
        return;
    }
    let target = match image.pixels() {
        PixelData::U8(data) if image.extra_planes().is_empty() && !image.has_modular_planes() => {
            Some(data.as_slice())
        }
        _ => None,
    };
    let mut best = (0, f64::INFINITY);
    for (rank, candidate) in candidates.iter().enumerate() {
        let reference = match (candidate.strategy, candidate.reference_index) {
            (Strategy::AddDelta, Some(i)) => match images[i].pixels() {
                PixelData::U8(data) => Some(data.as_slice()),
                _ => unreachable!("AddDelta candidates require U8 samples"),
            },
            _ => None,
        };
        let bits = if let Some(target) = target {
            let mut sum = 0.0;
            for &rect in &candidate.rects {
                sum += estimate_rect(image, target, reference, candidate, rect);
                if sum >= best.1 {
                    break;
                }
            }
            sum
        } else {
            candidate.estimated_work() as f64
        };
        if bits < best.1 {
            best = (rank, bits);
        }
    }
    let selected = candidates.swap_remove(best.0);
    candidates.clear();
    candidates.push(selected);
}

fn estimate_rect(
    image: &Image,
    target: &[u8],
    reference: Option<&[u8]>,
    candidate: &Candidate,
    rect: Rect,
) -> f64 {
    let channels = image.layout().channels();
    let multiply = candidate.multiply_mask();
    let mask = if multiply || candidate.mask {
        candidate.diff.as_deref()
    } else {
        None
    };
    let planes = channels + usize::from(multiply);
    let sample = |x, y| {
        let start = (y as usize * image.width() as usize + x as usize) * channels;
        let changed = mask.is_none_or(|diff| diff.changed(x, y));
        let mut values = [0i16; 5];
        if changed {
            for c in 0..channels {
                values[c] =
                    i16::from(target[start + c]) - reference.map_or(0, |r| i16::from(r[start + c]));
            }
        }
        // The multiply mask has the same keep/clear pattern in every channel.
        values[channels] = i16::from(multiply && !changed);
        values
    };
    let mut values = [[0u16; 511]; 5];
    let mut errors = [[0u16; 1021]; 5];
    // Stratified, repeatable positions avoid locking the grid to image patterns.
    // Include one west/north neighborhood to estimate spatial predictability.
    let nx = rect.width.min(16);
    let ny = rect.height.min(16);
    let mut jitter = 1u32;
    let mut position = |cell: u32, extent: u32, cells: u32| {
        let left = u64::from(cell) * u64::from(extent) / u64::from(cells);
        let right = u64::from(cell + 1) * u64::from(extent) / u64::from(cells);
        jitter = jitter.wrapping_mul(1664525).wrapping_add(1013904223);
        (left + (((right - left) * u64::from(jitter)) >> 32)) as u32
    };
    for row in 0..ny {
        for col in 0..nx {
            let x = rect.x + position(col, rect.width, nx);
            let y = rect.y + position(row, rect.height, ny);
            let w = x.saturating_sub(1).max(rect.x);
            let n = y.saturating_sub(1).max(rect.y);
            let v = sample(x, y);
            let west = sample(w, y);
            let north = sample(x, n);
            let nw = sample(w, n);
            for c in 0..planes {
                let pred = if y == rect.y {
                    if x == rect.x { 0 } else { west[c] }
                } else if x == rect.x {
                    north[c]
                } else {
                    (west[c] + north[c] - nw[c]).clamp(west[c].min(north[c]), west[c].max(north[c]))
                };
                values[c][(255 + v[c]) as usize] += 1;
                errors[c][(510 + v[c] - pred) as usize] += 1;
            }
        }
    }
    let samples = f64::from(nx * ny);
    let entropy = |counts: &[u16]| {
        counts
            .iter()
            .filter(|&&n| n != 0)
            .map(|&n| {
                let p = f64::from(n) / samples;
                -p * p.log2()
            })
            .sum::<f64>()
    };
    let bits_per_pixel: f64 = (0..planes)
        .map(|c| entropy(&values[c]).min(entropy(&errors[c])))
        .sum();
    let area = f64::from(rect.width) * f64::from(rect.height);
    // Even an apparently flat sample costs something to cover. Use the same
    // per-layer overhead as the rectangle partitioner.
    (1 + usize::from(multiply)) as f64 * 8.0 * LAYER_OVERHEAD_BYTES + area * (bits_per_pixel + 0.01)
}
