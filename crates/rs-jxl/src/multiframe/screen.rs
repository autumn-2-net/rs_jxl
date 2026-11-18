//! Estimate the relative size of frame candidates before allocating any
//! candidate images. The encoder still creates exact, lossless payloads.
use super::{
    Rect, Strategy,
    candidates::Candidate,
    partition::LAYER_OVERHEAD_BYTES,
    splits::{area, union},
};
use crate::{EncodeOptions, Image, PixelData, modular::palette_groups::ycocg};
use std::collections::HashMap;

pub(super) fn select(
    candidates: &mut Vec<Candidate>,
    images: &[Image],
    index: usize,
    options: &EncodeOptions,
) {
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
    // RCT search includes YCoCg directly after identity. Use its decorrelated
    // channels as a proxy when the encoder is choosing a color transform.
    let search_rct = image.color_channels() == 3
        && options
            .modular
            .rct_candidates
            .is_some_and(|count| count >= 2);
    // A wider grid captures sparse details on large canvases. Keep the cheaper
    // grid for the transform search, whose color model already reduces variance.
    let grid = if !search_rct && u64::from(image.width()) * u64::from(image.height()) >= 256 * 256 {
        48
    } else {
        16
    };
    let mut replace_costs = HashMap::new();
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
            // Dense splits give up spatial context at each independent layer
            // boundary. Discount that loss for splits that skip large gaps.
            let scale = if search_rct && candidate.rects.len() > 1 {
                let covered: u64 = candidate.rects.iter().copied().map(area).sum();
                let bounds = candidate.rects.iter().copied().reduce(union).unwrap();
                1.0 + 0.125 * covered as f64 / area(bounds) as f64
            } else {
                1.0
            };
            let layers = candidate.rects.len() * (1 + usize::from(candidate.multiply_mask()));
            let mut sum = layers as f64 * 8.0 * LAYER_OVERHEAD_BYTES;
            for &rect in &candidate.rects {
                let estimate = || {
                    if search_rct {
                        estimate_rect::<true, 1021, 2041>(
                            image, target, reference, candidate, rect, grid,
                        )
                    } else {
                        estimate_rect::<false, 511, 1021>(
                            image, target, reference, candidate, rect, grid,
                        )
                    }
                };
                // Ordinary Replace rectangles encode identical target samples
                // across reference choices and competing partitions.
                let payload = if candidate.replace() {
                    *replace_costs.entry(rect).or_insert_with(estimate)
                } else {
                    estimate()
                };
                sum += payload * scale + area(rect) as f64 * 0.01;
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

// Signed RGB differences range from -255 to 255. RCT6 doubles the chroma
// range, and subtracting a clamped predictor can double that range again.
fn estimate_rect<const RCT: bool, const VALUES: usize, const ERRORS: usize>(
    image: &Image,
    target: &[u8],
    reference: Option<&[u8]>,
    candidate: &Candidate,
    rect: Rect,
    grid: u32,
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
            if RCT {
                let [y, co, cg] = ycocg(values[0].into(), values[1].into(), values[2].into());
                values[..3].copy_from_slice(&[y as i16, co as i16, cg as i16]);
            }
        }
        // The multiply mask has the same keep/clear pattern in every channel.
        values[channels] = i16::from(multiply && !changed);
        values
    };
    let mut values = [[0u16; VALUES]; 5];
    let mut errors = [[0u16; ERRORS]; 5];
    // Stratified, repeatable positions avoid locking the grid to image patterns.
    // Include one west/north neighborhood to estimate spatial predictability.
    let nx = rect.width.min(grid);
    let ny = rect.height.min(grid);
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
                values[c][((VALUES / 2) as i16 + v[c]) as usize] += 1;
                errors[c][((ERRORS / 2) as i16 + v[c] - pred) as usize] += 1;
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
    area(rect) as f64 * bits_per_pixel
}
