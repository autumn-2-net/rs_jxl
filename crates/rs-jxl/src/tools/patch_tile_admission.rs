//! Cheap, deliberately conservative admission before allocating the full
//! residual or running any MA learner/entropy backend. This is an estimate,
//! not a compressed-size bound. Forced experiments remain available separately.

use super::{AtlasPacking, ImageView, PatchInfo};
use crate::{entropy::hybrid_uint::HybridUintConfig, modular::predictor::clamped_gradient};

pub(super) struct Estimate {
    pub saved_bits: f64,
    pub baseline_bits: f64,
}

// Fixed JXL predictors only. Independent tile boundaries intentionally charge
// the source and reference alike; boundary disruption is charged separately.
fn tile_bits(planes: &[Vec<i32>], stride: usize, origin: usize, side: usize) -> f64 {
    let config = HybridUintConfig::default();
    let count = (side * side) as f64;
    let mut bits = 0.0;
    for plane in planes {
        let mut histogram = [[0u64; 128]; 4];
        let mut extra = [0u64; 4];
        for y in 0..side {
            for x in 0..side {
                let p = origin + y * stride + x;
                let left = if x == 0 { 0 } else { plane[p - 1] };
                let top = if y == 0 { 0 } else { plane[p - stride] };
                let tl = if x == 0 || y == 0 {
                    0
                } else {
                    plane[p - stride - 1]
                };
                let predictions = [0, left, top, clamped_gradient(top, left, tl)];
                for (mode, prediction) in predictions.into_iter().enumerate() {
                    let residual = plane[p] - prediction;
                    let packed = ((residual as u32) << 1) ^ ((residual >> 31) as u32);
                    let token = config.encode(packed);
                    histogram[mode][token.token as usize] += 1;
                    extra[mode] += u64::from(token.nbits);
                }
            }
        }
        bits += histogram
            .iter()
            .zip(extra)
            .map(|(counts, extra)| {
                count * count.log2()
                    - counts
                        .iter()
                        .filter(|&&n| n != 0)
                        .map(|&n| n as f64 * (n as f64).log2())
                        .sum::<f64>()
                    + extra as f64
            })
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
    }
    bits
}

fn integer_bits(value: u64) -> f64 {
    // Conservative coordinate/model proxy, not the actual dictionary coder.
    1.0 + 2.0 * (64 - value.saturating_add(1).leading_zeros() - 1) as f64
}

fn signed_bits(value: i64) -> f64 {
    integer_bits(((value as u64) << 1) ^ ((value >> 63) as u64))
}

pub(super) fn estimate(
    image: ImageView<'_>,
    info: &[PatchInfo],
    packing: &AtlasPacking,
    side: usize,
) -> Estimate {
    let mut source_bits = 0.0;
    let mut saving = 0.0;
    let mut dictionary = 512.0 + 64.0 * image.planes.len() as f64;
    let boundary = (2 * side) as f64 * image.planes.len() as f64 * image.bit_depth as f64;
    for (patch, &(ax, ay)) in info.iter().zip(&packing.2) {
        let (x, y) = patch.positions[0];
        let original = tile_bits(
            image.planes,
            image.width,
            y as usize * image.width + x as usize,
            side,
        );
        let copies = patch.positions.len() as f64;
        source_bits += original * copies;
        // One original tile is retained in the reference. Charge the right and
        // bottom predictor boundaries after clearing every occurrence.
        saving += (copies - 1.0) * original - copies * boundary;
        dictionary += integer_bits(ax as u64)
            + integer_bits(ay as u64)
            + 2.0 * integer_bits((side - 1) as u64)
            + integer_bits(patch.positions.len() as u64 - 1)
            + 16.0;
        let mut previous = None;
        for &(x, y) in &patch.positions {
            dictionary += match previous {
                Some((px, py)) => {
                    signed_bits(i64::from(x) - i64::from(px))
                        + signed_bits(i64::from(y) - i64::from(py))
                }
                None => integer_bits(u64::from(x)) + integer_bits(u64::from(y)),
            };
            dictionary += 2.0 * (1 + image.planes.len() - image.color_channels) as f64;
            previous = Some((x, y));
        }
    }
    let saved_bits = saving - dictionary;
    if saved_bits <= 0.0 {
        return Estimate {
            saved_bits,
            baseline_bits: source_bits.max(1.0),
        };
    }
    // Deterministic stratification over the complete grid, not just repeated
    // content or the upper-left image. At most 128 complete tiles are scored.
    let columns = image.width / side;
    let count = columns * (image.height / side);
    let samples = count.min(128);
    let mut sampled_bits = 0.0;
    for sample in 0..samples {
        let tile = (count / samples) * sample + (count % samples) * sample / samples;
        let origin = (tile / columns) * side * image.width + (tile % columns) * side;
        sampled_bits += tile_bits(image.planes, image.width, origin, side);
    }
    let covered = count * side * side;
    let edge_pixels = image.width * image.height - covered;
    let baseline_bits = (sampled_bits * count as f64 / samples as f64
        + edge_pixels as f64 * image.planes.len() as f64 * image.bit_depth as f64)
        .max(source_bits)
        .max(1.0);
    Estimate {
        saved_bits,
        baseline_bits,
    }
}
