//! Bounded source statistics for prioritization, never a bound on JXL size.
use std::collections::{BTreeMap, BTreeSet};

use super::scan::{LosslessScan, ScanRegion};
use crate::Image;

#[derive(Clone, Debug, Default)]
pub struct Features {
    pub samples: usize,
    pub unique_fraction: f64,
    pub flat_fraction: f64,
    pub gradient_zero_fraction: f64,
    pub residual_bias: f64,
    pub cross_channel_gain: f64,
    /// Gradient, left, top. Computed on source samples, not transformed channels.
    pub residual_unique_fraction: [f64; 3],
    pub residual_zero_fraction: [f64; 3],
    /// Exact repeated tiles in interleaved layout channels only; separate extra
    /// planes are not part of this routing feature or a patch-equality proof.
    pub repeated_tile_fraction: f64,
    pub sampled_tiles: usize,
}

fn entropy(counts: impl Iterator<Item = u64>) -> f64 {
    let counts: Vec<_> = counts.collect();
    let n = counts.iter().sum::<u64>() as f64;
    if n == 0.0 {
        return 0.0;
    }
    counts
        .into_iter()
        .filter(|&n| n != 0)
        .map(|c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

fn bucket(r: i32) -> usize {
    if r == 0 {
        0
    } else {
        (32 - r.unsigned_abs().leading_zeros()) as usize * 2 - usize::from(r > 0)
    }
}

// Different deterministic jitter in each spatial stratum avoids a fixed stride
// aliasing regular text/tiles. It is not a whole-image cardinality estimate.
fn jitter(mut v: usize) -> usize {
    v ^= v >> 16;
    v = v.wrapping_mul(0x45d9f3b);
    v ^= v >> 16;
    v
}

pub(super) fn measure(image: &Image, regions: &[ScanRegion]) -> Features {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let channels = image.layout().channels();
    let color = if channels < 3 { 1 } else { 3 };
    let data = image.pixels();
    let mut tuples = BTreeSet::new();
    let mut residuals: [BTreeSet<[i32; 4]>; 3] = Default::default();
    let mut flat = 0;
    let mut zeros = [0; 3];
    let mut signed = 0i64;
    let mut magnitude = 0u64;
    let mut n = 0;
    let mut joint = [[0u64; 34]; 34];
    for (region_id, r) in regions.iter().enumerate() {
        let count = r.width * r.height;
        for i in 0..count.min(2048) {
            let lo = i * count / count.min(2048);
            let hi = (i + 1) * count / count.min(2048);
            let p = lo + jitter(i + region_id * 2048) % (hi - lo);
            let x = r.x + p % r.width;
            let y = r.y + p / r.width;
            let mut value = [0; 4];
            let mut deltas = [[0; 4]; 3];
            for c in 0..channels {
                let v = data.get((y * w + x) * channels + c);
                let left = if x > 0 {
                    data.get((y * w + x - 1) * channels + c)
                } else {
                    0
                };
                let top = if y > 0 {
                    data.get(((y - 1) * w + x) * channels + c)
                } else {
                    left
                };
                let tl = if x > 0 && y > 0 {
                    data.get(((y - 1) * w + x - 1) * channels + c)
                } else {
                    top
                };
                let gradient = (left + top - tl).clamp(left.min(top), left.max(top));
                value[c] = v;
                deltas[0][c] = v - gradient;
                deltas[1][c] = v - left;
                deltas[2][c] = v - top;
                if c < color {
                    signed += i64::from(v - gradient);
                    magnitude += u64::from((v - gradient).unsigned_abs());
                }
            }
            tuples.insert(value);
            flat += usize::from(deltas[1][..channels].iter().all(|&v| v == 0));
            for k in 0..3 {
                zeros[k] += usize::from(deltas[k][..channels].iter().all(|&v| v == 0));
                residuals[k].insert(deltas[k]);
            }
            for c in 1..color {
                joint[bucket(deltas[0][c - 1])][bucket(deltas[0][c])] += 1;
            }
            n += 1;
        }
    }
    let mut marginal = [0; 34];
    let total: u64 = joint.iter().flatten().sum();
    let conditional = joint
        .iter()
        .map(|row| {
            for (dst, src) in marginal.iter_mut().zip(row) {
                *dst += *src;
            }
            if total == 0 {
                0.0
            } else {
                entropy(row.iter().copied()) * row.iter().sum::<u64>() as f64 / total as f64
            }
        })
        .sum::<f64>();
    let marginal = entropy(marginal.into_iter());
    // Exact equality of sampled tiles, including alpha and invisible RGB.
    // The map owns at most 64 * 16 * 16 * 4 integer samples.
    let tw = w.min(16);
    let th = h.min(16);
    let mut tiles = BTreeMap::new();
    let mut origins = BTreeSet::new();
    let mut tile_count = 0;
    for gy in 0..8 {
        for gx in 0..8 {
            let x = gx * (w - tw) / 7;
            let y = gy * (h - th) / 7;
            if !origins.insert((x, y)) {
                continue;
            }
            let mut values = Vec::with_capacity(tw * th * channels);
            for yy in y..y + th {
                for xx in x..x + tw {
                    for c in 0..channels {
                        values.push(data.get((yy * w + xx) * channels + c));
                    }
                }
            }
            if values
                .chunks_exact(channels)
                .all(|v| v == &values[..channels])
            {
                continue;
            }
            *tiles.entry(values).or_insert(0usize) += 1;
            tile_count += 1;
        }
    }
    Features {
        samples: n,
        unique_fraction: tuples.len() as f64 / n.max(1) as f64,
        flat_fraction: flat as f64 / n.max(1) as f64,
        gradient_zero_fraction: zeros[0] as f64 / n.max(1) as f64,
        residual_bias: signed.unsigned_abs() as f64 / magnitude.max(1) as f64,
        cross_channel_gain: if marginal == 0.0 {
            0.0
        } else {
            ((marginal - conditional) / marginal).clamp(0.0, 1.0)
        },
        residual_unique_fraction: residuals.map(|v| v.len() as f64 / n.max(1) as f64),
        residual_zero_fraction: zeros.map(|v| v as f64 / n.max(1) as f64),
        repeated_tile_fraction: tiles.values().map(|&n| n.saturating_sub(1)).sum::<usize>() as f64
            / tile_count.max(1) as f64,
        sampled_tiles: tile_count,
    }
}

pub(super) fn regions(image: &Image, scan: &LosslessScan, side: usize) -> Vec<ScanRegion> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let pw = w.min(side);
    let ph = h.min(side);
    let make = |x: usize, y: usize| ScanRegion {
        x: x.saturating_sub(pw / 2).min(w - pw),
        y: y.saturating_sub(ph / 2).min(h - ph),
        width: pw,
        height: ph,
    };
    let mut result = Vec::new();
    // Include diverse coverage-selected activity regions, plus spatial anchors.
    let anchors = scan
        .regions()
        .into_iter()
        .map(|r| (r.x + r.width / 2, r.y + r.height / 2))
        .chain([(w / 2, h / 2), (pw / 2, ph / 2), (w - pw / 2, h - ph / 2)]);
    for (x, y) in anchors {
        let r = make(x, y);
        if !result.contains(&r) {
            result.push(r);
        }
        if result.len() == 3 {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    #[test]
    fn alpha_and_invisible_rgb_are_not_dropped() {
        let image = Image::new(
            64,
            64,
            PixelLayout::Rgba,
            PixelData::U8(
                (0..64 * 64)
                    .flat_map(|i| [(i % 251) as u8, 7, 9, 0])
                    .collect(),
            ),
        )
        .unwrap();
        let r = regions(&image, &LosslessScan::default(), 128);
        let f = measure(&image, &r);
        assert!(f.unique_fraction > 0.05);
        assert!(f.flat_fraction < 0.01);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn constants_and_wide_samples_have_finite_statistics() {
        for image in [
            Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![0])).unwrap(),
            Image::new(
                131,
                129,
                PixelLayout::Rgb,
                PixelData::U16((0..131 * 129 * 3).map(|i| (i % 2 * 65535) as u16).collect()),
            )
            .unwrap(),
        ] {
            let r = regions(&image, &LosslessScan::default(), 64);
            let f = measure(&image, &r);
            assert!(f.cross_channel_gain.is_finite());
            assert!((0.0..=1.0).contains(&f.cross_channel_gain));
            assert!((0.0..=1.0).contains(&f.residual_bias));
            assert!(f.samples <= 6144);
            assert!(r.iter().all(|r| r.x + r.width <= image.width() as usize
                && r.y + r.height <= image.height() as usize));
        }
    }
}
