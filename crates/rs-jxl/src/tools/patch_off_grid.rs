//! Encoder-only rolling 2D lookup inspired by QLIC's off-grid probe concept.
//! Seeds remain on the grid, destinations do not. This is not an exhaustive
//! rectangle dictionary and does not import QLIC syntax or predictors.
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use super::{ImageView, Tile, TilePatchOptions, TilePatchStats, compare};
use crate::{Error, Result};

const X: u64 = 0x9e37_79b1_85eb_ca87;
const Y: u64 = 0xc2b2_ae3d_27d4_eb4f;

fn pixel(image: ImageView<'_>, p: usize) -> u64 {
    image.planes.iter().fold(0u64, |h, plane| {
        let mut v = u64::from(plane[p] as u32).wrapping_add(X);
        v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h.wrapping_mul(Y).wrapping_add(v ^ (v >> 27))
    })
}

fn seed_hash(image: ImageView<'_>, origin: usize, side: usize) -> Option<u64> {
    let mut flat = true;
    let mut hash = 0u64;
    for y in 0..side {
        let mut row = 0u64;
        for x in 0..side {
            let p = origin + y * image.width + x;
            flat &= image.planes.iter().all(|plane| plane[p] == plane[origin]);
            row = row.wrapping_mul(X).wrapping_add(pixel(image, p));
        }
        hash = hash.wrapping_mul(Y).wrapping_add(row);
    }
    (!flat).then_some(hash)
}

fn cells(width: usize, origin: usize, side: usize) -> [usize; 4] {
    let columns = width.div_ceil(side);
    let (x, y) = (origin % width, origin / width);
    let (l, r, t, b) = (
        x / side,
        (x + side - 1) / side,
        y / side,
        (y + side - 1) / side,
    );
    [
        t * columns + l,
        t * columns + r,
        b * columns + l,
        b * columns + r,
    ]
}

/// O(width * side) rolling state, no full-frame hash image. Callback false
/// stops discovery when its explicit memory guard is reached.
fn windows(
    image: ImageView<'_>,
    side: usize,
    mut visit: impl FnMut(usize, u64) -> bool,
) -> Result<()> {
    let nx = image.width - side + 1;
    let len = nx
        .checked_mul(side)
        .ok_or(Error::Limit("off-grid hash rows"))?;
    let mut ring = Vec::new();
    ring.try_reserve_exact(len)
        .map_err(|_| Error::Limit("off-grid hash rows"))?;
    ring.resize(len, 0u64);
    let mut vertical = vec![0u64; nx];
    let mut pixels = vec![0u64; image.width];
    let xp = X.wrapping_pow((side - 1) as u32);
    let yp = Y.wrapping_pow(side as u32);
    for y in 0..image.height {
        for (x, p) in pixels.iter_mut().enumerate() {
            *p = pixel(image, y * image.width + x);
        }
        let mut row = pixels[..side]
            .iter()
            .fold(0u64, |h, &v| h.wrapping_mul(X).wrapping_add(v));
        for x in 0..nx {
            if x > 0 {
                row = row
                    .wrapping_sub(pixels[x - 1].wrapping_mul(xp))
                    .wrapping_mul(X)
                    .wrapping_add(pixels[x + side - 1]);
            }
            let slot = (y % side) * nx + x;
            vertical[x] = vertical[x]
                .wrapping_mul(Y)
                .wrapping_add(row)
                .wrapping_sub(ring[slot].wrapping_mul(yp));
            ring[slot] = row;
            if y + 1 >= side && !visit((y + 1 - side) * image.width + x, vertical[x]) {
                return Ok(());
            }
        }
    }
    Ok(())
}

pub(super) fn discover(
    image: ImageView<'_>,
    options: &TilePatchOptions,
    stats: &mut TilePatchStats,
) -> Result<Option<Vec<Tile>>> {
    let side = options.tile_size;
    // Bound seed storage too: a tiny tile on a large unique image must not
    // build a huge seed map before the occurrence guard gets its first chance.
    let seed_count = (image.width / side) * (image.height / side);
    if seed_count > options.max_discovery_positions {
        stats.discovery_limited = 1;
        return Ok(None);
    }
    let mut seeds = Vec::new();
    if seeds.try_reserve_exact(seed_count).is_err() {
        stats.discovery_limited = 1;
        return Ok(None);
    }
    for y in (0..=image.height - side).step_by(side) {
        for x in (0..=image.width - side).step_by(side) {
            let origin = y * image.width + x;
            stats.tiles_examined += 1;
            if let Some(hash) = seed_hash(image, origin, side) {
                seeds.push(Tile { hash, origin });
            } else {
                stats.constant_tiles += 1;
            }
        }
    }
    if seeds.is_empty() {
        return Ok(Some(Vec::new()));
    }
    seeds.sort_unstable_by_key(|s| (s.hash, s.origin));
    // Only sort full samples in a genuine collision bucket. Repeated texture
    // otherwise needs one equality scan per seed, not per sorting comparison.
    let mut start = 0;
    while start < seeds.len() {
        let end = start + seeds[start..].partition_point(|s| s.hash == seeds[start].hash);
        let first = seeds[start].origin;
        if !seeds[start + 1..end]
            .iter()
            .all(|s| compare(image, first, s.origin, side).is_eq())
        {
            seeds[start..end].sort_unstable_by(|a, b| {
                compare(image, a.origin, b.origin, side).then(a.origin.cmp(&b.origin))
            });
        }
        start = end;
    }
    seeds.dedup_by(|a, b| a.hash == b.hash && compare(image, a.origin, b.origin, side).is_eq());
    let mut lookup = HashMap::new();
    for (i, seed) in seeds.iter().enumerate() {
        lookup
            .entry(seed.hash)
            .and_modify(|r: &mut Range<usize>| r.end = i + 1)
            .or_insert(i..i + 1);
    }
    let mut occupied = HashSet::new();
    let mut found = Vec::new();
    // Group-local coarse occupancy bounds repeated equality work. It can reject
    // adjacent off-grid copies sharing a cell even when their pixels don't
    // overlap; this deliberately conservative heuristic never fabricates a hit.
    windows(image, side, |origin, hash| {
        stats.windows_examined += 1;
        let Some(range) = lookup.get(&hash) else {
            return true;
        };
        let touched = cells(image.width, origin, side);
        let candidates = &seeds[range.clone()];
        let group = if candidates.len() == 1 {
            let group = range.start;
            if touched
                .iter()
                .any(|&cell| occupied.contains(&(group, cell)))
            {
                return true;
            }
            if !compare(image, origin, seeds[group].origin, side).is_eq() {
                return true;
            }
            group
        } else {
            let Ok(i) = candidates.binary_search_by(|s| compare(image, s.origin, origin, side))
            else {
                return true;
            };
            range.start + i
        };
        if touched
            .iter()
            .any(|&cell| occupied.contains(&(group, cell)))
        {
            return true;
        }
        if found.len() == options.max_discovery_positions {
            stats.discovery_limited = 1;
            return false;
        }
        for cell in touched {
            occupied.insert((group, cell));
        }
        found.push(Tile { hash, origin });
        true
    })?;
    if stats.discovery_limited != 0 {
        Ok(None)
    } else {
        Ok(Some(found))
    }
}

/// Greedy coverage priority inherited from the grid detector. Disjoint coarse
/// cells avoid double-counting savings or zeroing already represented pixels.
pub(super) fn disjoint(
    width: usize,
    side: usize,
    index: &[Tile],
    ranges: &[Range<usize>],
    minimum: usize,
) -> (Vec<Tile>, Vec<Range<usize>>) {
    let mut occupied = HashSet::new();
    let mut result = Vec::new();
    let mut selected = Vec::new();
    for range in ranges {
        let start = result.len();
        for tile in &index[range.clone()] {
            if cells(width, tile.origin, side)
                .iter()
                .all(|c| !occupied.contains(c))
            {
                result.push(*tile);
            }
        }
        if result.len() - start < minimum {
            result.truncate(start);
            continue;
        }
        for tile in &result[start..] {
            occupied.extend(cells(width, tile.origin, side));
        }
        selected.push(start..result.len());
    }
    (result, selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::patches::{ReferenceImage, compose, discover_tiles};

    fn image(bits: u32) -> ReferenceImage {
        let (width, height) = (97, 69);
        let mask = (1 << bits) - 1;
        let mut image = ReferenceImage {
            width,
            height,
            bit_depth: bits,
            color_channels: 3,
            planes: (0..4)
                .map(|c| {
                    (0..width * height)
                        .map(|p| ((p * 1877 + c * 37) as i32) & mask)
                        .collect()
                })
                .collect(),
        };
        for (x, y) in [(0, 0), (37, 5), (71, 44)] {
            for c in 0..4 {
                for v in 0..16 {
                    for u in 0..16 {
                        image.planes[c][(y + v) * width + x + u] = if c == 3 {
                            0
                        } else {
                            ((u * 971 + v * 193 + c * 71 + u * v * 137) as i32) & mask
                        };
                    }
                }
            }
        }
        image
    }
    #[test]
    fn rolling_hash_matches_direct_windows_at_every_origin() {
        let image = image(16);
        for side in [1, 8, 16, 31] {
            let mut n = 0;
            windows(image.view(), side, |p, h| {
                if let Some(direct) = seed_hash(image.view(), p, side) {
                    assert_eq!(h, direct);
                }
                n += 1;
                true
            })
            .unwrap();
            assert_eq!(n, (image.width - side + 1) * (image.height - side + 1));
        }
    }
    #[test]
    fn shifted_copies_restore_hidden_rgb_and_full_sixteen_bits() {
        for bits in [8, 16] {
            let image = image(bits);
            let options = TilePatchOptions {
                off_grid: true,
                force: true,
                ..Default::default()
            };
            let plan = discover_tiles(image.view(), &options).unwrap().unwrap();
            assert!(
                plan.dictionary
                    .positions
                    .iter()
                    .any(|p| p.x % 16 != 0 || p.y % 16 != 0)
            );
            assert_eq!(
                compose(
                    ImageView {
                        planes: &plan.residual,
                        ..image.view()
                    },
                    plan.reference.view(),
                    &plan.dictionary
                )
                .unwrap(),
                image.planes
            );
        }
    }
    #[test]
    fn discovery_guard_declines_without_mutating_input() {
        let image = image(8);
        let original = image.planes.clone();
        let receipt = std::sync::Arc::new(super::super::TilePatchReceipt::default());
        let options = TilePatchOptions {
            off_grid: true,
            force: true,
            max_discovery_positions: 1,
            receipt: Some(receipt.clone()),
            ..Default::default()
        };
        assert!(discover_tiles(image.view(), &options).unwrap().is_none());
        assert_eq!(receipt.snapshot().discovery_limited, 1);
        assert_eq!(receipt.snapshot().tiles_examined, 0);
        assert_eq!(image.planes, original);
    }

    #[test]
    fn equality_cannot_ignore_alpha_or_invisible_rgb() {
        for channel in [0, 3] {
            let mut image = image(16);
            let mut stats = TilePatchStats::default();
            let options = TilePatchOptions {
                off_grid: true,
                ..Default::default()
            };
            let changed = 5 * image.width + 37;
            image.planes[channel][changed + 2 * image.width + 3] ^= 1;
            let found = discover(image.view(), &options, &mut stats)
                .unwrap()
                .unwrap();
            let seed = seed_hash(image.view(), 0, 16).unwrap();
            assert!(!found.iter().any(|t| t.origin == changed && t.hash == seed));
        }
    }

    #[test]
    fn selected_rectangles_do_not_overlap() {
        let image = image(8);
        let options = TilePatchOptions {
            off_grid: true,
            force: true,
            ..Default::default()
        };
        let plan = discover_tiles(image.view(), &options).unwrap().unwrap();
        let mut seen = vec![false; image.width * image.height];
        for position in &plan.dictionary.positions {
            for y in position.y as usize..position.y as usize + 16 {
                for x in position.x as usize..position.x as usize + 16 {
                    assert!(!std::mem::replace(&mut seen[y * image.width + x], true));
                }
            }
        }
    }
}
