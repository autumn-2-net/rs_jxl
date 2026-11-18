//! Bounded two-axis guillotine geometry proposal, additional to old partitions.
//! Each cut separates complete occupied tiles. Prefix/suffix tight boxes need
//! not fill the half-plane: this avoids forcing sparse columns into 2-D boxes.
use super::{Rect, diff::DiffMap};
use crate::{Error, Result};
use std::{cmp::Ordering, collections::BinaryHeap};
#[derive(Clone, Copy)]
struct Cut {
    gain: f64,
    balance: usize,
    axis: u8,
    coordinate: u32,
}
impl Cut {
    fn compare(self, other: Self) -> Ordering {
        self.gain
            .total_cmp(&other.gain)
            .then_with(|| other.balance.cmp(&self.balance))
            .then_with(|| other.axis.cmp(&self.axis))
            .then_with(|| other.coordinate.cmp(&self.coordinate))
    }
}
#[derive(Clone, Copy)]
struct Pending {
    cut: Cut,
    leaf: usize,
}
impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cut
            .compare(other.cut)
            .then_with(|| other.leaf.cmp(&self.leaf))
    }
}
impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pending {}
struct Leaf {
    start: usize,
    end: usize,
    bounds: Rect,
    active: bool,
}
struct Work {
    used: usize,
    limit: usize,
}
pub(super) fn union(a: Rect, b: Rect) -> Rect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    Rect {
        x,
        y,
        width: (a.x + a.width).max(b.x + b.width) - x,
        height: (a.y + a.height).max(b.y + b.height) - y,
    }
}
pub(super) fn area(r: Rect) -> u64 {
    u64::from(r.width) * u64::from(r.height)
}
fn coordinate(r: Rect, axis: u8, tile: u32) -> u32 {
    if axis == 0 { r.x / tile } else { r.y / tile }
}
fn sort(atoms: &mut [Rect], axis: u8, tile: u32) {
    atoms.sort_unstable_by_key(|r| (coordinate(*r, axis, tile), r.y, r.x, r.height, r.width));
}
/// Both axes are one bounded leaf evaluation. Sorting is allocation-free; the
/// prefix vector is reused, and the suffix union is maintained in reverse.
fn evaluate(
    atoms: &mut [Rect],
    bounds: Rect,
    tile: u32,
    pixel: f64,
    layer: f64,
    work: &mut Work,
    prefix: &mut Vec<Rect>,
) -> Result<Option<Cut>> {
    let n = atoms.len();
    if n < 2 || work.used.checked_add(n).is_none_or(|sum| sum > work.limit) {
        return Ok(None);
    }
    work.used += n;
    prefix.clear();
    prefix
        .try_reserve(n)
        .map_err(|_| Error::Limit("guillotine prefix allocation"))?;
    let mut best: Option<Cut> = None;
    for axis in 0..2 {
        sort(atoms, axis, tile);
        prefix.clear();
        let mut bbox = atoms[0];
        prefix.push(bbox);
        for &atom in &atoms[1..] {
            bbox = union(bbox, atom);
            prefix.push(bbox);
        }
        let mut suffix = atoms[n - 1];
        for right in (1..n).rev() {
            let split = coordinate(atoms[right], axis, tile);
            if coordinate(atoms[right - 1], axis, tile) < split {
                // The two child boxes occupy opposite tile half-planes, so
                // their areas sum to at most the parent's area.
                let saved = area(bounds) - area(prefix[right - 1]) - area(suffix);
                let cut = Cut {
                    gain: saved as f64 * pixel - layer,
                    balance: right.abs_diff(n - right),
                    axis,
                    coordinate: split,
                };
                if cut.gain > 0.0 && best.is_none_or(|old| cut.compare(old) == Ordering::Greater) {
                    best = Some(cut);
                }
            }
            suffix = union(suffix, atoms[right - 1]);
        }
    }
    Ok(best)
}
pub(super) fn propose(
    map: &DiffMap,
    tile_size: u32,
    max_rectangles: usize,
    bytes_per_pixel: f64,
) -> Result<Option<Vec<Rect>>> {
    let Some(bounds) = map.bounds() else {
        return Ok(None);
    };
    if tile_size == 0 || max_rectangles == 0 {
        return Err(Error::InvalidInput(
            "partition tile size and rectangle limit must be positive".into(),
        ));
    }
    if !bytes_per_pixel.is_finite() || bytes_per_pixel < 0.0 {
        return Err(Error::InvalidInput(
            "partition bytes per pixel must be finite and nonnegative".into(),
        ));
    }
    // A fully changed rectangle has no empty area that a cut could save.
    if max_rectangles == 1
        || bytes_per_pixel == 0.0
        || map.changed_pixels() == u64::from(bounds.width) * u64::from(bounds.height)
    {
        let mut result = Vec::new();
        result
            .try_reserve_exact(1)
            .map_err(|_| Error::Limit("guillotine result allocation"))?;
        result.push(bounds);
        return Ok(Some(result));
    }
    let mut atoms = map.tile_bounds(tile_size)?;
    let limit = atoms
        .len()
        .checked_mul(64)
        .ok_or(Error::Limit("guillotine evaluation work"))?;
    partition(
        &mut atoms,
        bounds,
        tile_size,
        max_rectangles,
        bytes_per_pixel,
        limit,
    )
    .map(Some)
}
fn partition(
    atoms: &mut [Rect],
    bounds: Rect,
    tile: u32,
    maximum: usize,
    rate: f64,
    limit: usize,
) -> Result<Vec<Rect>> {
    let mut leaves = Vec::new();
    leaves
        .try_reserve(1)
        .map_err(|_| Error::Limit("guillotine leaf allocation"))?;
    leaves.push(Leaf {
        start: 0,
        end: atoms.len(),
        bounds,
        active: true,
    });
    let mut prefix = Vec::new();
    let mut heap = BinaryHeap::new();
    let mut work = Work { used: 0, limit };
    let scale = rate.max(1.0);
    let pixel = rate / scale;
    let layer = 64.0 / scale;
    if maximum > 1
        && let Some(cut) = evaluate(atoms, bounds, tile, pixel, layer, &mut work, &mut prefix)?
    {
        heap.try_reserve(1)
            .map_err(|_| Error::Limit("guillotine heap allocation"))?;
        heap.push(Pending { cut, leaf: 0 });
    }
    let mut count = 1;
    while count < maximum {
        let Some(next) = heap.pop() else {
            break;
        };
        let parent = &leaves[next.leaf];
        let (start, end) = (parent.start, parent.end);
        let slice = &mut atoms[start..end];
        sort(slice, next.cut.axis, tile);
        let middle =
            slice.partition_point(|r| coordinate(*r, next.cut.axis, tile) < next.cut.coordinate);
        debug_assert!(middle > 0 && middle < slice.len());
        leaves[next.leaf].active = false;
        leaves
            .try_reserve(2)
            .map_err(|_| Error::Limit("guillotine leaf allocation"))?;
        for (a, b) in [(start, start + middle), (start + middle, end)] {
            let bounds = atoms[a..b]
                .iter()
                .copied()
                .reduce(union)
                .expect("nonempty guillotine child");
            let leaf = leaves.len();
            leaves.push(Leaf {
                start: a,
                end: b,
                bounds,
                active: true,
            });
            if let Some(cut) = evaluate(
                &mut atoms[a..b],
                bounds,
                tile,
                pixel,
                layer,
                &mut work,
                &mut prefix,
            )? {
                heap.try_reserve(1)
                    .map_err(|_| Error::Limit("guillotine heap allocation"))?;
                heap.push(Pending { cut, leaf });
            }
        }
        count += 1;
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("guillotine result allocation"))?;
    result.extend(leaves.into_iter().filter(|n| n.active).map(|n| n.bounds));
    result.sort_unstable_by_key(|r| (r.y, r.x, r.height, r.width));
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Image, PixelData, PixelLayout};
    fn map(w: u32, h: u32, changed: impl Fn(u32, u32) -> bool) -> DiffMap {
        let first = Image::new(
            w,
            h,
            PixelLayout::Gray,
            PixelData::U8(vec![0; (w * h) as usize]),
        )
        .unwrap();
        let second = Image::new(
            w,
            h,
            PixelLayout::Gray,
            PixelData::U8(
                (0..h)
                    .flat_map(|y| (0..w).map(|x| u8::from(changed(x, y))).collect::<Vec<_>>())
                    .collect(),
            ),
        )
        .unwrap();
        DiffMap::between(&first, &second).unwrap()
    }
    fn check(map: &DiffMap, rects: &[Rect], maximum: usize) {
        assert!(rects.len() <= maximum);
        for y in 0..map.height() {
            for x in 0..map.width() {
                let n = rects
                    .iter()
                    .filter(|r| x >= r.x && y >= r.y && x < r.x + r.width && y < r.y + r.height)
                    .count();
                assert!(n <= 1, "overlap at {x},{y}");
                if map.changed(x, y) {
                    assert_eq!(n, 1, "missing {x},{y}");
                }
            }
        }
        for r in rects {
            assert!(
                r.width > 0
                    && r.height > 0
                    && r.x + r.width <= map.width()
                    && r.y + r.height <= map.height()
            );
        }
    }
    #[test]
    fn sparse_columns_choose_axis_and_stop_on_header_cost() {
        let m = map(147, 227, |x, y| x % 16 == 1 && y % 16 == 1);
        let a = propose(&m, 16, 10, 0.3).unwrap().unwrap();
        check(&m, &a, 10);
        assert!(a.iter().all(|r| r.width == 1));
        assert!(a.iter().map(|r| area(*r)).sum::<u64>() < 3000);
        assert_eq!(a, propose(&m, 16, 10, 0.3).unwrap().unwrap());
    }
    #[test]
    fn edge_dense_empty_one_and_extreme_rates() {
        let empty = map(9, 7, |_, _| false);
        assert!(propose(&empty, 4, 3, 1.0).unwrap().is_none());
        for dense in [false, true] {
            let m = map(19, 13, |x, y| dense || x == 18 || y == 12);
            for tile in [1, 4, 32] {
                for maximum in [1, 3, 20] {
                    for rate in [0.0, 1.0, f64::MIN_POSITIVE, f64::MAX] {
                        let result = propose(&m, tile, maximum, rate).unwrap().unwrap();
                        check(&m, &result, maximum);
                        if dense {
                            assert_eq!(result.len(), 1);
                        }
                    }
                }
            }
            for (tile, k, rate) in [
                (0, 3, 1.0),
                (4, 0, 1.0),
                (4, 3, -1.0),
                (4, 3, f64::INFINITY),
                (4, 3, f64::NAN),
            ] {
                assert!(propose(&m, tile, k, rate).is_err());
            }
        }
    }
    #[test]
    fn interrupted_evaluation_keeps_exact_coverage() {
        let m = map(127, 61, |x, y| (x + 3 * y) % 17 == 0);
        for budget in [0, 1, 32, 128, 256] {
            let mut atoms = m.tile_bounds(4).unwrap();
            let result = partition(&mut atoms, m.bounds().unwrap(), 4, 64, 2.0, budget).unwrap();
            check(&m, &result, 64);
        }
        let result = propose(&m, 4, usize::MAX, 2.0).unwrap().unwrap();
        check(&m, &result, usize::MAX);
    }
}
