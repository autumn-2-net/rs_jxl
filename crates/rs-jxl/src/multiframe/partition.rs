//! Native geometry-only proposal, additional to the legacy tile candidate.
//!
//! A tile-aligned binary hierarchy guarantees disjoint output. Its unconstrained
//! cut minimizes estimated cost within that hierarchy; a sibling-merge heap
//! coarsens the cut when the caller's rectangle budget is smaller. This is not a
//! globally optimal partition. The parent measures actual encoded bytes and
//! retains existing candidates, so estimation error cannot replace their wins.
use super::{Rect, diff::DiffMap};
use crate::{Error, Result};
use std::{cmp::Ordering, collections::BinaryHeap};

// Heuristic bytes per layer for its frame header and independent coding setup,
// not a measured or normative JXL header size. The objective is
// sum(rectangle area * bytes_per_pixel + LAYER_OVERHEAD_BYTES).
const LAYER_OVERHEAD_BYTES: f64 = 64.0;
const NO_PARENT: usize = usize::MAX;

#[derive(Clone, Copy)]
struct Cost {
    pixel: f64,
    layer: f64,
}

impl Cost {
    fn new(bytes_per_pixel: f64) -> Self {
        // Uniform normalization preserves the objective while keeping even the
        // largest finite caller estimate from overflowing intermediate costs.
        let scale = bytes_per_pixel.max(1.0);
        Self {
            pixel: bytes_per_pixel / scale,
            layer: LAYER_OVERHEAD_BYTES / scale,
        }
    }

    fn rectangle(self, rect: Rect) -> f64 {
        let area = u64::from(rect.width) * u64::from(rect.height);
        area as f64 * self.pixel + self.layer
    }
}

struct Node {
    bounds: Rect,
    parent: usize,
    children: Option<[usize; 2]>,
    cost: f64,
    split: bool,
    selected: bool,
}

#[derive(Clone, Copy)]
struct Merge {
    penalty: f64,
    node: usize,
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: lowest penalty, then earliest tree node.
        other
            .penalty
            .total_cmp(&self.penalty)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Merge {}

pub(super) fn propose(
    map: &DiffMap,
    tile_size: u32,
    max_rectangles: usize,
    bytes_per_pixel: f64,
) -> Result<Option<Vec<Rect>>> {
    let Some(bounds) = map.bounds() else {
        return Ok(None);
    };
    validate(tile_size, max_rectangles, bytes_per_pixel)?;
    if max_rectangles == 1 || bytes_per_pixel == 0.0 {
        let mut result = Vec::new();
        result
            .try_reserve_exact(1)
            .map_err(|_| Error::Limit("partition rectangle allocation"))?;
        result.push(bounds);
        return Ok(Some(result));
    }
    let mut tiles = map.tile_bounds(tile_size)?;
    partition(&mut tiles, tile_size, max_rectangles, bytes_per_pixel).map(Some)
}

fn validate(tile_size: u32, max_rectangles: usize, bytes_per_pixel: f64) -> Result<()> {
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
    Ok(())
}

// Input is the DiffMap contract: one nonempty, tight, in-bounds box per occupied
// tile. Splits never cross a tile, so no pixels or sample values are revisited.
fn partition(
    tiles: &mut [Rect],
    tile_size: u32,
    max_rectangles: usize,
    bytes_per_pixel: f64,
) -> Result<Vec<Rect>> {
    validate(tile_size, max_rectangles, bytes_per_pixel)?;
    if tiles.is_empty() {
        return Ok(Vec::new());
    }
    let capacity = tiles
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or(Error::Limit("partition hierarchy size"))?;
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Limit("partition hierarchy allocation"))?;
    build(
        tiles,
        tile_size,
        NO_PARENT,
        Cost::new(bytes_per_pixel),
        &mut nodes,
    );
    let mut count = select(0, &mut nodes);

    if count > max_rectangles {
        let mut heap = BinaryHeap::new();
        // Each internal node above the selected cut can enter only once. There
        // are count - 1 such nodes; pruned descendants never enter the heap.
        heap.try_reserve_exact(count - 1)
            .map_err(|_| Error::Limit("partition merge allocation"))?;
        for index in 0..nodes.len() {
            queue_merge(index, &nodes, &mut heap);
        }
        while let Some(next) = heap.peek() {
            if count <= max_rectangles && next.penalty > 0.0 {
                break;
            }
            let next = heap.pop().expect("merge heap is nonempty");
            let [left, right] = nodes[next.node].children.expect("merge has children");
            debug_assert!(nodes[left].selected && nodes[right].selected);
            nodes[left].selected = false;
            nodes[right].selected = false;
            nodes[next.node].selected = true;
            count -= 1;
            let parent = nodes[next.node].parent;
            if parent != NO_PARENT {
                queue_merge(parent, &nodes, &mut heap);
            }
        }
    }
    debug_assert!(count <= max_rectangles);
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("partition rectangle allocation"))?;
    result.extend(nodes.iter().filter(|n| n.selected).map(|n| n.bounds));
    // Unstable sorting uses no auxiliary allocation. Disjoint rectangles cannot
    // share an origin, so this key also gives a total, reproducible raster order.
    result.sort_unstable_by_key(|r| (r.y, r.x, r.height, r.width));
    Ok(result)
}

fn union(a: Rect, b: Rect) -> Rect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    Rect {
        x,
        y,
        width: (a.x + a.width).max(b.x + b.width) - x,
        height: (a.y + a.height).max(b.y + b.height) - y,
    }
}

// Midpoint splits of the longest occupied tile-coordinate span halve that
// span. With u32 coordinates, recursion depth is at most 64: O(n * 64) building,
// O(n log n) heap work and O(n) checked storage, without all-pairs comparisons
// or rectangle-budget dynamic programming. Partitioning is in-place.
fn build(
    tiles: &mut [Rect],
    tile_size: u32,
    parent: usize,
    cost: Cost,
    nodes: &mut Vec<Node>,
) -> (usize, f64) {
    let bounds = tiles.iter().copied().reduce(union).expect("occupied tiles");
    let index = nodes.len();
    let own_cost = cost.rectangle(bounds);
    nodes.push(Node {
        bounds,
        parent,
        children: None,
        cost: own_cost,
        split: false,
        selected: false,
    });
    if tiles.len() == 1 {
        return (index, own_cost);
    }

    let first_x = bounds.x / tile_size;
    let first_y = bounds.y / tile_size;
    let span_x = (bounds.x + bounds.width - 1) / tile_size - first_x;
    let span_y = (bounds.y + bounds.height - 1) / tile_size - first_y;
    let horizontal = span_x >= span_y;
    let cut = if horizontal {
        first_x + span_x / 2 + 1
    } else {
        first_y + span_y / 2 + 1
    };
    let mut middle = 0;
    for i in 0..tiles.len() {
        let coordinate = if horizontal { tiles[i].x } else { tiles[i].y };
        if coordinate / tile_size < cut {
            tiles.swap(i, middle);
            middle += 1;
        }
    }
    debug_assert!(middle > 0 && middle < tiles.len());
    let (left, right) = tiles.split_at_mut(middle);
    let (left, left_cost) = build(left, tile_size, index, cost, nodes);
    let (right, right_cost) = build(right, tile_size, index, cost, nodes);
    let split_cost = left_cost + right_cost;
    nodes[index].children = Some([left, right]);
    nodes[index].split = split_cost < own_cost;
    (index, own_cost.min(split_cost))
}

fn select(index: usize, nodes: &mut [Node]) -> usize {
    if nodes[index].split {
        let [left, right] = nodes[index].children.expect("split has children");
        select(left, nodes) + select(right, nodes)
    } else {
        nodes[index].selected = true;
        1
    }
}

fn queue_merge(index: usize, nodes: &[Node], heap: &mut BinaryHeap<Merge>) {
    if let Some([left, right]) = nodes[index].children
        && nodes[left].selected
        && nodes[right].selected
    {
        heap.push(Merge {
            penalty: nodes[index].cost - (nodes[left].cost + nodes[right].cost),
            node: index,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Image, PixelData, PixelLayout};

    fn diff_map(width: u32, height: u32, changed: &impl Fn(u32, u32) -> bool) -> DiffMap {
        let mut pixels = vec![0; width as usize * height as usize];
        let previous = Image::new(
            width,
            height,
            PixelLayout::Gray,
            PixelData::U8(pixels.clone()),
        )
        .unwrap();
        for y in 0..height {
            for x in 0..width {
                pixels[y as usize * width as usize + x as usize] = u8::from(changed(x, y));
            }
        }
        let current = Image::new(width, height, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
        DiffMap::between(&previous, &current).unwrap()
    }

    fn checked_proposal(
        width: u32,
        height: u32,
        tile: u32,
        maximum: usize,
        bytes_per_pixel: f64,
        changed: impl Fn(u32, u32) -> bool,
    ) -> Vec<Rect> {
        let map = diff_map(width, height, &changed);
        let proposal = propose(&map, tile, maximum, bytes_per_pixel).unwrap();
        assert_eq!(proposal.is_none(), map.bounds().is_none());
        assert_eq!(
            proposal,
            propose(&map, tile, maximum, bytes_per_pixel).unwrap()
        );
        let result = proposal.unwrap_or_default();
        let mut tiles = map.tile_bounds(tile).unwrap();
        assert_eq!(
            result,
            partition(&mut tiles, tile, maximum, bytes_per_pixel).unwrap()
        );
        assert!(result.len() <= maximum);
        for rect in &result {
            assert!(rect.width > 0 && rect.height > 0);
            assert!(rect.x.checked_add(rect.width).unwrap() <= width);
            assert!(rect.y.checked_add(rect.height).unwrap() <= height);
        }
        assert!(
            result
                .windows(2)
                .all(|r| (r[0].y, r[0].x) < (r[1].y, r[1].x))
        );
        for y in 0..height {
            for x in 0..width {
                let count = result
                    .iter()
                    .filter(|r| x >= r.x && y >= r.y && x < r.x + r.width && y < r.y + r.height)
                    .count();
                assert!(count <= 1, "overlap at ({x}, {y})");
                if changed(x, y) {
                    assert_eq!(count, 1, "uncovered change at ({x}, {y})");
                }
            }
        }
        result
    }

    #[test]
    fn dense_changes_merge_to_one_rectangle() {
        for maximum in [1, 2, 32] {
            assert_eq!(
                checked_proposal(53, 37, 8, maximum, 3.0, |_, _| true),
                [Rect {
                    x: 0,
                    y: 0,
                    width: 53,
                    height: 37
                }]
            );
        }
    }

    #[test]
    fn distant_sparse_changes_stay_separate_when_budget_allows() {
        for maximum in [1, 2, 32] {
            let boxes = checked_proposal(129, 97, 8, maximum, 1.0, |x, y| {
                (x == 0 && y == 0) || (x == 128 && y == 96)
            });
            assert_eq!(boxes.len(), maximum.min(2));
            if maximum == 1 {
                assert_eq!(
                    boxes[0],
                    Rect {
                        x: 0,
                        y: 0,
                        width: 129,
                        height: 97
                    }
                );
            } else {
                assert!(boxes.iter().all(|r| r.width == 1 && r.height == 1));
            }
        }
    }

    #[test]
    fn nearby_fragments_trade_added_area_against_layer_overhead() {
        let changed = |x, y| y == 2 && (x == 7 || x == 10);
        for maximum in [1, 2, 32] {
            let boxes = checked_proposal(19, 5, 8, maximum, 1.0, changed);
            assert_eq!(
                boxes,
                [Rect {
                    x: 7,
                    y: 2,
                    width: 4,
                    height: 1
                }]
            );
        }
        let boxes = checked_proposal(19, 5, 8, 32, 128.0, changed);
        assert_eq!(boxes.len(), 2);
    }

    #[test]
    fn more_than_32_tiles_are_coarsened_not_discarded() {
        let changed = |x, y| x % 32 == 0 && y % 32 == 0;
        assert_eq!(
            diff_map(257, 193, &changed).tile_bounds(8).unwrap().len(),
            63
        );
        for maximum in [1, 2, 32, 64] {
            let boxes = checked_proposal(257, 193, 8, maximum, 8.0, changed);
            assert_eq!(boxes.len(), maximum.min(63));
        }
    }

    #[test]
    fn odd_dimensions_and_edge_changes_are_covered_without_overlap() {
        for tile in [1, 3, 8, 32, 128] {
            for maximum in [1, 2, 32] {
                checked_proposal(35, 27, tile, maximum, 2.5, |x, y| {
                    x == 34 || y == 26 || (x * 17 + y * 23) % 47 == 0
                });
            }
        }
    }

    #[test]
    fn tied_merges_are_deterministic() {
        let changed = |x, y| (x == 0 || x == 64) && (y == 0 || y == 64);
        let boxes = checked_proposal(65, 65, 8, 2, 4.0, changed);
        assert_eq!(boxes.len(), 2);
        let mut reversed = diff_map(65, 65, &changed).tile_bounds(8).unwrap();
        reversed.reverse();
        assert_eq!(boxes, partition(&mut reversed, 8, 2, 4.0).unwrap());
    }

    #[test]
    fn empty_and_single_tile_inputs() {
        assert!(checked_proposal(7, 5, 3, 32, 1.0, |_, _| false).is_empty());
        assert_eq!(
            checked_proposal(7, 5, 32, 32, 1.0, |x, y| x == 6 && y == 4),
            [Rect {
                x: 6,
                y: 4,
                width: 1,
                height: 1
            }]
        );
    }

    #[test]
    fn zero_and_extreme_finite_estimates_are_well_defined() {
        for estimate in [0.0, f64::MIN_POSITIVE, f64::MAX] {
            let boxes =
                checked_proposal(65, 3, 8, 32, estimate, |x, y| y == 1 && (x == 0 || x == 64));
            assert_eq!(boxes.len(), if estimate == f64::MAX { 2 } else { 1 });
        }
    }

    #[test]
    fn invalid_parameters_are_errors() {
        assert!(partition(&mut [], 0, 1, 1.0).is_err());
        assert!(partition(&mut [], 1, 0, 1.0).is_err());
        for estimate in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(partition(&mut [], 1, 1, estimate).is_err());
        }
    }

    #[test]
    fn large_coordinates_do_not_overflow_geometry_or_costs() {
        let last = u32::MAX - 1;
        let input = [
            Rect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            Rect {
                x: last,
                y: last,
                width: 1,
                height: 1,
            },
        ];
        for maximum in [1, 2, 32] {
            let mut tiles = input;
            let result = partition(&mut tiles, 1, maximum, f64::MAX).unwrap();
            assert_eq!(result.len(), maximum.min(2));
            if maximum == 1 {
                assert_eq!(
                    result[0],
                    Rect {
                        x: 0,
                        y: 0,
                        width: u32::MAX,
                        height: u32::MAX
                    }
                );
            } else {
                assert_eq!(result, input);
            }
        }
    }
}
