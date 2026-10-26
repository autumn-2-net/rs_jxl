//! Metadata-only matched samples on the largest candidate's real-group grid.
//! This is a child of `progressive`; scoring and racing belong to the parent.

use super::super::scan::{LosslessScan, ScanTile};
use crate::modular::layout::Rect;
use crate::{Error, Result};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Region {
    pub rect: Rect,
    /// Total pixel area of this stratum, repeated for every sample in it.
    /// Average sampled per-pixel costs within a stratum before applying this
    /// weight once; neither raw cropped costs nor repeated weights are additive.
    pub weight: f64,
    pub stratum: usize,
}

struct Cell {
    rect: Rect,
    activity: f64,
    spatial: u64,
}

fn grid(width: usize, height: usize, side: usize) -> Result<(usize, usize, usize)> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidInput("empty group sampling image".into()));
    }
    // Image dimensions are u32; all grid coordinates therefore fit the
    // 32-bit axes of the Morton key, independently of the number of cells.
    u32::try_from(width).map_err(|_| Error::Limit("group sampling image dimensions"))?;
    u32::try_from(height).map_err(|_| Error::Limit("group sampling image dimensions"))?;
    width
        .checked_mul(height)
        .ok_or(Error::Limit("group sampling pixel count"))?;
    let columns = width.div_ceil(side);
    let rows = height.div_ceil(side);
    let count = columns
        .checked_mul(rows)
        .ok_or(Error::Limit("group sampling grid size"))?;
    Ok((columns, rows, count))
}

fn rect_end(rect: Rect, width: usize, height: usize) -> Result<(usize, usize)> {
    let right = rect
        .x
        .checked_add(rect.width)
        .ok_or(Error::Limit("group sampling rectangle extent"))?;
    let bottom = rect
        .y
        .checked_add(rect.height)
        .ok_or(Error::Limit("group sampling rectangle extent"))?;
    if rect.width == 0 || rect.height == 0 || right > width || bottom > height {
        return Err(Error::InvalidInput(
            "group sampling rectangle is empty or outside the image".into(),
        ));
    }
    Ok((right, bottom))
}

// Morton order keeps midpoint sampling spatially useful inside activity ties.
// Interleave both complete axes, including bits above the former cell cap.
fn spatial_key(x: u32, y: u32) -> u64 {
    let mut key = 0;
    for bit in 0..u32::BITS {
        key |= ((x >> bit & 1) as u64) << (2 * bit);
        key |= ((y >> bit & 1) as u64) << (2 * bit + 1);
    }
    key
}

fn add_tile(
    cells: &mut [Cell],
    tile: &ScanTile,
    width: usize,
    height: usize,
    side: usize,
    columns: usize,
) -> Result<()> {
    let (right, bottom) = rect_end(
        Rect {
            x: tile.x,
            y: tile.y,
            width: tile.width,
            height: tile.height,
        },
        width,
        height,
    )?;
    let activity = if tile.observations == 0 {
        0.0
    } else {
        // Convert before adding so even extreme metadata counters cannot wrap.
        (tile.horizontal as f64 + tile.vertical as f64) / tile.observations as f64
    };
    for y in tile.y / side..bottom.div_ceil(side) {
        for x in tile.x / side..right.div_ceil(side) {
            let cell = &mut cells[y * columns + x];
            let r = cell.rect;
            let overlap_width = right.min(r.x + r.width) - tile.x.max(r.x);
            let overlap_height = bottom.min(r.y + r.height) - tile.y.max(r.y);
            cell.activity += activity * (overlap_width * overlap_height) as f64;
        }
    }
    Ok(())
}

fn center(rect: Rect, width: usize, height: usize) -> (f64, f64) {
    (
        (rect.x as f64 + rect.width as f64 / 2.0) / width as f64,
        (rect.y as f64 + rect.height as f64 / 2.0) / height as f64,
    )
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

fn representative(cells: &[Cell], previous: &[Region], width: usize, height: usize) -> usize {
    let median = cells[cells.len() / 2].activity;
    let mut centroid = (0.0, 0.0);
    let mut area = 0.0;
    for cell in cells {
        let weight = (cell.rect.width * cell.rect.height) as f64;
        let p = center(cell.rect, width, height);
        centroid.0 += p.0 * weight;
        centroid.1 += p.1 * weight;
        area += weight;
    }
    centroid.0 /= area;
    centroid.1 /= area;
    let mut chosen = cells.len() / 2;
    let mut best_separation = -1.0;
    let mut best_centrality = f64::INFINITY;
    for (index, cell) in cells.iter().enumerate() {
        if cell.activity != median {
            continue;
        }
        let p = center(cell.rect, width, height);
        let separation = previous.iter().fold(2.0_f64, |d, r| {
            d.min(distance(p, center(r.rect, width, height)))
        });
        let centrality = distance(p, centroid);
        if separation > best_separation
            || (separation == best_separation && centrality < best_centrality)
        {
            chosen = index;
            best_separation = separation;
            best_centrality = centrality;
        }
    }
    chosen
}

fn enqueue(queue: &mut VecDeque<(usize, usize)>, lo: usize, hi: usize) {
    if lo < hi {
        queue.push_back((lo, hi));
    }
}

/// `side` is the maximum candidate group dimension (128, 256, 512 or 1024).
/// Include every macrocell, even without scan evidence. Rank-based low/mid/high
/// bands remain nonempty when there are at least three cells, including on a
/// blank image. The first round covers each band once; later rounds interleave
/// breadth-first midpoint samples until every rectangle has appeared once.
pub(super) fn plan(
    width: usize,
    height: usize,
    side: usize,
    scan: &LosslessScan,
) -> Result<Vec<Region>> {
    if !matches!(side, 128 | 256 | 512 | 1024) {
        return Err(Error::InvalidInput("invalid sampling macro side".into()));
    }
    let (columns, rows, count) = grid(width, height, side)?;
    if scan.available() && (scan.width != width || scan.height != height) {
        return Err(Error::InvalidInput(
            "scan dimensions do not match group sampling image".into(),
        ));
    }
    let mut cells = Vec::new();
    cells
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("group sampling cell allocation"))?;
    for y in 0..rows {
        for x in 0..columns {
            cells.push(Cell {
                rect: Rect {
                    x: x * side,
                    y: y * side,
                    width: side.min(width - x * side),
                    height: side.min(height - y * side),
                },
                activity: 0.0,
                spatial: spatial_key(x as u32, y as u32),
            });
        }
    }
    for tile in &scan.tiles {
        add_tile(&mut cells, tile, width, height, side, columns)?;
    }
    for cell in &mut cells {
        cell.activity /= (cell.rect.width * cell.rect.height) as f64;
    }
    cells.sort_unstable_by(|a, b| {
        a.activity
            .total_cmp(&b.activity)
            .then_with(|| a.spatial.cmp(&b.spatial))
    });

    let bands = count.min(3);
    let mut weights = [0usize; 3];
    let mut queues: [VecDeque<(usize, usize)>; 3] = std::array::from_fn(|_| VecDeque::new());
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("group sampling region allocation"))?;
    for stratum in 0..bands {
        let lo = stratum * count / bands;
        let hi = (stratum + 1) * count / bands;
        weights[stratum] = cells[lo..hi].iter().try_fold(0usize, |sum, cell| {
            sum.checked_add(cell.rect.width * cell.rect.height)
                .ok_or(Error::Limit("group sampling stratum area"))
        })?;
        let chosen = lo + representative(&cells[lo..hi], &result, width, height);
        result.push(Region {
            rect: cells[chosen].rect,
            weight: weights[stratum] as f64,
            stratum,
        });
        // Disjoint nonempty ranges never outnumber the original band's cells.
        queues[stratum]
            .try_reserve_exact(hi - lo)
            .map_err(|_| Error::Limit("group sampling ordering allocation"))?;
        enqueue(&mut queues[stratum], lo, chosen);
        enqueue(&mut queues[stratum], chosen + 1, hi);
    }
    while result.len() < count {
        for stratum in 0..bands {
            if let Some((lo, hi)) = queues[stratum].pop_front() {
                let mid = lo + (hi - lo) / 2;
                result.push(Region {
                    rect: cells[mid].rect,
                    weight: weights[stratum] as f64,
                    stratum,
                });
                enqueue(&mut queues[stratum], lo, mid);
                enqueue(&mut queues[stratum], mid + 1, hi);
            }
        }
    }
    Ok(result)
}

/// Return full-image row-major group IDs whose clipped union is exactly `rect`.
/// Interior edges must align to real groups; only final image edges may clip.
pub(super) fn groups_for_rect(
    rect: Rect,
    width: usize,
    height: usize,
    shift: u8,
) -> Result<Vec<usize>> {
    if shift > 3 {
        return Err(Error::InvalidInput("invalid sampling group shift".into()));
    }
    let side = 128usize << shift;
    let (columns, _, _) = grid(width, height, side)?;
    let (right, bottom) = rect_end(rect, width, height)?;
    if !rect.x.is_multiple_of(side)
        || !rect.y.is_multiple_of(side)
        || (right != width && right % side != 0)
        || (bottom != height && bottom % side != 0)
    {
        return Err(Error::InvalidInput(
            "sampling rectangle is not aligned to real groups".into(),
        ));
    }
    let x0 = rect.x / side;
    let y0 = rect.y / side;
    let x1 = right.div_ceil(side);
    let y1 = bottom.div_ceil(side);
    let count = (x1 - x0)
        .checked_mul(y1 - y0)
        .ok_or(Error::Limit("sampled real-group count"))?;
    let mut groups = Vec::new();
    groups
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("sampled real-group allocation"))?;
    for y in y0..y1 {
        for x in x0..x1 {
            // grid() checked the whole-image group count, bounding this ID.
            groups.push(y * columns + x);
        }
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn empty_scan(width: usize, height: usize) -> LosslessScan {
        LosslessScan {
            width,
            height,
            ..Default::default()
        }
    }

    fn scan_with_scores(width: usize, height: usize, side: usize, scores: &[u64]) -> LosslessScan {
        let mut scan = empty_scan(width, height);
        for y in (0..height).step_by(side) {
            for x in (0..width).step_by(side) {
                let width = side.min(width - x);
                let height = side.min(height - y);
                let area = (width * height) as u64;
                scan.tiles.push(ScanTile {
                    x,
                    y,
                    width,
                    height,
                    anchor_x: x + width / 2,
                    anchor_y: y + height / 2,
                    horizontal: scores[scan.tiles.len()] * area,
                    observations: 16 * area,
                    ..Default::default()
                });
            }
        }
        assert_eq!(scan.tiles.len(), scores.len());
        scan
    }

    fn assert_exact_coverage(regions: &[Region], width: usize, height: usize, shift: u8) {
        let side = 128usize << shift;
        let columns = width.div_ceil(side);
        let mut visits = vec![0; columns * height.div_ceil(side)];
        for region in regions {
            let r = region.rect;
            let ids = groups_for_rect(r, width, height, shift).unwrap();
            assert!(!ids.is_empty());
            assert!(ids.windows(2).all(|ids| ids[0] < ids[1]));
            let mut area = 0;
            for id in ids {
                visits[id] += 1;
                let x = id % columns * side;
                let y = id / columns * side;
                let w = side.min(width - x);
                let h = side.min(height - y);
                assert!(x >= r.x && y >= r.y);
                assert!(x + w <= r.x + r.width && y + h <= r.y + r.height);
                area += w * h;
            }
            // Distinct grid groups are disjoint; containment plus equal area
            // proves exact pixel coverage, without allocating a pixel mask.
            assert_eq!(area, r.width * r.height);
        }
        assert!(visits.iter().all(|&n| n == 1));
    }

    #[test]
    fn exact_coverage_for_all_group_shifts_and_nonmultiple_tails() {
        for macro_shift in 0..=3 {
            let side = 128usize << macro_shift;
            for (width, height) in [
                (2 * side, 2 * side),
                (2 * side + 17, side + 9),
                (side + 1, 1),
                (1, side + 1),
                (7, 11),
            ] {
                let regions = plan(width, height, side, &empty_scan(width, height)).unwrap();
                assert_eq!(regions.len(), width.div_ceil(side) * height.div_ceil(side));
                for shift in 0..=macro_shift {
                    assert_exact_coverage(&regions, width, height, shift);
                }
            }
        }
    }

    #[test]
    fn tiny_grids_use_only_nonempty_strata() {
        for (width, height, expected) in [(1, 1, 1), (129, 1, 2), (1, 129, 2)] {
            let regions = plan(width, height, 128, &LosslessScan::default()).unwrap();
            assert_eq!(regions.len(), expected);
            for (stratum, region) in regions.iter().enumerate() {
                assert_eq!(region.stratum, stratum);
                assert_eq!(
                    region.weight,
                    (region.rect.width * region.rect.height) as f64
                );
            }
            assert_eq!(
                regions.iter().map(|r| r.weight).sum::<f64>(),
                (width * height) as f64
            );
            assert_exact_coverage(&regions, width, height, 0);
        }
    }

    #[test]
    fn blank_mid_and_dense_strata_have_whole_stratum_area_weights() {
        let (width, height, side) = (273, 137, 128);
        let scan = scan_with_scores(width, height, side, &[0, 0, 4, 4, 12, 12]);
        let regions = plan(width, height, side, &scan).unwrap();
        let expected = [2 * 128 * 128, 17 * 128 + 128 * 9, (128 + 17) * 9];
        assert_eq!(regions.len(), 6);
        assert_eq!(
            regions[..3].iter().map(|r| r.stratum).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            regions[..3].iter().map(|r| r.weight).sum::<f64>(),
            (width * height) as f64
        );
        for region in &regions {
            assert_eq!(region.weight, expected[region.stratum] as f64);
            let index = region.rect.y / side * 3 + region.rect.x / side;
            assert_eq!(region.stratum, index / 2);
        }
        assert_exact_coverage(&regions, width, height, 0);
    }

    #[test]
    fn first_round_uses_band_medians_before_activity_extremes() {
        let scan = scan_with_scores(9 * 128, 128, 128, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let regions = plan(scan.width, scan.height, 128, &scan).unwrap();
        assert_eq!(
            regions[..3].iter().map(|r| r.rect.x).collect::<Vec<_>>(),
            [128, 4 * 128, 7 * 128]
        );
        for round in regions.chunks_exact(3) {
            assert_eq!(
                round.iter().map(|r| r.stratum).collect::<Vec<_>>(),
                [0, 1, 2]
            );
        }
    }

    #[test]
    fn activity_ties_prefer_central_then_spatially_separated_representatives() {
        let regions = plan(12 * 128, 128, 128, &LosslessScan::default()).unwrap();
        assert!([128, 2 * 128].contains(&regions[0].rect.x));
        assert_eq!(regions[1].rect.x, 7 * 128);
        assert_eq!(regions[2].rect.x, 11 * 128);
    }

    #[test]
    fn scan_tile_overlap_scores_both_cells_without_using_its_anchor() {
        let mut scan = empty_scan(384, 128);
        scan.tiles.push(ScanTile {
            x: 64,
            y: 0,
            width: 128,
            height: 128,
            anchor_x: 128,
            anchor_y: 0,
            horizontal: 128 * 128,
            observations: 128 * 128,
            ..Default::default()
        });
        let regions = plan(384, 128, 128, &scan).unwrap();
        assert_eq!(
            regions.iter().map(|r| r.rect.x).collect::<Vec<_>>(),
            [256, 0, 128]
        );
        scan.tiles[0].anchor_x = 64;
        assert_eq!(regions, plan(384, 128, 128, &scan).unwrap());
    }

    #[test]
    fn clipped_cells_are_ranked_by_density_not_raw_activity_totals() {
        let scan = scan_with_scores(273, 128, 128, &[8, 8, 8]);
        let regions = plan(273, 128, 128, &scan).unwrap();
        assert_eq!(
            regions.iter().map(|r| r.rect.x).collect::<Vec<_>>(),
            [0, 128, 256]
        );
    }

    #[test]
    fn plans_are_deterministic_full_permutations_even_when_all_cells_are_blank() {
        let (width, height, side) = (7 * 128 + 13, 5 * 128 + 9, 128);
        for score in [0, 8] {
            let scan = scan_with_scores(width, height, side, &[score; 48]);
            let regions = plan(width, height, side, &scan).unwrap();
            assert_eq!(regions, plan(width, height, side, &scan).unwrap());
            assert_eq!(regions.len(), 48);
            let mut seen = HashSet::new();
            let mut areas = [0usize; 3];
            for region in &regions {
                let r = region.rect;
                assert!(seen.insert((r.x, r.y, r.width, r.height)));
                areas[region.stratum] += r.width * r.height;
            }
            for region in &regions {
                assert_eq!(region.weight, areas[region.stratum] as f64);
            }
            assert_eq!(
                regions[..3].iter().map(|r| r.weight).sum::<f64>(),
                (width * height) as f64
            );
            assert_exact_coverage(&regions, width, height, 0);
        }
    }

    #[test]
    fn zero_observations_and_large_counters_do_not_overflow() {
        let mut scan = scan_with_scores(128, 128, 128, &[0]);
        scan.tiles[0].horizontal = u64::MAX;
        scan.tiles[0].vertical = u64::MAX;
        for observations in [0, 1, u64::MAX] {
            scan.tiles[0].observations = observations;
            let regions = plan(128, 128, 128, &scan).unwrap();
            assert_eq!(regions.len(), 1);
            assert_eq!(regions[0].weight, 16384.0);
        }
    }

    #[test]
    fn plans_reject_invalid_geometry_scan_extents_and_arithmetic_overflow() {
        let scan = LosslessScan::default();
        for side in [0, 1, 127, 129, 384, 2048, usize::MAX] {
            assert!(matches!(
                plan(1, 1, side, &scan),
                Err(Error::InvalidInput(_))
            ));
        }
        for (width, height) in [(0, 1), (1, 0), (0, 0)] {
            assert!(matches!(
                plan(width, height, 128, &scan),
                Err(Error::InvalidInput(_))
            ));
        }
        for (width, height) in [(usize::MAX, 2), (usize::MAX, usize::MAX)] {
            assert!(matches!(
                plan(width, height, 128, &scan),
                Err(Error::Limit(_))
            ));
        }
        let mut scan = scan_with_scores(128, 128, 128, &[0]);
        assert!(matches!(
            plan(129, 128, 128, &scan),
            Err(Error::InvalidInput(_))
        ));
        scan.tiles[0].width = 129;
        assert!(matches!(
            plan(128, 128, 128, &scan),
            Err(Error::InvalidInput(_))
        ));
        scan.tiles[0].width = 0;
        assert!(matches!(
            plan(128, 128, 128, &scan),
            Err(Error::InvalidInput(_))
        ));
        scan.tiles[0].width = 1;
        scan.tiles[0].x = usize::MAX;
        assert!(matches!(plan(128, 128, 128, &scan), Err(Error::Limit(_))));
    }

    #[test]
    fn group_mapping_rejects_dilation_cropping_and_invalid_shifts() {
        let base = Rect {
            x: 0,
            y: 0,
            width: 128,
            height: 128,
        };
        for rect in [
            Rect { x: 1, ..base },
            Rect { y: 1, ..base },
            Rect { width: 127, ..base },
            Rect {
                height: 127,
                ..base
            },
            Rect { width: 0, ..base },
            Rect { height: 0, ..base },
            Rect { x: 256, ..base },
            Rect { y: 256, ..base },
        ] {
            assert!(matches!(
                groups_for_rect(rect, 257, 257, 0),
                Err(Error::InvalidInput(_))
            ));
        }
        // A smaller macrocell cannot silently expand to a larger candidate group.
        assert!(matches!(
            groups_for_rect(base, 257, 257, 1),
            Err(Error::InvalidInput(_))
        ));
        for shift in [4, u8::MAX] {
            assert!(matches!(
                groups_for_rect(base, 257, 257, shift),
                Err(Error::InvalidInput(_))
            ));
        }
        for (width, height) in [(0, 257), (257, 0)] {
            assert!(matches!(
                groups_for_rect(base, width, height, 0),
                Err(Error::InvalidInput(_))
            ));
        }
        assert!(matches!(
            groups_for_rect(base, usize::MAX, 2, 0),
            Err(Error::Limit(_))
        ));
        for rect in [
            Rect {
                x: usize::MAX,
                ..base
            },
            Rect {
                y: usize::MAX,
                ..base
            },
        ] {
            assert!(matches!(
                groups_for_rect(rect, 257, 257, 0),
                Err(Error::Limit(_))
            ));
        }
    }

    #[test]
    fn group_selections_can_exceed_the_former_cell_cap() {
        let former_cap = 1 << 20;
        let width = (former_cap + 1) * 128;
        assert_eq!(
            grid(width, 1, 128).unwrap(),
            (former_cap + 1, 1, former_cap + 1)
        );
        let rect = Rect {
            x: former_cap * 128,
            y: 0,
            width: 128,
            height: 1,
        };
        assert_eq!(groups_for_rect(rect, width, 1, 0).unwrap(), [former_cap]);
        let whole = Rect {
            x: 0,
            y: 0,
            width,
            height: 1,
        };
        let groups = groups_for_rect(whole, width, 1, 0).unwrap();
        assert_eq!(groups.len(), former_cap + 1);
        assert!(groups.into_iter().eq(0..=former_cap));
    }

    #[test]
    fn group_mapping_handles_the_last_u32_image_coordinate() {
        let width = u32::MAX as usize;
        let rect = Rect {
            x: width - 127,
            y: 0,
            width: 127,
            height: 1,
        };
        assert_eq!(groups_for_rect(rect, width, 1, 0).unwrap(), [width / 128]);
    }

    #[test]
    fn morton_keys_interleave_all_u32_coordinate_bits_without_collisions() {
        for bit in 0..u32::BITS {
            assert_eq!(spatial_key(1 << bit, 0), 1u64 << (2 * bit));
            assert_eq!(spatial_key(0, 1 << bit), 1u64 << (2 * bit + 1));
        }
        assert_eq!(spatial_key(u32::MAX, u32::MAX), u64::MAX);
        assert_eq!(spatial_key(0, 0), 0);
    }
}
