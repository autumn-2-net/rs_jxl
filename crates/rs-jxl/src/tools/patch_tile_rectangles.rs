//! Exact Replace experiment on the verified tile groups from patch_tiles and
//! patch_off_grid. Each input group already matches in every supplied plane.
//! Joining requires a one-to-one translation of the *entire* position list,
//! so the resulting rectangle covers exactly the same verified samples.
use super::{AtlasPacking, Tile};
use std::{collections::BTreeMap, ops::Range};

#[derive(Clone)]
pub(super) struct Group {
    pub range: Range<usize>,
    pub width: usize,
    pub height: usize,
}

fn join_axis(groups: Vec<Group>, index: &[Tile], stride: usize, horizontal: bool) -> Vec<Group> {
    let mut pending: BTreeMap<_, _> = groups
        .into_iter()
        .map(|g| (index[g.range.start].origin, g))
        .collect();
    let mut joined = Vec::with_capacity(pending.len());
    while let Some((origin, mut group)) = pending.pop_first() {
        loop {
            let (dx, dy) = if horizontal {
                (group.width, 0)
            } else {
                (0, group.height)
            };
            let Some(next) = dy
                .checked_mul(stride)
                .and_then(|n| origin.checked_add(n))
                .and_then(|n| n.checked_add(dx))
            else {
                break;
            };
            let Some(neighbor) = pending.get(&next) else {
                break;
            };
            if group.range.len() != neighbor.range.len()
                || (horizontal && group.height != neighbor.height)
                || (!horizontal && group.width != neighbor.width)
                || !index[group.range.clone()]
                    .iter()
                    .zip(&index[neighbor.range.clone()])
                    .all(|(a, b)| {
                        a.origin % stride + dx == b.origin % stride
                            && a.origin / stride + dy == b.origin / stride
                    })
            {
                break;
            }
            if horizontal {
                group.width += neighbor.width;
            } else {
                group.height += neighbor.height;
            }
            pending.remove(&next);
        }
        joined.push(group);
    }
    joined
}

pub(super) fn expand(
    stride: usize,
    index: &[Tile],
    groups: &[Group],
    old_width: usize,
    budget: usize,
) -> Option<(Vec<Group>, AtlasPacking)> {
    let rows = join_axis(groups.to_vec(), index, stride, true);
    let mut rects = join_axis(rows, index, stride, false);
    if rects.len() == groups.len() {
        return None;
    }
    // Tall rectangles first keep shelf packing linear after the sort. Charge
    // padded area to the same atlas budget; retain the square plan if it fails.
    rects.sort_by_key(|r| std::cmp::Reverse(r.height));
    let width = old_width.max(rects.iter().map(|r| r.width).max()?);
    let (mut x, mut y, mut row_height) = (0usize, 0usize, 0usize);
    let mut origins = Vec::with_capacity(rects.len());
    for rect in &rects {
        if rect.width > width - x {
            y = y.checked_add(row_height)?;
            x = 0;
            row_height = 0;
        }
        origins.push((x, y));
        x += rect.width;
        row_height = row_height.max(rect.height);
    }
    let height = y.checked_add(row_height)?;
    if height > i32::MAX as usize || width.checked_mul(height)? > budget {
        return None;
    }
    Some((rects, (width, height, origins)))
}
