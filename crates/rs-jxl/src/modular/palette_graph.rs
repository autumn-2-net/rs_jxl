//! Bounded graph reordering of explicit absolute palette tuples only.
//! Uses the existing adjacency sampler; graph cost is not final JXL bit cost.

use std::collections::BTreeSet;

use super::{PaletteMap, adjacency_order};
use crate::Result;

#[path = "palette_minla.rs"]
mod minla;

const MAX_COLORS: usize = 256;
const MAX_SAMPLES: usize = 262_144;

#[allow(clippy::too_many_arguments)]
pub(super) fn order(
    colors: &[Vec<i32>],
    frequencies: &[usize],
    map: &mut PaletteMap,
    planes: &[Vec<i32>],
    width: usize,
    height: usize,
    ezeng: bool,
    rounds: u8,
) -> Result<Option<Vec<usize>>> {
    let n = colors.len();
    if !(2..=MAX_COLORS).contains(&n) || !(3..=4).contains(&planes.len()) {
        crate::profile::note(
            "palette.graph.bypass",
            format_args!("colors={n} channels={}", planes.len()),
        );
        return Ok(None);
    }
    // Duplicate promoted implicit entries alias the last table slot. Their
    // effective index depends on the order, so a fixed graph would misprice it.
    if colors.iter().collect::<BTreeSet<_>>().len() != n {
        crate::profile::note("palette.graph.bypass", "duplicate explicit tuples");
        return Ok(None);
    }
    let _scope = crate::profile::scope("palette.order.graph");
    let build = crate::profile::scope("palette.order.graph.build");
    for (i, color) in colors.iter().enumerate() {
        map.get(color)?.index = i;
    }
    let len = width * height;
    let step = len.div_ceil(MAX_SAMPLES).max(1);
    let mut weights = vec![0u64; n * n];
    let mut pairs = Vec::new();
    let mut color = vec![0; planes.len()];
    let mut index = |pos: usize| -> Result<usize> {
        for (value, plane) in color.iter_mut().zip(planes) {
            *value = plane[pos];
        }
        Ok(map.get(&color)?.index)
    };
    let mut edges = 0usize;
    let mut add = |a: usize, b: usize| {
        // Non-promoted implicit entries remain outside the explicit graph.
        if a >= n || b >= n || a == b {
            return;
        }
        weights[a * n + b] += 1;
        weights[b * n + a] += 1;
        edges += 1;
        if !ezeng {
            pairs.push(((a.min(b) as u64) << 32) | a.max(b) as u64);
        }
    };
    for pos in (0..len).step_by(step) {
        let current = index(pos)?;
        if pos % width + 1 < width {
            add(current, index(pos + 1)?);
        }
        if pos / width + 1 < height {
            add(current, index(pos + width)?);
        }
    }
    drop(build);
    if edges == 0 {
        crate::profile::note("palette.graph.bypass", "no sampled explicit edges");
        return Ok(None);
    }
    let seed = crate::profile::scope("palette.order.graph.seed");
    let mut order = if ezeng {
        minla::ezeng(&weights, n)
    } else {
        adjacency_order(frequencies, pairs)
    };
    drop(seed);
    let refine = crate::profile::scope("palette.order.graph.refine");
    let stats = minla::refine(&weights, &mut order, rounds);
    drop(refine);
    crate::profile::note(
        "palette.graph.result",
        format_args!(
            "seed={} colors={n} samples={} edges={edges} rounds={} moves={} before={} after={}",
            if ezeng { "ezeng" } else { "adjacency" },
            len.div_ceil(step),
            stats.rounds,
            stats.moves,
            stats.before,
            stats.after
        ),
    );
    Ok(Some(order))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::transforms::PaletteMapping;

    #[test]
    fn graph_excludes_implicit_indices_and_keeps_hidden_rgb_distinct() {
        let colors = vec![vec![1, 3, 5, 0], vec![7, 9, 11, 0], vec![13, 15, 17, 255]];
        let implicit = vec![0; 4];
        let pixels = [
            &colors[0], &implicit, &colors[1], &colors[2], &colors[0], &implicit,
        ];
        let planes: Vec<Vec<i32>> = (0..4)
            .map(|c| pixels.iter().map(|p| p[c]).collect())
            .collect();
        for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
            let mut map = PaletteMap::new(4, mapping);
            map.get(&[0; 4]).unwrap().index = colors.len() + 7;
            let order = order(&colors, &[2, 1, 1], &mut map, &planes, 3, 2, true, 4)
                .unwrap()
                .unwrap();
            let mut sorted = order;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2]);
            assert_eq!(map.get(&[0; 4]).unwrap().index, colors.len() + 7);
        }
    }

    #[test]
    fn aliased_and_large_tables_bypass_before_sampling() {
        let mut map = PaletteMap::new(3, PaletteMapping::Tree);
        let aliases = vec![vec![4, 7, 9]; 2];
        let empty = vec![Vec::new(); 3];
        assert!(
            order(&aliases, &[1, 1], &mut map, &empty, 0, 0, true, 4)
                .unwrap()
                .is_none()
        );
        let large: Vec<_> = (0..257).map(|i| vec![i, 3, 5]).collect();
        assert!(
            order(&large, &vec![1; 257], &mut map, &empty, 0, 0, false, 4)
                .unwrap()
                .is_none()
        );
    }
}
