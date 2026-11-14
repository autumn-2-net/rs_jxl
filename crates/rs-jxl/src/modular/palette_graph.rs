//! Bounded graph reordering of explicit absolute palette tuples only.
//! Uses the existing adjacency sampler; graph cost is not final JXL bit cost.

use std::collections::BTreeSet;

use super::{PaletteMap, adjacency_weighted_order};
use crate::Result;

#[path = "palette_minla.rs"]
mod minla;

const MAX_COLORS: usize = 256;
const MAX_SAMPLES: usize = 262_144;

fn adjacency_from_weights(frequencies: &[usize], weights: &[u64]) -> Vec<usize> {
    let n = frequencies.len();
    let mut edges = Vec::new();
    // Each undirected edge is already counted once in each matrix triangle.
    for a in 0..n {
        for b in a + 1..n {
            let count = weights[a * n + b];
            if count != 0 {
                edges.push((a, b, count));
            }
        }
    }
    adjacency_weighted_order(frequencies, edges)
}

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
        adjacency_from_weights(frequencies, &weights)
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
    fn matrix_seed_matches_pair_seed_with_ties_and_disconnected_colors() {
        for n in [0, 1, 2, 8, 17, 64, 256] {
            for dense in [false, true] {
                let frequencies: Vec<_> = (0..n).map(|i| (i * 7) % 5).collect();
                let mut weights = vec![0; n * n];
                let mut pairs = Vec::new();
                for a in 0..n {
                    for b in a + 1..n {
                        let count = if dense || (a % 3 == 0 && b % 5 == 0) {
                            (a * 13 + b * 17) % 7
                        } else {
                            0
                        };
                        weights[a * n + b] = count as u64;
                        weights[b * n + a] = count as u64;
                        pairs.extend(std::iter::repeat_n(((a as u64) << 32) | b as u64, count));
                    }
                }
                let expected = super::super::adjacency_order(&frequencies, pairs);
                let actual = adjacency_from_weights(&frequencies, &weights);
                assert_eq!(actual, expected, "colors={n} dense={dense}");
                let mut sorted = actual;
                sorted.sort_unstable();
                assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    #[ignore = "manual seed-stage microbenchmark; not a performance assertion"]
    fn matrix_seed_timing() {
        use std::{hint::black_box, time::Instant};
        for n in [16, 64, 256] {
            let frequencies = vec![1; n];
            let mut weights = vec![0; n * n];
            let mut pairs = Vec::new();
            for i in 0..MAX_SAMPLES * 2 {
                let a = i % n;
                let b = (a + 1 + (i / n) % (n - 1)) % n;
                weights[a * n + b] += 1;
                weights[b * n + a] += 1;
                pairs.push(((a.min(b) as u64) << 32) | a.max(b) as u64);
            }
            assert_eq!(
                adjacency_from_weights(&frequencies, &weights),
                super::super::adjacency_order(&frequencies, pairs.clone()),
            );
            let (mut old, mut new) = (Vec::new(), Vec::new());
            for round in 0..9 {
                let input = pairs.clone();
                if round % 2 == 0 {
                    let start = Instant::now();
                    black_box(adjacency_from_weights(&frequencies, black_box(&weights)));
                    new.push(start.elapsed());
                }
                let start = Instant::now();
                black_box(super::super::adjacency_order(
                    &frequencies,
                    black_box(input),
                ));
                old.push(start.elapsed());
                if round % 2 != 0 {
                    let start = Instant::now();
                    black_box(adjacency_from_weights(&frequencies, black_box(&weights)));
                    new.push(start.elapsed());
                }
            }
            old.sort_unstable();
            new.sort_unstable();
            eprintln!(
                "palette seed colors={n} pairs={} old={:?} matrix={:?}",
                pairs.len(),
                old[4],
                new[4]
            );
        }
    }

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
