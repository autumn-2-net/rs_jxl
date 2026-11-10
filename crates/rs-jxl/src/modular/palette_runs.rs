//! Run-start statistics inspired by QLIC's hot-run palette ordering, adapted
//! independently to JXL's explicit absolute colors. No QLIC payload or varint
//! semantics are used. Existing implicit colors/delta entries are not sorted.

use super::PaletteMap;
use crate::Result;

pub(super) fn order(
    colors: &[Vec<i32>],
    map: &mut PaletteMap,
    planes: &[Vec<i32>],
    hot_colors: Option<usize>,
) -> Result<Vec<usize>> {
    let n = colors.len();
    let mut ranked: Vec<_> = (0..n).collect();
    if n < 2 {
        return Ok(ranked);
    }
    let lexical = |&a: &usize, &b: &usize| colors[a].cmp(&colors[b]).then(a.cmp(&b));
    if hot_colors.is_some_and(|hot| hot == 0 || hot >= n) {
        ranked.sort_unstable_by(lexical);
        return Ok(ranked);
    }

    let scores = scores(colors, map, planes)?;
    Ok(rank(colors, &scores, hot_colors))
}

pub(super) fn scores(
    colors: &[Vec<i32>],
    map: &mut PaletteMap,
    planes: &[Vec<i32>],
) -> Result<Vec<usize>> {
    let _scope = crate::profile::scope("palette.order.runs.scan");
    let n = colors.len();

    // The caller has already assigned non-promoted implicit colors indices >=n.
    // Aliased promotions share the last explicit slot, as in final remapping.
    // This scratch use of index changes neither occurrence counts nor samples.
    for (i, color) in colors.iter().enumerate() {
        map.get(color)?.index = i;
    }
    let mut counts = vec![0usize; n];
    let mut color = vec![0; planes.len()];
    for pos in 0..planes[0].len() {
        // A raster run continues across rows. Changes in any channel count,
        // including alpha and RGB below zero alpha. Implicit colors also break
        // runs, even though those colors are not candidates for reordering.
        if pos != 0 && planes.iter().all(|plane| plane[pos] == plane[pos - 1]) {
            continue;
        }
        for (value, plane) in color.iter_mut().zip(planes) {
            *value = plane[pos];
        }
        let index = map.get(&color)?.index;
        if index < n {
            counts[index] += 1;
        }
    }
    colors
        .iter()
        .map(|color| map.get(color).map(|entry| counts[entry.index]))
        .collect()
}

pub(super) fn rank(colors: &[Vec<i32>], scores: &[usize], hot_colors: Option<usize>) -> Vec<usize> {
    let n = colors.len();
    let mut ranked: Vec<_> = (0..n).collect();
    let lexical = |&a: &usize, &b: &usize| colors[a].cmp(&colors[b]).then(a.cmp(&b));
    if hot_colors.is_some_and(|hot| hot == 0 || hot >= n) {
        ranked.sort_unstable_by(lexical);
        return ranked;
    }
    ranked.sort_unstable_by(|&a, &b| scores[b].cmp(&scores[a]).then_with(|| lexical(&a, &b)));
    if let Some(limit) = hot_colors {
        let mut hot = vec![false; n];
        for &index in &ranked[..limit] {
            hot[index] = true;
        }
        ranked.sort_unstable_by(|&a, &b| hot[b].cmp(&hot[a]).then_with(|| lexical(&a, &b)));
    }
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::transforms::PaletteMapping;

    #[test]
    fn aliases_share_counts_and_nonpromoted_implicit_colors_keep_their_index() {
        let colors = vec![vec![90; 3], vec![10; 3], vec![90; 3], vec![50; 3]];
        let planes = vec![vec![90, 0, 90, 10, 50, 50, 50]; 3];
        for mapping in [PaletteMapping::Tree, PaletteMapping::Flat] {
            let mut map = PaletteMap::new(3, mapping);
            let implicit = map.get(&[0; 3]).unwrap();
            implicit.index = colors.len() + 7;
            implicit.implicit = true;
            assert_eq!(
                order(&colors, &mut map, &planes, None).unwrap(),
                [0, 2, 1, 3]
            );
            assert_eq!(map.get(&[0; 3]).unwrap().index, colors.len() + 7);
            assert_eq!(
                order(&colors, &mut map, &planes, Some(2)).unwrap(),
                [0, 2, 1, 3]
            );
        }
    }

    #[test]
    fn trivial_partitions_are_lexical_and_do_not_scan() {
        let colors = vec![vec![9; 3], vec![-5; 3], vec![9; 3]];
        let mut map = PaletteMap::new(3, PaletteMapping::Tree);
        for limit in [0, colors.len(), usize::MAX] {
            assert_eq!(
                order(&colors, &mut map, &[], Some(limit)).unwrap(),
                [1, 0, 2]
            );
        }
        assert!(order(&[], &mut map, &[], None).unwrap().is_empty());
        assert_eq!(order(&colors[..1], &mut map, &[], None).unwrap(), [0]);
    }
}
