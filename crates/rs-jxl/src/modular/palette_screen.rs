//! Experimental palette-order admission before MA learning. Costs are estimates,
//! not an encoded-size guarantee; ambiguous results retain the source default.

use super::{PaletteMap, luma, runs};
use crate::Result;
use crate::modular::cost::{self, ChannelView};

const TILE: usize = 64;
const GRID: usize = 4;
const MIN_GAIN: f64 = 0.03;

struct Sample {
    ids: Vec<i32>,
    width: usize,
    height: usize,
    population: usize,
    fold: usize,
}

#[derive(Clone, Copy, Debug)]
struct Score {
    total: f64,
    folds: [Option<f64>; 2],
    table: f64,
}

fn default_order(colors: &[Vec<i32>], frequencies: &[usize]) -> Vec<usize> {
    let mut order: Vec<_> = (0..colors.len()).collect();
    order.sort_by(|&a, &b| {
        luma(&colors[a], frequencies[a])
            .partial_cmp(&luma(&colors[b], frequencies[b]))
            .unwrap()
    });
    order
}

fn samples(indices: &[i32], width: usize, height: usize) -> Vec<Sample> {
    let _scope = crate::profile::scope("palette.order.screen.sample");
    let nx = width.div_ceil(TILE).min(GRID);
    let ny = height.div_ceil(TILE).min(GRID);
    let mut result = Vec::with_capacity(nx * ny);
    for gy in 0..ny {
        let y0 = gy * height / ny;
        let y1 = (gy + 1) * height / ny;
        let th = TILE.min(y1 - y0);
        let y = y0 + (y1 - y0 - th) / 2;
        for gx in 0..nx {
            let x0 = gx * width / nx;
            let x1 = (gx + 1) * width / nx;
            let tw = TILE.min(x1 - x0);
            let x = x0 + (x1 - x0 - tw) / 2;
            let mut ids = Vec::with_capacity(tw * th);
            for yy in y..y + th {
                let start = yy * width + x;
                ids.extend_from_slice(&indices[start..start + tw]);
            }
            result.push(Sample {
                ids,
                width: tw,
                height: th,
                population: (x1 - x0) * (y1 - y0),
                fold: (gx + gy) % 2,
            });
        }
    }
    result
}

fn remap_order(order: &[usize], canonical: &[usize]) -> Vec<i32> {
    let mut remap = vec![0; order.len()];
    for (index, &original) in order.iter().enumerate() {
        // Multiple promoted implicit colors can alias one tuple. Last assignment wins.
        remap[canonical[original]] = index as i32;
    }
    remap
}

fn score(
    order: &[usize],
    colors: &[Vec<i32>],
    canonical: &[usize],
    samples: &[Sample],
) -> Result<Score> {
    let _scope = crate::profile::scope("palette.order.screen.cost");
    let n = order.len();
    let remap = remap_order(order, canonical);
    let palette: Vec<_> = (0..colors[0].len())
        .flat_map(|c| order.iter().map(move |&i| colors[i][c]))
        .collect();
    let table = f64::from(cost::estimate_cost(&[ChannelView {
        pixels: &palette,
        width: n,
        height: colors[0].len(),
        stride: n,
    }])?);
    let mut bits = [0.0; 2];
    let mut populations = [0usize; 2];
    let mut indices = Vec::new();
    for sample in samples {
        indices.clear();
        indices.extend(
            sample
                .ids
                .iter()
                .map(|&id| remap.get(id as usize).copied().unwrap_or(id)),
        );
        // Contiguous original-resolution tiles preserve interior neighbors.
        // Tile boundaries use the cost kernel's own edge rules for every order.
        let cost = f64::from(cost::estimate_cost(&[ChannelView {
            pixels: &indices,
            width: sample.width,
            height: sample.height,
            stride: sample.width,
        }])?);
        bits[sample.fold] += cost * sample.population as f64 / sample.ids.len() as f64;
        populations[sample.fold] += sample.population;
    }
    let pixels: usize = populations.iter().sum();
    Ok(Score {
        total: bits.iter().sum::<f64>() + table,
        folds: std::array::from_fn(|fold| {
            (populations[fold] != 0)
                .then(|| bits[fold] * pixels as f64 / populations[fold] as f64 + table)
        }),
        table,
    })
}

fn admitted(candidate: Score, baseline: Score) -> bool {
    let improves = |new: f64, old: f64| {
        new.is_finite() && old.is_finite() && old > 0.0 && new < old * (1.0 - MIN_GAIN)
    };
    improves(candidate.total, baseline.total)
        && candidate
            .folds
            .iter()
            .zip(baseline.folds)
            .all(|(new, old)| match (new, old) {
                (Some(new), Some(old)) => improves(*new, old),
                (None, None) => true,
                _ => false,
            })
}

pub(super) fn order(
    colors: &[Vec<i32>],
    frequencies: &[usize],
    map: &mut PaletteMap,
    planes: &[Vec<i32>],
    width: usize,
    height: usize,
) -> Result<(Vec<usize>, Vec<i32>)> {
    let _scope = crate::profile::scope("palette.order.screen");
    let default = default_order(colors, frequencies);
    for (id, color) in colors.iter().enumerate() {
        map.get(color)?.index = id;
    }
    let mut run_counts = vec![0; colors.len()];
    let mut indices = super::raster_indices(planes, map, &mut run_counts)?;
    if colors.len() < 2 {
        let _decision = crate::profile::scope("palette.order.screen.default");
        return Ok((default, indices));
    }
    let canonical = colors
        .iter()
        .map(|color| map.get(color).map(|entry| entry.index))
        .collect::<Result<Vec<_>>>()?;
    let scores: Vec<_> = canonical.iter().map(|&id| run_counts[id]).collect();
    let samples = samples(&indices, width, height);
    let order = select_order(
        colors, default, &canonical, &scores, &samples, width, height,
    );
    let remap = remap_order(&order, &canonical);
    for id in &mut indices {
        *id = remap.get(*id as usize).copied().unwrap_or(*id);
    }
    Ok((order, indices))
}

fn select_order(
    colors: &[Vec<i32>],
    default: Vec<usize>,
    canonical: &[usize],
    scores: &[usize],
    samples: &[Sample],
    width: usize,
    height: usize,
) -> Vec<usize> {
    let baseline = match score(&default, colors, canonical, samples) {
        Ok(score) => score,
        Err(error) => {
            // A heuristic kernel's population/extra-bit limits must not become
            // new input restrictions on the actual lossless encoder.
            let _decision = crate::profile::scope("palette.order.screen.default");
            crate::profile::note("palette.screen.cost_unavailable", error);
            return default;
        }
    };
    let mut candidates = vec![("default", default, baseline)];
    for (name, hot) in [
        ("runs", None),
        ("lexical", Some(0)),
        ("hot32", Some(32)),
        ("hot128", Some(128)),
    ] {
        let candidate = runs::rank(colors, scores, hot);
        if candidates.iter().any(|(_, order, _)| *order == candidate) {
            continue;
        }
        let score = match score(&candidate, colors, canonical, samples) {
            Ok(score) => score,
            Err(error) => {
                crate::profile::note("palette.screen.cost_unavailable", error);
                continue;
            }
        };
        candidates.push((name, candidate, score));
    }
    let mut selected = 0;
    for (i, (name, _, score)) in candidates.iter().enumerate() {
        crate::profile::note(
            "palette.screen.cost",
            format_args!(
                "size={}x{} colors={} order={} bits={:.1} table={:.1} folds={:?}",
                width,
                height,
                colors.len(),
                name,
                score.total,
                score.table,
                score.folds
            ),
        );
        if admitted(*score, baseline) && score.total < candidates[selected].2.total {
            selected = i;
        }
    }
    let marker = match candidates[selected].0 {
        "runs" => "palette.order.screen.runs",
        "lexical" => "palette.order.screen.lexical",
        "hot32" => "palette.order.screen.hot32",
        "hot128" => "palette.order.screen.hot128",
        _ => "palette.order.screen.default",
    };
    let _decision = crate::profile::scope(marker);
    crate::profile::note(
        "palette.screen.selected",
        format_args!(
            "size={}x{} colors={} order={} estimated_ratio={:.6}",
            width,
            height,
            colors.len(),
            candidates[selected].0,
            candidates[selected].2.total / baseline.total.max(1.0)
        ),
    );
    candidates.swap_remove(selected).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_indices_match_final_dictionary_for_aliases_runs_and_implicit_colors() {
        let colors = vec![vec![4, -2, 7, 0], vec![8, 0, 1, 255], vec![4, -2, 7, 0]];
        let implicit = vec![0; 4];
        let keys = [&colors[0], &implicit, &colors[1]];
        // One run spans a scanline and all three colors recur in later rows.
        let ids = [0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 0, 1];
        let planes: Vec<_> = (0..4)
            .map(|c| ids.iter().map(|&id| keys[id][c]).collect())
            .collect();
        for mapping in [
            super::super::PaletteMapping::Tree,
            super::super::PaletteMapping::Flat,
        ] {
            let mut map = PaletteMap::new(4, mapping);
            map.get(&implicit).unwrap().index = 14;
            let (chosen, indices) = order(&colors, &[3, 3, 3], &mut map, &planes, 4, 3).unwrap();
            for (i, &original) in chosen.iter().enumerate() {
                map.get(&colors[original]).unwrap().index = i;
            }
            let expected: Vec<_> = ids
                .iter()
                .map(|&id| map.get(keys[id]).unwrap().index as i32)
                .collect();
            assert_eq!(indices, expected);
        }
    }

    #[test]
    fn admission_requires_consistent_gain_and_finite_nonzero_baseline() {
        let baseline = Score {
            total: 100.0,
            folds: [Some(90.0), Some(110.0)],
            table: 0.0,
        };
        assert!(admitted(
            Score {
                total: 90.0,
                folds: [Some(80.0), Some(100.0)],
                table: 0.0
            },
            baseline
        ));
        assert!(!admitted(
            Score {
                total: 95.0,
                folds: [Some(80.0), Some(110.0)],
                table: 0.0
            },
            baseline
        ));
        assert!(!admitted(baseline, baseline));
        assert!(!admitted(
            Score {
                total: f64::NAN,
                ..baseline
            },
            baseline
        ));
        assert!(!admitted(
            Score {
                total: 0.0,
                ..baseline
            },
            Score {
                total: 0.0,
                ..baseline
            }
        ));
    }

    #[test]
    fn sampling_is_bounded_contiguous_and_partitions_the_population() {
        let (width, height) = (277, 291);
        let samples = samples(&vec![5; width * height], width, height);
        assert_eq!(samples.len(), GRID * GRID);
        assert_eq!(
            samples.iter().map(|s| s.population).sum::<usize>(),
            width * height
        );
        assert!(samples.iter().map(|s| s.ids.len()).sum::<usize>() <= GRID * GRID * TILE * TILE);
        assert!(samples.iter().all(|s| s.ids.iter().all(|&id| id == 5)));
        assert!(samples.iter().any(|s| s.fold == 0));
        assert!(samples.iter().any(|s| s.fold == 1));
    }
}
