//! Experimental palette-order admission before MA learning. Costs are estimates,
//! not an encoded-size guarantee; ambiguous results retain the source default.

use super::{PaletteMap, luma, runs};
use crate::Result;
use crate::modular::cost::{self, ChannelView};

const TILE: usize = 64;
const GRID: usize = 4;
const MIN_GAIN: f64 = 0.03;

struct Sample {
    ids: Vec<usize>,
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

fn samples(
    map: &mut PaletteMap,
    planes: &[Vec<i32>],
    width: usize,
    height: usize,
) -> Result<Vec<Sample>> {
    let _scope = crate::profile::scope("palette.order.screen.sample");
    let nx = width.div_ceil(TILE).min(GRID);
    let ny = height.div_ceil(TILE).min(GRID);
    let mut result = Vec::with_capacity(nx * ny);
    let mut color = vec![0; planes.len()];
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
                for xx in x..x + tw {
                    for (value, plane) in color.iter_mut().zip(planes) {
                        *value = plane[yy * width + xx];
                    }
                    ids.push(map.get(&color)?.index);
                }
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
    Ok(result)
}

fn score(
    order: &[usize],
    colors: &[Vec<i32>],
    canonical: &[usize],
    samples: &[Sample],
) -> Result<Score> {
    let _scope = crate::profile::scope("palette.order.screen.cost");
    let n = order.len();
    let mut remap = vec![0i32; n];
    for (index, &original) in order.iter().enumerate() {
        // Duplicate implicit promotions use the last assignment, exactly like
        // the final palette mapping. Non-promoted implicit IDs are unchanged.
        remap[canonical[original]] = index as i32;
    }
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
                .map(|&id| if id < n { remap[id] } else { id as i32 }),
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
) -> Result<Vec<usize>> {
    let _scope = crate::profile::scope("palette.order.screen");
    let default = default_order(colors, frequencies);
    if colors.len() < 2 {
        let _decision = crate::profile::scope("palette.order.screen.default");
        return Ok(default);
    }
    let scores = runs::scores(colors, map, planes)?;
    let canonical = colors
        .iter()
        .map(|color| map.get(color).map(|entry| entry.index))
        .collect::<Result<Vec<_>>>()?;
    let samples = samples(map, planes, width, height)?;
    let baseline = match score(&default, colors, &canonical, &samples) {
        Ok(score) => score,
        Err(error) => {
            // A heuristic kernel's population/extra-bit limits must not become
            // new input restrictions on the actual lossless encoder.
            let _decision = crate::profile::scope("palette.order.screen.default");
            crate::profile::note("palette.screen.cost_unavailable", error);
            return Ok(default);
        }
    };
    let mut candidates = vec![("default", default, baseline)];
    for (name, hot) in [
        ("runs", None),
        ("lexical", Some(0)),
        ("hot32", Some(32)),
        ("hot128", Some(128)),
    ] {
        let candidate = runs::rank(colors, &scores, hot);
        if candidates.iter().any(|(_, order, _)| *order == candidate) {
            continue;
        }
        let score = match score(&candidate, colors, &canonical, &samples) {
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
    Ok(candidates.swap_remove(selected).1)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let planes = vec![vec![123; width * height]; 3];
        let mut map = PaletteMap::new(3, super::super::PaletteMapping::Flat);
        map.get(&[123; 3]).unwrap().index = 5;
        let samples = samples(&mut map, &planes, width, height).unwrap();
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
