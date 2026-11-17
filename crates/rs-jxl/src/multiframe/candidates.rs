//! Shared geometric candidate generation and work ranking for analysis and encoding.
use super::{
    AutoDiffReport, AutoOptions, Rect, Reference, Strategy, diff::DiffMap, lookahead::Lookahead,
    partition, references::ReferenceSlots, regions,
};
use crate::{Error, Image, Result};
use std::sync::Arc;

pub(super) struct Candidate {
    pub(super) strategy: Strategy,
    pub(super) tile_size: Option<u32>,
    pub(super) reference_index: Option<usize>,
    pub(super) diff: Option<Arc<DiffMap>>,
    pub(super) source: u8,
    pub(super) rects: Vec<Rect>,
    pub(super) mask: bool,
}

impl Candidate {
    // Ranking only. No candidate is discarded using this estimate.
    pub(super) fn estimated_work(&self) -> u64 {
        let area = self.rects.iter().fold(0u64, |sum, rect| {
            sum.saturating_add(u64::from(rect.width) * u64::from(rect.height))
        });
        // Sparse mask contents are cheaper, but materialization/transforms still
        // visit the entire bbox. Keep a scan-cost floor instead of treating that
        // work as almost free. This weight ranks trials, never bounds their bytes.
        let pixels = self.diff.as_ref().map_or(area, |diff| {
            diff.changed_pixels().saturating_add(area.div_ceil(8))
        });
        pixels.saturating_add((self.rects.len() as u64).saturating_mul(64))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn candidates(
    images: &[Image],
    index: usize,
    auto: &AutoOptions,
    bytes_per_pixel: f64,
    recent_full_bytes_per_pixel: Option<f64>,
    differences: &mut Vec<AutoDiffReport>,
    slots: &ReferenceSlots,
    lookahead: &Lookahead,
) -> Result<Vec<Candidate>> {
    let mut maps = Vec::new();
    if index != 0
        && auto.reference != Reference::Full
        && !images[index]
            .metadata()
            .extra_channels
            .iter()
            .any(|e| e.dim_shift != 0)
    {
        let sources = if auto.retained_reference {
            slots.sources(index)
        } else {
            reference_indices(index, auto.reference)
                .into_iter()
                .map(|reference| (reference, if reference == 0 { 1 } else { 2 }))
                .collect()
        };
        for (reference, source) in sources {
            let diff = match lookahead.map(reference, index) {
                Some(diff) => diff,
                None => Arc::new(DiffMap::between(&images[reference], &images[index])?),
            };
            differences.push(AutoDiffReport {
                reference,
                source,
                changed_pixels: diff.changed_pixels(),
                bounds: diff.bounds(),
            });
            maps.push((reference, source, diff));
        }
    }
    let mut result =
        candidates_from_references(&images[index], index, auto, bytes_per_pixel, &maps)?;
    extend_recent_rate(
        index,
        auto,
        bytes_per_pixel,
        recent_full_bytes_per_pixel,
        &maps,
        &mut result,
    )?;
    Ok(result)
}

// A winning complete Full frame supplies a second proposal cost, never a byte
// bound. Preserve the anchor and every original/extra-grid candidate. A 2x
// change in either direction can alter which neighboring regions should merge.
fn alternate_rate(anchor: f64, recent: Option<f64>) -> Option<f64> {
    recent.filter(|&recent| {
        anchor.is_finite()
            && anchor > 0.0
            && recent.is_finite()
            && recent > 0.0
            && (recent / anchor >= 2.0 || anchor / recent >= 2.0)
    })
}

fn extend_recent_rate(
    index: usize,
    auto: &AutoOptions,
    anchor: f64,
    recent: Option<f64>,
    maps: &[(usize, u8, Arc<DiffMap>)],
    result: &mut Vec<Candidate>,
) -> Result<()> {
    if !auto.adaptive_rate_feedback || !auto.allow_multi_rect || !auto.adaptive_rectangles {
        return Ok(());
    }
    let Some(rate) = alternate_rate(anchor, recent) else {
        return Ok(());
    };
    for &(reference, source, ref diff) in maps {
        let Some(rects) = partition::propose(diff, auto.tile_size, auto.max_rectangles, rate)?
        else {
            continue;
        };
        if rects.len() <= 1
            || result.iter().any(|candidate| {
                !candidate.mask && candidate.source == source && candidate.rects == rects
            })
        {
            continue;
        }
        let first = reference == 0 && (index != 1 || auto.reference == Reference::First);
        let previous = reference + 1 == index;
        result
            .try_reserve(1)
            .map_err(|_| Error::Limit("adaptive rate candidate allocation"))?;
        result.push(Candidate {
            strategy: if first {
                Strategy::MergedFirst
            } else if previous {
                Strategy::MergedPrevious
            } else {
                Strategy::MergedRetained
            },
            tile_size: Some(auto.tile_size),
            reference_index: Some(reference),
            diff: None,
            source,
            rects,
            mask: false,
        });
    }
    Ok(())
}

pub(super) fn reference_indices(index: usize, reference: Reference) -> Vec<usize> {
    if index == 0 || reference == Reference::Full {
        return Vec::new();
    }
    match reference {
        Reference::Previous => vec![index - 1],
        Reference::First => vec![0],
        _ if index == 1 => vec![0],
        _ => vec![index - 1, 0],
    }
}

/// Shared geometry proposal stage. Maps are computed once per reference pair;
/// analysis never materializes cropped images or starts an encode trial.
pub(super) fn candidates_from_maps(
    current: &Image,
    index: usize,
    auto: &AutoOptions,
    bytes_per_pixel: f64,
    maps: &[(usize, Arc<DiffMap>)],
) -> Result<Vec<Candidate>> {
    let maps: Vec<_> = maps
        .iter()
        .map(|(index, diff)| (*index, if *index == 0 { 1 } else { 2 }, Arc::clone(diff)))
        .collect();
    candidates_from_references(current, index, auto, bytes_per_pixel, &maps)
}

fn candidates_from_references(
    current: &Image,
    index: usize,
    auto: &AutoOptions,
    bytes_per_pixel: f64,
    maps: &[(usize, u8, Arc<DiffMap>)],
) -> Result<Vec<Candidate>> {
    let full = Rect {
        x: 0,
        y: 0,
        width: current.width(),
        height: current.height(),
    };
    let mut result = vec![Candidate {
        strategy: Strategy::Full,
        tile_size: None,
        reference_index: None,
        diff: None,
        source: 0,
        rects: vec![full],
        mask: false,
    }];
    if index == 0
        || auto.reference == Reference::Full
        || current
            .metadata()
            .extra_channels
            .iter()
            .any(|e| e.dim_shift != 0)
    {
        return Ok(result);
    }
    let mut merged = Vec::new();
    for &(reference, source, ref diff) in maps {
        let first = reference == 0 && (index != 1 || auto.reference == Reference::First);
        let previous = reference + 1 == index;
        let bounds = diff.bounds();
        let rect = bounds.unwrap_or(Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });
        // A full-size REPLACE crop is the full candidate already tested.
        if rect != full {
            result.push(Candidate {
                tile_size: None,
                strategy: if first {
                    Strategy::CropFirst
                } else if previous {
                    Strategy::CropPrevious
                } else {
                    Strategy::CropRetained
                },
                reference_index: Some(reference),
                diff: None,
                source,
                rects: vec![rect],
                mask: false,
            });
        }
        if bounds.is_none() {
            continue;
        }
        if auto.allow_mask && diff.mask_eligible() {
            result.push(Candidate {
                tile_size: None,
                strategy: if first {
                    Strategy::MaskFirst
                } else if previous {
                    Strategy::MaskPrevious
                } else {
                    Strategy::MaskRetained
                },
                reference_index: Some(reference),
                diff: Some(Arc::clone(diff)),
                source,
                rects: vec![rect],
                mask: true,
            });
        }
        if auto.allow_multi_rect
            && let Some(rects) = regions::tiles_from_map(diff, auto.tile_size, auto.max_rectangles)?
            && rects.len() > 1
        {
            result.push(Candidate {
                tile_size: Some(auto.tile_size),
                strategy: if first {
                    Strategy::TilesFirst
                } else if previous {
                    Strategy::TilesPrevious
                } else {
                    Strategy::TilesRetained
                },
                reference_index: Some(reference),
                diff: None,
                source,
                rects,
                mask: false,
            });
        }
        if auto.allow_multi_rect
            && auto.adaptive_rectangles
            && let Some(rects) =
                partition::propose(diff, auto.tile_size, auto.max_rectangles, bytes_per_pixel)?
            && rects.len() > 1
            && !result.iter().any(|candidate| {
                !candidate.mask && candidate.source == source && candidate.rects == rects
            })
        {
            merged.push(Candidate {
                tile_size: Some(auto.tile_size),
                strategy: if first {
                    Strategy::MergedFirst
                } else if previous {
                    Strategy::MergedPrevious
                } else {
                    Strategy::MergedRetained
                },
                reference_index: Some(reference),
                diff: None,
                source,
                rects,
                mask: false,
            });
        }
    }
    // Keep every legacy candidate ahead of new proposals for equal-size ties.
    result.extend(merged);
    super::scales::extend(index, auto, bytes_per_pixel, maps, &mut result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout};

    fn sparse_changes() -> (Image, Arc<DiffMap>) {
        let previous = Image::new(
            256,
            256,
            PixelLayout::Gray,
            PixelData::U8(vec![0; 256 * 256]),
        )
        .unwrap();
        let mut pixels = vec![0; 256 * 256];
        for y in 0..8 {
            for x in 0..8 {
                pixels[(y * 32 + 1) * 256 + x * 32 + 1] = 1;
            }
        }
        let current = Image::new(256, 256, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
        let map = Arc::new(DiffMap::between(&previous, &current).unwrap());
        (current, map)
    }

    #[test]
    fn recent_rate_requires_a_finite_positive_twofold_change() {
        for recent in [
            None,
            Some(0.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(-1.0),
            Some(0.5001),
            Some(1.0),
            Some(1.9999),
        ] {
            assert_eq!(alternate_rate(1.0, recent), None);
        }
        assert_eq!(alternate_rate(1.0, Some(2.0)), Some(2.0));
        assert_eq!(alternate_rate(1.0, Some(0.5)), Some(0.5));
        for anchor in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(alternate_rate(anchor, Some(2.0)), None);
        }
    }

    #[test]
    fn recent_rate_appends_unique_base_plans_after_all_existing_grids() {
        let (current, map) = sparse_changes();
        let maps = [(0, 1, Arc::clone(&map)), (1, 2, map)];
        let options = AutoOptions {
            adaptive_rate_feedback: true,
            allow_mask: false,
            extra_tile_sizes: vec![8, 16],
            ..Default::default()
        };
        let anchor = 0.001;
        let mut result = candidates_from_references(&current, 2, &options, anchor, &maps).unwrap();
        let original: Vec<_> = result
            .iter()
            .map(|c| {
                (
                    c.strategy,
                    c.tile_size,
                    c.reference_index,
                    c.source,
                    c.rects.clone(),
                    c.mask,
                )
            })
            .collect();
        let old_len = result.len();
        extend_recent_rate(2, &options, anchor, Some(3.0), &maps, &mut result).unwrap();
        assert_eq!(result.len(), old_len + 2);
        let retained: Vec<_> = result[..old_len]
            .iter()
            .map(|c| {
                (
                    c.strategy,
                    c.tile_size,
                    c.reference_index,
                    c.source,
                    c.rects.clone(),
                    c.mask,
                )
            })
            .collect();
        assert_eq!(retained, original);
        for (candidate, (strategy, source)) in result[old_len..]
            .iter()
            .zip([(Strategy::MergedFirst, 1), (Strategy::MergedPrevious, 2)])
        {
            assert_eq!(candidate.strategy, strategy);
            assert_eq!(candidate.source, source);
            assert_eq!(candidate.tile_size, Some(options.tile_size));
            assert_eq!(candidate.rects.len(), options.max_rectangles);
        }
        let new_len = result.len();
        extend_recent_rate(2, &options, anchor, Some(3.0), &maps, &mut result).unwrap();
        assert_eq!(
            result.len(),
            new_len,
            "identical source/rectangle plans are tried once"
        );
    }

    #[test]
    fn disabled_and_equivalent_proposals_do_not_add_trials() {
        let (current, map) = sparse_changes();
        let maps = [(1, 2, map)];
        for options in [
            AutoOptions {
                adaptive_rate_feedback: false,
                ..Default::default()
            },
            AutoOptions {
                adaptive_rate_feedback: true,
                adaptive_rectangles: false,
                ..Default::default()
            },
            AutoOptions {
                adaptive_rate_feedback: true,
                allow_multi_rect: false,
                ..Default::default()
            },
            AutoOptions {
                adaptive_rate_feedback: true,
                max_rectangles: 1,
                ..Default::default()
            },
        ] {
            let mut result =
                candidates_from_references(&current, 2, &options, 0.001, &maps).unwrap();
            let count = result.len();
            extend_recent_rate(2, &options, 0.001, Some(3.0), &maps, &mut result).unwrap();
            assert_eq!(result.len(), count);
        }
        let options = AutoOptions {
            adaptive_rate_feedback: true,
            ..Default::default()
        };
        let mut result = candidates_from_references(&current, 2, &options, 3.0, &maps).unwrap();
        let count = result.len();
        extend_recent_rate(2, &options, 3.0, Some(6.0), &maps, &mut result).unwrap();
        assert_eq!(
            result.len(),
            count,
            "same partition at another rate is redundant"
        );
        extend_recent_rate(2, &options, 3.0, Some(4.0), &maps, &mut result).unwrap();
        assert_eq!(
            result.len(),
            count,
            "nearby rates preserve the candidate set"
        );
    }
}
