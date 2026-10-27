//! Shared geometric candidate generation and work ranking for analysis and encoding.
use super::{
    AutoDiffReport, AutoOptions, Rect, Reference, Strategy, diff::DiffMap, lookahead::Lookahead,
    partition, references::ReferenceSlots, regions,
};
use crate::{Image, Result};
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

pub(super) fn candidates(
    images: &[Image],
    index: usize,
    auto: &AutoOptions,
    bytes_per_pixel: f64,
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
    candidates_from_references(&images[index], index, auto, bytes_per_pixel, &maps)
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
