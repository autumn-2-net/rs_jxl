//! Geometric candidate work, without entropy coding or byte-size prediction.
use super::{FrameAnalysis, GeometryCandidate, Suitability, area, suitability};
use crate::{Error, Result, multiframe::Strategy};
use std::collections::BTreeSet;

/// Unmasked rectangle work within a single logical input's candidate set.
/// Aggregates sum per-input counts; equal rectangles in different inputs are
/// never treated as reusable. Patches and pruning can prevent actual reuse.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RectangleWorkload {
    pub total_pixels: u64,
    pub total_layers: usize,
    /// Sum of the areas of distinct (x, y, width, height) keys, not a pixel union.
    pub unique_pixels: u64,
    pub unique_rectangles: usize,
    /// Repeated occurrences after the first use; an upper bound on reusable work.
    pub repeated_pixels: u64,
    pub repeated_layers: usize,
}

impl RectangleWorkload {
    fn add(&mut self, other: &Self) -> Result<()> {
        add_pixels(&mut self.total_pixels, other.total_pixels)?;
        add_count(&mut self.total_layers, other.total_layers)?;
        add_pixels(&mut self.unique_pixels, other.unique_pixels)?;
        add_count(&mut self.unique_rectangles, other.unique_rectangles)?;
        add_pixels(&mut self.repeated_pixels, other.repeated_pixels)?;
        add_count(&mut self.repeated_layers, other.repeated_layers)
    }
}

#[derive(Debug)]
pub struct GeometryDiagnostics {
    /// Minimum (layer_pixels, layer_count, canonical candidate index).
    pub best_area_candidate_index: usize,
    pub candidate_count: usize,
    /// Submitted areas/layers across every candidate, including masks and repeats.
    /// This is hypothetical work before pruning, not executed encoding trials.
    pub trial_pixels: u64,
    pub trial_layers: usize,
    pub unmasked_rectangles: RectangleWorkload,
}

#[derive(Debug)]
pub struct GeometrySummary {
    /// Sum of each input's best-area candidate, including the initial full frame.
    pub layer_pixels: u64,
    pub layer_area_ratio: f64,
    pub layer_count: usize,
    /// Subsequent inputs whose selected area is smaller than a complete canvas.
    pub crop_frames: usize,
    pub multi_rect_frames: usize,
    pub suitability: Suitability,
    pub candidate_count: usize,
    pub trial_pixels: u64,
    pub trial_layers: usize,
    pub unmasked_rectangles: RectangleWorkload,
}

fn add_pixels(total: &mut u64, value: u64) -> Result<()> {
    *total = total
        .checked_add(value)
        .ok_or(Error::Limit("analysis geometry pixel area"))?;
    Ok(())
}

fn add_count(total: &mut usize, value: usize) -> Result<()> {
    *total = total
        .checked_add(value)
        .ok_or(Error::Limit("analysis geometry layer or candidate count"))?;
    Ok(())
}

pub(super) fn diagnostics(candidates: &[GeometryCandidate]) -> Result<GeometryDiagnostics> {
    let best_area_candidate_index = candidates
        .iter()
        .enumerate()
        .min_by_key(|&(index, candidate)| (candidate.layer_pixels, candidate.layer_count, index))
        .map(|(index, _)| index)
        .ok_or(Error::InvalidInput(
            "analysis requires a full-frame candidate".into(),
        ))?;
    let mut result = GeometryDiagnostics {
        best_area_candidate_index,
        candidate_count: candidates.len(),
        trial_pixels: 0,
        trial_layers: 0,
        unmasked_rectangles: RectangleWorkload::default(),
    };
    let mut unique = BTreeSet::new();
    for candidate in candidates {
        add_pixels(&mut result.trial_pixels, candidate.layer_pixels)?;
        add_count(&mut result.trial_layers, candidate.layer_count)?;
        if matches!(
            candidate.strategy,
            Strategy::MaskPrevious | Strategy::MaskFirst | Strategy::MaskRetained
        ) {
            continue;
        }
        let work = &mut result.unmasked_rectangles;
        for &rect in &candidate.rectangles {
            let pixels = area(rect);
            add_pixels(&mut work.total_pixels, pixels)?;
            add_count(&mut work.total_layers, 1)?;
            if unique.insert((rect.x, rect.y, rect.width, rect.height)) {
                add_pixels(&mut work.unique_pixels, pixels)?;
                add_count(&mut work.unique_rectangles, 1)?;
            } else {
                add_pixels(&mut work.repeated_pixels, pixels)?;
                add_count(&mut work.repeated_layers, 1)?;
            }
        }
    }
    Ok(result)
}

pub(super) fn summarize(
    frames: &[FrameAnalysis],
    pixels: u64,
    full_pixels: u64,
    crop_supported: bool,
) -> Result<GeometrySummary> {
    let mut result = GeometrySummary {
        layer_pixels: 0,
        layer_area_ratio: 0.0,
        layer_count: 0,
        crop_frames: 0,
        multi_rect_frames: 0,
        suitability: Suitability::SingleFrame,
        candidate_count: 0,
        trial_pixels: 0,
        trial_layers: 0,
        unmasked_rectangles: RectangleWorkload::default(),
    };
    for frame in frames {
        let geometry = &frame.geometry;
        let best = &frame.candidates[geometry.best_area_candidate_index];
        add_pixels(&mut result.layer_pixels, best.layer_pixels)?;
        add_count(&mut result.layer_count, best.layer_count)?;
        add_count(
            &mut result.crop_frames,
            usize::from(best.layer_pixels < pixels),
        )?;
        add_count(
            &mut result.multi_rect_frames,
            usize::from(best.layer_count > 1),
        )?;
        add_count(&mut result.candidate_count, geometry.candidate_count)?;
        add_pixels(&mut result.trial_pixels, geometry.trial_pixels)?;
        add_count(&mut result.trial_layers, geometry.trial_layers)?;
        result
            .unmasked_rectangles
            .add(&geometry.unmasked_rectangles)?;
    }
    result.layer_area_ratio = result.layer_pixels as f64 / full_pixels as f64;
    result.suitability = suitability(
        frames.len(),
        crop_supported,
        result.layer_pixels,
        pixels,
        full_pixels,
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiframe::Rect;

    #[test]
    fn best_area_prefers_fewer_layers_then_canonical_index() {
        let one = Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        let two = Rect {
            x: 0,
            y: 0,
            width: 2,
            height: 1,
        };
        let candidate = |rectangles: Vec<Rect>, strategy| GeometryCandidate {
            strategy,
            tile_size: None,
            reference_index: None,
            source: 0,
            layer_pixels: rectangles.iter().copied().map(area).sum(),
            layer_area_ratio: 1.0,
            layer_count: rectangles.len(),
            rectangles,
        };
        let candidates = [
            candidate(vec![one, Rect { x: 1, ..one }], Strategy::TilesPrevious),
            candidate(vec![two], Strategy::CropPrevious),
            candidate(vec![two], Strategy::CropFirst),
        ];
        let result = diagnostics(&candidates).unwrap();
        assert_eq!(result.best_area_candidate_index, 1);
        assert_eq!(result.trial_pixels, 6);
        assert_eq!(result.trial_layers, 4);
        assert_eq!(result.unmasked_rectangles.unique_pixels, 4); // Not the union (2).
        assert_eq!(result.unmasked_rectangles.repeated_pixels, 2);
        assert!(diagnostics(&[]).is_err());
    }

    #[test]
    fn workload_counters_reject_overflow() {
        let mut pixels = u64::MAX;
        let mut layers = usize::MAX;
        assert!(add_pixels(&mut pixels, 1).is_err());
        assert!(add_count(&mut layers, 1).is_err());
        let mut work = RectangleWorkload {
            repeated_pixels: u64::MAX,
            ..Default::default()
        };
        assert!(
            work.add(&RectangleWorkload {
                repeated_pixels: 1,
                ..Default::default()
            })
            .is_err()
        );
    }
}
