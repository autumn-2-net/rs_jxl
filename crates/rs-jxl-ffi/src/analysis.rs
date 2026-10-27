//! Thin JSON adapter for native geometric multiframe analysis; no trial encodes.
use crate::{Result, message, pixels, request::Request};
use rs_jxl::{
    codec::Encoder,
    multiframe::{
        self, AnalysisReport, CropAnalysis, DiffStatistics, FrameAnalysis, GeometryCandidate,
        GeometryDiagnostics, GeometrySummary, HistoryAnalysis, Rect, RectangleWorkload, Reference,
        ReferenceSummary, Strategy, Suitability,
    },
};
use serde_json::{Value, json};

fn reference(value: Reference) -> &'static str {
    match value {
        Reference::Full => "full",
        Reference::Previous => "previous",
        Reference::First => "first",
        Reference::SmallestCrop => "smallest_crop",
    }
}

fn strategy(value: Strategy) -> &'static str {
    match value {
        Strategy::Full => "full",
        Strategy::CropPrevious => "crop_previous",
        Strategy::CropFirst => "crop_first",
        Strategy::CropRetained => "crop_retained",
        Strategy::MaskPrevious => "mask_previous",
        Strategy::MaskFirst => "mask_first",
        Strategy::MaskRetained => "mask_retained",
        Strategy::TilesPrevious => "tiles_previous",
        Strategy::TilesFirst => "tiles_first",
        Strategy::TilesRetained => "tiles_retained",
        Strategy::MergedPrevious => "merged_previous",
        Strategy::MergedFirst => "merged_first",
        Strategy::MergedRetained => "merged_retained",
    }
}

fn suitability(value: Suitability) -> &'static str {
    match value {
        Suitability::SingleFrame => "single_frame",
        Suitability::Strong => "strong",
        Suitability::Moderate => "moderate",
        Suitability::Limited => "limited",
        Suitability::FullFramesRequired => "full_frames_required",
    }
}

fn rect(value: Rect) -> Value {
    json!({"x": value.x, "y": value.y, "width": value.width, "height": value.height})
}

fn bounds(value: Option<Rect>) -> Value {
    value.map_or(Value::Null, rect)
}

fn statistics(value: &DiffStatistics) -> Value {
    json!({
        "bounds": bounds(value.bounds),
        "changed_pixels": value.changed_pixels,
        "changed_ratio": value.changed_ratio,
        "bbox_pixels": value.bbox_pixels,
        "bbox_ratio": value.bbox_ratio,
        "bbox_fill_ratio": value.bbox_fill_ratio,
        "mask_eligible": value.mask_eligible,
    })
}

fn plan(value: &CropAnalysis) -> Value {
    json!({
        "reference_index": value.reference_index,
        "source": value.source,
        "region": rect(value.region),
        "changed": bounds(value.changed),
        "layer_pixels": value.layer_pixels,
        "layer_area_ratio": value.layer_area_ratio,
    })
}

fn candidate(value: &GeometryCandidate) -> Value {
    json!({
        "strategy": strategy(value.strategy),
        "tile_size": value.tile_size,
        "reference_index": value.reference_index,
        "source": value.source,
        "rectangles": value.rectangles.iter().copied().map(rect).collect::<Vec<_>>(),
        "layer_pixels": value.layer_pixels,
        "layer_area_ratio": value.layer_area_ratio,
        "layer_count": value.layer_count,
    })
}

fn history(value: &HistoryAnalysis) -> Value {
    json!({
        "reference_index": value.reference_index,
        "stats": statistics(&value.stats),
        "improves_static_bbox": value.improves_static_bbox,
    })
}

fn rectangle_workload(value: &RectangleWorkload) -> Value {
    json!({
        "total_pixels": value.total_pixels,
        "total_layers": value.total_layers,
        "unique_pixels": value.unique_pixels,
        "unique_rectangles": value.unique_rectangles,
        "repeated_pixels": value.repeated_pixels,
        "repeated_layers": value.repeated_layers,
    })
}

fn geometry(value: &GeometryDiagnostics) -> Value {
    json!({
        "best_area_candidate_index": value.best_area_candidate_index,
        "candidate_count": value.candidate_count,
        "trial_pixels": value.trial_pixels,
        "trial_layers": value.trial_layers,
        "unmasked_rectangles": rectangle_workload(&value.unmasked_rectangles),
    })
}

fn geometry_summary(value: &GeometrySummary) -> Value {
    json!({
        "layer_pixels": value.layer_pixels,
        "layer_area_ratio": value.layer_area_ratio,
        "layer_count": value.layer_count,
        "crop_frames": value.crop_frames,
        "multi_rect_frames": value.multi_rect_frames,
        "suitability": suitability(value.suitability),
        "candidate_count": value.candidate_count,
        "trial_pixels": value.trial_pixels,
        "trial_layers": value.trial_layers,
        "unmasked_rectangles": rectangle_workload(&value.unmasked_rectangles),
    })
}

fn frame(value: &FrameAnalysis) -> Value {
    json!({
        "index": value.index,
        "previous": bounds(value.previous),
        "first": bounds(value.first),
        "previous_stats": value.previous_stats.as_ref().map(statistics),
        "first_stats": value.first_stats.as_ref().map(statistics),
        "history": value.history.iter().map(history).collect::<Vec<_>>(),
        "plan": plan(&value.plan),
        "candidates": value.candidates.iter().map(candidate).collect::<Vec<_>>(),
        "geometry": geometry(&value.geometry),
    })
}

fn summary(value: &ReferenceSummary) -> Value {
    json!({
        "reference": reference(value.reference),
        "layer_pixels": value.layer_pixels,
        "layer_area_ratio": value.layer_area_ratio,
        "average_changed_ratio": value.average_changed_ratio,
        "crop_frames": value.crop_frames,
        "identical_frames": value.identical_frames,
        "suitability": suitability(value.suitability),
    })
}

fn report(value: &AnalysisReport) -> Value {
    json!({
        "operation": "analyze_multiframe",
        "width": value.width,
        "height": value.height,
        "channels": value.channels,
        "frame_count": value.frame_count,
        "reference": reference(value.reference),
        "crop_supported": value.crop_supported,
        "rectangle_bytes_per_pixel": value.rectangle_bytes_per_pixel,
        "history_window": value.history_window,
        "frames": value.frames.iter().map(frame).collect::<Vec<_>>(),
        "reference_modes": value.reference_modes.iter().map(summary).collect::<Vec<_>>(),
        "geometry": geometry_summary(&value.geometry),
        "best_geometric_reference": reference(value.best_geometric_reference),
        "analysis_seconds": value.analysis_elapsed.as_secs_f64(),
    })
}

pub fn execute(
    request: &Request,
    data: &[u8],
    context: Option<&mut Encoder>,
) -> Result<(Value, Vec<u8>)> {
    let images = request
        .images()
        .iter()
        .map(|spec| {
            pixels::image(spec, data, request.options.strip_metadata.unwrap_or(false))
                .map(|(image, _)| image)
        })
        .collect::<Result<Vec<_>>>()?;
    let threads = request.options.threads.unwrap_or(1);
    let options = request.analysis_options.as_ref().map_or_else(
        || multiframe::AnalysisOptions {
            threads,
            ..Default::default()
        },
        |options| options.native(threads),
    );
    let analysis = match context {
        Some(encoder) => encoder.analyze_multiframe(&images, &options),
        None => multiframe::analyze(&images, &options),
    }
    .map_err(message)?;
    Ok((report(&analysis), Vec::new()))
}

#[cfg(test)]
#[path = "analysis/tests.rs"]
mod tests;
