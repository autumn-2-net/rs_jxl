//! Thin JSON adapter for native geometric multiframe analysis; no trial encodes.
use crate::{Result, message, pixels, request::Request};
use rs_jxl::{
    codec::Encoder,
    multiframe::{
        self, AnalysisReport, CropAnalysis, DiffStatistics, FrameAnalysis, GeometryCandidate, Rect,
        Reference, ReferenceSummary, Strategy, Suitability,
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
        Strategy::MaskPrevious => "mask_previous",
        Strategy::MaskFirst => "mask_first",
        Strategy::TilesPrevious => "tiles_previous",
        Strategy::TilesFirst => "tiles_first",
        Strategy::MergedPrevious => "merged_previous",
        Strategy::MergedFirst => "merged_first",
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
        "reference_index": value.reference_index,
        "source": value.source,
        "rectangles": value.rectangles.iter().copied().map(rect).collect::<Vec<_>>(),
        "layer_pixels": value.layer_pixels,
        "layer_area_ratio": value.layer_area_ratio,
        "layer_count": value.layer_count,
    })
}

fn frame(value: &FrameAnalysis) -> Value {
    json!({
        "index": value.index,
        "previous": bounds(value.previous),
        "first": bounds(value.first),
        "previous_stats": value.previous_stats.as_ref().map(statistics),
        "first_stats": value.first_stats.as_ref().map(statistics),
        "plan": plan(&value.plan),
        "candidates": value.candidates.iter().map(candidate).collect::<Vec<_>>(),
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
        "frames": value.frames.iter().map(frame).collect::<Vec<_>>(),
        "reference_modes": value.reference_modes.iter().map(summary).collect::<Vec<_>>(),
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
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn report_preserves_native_fields_nulls_and_seconds_without_size_estimates() {
        let region = Rect {
            x: 3,
            y: 4,
            width: 5,
            height: 6,
        };
        let analysis = AnalysisReport {
            width: 10,
            height: 12,
            channels: 4,
            frame_count: 1,
            reference: Reference::First,
            crop_supported: true,
            rectangle_bytes_per_pixel: 2.5,
            frames: vec![FrameAnalysis {
                index: 2,
                previous: Some(region),
                first: None,
                previous_stats: Some(DiffStatistics {
                    bounds: Some(region),
                    changed_pixels: 3,
                    changed_ratio: 0.025,
                    bbox_pixels: 30,
                    bbox_ratio: 0.25,
                    bbox_fill_ratio: 0.1,
                    mask_eligible: true,
                }),
                first_stats: None,
                plan: CropAnalysis {
                    reference_index: Some(0),
                    source: 1,
                    region,
                    changed: None,
                    layer_pixels: 30,
                    layer_area_ratio: 0.25,
                },
                candidates: vec![GeometryCandidate {
                    strategy: Strategy::MergedFirst,
                    reference_index: Some(0),
                    source: 1,
                    rectangles: vec![region],
                    layer_pixels: 30,
                    layer_area_ratio: 0.25,
                    layer_count: 1,
                }],
            }],
            reference_modes: vec![ReferenceSummary {
                reference: Reference::SmallestCrop,
                layer_pixels: 150,
                layer_area_ratio: 0.625,
                average_changed_ratio: 0.025,
                crop_frames: 1,
                identical_frames: 0,
                suitability: Suitability::Strong,
            }],
            best_geometric_reference: Reference::SmallestCrop,
            analysis_elapsed: Duration::from_millis(125),
        };
        let region = json!({"x": 3, "y": 4, "width": 5, "height": 6});
        assert_eq!(
            report(&analysis),
            json!({
                "operation": "analyze_multiframe",
                "width": 10, "height": 12, "channels": 4, "frame_count": 1,
                "reference": "first", "crop_supported": true,
                "rectangle_bytes_per_pixel": 2.5,
                "frames": [{
                    "index": 2, "previous": region, "first": null,
                    "previous_stats": {
                        "bounds": region, "changed_pixels": 3, "changed_ratio": 0.025,
                        "bbox_pixels": 30, "bbox_ratio": 0.25, "bbox_fill_ratio": 0.1,
                        "mask_eligible": true,
                    },
                    "first_stats": null,
                    "plan": {
                        "reference_index": 0, "source": 1, "region": region,
                        "changed": null, "layer_pixels": 30, "layer_area_ratio": 0.25,
                    },
                    "candidates": [{
                        "strategy": "merged_first", "reference_index": 0, "source": 1,
                        "rectangles": [region], "layer_pixels": 30,
                        "layer_area_ratio": 0.25, "layer_count": 1,
                    }],
                }],
                "reference_modes": [{
                    "reference": "smallest_crop", "layer_pixels": 150,
                    "layer_area_ratio": 0.625, "average_changed_ratio": 0.025,
                    "crop_frames": 1, "identical_frames": 0, "suitability": "strong",
                }],
                "best_geometric_reference": "smallest_crop", "analysis_seconds": 0.125,
            })
        );
    }

    #[test]
    fn analysis_enums_have_explicit_snake_case_wire_names() {
        for (value, expected) in [
            (Reference::Full, "full"),
            (Reference::Previous, "previous"),
            (Reference::First, "first"),
            (Reference::SmallestCrop, "smallest_crop"),
        ] {
            assert_eq!(reference(value), expected);
        }
        for (value, expected) in [
            (Strategy::Full, "full"),
            (Strategy::CropPrevious, "crop_previous"),
            (Strategy::CropFirst, "crop_first"),
            (Strategy::MaskPrevious, "mask_previous"),
            (Strategy::MaskFirst, "mask_first"),
            (Strategy::TilesPrevious, "tiles_previous"),
            (Strategy::TilesFirst, "tiles_first"),
            (Strategy::MergedPrevious, "merged_previous"),
            (Strategy::MergedFirst, "merged_first"),
        ] {
            assert_eq!(strategy(value), expected);
        }
        for (value, expected) in [
            (Suitability::SingleFrame, "single_frame"),
            (Suitability::Strong, "strong"),
            (Suitability::Moderate, "moderate"),
            (Suitability::Limited, "limited"),
            (Suitability::FullFramesRequired, "full_frames_required"),
        ] {
            assert_eq!(suitability(value), expected);
        }
    }
}
