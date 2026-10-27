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
            history_window: 4,
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
                history: vec![HistoryAnalysis {
                    reference_index: 1,
                    stats: DiffStatistics {
                        bounds: None,
                        changed_pixels: 0,
                        changed_ratio: 0.0,
                        bbox_pixels: 0,
                        bbox_ratio: 0.0,
                        bbox_fill_ratio: 0.0,
                        mask_eligible: false,
                    },
                    improves_static_bbox: true,
                }],
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
                    tile_size: Some(16),
                    reference_index: Some(0),
                    source: 1,
                    rectangles: vec![region],
                    layer_pixels: 30,
                    layer_area_ratio: 0.25,
                    layer_count: 1,
                }],
                geometry: GeometryDiagnostics {
                    best_area_candidate_index: 0,
                    candidate_count: 1,
                    trial_pixels: 30,
                    trial_layers: 1,
                    unmasked_rectangles: RectangleWorkload {
                        total_pixels: 30,
                        total_layers: 1,
                        unique_pixels: 30,
                        unique_rectangles: 1,
                        repeated_pixels: 0,
                        repeated_layers: 0,
                    },
                },
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
            geometry: GeometrySummary {
                layer_pixels: 30,
                layer_area_ratio: 0.25,
                layer_count: 1,
                crop_frames: 1,
                multi_rect_frames: 0,
                suitability: Suitability::Strong,
                candidate_count: 1,
                trial_pixels: 30,
                trial_layers: 1,
                unmasked_rectangles: RectangleWorkload {
                    total_pixels: 30,
                    total_layers: 1,
                    unique_pixels: 30,
                    unique_rectangles: 1,
                    repeated_pixels: 0,
                    repeated_layers: 0,
                },
            },
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
                "rectangle_bytes_per_pixel": 2.5, "history_window": 4,
                "frames": [{
                    "index": 2, "previous": region, "first": null,
                    "previous_stats": {
                        "bounds": region, "changed_pixels": 3, "changed_ratio": 0.025,
                        "bbox_pixels": 30, "bbox_ratio": 0.25, "bbox_fill_ratio": 0.1,
                        "mask_eligible": true,
                    },
                    "first_stats": null,
                    "history": [{
                        "reference_index": 1,
                        "stats": {
                            "bounds": null, "changed_pixels": 0, "changed_ratio": 0.0,
                            "bbox_pixels": 0, "bbox_ratio": 0.0, "bbox_fill_ratio": 0.0,
                            "mask_eligible": false,
                        },
                        "improves_static_bbox": true,
                    }],
                    "plan": {
                        "reference_index": 0, "source": 1, "region": region,
                        "changed": null, "layer_pixels": 30, "layer_area_ratio": 0.25,
                    },
                    "candidates": [{
                        "strategy": "merged_first", "tile_size": 16,
                        "reference_index": 0, "source": 1,
                        "rectangles": [region], "layer_pixels": 30,
                        "layer_area_ratio": 0.25, "layer_count": 1,
                    }],
                    "geometry": {
                        "best_area_candidate_index": 0, "candidate_count": 1,
                        "trial_pixels": 30, "trial_layers": 1,
                        "unmasked_rectangles": {
                            "total_pixels": 30, "total_layers": 1,
                            "unique_pixels": 30, "unique_rectangles": 1,
                            "repeated_pixels": 0, "repeated_layers": 0,
                        },
                    },
                }],
                "geometry": {
                    "layer_pixels": 30, "layer_area_ratio": 0.25, "layer_count": 1,
                    "crop_frames": 1, "multi_rect_frames": 0, "suitability": "strong",
                    "candidate_count": 1, "trial_pixels": 30, "trial_layers": 1,
                    "unmasked_rectangles": {
                        "total_pixels": 30, "total_layers": 1,
                        "unique_pixels": 30, "unique_rectangles": 1,
                        "repeated_pixels": 0, "repeated_layers": 0,
                    },
                },
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
    fn execute_projects_history_and_geometry_without_returning_encoded_data() {
        let images: Vec<_> = (0..4)
            .map(|index| {
                json!({
                    "width": 4, "height": 2, "channels": 1, "dtype": "u8",
                    "offset": index * 8, "length": 8,
                })
            })
            .collect();
        let request: Request = serde_json::from_value(json!({
            "operation": "analyze_multiframe",
            "images": images,
            "analysis_options": {"history_window": 2, "extra_tile_sizes": [2, 8]},
        }))
        .unwrap();
        request.validate().unwrap();
        let data: Vec<_> = [0u8, 17, 29, 17]
            .into_iter()
            .flat_map(|value| [value; 8])
            .collect();
        let (report, output) = execute(&request, &data, None).unwrap();
        assert!(output.is_empty());
        assert_eq!(report["history_window"], 2);
        assert_eq!(report["frames"][0]["history"], json!([]));
        let history = &report["frames"][3]["history"][0];
        assert_eq!(history["reference_index"], 1);
        assert_eq!(history["stats"]["changed_pixels"], 0);
        assert_eq!(history["improves_static_bbox"], true);
        assert!(history.get("source").is_none()); // Logical opportunity, not a slot.
        for frame in report["frames"].as_array().unwrap() {
            assert_eq!(frame["geometry"]["best_area_candidate_index"], 0);
            assert_eq!(frame["geometry"]["trial_pixels"], 8);
            assert_eq!(frame["geometry"]["trial_layers"], 1);
            assert!(frame["candidates"][0]["tile_size"].is_null());
            assert!(frame["candidates"][0].get("encoded_bytes").is_none());
            assert!(frame["candidates"][0].get("estimated_bytes").is_none());
        }
        assert_eq!(report["geometry"]["layer_pixels"], 32);
        assert_eq!(report["geometry"]["layer_count"], 4);
        assert_eq!(report["geometry"]["suitability"], "limited");
        assert_eq!(
            report["geometry"]["unmasked_rectangles"],
            json!({
                "total_pixels": 32, "total_layers": 4,
                "unique_pixels": 32, "unique_rectangles": 4,
                "repeated_pixels": 0, "repeated_layers": 0,
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
            (Strategy::CropRetained, "crop_retained"),
            (Strategy::MaskPrevious, "mask_previous"),
            (Strategy::MaskFirst, "mask_first"),
            (Strategy::MaskRetained, "mask_retained"),
            (Strategy::TilesPrevious, "tiles_previous"),
            (Strategy::TilesFirst, "tiles_first"),
            (Strategy::TilesRetained, "tiles_retained"),
            (Strategy::MergedPrevious, "merged_previous"),
            (Strategy::MergedFirst, "merged_first"),
            (Strategy::MergedRetained, "merged_retained"),
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
