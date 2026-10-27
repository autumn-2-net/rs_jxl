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
