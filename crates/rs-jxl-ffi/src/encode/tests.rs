use super::multiframe::auto_frame_report;
use rs_jxl::multiframe::{
    self, AutoCandidateReport, AutoDiffReport, AutoFrameReport, AutoPrunedReport, Strategy,
};
use serde_json::json;
use std::time::Duration;

#[test]
fn auto_report_keeps_native_statistics_and_separates_pruned_bounds() {
    let frame = AutoFrameReport {
        winner_index: 0,
        retained_reference: true,
        reference_lookahead: true,
        lookahead_comparisons: 3,
        reused_payloads: 2,
        reused_payload_pixels: 25,
        references_before: [None, Some(0), Some(1), None],
        references_after: [None, Some(2), Some(1), None],
        candidates: vec![AutoCandidateReport {
            strategy: Strategy::Full,
            tile_size: None,
            reference_index: None,
            source: 0,
            save_as_reference: 1,
            encoded_bytes: 80,
            encoded_frames: 1,
            trial_pixels: 100,
            elapsed: Duration::from_millis(125),
        }],
        pruned: vec![AutoPrunedReport {
            strategy: Strategy::MergedRetained,
            tile_size: Some(16),
            reference_index: Some(1),
            source: 2,
            save_as_reference: 1,
            lower_bound: 90,
            encoded_frames: 1,
            planned_frames: 3,
            trial_pixels: 30,
            elapsed: Duration::from_millis(250),
        }],
        analysis_elapsed: Duration::from_millis(500),
        differences: vec![
            AutoDiffReport {
                reference: 0,
                source: 1,
                changed_pixels: 0,
                bounds: None,
            },
            AutoDiffReport {
                reference: 1,
                source: 2,
                changed_pixels: 2,
                bounds: Some(multiframe::Rect {
                    x: 3,
                    y: 4,
                    width: 5,
                    height: 6,
                }),
            },
        ],
    };
    assert_eq!(
        auto_frame_report(2, &frame),
        json!({
            "index": 2,
            "winner_index": 0,
            "retained_reference": true,
            "reference_lookahead": true,
            "lookahead_comparisons": 3,
            "reused_payloads": 2,
            "reused_payload_pixels": 25,
            "references_before": [null, 0, 1, null],
            "references_after": [null, 2, 1, null],
            "candidates": [{
                "strategy": "Full", "encoded_bytes": 80, "encoded_frames": 1,
                "tile_size": null,
                "reference_index": null, "source": 0, "save_as_reference": 1,
                "trial_pixels": 100, "elapsed_seconds": 0.125,
            }],
            "pruned": [{
                "strategy": "MergedRetained", "lower_bound": 90,
                "tile_size": 16,
                "reference_index": 1, "source": 2, "save_as_reference": 1,
                "encoded_frames": 1, "planned_frames": 3,
                "trial_pixels": 30, "elapsed_seconds": 0.25,
            }],
            "analysis_seconds": 0.5,
            "differences": [
                {"reference": 0, "source": 1, "changed_pixels": 0, "bounds": null},
                {"reference": 1, "source": 2, "changed_pixels": 2,
                 "bounds": {"x": 3, "y": 4, "width": 5, "height": 6}},
            ],
        })
    );
}
