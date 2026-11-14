use super::multiframe::auto_frame_report;
use rs_jxl::multiframe::{
    self, AutoCandidateReport, AutoDiffReport, AutoFrameReport, AutoPrunedReport, Strategy,
};
use serde_json::json;
use std::time::Duration;

#[test]
fn source_patch_parallel_gather_reaches_pixel_projection_and_preserves_other_options() {
    use crate::request::Options;
    use rs_jxl::{Image, PixelData, PixelLayout};

    let image = Image::new(16, 16, PixelLayout::Gray, PixelData::U8(vec![7; 256])).unwrap();
    for setting in [None, Some(false), Some(true)] {
        for threads in [1, 2] {
            let options = Options {
                effort: Some(9),
                threads: Some(threads),
                source_patch_parallel_gather: setting,
                source_ma_defaults: Some(false),
                ma_max_leaves: Some(32),
                ..Default::default()
            };
            let (params, base) = super::candidate(&options).unwrap();
            assert_eq!(params.speed_tier, 1);
            assert_eq!(base.source_patch_parallel_gather, setting.unwrap_or(true));
            let projected = super::pixel_options(&image, &options).unwrap();
            assert_eq!(
                projected.source_patch_parallel_gather,
                setting.unwrap_or(true)
            );
            assert_eq!(projected.threads, threads);
            assert!(!projected.modular.source_ma_defaults);
            assert_eq!(projected.modular.ma_max_leaves, Some(32));
        }
    }
}

#[test]
fn source_ma_defaults_reaches_pixel_projection_without_changing_effort_defaults() {
    use crate::request::Options;
    use rs_jxl::modular::ma_train::{DescriptionCost, TreeLearning};
    use rs_jxl::modular::refinement::OffsetMode;
    use rs_jxl::{Image, PixelData, PixelLayout};

    let image = Image::new(16, 16, PixelLayout::Gray, PixelData::U8(vec![7; 256])).unwrap();
    for setting in [None, Some(false), Some(true)] {
        let options = Options {
            effort: Some(9),
            source_ma_defaults: setting,
            ..Default::default()
        };
        let (params, base) = super::candidate(&options).unwrap();
        assert_eq!(params.speed_tier, 1);
        assert_eq!(base.modular.source_ma_defaults, setting.unwrap_or(true));
        let projected = super::pixel_options(&image, &options).unwrap();
        if setting == Some(false) {
            assert_eq!(projected.modular.tree_learning, TreeLearning::Legacy);
            assert!(projected.modular.refinement.offsets.is_none());
            assert_eq!(
                projected.modular.source_ma_scoring.description,
                DescriptionCost::Disabled
            );
        } else {
            assert_eq!(projected.modular.tree_learning, TreeLearning::Symmetric);
            assert_eq!(
                projected.modular.refinement.offsets.unwrap().mode,
                OffsetMode::DistributionSinglePass
            );
            assert_eq!(
                projected.modular.source_ma_scoring.description,
                DescriptionCost::Half
            );
        }
    }
    let (params, _) = super::candidate(&Options::default()).unwrap();
    assert_eq!(params.speed_tier, 3);
}

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
