use super::*;
use rs_jxl::multiframe::Reference;
use serde_json::json;

#[test]
fn source_tool_defaults_is_presence_aware_and_pixel_only() {
    for operation in ["encode", "encode_multiframe", "encode_layers"] {
        for setting in [None, Some(false), Some(true)] {
            let mut options = json!({});
            if let Some(value) = setting {
                options["source_tool_defaults"] = json!(value);
            }
            let request: Request = serde_json::from_value(json!({
                "operation": operation, "images": [{"length": 0}], "options": options,
            }))
            .unwrap();
            request.validate_typed().unwrap();
            assert_eq!(request.options.source_tool_defaults, setting);
        }
    }
    for operation in [
        "transcode_jpeg",
        "reconstruct_jpeg",
        "decode",
        "decode_to_png",
        "info",
        "read_image",
        "analyze_multiframe",
    ] {
        for value in [false, true] {
            let mut wire = json!({
                "operation": operation, "options": {"source_tool_defaults": value},
            });
            if operation == "analyze_multiframe" {
                wire["images"] = json!([{"length": 0}]);
            }
            let request: Request = serde_json::from_value(wire).unwrap();
            assert!(
                request
                    .validate()
                    .unwrap_err()
                    .message
                    .starts_with("option source_tool_defaults is unsupported")
            );
        }
    }
    for value in [false, true] {
        let request: Request = serde_json::from_value(json!({
            "operation": "encode", "images": [{"length": 0}],
            "options": {"source_tool_defaults": value, "source_search": true},
        }))
        .unwrap();
        request.validate_typed().unwrap();
    }
}

#[test]
fn source_tool_defaults_rejects_null_wrong_types_and_duplicates() {
    for value in [
        json!(null),
        json!(0),
        json!(1),
        json!(1.0),
        json!("false"),
        json!([]),
        json!({}),
    ] {
        assert!(
            serde_json::from_value::<Request>(json!({
                "operation": "encode", "images": [{"length": 0}],
                "options": {"source_tool_defaults": value},
            }))
            .is_err()
        );
    }
    assert!(
        serde_json::from_str::<Request>(
            r#"{"operation":"encode","options":{"source_tool_defaults":false,"source_tool_defaults":true}}"#
        )
        .is_err()
    );
}

#[test]
fn source_patch_parallel_gather_is_presence_aware_and_pixel_only() {
    for operation in ["encode", "encode_multiframe", "encode_layers"] {
        for setting in [None, Some(false), Some(true)] {
            let mut options = json!({"source_ma_defaults": false});
            if let Some(value) = setting {
                options["source_patch_parallel_gather"] = json!(value);
            }
            let request: Request = serde_json::from_value(json!({
                "operation": operation,
                "images": [{"length": 0}],
                "options": options,
            }))
            .unwrap();
            request.validate_typed().unwrap();
            assert_eq!(request.options.source_patch_parallel_gather, setting);
            assert_eq!(request.options.source_ma_defaults, Some(false));
        }
    }
    for operation in [
        "transcode_jpeg",
        "reconstruct_jpeg",
        "decode",
        "decode_to_png",
        "info",
        "read_image",
        "analyze_multiframe",
    ] {
        for value in [false, true] {
            let mut wire = json!({
                "operation": operation,
                "options": {"source_patch_parallel_gather": value},
            });
            if operation == "analyze_multiframe" {
                wire["images"] = json!([{"length": 0}]);
            }
            let request: Request = serde_json::from_value(wire).unwrap();
            assert!(
                request
                    .validate()
                    .unwrap_err()
                    .message
                    .starts_with("option source_patch_parallel_gather is unsupported")
            );
        }
    }
    for value in [false, true] {
        let request: Request = serde_json::from_value(json!({
            "operation": "encode",
            "images": [{"length": 0}],
            "options": {"source_patch_parallel_gather": value, "source_search": true},
        }))
        .unwrap();
        request.validate_typed().unwrap();
    }
}

#[test]
fn source_patch_parallel_gather_rejects_null_wrong_types_and_duplicates() {
    for value in [
        json!(null),
        json!(0),
        json!(1),
        json!(1.0),
        json!("false"),
        json!([]),
        json!({}),
    ] {
        assert!(
            serde_json::from_value::<Request>(json!({
                "operation": "encode",
                "images": [{"length": 0}],
                "options": {"source_patch_parallel_gather": value},
            }))
            .is_err()
        );
    }
    assert!(
        serde_json::from_str::<Request>(
            r#"{"operation":"encode","options":{"source_patch_parallel_gather":false,"source_patch_parallel_gather":true}}"#
        )
        .is_err()
    );
}

#[test]
fn source_ma_defaults_is_presence_aware_and_pixel_only() {
    for operation in ["encode", "encode_multiframe", "encode_layers"] {
        for setting in [None, Some(false), Some(true)] {
            let mut options = json!({});
            if let Some(value) = setting {
                options["source_ma_defaults"] = json!(value);
            }
            let request: Request = serde_json::from_value(json!({
                "operation": operation,
                "images": [{"length": 0}],
                "options": options,
            }))
            .unwrap();
            request.validate().unwrap();
            assert_eq!(request.options.source_ma_defaults, setting);
        }
    }
    for operation in ["transcode_jpeg", "reconstruct_jpeg", "decode", "info"] {
        for value in [false, true] {
            let request: Request = serde_json::from_value(json!({
                "operation": operation,
                "options": {"source_ma_defaults": value},
            }))
            .unwrap();
            assert!(
                request
                    .validate()
                    .unwrap_err()
                    .message
                    .starts_with("option source_ma_defaults is unsupported")
            );
        }
    }
    for value in [false, true] {
        let request: Request = serde_json::from_value(json!({
            "operation": "encode",
            "images": [{"length": 0}],
            "options": {"source_ma_defaults": value, "source_search": true},
        }))
        .unwrap();
        request.validate().unwrap();
    }
}

#[test]
fn source_ma_defaults_rejects_null_wrong_types_and_duplicates() {
    for value in [json!(null), json!(0), json!("false"), json!([]), json!({})] {
        assert!(
            serde_json::from_value::<Request>(json!({
                "operation": "encode",
                "images": [{"length": 0}],
                "options": {"source_ma_defaults": value},
            }))
            .is_err()
        );
    }
    assert!(
        serde_json::from_str::<Request>(
            r#"{"operation":"encode","options":{"source_ma_defaults":false,"source_ma_defaults":true}}"#
        )
        .is_err()
    );
}

#[test]
fn legacy_gradient_is_explicit_and_projects_to_native_decode_options() {
    for operation in ["decode", "decode_to_png"] {
        for (wire, expected) in [
            (json!({}), false),
            (json!({"legacy_gradient": false}), false),
            (json!({"legacy_gradient": true}), true),
        ] {
            let request: Request = serde_json::from_value(json!({
                "operation": operation, "options": wire,
            }))
            .unwrap();
            request.validate_typed().unwrap();
            assert_eq!(
                crate::decode::options(&request.options)
                    .unwrap()
                    .legacy_gradient,
                expected
            );
        }
    }
}

#[test]
fn legacy_gradient_rejects_wrong_types_null_and_duplicates() {
    for value in [
        json!(null),
        json!(0),
        json!(1),
        json!(1.0),
        json!("true"),
        json!("false"),
        json!([]),
        json!({}),
    ] {
        assert!(
            serde_json::from_value::<Request>(json!({
                "operation": "decode", "options": {"legacy_gradient": value},
            }))
            .is_err(),
            "{value}"
        );
    }
    assert!(
        serde_json::from_str::<Request>(
            r#"{"operation":"decode","options":{"legacy_gradient":true,"legacy_gradient":false}}"#
        )
        .is_err()
    );
}

#[test]
fn legacy_gradient_is_rejected_by_nondecode_operations_even_when_false() {
    for operation in [
        "encode",
        "encode_multiframe",
        "encode_layers",
        "analyze_multiframe",
        "transcode_jpeg",
        "reconstruct_jpeg",
        "info",
        "read_image",
    ] {
        for value in [false, true] {
            let mut wire = json!({
                "operation": operation, "options": {"legacy_gradient": value},
            });
            if matches!(
                operation,
                "encode" | "encode_multiframe" | "encode_layers" | "analyze_multiframe"
            ) {
                wire["images"] = json!([{"length": 0}]);
            }
            let request: Request = serde_json::from_value(wire).unwrap();
            let error = request.validate().unwrap_err();
            assert!(
                error
                    .message
                    .starts_with("option legacy_gradient is unsupported"),
                "{operation}: {error}"
            );
        }
    }
}

fn automatic(options: serde_json::Value) -> Request {
    serde_json::from_value(json!({
        "operation": "encode_multiframe",
        "images": [{"length": 0}],
        "auto": true,
        "reference": "first",
        "auto_options": options,
    }))
    .unwrap()
}

#[test]
fn auto_options_default_to_native_and_keep_outer_reference() {
    let request = automatic(json!({}));
    let ValidatedRequest::EncodeMultiframe { sequence, .. } = request.validate_typed().unwrap()
    else {
        panic!("expected multiframe");
    };
    let wire = sequence.auto_options.unwrap();
    assert!(std::ptr::eq(wire, request.auto_options.as_ref().unwrap()));
    let actual = wire.native(sequence.reference);
    let default = rs_jxl::multiframe::AutoOptions::default();
    assert_eq!(actual.reference, Reference::First);
    assert_eq!(actual.retained_reference, default.retained_reference);
    assert!(!actual.retained_reference);
    assert_eq!(actual.reference_lookahead, default.reference_lookahead);
    assert!(!actual.reference_lookahead);
    assert_eq!(actual.allow_mask, default.allow_mask);
    assert_eq!(actual.allow_multi_rect, default.allow_multi_rect);
    assert_eq!(actual.tile_size, default.tile_size);
    assert_eq!(actual.extra_tile_sizes, default.extra_tile_sizes);
    assert_eq!(actual.max_rectangles, default.max_rectangles);
    assert_eq!(actual.prune, default.prune);
    assert_eq!(actual.adaptive_rectangles, default.adaptive_rectangles);
    assert_eq!(actual.max_frames, default.max_frames);
    assert_eq!(actual.max_total_trials, default.max_total_trials);
    assert_eq!(actual.max_trial_pixels, default.max_trial_pixels);
    let omitted: Request = serde_json::from_value(json!({
        "operation": "encode_multiframe", "images": [{"length": 0}], "auto": true,
    }))
    .unwrap();
    assert!(omitted.auto_options.is_none());
    omitted.validate().unwrap();
    assert!(automatic(serde_json::Value::Null).auto_options.is_none());
}

#[test]
fn auto_options_forward_all_overrides_and_zero_budgets() {
    let request = automatic(json!({
        "allow_mask": false,
        "allow_multi_rect": false,
        "tile_size": 8,
        "extra_tile_sizes": [16, 64],
        "max_rectangles": 4097,
        "prune": false,
        "adaptive_rectangles": false,
        "max_frames": 0,
        "max_total_trials": 0,
        "max_trial_pixels": 0,
    }));
    request.validate().unwrap();
    let auto = request.auto_options.unwrap().native(Reference::Previous);
    assert_eq!(auto.reference, Reference::Previous);
    assert!(!auto.retained_reference);
    assert!(!auto.reference_lookahead);
    assert!(!auto.allow_mask && !auto.allow_multi_rect);
    assert!(!auto.prune && !auto.adaptive_rectangles);
    assert_eq!((auto.tile_size, auto.max_rectangles), (8, 4097));
    assert_eq!(auto.extra_tile_sizes, [16, 64]);
    assert_eq!(
        (
            auto.max_frames,
            auto.max_total_trials,
            auto.max_trial_pixels
        ),
        (0, 0, 0)
    );
}

#[test]
fn retained_reference_requires_automatic_smallest_crop_and_forwards_native_flag() {
    let mut request = automatic(json!({"retained_reference": true}));
    for reference in [None, Some("previous"), Some("first"), Some("full")] {
        request.reference = reference.map(str::to_owned);
        assert_eq!(
            request.validate().unwrap_err().message,
            "retained_reference requires reference=smallest_crop"
        );
    }
    request.reference = Some("smallest_crop".into());
    request.validate().unwrap();
    let actual = request
        .auto_options
        .as_ref()
        .unwrap()
        .native(Reference::SmallestCrop);
    assert!(actual.retained_reference);
    assert_eq!(actual.reference, Reference::SmallestCrop);
    request.auto = Some(false);
    assert!(request.validate().is_err());
    for reference in [
        None,
        Some("previous"),
        Some("first"),
        Some("full"),
        Some("smallest_crop"),
    ] {
        let mut disabled = automatic(json!({"retained_reference": false}));
        disabled.reference = reference.map(str::to_owned);
        disabled.validate().unwrap();
    }
}

#[test]
fn reference_lookahead_requires_retained_reference_and_forwards_native_flag() {
    for options in [
        json!({"reference_lookahead": true}),
        json!({"reference_lookahead": true, "retained_reference": false}),
    ] {
        let mut request = automatic(options);
        request.images = None;
        assert_eq!(
            request.validate().unwrap_err().message,
            "reference_lookahead requires retained_reference=true"
        );
    }
    let mut request = automatic(json!({
        "reference_lookahead": true, "retained_reference": true,
    }));
    for reference in [None, Some("previous"), Some("first"), Some("full")] {
        request.reference = reference.map(str::to_owned);
        assert_eq!(
            request.validate().unwrap_err().message,
            "retained_reference requires reference=smallest_crop"
        );
    }
    request.reference = Some("smallest_crop".into());
    let ValidatedRequest::EncodeMultiframe { sequence, .. } = request.validate_typed().unwrap()
    else {
        panic!("expected multiframe");
    };
    let native = sequence.auto_options.unwrap().native(sequence.reference);
    assert!(native.reference_lookahead && native.retained_reference);
    assert_eq!(native.reference, Reference::SmallestCrop);
    for auto in [None, Some(false)] {
        request.auto = auto;
        assert_eq!(
            request.validate().unwrap_err().message,
            "auto_options require auto=true"
        );
    }
    for reference in [
        None,
        Some("previous"),
        Some("first"),
        Some("full"),
        Some("smallest_crop"),
    ] {
        let mut disabled = automatic(json!({"reference_lookahead": false}));
        disabled.reference = reference.map(str::to_owned);
        disabled.validate().unwrap();
    }
}

#[test]
fn auto_options_require_automatic_multiframe_and_positive_geometry() {
    for operation in [
        Operation::Encode,
        Operation::AnalyzeMultiframe,
        Operation::TranscodeJpeg,
        Operation::ReconstructJpeg,
        Operation::Decode,
        Operation::DecodeToPng,
        Operation::Info,
        Operation::ReadImage,
    ] {
        let mut request = automatic(json!({}));
        request.operation = operation;
        request.auto = None;
        request.reference = None;
        assert!(request.validate().is_err());
    }
    for auto in [None, Some(false)] {
        let mut request = automatic(json!({}));
        request.auto = auto;
        assert!(request.validate().is_err());
    }
    for field in ["tile_size", "max_rectangles"] {
        assert!(automatic(json!({field: 0})).validate().is_err());
    }
}

#[test]
fn auto_options_serde_rejects_unknown_duplicate_and_wrong_types() {
    assert!(
        automatic(json!({"extra_tile_sizes": [0]}))
            .validate()
            .is_err()
    );
    for value in [
        json!(null),
        json!(false),
        json!(8),
        json!("8"),
        json!([false]),
        json!([-1]),
        json!([1.5]),
        json!([4294967296u64]),
    ] {
        assert!(serde_json::from_value::<AutoOptions>(json!({"extra_tile_sizes": value})).is_err());
    }
    for options in [
        r#"{"unknown":true}"#,
        r#"{"reference":"first"}"#,
        r#"{"prune":true,"prune":false}"#,
        r#"{"retained_reference":true,"retained_reference":false}"#,
        r#"{"reference_lookahead":true,"reference_lookahead":false}"#,
        r#"{"max_frames":1,"max_frames":2}"#,
        r#"[]"#,
        r#"true"#,
        r#"{"tile_size":4294967296}"#,
        r#"{"max_trial_pixels":18446744073709551616}"#,
    ] {
        let request = format!(r#"{{"operation":"encode_multiframe","auto_options":{options}}}"#);
        assert!(
            serde_json::from_str::<Request>(&request).is_err(),
            "{options}"
        );
    }
    for field in [
        "retained_reference",
        "reference_lookahead",
        "allow_mask",
        "allow_multi_rect",
        "prune",
        "adaptive_rectangles",
    ] {
        for value in [
            json!(0),
            json!(1),
            json!(1.0),
            json!("true"),
            json!(null),
            json!([]),
            json!({}),
        ] {
            assert!(serde_json::from_value::<AutoOptions>(json!({field: value})).is_err());
        }
    }
    for field in [
        "tile_size",
        "max_rectangles",
        "max_frames",
        "max_total_trials",
        "max_trial_pixels",
    ] {
        for value in [json!(true), json!(-1), json!(1.5), json!("1"), json!(null)] {
            assert!(serde_json::from_value::<AutoOptions>(json!({field: value})).is_err());
        }
    }
    assert!(
        serde_json::from_str::<Request>(
            r#"{"operation":"encode_multiframe","auto_options":{},"auto_options":{}}"#,
        )
        .is_err()
    );
}

#[test]
fn multiframe_reference_projection_preserves_wire_names_defaults_and_errors() {
    let mut request = automatic(json!({}));
    request.reference = None;
    let ValidatedRequest::EncodeMultiframe { sequence, .. } = request.validate_typed().unwrap()
    else {
        panic!("expected multiframe");
    };
    assert_eq!(sequence.reference, Reference::Previous);
    for (name, expected) in [
        ("full", Reference::Full),
        ("previous", Reference::Previous),
        ("first", Reference::First),
        ("smallest_crop", Reference::SmallestCrop),
    ] {
        request.reference = Some(name.into());
        let ValidatedRequest::EncodeMultiframe { sequence, .. } = request.validate_typed().unwrap()
        else {
            panic!("expected multiframe");
        };
        assert_eq!(sequence.reference, expected);
        let analysis: AnalysisOptions = serde_json::from_value(json!({"reference": name})).unwrap();
        assert_eq!(analysis.reference, expected);
    }
    for name in ["", "Previous", "auto"] {
        request.reference = Some(name.into());
        assert_eq!(
            request.validate().unwrap_err().message,
            "unsupported reference"
        );
        assert_eq!(request.multiframe_reference(), Reference::Previous);
        let error =
            serde_json::from_value::<AnalysisOptions>(json!({"reference": name})).unwrap_err();
        assert!(error.to_string().contains("unsupported analysis reference"));
    }
}

#[test]
fn auto_options_validation_preserves_error_precedence() {
    for (options, auto_enabled, reference, expected) in [
        (
            json!({"extra_tile_sizes": [0], "tile_size": 0}),
            false,
            "invalid",
            "extra_tile_sizes must be positive",
        ),
        (
            json!({"reference_lookahead": true, "tile_size": 0}),
            false,
            "invalid",
            "auto_options require auto=true",
        ),
        (
            json!({"reference_lookahead": true, "tile_size": 0}),
            true,
            "invalid",
            "reference_lookahead requires retained_reference=true",
        ),
        (
            json!({"retained_reference": true, "tile_size": 0}),
            true,
            "invalid",
            "retained_reference requires reference=smallest_crop",
        ),
        (
            json!({"tile_size": 0}),
            true,
            "invalid",
            "tile_size and max_rectangles must be positive",
        ),
    ] {
        let mut request = automatic(options);
        request.auto = Some(auto_enabled);
        request.reference = Some(reference.into());
        request.images = None;
        assert_eq!(request.validate().unwrap_err().message, expected);
    }
}
