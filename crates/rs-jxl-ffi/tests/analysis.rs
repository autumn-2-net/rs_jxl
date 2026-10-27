use rs_jxl::multiframe::{AnalysisOptions as NativeOptions, Reference};
use rs_jxl_ffi::{
    request::{AnalysisOptions, Operation, Request, ValidatedRequest},
    *,
};
use serde_json::{Value, json};
use std::{ptr, slice};

fn request(analysis_options: Value) -> Request {
    serde_json::from_value(json!({
        "operation": "analyze_multiframe",
        "images": [{"length": 0}],
        "analysis_options": analysis_options,
    }))
    .unwrap()
}

#[test]
fn analysis_options_use_native_defaults_and_forward_all_overrides() {
    let request = request(json!({}));
    let ValidatedRequest::AnalyzeMultiframe {
        images,
        strip_metadata,
        options,
        analysis_options,
    } = request.validate_typed().unwrap()
    else {
        panic!()
    };
    assert_eq!(images.len(), 1);
    assert!(!strip_metadata);
    assert_eq!(options.threads, None);
    let actual = analysis_options.unwrap().native(1);
    let default = NativeOptions::default();
    assert_eq!(actual.reference, default.reference);
    assert_eq!(actual.allow_mask, default.allow_mask);
    assert_eq!(actual.allow_multi_rect, default.allow_multi_rect);
    assert_eq!(actual.adaptive_rectangles, default.adaptive_rectangles);
    assert_eq!(actual.tile_size, default.tile_size);
    assert_eq!(actual.extra_tile_sizes, default.extra_tile_sizes);
    assert_eq!(actual.history_window, default.history_window);
    assert_eq!(actual.max_rectangles, default.max_rectangles);
    assert_eq!(
        actual.rectangle_bytes_per_pixel,
        default.rectangle_bytes_per_pixel
    );
    assert_eq!(actual.threads, default.threads);

    for (wire, reference) in [
        ("full", Reference::Full),
        ("previous", Reference::Previous),
        ("first", Reference::First),
        ("smallest_crop", Reference::SmallestCrop),
    ] {
        let mut request = self::request(json!({
            "reference": wire,
            "allow_mask": false,
            "allow_multi_rect": false,
            "adaptive_rectangles": false,
            "tile_size": 8,
            "extra_tile_sizes": [16, 64],
            "history_window": 4,
            "max_rectangles": 17,
            "rectangle_bytes_per_pixel": 0.0,
        }));
        request.options.threads = Some(3);
        request.validate().unwrap();
        let actual = request.analysis_options.unwrap().native(3);
        assert_eq!(actual.reference, reference);
        assert!(!actual.allow_mask && !actual.allow_multi_rect && !actual.adaptive_rectangles);
        assert_eq!((actual.tile_size, actual.max_rectangles), (8, 17));
        assert_eq!(actual.extra_tile_sizes, [16, 64]);
        assert_eq!(actual.history_window, 4);
        assert_eq!(actual.rectangle_bytes_per_pixel, 0.0);
        assert_eq!(actual.threads, 3);
    }
    let omitted: Request = serde_json::from_value(json!({
        "operation": "analyze_multiframe", "images": [{"length": 0}],
    }))
    .unwrap();
    omitted.validate().unwrap();
    assert!(omitted.analysis_options.is_none());
}

#[test]
fn analysis_options_reject_other_operations_and_invalid_geometry() {
    for operation in [
        Operation::Encode,
        Operation::EncodeMultiframe,
        Operation::TranscodeJpeg,
        Operation::ReconstructJpeg,
        Operation::Decode,
        Operation::DecodeToPng,
        Operation::Info,
        Operation::ReadImage,
    ] {
        let mut request = request(json!({}));
        request.operation = operation;
        assert_eq!(
            request.validate().unwrap_err().message,
            "analysis_options require analyze_multiframe"
        );
    }
    for options in [
        json!({"tile_size": 0, "allow_multi_rect": false}),
        json!({"extra_tile_sizes": [0]}),
        json!({"max_rectangles": 0, "adaptive_rectangles": false}),
        json!({"rectangle_bytes_per_pixel": -0.1}),
    ] {
        assert!(request(options).validate().is_err());
    }
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut request = request(json!({}));
        request
            .analysis_options
            .as_mut()
            .unwrap()
            .rectangle_bytes_per_pixel = invalid;
        assert!(request.validate().is_err());
    }
    for threads in [0, 257] {
        let mut request = request(json!({}));
        request.options.threads = Some(threads);
        assert_eq!(
            request.validate().unwrap_err().message,
            "threads must be 1..256"
        );
    }
}

#[test]
fn analysis_options_reject_null_nonobjects_unknown_duplicates_and_wrong_types() {
    for options in [
        "null",
        "[]",
        "[{}]",
        "true",
        "1",
        "\"options\"",
        r#"{"threads":2}"#,
        r#"{"unknown":true}"#,
        r#"{"prune":false}"#,
        r#"{"effort":7}"#,
        r#"{"reference":"Previous"}"#,
        r#"{"reference":"auto"}"#,
        r#"{"reference":null}"#,
        r#"{"reference":1}"#,
        r#"{"tile_size":4294967296}"#,
        r#"{"max_rectangles":18446744073709551616}"#,
        r#"{"rectangle_bytes_per_pixel":1e999}"#,
        r#"{"reference":"first","reference":"previous"}"#,
        r#"{"allow_mask":true,"allow_mask":false}"#,
        r#"{"allow_multi_rect":true,"allow_multi_rect":false}"#,
        r#"{"adaptive_rectangles":true,"adaptive_rectangles":false}"#,
        r#"{"tile_size":1,"tile_size":2}"#,
        r#"{"max_rectangles":1,"max_rectangles":2}"#,
        r#"{"rectangle_bytes_per_pixel":1,"rectangle_bytes_per_pixel":2}"#,
    ] {
        let wire = format!(r#"{{"operation":"analyze_multiframe","analysis_options":{options}}}"#);
        assert!(serde_json::from_str::<Request>(&wire).is_err(), "{options}");
    }
    for field in ["allow_mask", "allow_multi_rect", "adaptive_rectangles"] {
        for value in [json!(0), json!(1), json!("true"), json!(null)] {
            assert!(serde_json::from_value::<AnalysisOptions>(json!({field: value})).is_err());
        }
    }
    for field in ["tile_size", "max_rectangles"] {
        for value in [json!(true), json!(-1), json!(1.5), json!("1"), json!(null)] {
            assert!(serde_json::from_value::<AnalysisOptions>(json!({field: value})).is_err());
        }
    }
    for value in [json!(true), json!("1.5"), json!(null)] {
        assert!(
            serde_json::from_value::<AnalysisOptions>(json!({
                "rectangle_bytes_per_pixel": value,
            }))
            .is_err()
        );
    }
    for wire in [
        r#"{"operation":"analyze_multiframe","analysis_options":{},"analysis_options":{}}"#,
        r#"{"operation":"analyze_multiframe","options":null}"#,
        r#"{"operation":"analyze_multiframe","options":{"unknown":1}}"#,
        r#"{"operation":"analyze_multiframe","options":{"threads":1,"threads":2}}"#,
        r#"{"operation":"analyze_multiframe","options":{},"options":{}}"#,
        r#"{"operation":"analyze_multiframe","options":{"analysis_options":{}}}"#,
    ] {
        assert!(serde_json::from_str::<Request>(wire).is_err(), "{wire}");
    }
}

#[test]
fn analysis_rejects_encoding_controls_and_auto_options() {
    for options in [
        json!({"effort": 7}),
        json!({"source_search": false}),
        json!({"group_size_shift": 1}),
        json!({"rct": 0}),
        json!({"predictor": 0}),
        json!({"palette_colors": 0}),
        json!({"patches": false}),
        json!({"learning_fraction": 0.5}),
        json!({"ma_max_leaves": 1}),
        json!({"limits": {}}),
    ] {
        let request: Request = serde_json::from_value(json!({
            "operation": "analyze_multiframe", "images": [{"length": 0}], "options": options,
        }))
        .unwrap();
        assert!(request.validate().is_err());
    }
    let request: Request = serde_json::from_value(json!({
        "operation": "analyze_multiframe", "images": [{"length": 0}], "auto_options": {},
    }))
    .unwrap();
    assert_eq!(
        request.validate().unwrap_err().message,
        "sequence fields require encode_multiframe"
    );
}

struct OwnedResult(*mut RsJxlResult);

impl Drop for OwnedResult {
    fn drop(&mut self) {
        unsafe { rs_jxl_result_free(self.0) };
    }
}

impl OwnedResult {
    fn response(&self) -> (i32, Value) {
        assert!(!self.0.is_null());
        let mut length = 0;
        let json = unsafe { rs_jxl_result_json(self.0, &mut length) };
        assert!(!json.is_null());
        let json = serde_json::from_slice(unsafe { slice::from_raw_parts(json, length) }).unwrap();
        assert!(unsafe { rs_jxl_result_data(self.0, &mut length) }.is_null());
        assert_eq!(length, 0);
        (unsafe { rs_jxl_result_status(self.0) }, json)
    }
}

fn wire(json: &[u8], data: &[u8]) -> RsJxlRequest {
    RsJxlRequest {
        abi_version: 1,
        struct_size: rs_jxl_request_size(),
        json: json.as_ptr(),
        json_len: json.len(),
        data: data.as_ptr(),
        data_len: data.len(),
    }
}

fn call(context: Option<*mut RsJxlContext>, value: Value, data: &[u8]) -> OwnedResult {
    let json = serde_json::to_vec(&value).unwrap();
    let request = wire(&json, data);
    OwnedResult(unsafe {
        match context {
            Some(context) => rs_jxl_context_execute(context, &request),
            None => rs_jxl_execute(&request),
        }
    })
}

struct OwnedContext(*mut RsJxlContext);

impl Drop for OwnedContext {
    fn drop(&mut self) {
        unsafe { rs_jxl_context_free(self.0) };
    }
}

fn context(kind: &str) -> OwnedContext {
    let json = serde_json::to_vec(&json!({"kind": kind, "threads": 2})).unwrap();
    let mut context = ptr::null_mut();
    let result = OwnedResult(unsafe { rs_jxl_context_create(&wire(&json, &[]), &mut context) });
    let (status, metadata) = result.response();
    assert_eq!(status, 0, "{metadata}");
    assert!(!context.is_null());
    OwnedContext(context)
}

fn sequence() -> Value {
    let images: Vec<_> = (0..3)
        .map(|index| {
            json!({
                "width": 4, "height": 2, "channels": 1, "dtype": "u8",
                "offset": index * 8, "length": 8,
            })
        })
        .collect();
    json!({"operation": "analyze_multiframe", "images": images})
}

fn samples() -> Vec<u8> {
    let mut pixels = vec![0; 24];
    pixels[9] = 1;
    pixels[17] = 1;
    pixels
}

fn ok(context: Option<*mut RsJxlContext>, request: Value, data: &[u8]) -> Value {
    let (status, metadata) = call(context, request, data).response();
    assert_eq!(status, 0, "{metadata}");
    metadata
}

#[test]
fn abi_analysis_retains_legacy_bounds_and_exposes_native_statistics() {
    let report = ok(None, sequence(), &samples());
    assert_eq!(report["operation"], "analyze_multiframe");
    assert_eq!(report["width"], 4);
    assert_eq!(report["height"], 2);
    assert_eq!(report["channels"], 1);
    assert_eq!(report["frame_count"], 3);
    assert_eq!(report["reference"], "smallest_crop");
    assert_eq!(report["crop_supported"], true);
    assert_eq!(report["rectangle_bytes_per_pixel"], 1.0);
    assert!(report["analysis_seconds"].as_f64().unwrap() >= 0.0);
    let full = json!({"x": 0, "y": 0, "width": 4, "height": 2});
    let changed = json!({"x": 1, "y": 0, "width": 1, "height": 1});
    let frames = report["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0]["previous"], full);
    assert_eq!(frames[0]["first"], full);
    assert!(frames[0]["previous_stats"].is_null());
    assert!(frames[0]["first_stats"].is_null());
    assert!(frames[0]["plan"]["reference_index"].is_null());
    assert_eq!(frames[0]["plan"]["region"], full);
    assert_eq!(frames[0]["plan"]["layer_pixels"], 8);
    assert_eq!(frames[1]["previous"], changed);
    assert_eq!(frames[1]["first"], changed);
    assert_eq!(
        frames[1]["previous_stats"],
        json!({
            "bounds": changed, "changed_pixels": 1, "changed_ratio": 0.125,
            "bbox_pixels": 1, "bbox_ratio": 0.125, "bbox_fill_ratio": 1.0,
            "mask_eligible": false,
        })
    );
    assert_eq!(frames[1]["previous_stats"], frames[1]["first_stats"]);
    assert!(frames[2]["previous"].is_null());
    assert_eq!(frames[2]["first"], changed);
    assert_eq!(frames[2]["previous_stats"]["changed_pixels"], 0);
    assert_eq!(frames[2]["previous_stats"]["bbox_fill_ratio"], 0.0);
    assert!(frames[2]["plan"]["changed"].is_null());
    assert_eq!(frames[2]["plan"]["layer_pixels"], 1);

    let summaries = report["reference_modes"].as_array().unwrap();
    assert_eq!(summaries.len(), 4);
    let full = summaries.iter().find(|s| s["reference"] == "full").unwrap();
    assert_eq!(full["layer_pixels"], 24);
    assert_eq!(full["layer_area_ratio"], 1.0);
    let previous = summaries
        .iter()
        .find(|s| s["reference"] == "previous")
        .unwrap();
    assert_eq!(previous["layer_pixels"], 10);
    assert_eq!(previous["layer_area_ratio"], 10.0 / 24.0);
    assert_eq!(previous["average_changed_ratio"], 0.0625);
    assert_eq!(previous["identical_frames"], 1);
    assert_eq!(previous["crop_frames"], 2);
    for frame in frames {
        for candidate in frame["candidates"].as_array().unwrap() {
            assert!(
                candidate["strategy"]
                    .as_str()
                    .unwrap()
                    .chars()
                    .all(|c| !c.is_uppercase())
            );
            assert!(candidate["rectangles"].is_array());
            assert!(candidate["layer_count"].as_u64().unwrap() > 0);
            assert!(candidate.get("encoded_bytes").is_none());
            assert!(candidate.get("estimated_bytes").is_none());
        }
    }
}

#[test]
fn abi_analysis_context_inherits_workers_rejects_mismatch_and_recovers() {
    let encoder = context("encoder");
    let pixels = samples();
    let mut baseline = ok(None, sequence(), &pixels);
    baseline.as_object_mut().unwrap().remove("analysis_seconds");
    let mut threaded = sequence();
    threaded["options"] = json!({"threads": 2});
    let mut report = ok(None, threaded, &pixels);
    report.as_object_mut().unwrap().remove("analysis_seconds");
    assert_eq!(report, baseline);
    for threads in [Value::Null, json!(2)] {
        let mut request = sequence();
        request["options"] = json!({"threads": threads});
        let mut report = ok(Some(encoder.0), request, &pixels);
        report.as_object_mut().unwrap().remove("analysis_seconds");
        assert_eq!(report, baseline);
    }
    let mut mismatch = sequence();
    mismatch["options"] = json!({"threads": 1});
    let (status, error) = call(Some(encoder.0), mismatch, &pixels).response();
    assert_eq!(status, 1);
    assert_eq!(error["error_kind"], "invalid_input");
    assert_eq!(error["error"], "request threads must match the context");

    let mut overridden = sequence();
    overridden["analysis_options"] = json!({
        "reference": "full", "rectangle_bytes_per_pixel": 2.5,
    });
    let report = ok(Some(encoder.0), overridden, &pixels);
    assert_eq!(report["reference"], "full");
    assert_eq!(report["rectangle_bytes_per_pixel"], 2.5);
    assert_eq!(report["frames"][1]["plan"]["layer_pixels"], 8);
    let result = call(Some(encoder.0), sequence(), &pixels);
    drop(encoder);
    let (status, mut report) = result.response();
    assert_eq!(status, 0, "{report}");
    report.as_object_mut().unwrap().remove("analysis_seconds");
    assert_eq!(report, baseline);

    let decoder = context("decoder");
    let (status, error) = call(Some(decoder.0), sequence(), &pixels).response();
    assert_eq!(status, 1);
    assert_eq!(error["error_kind"], "unsupported");
}

#[test]
fn abi_analysis_validates_images_and_single_frame_reports() {
    let mut request = sequence();
    request["images"].as_array_mut().unwrap().truncate(1);
    let report = ok(None, request, &samples());
    assert_eq!(report["frame_count"], 1);
    for summary in report["reference_modes"].as_array().unwrap() {
        assert_eq!(summary["suitability"], "single_frame");
        assert_eq!(summary["average_changed_ratio"], 0.0);
        assert_eq!(summary["layer_pixels"], 8);
        assert_eq!(summary["layer_area_ratio"], 1.0);
    }
    let mut mismatched = sequence();
    mismatched["images"][1]["width"] = json!(2);
    mismatched["images"][1]["height"] = json!(4);
    for request in [
        mismatched,
        json!({"operation": "analyze_multiframe", "images": []}),
        json!({"operation": "analyze_multiframe"}),
    ] {
        let (status, error) = call(None, request, &samples()).response();
        assert_eq!(status, 1);
        assert_eq!(error["error_kind"], "invalid_input");
    }
}

#[test]
fn abi_analysis_options_control_mask_and_rectangle_candidates() {
    let mut pixels = vec![0; 256];
    for pixel in pixels.chunks_exact_mut(2) {
        pixel[1] = 255;
    }
    pixels[128] = 10;
    pixels[254] = 10;
    let images: Vec<_> = (0..2)
        .map(|index| {
            json!({
                "width": 8, "height": 8, "channels": 2, "dtype": "u8",
                "offset": index * 128, "length": 128,
            })
        })
        .collect();
    let mut request = json!({
        "operation": "analyze_multiframe", "images": images,
        "analysis_options": {
            "reference": "previous", "tile_size": 2, "max_rectangles": 8,
            "rectangle_bytes_per_pixel": 10.0,
        },
    });
    let report = ok(None, request.clone(), &pixels);
    assert_eq!(report["frames"][1]["previous_stats"]["mask_eligible"], true);
    let candidates = report["frames"][1]["candidates"].as_array().unwrap();
    assert!(candidates.iter().any(|c| c["strategy"] == "mask_previous"));
    assert!(candidates.iter().any(|c| c["strategy"] == "tiles_previous"));

    request["analysis_options"]["allow_mask"] = json!(false);
    request["analysis_options"]["allow_multi_rect"] = json!(false);
    request["analysis_options"]["adaptive_rectangles"] = json!(false);
    let report = ok(None, request, &pixels);
    assert_eq!(report["frames"][1]["previous_stats"]["mask_eligible"], true);
    let candidates = report["frames"][1]["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0]["strategy"], "full");
}

#[test]
fn abi_analysis_preserves_subsampled_extra_statistics_and_full_frame_fallback() {
    let mut pixels = vec![0; 40];
    pixels[36] = 1;
    let images: Vec<_> = (0..2)
        .map(|index| {
            json!({
                "width": 4, "height": 4, "channels": 1, "dtype": "u8",
                "offset": index * 20, "length": 16,
                "extra_channels": [{
                    "type": "depth", "dtype": "u8", "dim_shift": 1,
                    "offset": index * 20 + 16, "length": 4,
                }],
            })
        })
        .collect();
    let report = ok(
        None,
        json!({
            "operation": "analyze_multiframe", "images": images,
        }),
        &pixels,
    );
    assert_eq!(report["channels"], 2);
    assert_eq!(report["crop_supported"], false);
    assert_eq!(report["frames"][1]["previous_stats"]["changed_pixels"], 4);
    assert_eq!(
        report["frames"][1]["previous"],
        json!({
            "x": 0, "y": 0, "width": 2, "height": 2,
        })
    );
    assert_eq!(report["frames"][1]["plan"]["layer_pixels"], 16);
    assert!(report["frames"][1]["plan"]["reference_index"].is_null());
    for summary in report["reference_modes"].as_array().unwrap() {
        assert_eq!(summary["suitability"], "full_frames_required");
        assert_eq!(summary["layer_pixels"], 32);
    }
}
