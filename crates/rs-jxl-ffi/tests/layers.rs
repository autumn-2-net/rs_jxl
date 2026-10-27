use std::{ptr, slice};

use rs_jxl::frame::BlendMode;
use rs_jxl_ffi::{
    request::{Request, ValidatedRequest},
    *,
};
use serde_json::{Value, json};

struct OwnedResult(*mut RsJxlResult);

impl Drop for OwnedResult {
    fn drop(&mut self) {
        unsafe { rs_jxl_result_free(self.0) };
    }
}

impl OwnedResult {
    fn response(&self) -> (i32, Value, Vec<u8>) {
        assert!(!self.0.is_null());
        let mut length = 0;
        let json = unsafe { rs_jxl_result_json(self.0, &mut length) };
        assert!(!json.is_null());
        let json = serde_json::from_slice(unsafe { slice::from_raw_parts(json, length) }).unwrap();
        let data = unsafe { rs_jxl_result_data(self.0, &mut length) };
        let data = if length == 0 {
            assert!(data.is_null());
            Vec::new()
        } else {
            assert!(!data.is_null());
            unsafe { slice::from_raw_parts(data, length) }.to_vec()
        };
        (unsafe { rs_jxl_result_status(self.0) }, json, data)
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

fn raw(context: Option<*mut RsJxlContext>, json: &[u8], data: &[u8]) -> OwnedResult {
    let request = wire(json, data);
    OwnedResult(unsafe {
        match context {
            Some(context) => rs_jxl_context_execute(context, &request),
            None => rs_jxl_execute(&request),
        }
    })
}

fn call(context: Option<*mut RsJxlContext>, request: Value, data: &[u8]) -> OwnedResult {
    raw(context, &serde_json::to_vec(&request).unwrap(), data)
}

fn ok(context: Option<*mut RsJxlContext>, request: Value, data: &[u8]) -> (Value, Vec<u8>) {
    let (status, report, bytes) = call(context, request, data).response();
    assert_eq!(status, 0, "{report}");
    (report, bytes)
}

fn invalid(result: OwnedResult) {
    let (status, report, data) = result.response();
    assert_eq!(status, 1, "{report}");
    assert_eq!(report["error_kind"], "invalid_input", "{report}");
    assert!(report["error"].is_string());
    assert!(data.is_empty());
}

fn malformed(request: Value) {
    let wire = serde_json::to_vec(&request).unwrap();
    assert!(
        serde_json::from_slice::<Request>(&wire).is_err(),
        "{request}"
    );
    invalid(raw(None, &wire, &[]));
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
    let (status, report, data) = result.response();
    assert_eq!(status, 0, "{report}");
    assert!(data.is_empty());
    assert!(!context.is_null());
    OwnedContext(context)
}

fn sequence() -> Value {
    json!({
        "operation": "encode_layers",
        "options": {"effort": 2},
        "images": [
            {"width": 2, "height": 1, "channels": 1, "dtype": "u8", "length": 2},
            {"width": 1, "height": 2, "channels": 1, "dtype": "u8", "offset": 2, "length": 2},
        ],
    })
}

#[test]
fn typed_layers_project_native_defaults_and_all_wire_overrides() {
    let mut value = sequence();
    let request: Request = serde_json::from_value(value.clone()).unwrap();
    let ValidatedRequest::EncodeLayers {
        images,
        options,
        layer_options,
    } = request.validate_typed().unwrap()
    else {
        panic!("expected layers");
    };
    assert!(ptr::eq(images, request.images()));
    assert!(ptr::eq(options, &request.options));
    assert!(layer_options.is_none());
    value["layer_options"] = json!({});
    let request: Request = serde_json::from_value(value.clone()).unwrap();
    request.validate().unwrap();
    let native = request.layer_options.unwrap().native();
    assert!(native.canvas.is_none() && native.frames.is_empty() && native.animation.is_none());
    value["layer_options"] = json!({"frames": [{}, {}], "animation": {}});
    let request: Request = serde_json::from_value(value.clone()).unwrap();
    request.validate().unwrap();
    let native = request.layer_options.unwrap().native();
    assert_eq!(
        native.animation.unwrap(),
        rs_jxl::metadata::Animation::default()
    );
    for frame in native.frames {
        assert_eq!(frame, rs_jxl::layers::default_frame_options());
        assert_eq!(frame.blend.source, 1);
        assert_eq!(frame.save_as_reference, 0);
        assert!(!frame.reference_only && !frame.save_before_color_transform);
    }
    for (mode, expected) in [
        ("replace", BlendMode::Replace),
        ("add", BlendMode::Add),
        ("blend", BlendMode::Blend),
        ("alpha_weighted_add", BlendMode::AlphaWeightedAdd),
        ("multiply", BlendMode::Multiply),
    ] {
        value["layer_options"] = json!({
            "canvas": [5, 6],
            "animation": {"ticks_per_second_numerator": 24, "ticks_per_second_denominator": 1, "loops": 7, "timecodes": true},
            "frames": [{
                "origin": [-1, 2], "duration": 3, "timecode": 9, "name": "first",
                "blend": {"mode": mode, "source": 2, "alpha_channel": 1, "clamp": true},
                "extra_blend": [{"mode": "multiply", "source": 3}], "save_as_reference": 2,
            }, {}],
        });
        let request: Request = serde_json::from_value(value.clone()).unwrap();
        request.validate().unwrap();
        let native = request.layer_options.unwrap().native();
        assert_eq!(native.canvas, Some((5, 6)));
        let animation = native.animation.unwrap();
        assert_eq!(
            (
                animation.ticks_per_second_numerator,
                animation.ticks_per_second_denominator,
                animation.loops
            ),
            (24, 1, 7)
        );
        assert!(animation.timecodes);
        let frame = &native.frames[0];
        assert_eq!(
            (
                frame.origin,
                frame.duration,
                frame.timecode,
                frame.name.as_str()
            ),
            ((-1, 2), 3, 9, "first")
        );
        assert_eq!(frame.blend.mode, expected);
        assert_eq!(
            (
                frame.blend.source,
                frame.blend.alpha_channel,
                frame.blend.clamp
            ),
            (2, 1, true)
        );
        assert_eq!(frame.extra_blend.len(), 1);
        assert_eq!(frame.extra_blend[0].mode, BlendMode::Multiply);
        assert_eq!(frame.extra_blend[0].source, 3);
        assert_eq!(frame.save_as_reference, 2);
        assert!(!frame.reference_only && !frame.save_before_color_transform);
    }
}

#[test]
fn raw_wire_rejects_nonobjects_unknown_fields_duplicates_and_wrong_types() {
    for options in [
        "null",
        "[]",
        "[null,null,null]",
        "true",
        "1",
        "\"options\"",
        r#"{"unknown":1}"#,
        r#"{"frames":[],"frames":null}"#,
        r#"{"canvas":null,"canvas":[1,1]}"#,
        r#"{"animation":{},"animation":{}}"#,
        r#"{"canvas":[1]}"#,
        r#"{"canvas":[1,2,3]}"#,
        r#"{"canvas":[-1,2]}"#,
        r#"{"canvas":[1.0,2]}"#,
        r#"{"canvas":[true,2]}"#,
        r#"{"canvas":["1",2]}"#,
        r#"{"canvas":[4294967296,2]}"#,
        r#"{"frames":{}}"#,
        r#"{"frames":[null]}"#,
        r#"{"frames":[[]]}"#,
        r#"{"frames":[[null,null,null,null,null,null,null]]}"#,
        r#"{"frames":[{"reference_only":false}]}"#,
        r#"{"frames":[{"save_before_color_transform":false}]}"#,
        r#"{"frames":[{"origin":[-2147483649,0]}]}"#,
        r#"{"frames":[{"origin":[2147483648,0]}]}"#,
        r#"{"frames":[{"origin":[0]}]}"#,
        r#"{"frames":[{"origin":[false,0]}]}"#,
        r#"{"frames":[{"origin":[0.5,0]}]}"#,
        r#"{"frames":[{"origin":null}]}"#,
        r#"{"frames":[{"name":null}]}"#,
        r#"{"frames":[{"name":1}]}"#,
        r#"{"frames":[{"duration":0,"duration":0}]}"#,
        r#"{"frames":[{"blend":null}]}"#,
        r#"{"frames":[{"blend":[]}]}"#,
        r#"{"frames":[{"blend":["replace",1,0,false]}]}"#,
        r#"{"frames":[{"blend":{"mode":"Replace"}}]}"#,
        r#"{"frames":[{"blend":{"mode":{"replace":null}}}]}"#,
        r#"{"frames":[{"blend":{"mode":"screen"}}]}"#,
        r#"{"frames":[{"blend":{"unknown":0}}]}"#,
        r#"{"frames":[{"blend":{"source":1,"source":1}}]}"#,
        r#"{"frames":[{"blend":{"source":true}}]}"#,
        r#"{"frames":[{"blend":{"source":"1"}}]}"#,
        r#"{"frames":[{"blend":{"clamp":0}}]}"#,
        r#"{"frames":[{"blend":{"clamp":"false"}}]}"#,
        r#"{"frames":[{"extra_blend":null}]}"#,
        r#"{"frames":[{"extra_blend":[null]}]}"#,
        r#"{"frames":[{"extra_blend":[[]]}]}"#,
        r#"{"frames":[{"extra_blend":[{"clamp":false,"clamp":false}]}]}"#,
        r#"{"animation":[]}"#,
        r#"{"animation":[1000,1,0,false]}"#,
        r#"{"animation":{"timebase":[1000,1]}}"#,
        r#"{"animation":{"loops":0,"loops":0}}"#,
        r#"{"animation":{"loops":true}}"#,
        r#"{"animation":{"timecodes":0}}"#,
    ] {
        let wire = format!(
            r#"{{"operation":"encode_layers","images":[{{"length":0}}],"layer_options":{options}}}"#
        );
        assert!(serde_json::from_str::<Request>(&wire).is_err(), "{options}");
        invalid(raw(None, wire.as_bytes(), &[]));
    }
    for field in ["duration", "timecode", "save_as_reference"] {
        for value in [
            json!(null),
            json!(true),
            json!(-1),
            json!(1.5),
            json!("1"),
            json!(4294967296u64),
        ] {
            let mut request = sequence();
            request["layer_options"] = json!({"frames": [{field: value}, {}]});
            malformed(request);
        }
    }
    for field in ["source", "alpha_channel", "clamp", "mode"] {
        for value in [json!(null), json!(-1), json!(1.5), json!(256)] {
            let mut request = sequence();
            request["layer_options"] = json!({"frames": [{"blend": {field: value}}, {}]});
            malformed(request);
        }
    }
    for field in [
        "ticks_per_second_numerator",
        "ticks_per_second_denominator",
        "loops",
        "timecodes",
    ] {
        for value in [
            json!(null),
            json!(-1),
            json!(1.5),
            json!("1"),
            json!(4294967296u64),
        ] {
            let mut request = sequence();
            request["layer_options"] = json!({"animation": {field: value}});
            malformed(request);
        }
    }
    for duplicate in [
        "layer_options",
        "options",
        "images",
        "durations",
        "auto_options",
    ] {
        let value = match duplicate {
            "layer_options" | "options" => "{}",
            "images" => "[]",
            _ => "null",
        };
        let wire = format!(
            r#"{{"operation":"encode_layers","{duplicate}":{value},"{duplicate}":{value}}}"#
        );
        invalid(raw(None, wire.as_bytes(), &[]));
    }
}

#[test]
fn operation_restrictions_preserve_old_null_semantics() {
    for (field, value) in [
        ("durations", json!([0, 0])),
        ("reference", json!("previous")),
        ("timebase", json!([1000, 1])),
        ("loops", json!(0)),
        ("auto", json!(false)),
        ("auto_options", json!({})),
        ("analysis_options", json!({})),
    ] {
        for value in [value, Value::Null] {
            // Operation intentionally comes last: rejection must not depend on key order.
            let wire = format!(
                r#"{{"{field}":{value},"images":[{{"length":0}}],"operation":"encode_layers"}}"#
            );
            invalid(raw(None, wire.as_bytes(), &[]));
        }
    }
    for operation in [
        "encode",
        "encode_multiframe",
        "analyze_multiframe",
        "transcode_jpeg",
        "reconstruct_jpeg",
        "decode",
        "decode_to_png",
        "info",
        "read_image",
    ] {
        invalid(call(
            None,
            json!({"operation": operation, "layer_options": {}}),
            &[],
        ));
    }
    let legacy: Request = serde_json::from_value(json!({
        "operation": "info", "images": null, "durations": null, "reference": null,
        "timebase": null, "loops": null, "auto": null, "auto_options": null,
    }))
    .unwrap();
    legacy.validate().unwrap();
    for images in [json!(null), json!([])] {
        invalid(call(
            None,
            json!({"operation": "encode_layers", "images": images}),
            &[],
        ));
    }
    invalid(call(None, json!({"operation": "encode_layers"}), &[]));
    for frames in [json!([]), json!([{}]), json!([{}, {}, {}])] {
        let mut request = sequence();
        request["layer_options"] = json!({"frames": frames});
        invalid(call(None, request, &[]));
    }
    for options in [
        json!({"source_search": true}),
        json!({"frame_index": 0}),
        json!({"coalescing": false}),
        json!({"format": "native"}),
        json!({"include_extra_channels": true}),
        json!({"adjust_orientation": true}),
        json!({"dither": "jxl-rs"}),
        json!({"limits": {}}),
        json!({"brotli_quality": 1}),
        json!({"no_cfl": false}),
        json!({"order": "source"}),
        json!({"layer_options": {}}),
        json!({"threads": 0}),
        json!({"threads": 257}),
        json!({"effort": 0}),
        json!({"effort": 12}),
        json!({"ma_max_leaves": 0}),
        json!({"group_size_shift": 4}),
        json!({"predictor": 1000}),
        json!({"rct": u32::MAX}),
        json!({"learning_fraction": -1}),
    ] {
        let mut request = sequence();
        request["options"] = options;
        invalid(call(None, request, &[7, 9, 11, 13]));
    }
}

fn frame_bytes<'a>(frame: &Value, data: &'a [u8]) -> &'a [u8] {
    let offset = frame["offset"].as_u64().unwrap() as usize;
    let length = frame["length"].as_u64().unwrap() as usize;
    &data[offset..offset + length]
}

#[test]
fn abi_heterogeneous_layers_keep_raw_shapes_and_native_sample_words() {
    for (dtype, input) in [
        ("u8", vec![7, 9, 11, 13]),
        (
            "u16",
            [0x1234u16, 0xabcd, 0, u16::MAX]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
        (
            "u32",
            [0x7fffffffu32, 0x1234567, 0, 17]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        ),
        (
            "f16",
            [0x8000u16, 0x7e01, 0x3c00, 0x7c00]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
        (
            "f32",
            [0x80000000u32, 0x7fc00123, 0x3f800000, 0x7f800000]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        ),
    ] {
        let mut request = sequence();
        let length = input.len() / 2;
        for index in 0..2 {
            request["images"][index]["dtype"] = json!(dtype);
            request["images"][index]["length"] = json!(length);
            request["images"][index]["offset"] = json!(index * length);
        }
        let (report, bytes) = ok(None, request, &input);
        assert_eq!(report["operation"], "encode_layers");
        assert_eq!(report["bytes"], bytes.len());
        assert_eq!(report["jpeg_input"], false);
        assert_eq!(report["canvas"], json!({"width": 2, "height": 2}));
        let seconds = report["codec_seconds"].as_f64().unwrap();
        assert!(seconds.is_finite() && seconds >= 0.0);
        for (index, (width, height)) in [(2, 1), (1, 2)].into_iter().enumerate() {
            assert_eq!(
                report["layers"][index],
                json!({
                    "index": index, "width": width, "height": height,
                    "origin": [0, 0], "duration": 0, "timecode": 0, "name": "",
                    "blend": {"mode": "replace", "source": 1, "alpha_channel": 0, "clamp": false},
                    "extra_blend": [], "save_as_reference": 0,
                })
            );
        }
        let (decoded, pixels) = ok(
            None,
            json!({"operation": "decode", "options": {"coalescing": false}}),
            &bytes,
        );
        assert_eq!(decoded["frames"].as_array().unwrap().len(), 2);
        // Native half-float decoding expands the words exactly to F32.
        let expected = if dtype == "f16" {
            [0x80000000u32, 0x7fc02000, 0x3f800000, 0x7f800000]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        } else {
            input.clone()
        };
        let output_length = expected.len() / 2;
        for (index, (width, height)) in [(2, 1), (1, 2)].into_iter().enumerate() {
            let frame = &decoded["frames"][index];
            assert_eq!(frame["width"], width);
            assert_eq!(frame["height"], height);
            assert_eq!(frame["dtype"], if dtype == "f16" { "f32" } else { dtype });
            assert_eq!(
                frame_bytes(frame, &pixels),
                &expected[index * output_length..(index + 1) * output_length]
            );
        }
        if dtype == "u8" {
            let (decoded, pixels) = ok(None, json!({"operation": "decode"}), &bytes);
            assert_eq!(decoded["frames"].as_array().unwrap().len(), 1);
            assert_eq!(pixels, [11, 0, 13, 0]);
        }
    }
}

#[test]
fn abi_explicit_canvas_offsets_timing_and_reference_overrides() {
    let input = [7, 9, 11, 13];
    let mut request = sequence();
    request["layer_options"] = json!({
        "canvas": [3, 2],
        "animation": {"ticks_per_second_numerator": 24, "ticks_per_second_denominator": 1, "loops": 2, "timecodes": true},
        "frames": [
            {"origin": [-1, 0], "duration": 2, "timecode": 5, "name": "left", "save_as_reference": 2},
            {"origin": [2, 0], "duration": 3, "timecode": 6, "name": "right", "blend": {"source": 2}},
        ],
    });
    let (report, bytes) = ok(None, request, &input);
    assert_eq!(report["canvas"], json!({"width": 3, "height": 2}));
    assert_eq!(report["layers"][0]["origin"], json!([-1, 0]));
    assert_eq!(report["layers"][0]["save_as_reference"], 2);
    assert_eq!(report["layers"][1]["blend"]["source"], 2);
    let (decoded, pixels) = ok(
        None,
        json!({"operation": "decode", "options": {"coalescing": false}}),
        &bytes,
    );
    assert_eq!(
        decoded["animation"],
        json!({"tps_numerator": 24, "tps_denominator": 1, "num_loops": 2, "have_timecodes": true})
    );
    assert_eq!(pixels, input);
    for (index, (duration, timecode, name)) in
        [(2, 5, "left"), (3, 6, "right")].into_iter().enumerate()
    {
        assert_eq!(report["layers"][index]["duration"], duration);
        assert_eq!(report["layers"][index]["timecode"], timecode);
        assert_eq!(report["layers"][index]["name"], name);
        assert_eq!(decoded["frames"][index]["header"]["timecode"], timecode);
    }
    let (_, pixels) = ok(
        None,
        json!({"operation": "decode", "options": {"frame_index": 1}}),
        &bytes,
    );
    assert_eq!(pixels, [9, 0, 11, 0, 0, 13]);
}

#[test]
fn abi_context_reuses_resources_after_errors_and_results_outlive_context() {
    let context = context("encoder");
    for options in [
        json!({"frames": []}),
        json!({"canvas": [0, 1]}),
        json!({"frames": [{"origin": [-1, 0]}, {}]}),
        json!({"frames": [{}, {"save_as_reference": 1}]}),
        json!({"frames": [{"blend": {"source": 4}}, {}]}),
        json!({"frames": [{"extra_blend": [{}]}, {}]}),
        json!({"animation": {"ticks_per_second_numerator": 0}}),
        json!({"animation": {"ticks_per_second_denominator": 1025}}),
        json!({"animation": {"timecodes": false}, "frames": [{"timecode": 1}, {}]}),
    ] {
        let mut request = sequence();
        request["layer_options"] = options;
        invalid(call(Some(context.0), request, &[7, 9, 11, 13]));
    }
    let mut mismatch = sequence();
    mismatch["options"]["threads"] = json!(1);
    invalid(call(Some(context.0), mismatch, &[7, 9, 11, 13]));
    let mut mismatch = sequence();
    mismatch["images"][1]["color_encoding"] = json!("linear_srgb");
    invalid(call(Some(context.0), mismatch, &[7, 9, 11, 13]));
    let result = call(Some(context.0), sequence(), &[7, 9, 11, 13]);
    let before = result.response();
    assert_eq!(before.0, 0, "{}", before.1);
    let mut timed = sequence();
    timed["layer_options"] = json!({"frames": [
        {"duration": 1}, {"origin": [2, 1], "duration": 2, "timecode": 7}
    ]});
    let (report, bytes) = ok(Some(context.0), timed, &[7, 9, 11, 13]);
    assert_eq!(report["canvas"], json!({"width": 3, "height": 3}));
    let (info, _) = ok(None, json!({"operation": "info"}), &bytes);
    assert_eq!(info["animation"]["tps_numerator"], 1000);
    assert_eq!(info["animation"]["have_timecodes"], true);
    let (_, bytes) = ok(Some(context.0), sequence(), &[7, 9, 11, 13]);
    let (info, _) = ok(None, json!({"operation": "info"}), &bytes);
    assert!(info["animation"].is_null());
    drop(context);
    assert_eq!(result.response(), before);
    let decoder = self::context("decoder");
    let (status, report, data) = call(Some(decoder.0), sequence(), &[7, 9, 11, 13]).response();
    assert_eq!(status, 1);
    assert_eq!(report["error_kind"], "unsupported");
    assert!(data.is_empty());
}
