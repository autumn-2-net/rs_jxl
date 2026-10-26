use rs_jxl_ffi::*;
use serde_json::{Value, json};
use std::{mem, ptr, slice};

struct Owned(*mut RsJxlResult);
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { rs_jxl_result_free(self.0) };
    }
}

fn response(request: *const RsJxlRequest) -> (i32, Value, Vec<u8>) {
    let result = Owned(unsafe { rs_jxl_execute(request) });
    assert!(!result.0.is_null());
    let mut length = 0;
    let json = unsafe { rs_jxl_result_json(result.0, &mut length) };
    assert!(!json.is_null());
    let json = serde_json::from_slice(unsafe { slice::from_raw_parts(json, length) }).unwrap();
    let data = unsafe { rs_jxl_result_data(result.0, &mut length) };
    let data = if length == 0 {
        assert!(data.is_null());
        Vec::new()
    } else {
        unsafe { slice::from_raw_parts(data, length) }.to_vec()
    };
    (unsafe { rs_jxl_result_status(result.0) }, json, data)
}

fn call(json: Value, data: &[u8]) -> (i32, Value, Vec<u8>) {
    let json = serde_json::to_vec(&json).unwrap();
    response(&RsJxlRequest {
        abi_version: 1,
        struct_size: rs_jxl_request_size(),
        json: json.as_ptr(),
        json_len: json.len(),
        data: data.as_ptr(),
        data_len: data.len(),
    })
}

fn ok(json: Value, data: &[u8]) -> (Value, Vec<u8>) {
    let (code, meta, data) = call(json, data);
    assert_eq!(code, 0, "{meta}");
    (meta, data)
}

fn spec(dtype: &str, channels: u8, length: usize) -> Value {
    json!({"format":"pixels","width":2,"height":1,"channels":channels,"dtype":dtype,"length":length})
}

fn encode(spec: Value, data: &[u8]) -> Vec<u8> {
    ok(
        json!({"operation":"encode","options":{"effort":2},"images":[spec]}),
        data,
    )
    .1
}

#[test]
fn version_null_and_short_prefix_contracts() {
    assert_eq!(rs_jxl_abi_version(), 1);
    assert_eq!(rs_jxl_request_size(), mem::size_of::<RsJxlRequest>());
    assert_eq!(response(ptr::null()).0, 1);
    #[repr(C)]
    struct Prefix {
        version: u32,
        size: usize,
    }
    for prefix in [
        Prefix {
            version: 2,
            size: rs_jxl_request_size(),
        },
        Prefix {
            version: 1,
            size: mem::size_of::<Prefix>(),
        },
        Prefix {
            version: 1,
            size: usize::MAX,
        },
    ] {
        // Deliberately only prefix storage: size mismatch must not read pointers.
        let (code, meta, data) = response((&prefix as *const Prefix).cast());
        assert_eq!(code, 1);
        assert!(meta["error"].is_string());
        assert!(data.is_empty());
    }
    unsafe {
        let mut n = 7;
        assert_eq!(rs_jxl_result_status(ptr::null()), -1);
        assert!(rs_jxl_result_json(ptr::null(), &mut n).is_null());
        assert_eq!(n, 0);
        n = 7;
        assert!(rs_jxl_result_data(ptr::null(), &mut n).is_null());
        assert_eq!(n, 0);
        rs_jxl_result_free(ptr::null_mut());
    }
}

#[test]
fn rejects_invalid_buffer_lengths_before_access() {
    let mut request = RsJxlRequest {
        abi_version: 1,
        struct_size: rs_jxl_request_size(),
        json: ptr::null(),
        json_len: 1,
        data: ptr::null(),
        data_len: 0,
    };
    assert_eq!(response(&request).0, 1);
    request.json = ptr::dangling();
    request.json_len = isize::MAX as usize + 1;
    assert_eq!(response(&request).0, 1);
    request.json = (usize::MAX - 3) as *const u8;
    request.json_len = 8;
    assert_eq!(response(&request).0, 1);
    request.json = ptr::null();
    request.json_len = 0;
    request.data_len = 1;
    assert_eq!(response(&request).0, 1);
}

#[test]
fn results_are_independent_owned_and_json_has_no_nul() {
    let a = Owned(unsafe { rs_jxl_execute(ptr::null()) });
    let json = b"{invalid";
    let request = RsJxlRequest {
        abi_version: 1,
        struct_size: rs_jxl_request_size(),
        json: json.as_ptr(),
        json_len: json.len(),
        data: ptr::null(),
        data_len: 0,
    };
    let b = Owned(unsafe { rs_jxl_execute(&request) });
    let mut n = 0;
    let pointer = unsafe { rs_jxl_result_json(a.0, &mut n) };
    let before = unsafe { slice::from_raw_parts(pointer, n) }.to_vec();
    drop(b);
    assert_eq!(unsafe { slice::from_raw_parts(pointer, n) }, before);
    assert!(!before.contains(&0));
    assert!(unsafe { rs_jxl_result_json(a.0, ptr::null_mut()) } == pointer);
}

#[test]
fn rejects_unknown_inapplicable_and_out_of_bounds_fields() {
    for request in [
        json!({"operation":"info","surprise":true}),
        json!({"operation":"decode","options":{"cjxl_flags":"-e9"}}),
        json!({"operation":"decode","options":{"effort":7}}),
        json!({"operation":"info","images":[]}),
        json!({"operation":"analyze_multiframe","images":[spec("u8",1,2)],"auto":false}),
        json!({"operation":"info","options":{"limits":{"max_tokens":1}}}),
        json!({"operation":"encode","images":[{"width":1,"height":1,"channels":1,"dtype":"u8","offset":usize::MAX,"length":1}]}),
        json!({"operation":"encode","images":[{"width":u32::MAX,"height":u32::MAX,"channels":4,"dtype":"f32","length":1}]}),
        json!({"operation":"encode","images":[spec("u16",1,3)]}),
        json!({"operation":"encode","images":[{"format":"encoded","length":1,"orientation":1}]}),
    ] {
        let (code, meta, data) = call(request, &[0, 0, 0]);
        assert_eq!(code, 1, "{meta}");
        assert!(meta["error"].is_string());
        assert!(data.is_empty());
    }
}

#[test]
fn analysis_rejects_every_encoding_only_sequence_field() {
    let base = json!({
        "operation": "analyze_multiframe",
        "images": [spec("u8", 1, 2)],
    });
    assert_eq!(call(base.clone(), &[0, 1]).0, 0);
    for (field, value) in [
        ("durations", json!([1])),
        ("timebase", json!([1000, 1])),
        ("loops", json!(0)),
        ("reference", json!("previous")),
        ("auto", json!(false)),
    ] {
        let mut request = base.clone();
        request[field] = value;
        let (code, metadata, data) = call(request, &[0, 1]);
        assert_eq!(code, 1, "{field}: {metadata}");
        assert_eq!(
            metadata["error"],
            "sequence fields require encode_multiframe"
        );
        assert!(data.is_empty());
    }
}

#[test]
fn unaligned_little_endian_pixels_roundtrip_and_color_info() {
    let input = [0xff, 0x34, 0x12, 0xcd, 0xab];
    let mut image = spec("u16", 1, 4);
    image["offset"] = json!(1);
    image["color_encoding"] = json!("linear_srgb");
    let encoded = encode(image, &input);
    let (meta, pixels) = ok(
        json!({"operation":"decode","options":{"frame_index":0}}),
        &encoded,
    );
    assert_eq!(pixels, &input[1..]);
    assert_eq!(meta["frames"][0]["dtype"], "u16");
    assert_eq!(meta["frames"][0]["length"], 4);
    assert_eq!(meta["color_encoding"]["transfer_function"], "linear");
    let (meta, data) = ok(json!({"operation":"info"}), &encoded);
    assert!(data.is_empty());
    assert_eq!(meta["bits_per_sample"], 16);
    assert_eq!(meta["color_encoding"]["name"], "linear_srgb");
    assert_eq!(
        call(
            json!({"operation":"decode","options":{"limits":{"max_output_bytes":1}}}),
            &encoded
        )
        .0,
        1
    );
}

#[test]
fn separate_depth_channel_and_alpha_metadata() {
    let input = [3, 0, 7, 255, 41, 99];
    let mut image = spec("u8", 2, 4);
    image["extra_channels"] =
        json!([{"type":"depth","name":"z","dtype":"u8","offset":4,"length":2}]);
    let encoded = encode(image, &input);
    let (meta, data) = ok(json!({"operation":"decode"}), &encoded);
    let extras = &meta["frames"][0]["extra_channels"];
    assert_eq!(extras[0]["type"], "alpha");
    assert_eq!(extras[1]["type"], "depth");
    assert_eq!(extras[1]["name"], "z");
    let offset = extras[1]["offset"].as_u64().unwrap() as usize;
    assert_eq!(&data[offset..offset + 2], &input[4..]);
    assert_eq!(call(json!({"operation":"decode_to_png"}), &encoded).0, 1);
}

#[test]
fn png_export_preserves_sixteen_bit_samples() {
    let input = [0x34, 0x12, 0xcd, 0xab];
    let encoded = encode(spec("u16", 1, 4), &input);
    let (_, png) = ok(json!({"operation":"decode_to_png"}), &encoded);
    assert!(png.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]));
    let (meta, data) = ok(json!({"operation":"read_image"}), &png);
    assert_eq!(meta["frames"][0]["dtype"], "u16");
    assert_eq!(data, input);
}

#[test]
fn raw_zero_duration_and_difference_bounds_are_retained() {
    let mut a = spec("u8", 1, 2);
    a["offset"] = json!(0);
    let mut b = spec("u8", 1, 2);
    b["offset"] = json!(2);
    let input = [0, 1, 0, 2];
    let (analysis, _) = ok(
        json!({"operation":"analyze_multiframe","images":[a,b]}),
        &input,
    );
    assert_eq!(
        analysis["frames"][1]["previous"],
        json!({"x":1,"y":0,"width":1,"height":1})
    );
    let (_, encoded) = ok(
        json!({"operation":"encode_multiframe","options":{"effort":2},"images":[a,b],"durations":[0,3],"timebase":[1000,1],"reference":"previous"}),
        &input,
    );
    let (meta, _) = ok(
        json!({"operation":"decode","options":{"coalescing":false}}),
        &encoded,
    );
    assert_eq!(meta["frames"][0]["duration"], 0);
    assert_eq!(meta["frames"][1]["duration"], 3);
    assert!(meta["frames"][1]["header"]["layer_info"]["crop_origin"].is_array());
    assert_eq!(meta["animation"]["tps_numerator"], 1000);
}

#[test]
fn jpeg_reconstruction_is_byte_exact() {
    let mut jpeg = vec![0xff, 0xd8, 0xff, 0xdb, 0, 67, 0];
    jpeg.extend([16; 64]);
    jpeg.extend([
        0xff, 0xc0, 0, 11, 8, 0, 1, 0, 1, 1, 1, 0x11, 0, 0xff, 0xc4, 0, 38,
    ]);
    for selector in [0, 0x10] {
        jpeg.extend([selector, 1]);
        jpeg.extend([0; 15]);
        jpeg.push(0);
    }
    jpeg.extend([0xff, 0xda, 0, 8, 1, 1, 0, 0, 63, 0, 0x3f, 0xff, 0xd9]);
    jpeg.extend(b"trailing bytes");
    let (_, encoded) = ok(
        json!({"operation":"transcode_jpeg","options":{"effort":1}}),
        &jpeg,
    );
    let (_, restored) = ok(json!({"operation":"reconstruct_jpeg"}), &encoded);
    assert_eq!(restored, jpeg);
}
