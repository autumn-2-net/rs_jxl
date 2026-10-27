use std::{error::Error as StdError, fmt, io, ptr, slice};

use rs_jxl_ffi::{
    request::{Request, ValidatedRequest},
    *,
};
use serde_json::{Value, json};

#[test]
fn core_categories_preserve_messages_without_text_classification() {
    for (source, kind) in [
        (
            rs_jxl::Error::InvalidInput("resource limit: misleading text".into()),
            ErrorKind::InvalidInput,
        ),
        (
            rs_jxl::Error::Limit("test budget"),
            ErrorKind::ResourceLimit,
        ),
        (
            rs_jxl::Error::Encoding("invalid input".into()),
            ErrorKind::Encoding,
        ),
        (
            rs_jxl::Error::Decoding("unsupported format".into()),
            ErrorKind::Decoding,
        ),
        (
            rs_jxl::Error::Io(io::Error::other("encoding failed")),
            ErrorKind::Io,
        ),
    ] {
        let expected = source.to_string();
        let error = message(source);
        assert_eq!(error.kind, kind);
        assert_eq!(error.message, expected);
        assert_eq!(error.to_string(), expected);
        assert_eq!(error.stage, None);
    }
    for error in [
        Error::from("resource limit: text"),
        Error::from(String::from("internal panic")),
    ] {
        assert_eq!(error.kind, ErrorKind::InvalidInput);
    }
    let error = Error::new(ErrorKind::Unsupported, "message")
        .at_stage("image")
        .at_stage("execute");
    assert_eq!(message(error.clone()), error);
    assert_eq!(error.stage, Some("image"));
    assert_eq!(error.to_string(), "message");
}

#[derive(Debug)]
struct Wrapped(rs_jxl::Error);

impl fmt::Display for Wrapped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reader failed: {}", self.0)
    }
}

impl StdError for Wrapped {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.0)
    }
}

#[test]
fn boxed_reader_errors_recover_known_types_and_outer_messages() {
    let source: Box<dyn StdError> = Box::new(rs_jxl::Error::Limit("boxed budget"));
    assert_eq!(message(source).kind, ErrorKind::ResourceLimit);
    let source: Box<dyn StdError> = Box::new(Wrapped(rs_jxl::Error::Decoding("bad input".into())));
    let error = message(source);
    assert_eq!(error.kind, ErrorKind::Decoding);
    assert_eq!(error.message, "reader failed: bad input");
    let source: Box<dyn StdError> = "resource limit: unknown reader text".into();
    assert_eq!(message(source).kind, ErrorKind::InvalidInput);
    let source: Box<dyn StdError + Send + Sync> =
        Box::new(Error::new(ErrorKind::ResourceLimit, "bounded output").at_stage("png"));
    let error = message(source);
    assert_eq!(error.kind, ErrorKind::ResourceLimit);
    assert_eq!(error.stage, Some("png"));
    let error = message(rs_jxl_image::decode(b"not an image", false).err().unwrap());
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(
        error.message,
        "unsupported input: expected a PNG or JPEG signature"
    );
}

#[test]
fn serde_allocation_and_png_errors_keep_known_categories() {
    let source = serde_json::from_slice::<Value>(b"{").unwrap_err();
    let expected = source.to_string();
    let error = message(source);
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.message, expected);
    let source = serde_json::Error::io(io::Error::other("JSON reader failed"));
    assert_eq!(message(source).kind, ErrorKind::Io);
    let source = Vec::<u8>::new().try_reserve(usize::MAX).unwrap_err();
    assert_eq!(message(source).kind, ErrorKind::ResourceLimit);
    assert_eq!(
        message(png::EncodingError::LimitsExceeded).kind,
        ErrorKind::ResourceLimit
    );
    let source: Box<dyn StdError> = Box::new(png::DecodingError::LimitsExceeded);
    assert_eq!(message(source).kind, ErrorKind::ResourceLimit);
    let source = png::EncodingError::IoError(io::Error::other(
        Error::new(ErrorKind::ResourceLimit, "output limit").at_stage("png"),
    ));
    let error = message(source);
    assert_eq!(error.kind, ErrorKind::ResourceLimit);
    assert_eq!(error.message, "output limit");
    assert_eq!(error.stage, Some("png"));
    assert_eq!(
        message(io::Error::other("resource limit: only text")).kind,
        ErrorKind::Io
    );
}

fn parsed(value: Value) -> Request {
    serde_json::from_value(value).unwrap()
}

#[test]
fn typed_requests_keep_shape_defaults_and_explicit_zero_durations() {
    let request = parsed(json!({"operation":"encode", "images":[{"length":0}]}));
    request.validate().unwrap();
    let ValidatedRequest::Encode { image, .. } = request.validate_typed().unwrap() else {
        panic!()
    };
    assert!(ptr::eq(image, &request.images()[0]));
    // Image sample validation is intentionally still deferred to pixels.rs.
    assert!(image.width.is_none());
    let request =
        parsed(json!({"operation":"encode_multiframe", "images":[{"length":0}], "durations":[0]}));
    let ValidatedRequest::EncodeMultiframe {
        sequence, images, ..
    } = request.validate_typed().unwrap()
    else {
        panic!()
    };
    assert_eq!(images.len(), 1);
    assert_eq!(sequence.durations, Some(&[0][..]));
    assert_eq!(sequence.reference, rs_jxl::multiframe::Reference::Previous);
    assert_eq!(sequence.animation.ticks_per_second_numerator, 1000);
    assert_eq!(sequence.animation.ticks_per_second_denominator, 1);
    assert_eq!(sequence.animation.loops, 0);
    assert!(!sequence.automatic);
    let request = parsed(json!({"operation":"decode_to_png"}));
    assert!(matches!(
        request.validate_typed().unwrap(),
        ValidatedRequest::DecodeToPng { frame_index: 0, .. }
    ));
    let request = parsed(json!({"operation":"decode"}));
    assert!(matches!(
        request.validate_typed().unwrap(),
        ValidatedRequest::Decode {
            frame_index: None,
            ..
        }
    ));
    for operation in ["transcode_jpeg", "reconstruct_jpeg", "info", "read_image"] {
        parsed(json!({"operation":operation}))
            .validate_typed()
            .unwrap();
    }
}

#[test]
fn validation_keeps_presence_rules_and_error_precedence() {
    for (value, expected) in [
        (
            json!({"operation":"analyze_multiframe", "auto":false}),
            "sequence fields require encode_multiframe",
        ),
        (
            json!({"operation":"info", "images":[]}),
            "this operation consumes the entire data buffer and does not accept images",
        ),
        (
            json!({"operation":"encode"}),
            "encode requires exactly one image",
        ),
        (
            json!({"operation":"encode_multiframe", "images":[]}),
            "multiframe requires nonempty images",
        ),
        (
            json!({"operation":"decode", "options":{"effort":7}}),
            "option effort is unsupported for Decode",
        ),
        (
            json!({"operation":"decode", "options":{"threads":0}}),
            "threads must be 1..256",
        ),
    ] {
        let request = parsed(value);
        let error = request.validate_typed().unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput);
        assert_eq!(error.message, expected);
        assert_eq!(request.validate().unwrap_err(), error);
    }
    parsed(json!({"operation":"info", "images":null, "auto":null, "options":{"effort":null}}))
        .validate_typed()
        .unwrap();
    for json in [
        r#"{"operation":"info","unknown":true}"#,
        r#"{"operation":"info","operation":"info"}"#,
        r#"{"operation":"info","options":{"threads":1,"threads":1}}"#,
        r#"{"operation":"info","options":{"limits":{"unknown":1}}}"#,
    ] {
        assert!(serde_json::from_str::<Request>(json).is_err());
    }
}

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

fn context(kind: &str, threads: usize) -> OwnedContext {
    let json = serde_json::to_vec(&json!({"kind":kind,"threads":threads})).unwrap();
    let mut context = ptr::null_mut();
    let result = OwnedResult(unsafe { rs_jxl_context_create(&wire(&json, &[]), &mut context) });
    let (status, metadata, data) = result.response();
    assert_eq!(status, 0, "{metadata}");
    assert_eq!(metadata, json!({"kind":kind,"threads":threads}));
    assert!(data.is_empty());
    assert!(!context.is_null());
    OwnedContext(context)
}

#[test]
fn abi_error_messages_status_and_known_resource_category_are_preserved() {
    let (status, value, data) = OwnedResult(unsafe { rs_jxl_execute(ptr::null()) }).response();
    assert_eq!(status, 1);
    assert_eq!(
        value,
        json!({"error":"null request","error_kind":"invalid_input"})
    );
    assert!(data.is_empty());
    let mut request = wire(&[], &[]);
    request.abi_version = 2;
    let (status, value, _) = OwnedResult(unsafe { rs_jxl_execute(&request) }).response();
    assert_eq!(status, 1);
    assert_eq!(value["error"], "unsupported ABI version");
    assert_eq!(value["error_kind"], "unsupported");
    let (status, value, data) = call(
        None,
        json!({"operation":"decode","options":{"limits":{"max_input_bytes":0}}}),
        &[0],
    )
    .response();
    assert_eq!(status, 1);
    assert_eq!(value["error_kind"], "resource_limit");
    assert_eq!(value["error"], "resource limit: JXL input bytes");
    assert!(data.is_empty());
}

#[test]
fn context_constructor_rejects_invalid_specs_and_clears_output() {
    for value in [
        json!({"kind":"encoder","threads":0}),
        json!({"kind":"decoder","threads":257}),
        json!({"kind":"other","threads":1}),
        json!({"kind":"decoder","options":{}}),
        json!({"kind":"encoder","operation":"encode"}),
    ] {
        let json = serde_json::to_vec(&value).unwrap();
        let mut output = ptr::dangling_mut();
        let result = OwnedResult(unsafe { rs_jxl_context_create(&wire(&json, &[]), &mut output) });
        let (status, value, data) = result.response();
        assert_eq!(status, 1);
        assert_eq!(value["error_kind"], "invalid_input");
        assert_eq!(value["error_stage"], "context_create");
        assert!(output.is_null());
        assert!(data.is_empty());
    }
    let result = OwnedResult(unsafe { rs_jxl_context_create(ptr::null(), ptr::null_mut()) });
    assert_eq!(result.response().1["error"], "null context output pointer");
    let json = br#"{"kind":"encoder","threads":1}"#;
    let mut output = ptr::null_mut();
    let result = OwnedResult(unsafe { rs_jxl_context_create(&wire(json, &[0]), &mut output) });
    assert_eq!(
        result.response().1["error"],
        "context creation does not accept data"
    );
    assert!(output.is_null());
    assert_eq!(
        call(Some(ptr::null_mut()), json!({"operation":"decode"}), &[])
            .response()
            .1["error"],
        "null context"
    );
    unsafe { rs_jxl_context_free(ptr::null_mut()) };
}

#[test]
fn contexts_inherit_threads_recover_after_errors_and_reset_decoder_options() {
    let encoder = context("encoder", 2);
    let request = json!({"operation":"encode","options":{"effort":2},"images":[{
        "length":1,"width":1,"height":1,"channels":1,"dtype":"u8"
    }]});
    let mut mismatch = request.clone();
    mismatch["options"]["threads"] = json!(1);
    let (status, error, _) = call(Some(encoder.0), mismatch, &[17]).response();
    assert_eq!(status, 1);
    assert_eq!(error["error"], "request threads must match the context");
    assert_eq!(error["error_kind"], "invalid_input");
    let (status, error, _) = call(Some(encoder.0), json!({"operation":"decode"}), &[]).response();
    assert_eq!(status, 1);
    assert_eq!(error["error_kind"], "unsupported");
    let (status, metadata, encoded) = call(Some(encoder.0), request, &[17]).response();
    assert_eq!(status, 0, "{metadata}");
    assert!(!encoded.is_empty());
    let decoder = context("decoder", 2);
    let (status, error, _) = call(
        Some(decoder.0),
        json!({"operation":"decode","options":{"limits":{"max_input_bytes":0}}}),
        &encoded,
    )
    .response();
    assert_eq!(status, 1);
    assert_eq!(error["error_kind"], "resource_limit");
    let (status, metadata, pixels) = call(
        Some(decoder.0),
        json!({"operation":"decode","options":{"format":"u16"}}),
        &encoded,
    )
    .response();
    assert_eq!(status, 0, "{metadata}");
    assert_eq!(metadata["frames"][0]["dtype"], "u16");
    assert_eq!(pixels.len(), 2);
    let result = call(Some(decoder.0), json!({"operation":"decode"}), &encoded);
    drop(decoder);
    drop(encoder);
    let (status, metadata, pixels) = result.response();
    assert_eq!(status, 0, "{metadata}");
    assert_eq!(metadata["frames"][0]["dtype"], "u8");
    assert_eq!(pixels, [17]);
}
