use std::{
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

use rs_jxl::{
    codec::Encoder,
    decode::{DecodeOptions, Decoder},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Error, ErrorKind, Result, RsJxlRequest, RsJxlResult, abi, analysis, decode, encode, layers,
    message,
    request::{Operation, Request, ValidatedRequest},
};

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Encoder,
    Decoder,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextSpec {
    kind: Kind,
    #[serde(default = "default_threads")]
    threads: usize,
}

fn default_threads() -> usize {
    1
}

enum Resources {
    Encoder(Encoder),
    Decoder(Decoder),
}

/// Opaque, caller-owned resources. All access, including free, must be exclusive.
pub struct RsJxlContext {
    resources: Resources,
    threads: usize,
    poisoned: bool,
}

impl RsJxlContext {
    fn execute(&mut self, json: &[u8], data: &[u8]) -> Result<(Value, Vec<u8>)> {
        if self.poisoned {
            return Err(Error::new(
                ErrorKind::Internal,
                "context is unusable after a panic",
            ));
        }
        let mut request: Request = serde_json::from_slice(json).map_err(message)?;
        // Validate supplied fields before adding a context's implicit threads.
        request.validate()?;
        if let Some(threads) = request.options.threads {
            if threads != self.threads {
                return Err("request threads must match the context".into());
            }
        } else if request.operation != Operation::ReadImage {
            request.options.threads = Some(self.threads);
        }
        let request_kind = request.validate_typed()?;
        // Ordinary errors are reusable. A panic leaves this flag set and the
        // outer C boundary catches it without allowing damaged resources reuse.
        self.poisoned = true;
        use ValidatedRequest::*;
        let result = match (&mut self.resources, request_kind) {
            (Resources::Encoder(encoder), Encode { .. } | EncodeMultiframe { .. }) => {
                encode::execute_with_context(&request, data, Some(encoder))
            }
            (Resources::Encoder(encoder), EncodeLayers { .. }) => {
                layers::execute(&request, data, Some(encoder))
            }
            (Resources::Encoder(encoder), TranscodeJpeg { .. }) => {
                encode::transcode_with_context(&request, data, Some(encoder))
            }
            (Resources::Encoder(encoder), AnalyzeMultiframe { .. }) => {
                analysis::execute(&request, data, Some(encoder))
            }
            (
                Resources::Decoder(decoder),
                Decode { .. } | DecodeToPng { .. } | ReconstructJpeg { .. },
            ) => decode::execute_with_context(&request, data, Some(decoder)),
            (Resources::Decoder(_), Info { .. } | ReadImage { .. }) => {
                decode::execute(&request, data)
            }
            _ => Err(Error::new(
                ErrorKind::Unsupported,
                "operation is unsupported by this context kind",
            )),
        };
        self.poisoned = false;
        result
    }
}

/// Create resources from separate JSON: {"kind":"encoder"|"decoder","threads":N}.
/// Threads default to one; no data payload or operation/options keys are accepted.
/// On failure `*out_context` is null. The result always has independent ownership.
///
/// # Safety
/// The request has the same readable-prefix and buffer contract as rs_jxl_execute.
/// out_context must be null or writable for one pointer, with no overlap with
/// request/input storage. Unaligned output pointer storage is supported. A
/// nonnull existing handle must be freed before reusing its output slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_context_create(
    request: *const RsJxlRequest,
    out_context: *mut *mut RsJxlContext,
) -> *mut RsJxlResult {
    Box::into_raw(Box::new(abi::protect(|| {
        let create = || -> Result<RsJxlResult> {
            if out_context.is_null() {
                return Err("null context output pointer".into());
            }
            unsafe { ptr::write_unaligned(out_context, ptr::null_mut()) };
            let (json, data) = unsafe { abi::request_input(request) }?;
            let spec: ContextSpec = serde_json::from_slice(json).map_err(message)?;
            if !data.is_empty() {
                return Err("context creation does not accept data".into());
            }
            if !(1..=256).contains(&spec.threads) {
                return Err("threads must be 1..256".into());
            }
            let (kind, resources) = match spec.kind {
                Kind::Encoder => (
                    "encoder",
                    Resources::Encoder(Encoder::new(spec.threads).map_err(message)?),
                ),
                Kind::Decoder => (
                    "decoder",
                    Resources::Decoder(
                        Decoder::new(DecodeOptions {
                            threads: spec.threads,
                            ..Default::default()
                        })
                        .map_err(message)?,
                    ),
                ),
            };
            // Serialize success before transferring ownership to the caller.
            let result =
                RsJxlResult::success(json!({"kind": kind, "threads": spec.threads}), Vec::new())?;
            let context = Box::new(RsJxlContext {
                resources,
                threads: spec.threads,
                poisoned: false,
            });
            unsafe { ptr::write_unaligned(out_context, Box::into_raw(context)) };
            Ok(result)
        };
        create().map_err(|error| error.at_stage("context_create"))
    })))
}

/// Execute JSON v1 with fixed context threads. Omitted threads inherit; explicit
/// mismatches fail. All other options use per-call defaults, never stored defaults.
/// Encoder contexts also reuse workers for analysis; decoder contexts allow
/// stateless info/read_image. Ordinary errors permit reuse; a panic poisons it.
///
/// # Safety
/// context must be null or a live handle created by rs_jxl_context_create. The
/// caller must serialize all calls/free for this handle. Request storage follows
/// rs_jxl_execute and must not alias context storage. No input pointers are retained.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_context_execute(
    context: *mut RsJxlContext,
    request: *const RsJxlRequest,
) -> *mut RsJxlResult {
    Box::into_raw(Box::new(abi::protect(|| {
        let execute = || -> Result<RsJxlResult> {
            if context.is_null() {
                return Err("null context".into());
            }
            let (json, data) = unsafe { abi::request_input(request) }?;
            let (json, data) = unsafe { &mut *context }.execute(json, data)?;
            RsJxlResult::success(json, data)
        };
        execute().map_err(|error| error.at_stage("context_execute"))
    })))
}

/// # Safety
/// context must be null or a live context, freed exactly once with no concurrent
/// execution/free. Results remain independently valid after the context is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_context_free(context: *mut RsJxlContext) {
    if !context.is_null()
        && let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(context) });
        }))
    {
        // As at the execute boundary, even a hostile panic payload must not unwind.
        mem::forget(payload);
    }
}
