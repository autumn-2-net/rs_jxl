use std::{
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

pub const RS_JXL_ABI_VERSION: u32 = 1;

#[repr(C)]
pub struct RsJxlRequest {
    pub abi_version: u32,
    pub struct_size: usize,
    pub json: *const u8,
    pub json_len: usize,
    pub data: *const u8,
    pub data_len: usize,
}

/// Opaque to C. Immutable buffers are owned until rs_jxl_result_free.
pub struct RsJxlResult {
    json: Vec<u8>,
    data: Vec<u8>,
    code: i32,
}

impl RsJxlResult {
    fn error(code: i32, error: crate::Error) -> Self {
        let mut json = serde_json::json!({
            "error": error.message,
            "error_kind": error.kind,
        });
        if let Some(stage) = error.stage {
            json["error_stage"] = stage.into();
        }
        Self {
            json: json.to_string().into_bytes(),
            data: Vec::new(),
            code,
        }
    }

    pub(crate) fn success(json: serde_json::Value, data: Vec<u8>) -> crate::Result<Self> {
        Ok(Self {
            json: serde_json::to_vec(&json).map_err(|error| {
                crate::Error::new(crate::ErrorKind::Internal, error.to_string())
                    .at_stage("serialize_response")
            })?,
            data,
            code: 0,
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rs_jxl_abi_version() -> u32 {
    RS_JXL_ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn rs_jxl_request_size() -> usize {
    mem::size_of::<RsJxlRequest>()
}

fn bounds(pointer: *const u8, length: usize) -> crate::Result<()> {
    if length > isize::MAX as usize {
        return Err("buffer length exceeds isize::MAX".into());
    }
    if length != 0 && pointer.is_null() {
        return Err("nonempty buffer has null pointer".into());
    }
    if (pointer as usize).checked_add(length).is_none() {
        return Err("buffer address overflow".into());
    }
    Ok(())
}

unsafe fn input<'a>(pointer: *const u8, length: usize) -> crate::Result<&'a [u8]> {
    bounds(pointer, length)?;
    if length == 0 {
        return Ok(&[]);
    }
    // Caller guarantees readable immutable storage for the validated length.
    Ok(unsafe { slice::from_raw_parts(pointer, length) })
}

pub(crate) unsafe fn request_input<'a>(
    request: *const RsJxlRequest,
) -> crate::Result<(&'a [u8], &'a [u8])> {
    if request.is_null() {
        return Err("null request".into());
    }
    // Only the two-field ABI prefix is read before version/size validation.
    let size_offset = mem::offset_of!(RsJxlRequest, struct_size);
    bounds(request.cast(), size_offset + mem::size_of::<usize>())?;
    let version = unsafe { ptr::read_unaligned(request.cast::<u32>()) };
    if version != RS_JXL_ABI_VERSION {
        return Err(crate::Error::new(
            crate::ErrorKind::Unsupported,
            "unsupported ABI version",
        ));
    }
    let size =
        unsafe { ptr::read_unaligned(request.cast::<u8>().add(size_offset).cast::<usize>()) };
    if size != mem::size_of::<RsJxlRequest>() {
        return Err("request struct_size mismatch".into());
    }
    bounds(request.cast(), size)?;
    let request = unsafe { ptr::read_unaligned(request) };
    // Validate both regions before creating either borrowed slice.
    bounds(request.json, request.json_len)?;
    bounds(request.data, request.data_len)?;
    let json = unsafe { input(request.json, request.json_len)? };
    let data = unsafe { input(request.data, request.data_len)? };
    Ok((json, data))
}

unsafe fn execute(request: *const RsJxlRequest) -> crate::Result<RsJxlResult> {
    let (json, data) = unsafe { request_input(request) }?;
    let (json, data) = crate::execute_json(json, data)?;
    RsJxlResult::success(json, data)
}

/// Execute synchronously. No input pointers are retained.
///
/// # Safety
/// A nonnull request must point to readable storage for the native ABI prefix
/// (abi_version and struct_size). If that prefix matches v1, the full struct
/// must be readable. Nonempty input buffers must each be within one allocation,
/// readable and immutable for the call. Invalid/dangling pointers cannot be
/// detected by this API. Unaligned structs and sample buffers are supported.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_execute(request: *const RsJxlRequest) -> *mut RsJxlResult {
    Box::into_raw(Box::new(protect(|| unsafe { execute(request) })))
}

pub(crate) fn protect(operation: impl FnOnce() -> crate::Result<RsJxlResult>) -> RsJxlResult {
    let result = catch_unwind(AssertUnwindSafe(operation));
    match result {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => RsJxlResult::error(1, error),
        Err(payload) => {
            // A hostile panic payload may itself panic in Drop; do not unwind into C.
            mem::forget(payload);
            RsJxlResult::error(
                2,
                crate::Error::new(crate::ErrorKind::Internal, "Rust panic during execution"),
            )
        }
    }
}

/// # Safety
/// result must be null or a live result from rs_jxl_execute, not concurrently freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_result_status(result: *const RsJxlResult) -> i32 {
    if result.is_null() {
        -1
    } else {
        unsafe { (*result).code }
    }
}

unsafe fn buffer(result: *const RsJxlResult, length: *mut usize, json: bool) -> *const u8 {
    if !length.is_null() {
        unsafe { ptr::write_unaligned(length, 0) };
    }
    if result.is_null() {
        return ptr::null();
    }
    let result = unsafe { &*result };
    let bytes = if json { &result.json } else { &result.data };
    if !length.is_null() {
        unsafe { ptr::write_unaligned(length, bytes.len()) };
    }
    if bytes.is_empty() {
        ptr::null()
    } else {
        bytes.as_ptr()
    }
}

/// # Safety
/// result must be null or live. length must be null or writable for one usize,
/// and must not alias result storage. Returned bytes must not be modified and
/// remain valid only until result is freed; concurrent free is forbidden.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_result_json(
    result: *const RsJxlResult,
    length: *mut usize,
) -> *const u8 {
    unsafe { buffer(result, length, true) }
}

/// # Safety
/// Same pointer, aliasing, lifetime and concurrency contract as rs_jxl_result_json.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_result_data(
    result: *const RsJxlResult,
    length: *mut usize,
) -> *const u8 {
    unsafe { buffer(result, length, false) }
}

/// # Safety
/// result must be null or a live result from rs_jxl_execute. Free exactly once,
/// with no concurrent getters/readers; all returned buffer pointers expire.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rs_jxl_result_free(result: *mut RsJxlResult) {
    if !result.is_null() {
        drop(unsafe { Box::from_raw(result) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_is_a_result_not_an_unwind() {
        let result = protect(|| panic!("test panic"));
        assert_eq!(result.code, 2);
        let json: serde_json::Value = serde_json::from_slice(&result.json).unwrap();
        assert_eq!(json["error"], "Rust panic during execution");
        assert_eq!(json["error_kind"], "internal");
        assert!(result.data.is_empty());
    }
}
