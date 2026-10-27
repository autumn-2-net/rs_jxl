//! Pure Rust, caller-owned C ABI. Wire protocol: ../protocol.md.
#![deny(unsafe_op_in_unsafe_fn)]

mod abi;
mod contexts;
mod decode;
mod encode;
mod error;
mod pixels;
pub mod request;

pub use abi::*;
pub use contexts::*;
pub use error::{Error, ErrorKind, Result, message};

fn execute_json(json: &[u8], data: &[u8]) -> Result<(serde_json::Value, Vec<u8>)> {
    let request: request::Request = serde_json::from_slice(json).map_err(message)?;
    use request::ValidatedRequest::*;
    match request.validate_typed()? {
        Encode { .. } | EncodeMultiframe { .. } | AnalyzeMultiframe { .. } => {
            encode::execute(&request, data)
        }
        TranscodeJpeg { .. } => encode::transcode(&request, data),
        Decode { .. }
        | DecodeToPng { .. }
        | Info { .. }
        | ReconstructJpeg { .. }
        | ReadImage { .. } => decode::execute(&request, data),
    }
}
