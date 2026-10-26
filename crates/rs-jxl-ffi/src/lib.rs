//! Pure Rust, caller-owned C ABI. Wire protocol: ../protocol.md.
#![deny(unsafe_op_in_unsafe_fn)]

mod abi;
mod decode;
mod encode;
mod pixels;
mod request;

pub use abi::*;

type Result<T> = std::result::Result<T, String>;

fn message(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn execute_json(json: &[u8], data: &[u8]) -> Result<(serde_json::Value, Vec<u8>)> {
    let request: request::Request = serde_json::from_slice(json).map_err(message)?;
    request.validate()?;
    use request::Operation::*;
    match request.operation {
        Encode | EncodeMultiframe | AnalyzeMultiframe => encode::execute(&request, data),
        TranscodeJpeg => encode::transcode(&request, data),
        Decode | DecodeToPng | Info | ReconstructJpeg | ReadImage => {
            decode::execute(&request, data)
        }
    }
}
