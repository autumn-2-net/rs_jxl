//! Native Rust PJA algorithms and an experimental hashless jxl_plus backend.
//!
//! The currently exported API reuses rs-jxl's JPEG reconstruction. The
//! broader port is internal while the subsequent JXL-plus-extension format
//! is designed; PJA archive interoperability is not a public contract.
//! See PROVENANCE.md for translated sources and their separate licenses.
#![deny(unsafe_op_in_unsafe_fn)]

mod arith;
#[allow(dead_code)]
mod codec;
mod container;
// These source-port modules are exercised through the internal entry and tests.
// Their public format integration follows the algorithm port; retain source
// entry points until then without warning on every normal build.
#[allow(dead_code)]
mod j2k;
#[allow(dead_code)]
mod jls;
#[allow(dead_code)]
mod jpeg_native;
#[allow(dead_code)]
mod legacy_model;
mod mix_simd;
mod model;
#[allow(dead_code)]
mod ported;
mod predict;
mod probability;
#[allow(dead_code)]
mod raw_model;
#[allow(dead_code)]
mod sample_model;

#[cfg(test)]
mod oracle_tests;

#[cfg(test)]
mod malformed_tests;

#[cfg(test)]
mod static_tests;

#[cfg(test)]
mod tool_tests;

#[cfg(test)]
mod ma_tool_tests;

#[cfg(test)]
mod dynamic_ma_tests;

#[cfg(test)]
mod search_tools_tests;

#[cfg(test)]
mod speed_tools_tests;

#[cfg(test)]
mod corpus_tools_tests;

#[cfg(test)]
mod api_speed_tests;

#[cfg(test)]
mod standard_corpus_tests;

#[cfg(test)]
mod fast_parallel_speed_tests;

#[cfg(test)]
mod parallel_corpus_tests;

#[cfg(test)]
mod pixel_compose_tests;
#[cfg(test)]
mod pixel_model;
#[cfg(test)]
mod pixel_tests;

pub use container::{DecodeLimits, EncodeOptions, MAGIC, VERSION, decode_jpeg, encode_jpeg};

/// Errors from the experimental codec and internal translated algorithms.
#[derive(Debug)]
pub enum Error {
    Invalid(&'static str),
    Unsupported(&'static str),
    Limit(&'static str),
    Internal(&'static str),
    Jpeg(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid jxl_plus input: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported jxl_plus operation: {s}"),
            Self::Limit(s) => write!(f, "jxl_plus resource limit: {s}"),
            Self::Internal(s) => write!(f, "jxl_plus internal error: {s}"),
            Self::Jpeg(s) => write!(f, "JPEG: {s}"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
