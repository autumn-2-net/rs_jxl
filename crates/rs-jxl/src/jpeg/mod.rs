//! Dedicated coefficient/reconstruction input path, ported from libjxl.
//! Marker archives, block primitives and bounded scan traversal are independent.
//! No JPEG-to-RGB decoder is used here. `transcode` and `transcode_to` produce
//! byte-reversible JPEG containers. Full optimized source-tool parity remains
//! tracked separately from this independently verified coefficient path.
pub mod ac_tokens;
mod archival;
pub mod bitstream;
pub mod cfl;
pub mod coefficient_order;
pub mod coefficient_reader;
pub mod coefficient_writer;
pub mod coefficients;
pub mod color_metadata;
pub mod compatibility;
pub mod container;
pub mod data;
pub mod dc_streams;
#[cfg(test)]
mod default_budget_tests;
pub mod entropy;
pub mod entropy_writer;
pub mod frame_header;
pub mod huffman;
pub mod modular_policy;
pub mod quantization;
pub mod reader;
pub mod reconstruction_brotli;
pub mod reconstruction_decode;
pub mod reconstruction_metadata;
mod reconstruction_validation;
mod refinement;
pub mod scan_progression;
pub mod scan_writer;
pub mod search;
mod sequential;
pub mod sof;
mod transcode;
pub mod transcode_layout;
#[cfg(test)]
mod transcode_reference_tests;

pub use archival::{TranscodeLimits, TranscodeOptions, transcode, transcode_to};
pub use search::transcode_search;
