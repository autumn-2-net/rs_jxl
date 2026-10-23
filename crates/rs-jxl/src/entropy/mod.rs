//! Entropy-coding primitives directly ported from libjxl.
//!
//! Prefix/rANS streams, HybridUint and context mapping are independent tools.

pub mod ans;
mod ans_histogram;
pub mod cluster;
pub mod config_search;
pub mod context_map;
pub mod hybrid_uint;
mod jobs;
pub mod options;
pub mod prefix_code;
mod simd;
pub mod stream;

#[cfg(test)]
mod worker_test;

pub use hybrid_uint::{HybridUintConfig, HybridUintConfigError, HybridUintEncoding};
