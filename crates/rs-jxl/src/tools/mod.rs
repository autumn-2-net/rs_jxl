//! Standalone lossless encoder kernels migrated from the frozen libjxl fork.
//!
//! Standalone direct ports; the complete encoder is still being migrated.
//! See `docs/PORTING_KERNELS.md` for provenance and integration requirements.
//! Native Rust SIMD is isolated in prefix_simd and ma_simd, with scalar fallback.
//! No end-to-end codec performance or compression-ratio improvement is claimed.

pub mod lz77;
pub mod lz77_prefix;
pub mod ma_entropy;
pub mod ma_simd;
pub mod patches;
mod prefix_simd;
