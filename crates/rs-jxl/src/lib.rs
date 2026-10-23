//! Direct, staged port of the optimized libjxl lossless encoder.
//! Pixel-lossless encoding is built from the directly ported Modular primitives.
//!
//! ```
//! use rs_jxl::{encode, encode_to, EncodeOptions, Image, PixelData, PixelLayout};
//!
//! // Interleaved sRGB RGBA, including nonzero RGB under zero alpha.
//! let image = Image::new(2, 1, PixelLayout::Rgba,
//!     PixelData::U8(vec![31, 97, 221, 0, 0, 255, 0, 255]))?;
//! let options = EncodeOptions::default();
//! let jxl_bytes = encode(&image, &options)?;
//! assert!(!jxl_bytes.is_empty());
//! let mut sink = Vec::new();
//! encode_to(&image, &options, &mut sink)?;
//! # Ok::<(), rs_jxl::Error>(())
//! ```
//!
//! Pixel encoding includes explicit Modular tools, animation and zero-duration
//! layers. Full source scheduler/tool parity is tracked in the port status;
//! a valid stream alone does not establish migration completion. Byte-reversible
//! JPEG coefficient transcoding has its own [`jpeg::transcode`] entry point;
//! it never falls back to decoding and recompressing pixels.
//! Runtime decoding uses maintained official Rust decoder sources; see [`decode`].
#![deny(unsafe_op_in_unsafe_fn)]

pub mod bit_writer;
pub mod color;
pub mod decode;
pub mod entropy;
pub mod frame;
pub mod icc;
pub mod jpeg;
pub mod metadata;
pub mod modular;
pub mod multiframe;
#[doc(hidden)]
pub mod profile;
pub mod region_input;
pub mod scheduler;
pub mod tools;

mod codestream;
pub mod container;
mod encode;
mod error;
mod execution;
pub mod fast_lossless;
/// Experimental source transcript utilities. Codec capture/replay hooks are
/// not yet connected; ordinary encoding does not consult these records.
pub mod final_plan;
mod image;
mod patch_frame;
pub mod streaming;
pub use encode::{EncodeOptions, encode, encode_frames, encode_frames_to, encode_to};
pub use error::{Error, Result};
pub use image::{Image, PixelData, PixelLayout};
pub use modular::encode::ModularOptions;
