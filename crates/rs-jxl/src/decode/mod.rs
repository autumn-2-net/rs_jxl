//! Pure Rust decoding, backed by maintained official sources in `vendor/jxl-rs`.
//!
//! By default output is full-canvas, coalesced display frames. Raw layers and
//! previews are explicit options. Native samples retain integer ranges through
//! 16 bits; wider integers become normalized F32, not exact native U31.
//! Budgets bound input, samples and retained output, not total decoder RSS.

mod boxes;
mod buffers;
mod color;
mod color_lut;
mod convert;
mod engine;
#[cfg(test)]
mod error_tests;
mod inspect;
mod jpeg;
mod runner;
pub use color::{ColorOptions, ColorProfile, parse_background, parse_color_description};
pub use engine::{decode, decode_frame};
pub use inspect::{ImageInspection, inspect, read_basic_info};
pub use jpeg::reconstruct_jpeg;

use crate::{Error, PixelData, PixelLayout, Result};
pub use jxl_decoder::api::{
    AlphaMode, ExtraChannel, JxlAnimation, JxlBasicInfo, JxlBitDepth, JxlBlendInfo, JxlBlendMode,
    JxlColorEncoding, JxlColorProfile, JxlExtraChannel, JxlFrameHeader, JxlLayerInfo, JxlPrimaries,
    JxlTransferFunction, JxlWhitePoint, Orientation, RenderingIntent,
};
use jxl_decoder::api::{JxlDecoderOptions, ProfileLevel};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SampleFormat {
    #[default]
    Native,
    /// Rescale/clamp to the full 8-bit range.
    U8,
    /// Rescale/clamp to the full 16-bit range.
    U16,
    /// Rescale to 1..=8 significant bits, stored in U8 samples.
    U8Bits(u8),
    /// Rescale to 1..=16 significant bits, stored in U16 samples.
    U16Bits(u8),
    /// Integers become normalized floats; floating samples retain their range.
    F32,
}

#[derive(Debug, Clone)]
pub struct DecodeLimits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_pixels: usize,
    pub max_samples: usize,
    /// Returned frames/layers, excluding the optional preview.
    pub max_frames: usize,
}
impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 512 * 1024 * 1024,
            max_output_bytes: 512 * 1024 * 1024,
            max_metadata_bytes: 64 * 1024 * 1024,
            max_pixels: 64_000_000,
            max_samples: 256_000_000,
            max_frames: 10_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodeOptions {
    pub threads: usize,
    pub format: SampleFormat,
    /// Allow legacy Main10 complexity without a jxll box; budgets still apply.
    pub allow_implicit_main10: bool,
    pub adjust_orientation: bool,
    pub render_spot_colors: bool,
    /// Return all extra channels (including alpha) as separate full-size planes.
    pub include_extra_channels: bool,
    /// False returns unblended crop layers, including zero-duration layers.
    pub coalescing: bool,
    pub include_preview: bool,
    /// Opt into incomplete pixel output on EOF, never on malformed data.
    pub allow_partial: bool,
    /// Progressive quality selection (1, 2, 4, 8), not spatial resizing.
    pub downsampling: u8,
    pub alpha_mode: AlphaMode,
    pub color: ColorOptions,
    pub limits: DecodeLimits,
}
impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            format: SampleFormat::Native,
            allow_implicit_main10: false,
            adjust_orientation: true,
            render_spot_colors: true,
            include_extra_channels: true,
            coalescing: true,
            include_preview: false,
            allow_partial: false,
            downsampling: 1,
            alpha_mode: AlphaMode::AsIs,
            color: ColorOptions::default(),
            limits: DecodeLimits::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub index: usize,
    pub width: usize,
    pub height: usize,
    pub layout: PixelLayout,
    pub pixels: PixelData,
    /// In basic_info.extra_channels order. Separate alpha is authoritative when
    /// its depth differs from the main image. Empty only when explicitly disabled.
    pub extra_planes: Vec<PixelData>,
    pub duration_ticks: u32,
    pub name: String,
    pub header: JxlFrameHeader,
    /// False for explicitly requested partial/progressive output.
    pub complete: bool,
}
#[derive(Debug, Clone)]
pub struct DecodedBox {
    pub kind: [u8; 4],
    /// Decompressed contents. Exif includes the TIFF offset prefix.
    pub data: Vec<u8>,
}
pub struct DecodedImage {
    pub info: JxlBasicInfo,
    pub frames: Vec<DecodedFrame>,
    pub preview: Option<DecodedFrame>,
    pub embedded_profile: Option<JxlColorProfile>,
    pub output_profile: Option<JxlColorProfile>,
    /// Resolved display peak after color processing; source tone mapping stays in info.
    pub output_intensity_target: f32,
    pub boxes: Vec<DecodedBox>,
    /// False for early frame selection, partial or progressive decode.
    pub complete: bool,
    /// Input bytes consumed, including any parser read-ahead.
    pub bytes_read: usize,
}

fn failure(error: impl std::fmt::Display) -> Error {
    Error::Decoding(error.to_string())
}

fn backend_error(error: jxl_decoder::error::Error) -> Error {
    use jxl_decoder::error::Error as BackendError;
    match error {
        BackendError::OutOfMemory(_) | BackendError::ImageOutOfMemory(..) => {
            Error::Limit("decoder allocation")
        }
        BackendError::ImageSizeTooLarge(..) | BackendError::ImageDimensionTooLarge(_) => {
            Error::Limit("decoder image size")
        }
        BackendError::IOError(error) => Error::Io(error),
        other => failure(other),
    }
}

fn check_info(info: &JxlBasicInfo, options: &DecodeOptions) -> Result<()> {
    if !options.coalescing
        && options.render_spot_colors
        && info
            .extra_channels
            .iter()
            .any(|ec| ec.ec_type == ExtraChannel::SpotColor)
    {
        return Err(Error::InvalidInput(
            "raw layers with spot channels require render_spot_colors=false to preserve blend/reference semantics".into(),
        ));
    }
    let pixels = info
        .size
        .0
        .checked_mul(info.size.1)
        .ok_or(Error::Limit("pixel count overflow"))?;
    if pixels > options.limits.max_pixels {
        return Err(Error::Limit("decoded pixel count"));
    }
    Ok(())
}

fn backend_options(input: &[u8], options: &DecodeOptions) -> Result<JxlDecoderOptions> {
    if input.len() > options.limits.max_input_bytes {
        return Err(Error::Limit("JXL input bytes"));
    }
    if !(1..=256).contains(&options.threads) {
        return Err(Error::InvalidInput(
            "decoder threads must be in 1..=256".into(),
        ));
    }
    match options.format {
        SampleFormat::U8Bits(n) if !(1..=8).contains(&n) => {
            return Err(Error::InvalidInput("U8 bit depth must be in 1..=8".into()));
        }
        SampleFormat::U16Bits(n) if !(1..=16).contains(&n) => {
            return Err(Error::InvalidInput(
                "U16 bit depth must be in 1..=16".into(),
            ));
        }
        _ => {}
    }
    if ![1, 2, 4, 8].contains(&options.downsampling) {
        return Err(Error::InvalidInput(
            "downsampling must be 1, 2, 4 or 8".into(),
        ));
    }
    if !options.coalescing && (!options.color.is_noop() || options.alpha_mode != AlphaMode::AsIs) {
        return Err(Error::InvalidInput(
            "raw layers must keep source color and alpha association to preserve blend/reference semantics; coalesce before transforming".into(),
        ));
    }
    let mut backend = JxlDecoderOptions::default();
    backend.adjust_orientation = options.adjust_orientation;
    backend.render_spot_colors = options.render_spot_colors;
    backend.coalescing = options.coalescing;
    backend.skip_preview = !options.include_preview;
    backend.alpha_mode = options.alpha_mode;
    backend.max_downsampling = usize::from(options.downsampling);
    backend.sample_limit = Some(options.limits.max_samples);
    backend.max_profile_level = ProfileLevel::Main10;
    backend.allow_implicit_main10 = options.allow_implicit_main10;
    // libjxl/common.h defaults JXL_HIGH_PRECISION to 1; upstream Rust does not.
    backend.high_precision = true;
    Ok(backend)
}
