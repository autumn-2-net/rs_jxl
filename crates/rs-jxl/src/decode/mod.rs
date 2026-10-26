//! Pure Rust decoding, backed by maintained official sources in `vendor/jxl-rs`.
//!
//! By default output is full-canvas, coalesced display frames. Raw layers and
//! previews are explicit options. Native wide integers use an exact Modular
//! capture path, not a floating-point roundtrip. Unsupported render operations
//! require explicitly selecting floating output.
//! Floating-to-U8 output uses upstream jxl-rs dithering by default; select
//! [`DitherMode::Libjxl`] for the libjxl output pattern. This does not change
//! floating output, native integer output, JPEG bytes, or Modular smoothing.
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
mod native;
mod runner;
mod stream;
pub use color::{ColorOptions, ColorProfile, parse_background, parse_color_description};
pub use engine::{decode, decode_frame};
pub use inspect::{ImageInspection, inspect, read_basic_info};
pub use jpeg::reconstruct_jpeg;
pub use stream::{
    StreamDecoder, StreamSummary, decode_reader, decode_reader_with, decode_reader_with_chunk_size,
};

use crate::{Error, PixelData, PixelLayout, Result};
pub use jxl_decoder::api::{
    AlphaMode, DitherMode, ExtraChannel, JxlAnimation, JxlBasicInfo, JxlBitDepth, JxlBlendInfo, JxlBlendMode,
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
    /// Backend floating-to-U8 output dithering; default preserves upstream jxl-rs.
    /// Floating output, native integer output and JPEG reconstruction ignore it.
    /// Explicit color postprocessing retains its existing nondithered quantizer.
    pub dither: DitherMode,
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
            dither: DitherMode::JxlRs,
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

/// Reusable decoder resources for independent complete inputs.
///
/// Owns one thread pool (none for `threads == 1`) and immutable options. Every
/// call creates fresh parser, frame and reference state; neither decoded data
/// nor input buffers are retained, including after an error. Limits apply per
/// input, not cumulatively across calls. Calls require exclusive access.
pub struct Decoder {
    options: DecodeOptions,
    runner: runner::Runner,
}

impl Decoder {
    /// Validate options and create the caller-owned worker pool once.
    /// Input-dependent limits are checked by each decoding operation.
    pub fn new(options: DecodeOptions) -> Result<Self> {
        validate_options(&options)?;
        let runner = runner::Runner::new(options.threads)?;
        Ok(Self { options, runner })
    }

    pub fn options(&self) -> &DecodeOptions {
        &self.options
    }

    /// Decode every display frame, or every regular layer without coalescing.
    pub fn decode(&mut self, input: &[u8]) -> Result<DecodedImage> {
        self.decode_impl(input, None)
    }

    /// Decode a zero-based display frame (or raw layer without coalescing),
    /// reconstructing references as needed.
    /// Later data is not checked, and the returned image is marked incomplete.
    pub fn decode_frame(&mut self, input: &[u8], index: usize) -> Result<DecodedImage> {
        if index >= self.options.limits.max_frames {
            return Err(Error::Limit("requested frame index"));
        }
        self.decode_impl(input, Some(index))
    }

    fn decode_impl(&mut self, input: &[u8], target: Option<usize>) -> Result<DecodedImage> {
        let _total = crate::profile::scope("decode.pixels.total");
        check_input_limit(input, &self.options)?;
        engine::decode_with_runner(
            input,
            target,
            &self.options,
            backend_options_from_validated(&self.options),
            &mut self.runner,
        )
    }

    /// Reconstruct original JPEG bytes under the same requirements as
    /// [`reconstruct_jpeg`], reusing this context's pool for capture and writing.
    pub fn reconstruct_jpeg(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        let _total = crate::profile::scope("decode.jpeg.total");
        check_input_limit(input, &self.options)?;
        let backend =
            jpeg::prepare_backend(backend_options_from_validated(&self.options), &self.options)?;
        jpeg::reconstruct_with_runner(input, &self.options, backend, &mut self.runner)
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
    check_input_limit(input, options)?;
    validate_options(options)?;
    Ok(backend_options_from_validated(options))
}

fn check_input_limit(input: &[u8], options: &DecodeOptions) -> Result<()> {
    if input.len() > options.limits.max_input_bytes {
        return Err(Error::Limit("JXL input bytes"));
    }
    Ok(())
}

fn validate_options(options: &DecodeOptions) -> Result<()> {
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
    Ok(())
}

fn backend_options_from_validated(options: &DecodeOptions) -> JxlDecoderOptions {
    let mut backend = JxlDecoderOptions::default();
    backend.adjust_orientation = options.adjust_orientation;
    backend.render_spot_colors = options.render_spot_colors;
    backend.coalescing = options.coalescing;
    backend.skip_preview = !options.include_preview;
    backend.alpha_mode = options.alpha_mode;
    backend.dither = options.dither;
    backend.max_downsampling = usize::from(options.downsampling);
    backend.sample_limit = Some(options.limits.max_samples);
    backend.max_profile_level = ProfileLevel::Main10;
    backend.allow_implicit_main10 = options.allow_implicit_main10;
    // libjxl/common.h defaults JXL_HIGH_PRECISION to 1; upstream Rust does not.
    backend.high_precision = true;
    backend
}

#[cfg(test)]
mod dither_tests {
    use super::*;

    #[test]
    fn dither_defaults_and_backend_options() {
        assert_eq!(DecodeOptions::default().dither, DitherMode::JxlRs);
        assert_eq!(JxlDecoderOptions::default().dither, DitherMode::JxlRs);
        for dither in [DitherMode::JxlRs, DitherMode::Libjxl] {
            let options = DecodeOptions {
                dither,
                ..DecodeOptions::default()
            };
            assert_eq!(backend_options_from_validated(&options).dither, dither);
        }
    }
}
