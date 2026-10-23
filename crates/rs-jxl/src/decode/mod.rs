//! Pure Rust decoding, backed by maintained official sources in `vendor/jxl-rs`.
//! Visible frames are coalesced full canvases. Invisible zero-duration layers are
//! applied internally, not returned as independent frames.
//!
//! `Native` preserves integer ranges through 16 bits and widens floats to F32.
//! Wider integers become normalized F32, like djxl's float output, NOT exact U31.
//! Limits bound input, samples and retained outputs, not total decoder RSS.

mod boxes;
mod buffers;
mod jpeg;
mod runner;
pub use jpeg::reconstruct_jpeg;

use crate::{Error, PixelData, PixelLayout, Result};
use jxl_decoder::api::{Event, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, ProfileLevel};
pub use jxl_decoder::api::{
    ExtraChannel, JxlAnimation, JxlBasicInfo, JxlBitDepth, JxlColorProfile, JxlExtraChannel,
    Orientation,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SampleFormat {
    #[default]
    Native,
    /// Rescale/clamp to the full 8-bit range.
    U8,
    /// Rescale/clamp to the full 16-bit range.
    U16,
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
    /// Legacy djxl compatibility: allow Main10 complexity without a jxll box.
    /// Off by default; enabling this does not disable the configured budgets.
    pub allow_implicit_main10: bool,
    /// Apply encoded orientation, as djxl does by default.
    pub adjust_orientation: bool,
    /// Render spot colors into the main color output.
    pub render_spot_colors: bool,
    /// Return every extra channel, including alpha, as a separate plane.
    /// Planes are rendered at canvas resolution (dim_shift is still reported).
    pub include_extra_channels: bool,
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
    /// In basic_info.extra_channels order. Empty when explicitly disabled.
    /// For alpha of a different source depth this is the authoritative output:
    /// interleaved alpha uses the main image's output depth instead.
    pub extra_planes: Vec<PixelData>,
    pub duration_ticks: u32,
    pub name: String,
}
#[derive(Debug, Clone)]
pub struct DecodedBox {
    pub kind: [u8; 4],
    /// Decompressed contents. For Exif this includes the TIFF offset prefix.
    pub data: Vec<u8>,
}
pub struct DecodedImage {
    pub info: JxlBasicInfo,
    pub frames: Vec<DecodedFrame>,
    pub embedded_profile: Option<JxlColorProfile>,
    pub output_profile: Option<JxlColorProfile>,
    pub boxes: Vec<DecodedBox>,
    /// False when decode_frame stopped early; trailing boxes were not scanned.
    pub complete: bool,
}

fn failure(error: impl std::fmt::Display) -> Error {
    Error::Decoding(error.to_string())
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
    let mut backend = JxlDecoderOptions::default();
    backend.adjust_orientation = options.adjust_orientation;
    backend.render_spot_colors = options.render_spot_colors;
    backend.sample_limit = Some(options.limits.max_samples);
    backend.max_profile_level = ProfileLevel::Main10;
    backend.allow_implicit_main10 = options.allow_implicit_main10;
    // libjxl/common.h defaults JXL_HIGH_PRECISION to 1; upstream Rust does not.
    backend.high_precision = true;
    // Alpha stays AsIs: never discard invisible RGB.
    Ok(backend)
}

/// Read image/color headers only, without decoding a pixel frame. Does not
/// validate the remaining codestream or prove that the file is complete.
pub fn read_basic_info(mut input: &[u8], options: &DecodeOptions) -> Result<JxlBasicInfo> {
    let mut decoder = JxlDecoder::new(backend_options(input, options)?);
    decoder.close_input();
    match decoder.process(&mut input, None, None).map_err(failure)? {
        Event::BasicInfo => {
            let info = decoder
                .basic_info()
                .ok_or_else(|| failure("missing basic info"))?;
            let pixels = info
                .size
                .0
                .checked_mul(info.size.1)
                .ok_or(Error::Limit("pixel count overflow"))?;
            if pixels > options.limits.max_pixels {
                return Err(Error::Limit("decoded pixel count"));
            }
            Ok(info.clone())
        }
        _ => Err(failure("incomplete JPEG XL image header")),
    }
}
/// Decode all visible frames from a complete input buffer.
pub fn decode(input: &[u8], options: &DecodeOptions) -> Result<DecodedImage> {
    decode_impl(input, None, options)
}
/// Decode one zero-based visible frame, reconstructing dependencies as needed.
/// Stops after that frame; later frames/trailing metadata are not validated.
/// This is sequential dependency-aware decoding, not random access.
pub fn decode_frame(input: &[u8], index: usize, options: &DecodeOptions) -> Result<DecodedImage> {
    if index >= options.limits.max_frames {
        return Err(Error::Limit("requested frame index"));
    }
    decode_impl(input, Some(index), options)
}

fn decode_impl(
    mut input: &[u8],
    target: Option<usize>,
    options: &DecodeOptions,
) -> Result<DecodedImage> {
    let mut backend = backend_options(input, options)?;
    backend.request_aux_boxes = boxes::REQUESTED.to_vec();
    let mut decoder = JxlDecoder::new(backend);
    decoder.close_input();
    let mut runner = runner::Runner::new(options.threads)?;
    let mut buffers = None;
    let mut frames = Vec::new();
    let mut index = 0;
    let mut retained = 0usize;
    let mut complete = false;
    loop {
        let mut views: Vec<_> = buffers
            .as_mut()
            .map(|b: &mut buffers::Buffers| {
                b.images
                    .iter_mut()
                    .map(|image| JxlOutputBuffer::from_image_rect_mut(image.as_rect_mut()))
                    .collect()
            })
            .unwrap_or_default();
        let wanted = target.is_none_or(|n| n == index);
        let output = if views.is_empty() || !wanted {
            None
        } else {
            Some(views.as_mut_slice())
        };
        let event = decoder
            .process(&mut input, output, Some(&mut runner))
            .map_err(failure)?;
        match event {
            Event::BasicInfo => {
                let info = decoder
                    .basic_info()
                    .ok_or_else(|| failure("missing basic info"))?;
                let color = decoder
                    .current_pixel_format()
                    .ok_or_else(|| failure("missing pixel format"))?
                    .color_type;
                let (allocated, format) = buffers::Buffers::new(info, color, options)?;
                decoder.set_pixel_format(format).map_err(failure)?;
                buffers = Some(allocated);
            }
            Event::FrameHeader => {
                if index >= options.limits.max_frames {
                    return Err(Error::Limit("visible frame count"));
                }
                if wanted {
                    let bytes = buffers
                        .as_ref()
                        .ok_or_else(|| failure("missing output buffers"))?
                        .byte_count;
                    retained = retained
                        .checked_add(bytes)
                        .ok_or(Error::Limit("output size overflow"))?;
                    if retained > options.limits.max_output_bytes {
                        return Err(Error::Limit("retained decoded outputs"));
                    }
                }
            }
            Event::FrameComplete { .. } => {
                if wanted {
                    frames
                        .try_reserve(1)
                        .map_err(|_| Error::Limit("decoded frame list"))?;
                    let header = decoder
                        .scanned_frames()
                        .last()
                        .ok_or_else(|| failure("missing visible frame header"))?;
                    frames.push(
                        buffers
                            .as_ref()
                            .ok_or_else(|| failure("missing output buffers"))?
                            .copy_frame(index, header)?,
                    );
                }
                if target == Some(index) {
                    break;
                }
                index += 1;
            }
            Event::Complete => {
                complete = true;
                break;
            }
            Event::NeedMoreInput { .. } => return Err(failure("truncated JPEG XL input")),
        }
    }
    if frames.is_empty() {
        return Err(failure(if target.is_some() {
            "requested visible frame does not exist"
        } else {
            "JXL contains no visible frames"
        }));
    }
    let info = decoder
        .basic_info()
        .ok_or_else(|| failure("missing basic info"))?
        .clone();
    let metadata = if complete {
        boxes::extract(&decoder, input, options.limits.max_metadata_bytes)?
    } else {
        Vec::new()
    };
    Ok(DecodedImage {
        info,
        frames,
        embedded_profile: decoder.embedded_color_profile().cloned(),
        output_profile: decoder.output_color_profile().cloned(),
        boxes: metadata,
        complete,
    })
}
