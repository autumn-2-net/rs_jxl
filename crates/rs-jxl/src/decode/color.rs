//! Opt-in output color processing. The parent decoder supplies F32 main pixels
//! (and F32 Black for CMYK), then quantizes to the requested output depth.
//!
//! Source provenance: libjxl at b866b3defec13494bfbaf705db93581cda118e37:
//! lib/extras/dec/color_description.cc, lib/extras/alpha_blend.cc,
//! lib/jxl/cms/{transfer_functions.h,transfer_functions-inl.h,tone_mapping.h},
//! lib/jxl/render_pipeline/stage_tone_mapping.cc. CMS routing also follows the
//! official jxl-rs jxl_cms/src/moxcms.rs, but uses the destination intent.
//!
//! Translated portions: Copyright (c) the JPEG XL Project Authors.
//! All rights reserved. Governed by the BSD-style license in
//! vendor/jxl-rs/LICENSE. This attribution does not license unrelated code.
//!
//! Default options are an unconditional, allocation-free, bit-exact no-op.
//! Active processing uses output_profile, NEVER embedded_profile, as the pixel
//! source. Embedded profile, source basic info, frame headers, and all extra
//! planes stay unchanged. Successful conversion changes output_profile.
//!
//! Interleaved alpha bypasses CMS and keeps its F32 bits. Associated color is
//! divided by positive alpha and reassociated after conversion. At alpha zero
//! the stored color is transformed as-is and is NOT multiplied by zero: hidden
//! samples are not erased, although they necessarily change color space.
//! Background compositing deliberately removes interleaved alpha, but retains
//! the original alpha extra plane. It blends in output-encoded coordinates,
//! matching djxl, not in linear light. Frame layout describes output alpha;
//! basic_info.extra_channels continues describing the source.
//!
//! The output and temporary sample buffers are checked against DecodeLimits.
//! CMS-internal allocations are not a strict RSS bound. All frames (including
//! preview) are staged before publication, so errors leave the image unchanged.
//! Structured encodings use unclamped F32 matrix/transfer processing. Opaque
//! ICC transforms use moxcms's conventional table domain: input outside [0, 1]
//! is rejected, not silently clipped. Their output follows the ICC tables and
//! does not promise extended-range gamut preservation.
//! CMYK mft1/mft2 -> structured RGB/gray defaults to the local skcms-based
//! djxl conventions; see ColorOptions::cmyk_icc_conventions. This direct route
//! preserves extended PCS/RGB output, but still rejects out-of-range device
//! input (after alpha unassociation and JXL CMYK inversion), unlike skcms's
//! input clamp. Other ICC forms continue through moxcms, not a skcms emulator.
//! Structured sRGB -> structured linear and default CMYK -> structured sRGB
//! use skcms's quantized ICC sRGB parameters and scalar power approximation.
//! This numerical compatibility never activates processing by itself.
//! F32 CMS results are
//! not promised bit-identical to a different CMS or libjxl's SIMD math.

use std::{str::FromStr, sync::Arc};

use jxl_decoder::api::{
    JxlColorEncoding, JxlPrimaries, JxlTransferFunction, JxlWhitePoint, RenderingIntent,
};
use moxcms::{
    ColorProfile as CmsProfile, DataColorSpace, Layout, Matrix3d, ParsingOptions,
    RenderingIntent as CmsIntent, ToneCurveEvaluator, TransformF32Executor, TransformOptions, Xyzd,
};

use super::{DecodeLimits, DecodedFrame, DecodedImage, ExtraChannel, JxlColorProfile};
use crate::{Error, PixelData, PixelLayout, Result};

/// Destination color coordinates. ICC bytes are explicit, never a file path.
#[derive(Debug, Clone, PartialEq)]
pub enum ColorProfile {
    Encoding(JxlColorEncoding),
    Icc(Vec<u8>),
}

/// Output processing only; not a second copy of DecodeOptions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColorOptions {
    pub output_profile: Option<ColorProfile>,
    /// CMYK mft1/mft2 -> structured RGB/gray policy; does not activate CMS.
    /// False (default): skcms-based djxl's A2B0-then-A2B1 selection (not A2B2),
    /// unit-normalized Lab/XYZ tables, no absolute media-white scaling, and
    /// gray from PCS X / header-white X, with skcms sRGB output transfer math.
    /// Destination intent is retained as
    /// metadata but does not alter this route's pixels.
    /// True: destination-intent selection with A2B0 fallback, legacy 16-bit
    /// Lab 0xff00 normalization, ICC XYZ scaling, absolute media-white ratios,
    /// and gray from PCS Y with analytic output transfer math. This is not a
    /// full ICC conformance mode and does not change non-CMYK processing.
    /// Neither mode clips extended output; both reject nonfinite or out-of-
    /// range CMYK input. Other ICC forms/destinations use the existing moxcms
    /// path and are not covered by this compatibility policy.
    pub cmyk_icc_conventions: bool,
    /// Positive finite display peak. PQ down-mapping follows Rec.2408; HLG
    /// adapts its OOTF when the destination is not HLG. SDR is unchanged.
    /// HDR adjustment requires structured source AND destination encodings;
    /// arbitrary ICC luminance semantics cannot be inferred and are rejected.
    pub display_nits: Option<f32>,
    /// Encoded destination RGB in [0, 1]. Gray uses element zero, as djxl does.
    pub background: Option<[f32; 3]>,
}

impl ColorOptions {
    /// The parent decoder must supply F32 main pixels for any active option.
    pub fn needs_float(&self) -> bool {
        !self.is_noop()
    }

    pub fn is_noop(&self) -> bool {
        self.output_profile.is_none() && self.display_nits.is_none() && self.background.is_none()
    }
}

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

fn cms_error(error: impl std::fmt::Display) -> Error {
    Error::Decoding(format!("color management: {error}"))
}

impl FromStr for ColorProfile {
    type Err = Error;

    fn from_str(description: &str) -> Result<Self> {
        parse_color_description(description).map(Self::Encoding)
    }
}

impl ColorProfile {
    /// Parse a djxl descriptor or alias. Whitespace/trailing tokens are errors.
    pub fn parse(description: &str) -> Result<Self> {
        description.parse()
    }

    fn as_jxl(&self) -> JxlColorProfile {
        match self {
            Self::Encoding(encoding) => JxlColorProfile::Simple(encoding.clone()),
            Self::Icc(bytes) => JxlColorProfile::Icc(bytes.clone()),
        }
    }
}

/// djxl descriptors plus the convenient linear (= linear sRGB) alias.
pub fn parse_color_description(description: &str) -> Result<JxlColorEncoding> {
    let description = match description {
        "sRGB" | "srgb" => "RGB_D65_SRG_Rel_SRG",
        "linear" | "linear-sRGB" => "RGB_D65_SRG_Rel_Lin",
        "DisplayP3" => "RGB_D65_DCI_Rel_SRG",
        "Adobe98" => "RGB_D65_Ado_Rel_Ado",
        "ProPhoto" => "RGB_D50_Pro_Rel_Pro",
        "Rec2100PQ" => "RGB_D65_202_Rel_PeQ",
        "Rec2100HLG" => "RGB_D65_202_Rel_HLG",
        value => value,
    };
    let mut tokens = description.split('_');
    let space = tokens.next().unwrap_or_default();
    let mut next = || {
        tokens
            .next()
            .ok_or_else(|| invalid("incomplete color descriptor"))
    };
    let white_point = parse_white(next()?)?;
    let primaries = match space {
        "RGB" => Some(parse_primaries(next()?)?),
        "Gra" => None,
        _ => {
            return Err(invalid(
                "output descriptors support RGB or Gra, not XYB/unknown",
            ));
        }
    };
    let rendering_intent = match next()? {
        "Per" => RenderingIntent::Perceptual,
        "Rel" => RenderingIntent::Relative,
        "Sat" => RenderingIntent::Saturation,
        "Abs" => RenderingIntent::Absolute,
        _ => return Err(invalid("invalid color rendering intent")),
    };
    let transfer_function = parse_transfer(next()?)?;
    if tokens.next().is_some() {
        return Err(invalid("trailing color descriptor tokens"));
    }
    Ok(match primaries {
        Some(primaries) => JxlColorEncoding::RgbColorSpace {
            white_point,
            primaries,
            transfer_function,
            rendering_intent,
        },
        None => JxlColorEncoding::GrayscaleColorSpace {
            white_point,
            transfer_function,
            rendering_intent,
        },
    })
}

fn numbers<const N: usize>(value: &str) -> Result<[f32; N]> {
    let mut tokens = value.split(';');
    let mut result = [0.0; N];
    for number in &mut result {
        *number = tokens
            .next()
            .ok_or_else(|| invalid("missing chromaticity"))?
            .parse::<f32>()
            .map_err(|_| invalid("invalid chromaticity"))?;
        if !number.is_finite() {
            return Err(invalid("chromaticities must be finite"));
        }
    }
    if tokens.next().is_some() {
        return Err(invalid("too many chromaticities"));
    }
    Ok(result)
}

fn parse_white(value: &str) -> Result<JxlWhitePoint> {
    Ok(match value {
        "D65" => JxlWhitePoint::D65,
        "EER" => JxlWhitePoint::E,
        "DCI" => JxlWhitePoint::DCI,
        "D50" => JxlWhitePoint::Chromaticity {
            wx: 0.345669,
            wy: 0.358496,
        },
        value => {
            let [wx, wy] = numbers(value)?;
            if !(0.0..=1.0).contains(&wx) || !(0.0 < wy && wy <= 1.0) {
                return Err(invalid("white point outside xy coordinate range"));
            }
            JxlWhitePoint::Chromaticity { wx, wy }
        }
    })
}

fn parse_primaries(value: &str) -> Result<JxlPrimaries> {
    Ok(match value {
        "SRG" => JxlPrimaries::SRGB,
        "202" => JxlPrimaries::BT2100,
        "DCI" => JxlPrimaries::P3,
        value => {
            let [rx, ry, gx, gy, bx, by] = match value {
                "Ado" => [0.64, 0.33, 0.21, 0.71, 0.15, 0.06],
                "Pro" => [0.734699, 0.265301, 0.159597, 0.840403, 0.036598, 0.000105],
                value => numbers(value)?,
            };
            // Negative primary coordinates can be legitimate (e.g. ACES).
            JxlPrimaries::Chromaticities {
                rx,
                ry,
                gx,
                gy,
                bx,
                by,
            }
        }
    })
}

fn parse_transfer(value: &str) -> Result<JxlTransferFunction> {
    Ok(match value {
        "709" => JxlTransferFunction::BT709,
        "Lin" => JxlTransferFunction::Linear,
        "SRG" => JxlTransferFunction::SRGB,
        "PeQ" => JxlTransferFunction::PQ,
        "DCI" => JxlTransferFunction::DCI,
        "HLG" => JxlTransferFunction::HLG,
        "Ado" => JxlTransferFunction::Gamma(256.0 / 563.0),
        "Pro" => JxlTransferFunction::Gamma(1.0 / 1.8),
        value => {
            let gamma: f32 = value
                .strip_prefix('g')
                .ok_or_else(|| invalid("invalid transfer function"))?
                .parse()
                .map_err(|_| invalid("invalid gamma"))?;
            if !gamma.is_finite() || !(0.0 < gamma && gamma <= 1.0) {
                return Err(invalid("JPEG XL gamma must be in (0, 1]"));
            }
            JxlTransferFunction::Gamma(gamma)
        }
    })
}

/// Strict djxl --background parser: black, white, or #RRGGBB.
pub fn parse_background(value: &str) -> Result<[f32; 3]> {
    match value {
        "black" => return Ok([0.0; 3]),
        "white" => return Ok([1.0; 3]),
        _ => {}
    }
    let hex = value.as_bytes();
    if hex.len() != 7 || hex[0] != b'#' || !hex[1..].iter().all(u8::is_ascii_hexdigit) {
        return Err(invalid("background must be black, white, or #RRGGBB"));
    }
    let rgb = u32::from_str_radix(&value[1..], 16).map_err(|_| invalid("invalid background"))?;
    Ok([
        ((rgb >> 16) & 255) as f32 / 255.0,
        ((rgb >> 8) & 255) as f32 / 255.0,
        (rgb & 255) as f32 / 255.0,
    ])
}

struct Endpoint {
    cms: CmsProfile,
    layout: Layout,
    curve: Option<Curve>,
}

#[derive(Clone)]
struct Curve {
    transfer: JxlTransferFunction,
    luminances: [f32; 3],
    native_white: [f64; 3],
}

fn intent(value: RenderingIntent) -> CmsIntent {
    match value {
        RenderingIntent::Perceptual => CmsIntent::Perceptual,
        RenderingIntent::Relative => CmsIntent::RelativeColorimetric,
        RenderingIntent::Saturation => CmsIntent::Saturation,
        RenderingIntent::Absolute => CmsIntent::AbsoluteColorimetric,
    }
}

fn encoding_parts(encoding: &JxlColorEncoding) -> Result<(&JxlTransferFunction, RenderingIntent)> {
    match encoding {
        JxlColorEncoding::RgbColorSpace {
            transfer_function,
            rendering_intent,
            ..
        }
        | JxlColorEncoding::GrayscaleColorSpace {
            transfer_function,
            rendering_intent,
            ..
        } => Ok((transfer_function, *rendering_intent)),
        JxlColorEncoding::XYB { .. } => Err(invalid("XYB is not an output pixel color space")),
    }
}

fn linear_encoding(encoding: &JxlColorEncoding) -> Result<JxlColorEncoding> {
    let mut result = encoding.clone();
    match &mut result {
        JxlColorEncoding::RgbColorSpace {
            transfer_function, ..
        }
        | JxlColorEncoding::GrayscaleColorSpace {
            transfer_function, ..
        } => *transfer_function = JxlTransferFunction::Linear,
        JxlColorEncoding::XYB { .. } => return Err(invalid("XYB output is unsupported")),
    }
    Ok(result)
}

fn parse_icc(bytes: &[u8], limits: &DecodeLimits) -> Result<CmsProfile> {
    if bytes.len() > limits.max_metadata_bytes {
        return Err(Error::Limit("color profile bytes"));
    }
    CmsProfile::new_from_slice_with_options(
        bytes,
        ParsingOptions {
            max_profile_size: limits.max_metadata_bytes,
            max_allowed_clut_size: limits.max_metadata_bytes.min(10_000_000),
            max_allowed_trc_size: (limits.max_metadata_bytes / 4).min(40_000),
        },
    )
    .map_err(cms_error)
}

impl Endpoint {
    fn new(profile: &JxlColorProfile, limits: &DecodeLimits, output: bool) -> Result<Self> {
        let (mut cms, curve) = match profile {
            JxlColorProfile::Icc(bytes) => (parse_icc(bytes, limits)?, None),
            JxlColorProfile::Simple(encoding) => {
                let (transfer, rendering_intent) = encoding_parts(encoding)?;
                if let JxlTransferFunction::Gamma(gamma) = transfer
                    && (!gamma.is_finite() || !(0.0 < *gamma && *gamma <= 1.0))
                {
                    return Err(invalid("JPEG XL gamma must be in (0, 1]"));
                }
                let curve = Curve {
                    transfer: transfer.clone(),
                    luminances: luminances(encoding)?,
                    native_white: native_white(encoding)?,
                };
                // HDR ICC compatibility LUTs are display adaptations, not
                // reversible PQ/HLG transfer functions. Use a linear ICC and
                // perform structured transfer functions explicitly instead.
                let linear = JxlColorProfile::Simple(linear_encoding(encoding)?);
                let bytes = linear
                    .try_as_icc()
                    .ok_or_else(|| invalid("cannot create linear ICC"))?;
                let mut cms = parse_icc(&bytes, limits)?;
                cms.rendering_intent = intent(rendering_intent);
                (cms, Some(curve))
            }
        };
        // Structured profiles already have a linear TRC; avoid any CICP
        // override. Explicit ICC profiles retain their original tag semantics.
        if curve.is_some() {
            cms.cicp = None;
        }
        let layout = match cms.color_space {
            DataColorSpace::Rgb => Layout::Rgb,
            DataColorSpace::Gray => Layout::Gray,
            DataColorSpace::Cmyk if !output => Layout::Rgba,
            _ => {
                return Err(invalid(if output {
                    "output ICC must describe RGB or grayscale (CMYK output is unsupported)"
                } else {
                    "source ICC must describe RGB, grayscale, or CMYK"
                }));
            }
        };
        Ok(Self { cms, layout, curve })
    }
}

/// Transform main pixels and preview transactionally. Call before final
/// quantization, with backend alpha left AsIs. Active calls require F32 main
/// storage. CMYK additionally requires the Black plane in F32; other extras may
/// use any storage and are retained verbatim. No automatic CMYK conversion is
/// performed without an explicit destination. The parent owns output bit-depth
/// metadata; info.bit_depth remains the encoded/source depth. Returns the
/// resolved output intensity target (nits), for storage alongside the output
/// profile. An ignored SDR display request does not change that value.
pub fn process_image(
    image: &mut DecodedImage,
    options: &ColorOptions,
    limits: &DecodeLimits,
) -> Result<f32> {
    if options.is_noop() {
        return Ok(image.info.tone_mapping.intensity_target);
    }
    validate_options(options)?;
    if image.frames.len() > limits.max_frames {
        return Err(Error::Limit("color processing frame count"));
    }
    if let Some(ColorProfile::Icc(bytes)) = &options.output_profile
        && bytes.len() > limits.max_metadata_bytes
    {
        return Err(Error::Limit("output color profile bytes"));
    }
    let source = image
        .output_profile
        .as_ref()
        .ok_or_else(|| invalid("color processing requires image.output_profile"))?;
    let target = options.output_profile.as_ref().map(ColorProfile::as_jxl);
    let destination = target.as_ref().unwrap_or(source);
    let pipeline = Pipeline::new(
        source,
        destination,
        image.info.tone_mapping.intensity_target,
        if options.display_nits.is_some() {
            hdr_origin(image.embedded_profile.as_ref(), limits)?
        } else {
            None
        },
        options,
        limits,
    )?;
    let alpha = image
        .info
        .extra_channels
        .iter()
        .find(|ec| ec.ec_type == ExtraChannel::Alpha);
    let associated = alpha.is_some_and(|ec| ec.alpha_associated);
    let black = if pipeline.input.layout == Layout::Rgba {
        let mut indices = image
            .info
            .extra_channels
            .iter()
            .enumerate()
            .filter(|(_, ec)| ec.ec_type == ExtraChannel::Black);
        let index = indices
            .next()
            .map(|(i, _)| i)
            .ok_or_else(|| invalid("CMYK source is missing Black extra-channel metadata"))?;
        if indices.next().is_some() {
            return Err(invalid("CMYK source has ambiguous Black planes"));
        }
        Some(index)
    } else {
        None
    };

    // Account for original outputs plus staged replacements and the bounded
    // transform scratch. None of the original buffers is freed until commit.
    let mut bytes = 0usize;
    let mut scratch_pixels = 0usize;
    for frame in image.preview.iter().chain(&image.frames) {
        let count = checked_product(frame.width, frame.height)?;
        scratch_pixels = scratch_pixels.max(count.min(CHUNK_PIXELS));
        if count > limits.max_pixels {
            return Err(Error::Limit("color processing pixels"));
        }
        let main_len = checked_product(count, frame.layout.channels())?;
        if frame.pixels.len() != main_len {
            return Err(invalid("main sample count does not match frame dimensions"));
        }
        f32_pixels(&frame.pixels)?;
        let input_colors = frame.layout.channels() - usize::from(frame.layout.has_alpha());
        let required_colors = if black.is_some() {
            3
        } else {
            pipeline.input.layout.channels()
        };
        if input_colors != required_colors {
            return Err(invalid(
                "source profile channel count disagrees with frame layout",
            ));
        }
        if frame.layout.has_alpha() && alpha.is_none() {
            return Err(invalid("interleaved alpha has no source metadata"));
        }
        if !frame.extra_planes.is_empty()
            && frame.extra_planes.len() != image.info.extra_channels.len()
        {
            return Err(invalid("extra planes must be in complete metadata order"));
        }
        let mut samples = main_len;
        for plane in &frame.extra_planes {
            if plane.len() != count {
                return Err(invalid("extra plane is not at frame resolution"));
            }
            samples = checked_sum(samples, plane.len())?;
            bytes = checked_sum(
                bytes,
                checked_product(plane.len(), plane.bits() as usize / 8)?,
            )?;
        }
        if let Some(index) = black {
            f32_pixels(
                frame
                    .extra_planes
                    .get(index)
                    .ok_or_else(|| invalid("CMYK conversion requires the Black extra plane"))?,
            )?;
        }
        let output_len = checked_product(count, pipeline.output_layout(frame.layout).channels())?;
        let output_samples = checked_sum(samples - main_len, output_len)?;
        if samples > limits.max_samples || output_samples > limits.max_samples {
            return Err(Error::Limit("color processing samples"));
        }
        bytes = checked_sum(bytes, checked_product(main_len, 4)?)?;
        bytes = checked_sum(bytes, checked_product(output_len, 4)?)?;
    }
    bytes = checked_sum(
        bytes,
        checked_product(
            scratch_pixels,
            checked_product(
                pipeline.input.layout.channels()
                    + pipeline.output.layout.channels()
                    + if pipeline.absolute_output.is_some() {
                        3
                    } else {
                        0
                    },
                4,
            )?,
        )?,
    )?;
    if bytes > limits.max_output_bytes {
        return Err(Error::Limit("color processing outputs and scratch"));
    }
    let mut staged = Vec::new();
    staged
        .try_reserve_exact(image.frames.len() + usize::from(image.preview.is_some()))
        .map_err(|_| Error::Limit("color frame list"))?;
    for frame in image.preview.iter().chain(&image.frames) {
        staged.push(pipeline.process_frame(frame, associated, black)?);
    }
    for (frame, (layout, pixels)) in image
        .preview
        .iter_mut()
        .chain(&mut image.frames)
        .zip(staged)
    {
        frame.layout = layout;
        frame.pixels = PixelData::F32(pixels);
    }
    if let Some(target) = target {
        image.output_profile = Some(target);
    }
    Ok(pipeline.destination_nits)
}

fn hdr_origin(
    profile: Option<&JxlColorProfile>,
    limits: &DecodeLimits,
) -> Result<Option<JxlTransferFunction>> {
    Ok(match profile {
        Some(JxlColorProfile::Simple(JxlColorEncoding::XYB { .. })) => None,
        Some(JxlColorProfile::Simple(encoding)) => {
            let (transfer, _) = encoding_parts(encoding)?;
            matches!(transfer, JxlTransferFunction::PQ | JxlTransferFunction::HLG)
                .then(|| transfer.clone())
        }
        Some(JxlColorProfile::Icc(bytes)) => {
            parse_icc(bytes, limits)?
                .cicp
                .and_then(|cicp| match cicp.transfer_characteristics {
                    moxcms::TransferCharacteristics::Smpte2084 => Some(JxlTransferFunction::PQ),
                    moxcms::TransferCharacteristics::Hlg => Some(JxlTransferFunction::HLG),
                    _ => None,
                })
        }
        None => None,
    })
}

fn validate_options(options: &ColorOptions) -> Result<()> {
    if options
        .display_nits
        .is_some_and(|n| !n.is_finite() || n <= 0.0)
    {
        return Err(invalid("display nits must be finite and positive"));
    }
    if options
        .background
        .is_some_and(|rgb| rgb.iter().any(|v| !(0.0..=1.0).contains(v)))
    {
        return Err(invalid(
            "background components must be finite and in [0, 1]",
        ));
    }
    Ok(())
}

fn checked_product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(Error::Limit("color sample size overflow"))
}

fn checked_sum(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(Error::Limit("color buffer size overflow"))
}

fn floats(count: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("color sample allocation"))?;
    values.resize(count, 0.0);
    Ok(values)
}

fn f32_pixels(data: &PixelData) -> Result<&[f32]> {
    match data {
        PixelData::F32(values) => Ok(values),
        _ => Err(invalid(
            "active color processing requires backend F32 samples",
        )),
    }
}

const CHUNK_PIXELS: usize = 4096;

struct Pipeline {
    input: Endpoint,
    output: Endpoint,
    transform: Option<Arc<TransformF32Executor>>,
    absolute_output: Option<Arc<TransformF32Executor>>,
    absolute_scale: [f32; 3],
    cmyk_lut: Option<super::color_lut::CmykLut>,
    djxl_cmyk: bool,
    cmyk_gray_component: usize,
    cmyk_gray_scale: f32,
    matrix: Option<Matrix3d>,
    post_matrix: Option<Matrix3d>,
    gray_curve: Option<Box<dyn ToneCurveEvaluator + Send + Sync>>,
    srgb_input: Option<skcms::Transfer>,
    srgb_output: Option<skcms::Transfer>,
    source_nits: f32,
    destination_nits: f32,
    tone: Option<DisplayMap>,
    background: Option<[f32; 3]>,
    preserve_encoding: bool,
}

impl Pipeline {
    fn new(
        source: &JxlColorProfile,
        destination: &JxlColorProfile,
        source_nits: f32,
        hdr_origin: Option<JxlTransferFunction>,
        options: &ColorOptions,
        limits: &DecodeLimits,
    ) -> Result<Self> {
        if !source_nits.is_finite() || source_nits <= 0.0 {
            return Err(invalid(
                "source intensity target must be finite and positive",
            ));
        }
        let input = Endpoint::new(source, limits, false)?;
        let output = Endpoint::new(destination, limits, true)?;
        let tone = DisplayMap::new(
            &input,
            &output,
            hdr_origin.as_ref(),
            source_nits,
            options.display_nits,
        )?;
        let destination_nits = if tone.is_some() {
            options.display_nits.unwrap_or(source_nits)
        } else {
            source_nits
        };
        let same_linear = match (source, destination) {
            (JxlColorProfile::Simple(a), JxlColorProfile::Simple(b)) => {
                linear_encoding(a)? == linear_encoding(b)?
            }
            _ => source == destination,
        };
        let matrix = if !same_linear && input.curve.is_some() && output.curve.is_some() {
            Some(structured_matrix(&input, &output)?)
        } else {
            None
        };
        let gray_curve = if input.curve.is_none()
            && input.layout == Layout::Gray
            && input.cms.pcs == DataColorSpace::Xyz
            && output.cms.rendering_intent != CmsIntent::AbsoluteColorimetric
            && output.curve.is_some()
            && input.cms.lut_a_to_b_perceptual.is_none()
            && input.cms.lut_a_to_b_colorimetric.is_none()
            && input.cms.lut_a_to_b_saturation.is_none()
        {
            input
                .cms
                .gray_trc
                .as_ref()
                .map(|curve| curve.make_linear_evaluator())
                .transpose()
                .map_err(cms_error)?
        } else {
            None
        };
        let cmyk_lut = if input.layout == Layout::Rgba && output.curve.is_some() {
            super::color_lut::CmykLut::new(&input.cms, &output.cms, options.cmyk_icc_conventions)?
        } else {
            None
        };
        let djxl_cmyk = cmyk_lut.is_some() && !options.cmyk_icc_conventions;
        let absolute_scale = if djxl_cmyk {
            // skcms_Transform has no rendering-intent/media-white operation.
            [1.0; 3]
        } else {
            super::color_lut::absolute_scale(&input.cms, &output.cms)?
        };
        // A CMYK LUT can produce negative XYZ as well as out-of-gamut RGB.
        // Keep PCS conversion and the destination matrix outside sampled CMS.
        let mut cms_destination = output.cms.clone();
        let post_matrix = if input.layout == Layout::Rgba
            && output.curve.is_some()
            && output.layout == Layout::Rgb
        {
            cms_destination.red_colorant = Xyzd::new(1.0, 0.0, 0.0);
            cms_destination.green_colorant = Xyzd::new(0.0, 1.0, 0.0);
            cms_destination.blue_colorant = Xyzd::new(0.0, 0.0, 1.0);
            let mut matrix = checked_inverse(output.cms.colorant_matrix())?;
            for row in &mut matrix.v {
                for (entry, scale) in row.iter_mut().zip(absolute_scale) {
                    *entry *= scale;
                }
            }
            Some(matrix)
        } else {
            None
        };
        let cmyk_gray =
            input.layout == Layout::Rgba && output.layout == Layout::Gray && output.curve.is_some();
        let cmyk_gray_component = usize::from(!djxl_cmyk);
        let cmyk_gray_scale = if cmyk_gray && djxl_cmyk {
            // skcms parses gray as a diagonal header-illuminant matrix;
            // libjxl contracts its RGB result by retaining channel zero.
            let white_x = output.cms.white_point.x;
            if !white_x.is_finite() || white_x <= 0.0 {
                return Err(invalid("invalid ICC PCS white point"));
            }
            (1.0 / white_x) as f32
        } else if cmyk_gray {
            absolute_scale[1] as f32
        } else {
            1.0
        };
        let absolute_output = if !same_linear
            && matrix.is_none()
            && post_matrix.is_none()
            && !(cmyk_gray && cmyk_lut.is_some())
            && absolute_scale != [1.0; 3]
        {
            // moxcms selects A2B1/B2A1 but omits the absolute media-white scale.
            // Keep that operation between explicit device->PCS->device stages.
            let mut pcs =
                Endpoint::new(&ColorProfile::parse("linear")?.as_jxl(), limits, true)?.cms;
            pcs.red_colorant = Xyzd::new(1.0, 0.0, 0.0);
            pcs.green_colorant = Xyzd::new(0.0, 1.0, 0.0);
            pcs.blue_colorant = Xyzd::new(0.0, 0.0, 1.0);
            let transform = pcs
                .create_transform_f32(
                    Layout::Rgb,
                    &output.cms,
                    output.layout,
                    TransformOptions {
                        rendering_intent: output.cms.rendering_intent,
                        prefer_fixed_point: false,
                        allow_extended_range_rgb_xyz: true,
                        ..Default::default()
                    },
                )
                .map_err(cms_error)?;
            cms_destination = pcs;
            Some(transform)
        } else {
            None
        };
        let transform =
            if same_linear || matrix.is_some() || gray_curve.is_some() || cmyk_lut.is_some() {
                None
            } else {
                Some(
                    input
                        .cms
                        .create_transform_f32(
                            input.layout,
                            &cms_destination,
                            if absolute_output.is_some() {
                                Layout::Rgb
                            } else {
                                output.layout
                            },
                            TransformOptions {
                                rendering_intent: output.cms.rendering_intent,
                                prefer_fixed_point: false,
                                allow_extended_range_rgb_xyz: true,
                                ..Default::default()
                            },
                        )
                        .map_err(cms_error)?,
                )
            };
        let preserve_encoding = source == destination && tone.is_none();
        let srgb_input = if output
            .curve
            .as_ref()
            .is_some_and(|c| c.transfer == JxlTransferFunction::Linear)
        {
            input.curve.as_ref().and_then(|curve| match curve.transfer {
                JxlTransferFunction::SRGB => Some(skcms::Transfer::srgb()),
                JxlTransferFunction::Gamma(gamma) => Some(skcms::Transfer::gamma(gamma)),
                JxlTransferFunction::DCI => Some(skcms::Transfer::gamma(1.0 / 2.6)),
                _ => None,
            })
        } else {
            None
        };
        let srgb_output = (djxl_cmyk
            && output
                .curve
                .as_ref()
                .is_some_and(|c| c.transfer == JxlTransferFunction::SRGB))
        .then(|| skcms::Transfer::srgb().inverse());
        Ok(Self {
            input,
            output,
            transform,
            absolute_output,
            absolute_scale: absolute_scale.map(|v| v as f32),
            cmyk_lut,
            djxl_cmyk,
            cmyk_gray_component,
            cmyk_gray_scale,
            matrix,
            post_matrix,
            gray_curve,
            srgb_input,
            srgb_output,
            source_nits,
            destination_nits,
            tone,
            background: options.background,
            preserve_encoding,
        })
    }

    fn output_layout(&self, input: PixelLayout) -> PixelLayout {
        match (
            self.output.layout,
            input.has_alpha() && self.background.is_none(),
        ) {
            (Layout::Gray, false) => PixelLayout::Gray,
            (Layout::Gray, true) => PixelLayout::GrayAlpha,
            (_, false) => PixelLayout::Rgb,
            (_, true) => PixelLayout::Rgba,
        }
    }

    fn process_frame(
        &self,
        frame: &DecodedFrame,
        associated: bool,
        black: Option<usize>,
    ) -> Result<(PixelLayout, Vec<f32>)> {
        let source = f32_pixels(&frame.pixels)?;
        let black_plane = black
            .map(|i| f32_pixels(&frame.extra_planes[i]))
            .transpose()?;
        let layout = self.output_layout(frame.layout);
        let count = checked_product(frame.width, frame.height)?;
        let mut result = floats(checked_product(count, layout.channels())?)?;
        let in_channels = self.input.layout.channels();
        let out_channels = self.output.layout.channels();
        let mut input = floats(count.min(CHUNK_PIXELS) * in_channels)?;
        let mut output = floats(count.min(CHUNK_PIXELS) * out_channels)?;
        let mut pcs = floats(if self.absolute_output.is_some() {
            count.min(CHUNK_PIXELS) * 3
        } else {
            0
        })?;
        for start in (0..count).step_by(CHUNK_PIXELS) {
            let len = (count - start).min(CHUNK_PIXELS);
            for p in 0..len {
                let original =
                    &source[(start + p) * frame.layout.channels()..][..frame.layout.channels()];
                let alpha = frame_alpha(original, frame.layout);
                if frame.layout.has_alpha()
                    && (associated || self.background.is_some())
                    && !(0.0..=1.0).contains(&alpha)
                {
                    return Err(invalid(
                        "associated/composited alpha must be finite and in [0, 1]",
                    ));
                }
                let pixel = &mut input[p * in_channels..][..in_channels];
                gather_color(
                    original,
                    pixel,
                    black_plane.map(|plane| plane[start + p]),
                    associated,
                    alpha,
                );
                if let Some(curve) = self
                    .input
                    .curve
                    .as_ref()
                    .filter(|_| !self.preserve_encoding)
                {
                    if let Some(srgb) = &self.srgb_input {
                        for value in pixel.iter_mut() {
                            *value = srgb.apply(*value);
                        }
                    } else {
                        curve.decode(pixel, self.source_nits);
                    }
                }
                if pixel.iter().any(|v| !v.is_finite()) {
                    return Err(invalid(
                        "non-finite color input or transfer-function overflow",
                    ));
                }
            }
            let input = &input[..len * in_channels];
            let output = &mut output[..len * out_channels];
            if let Some(lut) = &self.cmyk_lut {
                if out_channels == 1 {
                    for (input, value) in input.chunks_exact(4).zip(output.iter_mut()) {
                        let mut xyz = [0.0; 3];
                        lut.transform(input, &mut xyz)?;
                        *value = xyz[self.cmyk_gray_component] * self.cmyk_gray_scale;
                    }
                } else {
                    lut.transform(input, output)?;
                }
            } else if let Some(transform) = &self.transform {
                if input.iter().any(|value| !(0.0..=1.0).contains(value)) {
                    return Err(invalid(
                        "extended-range ICC input is unsupported by this CMS configuration; use structured profiles",
                    ));
                }
                if let Some(final_transform) = &self.absolute_output {
                    let pcs = &mut pcs[..len * 3];
                    transform.transform(input, pcs).map_err(cms_error)?;
                    for pixel in pcs.chunks_exact_mut(3) {
                        for (value, scale) in pixel.iter_mut().zip(self.absolute_scale) {
                            *value *= scale;
                        }
                    }
                    if pcs
                        .iter()
                        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                    {
                        return Err(invalid(
                            "absolute ICC result exceeds supported PCS domain; no clipping was applied",
                        ));
                    }
                    final_transform.transform(pcs, output).map_err(cms_error)?;
                } else {
                    transform.transform(input, output).map_err(cms_error)?;
                }
            } else if let Some(curve) = &self.gray_curve {
                for (&gray, pixel) in input.iter().zip(output.chunks_exact_mut(out_channels)) {
                    pixel.fill(curve.evaluate_value(gray));
                }
            } else if let Some(matrix) = &self.matrix {
                for (src, dst) in input
                    .chunks_exact(in_channels)
                    .zip(output.chunks_exact_mut(out_channels))
                {
                    for (row, dst) in matrix.v.iter().zip(dst) {
                        *dst = row
                            .iter()
                            .zip(src)
                            .map(|(coefficient, value)| coefficient * *value as f64)
                            .sum::<f64>() as f32;
                    }
                }
            } else {
                output.copy_from_slice(input);
            }
            if let Some(matrix) = &self.post_matrix {
                for pixel in output.chunks_exact_mut(3) {
                    let xyz = [pixel[0] as f64, pixel[1] as f64, pixel[2] as f64];
                    for (row, value) in matrix.v.iter().zip(pixel) {
                        *value = if self.djxl_cmyk {
                            // skcms stores the inverse matrix and evaluates it
                            // in F32, without fused multiply-add operations.
                            row[0] as f32 * xyz[0] as f32
                                + row[1] as f32 * xyz[1] as f32
                                + row[2] as f32 * xyz[2] as f32
                        } else {
                            row.iter().zip(xyz).map(|(a, b)| a * b).sum::<f64>() as f32
                        };
                    }
                }
            }
            for p in 0..len {
                let pixel = &mut output[p * out_channels..][..out_channels];
                if let Some(tone) = &self.tone {
                    tone.apply(pixel);
                }
                if let Some(curve) = self
                    .output
                    .curve
                    .as_ref()
                    .filter(|_| !self.preserve_encoding)
                {
                    if let Some(srgb) = &self.srgb_output {
                        for value in pixel.iter_mut() {
                            *value = srgb.apply(*value);
                        }
                    } else {
                        curve.encode(pixel, self.destination_nits);
                    }
                }
                if pixel.iter().any(|v| !v.is_finite()) {
                    return Err(invalid(
                        "non-finite CMS output or transfer-function overflow",
                    ));
                }
                let original =
                    &source[(start + p) * frame.layout.channels()..][..frame.layout.channels()];
                let alpha = frame_alpha(original, frame.layout);
                let dst = &mut result[(start + p) * layout.channels()..][..layout.channels()];
                for c in 0..out_channels {
                    dst[c] = if let Some(background) =
                        self.background.filter(|_| frame.layout.has_alpha())
                    {
                        // Alpha zero must not leak hidden color (including
                        // nonzero associated samples) into the flattened view.
                        if alpha == 0.0 {
                            background[c]
                        } else {
                            pixel[c] * alpha + background[c] * (1.0 - alpha)
                        }
                    } else if associated && frame.layout.has_alpha() && alpha > 0.0 {
                        pixel[c] * alpha
                    } else {
                        pixel[c]
                    };
                }
                if layout.has_alpha() {
                    dst[out_channels] = alpha;
                }
            }
        }
        Ok((layout, result))
    }
}

fn native_white(encoding: &JxlColorEncoding) -> Result<[f64; 3]> {
    let white = match encoding {
        JxlColorEncoding::RgbColorSpace { white_point, .. }
        | JxlColorEncoding::GrayscaleColorSpace { white_point, .. } => white_point,
        JxlColorEncoding::XYB { .. } => return Err(invalid("XYB output is unsupported")),
    };
    let (x, y) = white.to_xy_coords();
    if !x.is_finite() || !y.is_finite() || !(0.0..=1.0).contains(&x) || !(0.0 < y && y <= 1.0) {
        return Err(invalid("invalid white point"));
    }
    Ok([
        x as f64 / y as f64,
        1.0,
        (1.0 - x as f64 - y as f64) / y as f64,
    ])
}

fn checked_inverse(matrix: Matrix3d) -> Result<Matrix3d> {
    if matrix.v.iter().flatten().any(|value| !value.is_finite())
        || matrix
            .determinant()
            .is_none_or(|det| !det.is_finite() || det.abs() < 1e-12)
    {
        return Err(invalid("singular color transform matrix"));
    }
    Ok(matrix.inverse())
}

/// Matrix profiles have no perceptual/saturation LUT, so those intents use
/// relative colorimetry. Absolute preserves native whites by undoing CHAD.
/// Matrix operations use moxcms utilities, without its clamping TRC tables.
fn structured_matrix(input: &Endpoint, output: &Endpoint) -> Result<Matrix3d> {
    let absolute = output.cms.rendering_intent == CmsIntent::AbsoluteColorimetric;
    let rgb_to_xyz = |endpoint: &Endpoint| -> Result<Matrix3d> {
        let matrix = endpoint.cms.colorant_matrix();
        if absolute {
            let chad = endpoint
                .cms
                .chromatic_adaptation
                .ok_or_else(|| invalid("structured RGB profile is missing chromatic adaptation"))?;
            Ok(checked_inverse(chad)?.mat_mul(matrix))
        } else {
            Ok(matrix)
        }
    };
    let input_to_xyz = if input.layout == Layout::Gray {
        let white = if absolute {
            input
                .curve
                .as_ref()
                .ok_or_else(|| invalid("structured source required"))?
                .native_white
        } else {
            [0.964203, 1.0, 0.824905]
        };
        Matrix3d {
            v: white.map(|v| [v, 0.0, 0.0]),
        }
    } else {
        rgb_to_xyz(input)?
    };
    let xyz_to_output = if output.layout == Layout::Gray {
        Matrix3d {
            v: [[0.0, 1.0, 0.0], [0.0; 3], [0.0; 3]],
        }
    } else {
        checked_inverse(rgb_to_xyz(output)?)?
    };
    let matrix = xyz_to_output.mat_mul(input_to_xyz);
    if matrix.v.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite color transform matrix"));
    }
    Ok(matrix)
}

// Native-white RGB luminances, matching PrimariesToXYZ in libjxl. Do not use
// the D50-adapted ICC colorants for HDR luminance calculations.
fn luminances(encoding: &JxlColorEncoding) -> Result<[f32; 3]> {
    let JxlColorEncoding::RgbColorSpace {
        white_point,
        primaries,
        ..
    } = encoding
    else {
        encoding_parts(encoding)?;
        return Ok([1.0, 0.0, 0.0]);
    };
    if *white_point == JxlWhitePoint::D65 && *primaries == JxlPrimaries::SRGB {
        return Ok([0.2126, 0.7152, 0.0722]);
    }
    let (wx, wy) = white_point.to_xy_coords();
    if !wx.is_finite() || !wy.is_finite() || !(0.0..=1.0).contains(&wx) || !(0.0 < wy && wy <= 1.0)
    {
        return Err(invalid("invalid white point"));
    }
    let primaries = primaries.to_xy_coords();
    let columns = primaries.map(|(x, y)| [x, y, 1.0 - x - y]);
    let matrix = Matrix3d {
        v: std::array::from_fn(|r| std::array::from_fn(|c| columns[c][r] as f64)),
    };
    let inverse = checked_inverse(matrix)?;
    let white = [wx / wy, 1.0, (1.0 - wx - wy) / wy];
    // PrimariesToXYZ uses F32 matrices with double arithmetic between stores.
    // Preserve those boundaries: a one-ULP weight difference is amplified by
    // gamut mapping when a highlight's third component is near black.
    let result = std::array::from_fn(|i| {
        let scale = inverse.v[i]
            .iter()
            .zip(white)
            .map(|(a, b)| (*a as f32) as f64 * b as f64)
            .sum::<f64>() as f32;
        (scale as f64 * columns[i][1] as f64) as f32
    });
    if result.iter().any(|v: &f32| !v.is_finite()) {
        return Err(invalid("invalid primary luminances"));
    }
    Ok(result)
}

fn luma(pixel: &[f32], weights: [f32; 3]) -> f32 {
    if pixel.len() == 1 {
        pixel[0]
    } else {
        weights[0].mul_add(
            pixel[0],
            weights[1].mul_add(pixel[1], weights[2] * pixel[2]),
        )
    }
}

impl Curve {
    fn decode(&self, pixel: &mut [f32], nits: f32) {
        for value in pixel.iter_mut() {
            *value = decode_transfer(&self.transfer, *value, nits);
        }
        if self.transfer == JxlTransferFunction::HLG {
            apply_ootf(pixel, self.luminances, hlg_gamma(nits) - 1.0);
        }
    }

    fn encode(&self, pixel: &mut [f32], nits: f32) {
        if self.transfer == JxlTransferFunction::HLG {
            apply_ootf(pixel, self.luminances, 1.0 / hlg_gamma(nits) - 1.0);
        }
        for value in pixel {
            *value = encode_transfer(&self.transfer, *value, nits);
        }
    }
}

fn decode_transfer(transfer: &JxlTransferFunction, value: f32, nits: f32) -> f32 {
    use JxlTransferFunction::*;
    let x = value.abs() as f64;
    let y = match transfer {
        Linear => return value,
        PQ => return pq_display(value, nits),
        SRGB => {
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        }
        BT709 => {
            // BT.709's negative extension is linear, unlike sRGB's mirrored
            // curve. This follows TF_709 in transfer_functions-inl.h.
            return if value < 0.081 {
                value / 4.5
            } else {
                (((value as f64 + 0.099) / 1.099).powf(1.0 / 0.45)) as f32
            };
        }
        DCI => x.powf(2.6),
        Gamma(gamma) => x.powf(1.0 / *gamma as f64),
        HLG => {
            if x <= 0.5 {
                x * x / 3.0
            } else {
                (((x - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
            }
        }
    };
    (y as f32).copysign(value)
}

fn encode_transfer(transfer: &JxlTransferFunction, value: f32, nits: f32) -> f32 {
    use JxlTransferFunction::*;
    let x = value.abs() as f64;
    let y = match transfer {
        Linear => return value,
        // libjxl's output stage normalizes to 10000 nits before selecting the
        // approximation branch. Selecting on display-relative values changes
        // near-black output when the display peak is not 10000 nits.
        PQ => return pq_encoded(value * (nits / 10000.0), 10000.0),
        SRGB => {
            if x <= 0.0031308 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            }
        }
        BT709 => {
            return if value <= 0.018 {
                4.5 * value
            } else {
                (1.099 * (value as f64).powf(0.45) - 0.099) as f32
            };
        }
        DCI => x.powf(1.0 / 2.6),
        Gamma(gamma) => x.powf(*gamma as f64),
        HLG => {
            if x <= 1.0 / 12.0 {
                (3.0 * x).sqrt()
            } else {
                HLG_A * (12.0 * x - HLG_B).ln() + HLG_C
            }
        }
    };
    (y as f32).copysign(value)
}

const HLG_A: f64 = 0.17883277;
const HLG_B: f64 = 0.28466892;
const HLG_C: f64 = 0.55991073;

fn rational(x: f32, numerator: [f32; 5], denominator: [f32; 5]) -> f32 {
    let evaluate = |coefficients: [f32; 5]| {
        coefficients[..4]
            .iter()
            .rev()
            .fold(coefficients[4], |v, c| v.mul_add(x, *c))
    };
    evaluate(numerator) / evaluate(denominator)
}

// libjxl TF_PQ's SIMD approximation is deliberately distinct from TF_PQ_Base.
// Its nonzero black offset matters after tone mapping; exact powf is not an
// interchangeable implementation near black.
#[allow(clippy::excessive_precision)] // Retain the upstream polynomial literals.
fn pq_display(encoded: f32, nits: f32) -> f32 {
    let x = encoded.abs();
    let y = rational(
        x.mul_add(x, x),
        [
            2.62975656e-4,
            -6.23553089e-3,
            7.38602301e-1,
            2.64553172,
            5.50034862e-1,
        ],
        [
            4.21350107e2,
            -4.28736818e2,
            1.74364667e2,
            -3.39078883e1,
            2.67718770,
        ],
    );
    (y * (10000.0 / nits)).copysign(encoded)
}

#[allow(clippy::excessive_precision)] // Retain the upstream polynomial literals.
fn pq_encoded(linear: f32, nits: f32) -> f32 {
    let x = linear.abs();
    let root = (x * (nits * 1e-4)).sqrt().sqrt();
    let value = if x < 1e-4 {
        rational(
            root,
            [
                9.863406e-6,
                3.881234e-1,
                1.352821e2,
                6.889862e4,
                -2.864824e5,
            ],
            [3.371868e1, 1.477719e3, 1.608477e4, -4.389884e4, -2.072546e5],
        )
    } else {
        rational(
            root,
            [1.351392e-2, -1.095778, 5.522776e1, 1.492516e2, 4.838434e1],
            [1.012416, 2.016708e1, 9.263710e1, 1.120607e2, 2.590418e1],
        )
    };
    value.copysign(linear)
}

fn hlg_gamma(nits: f32) -> f32 {
    1.2 * 1.111f32.powf((nits / 1000.0).log2())
}

fn apply_ootf(pixel: &mut [f32], weights: [f32; 3], exponent: f32) {
    if exponent.abs() <= 0.01 {
        return;
    }
    let luminance = luma(pixel, weights).max(0.0);
    let ratio = luminance.powf(exponent).min(1e9);
    for value in pixel {
        *value *= ratio;
    }
}

// TF_PQ_Base with display_intensity_target = 1: linear values are in nits.
#[cfg(test)]
fn pq_decode(value: f32) -> f32 {
    if value == 0.0 {
        return value;
    }
    let power = (value.abs() as f64).powf(1.0 / (2523.0 / 32.0));
    let numerator = (power - 3424.0 / 4096.0).max(0.0);
    let denominator = 2413.0 / 128.0 - 2392.0 / 128.0 * power;
    ((numerator / denominator).powf(1.0 / (2610.0 / 16384.0)) as f32 * 10000.0).copysign(value)
}

fn pq_encode(nits: f32) -> f32 {
    if nits == 0.0 {
        return nits;
    }
    let power = (nits.abs() as f64 * 1e-4_f32 as f64).powf(2610.0 / 16384.0);
    let numerator = 3424.0 / 4096.0 + 2413.0 / 128.0 * power;
    let denominator = 1.0 + 2392.0 / 128.0 * power;
    ((numerator / denominator).powf(2523.0 / 32.0) as f32).copysign(nits)
}

enum DisplayMap {
    Pq {
        source: f32,
        target: f32,
        weights: [f32; 3],
    },
    Hlg {
        exponent: f32,
        weights: [f32; 3],
    },
}

impl DisplayMap {
    fn new(
        input: &Endpoint,
        output: &Endpoint,
        hdr_origin: Option<&JxlTransferFunction>,
        source: f32,
        desired: Option<f32>,
    ) -> Result<Option<Self>> {
        let Some(target) = desired.filter(|target| *target != source) else {
            return Ok(None);
        };
        let (Some(input), Some(output)) = (&input.curve, &output.curve) else {
            return Err(invalid(
                "display-nits adjustment requires structured source and destination profiles",
            ));
        };
        let weights = output.luminances;
        let origin = hdr_origin.unwrap_or(&input.transfer);
        Ok(if *origin == JxlTransferFunction::PQ && target < source {
            Some(Self::Pq {
                source,
                target,
                weights,
            })
        } else if *origin == JxlTransferFunction::HLG && output.transfer != JxlTransferFunction::HLG
        {
            Some(Self::Hlg {
                exponent: 1.111f32.powf((target / source).log2()) - 1.0,
                weights,
            })
        } else {
            None
        })
    }

    fn apply(&self, pixel: &mut [f32]) {
        match *self {
            Self::Pq {
                source,
                target,
                weights,
            } => {
                let luminance = source * luma(pixel, weights);
                let pq_min = pq_encode(0.0);
                let pq_range = pq_encode(source) - pq_min;
                let inverse_range = 1.0 / pq_range;
                let normalized = ((pq_encoded(luminance, 1.0) - pq_min) * inverse_range).min(1.0);
                let max_lum = (pq_encode(target) - pq_min) * inverse_range;
                let ks = 1.5 * max_lum - 0.5;
                let e2 = if normalized < ks {
                    normalized
                } else {
                    let t = (normalized - ks) * (1.0 / (1.0 - ks).max(1e-6));
                    let t2 = t * t;
                    let t3 = t2 * t;
                    2.0_f32.mul_add(t3, (-3.0_f32).mul_add(t2, 1.0)).mul_add(
                        ks,
                        (t3 + (-2.0_f32).mul_add(t2, t))
                            .mul_add(1.0 - ks, (-2.0_f32).mul_add(t3, 3.0 * t2) * max_lum),
                    )
                };
                // Both black levels are zero, so Rec2408's min_lum term is 0.
                let new_luminance =
                    pq_display(e2.mul_add(pq_range, pq_min), 1.0).clamp(0.0, target);
                for value in pixel.iter_mut() {
                    *value = if luminance <= 1e-6 {
                        new_luminance / target
                    } else {
                        *value * ((new_luminance / luminance) * (source / target))
                    };
                }
                gamut_map(pixel, weights);
            }
            Self::Hlg { exponent, weights } => {
                apply_ootf(pixel, weights, exponent);
                if exponent < -0.01 {
                    gamut_map(pixel, weights);
                }
            }
        }
    }
}

// Scalar translation of libjxl cms/tone_mapping.h:GamutMapScalar.
fn gamut_map(pixel: &mut [f32], weights: [f32; 3]) {
    let luminance = luma(pixel, weights);
    let mut saturation: f32 = 0.0;
    let mut lightness: f32 = 0.0;
    for &value in pixel.iter() {
        let difference = value - luminance;
        let reciprocal = 1.0 / if difference == 0.0 { 1.0 } else { difference };
        let ratio = value * reciprocal;
        if difference < 0.0 {
            saturation = saturation.max(ratio);
        }
        lightness = lightness.max(if difference <= 0.0 {
            saturation
        } else {
            ratio - reciprocal
        });
    }
    let mix = 0.1_f32
        .mul_add(saturation - lightness, lightness)
        .clamp(0.0, 1.0);
    for value in pixel.iter_mut() {
        *value = mix.mul_add(luminance - *value, *value);
    }
    let maximum = pixel.iter().copied().fold(1.0f32, f32::max);
    for value in pixel {
        *value *= 1.0 / maximum;
    }
}

fn frame_alpha(pixel: &[f32], layout: PixelLayout) -> f32 {
    if layout.has_alpha() {
        pixel[layout.channels() - 1]
    } else {
        1.0
    }
}

fn gather_color(
    source: &[f32],
    target: &mut [f32],
    black: Option<f32>,
    associated: bool,
    alpha: f32,
) {
    let unassociate = |v| {
        if associated && alpha > 0.0 {
            v / alpha
        } else {
            v
        }
    };
    if let Some(black) = black {
        // JXL stores inverted CMYK. The Rgba CMS layout's fourth sample is K,
        // not the frame's interleaved alpha. Extra-plane K is never modified.
        for c in 0..3 {
            target[c] = 1.0 - unassociate(source[c]);
        }
        target[3] = 1.0 - unassociate(black);
    } else {
        for (target, source) in target.iter_mut().zip(source) {
            *target = unassociate(*source);
        }
    }
}

#[cfg(test)]
#[path = "color_tests.rs"]
mod tests;

#[path = "color_skcms.rs"]
mod skcms;
