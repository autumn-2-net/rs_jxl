//! Pixel decoding and bounded, transactional CLI output serialization.

use std::borrow::Cow;
use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use rs_jxl::decode::{
    self, DecodeOptions, DecodedBox, DecodedFrame, DecodedImage, ExtraChannel, JxlBasicInfo,
    JxlBitDepth, SampleFormat,
};
use rs_jxl::{PixelData, PixelLayout};

use crate::{Result, io, parse_threads};

const MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Png,
    Npy,
    Ppm,
    Pgm,
    Pfm,
    #[value(alias = "jpeg")]
    Jpg,
}

#[derive(Debug, Args)]
#[command(
    after_long_help = "PNG writes all displayed frames as APNG with exact timing when representable. \
NPY writes one H x W x C array, or an F x H x W x C stack for multiple frames. \
NPY writes main color channels followed by all extra planes, including alpha once, \
in codestream order. Shapes and sample types must agree; mixed native storage types \
require --bits 32. PPM/PGM/PFM require a single frame and cannot store alpha \
or extra channels. Use --frame to explicitly select a zero-based displayed frame.\n\
PNG/PPM/PGM choose 8 or 16 bits from the source color/alpha depth. Float or wider-than-16-bit sources \
require an explicit --bits for these formats. PFM defaults to float32; NPY defaults \
to the decoder's native sample format. --bits 32 means float32, not uint32.\n\
PNG requires straight alpha and retains the OUTPUT ICC profile, EXIF and UTF-8 XMP. \
--frame stops early, so trailing metadata is unavailable. Other formats contain \
untagged samples, without color conversion, profiles, frame timing or metadata; \
diagnostics report these omissions. JPEG output reconstructs the original JPEG bytes \
without pixel re-encoding or a fallback; --frame and --bits are not accepted.\n\
Use '-' for stdin/stdout; stdout requires --format. Diagnostics go to stderr. \
Output is published only after serialization succeeds, with a 512 MiB output limit. \
Existing output is not replaced unless --force is supplied."
)]
pub(crate) struct DecodeArgs {
    /// JPEG XL input path, or '-' for stdin
    input: PathBuf,
    /// PNG, NPY, PPM, PGM, PFM or reconstructed JPEG output path, or '-'
    output: PathBuf,
    /// Output file format (required for stdout, otherwise inferred from extension)
    #[arg(long, value_enum)]
    format: Option<OutputFormat>,
    /// Select one zero-based displayed frame instead of exporting all frames
    #[arg(long, value_name = "INDEX")]
    frame: Option<usize>,
    /// Decoder worker count (1..=256)
    #[arg(long, value_name = "N", default_value = "1", value_parser = parse_threads)]
    threads: usize,
    /// Output sample depth: 8/16-bit unsigned integer or 32-bit float
    #[arg(long, value_name = "8|16|32", value_parser = parse_bits)]
    bits: Option<u8>,
    /// Leave pixels in stored orientation instead of applying the JXL transform
    #[arg(long)]
    keep_orientation: bool,
    /// Allow Level 10 complexity without a jxll declaration (strict by default)
    #[arg(long)]
    allow_implicit_level10: bool,
    /// Replace an existing output file; input and output must still differ
    #[arg(long)]
    force: bool,
}

fn parse_bits(value: &str) -> std::result::Result<u8, String> {
    match value {
        "8" => Ok(8),
        "16" => Ok(16),
        "32" => Ok(32),
        _ => Err("bits must be 8, 16 or 32 (float32)".into()),
    }
}

impl DecodeArgs {
    fn output_format(&self) -> Result<OutputFormat> {
        let format = match self.format {
            Some(format) => format,
            None if self.output == Path::new("-") => {
                return Err("binary stdout requires --format png|npy|ppm|pgm|pfm|jpg".into());
            }
            None => {
                let extension = self
                    .output
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .ok_or("output has no recognized extension; specify --format")?;
                match extension.to_ascii_lowercase().as_str() {
                    "png" | "apng" => OutputFormat::Png,
                    "npy" => OutputFormat::Npy,
                    "ppm" => OutputFormat::Ppm,
                    "pgm" => OutputFormat::Pgm,
                    "pfm" => OutputFormat::Pfm,
                    "jpg" | "jpeg" => OutputFormat::Jpg,
                    _ => return Err("unsupported output extension; specify --format".into()),
                }
            }
        };
        match (format, self.bits) {
            (OutputFormat::Jpg, _) if self.frame.is_some() || self.bits.is_some() => {
                Err("JPEG reconstruction does not accept --frame or --bits; original bytes are reconstructed without pixel re-encoding".into())
            }
            (OutputFormat::Png | OutputFormat::Ppm | OutputFormat::Pgm, Some(32)) => {
                Err("PNG/PPM/PGM cannot store float32; use --bits 8/16, NPY or PFM".into())
            }
            (OutputFormat::Pfm, Some(8 | 16)) => {
                Err("PFM requires float32 samples; use --bits 32 or omit --bits".into())
            }
            _ => Ok(format),
        }
    }

    fn options(&self, format: OutputFormat) -> DecodeOptions {
        let samples = match self.bits {
            Some(8) => SampleFormat::U8,
            Some(16) => SampleFormat::U16,
            Some(32) => SampleFormat::F32,
            _ => match format {
                OutputFormat::Pfm => SampleFormat::F32,
                _ => SampleFormat::Native,
            },
        };
        DecodeOptions {
            threads: self.threads,
            format: samples,
            allow_implicit_main10: self.allow_implicit_level10,
            adjust_orientation: !self.keep_orientation,
            render_spot_colors: format != OutputFormat::Npy,
            include_extra_channels: true,
            ..DecodeOptions::default()
        }
    }
}

pub(crate) fn run(args: DecodeArgs) -> Result<()> {
    let _total = rs_jxl::profile::scope("cli.total");
    let format = args.output_format()?;
    io::check_paths(&args.input, &args.output, args.force)?;
    let bytes = io::read_input(&args.input)?;
    let mut options = args.options(format);
    if format == OutputFormat::Jpg {
        let reconstructed = decode::reconstruct_jpeg(&bytes, &options)?;
        if reconstructed.len() > MAX_OUTPUT_BYTES {
            return Err("reconstructed JPEG exceeds the CLI's 512 MiB output limit".into());
        }
        return io::write_output(&args.input, &args.output, args.force, &reconstructed);
    }
    if args.bits.is_none()
        && matches!(
            format,
            OutputFormat::Png | OutputFormat::Ppm | OutputFormat::Pgm
        )
    {
        let info = decode::read_basic_info(&bytes, &options)?;
        options.format = default_integer_format(&info)?;
    }
    let image = {
        let _decoding = rs_jxl::profile::scope("cli.decode");
        match args.frame {
            Some(index) => decode::decode_frame(&bytes, index, &options)?,
            None => decode::decode(&bytes, &options)?,
        }
    };
    if args.frame.is_none() && !image.complete {
        return Err("decoder returned an incomplete image; output was not written".into());
    }
    if args.frame.is_some() && image.frames.len() != 1 {
        return Err("frame selection did not return exactly one frame".into());
    }
    if !image.complete {
        eprintln!(
            "warning: --frame stops before trailing metadata and later frames are validated; EXIF/XMP may be unavailable."
        );
    }
    if format == OutputFormat::Npy && args.bits.is_none()
        && std::iter::once(&image.info.bit_depth).chain(image.info.extra_channels.iter().map(|ec| &ec.bit_depth))
            .any(|depth| matches!(depth, JxlBitDepth::Int { bits_per_sample } if *bits_per_sample > 16))
    {
        eprintln!("warning: native integers above 16 bits are normalized F32 by this decoder, not exact integer output.");
    }
    let output = serialize(&image, format, args.frame.is_some(), args.keep_orientation)?;
    io::write_output(&args.input, &args.output, args.force, &output)
}

fn default_integer_format(info: &JxlBasicInfo) -> Result<SampleFormat> {
    let mut bits = 0;
    let alpha = info
        .extra_channels
        .iter()
        .find(|ec| ec.ec_type == ExtraChannel::Alpha);
    for depth in std::iter::once(&info.bit_depth).chain(alpha.map(|ec| &ec.bit_depth)) {
        match depth {
            JxlBitDepth::Int { bits_per_sample } if (1..=16).contains(bits_per_sample) => {
                bits = bits.max(*bits_per_sample);
            }
            _ => return Err("float/wide color or alpha requires explicit --bits 8/16 for integer output; use NPY or PFM to avoid this quantization".into()),
        }
    }
    Ok(if bits <= 8 {
        SampleFormat::U8
    } else {
        SampleFormat::U16
    })
}

// The limit applies to compressed PNG bytes too, including metadata and headers.
#[derive(Default)]
struct Output(Vec<u8>);

impl Write for Output {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let new_len = self.0.len().checked_add(bytes.len());
        if new_len.is_none_or(|len| len > MAX_OUTPUT_BYTES) {
            return Err(std::io::Error::other(
                "output exceeds the CLI's 512 MiB limit",
            ));
        }
        self.0
            .try_reserve(bytes.len())
            .map_err(std::io::Error::other)?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn validate_frames(frames: &[DecodedFrame]) -> Result<&DecodedFrame> {
    let first = frames
        .first()
        .ok_or("decoder returned no displayed frames")?;
    for frame in frames {
        if frame.width == 0 || frame.height == 0 {
            return Err("decoded frame dimensions must be nonzero".into());
        }
        let pixels = frame
            .width
            .checked_mul(frame.height)
            .ok_or("pixel count overflow")?;
        let samples = pixels
            .checked_mul(frame.layout.channels())
            .ok_or("sample count overflow")?;
        if frame.pixels.len() != samples {
            return Err("decoded frame sample count does not match its dimensions/layout".into());
        }
        if frame.width != first.width
            || frame.height != first.height
            || frame.layout != first.layout
            || dtype(&frame.pixels) != dtype(&first.pixels)
            || frame.extra_planes.len() != first.extra_planes.len()
        {
            return Err("output requires matching frame dimensions, layouts and sample types; select --frame".into());
        }
        for plane in &frame.extra_planes {
            if plane.len() != pixels || dtype(plane) != dtype(&frame.pixels) {
                return Err("extra planes require full-resolution matching sample types; try explicit --bits or use the decoder API".into());
            }
        }
    }
    Ok(first)
}

fn serialize(
    image: &DecodedImage,
    format: OutputFormat,
    selected: bool,
    keep_orientation: bool,
) -> Result<Vec<u8>> {
    let first = validate_frames(&image.frames)?;
    if format == OutputFormat::Png {
        check_png_planes(image)?;
    }
    if format != OutputFormat::Png {
        eprintln!(
            "warning: output contains untagged samples in the decoder's output color space; ICC, EXIF/XMP, orientation metadata and animation timing are not stored. No additional color conversion is applied."
        );
    }
    let mut output = Output::default();
    match format {
        OutputFormat::Png => write_png(&mut output, image, selected, keep_orientation)?,
        OutputFormat::Npy => write_npy(&mut output, &image.frames)?,
        OutputFormat::Ppm | OutputFormat::Pgm | OutputFormat::Pfm => {
            if image.frames.len() != 1 {
                return Err(
                    "PPM/PGM/PFM cannot store multiple frames; select --frame or use PNG/NPY"
                        .into(),
                );
            }
            write_portable(&mut output, first, format)?;
        }
        OutputFormat::Jpg => {
            return Err("JPEG reconstruction must not use pixel serialization".into());
        }
    }
    Ok(output.0)
}

fn check_png_planes(image: &DecodedImage) -> Result<()> {
    let first = image
        .frames
        .first()
        .ok_or("decoder returned no displayed frames")?;
    let primary_alpha = image
        .info
        .extra_channels
        .iter()
        .position(|ec| ec.ec_type == ExtraChannel::Alpha);
    for frame in &image.frames {
        if frame.extra_planes.len() != image.info.extra_channels.len() {
            return Err("decoded extra planes do not match channel metadata".into());
        }
        if frame
            .extra_planes
            .iter()
            .enumerate()
            .any(|(index, _)| !frame.layout.has_alpha() || Some(index) != primary_alpha)
        {
            return Err(
                "this output cannot preserve non-primary extra planes; use NPY or the decoder API"
                    .into(),
            );
        }
        if let Some(alpha) = primary_alpha.and_then(|index| frame.extra_planes.get(index)) {
            let stride = frame.layout.channels();
            let matches = match (&frame.pixels, alpha) {
                (PixelData::U8(pixels), PixelData::U8(alpha)) => pixels
                    .chunks_exact(stride)
                    .map(|pixel| pixel[stride - 1])
                    .eq(alpha.iter().copied()),
                (PixelData::U16(pixels), PixelData::U16(alpha)) => pixels
                    .chunks_exact(stride)
                    .map(|pixel| pixel[stride - 1])
                    .eq(alpha.iter().copied()),
                _ => false,
            };
            if !matches {
                return Err("the separate primary alpha differs from interleaved alpha; PNG cannot preserve both, use NPY".into());
            }
        }
    }
    if first.layout.has_alpha()
        && primary_alpha.is_some_and(|index| image.info.extra_channels[index].alpha_associated)
    {
        return Err("PNG requires straight alpha but the decoder returned associated alpha; use NPY to preserve samples".into());
    }
    Ok(())
}

fn dtype(pixels: &PixelData) -> &'static str {
    match pixels {
        PixelData::U8(_) => "|u1",
        PixelData::U16(_) => "<u2",
        PixelData::U32(_) => "<u4",
        PixelData::F16(_) => "<f2",
        PixelData::F32(_) => "<f4",
    }
}

fn write_sample(output: &mut impl Write, pixels: &PixelData, index: usize) -> Result<()> {
    match pixels {
        PixelData::U8(values) => output.write_all(&[values[index]])?,
        PixelData::U16(values) | PixelData::F16(values) => {
            output.write_all(&values[index].to_le_bytes())?
        }
        PixelData::U32(values) => output.write_all(&values[index].to_le_bytes())?,
        PixelData::F32(values) => output.write_all(&values[index].to_le_bytes())?,
    }
    Ok(())
}

fn write_npy(output: &mut impl Write, frames: &[DecodedFrame]) -> Result<()> {
    let first = validate_frames(frames)?;
    if first.layout.has_alpha() && first.extra_planes.is_empty() {
        return Err(
            "NPY requires the separate primary alpha plane to preserve its source precision".into(),
        );
    }
    let color_channels = first.layout.channels() - usize::from(first.layout.has_alpha());
    let channels = color_channels
        .checked_add(first.extra_planes.len())
        .ok_or("channel count overflow")?;
    let shape = if frames.len() == 1 {
        format!("({}, {}, {})", first.height, first.width, channels)
    } else {
        format!(
            "({}, {}, {}, {})",
            frames.len(),
            first.height,
            first.width,
            channels
        )
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}",
        dtype(&first.pixels)
    );
    // NPY 1.0 has a ten-byte prefix and a newline-terminated, 64-byte-aligned header.
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let header_len = u16::try_from(header.len())?;
    first
        .width
        .checked_mul(first.height)
        .and_then(|n| n.checked_mul(channels))
        .and_then(|n| n.checked_mul(usize::from(first.pixels.bits() / 8)))
        .and_then(|n| n.checked_mul(frames.len()))
        .and_then(|n| n.checked_add(10 + header.len()))
        .filter(|&n| n <= MAX_OUTPUT_BYTES)
        .ok_or("NPY output exceeds the CLI's 512 MiB limit")?;
    output.write_all(b"\x93NUMPY\x01\x00")?;
    output.write_all(&header_len.to_le_bytes())?;
    output.write_all(header.as_bytes())?;
    for frame in frames {
        let stride = frame.layout.channels();
        for pixel in 0..frame.width * frame.height {
            for channel in 0..color_channels {
                write_sample(output, &frame.pixels, pixel * stride + channel)?;
            }
            for plane in &frame.extra_planes {
                write_sample(output, plane, pixel)?;
            }
        }
    }
    Ok(())
}

fn write_portable(
    output: &mut impl Write,
    frame: &DecodedFrame,
    format: OutputFormat,
) -> Result<()> {
    validate_frames(std::slice::from_ref(frame))?;
    if !frame.extra_planes.is_empty() || frame.layout.has_alpha() {
        return Err("PPM/PGM/PFM cannot store alpha or extra planes; use PNG or NPY".into());
    }
    let magic = match (format, frame.layout) {
        (OutputFormat::Ppm, PixelLayout::Rgb) => "P6",
        (OutputFormat::Pgm, PixelLayout::Gray) => "P5",
        (OutputFormat::Pfm, PixelLayout::Rgb) => "PF",
        (OutputFormat::Pfm, PixelLayout::Gray) => "Pf",
        _ => return Err("PPM requires RGB and PGM requires grayscale; no implicit color conversion is performed".into()),
    };
    match (&frame.pixels, format) {
        (PixelData::F32(values), OutputFormat::Pfm) => {
            write!(output, "{magic}\n{} {}\n-1.0\n", frame.width, frame.height)?;
            let row = frame.width * frame.layout.channels();
            for y in (0..frame.height).rev() {
                for sample in &values[y * row..(y + 1) * row] {
                    output.write_all(&sample.to_le_bytes())?;
                }
            }
        }
        (PixelData::U8(values), OutputFormat::Ppm | OutputFormat::Pgm) => {
            write!(output, "{magic}\n{} {}\n255\n", frame.width, frame.height)?;
            output.write_all(values)?;
        }
        (PixelData::U16(values), OutputFormat::Ppm | OutputFormat::Pgm) => {
            write!(output, "{magic}\n{} {}\n65535\n", frame.width, frame.height)?;
            for sample in values {
                output.write_all(&sample.to_be_bytes())?;
            }
        }
        _ => return Err("PPM/PGM require U8/U16 samples and PFM requires F32 samples".into()),
    }
    Ok(())
}

fn png_depth(pixels: &PixelData) -> Result<png::BitDepth> {
    match pixels {
        PixelData::U8(_) => Ok(png::BitDepth::Eight),
        PixelData::U16(_) => Ok(png::BitDepth::Sixteen),
        _ => Err("PNG supports only U8/U16 samples; explicitly choose --bits 8 or 16".into()),
    }
}

fn png_delay(ticks: u32, tps_numerator: u32, tps_denominator: u32) -> Result<(u16, u16)> {
    if tps_numerator == 0 || tps_denominator == 0 {
        return Err("invalid animation timebase".into());
    }
    let numerator = u64::from(ticks) * u64::from(tps_denominator);
    let denominator = u64::from(tps_numerator);
    let (mut a, mut b) = (numerator, denominator);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let numerator = u16::try_from(numerator / a);
    let denominator = u16::try_from(denominator / a);
    match (numerator, denominator) {
        (Ok(n), Ok(d)) => Ok((n, d)),
        _ => Err("frame timing cannot be represented exactly in APNG; select --frame or use NPY (without timing)".into()),
    }
}

fn write_png(
    output: &mut impl Write,
    image: &DecodedImage,
    selected: bool,
    keep_orientation: bool,
) -> Result<()> {
    let first = validate_frames(&image.frames)?;
    check_png_planes(image)?;
    let mut info = png::Info::with_size(u32::try_from(first.width)?, u32::try_from(first.height)?);
    info.bit_depth = png_depth(&first.pixels)?;
    info.color_type = match first.layout {
        PixelLayout::Gray => png::ColorType::Grayscale,
        PixelLayout::GrayAlpha => png::ColorType::GrayscaleAlpha,
        PixelLayout::Rgb => png::ColorType::Rgb,
        PixelLayout::Rgba => png::ColorType::Rgba,
    };
    // An embedded profile can describe a different color space than output pixels.
    let profile = image
        .output_profile
        .as_ref()
        .ok_or("PNG output requires the decoder's output color profile; refusing to assume sRGB")?;
    let icc = profile
        .try_as_icc()
        .ok_or("output color profile cannot be represented as ICC; refusing untagged PNG output")?;
    let color_space = rs_jxl::icc::validate_profile(icc.as_slice())?;
    let expected = if first.layout.channels() < 3 {
        rs_jxl::icc::IccColorSpace::Gray
    } else {
        rs_jxl::icc::IccColorSpace::Rgb
    };
    if color_space != expected {
        return Err("output ICC color space does not match the PNG pixel layout".into());
    }
    info.icc_profile = Some(Cow::Borrowed(icc.as_slice()));
    let orientation = if keep_orientation {
        image.info.orientation as u16
    } else {
        1
    };
    let mut exif = None;
    for metadata in &image.boxes {
        match &metadata.kind {
            b"Exif" => {
                if exif.is_some() {
                    return Err("multiple EXIF boxes cannot be preserved in one PNG".into());
                }
                exif = Some(png_exif(&metadata.data, orientation)?);
            }
            b"xml " => {
                let text = std::str::from_utf8(&metadata.data)
                    .map_err(|_| "XMP is not UTF-8 and cannot be preserved as PNG iTXt")?;
                info.utf8_text.push(png::text_metadata::ITXtChunk::new(
                    "XML:com.adobe.xmp",
                    text,
                ));
                if !keep_orientation && image.info.orientation as u16 != 1 {
                    eprintln!(
                        "warning: XMP is retained verbatim and may still describe the original orientation/dimensions."
                    );
                }
            }
            _ => warn_box(metadata),
        }
    }
    if exif.is_none() && orientation != 1 {
        exif = Some(orientation_exif(orientation));
    }
    info.exif_metadata = exif.map(Cow::Owned);
    let animation = if selected {
        None
    } else {
        image.info.animation.as_ref()
    };
    if image.frames.len() > 1 && animation.is_none() {
        return Err("multiple frames without an animation timebase cannot be written as APNG; use NPY or --frame".into());
    }
    let delays = animation
        .map(|animation| {
            image
                .frames
                .iter()
                .map(|frame| {
                    png_delay(
                        frame.duration_ticks,
                        animation.tps_numerator,
                        animation.tps_denominator,
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut encoder = png::Encoder::with_info(output, info)?;
    encoder.validate_sequence(true);
    if let Some(animation) = animation {
        encoder.set_animated(u32::try_from(image.frames.len())?, animation.num_loops)?;
        encoder.set_blend_op(png::BlendOp::Source)?;
        encoder.set_dispose_op(png::DisposeOp::None)?;
    }
    let mut writer = encoder.write_header()?;
    for (index, frame) in image.frames.iter().enumerate() {
        if let Some(delays) = &delays {
            writer.set_frame_delay(delays[index].0, delays[index].1)?;
        }
        match &frame.pixels {
            PixelData::U8(values) => writer.write_image_data(values)?,
            PixelData::U16(values) => {
                let length = values
                    .len()
                    .checked_mul(2)
                    .filter(|&n| n <= MAX_OUTPUT_BYTES)
                    .ok_or("PNG frame exceeds 512 MiB")?;
                let mut bytes = Vec::new();
                bytes.try_reserve_exact(length)?;
                for value in values {
                    bytes.extend_from_slice(&value.to_be_bytes());
                }
                writer.write_image_data(&bytes)?;
            }
            _ => return Err("PNG requires integer samples".into()),
        }
    }
    writer.finish()?;
    Ok(())
}

fn warn_box(metadata: &DecodedBox) {
    eprintln!(
        "warning: PNG cannot preserve JXL box {:?} ({} bytes).",
        String::from_utf8_lossy(&metadata.kind),
        metadata.data.len()
    );
}

fn orientation_exif(orientation: u16) -> Vec<u8> {
    let mut tiff = b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x01\0\0\0\0\0\0\0".to_vec();
    tiff[18..20].copy_from_slice(&orientation.to_le_bytes());
    tiff
}

fn png_exif(data: &[u8], orientation: u16) -> Result<Vec<u8>> {
    let offset = u32::from_be_bytes(
        data.get(..4)
            .ok_or("truncated JXL EXIF offset")?
            .try_into()?,
    );
    let start = usize::try_from(offset)?
        .checked_add(4)
        .ok_or("EXIF offset overflow")?;
    let mut tiff = data
        .get(start..)
        .ok_or("JXL EXIF offset is outside the box")?
        .to_vec();
    let little = match tiff.get(..4) {
        Some(b"II\x2a\0") => true,
        Some(b"MM\0\x2a") => false,
        _ => return Err("EXIF does not contain a supported TIFF header".into()),
    };
    let word = |bytes: &[u8]| -> u16 {
        let bytes = [bytes[0], bytes[1]];
        if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        }
    };
    let dword = |bytes: &[u8]| -> u32 {
        let bytes = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        }
    };
    let encode_word = |value: u16| {
        if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        }
    };
    let encode_dword = |value: u32| {
        if little {
            value.to_le_bytes()
        } else {
            value.to_be_bytes()
        }
    };
    let directory = usize::try_from(dword(tiff.get(4..8).ok_or("truncated EXIF header")?))?;
    if directory < 8 {
        return Err("invalid EXIF IFD0 offset".into());
    }
    let count = word(
        tiff.get(directory..directory.checked_add(2).ok_or("EXIF offset overflow")?)
            .ok_or("truncated EXIF IFD0")?,
    );
    let entries = directory.checked_add(2).ok_or("EXIF offset overflow")?;
    let end = entries
        .checked_add(usize::from(count) * 12)
        .ok_or("EXIF directory overflow")?;
    let next_end = end
        .checked_add(4)
        .filter(|&n| n <= tiff.len())
        .ok_or("truncated EXIF directory")?;
    let mut found = false;
    for index in 0..usize::from(count) {
        let entry = entries + index * 12;
        if word(&tiff[entry..entry + 2]) == 274 {
            if word(&tiff[entry + 2..entry + 4]) != 3 || dword(&tiff[entry + 4..entry + 8]) != 1 {
                return Err("malformed EXIF orientation tag".into());
            }
            tiff[entry + 8..entry + 10].copy_from_slice(&encode_word(orientation));
            found = true;
        }
    }
    if !found && orientation != 1 {
        // Append a replacement IFD0 so every existing absolute TIFF offset remains valid.
        let new_count = count.checked_add(1).ok_or("EXIF IFD0 is full")?;
        if tiff.len() % 2 != 0 {
            tiff.push(0);
        }
        let new_offset = u32::try_from(tiff.len())?;
        let mut replacement = Vec::new();
        replacement.extend_from_slice(&encode_word(new_count));
        let insertion = (0..usize::from(count))
            .find(|&index| word(&tiff[entries + index * 12..entries + index * 12 + 2]) > 274)
            .unwrap_or(usize::from(count));
        let split = entries + insertion * 12;
        replacement.extend_from_slice(&tiff[entries..split]);
        replacement.extend_from_slice(&encode_word(274));
        replacement.extend_from_slice(&encode_word(3));
        replacement.extend_from_slice(&encode_dword(1));
        replacement.extend_from_slice(&encode_word(orientation));
        replacement.extend_from_slice(&[0, 0]);
        replacement.extend_from_slice(&tiff[split..next_end]);
        tiff.extend_from_slice(&replacement);
        tiff[4..8].copy_from_slice(&encode_dword(new_offset));
    }
    Ok(tiff)
}
