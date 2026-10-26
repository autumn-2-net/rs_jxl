//! Decoder CLI orchestration. Controls, codecs, metadata and publication are separate.
use crate::{Result, io};
use rs_jxl::decode::{
    self, DecodeOptions, DecodedBox, DecodedFrame, DecodedImage, ExtraChannel, JxlBasicInfo,
    JxlBitDepth, SampleFormat,
};
use rs_jxl::{PixelData, PixelLayout};
use std::path::{Path, PathBuf};

mod controls;
mod metadata;
mod outputs;
mod streaming;
mod writers;
pub(crate) use controls::DecodeArgs;
use controls::OutputFormat;

const MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
#[cfg(test)]
use writers::{png_depth, write_npy, write_portable};

pub(crate) fn run(args: DecodeArgs) -> Result<()> {
    let _total = rs_jxl::profile::scope("cli.total");
    let format = args.output_format()?;
    outputs::preflight(&args)?;
    if args.stream_input {
        return streaming::run(&args, format);
    }
    let bytes = io::read_input(&args.input)?;
    let mut options = args.options(format);
    args.configure_color(&mut options, format)?;
    if args.verbose > 0 {
        args.diagnostic(format!("Read {} input bytes.", bytes.len()));
    }
    let started = std::time::Instant::now();
    if format == OutputFormat::Jpg {
        return reconstruct(&args, &bytes, &options);
    }
    let inspection_only = !args.disable_output
        && (format.metadata()
            || (args.output.is_none() && args.preview_out.is_none() && options.color.is_noop()));
    if inspection_only {
        let mut inspected = None;
        for _ in 0..args.num_reps {
            drop(inspected.take());
            inspected = Some(decode::inspect(&bytes, &options)?);
        }
        let image = inspected.ok_or("no inspection repetitions")?;
        let mut batch = outputs::Batch::new(&args);
        if let Some(path) = &args.output {
            let output = metadata::extract(&image.boxes, format, batch.remaining())?;
            batch.add(path, output, true)?;
        }
        batch.sidecars(&metadata::Metadata::from(&image))?;
        report(&args, image.bytes_read, started);
        return batch.publish();
    }
    if format == OutputFormat::Png && matches!(options.format, SampleFormat::Native) {
        options.format = default_integer_format(&decode::read_basic_info(&bytes, &options)?)?;
    }
    let mut decoder = decode::Decoder::new(options.clone())?;
    let mut decoded = None;
    for _ in 0..args.num_reps {
        drop(decoded.take());
        let _decoding = rs_jxl::profile::scope("cli.decode");
        decoded = Some(match args.frame {
            Some(index) => decoder.decode_frame(&bytes, index)?,
            None => decoder.decode(&bytes)?,
        });
    }
    let image = decoded.ok_or("no decoding repetitions")?;
    if args.frame.is_none()
        && !image.complete
        && !(args.allow_partial_files || args.downsampling > 1)
    {
        return Err(
            "decoder returned incomplete output without an explicit partial/progressive request"
                .into(),
        );
    }
    if args.frame.is_some() && image.frames.len() != 1 {
        return Err("frame selection did not return exactly one frame".into());
    }
    if !image.complete {
        args.diagnostic("warning: selected/partial/progressive output is incomplete; later frames or trailing metadata may not have been validated.");
    }
    report(&args, image.bytes_read, started);
    if args.disable_output {
        return Ok(());
    }
    diagnostics(&args, &image, format);
    let mut batch = outputs::Batch::new(&args);
    outputs::pixels(&mut batch, &image, format, &options)?;
    outputs::preview(&mut batch, &image, &options)?;
    if args.frame.is_some() && (args.metadata_out.is_some() || args.orig_icc_out.is_some()) {
        let inspected = decode::inspect(&bytes, &options)?;
        let metadata = metadata::Metadata {
            output_profile: image.output_profile.as_ref(),
            output_intensity_target: image.output_intensity_target,
            complete: image.complete,
            ..metadata::Metadata::from(&inspected)
        };
        batch.sidecars(&metadata)?;
    } else {
        batch.sidecars(&metadata::Metadata::from(&image))?;
    }
    batch.publish()
}

fn report(args: &DecodeArgs, bytes_read: usize, started: std::time::Instant) {
    if args.print_read_bytes {
        args.diagnostic(format!("Decoded bytes: {bytes_read}"));
    }
    if args.verbose > 0 || args.num_reps > 1 {
        args.diagnostic(format!(
            "{} repetition(s) in {:.6} s.",
            args.num_reps,
            started.elapsed().as_secs_f64()
        ));
    }
}

fn reconstruct(args: &DecodeArgs, bytes: &[u8], options: &DecodeOptions) -> Result<()> {
    let started = std::time::Instant::now();
    let mut decoder = decode::Decoder::new(options.clone())?;
    let mut output = Vec::new();
    for _ in 0..args.num_reps {
        drop(std::mem::take(&mut output));
        output = decoder.reconstruct_jpeg(bytes)?;
        if output.len() > MAX_OUTPUT_BYTES {
            return Err("reconstructed JPEG exceeds 512 MiB".into());
        }
    }
    let inspected = if args.print_read_bytes
        || (!args.disable_output
            && (args.icc_out.is_some()
                || args.orig_icc_out.is_some()
                || args.metadata_out.is_some()))
    {
        Some(decode::inspect(bytes, options)?)
    } else {
        None
    };
    report(
        args,
        inspected.as_ref().map_or(0, |i| i.bytes_read),
        started,
    );
    if args.disable_output {
        return Ok(());
    }
    let mut batch = outputs::Batch::new(args);
    if let Some(path) = &args.output {
        batch.add(path, output, true)?;
    } else {
        return Err("JPEG reconstruction requires an output or --disable-output".into());
    }
    if let Some(image) = &inspected {
        batch.sidecars(&metadata::Metadata::from(image))?;
    }
    batch.publish()
}

fn diagnostics(args: &DecodeArgs, image: &DecodedImage, format: OutputFormat) {
    if args.output.is_some() && format != OutputFormat::Png {
        args.diagnostic("warning: this raster format does not embed JXL boxes or animation timing; use --metadata-out and metadata/ICC sidecars as needed.");
        if format != OutputFormat::Exr {
            args.diagnostic("warning: samples use the decoder's output color space without an embedded ICC profile.");
        }
    }
    if format == OutputFormat::Png {
        for metadata in &image.boxes {
            if !matches!(&metadata.kind, b"Exif" | b"xml ") {
                args.diagnostic(format!(
                    "warning: PNG cannot preserve box {:?} ({} bytes).",
                    String::from_utf8_lossy(&metadata.kind),
                    metadata.data.len()
                ));
            }
        }
        if !args.keep_orientation
            && image.info.orientation as u16 != 1
            && image.boxes.iter().any(|b| &b.kind == b"xml ")
        {
            args.diagnostic(
                "warning: retained XMP may describe the original orientation/dimensions.",
            );
        }
    }
}

fn default_integer_format(info: &JxlBasicInfo) -> Result<SampleFormat> {
    let mut bits = 0;
    for depth in
        std::iter::once(&info.bit_depth).chain(info.extra_channels.iter().map(|ec| &ec.bit_depth))
    {
        match depth {
            JxlBitDepth::Int { bits_per_sample } if (1..=16).contains(bits_per_sample) => bits = bits.max(*bits_per_sample),
            _ => return Err("float/wide color or extra channels require explicit --bits 8/16 for PNG; use EXR/NPY to avoid quantization".into()),
        }
    }
    Ok(if bits <= 8 {
        SampleFormat::U8
    } else {
        SampleFormat::U16
    })
}

#[cfg(test)]
fn serialize(
    image: &DecodedImage,
    format: OutputFormat,
    selected: bool,
    keep_orientation: bool,
) -> Result<Vec<u8>> {
    writers::serialize_raster(
        writers::Raster::from(image),
        format,
        selected,
        keep_orientation,
        SampleFormat::Native,
        false,
        MAX_OUTPUT_BYTES,
    )
}
