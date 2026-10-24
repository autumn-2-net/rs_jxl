//! Output-set planning and preflight; io.rs remains the only publication path.
use super::metadata::Metadata;
use super::writers::{Raster, integer_depth, serialize_plane, serialize_raster};
use super::*;
use std::fs;

struct Artifact {
    path: PathBuf,
    bytes: Vec<u8>,
    primary: bool,
}

pub(super) struct Batch<'a> {
    args: &'a DecodeArgs,
    artifacts: Vec<Artifact>,
    used: usize,
}

impl<'a> Batch<'a> {
    pub(super) fn new(args: &'a DecodeArgs) -> Self {
        Self {
            args,
            artifacts: Vec::new(),
            used: 0,
        }
    }
    pub(super) fn remaining(&self) -> usize {
        MAX_OUTPUT_BYTES - self.used
    }

    pub(super) fn add(&mut self, path: &Path, bytes: Vec<u8>, primary: bool) -> Result<()> {
        let used = self
            .used
            .checked_add(bytes.len())
            .filter(|&n| n <= MAX_OUTPUT_BYTES)
            .ok_or("output set exceeds the CLI's 512 MiB aggregate limit")?;
        check_destination(self.args, path)?;
        for previous in &self.artifacts {
            if path == Path::new("-") && previous.path == Path::new("-") {
                if !(primary && previous.primary) {
                    return Err("sidecars and primary output cannot share stdout".into());
                }
            } else if path != Path::new("-")
                && previous.path != Path::new("-")
                && path_key(path)? == path_key(&previous.path)?
            {
                return Err(format!(
                    "output destinations collide: '{}' and '{}'",
                    path.display(),
                    previous.path.display()
                )
                .into());
            }
        }
        self.artifacts.try_reserve(1)?;
        self.artifacts.push(Artifact {
            path: path.to_owned(),
            bytes,
            primary,
        });
        self.used = used;
        Ok(())
    }

    pub(super) fn sidecars(&mut self, metadata: &Metadata<'_>) -> Result<()> {
        if let Some(path) = &self.args.icc_out {
            let bytes = metadata::icc(metadata.output_profile, self.remaining())?;
            self.add(path, bytes, false)?;
        }
        if let Some(path) = &self.args.orig_icc_out {
            let bytes = metadata::icc(metadata.embedded_profile, self.remaining())?;
            self.add(path, bytes, false)?;
        }
        if let Some(path) = &self.args.metadata_out {
            let bytes = metadata::json_metadata(metadata, self.remaining())?;
            self.add(path, bytes, false)?;
        }
        Ok(())
    }

    pub(super) fn publish(self) -> Result<()> {
        // Preflight the entire set once more before the first file/stdout write.
        for artifact in &self.artifacts {
            check_destination(self.args, &artifact.path)?;
        }
        // Write regular files first, so stdout stays untouched on file failures.
        for stdout in [false, true] {
            for artifact in &self.artifacts {
                if (artifact.path == Path::new("-")) != stdout {
                    continue;
                }
                io::write_output(
                    &self.args.input,
                    &artifact.path,
                    self.args.force,
                    &artifact.bytes,
                )?;
                if self.args.verbose > 0 {
                    self.args.diagnostic(format!(
                        "Wrote {} bytes to {}",
                        artifact.bytes.len(),
                        artifact.path.display()
                    ));
                }
            }
        }
        Ok(())
    }
}

fn path_key(path: &Path) -> Result<PathBuf> {
    let name = path.file_name().ok_or("output must name a file")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let absolute = fs::canonicalize(parent)?.join(name);
    #[cfg(windows)]
    let absolute = PathBuf::from(absolute.to_string_lossy().to_lowercase());
    Ok(absolute)
}

fn check_destination(args: &DecodeArgs, path: &Path) -> Result<()> {
    io::check_paths(&args.input, path, args.force)?;
    if let Some(profile) = &args.color_profile {
        io::check_paths(profile, path, args.force)?;
    }
    if path != Path::new("-") {
        path_key(path)?;
    }
    Ok(())
}

pub(super) fn preflight(args: &DecodeArgs) -> Result<()> {
    if args.disable_output {
        return Ok(());
    }
    let paths: Vec<_> = args
        .output
        .iter()
        .chain(args.preview_out.iter())
        .chain(args.icc_out.iter())
        .chain(args.orig_icc_out.iter())
        .chain(args.metadata_out.iter())
        .collect();
    for (index, path) in paths.iter().enumerate() {
        check_destination(args, path)?;
        for previous in &paths[..index] {
            if path == previous
                || (path.as_path() != Path::new("-")
                    && previous.as_path() != Path::new("-")
                    && path_key(path)? == path_key(previous)?)
            {
                return Err("output/sidecar destinations must be distinct".into());
            }
        }
    }
    Ok(())
}

pub(super) fn numbered_path(
    path: &Path,
    frame: Option<(usize, usize)>,
    extra: Option<usize>,
    extension: Option<&str>,
) -> Result<PathBuf> {
    if path == Path::new("-") {
        return Ok(path.to_owned());
    }
    let mut name = path
        .file_stem()
        .ok_or("output must name a file")?
        .to_os_string();
    if let Some((index, count)) = frame {
        if index >= count {
            return Err("invalid output frame index".into());
        }
        let width = (count - 1).to_string().len();
        name.push(format!("-{index:0width$}"));
    }
    if let Some(index) = extra {
        name.push(format!("-ec{index}"));
    }
    if let Some(extension) = extension {
        name.push(".");
        name.push(extension);
    } else if let Some(extension) = path.extension() {
        name.push(".");
        name.push(extension);
    }
    Ok(path.with_file_name(name))
}

fn alpha_matches(frame: &DecodedFrame, index: usize) -> bool {
    if !frame.layout.has_alpha() {
        return false;
    }
    let stride = frame.layout.channels();
    match (&frame.pixels, &frame.extra_planes[index]) {
        (PixelData::U8(main), PixelData::U8(alpha)) => main
            .chunks_exact(stride)
            .map(|p| p[stride - 1])
            .eq(alpha.iter().copied()),
        (PixelData::U16(main), PixelData::U16(alpha)) => main
            .chunks_exact(stride)
            .map(|p| p[stride - 1])
            .eq(alpha.iter().copied()),
        _ => false,
    }
}

pub(super) fn pixels(
    batch: &mut Batch<'_>,
    image: &DecodedImage,
    format: OutputFormat,
    options: &DecodeOptions,
) -> Result<()> {
    let args = batch.args;
    let Some(path) = &args.output else {
        return Ok(());
    };
    let view = Raster {
        alpha_mode: options.alpha_mode,
        ..Raster::from(image)
    };
    let raw = !options.coalescing;
    let extras_separate = args.output_extra_channels
        || (args.alpha_blend
            && image.info.extra_channels.len() == 1
            && image.info.extra_channels[0].ec_type == ExtraChannel::Alpha);
    let split =
        args.output_frames || raw || !matches!(format, OutputFormat::Png | OutputFormat::Npy);
    if split && image.frames.len() > 1 && !(args.output_frames || raw) {
        return Err(
            "this format cannot store all frames; use --output-frames or select --frame".into(),
        );
    }
    if split {
        for (index, frame) in image.frames.iter().enumerate() {
            let destination = numbered_path(
                path,
                (image.frames.len() > 1).then_some((index, image.frames.len())),
                None,
                None,
            )?;
            let bytes = serialize_raster(
                Raster {
                    frames: std::slice::from_ref(frame),
                    ..view
                },
                format,
                true,
                args.keep_orientation,
                options.format,
                extras_separate,
                batch.remaining(),
            )?;
            batch.add(&destination, bytes, true)?;
        }
    } else {
        let bytes = serialize_raster(
            view,
            format,
            args.frame.is_some(),
            args.keep_orientation,
            options.format,
            extras_separate,
            batch.remaining(),
        )?;
        batch.add(path, bytes, true)?;
    }
    if args.output_extra_channels
        && !matches!(
            format,
            OutputFormat::Npy | OutputFormat::Pam | OutputFormat::Exr
        )
    {
        let primary = image
            .info
            .extra_channels
            .iter()
            .position(|ec| ec.ec_type == ExtraChannel::Alpha);
        for (index, info) in image.info.extra_channels.iter().enumerate() {
            for (frame_index, frame) in image.frames.iter().enumerate() {
                if format.stores_alpha() && primary == Some(index) && alpha_matches(frame, index) {
                    continue;
                }
                let plane = frame
                    .extra_planes
                    .get(index)
                    .ok_or("missing output extra plane")?;
                let (plane_format, extension, depth) = match plane {
                    PixelData::F32(_) => (OutputFormat::Pfm, "pfm", 32),
                    PixelData::U8(_) | PixelData::U16(_) => {
                        let depth = integer_depth(&info.bit_depth, options.format)?;
                        if format == OutputFormat::Png {
                            (OutputFormat::Png, "png", depth)
                        } else if format == OutputFormat::Pgx {
                            (OutputFormat::Pgx, "pgx", depth)
                        } else {
                            (OutputFormat::Pgm, "pgm", depth)
                        }
                    }
                    _ => {
                        return Err(
                            "extra-plane storage is unsupported; use NPY or --bits 32".into()
                        );
                    }
                };
                let destination = numbered_path(
                    path,
                    (image.frames.len() > 1).then_some((frame_index, image.frames.len())),
                    Some(index),
                    Some(extension),
                )?;
                let bytes = serialize_plane(frame, index, plane_format, depth, batch.remaining())?;
                batch.add(&destination, bytes, true)?;
            }
        }
    }
    Ok(())
}

pub(super) fn preview(
    batch: &mut Batch<'_>,
    image: &DecodedImage,
    options: &DecodeOptions,
) -> Result<()> {
    let args = batch.args;
    let Some(path) = &args.preview_out else {
        return Ok(());
    };
    if path == Path::new("-") {
        return Err("preview stdout has no independent --format; use a preview filename".into());
    }
    let format = OutputFormat::from_path(path)?;
    if format == OutputFormat::Jpg || format.metadata() {
        return Err("preview requires a pixel format".into());
    }
    let frame = image
        .preview
        .as_ref()
        .ok_or("input contains no decoded preview")?;
    let view = Raster {
        frames: std::slice::from_ref(frame),
        alpha_mode: options.alpha_mode,
        boxes: &[],
        ..Raster::from(image)
    };
    let bytes = serialize_raster(
        view,
        format,
        true,
        args.keep_orientation,
        options.format,
        false,
        batch.remaining(),
    )?;
    batch.add(path, bytes, false)
}
