//! Decoder controls and djxl-compatible option spellings.
use super::*;
use clap::{Args, ValueEnum};
use rs_jxl::decode::{AlphaMode, ColorProfile, DitherMode};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(super) enum DitherControl {
    #[default]
    #[value(name = "jxl-rs")]
    JxlRs,
    #[value(name = "libjxl")]
    Libjxl,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub(super) enum AlphaControl {
    #[default]
    AsIs,
    Premultiplied,
    Unpremultiplied,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(super) enum OutputFormat {
    #[value(alias = "apng")]
    Png,
    Npy,
    Ppm,
    Pgm,
    Pnm,
    Pam,
    Pfm,
    Pgx,
    Exr,
    Exif,
    #[value(alias = "xml")]
    Xmp,
    #[value(alias = "jumb")]
    Jumbf,
    #[value(alias = "jpeg")]
    Jpg,
}

impl OutputFormat {
    pub(super) fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|s| s.to_str())
            .ok_or("output has no recognized extension; specify --format")?;
        Self::from_str(extension, true)
            .map_err(|_| "unsupported output extension; specify --format".into())
    }

    pub(super) fn metadata(self) -> bool {
        matches!(self, Self::Exif | Self::Xmp | Self::Jumbf)
    }

    pub(super) fn integer(self) -> bool {
        matches!(
            self,
            Self::Png | Self::Ppm | Self::Pgm | Self::Pnm | Self::Pam | Self::Pgx
        )
    }

    pub(super) fn stores_alpha(self) -> bool {
        matches!(self, Self::Png | Self::Pam | Self::Npy | Self::Exr)
    }
}

#[derive(Debug, Args)]
#[command(
    after_long_help = "PNG stores all displayed frames as APNG; NPY stores all frames and extra planes. Other pixel formats require --frame or --output-frames for multiple frames. Output never silently drops extra planes: use --output-extra-channels for separate -ecN files. Numbered files use -N before the extension. Multiple primary streams are concatenated on stdout; sidecars must have distinct destinations.\nPNM/PPM/PGM/PAM/PGX default to original integer depth (1..16); PNG defaults to full-range 8/16-bit storage. --bits 32 means float32, not uint32. NPY preserves native wide integers and widens mixed integer planes without normalization. EXR uses float32 linear sRGB; no lossy JPEG encoder is included. JPEG output always requires successful original-byte reconstruction.\n--stream-input feeds one persistent pixel decoder from bounded reads; PNG then requires explicit --bits 8/16. It does not support JPEG reconstruction, metadata-only output, frame selection or repetitions. The CLI retains decoded frames for transactional publication; use the Rust streaming API to consume frames as they arrive.\nUse '-' for stdin/stdout; stdout requires --format. Diagnostics go to stderr. All serialization completes before publication; each file uses transactional io.rs publication. The output set has a 512 MiB byte budget; publishing multiple files is not a filesystem-wide atomic transaction."
)]
pub(crate) struct DecodeArgs {
    /// JPEG XL input path, or '-' for stdin
    pub(super) input: PathBuf,
    /// Decode pixel input incrementally without retaining the complete JXL file
    #[arg(long, conflicts_with_all = ["frame", "num_reps", "reconstruct_jpeg"])]
    pub(super) stream_input: bool,
    /// Incremental input read size (1..=1048576 bytes)
    #[arg(long, requires = "stream_input", value_parser = clap::value_parser!(u32).range(1..=1_048_576))]
    pub(super) input_chunk_size: Option<u32>,
    /// Output path, or '-'; optional with --disable-output or sidecars
    pub(super) output: Option<PathBuf>,
    /// Output format, otherwise inferred from the extension
    #[arg(long, aliases = ["output_format", "output-format"], value_enum, ignore_case = true)]
    pub(super) format: Option<OutputFormat>,
    /// Select one zero-based frame
    #[arg(long, value_name = "INDEX", conflicts_with = "output_frames")]
    pub(super) frame: Option<usize>,
    /// Worker count (1..=256); djxl --num_threads also accepts -1 and 0
    #[arg(long, default_value = "1", value_parser = crate::parse_threads)]
    pub(super) threads: usize,
    #[arg(long = "num_threads", alias = "num-threads", allow_hyphen_values = true, value_parser = parse_djxl_threads, conflicts_with = "threads")]
    pub(super) num_threads: Option<usize>,
    /// Sample depth: -1 format default, 0 original, 1..16 integer, 32 float
    #[arg(long, aliases = ["bits_per_sample", "bits-per-sample"], allow_hyphen_values = true, value_parser = parse_bits)]
    pub(super) bits: Option<i16>,
    /// Floating-to-U8 output pattern; does not affect JPEG reconstruction or native integers
    #[arg(long, value_enum, default_value = "jxl-rs")]
    pub(super) dither: DitherControl,
    #[arg(long, alias = "keep_orientation")]
    pub(super) keep_orientation: bool,
    #[arg(long)]
    pub(super) allow_implicit_level10: bool,
    #[arg(long)]
    pub(super) force: bool,
    /// Suppress diagnostics except errors
    #[arg(long)]
    pub(super) quiet: bool,
    #[arg(short = 'v', long, action = clap::ArgAction::Count)]
    pub(super) verbose: u8,
    /// Repeat decoding without repeating publication
    #[arg(long, alias = "num_reps", default_value = "1", value_parser = parse_reps)]
    pub(super) num_reps: usize,
    /// Decode without writing any image or sidecar
    #[arg(long, alias = "disable_output")]
    pub(super) disable_output: bool,
    #[arg(long, alias = "print_read_bytes")]
    pub(super) print_read_bytes: bool,
    /// Write numbered still files for every frame, including PNG and NPY
    #[arg(long, alias = "output_frames")]
    pub(super) output_frames: bool,
    #[arg(long, alias = "output_extra_channels")]
    pub(super) output_extra_channels: bool,
    #[arg(short = 'J', long, alias = "reconstruct_jpeg")]
    pub(super) reconstruct_jpeg: bool,
    #[arg(long = "norender-spotcolors", alias = "norender_spotcolors")]
    pub(super) norender_spotcolors: bool,
    #[arg(long, alias = "icc_out")]
    pub(super) icc_out: Option<PathBuf>,
    #[arg(long, alias = "orig_icc_out")]
    pub(super) orig_icc_out: Option<PathBuf>,
    #[arg(long, alias = "metadata_out")]
    pub(super) metadata_out: Option<PathBuf>,
    #[arg(long, alias = "preview_out")]
    pub(super) preview_out: Option<PathBuf>,
    #[arg(long, alias = "no_coalescing")]
    pub(super) no_coalescing: bool,
    #[arg(long, alias = "allow_partial_files")]
    pub(super) allow_partial_files: bool,
    #[arg(short = 's', long, default_value = "1", value_parser = parse_downsampling)]
    pub(super) downsampling: u8,
    /// Destination color descriptor (djxl spelling) or sRGB/linear alias
    #[arg(long, alias = "color_space", conflicts_with = "color_profile")]
    pub(super) color_space: Option<String>,
    /// Destination ICC profile file
    #[arg(long, alias = "icc-profile")]
    pub(super) color_profile: Option<PathBuf>,
    /// Use intent-aware legacy ICC CMYK tables instead of djxl conventions
    #[arg(long, requires = "color_space")]
    pub(super) cmyk_icc_conventions: bool,
    #[arg(long, alias = "display_nits", value_parser = parse_nits)]
    pub(super) display_nits: Option<f32>,
    #[arg(long, alias = "alpha_blend")]
    pub(super) alpha_blend: bool,
    #[arg(long, requires = "alpha_blend")]
    pub(super) background: Option<String>,
    #[arg(long, value_enum, default_value = "as-is")]
    pub(super) alpha_mode: AlphaControl,
}

fn parse_nits(value: &str) -> std::result::Result<f32, String> {
    value
        .parse::<f32>()
        .ok()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or("display-nits must be finite and nonnegative".into())
}

fn parse_bits(value: &str) -> std::result::Result<i16, String> {
    let bits = value.parse::<i16>().map_err(|_| "invalid bit depth")?;
    if (-1..=16).contains(&bits) || bits == 32 {
        Ok(bits)
    } else {
        Err("bits must be -1, 0, 1..16 or 32 (float32)".into())
    }
}

fn parse_reps(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or("num-reps must be positive".into())
}

fn parse_downsampling(value: &str) -> std::result::Result<u8, String> {
    match value {
        "1" => Ok(1),
        "2" => Ok(2),
        "4" => Ok(4),
        "8" => Ok(8),
        _ => Err("downsampling must be 1, 2, 4 or 8".into()),
    }
}

fn parse_djxl_threads(value: &str) -> std::result::Result<usize, String> {
    match value {
        "-1" => Ok(std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(256)),
        "0" => Ok(1),
        _ => crate::parse_threads(value),
    }
}

impl DecodeArgs {
    pub(super) fn output_format(&self) -> Result<OutputFormat> {
        let format = if let Some(format) = self.format {
            format
        } else if self.disable_output {
            if self.reconstruct_jpeg {
                OutputFormat::Jpg
            } else {
                OutputFormat::Npy
            }
        } else if let Some(path) = &self.output {
            if path == Path::new("-") {
                return Err("binary stdout requires --format".into());
            }
            OutputFormat::from_path(path)?
        } else if self.icc_out.is_some()
            || self.orig_icc_out.is_some()
            || self.metadata_out.is_some()
            || self.preview_out.is_some()
        {
            OutputFormat::Npy
        } else {
            return Err("specify an output, sidecar, or --disable-output".into());
        };
        if self.reconstruct_jpeg && format != OutputFormat::Jpg {
            return Err("--reconstruct-jpeg requires JPEG output".into());
        }
        if (format == OutputFormat::Jpg || format.metadata())
            && (self.color_space.is_some()
                || self.color_profile.is_some()
                || self.display_nits.is_some()
                || self.alpha_blend
                || !matches!(self.alpha_mode, AlphaControl::AsIs))
        {
            return Err("JPEG restoration and metadata extraction do not accept color/alpha transformations".into());
        }
        if self.no_coalescing && self.metadata_out.is_none() && !self.disable_output {
            return Err("--no-coalescing requires --metadata-out to preserve crop origins and blend/reference metadata".into());
        }
        if format == OutputFormat::Jpg
            && (self.frame.is_some()
                || self.bits.is_some()
                || self.output_frames
                || self.output_extra_channels
                || self.no_coalescing
                || self.allow_partial_files
                || self.downsampling != 1
                || self.keep_orientation
                || self.norender_spotcolors
                || self.preview_out.is_some())
        {
            return Err("JPEG reconstruction does not accept pixel/frame/partial controls; original bytes are reconstructed without pixel re-encoding".into());
        }
        if format.metadata()
            && (self.frame.is_some()
                || self.preview_out.is_some()
                || self.output_frames
                || self.output_extra_channels
                || self.bits.is_some()
                || self.allow_partial_files
                || self.downsampling != 1
                || self.no_coalescing)
        {
            return Err("metadata extraction requires complete input and does not accept pixel/frame controls".into());
        }
        match (format, self.bits) {
            (format, Some(32)) if format.integer() => {
                return Err("integer output cannot store float32; use EXR, PFM or NPY".into());
            }
            (OutputFormat::Png, Some(1..=7 | 9..=15)) => {
                return Err(
                    "PNG requires full-range --bits 8 or 16; use portable output for custom depths"
                        .into(),
                );
            }
            (OutputFormat::Pfm | OutputFormat::Exr, Some(0..=16)) => {
                return Err("PFM/EXR require --bits 32 or the format default".into());
            }
            _ => (),
        }
        Ok(format)
    }

    pub(super) fn options(&self, format: OutputFormat) -> DecodeOptions {
        let samples = match self.bits {
            Some(8) => SampleFormat::U8,
            Some(16) => SampleFormat::U16,
            Some(n @ 1..=7) => SampleFormat::U8Bits(n as u8),
            Some(n @ 9..=15) => SampleFormat::U16Bits(n as u8),
            Some(32) => SampleFormat::F32,
            _ if matches!(format, OutputFormat::Pfm | OutputFormat::Exr) => SampleFormat::F32,
            _ => SampleFormat::Native,
        };
        DecodeOptions {
            threads: self.num_threads.unwrap_or(self.threads),
            format: samples,
            dither: match self.dither {
                DitherControl::JxlRs => DitherMode::JxlRs,
                DitherControl::Libjxl => DitherMode::Libjxl,
            },
            allow_implicit_main10: self.allow_implicit_level10,
            adjust_orientation: !self.keep_orientation,
            render_spot_colors: !self.norender_spotcolors && format != OutputFormat::Npy,
            include_extra_channels: true,
            coalescing: !self.no_coalescing,
            include_preview: self.preview_out.is_some(),
            allow_partial: self.allow_partial_files,
            downsampling: self.downsampling,
            alpha_mode: match self.alpha_mode {
                AlphaControl::AsIs => AlphaMode::AsIs,
                AlphaControl::Premultiplied => AlphaMode::Premultiplied,
                AlphaControl::Unpremultiplied => AlphaMode::Unpremultiplied,
            },
            ..DecodeOptions::default()
        }
    }

    pub(super) fn configure_color(
        &self,
        options: &mut DecodeOptions,
        format: OutputFormat,
    ) -> Result<()> {
        options.color.output_profile = if format == OutputFormat::Exr {
            if self.color_profile.is_some()
                || self
                    .color_space
                    .as_deref()
                    .is_some_and(|s| !matches!(s, "RGB_D65_SRG_Rel_Lin" | "linear" | "linear-sRGB"))
            {
                return Err(
                    "EXR requires linear sRGB; omit --color-space/--color-profile or select linear"
                        .into(),
                );
            }
            Some(ColorProfile::parse("RGB_D65_SRG_Rel_Lin")?)
        } else if let Some(description) = &self.color_space {
            Some(ColorProfile::parse(description)?)
        } else if let Some(path) = &self.color_profile {
            if path == Path::new("-") {
                return Err("ICC input must be a file, not stdin".into());
            }
            let bytes = io::read_input(path)?;
            if bytes.len() > options.limits.max_metadata_bytes {
                return Err("ICC profile exceeds metadata byte budget".into());
            }
            rs_jxl::icc::validate_profile(&bytes)?;
            Some(ColorProfile::Icc(bytes))
        } else {
            None
        };
        if self.display_nits.is_some_and(|n| !n.is_finite() || n < 0.0) {
            return Err(
                "display-nits must be finite and nonnegative; zero disables the override".into(),
            );
        }
        options.color.display_nits = self.display_nits.filter(|&n| n != 0.0);
        options.color.cmyk_icc_conventions = self.cmyk_icc_conventions;
        if self.alpha_blend {
            options.color.background = Some(decode::parse_background(
                self.background.as_deref().unwrap_or("white"),
            )?);
        }
        Ok(())
    }

    pub(super) fn diagnostic(&self, message: impl std::fmt::Display) {
        if !self.quiet {
            eprintln!("{message}");
        }
    }
}
