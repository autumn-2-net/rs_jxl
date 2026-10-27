//! Ordered still-image input for the automatic lossless multiframe encoder.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use rs_jxl::codec::Encoder;
use rs_jxl::metadata::Animation;
use rs_jxl::multiframe::{AutoEncodeResult, AutoOptions, Reference};
use rs_jxl::{EncodeOptions, Image};

use crate::{Result, input, io, options::Tools, validators::parse_threads};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ReferenceArg {
    Auto,
    Previous,
    First,
    Full,
}

impl ReferenceArg {
    fn core(self) -> Reference {
        match self {
            Self::Auto => Reference::SmallestCrop,
            Self::Previous => Reference::Previous,
            Self::First => Reference::First,
            Self::Full => Reference::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Timebase {
    numerator: u32,
    denominator: u32,
}

fn parse_tps(value: &str) -> std::result::Result<Timebase, String> {
    let (numerator, denominator) = value.split_once('/').unwrap_or((value, "1"));
    let parse = |part: &str| {
        part.parse::<u32>()
            .map_err(|_| "tps must be an integer or NUM/DEN".to_owned())
    };
    let timebase = Timebase {
        numerator: parse(numerator)?,
        denominator: parse(denominator)?,
    };
    Animation {
        ticks_per_second_numerator: timebase.numerator,
        ticks_per_second_denominator: timebase.denominator,
        ..Animation::default()
    }
    .validate()
    .map_err(|e| e.to_string())?;
    Ok(timebase)
}

fn positive_usize(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| "limit must be a positive integer fitting usize".into())
}

#[derive(Debug, Args)]
#[command(
    after_long_help = "Inputs are processed in exactly the given order, without sorting. \
All frames must have identical dimensions, decoded sample/alpha types and metadata. \
No bit-depth, alpha, color or orientation conversion is performed. RGB under zero \
alpha is retained. PNG palette/packed gray and tRNS use the existing lossless \
input expansion. APNG and CMYK/YCCK or non-8-bit JPEG are unsupported.\n\
JPEG input is pixel-lossless relative to zune-jpeg, not byte-reversible. Supported \
PNG/JPEG color metadata is retained. Unsupported metadata requires explicit \
--strip-metadata; supported color metadata is retained even when stripping. \
Discarding unsupported metadata may change displayed orientation.\n\
Durations are integer ticks: seconds = ticks * tps denominator / tps numerator. \
--duration applies to every input; --durations requires one comma-separated \
value per input. Zero ticks mean a non-display layer, not a one-tick substitute.\n\
Automatic selection measures encoded bytes, not elapsed time. --reference auto \
permits previous and first references. Mask and multi-rectangle candidates are \
enabled by default when lossless and compatible. These are multiframe strategies, \
not source-search or an effort mapping. Manual tool options and --threads apply.\n\
No extra CLI input, decoded-sample or retained-sequence byte cap is imposed. \
Frame, trial and trial-pixel budgets are unlimited unless explicitly supplied. \
Format bounds, size-overflow checks and allocation failures still apply; errors \
do not silently disable tools or discard frames.\n\
Use '-' for at most one input from stdin, or -o - for binary stdout. Candidate \
reports and warnings go to stderr. Encoding finishes before output is written. \
Output must differ from every input, including with --force."
)]
pub(crate) struct EncodeFramesArgs {
    #[command(flatten)]
    tools: Tools,
    /// Ordered PNG/JPEG input paths (format detected by signature)
    #[arg(required = true, num_args = 1.., value_name = "INPUT")]
    inputs: Vec<PathBuf>,
    /// JXL output path, or '-' for binary stdout
    #[arg(short, long, value_name = "OUTPUT")]
    output: PathBuf,
    /// Uniform duration in ticks, including zero for a non-display layer
    #[arg(
        long,
        default_value_t = 100,
        conflicts_with = "durations",
        value_name = "TICKS"
    )]
    duration: u32,
    /// One tick count per input, comma-separated; may be repeated
    #[arg(long, value_delimiter = ',', num_args = 1, value_name = "TICKS,...")]
    durations: Vec<u32>,
    /// Ticks per second: positive integer or numerator/denominator
    #[arg(long, default_value = "1000", value_parser = parse_tps, value_name = "NUM[/DEN]")]
    tps: Timebase,
    /// Animation loop count; zero repeats indefinitely
    #[arg(long, default_value_t = 0, value_name = "N")]
    loops: u32,
    /// Reference candidates: auto permits both previous and first
    #[arg(long, value_enum, default_value = "auto")]
    reference: ReferenceArg,
    /// Disable the safe mask candidate
    #[arg(long)]
    no_mask: bool,
    /// Disable the safe multi-rectangle candidate
    #[arg(long)]
    no_multi_rect: bool,
    /// Tile side length for multi-rectangle planning
    #[arg(long, default_value_t = AutoOptions::default().tile_size, value_parser = clap::value_parser!(u32).range(1..), value_name = "PIXELS")]
    tile_size: u32,
    /// Maximum rectangles in a multi-rectangle candidate (positive integer)
    #[arg(long, default_value_t = AutoOptions::default().max_rectangles, value_parser = positive_usize, value_name = "N")]
    max_rectangles: usize,
    /// Logical input frame budget (omitted: unlimited; generated layers not counted)
    #[arg(long, default_value_t = AutoOptions::default().max_frames, hide_default_value = true, value_parser = positive_usize, value_name = "N")]
    max_frames: usize,
    /// Hard total trial-encode budget (omitted: unlimited)
    #[arg(long, default_value_t = AutoOptions::default().max_total_trials, hide_default_value = true, value_parser = positive_usize, value_name = "N")]
    max_total_trials: usize,
    /// Hard trial-pixel budget (omitted: unlimited)
    #[arg(long, default_value_t = AutoOptions::default().max_trial_pixels, hide_default_value = true, value_parser = clap::value_parser!(u64).range(1..), value_name = "N")]
    max_trial_pixels: u64,
    /// Encoder worker count (1..=256)
    #[arg(long, default_value = "1", value_parser = parse_threads, value_name = "N")]
    threads: usize,
    /// Discard unsupported metadata; retain supported color metadata and samples
    #[arg(long)]
    strip_metadata: bool,
    /// Replace an existing output file; all input paths remain protected
    #[arg(long)]
    force: bool,
}

impl EncodeFramesArgs {
    fn auto_options(&self) -> AutoOptions {
        AutoOptions {
            reference: self.reference.core(),
            allow_mask: !self.no_mask,
            allow_multi_rect: !self.no_multi_rect,
            tile_size: self.tile_size,
            max_rectangles: self.max_rectangles,
            max_frames: self.max_frames,
            max_total_trials: self.max_total_trials,
            max_trial_pixels: self.max_trial_pixels,
        }
    }

    fn animation(&self) -> Animation {
        Animation {
            ticks_per_second_numerator: self.tps.numerator,
            ticks_per_second_denominator: self.tps.denominator,
            loops: self.loops,
            timecodes: false,
        }
    }

    fn frame_durations(&self) -> Result<Vec<u32>> {
        if !self.durations.is_empty() {
            if self.durations.len() != self.inputs.len() {
                return Err(format!(
                    "--durations requires one value per input ({} inputs, {} durations)",
                    self.inputs.len(),
                    self.durations.len()
                )
                .into());
            }
            return Ok(self.durations.clone());
        }
        let mut durations = Vec::new();
        durations.try_reserve_exact(self.inputs.len())?;
        durations.resize(self.inputs.len(), self.duration);
        Ok(durations)
    }

    fn check_paths(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err("at least one input frame is required".into());
        }
        if self.inputs.len() > self.max_frames {
            return Err("input frame count exceeds --max-frames".into());
        }
        if self
            .inputs
            .iter()
            .filter(|p| p.as_path() == Path::new("-"))
            .count()
            > 1
        {
            return Err("stdin ('-') may appear at most once in the input sequence".into());
        }
        for path in &self.inputs {
            io::check_paths(path, &self.output, self.force)?;
        }
        Ok(())
    }
}

fn compatible(first: &Image, current: &Image) -> Result<()> {
    if first.width() != current.width() || first.height() != current.height() {
        return Err("all frames must have identical dimensions; no resizing is performed".into());
    }
    if first.layout() != current.layout()
        || std::mem::discriminant(first.pixels()) != std::mem::discriminant(current.pixels())
        || first.extra_planes().len() != current.extra_planes().len()
        || first
            .extra_planes()
            .iter()
            .zip(current.extra_planes())
            .any(|(a, b)| std::mem::discriminant(a) != std::mem::discriminant(b))
    {
        return Err("all frames must have identical sample storage and alpha layout; no conversion is performed".into());
    }
    if first.metadata() != current.metadata() {
        return Err("all frames must have identical metadata, including color and declared sample/alpha types; --strip-metadata does not remove supported color metadata or convert samples".into());
    }
    Ok(())
}

fn with_animation(image: Image, animation: Animation) -> Result<Image> {
    if image.metadata().animation.is_some() {
        return Err(
            "encode-frames requires still inputs; existing animation is not discarded".into(),
        );
    }
    let mut metadata = image.metadata().clone();
    metadata.animation = Some(animation);
    let extras = image.extra_planes().to_vec();
    Ok(image.with_metadata(metadata, extras)?)
}

fn retained_bytes(image: &Image) -> Result<usize> {
    let icc_bytes = image
        .metadata()
        .color_encoding
        .icc_profile()
        .map_or(0, <[u8]>::len);
    std::iter::once(image.pixels())
        .chain(image.extra_planes())
        .try_fold(icc_bytes, |total, samples| {
            samples
                .len()
                .checked_mul(usize::from(samples.bits() / 8))
                .and_then(|n| total.checked_add(n))
                .ok_or_else(|| "retained sequence size overflow".into())
        })
}

fn add_retained_bytes(total: usize, additional: usize) -> Result<usize> {
    total
        .checked_add(additional)
        .ok_or_else(|| "retained sequence size overflow".into())
}

fn load_images(args: &EncodeFramesArgs) -> Result<Vec<Image>> {
    let mut images = Vec::new();
    images.try_reserve_exact(args.inputs.len())?;
    let mut retained = 0;
    let mut warned_jpeg = false;
    for (index, path) in args.inputs.iter().enumerate() {
        let read_frame = || -> Result<input::Decoded> {
            let bytes = io::read_input(path)?;
            input::decode(&bytes, args.strip_metadata)
        };
        let decoded = read_frame()
            .map_err(|error| format!("frame {} '{}': {error}", index + 1, path.display()))?;
        if decoded.jpeg && !warned_jpeg {
            eprintln!(
                "warning: JPEG frames are PIXEL LOSSLESS relative to zune-jpeg's decoded samples, NOT byte-reversible; original JPEG bytes cannot be reconstructed."
            );
            warned_jpeg = true;
        }
        // Compare original input metadata before attaching our common timebase.
        if let Some(first) = images.first() {
            compatible(first, &decoded.image)
                .map_err(|error| format!("frame {} '{}': {error}", index + 1, path.display()))?;
        }
        retained = add_retained_bytes(retained, retained_bytes(&decoded.image)?)?;
        images.push(decoded.image);
    }
    images
        .into_iter()
        .map(|image| with_animation(image, args.animation()))
        .collect()
}

fn write_report(mut output: impl Write, result: &AutoEncodeResult) -> std::io::Result<()> {
    for (frame_index, frame) in result.frames.iter().enumerate() {
        writeln!(
            output,
            "frame {}: winner={} candidates={}",
            frame_index + 1,
            frame.winner_index,
            frame.candidates.len()
        )?;
        for (index, candidate) in frame.candidates.iter().enumerate() {
            writeln!(
                output,
                "  candidate={index} strategy={:?} encoded_bytes={} encoded_frames={} elapsed={:?}{}",
                candidate.strategy,
                candidate.encoded_bytes,
                candidate.encoded_frames,
                candidate.elapsed,
                if index == frame.winner_index {
                    " winner"
                } else {
                    ""
                }
            )?;
        }
    }
    writeln!(
        output,
        "sequence: inputs={} output_bytes={}",
        result.frames.len(),
        result.bytes.len()
    )
}

pub(crate) fn run(args: EncodeFramesArgs) -> Result<()> {
    let _total = rs_jxl::profile::scope("cli.total");
    args.check_paths()?;
    let durations = args.frame_durations()?;
    let mut options = EncodeOptions {
        threads: args.threads,
        ..EncodeOptions::default()
    };
    args.tools.apply(&mut options)?;
    if args.strip_metadata {
        eprintln!(
            "warning: --strip-metadata discards unsupported metadata without color conversion or EXIF orientation transforms; supported color metadata is retained, but displayed orientation may change."
        );
    }
    let images = {
        let _input = rs_jxl::profile::scope("cli.input_decode");
        load_images(&args)?
    };
    eprintln!(
        "note: encoding ordered decoded samples without sample, alpha, color or orientation conversion; automatic candidates are selected by encoded bytes, not elapsed time."
    );
    let result = {
        let _encoding = rs_jxl::profile::scope("cli.encode_frames");
        let mut encoder = Encoder::new(options.threads)?;
        encoder.encode_auto(&images, &durations, &options, &args.auto_options())?
    };
    write_report(std::io::stderr().lock(), &result)?;
    // Protect every source again before the existing transactional publisher.
    args.check_paths()?;
    io::write_output(&args.inputs[0], &args.output, args.force, &result.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;
    use rs_jxl::{PixelData, PixelLayout};
    use std::ffi::OsString;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn parse(extra: &[&str]) -> EncodeFramesArgs {
        let mut argv = vec!["rs-jxl", "encode-frames", "b.png", "a.jpg", "-o", "out.jxl"];
        argv.extend_from_slice(extra);
        let Command::EncodeFrames(args) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("expected encode-frames subcommand");
        };
        args
    }

    fn gray8() -> Image {
        Image::new(2, 1, PixelLayout::Gray, PixelData::U8(vec![0, 1])).unwrap()
    }

    #[test]
    fn defaults_preserve_order_and_enable_safe_strategies() {
        let args = parse(&[]);
        assert_eq!(
            args.inputs,
            [PathBuf::from("b.png"), PathBuf::from("a.jpg")]
        );
        assert_eq!(args.frame_durations().unwrap(), [100, 100]);
        assert_eq!(args.animation(), Animation::default());
        assert_eq!(args.threads, 1);
        let auto = args.auto_options();
        assert_eq!(auto.reference, Reference::SmallestCrop);
        assert!(auto.allow_mask && auto.allow_multi_rect);
        assert_eq!(auto.tile_size, AutoOptions::default().tile_size);
        assert_eq!(auto.max_rectangles, AutoOptions::default().max_rectangles);
        assert_eq!(auto.max_frames, usize::MAX);
        assert_eq!(auto.max_total_trials, usize::MAX);
        assert_eq!(auto.max_trial_pixels, u64::MAX);
    }

    #[test]
    fn default_frame_budget_accepts_more_than_ten_thousand_input_paths() {
        let mut args = parse(&[]);
        args.inputs = vec![PathBuf::from("frame.png"); 10_001];
        args.output = PathBuf::from("-");
        args.check_paths().unwrap();
        assert_eq!(args.frame_durations().unwrap().len(), 10_001);
        args.max_frames = 10_000;
        assert!(args.check_paths().is_err());
    }

    #[test]
    fn rectangle_budget_accepts_values_above_the_former_4096_cap() {
        for limit in [4097usize, usize::MAX] {
            assert_eq!(
                parse(&["--max-rectangles", &limit.to_string()])
                    .auto_options()
                    .max_rectangles,
                limit
            );
        }
    }

    #[test]
    fn explicit_timing_and_strategy_options_are_forwarded() {
        let args = parse(&[
            "--durations",
            "0,17",
            "--tps",
            "30000/1001",
            "--loops",
            "2",
            "--no-mask",
            "--no-multi-rect",
            "--tile-size",
            "8",
            "--max-rectangles",
            "7",
            "--max-frames",
            "100",
            "--max-total-trials",
            "300",
            "--max-trial-pixels",
            "4000",
            "--threads",
            "2",
            "--rct",
            "0",
            "--predictor",
            "0",
        ]);
        assert_eq!(args.frame_durations().unwrap(), [0, 17]);
        assert_eq!(
            args.animation(),
            Animation {
                ticks_per_second_numerator: 30000,
                ticks_per_second_denominator: 1001,
                loops: 2,
                timecodes: false,
            }
        );
        let auto = args.auto_options();
        assert!(!auto.allow_mask && !auto.allow_multi_rect);
        assert_eq!(
            (auto.tile_size, auto.max_rectangles, auto.max_frames),
            (8, 7, 100)
        );
        assert_eq!(auto.max_total_trials, 300);
        assert_eq!(auto.max_trial_pixels, 4000);
        let mut options = EncodeOptions::default();
        args.tools.apply(&mut options).unwrap();
        assert_eq!(options.modular.rct, Some(0));
        assert_eq!(args.threads, 2);
        for (name, reference) in [
            ("auto", Reference::SmallestCrop),
            ("previous", Reference::Previous),
            ("first", Reference::First),
            ("full", Reference::Full),
        ] {
            assert_eq!(
                parse(&["--reference", name]).auto_options().reference,
                reference
            );
        }
    }

    #[test]
    fn durations_are_exact_and_never_implicitly_repeated_or_clamped() {
        assert_eq!(
            parse(&["--duration", "13"]).frame_durations().unwrap(),
            [13, 13]
        );
        assert_eq!(
            parse(&["--durations", "1", "--durations", "4294967295"])
                .frame_durations()
                .unwrap(),
            [1, u32::MAX]
        );
        assert!(parse(&["--durations", "1"]).frame_durations().is_err());
        assert!(parse(&["--durations", "1,2,3"]).frame_durations().is_err());
        assert!(
            Cli::try_parse_from([
                "rs-jxl",
                "encode-frames",
                "a",
                "-o",
                "b",
                "--duration",
                "1",
                "--durations",
                "2"
            ])
            .is_err()
        );
    }

    #[test]
    fn invalid_timebases_limits_and_unsupported_search_fail() {
        for tps in [
            "0",
            "1/0",
            "1/1025",
            "1073741825",
            "1/2/3",
            "1/",
            "/2",
            "1.5",
            "x",
        ] {
            assert!(parse_tps(tps).is_err(), "{tps}");
        }
        assert!(parse_tps("1073741824/1024").is_ok());
        for extra in [
            vec!["--tile-size", "0"],
            vec!["--max-rectangles", "0"],
            vec!["--max-frames", "0"],
            vec!["--max-total-trials", "0"],
            vec!["--max-trial-pixels", "0"],
            vec!["--threads", "257"],
            vec!["--duration", "4294967296"],
            vec!["--durations", "1,,2"],
            vec!["--reference", "unknown"],
            vec!["--effort", "3"],
            vec!["--source-effort", "3"],
            vec!["--source-search"],
        ] {
            let mut argv = vec!["rs-jxl", "encode-frames", "a", "-o", "b"];
            argv.extend(extra);
            assert!(Cli::try_parse_from(argv).is_err());
        }
        assert!(Cli::try_parse_from(["rs-jxl", "encode-frames", "-o", "b"]).is_err());
        assert!(Cli::try_parse_from(["rs-jxl", "encode-frames", "a"]).is_err());
        let mut args = parse(&[]);
        args.inputs = vec!["-".into(), "-".into()];
        assert!(
            args.check_paths()
                .unwrap_err()
                .to_string()
                .contains("at most once")
        );
        args.inputs = vec!["-".into()];
        args.output = "-".into();
        assert!(args.check_paths().is_ok());
        assert!(parse(&["--max-frames", "1"]).check_paths().is_err());
    }

    #[test]
    fn compatibility_checks_geometry_storage_declared_depth_and_color() {
        let a = gray8();
        compatible(&a, &a).unwrap();
        let geometry = Image::new(1, 2, PixelLayout::Gray, PixelData::U8(vec![0, 1])).unwrap();
        assert!(compatible(&a, &geometry).is_err());
        let storage = Image::new(2, 1, PixelLayout::Gray, PixelData::U16(vec![0, 1]))
            .unwrap()
            .with_metadata(a.metadata().clone(), vec![])
            .unwrap();
        assert!(compatible(&a, &storage).is_err());
        let mut metadata = a.metadata().clone();
        metadata.bit_depth.bits_per_sample = 7;
        let declared = a.clone().with_metadata(metadata, vec![]).unwrap();
        assert!(compatible(&a, &declared).is_err());
        let mut metadata = a.metadata().clone();
        metadata.color_encoding =
            rs_jxl::color::ColorEncoding::linear_srgb(rs_jxl::color::ColorSpace::Gray);
        let color = a.clone().with_metadata(metadata, vec![]).unwrap();
        assert!(compatible(&a, &color).is_err());
        let alpha = Image::new(
            2,
            1,
            PixelLayout::GrayAlpha,
            PixelData::U8(vec![0, 0, 1, 1]),
        )
        .unwrap();
        assert!(compatible(&a, &alpha).is_err());
        let mut metadata = alpha.metadata().clone();
        metadata.extra_channels[0].bit_depth.bits_per_sample = 7;
        let declared_alpha = alpha.clone().with_metadata(metadata, vec![]).unwrap();
        assert!(compatible(&alpha, &declared_alpha).is_err());
        let mut metadata = alpha.metadata().clone();
        metadata.extra_channels[0].alpha_associated = true;
        let associated_alpha = alpha.clone().with_metadata(metadata, vec![]).unwrap();
        assert!(compatible(&alpha, &associated_alpha).is_err());
    }

    #[test]
    fn animation_attachment_retains_samples_and_other_metadata() {
        let original = Image::new(
            1,
            1,
            PixelLayout::Rgba,
            PixelData::U16(vec![17, 1000, 65535, 0]),
        )
        .unwrap();
        let animation = parse(&["--tps", "60", "--loops", "3"]).animation();
        let animated = with_animation(original.clone(), animation).unwrap();
        assert_eq!(animated.pixels(), original.pixels());
        let mut metadata = original.metadata().clone();
        metadata.animation = Some(animation);
        assert_eq!(animated.metadata(), &metadata);
        assert!(with_animation(animated, animation).is_err());
        assert_eq!(retained_bytes(&original).unwrap(), 8);
        assert_eq!(
            add_retained_bytes(512 * 1024 * 1024, 8).unwrap(),
            512 * 1024 * 1024 + 8
        );
        assert_eq!(add_retained_bytes(usize::MAX - 8, 8).unwrap(), usize::MAX);
        assert!(add_retained_bytes(usize::MAX, 1).is_err());
    }

    #[test]
    fn report_uses_actual_candidate_statistics_and_core_winner() {
        use rs_jxl::multiframe::{AutoCandidateReport, AutoFrameReport, Strategy};
        use std::time::Duration;

        let result = AutoEncodeResult {
            bytes: vec![0; 123],
            frames: vec![AutoFrameReport {
                winner_index: 1,
                candidates: vec![
                    AutoCandidateReport {
                        strategy: Strategy::Full,
                        encoded_bytes: 100,
                        encoded_frames: 1,
                        elapsed: Duration::from_secs(1),
                    },
                    AutoCandidateReport {
                        strategy: Strategy::CropPrevious,
                        encoded_bytes: 70,
                        encoded_frames: 2,
                        elapsed: Duration::from_secs(3),
                    },
                ],
            }],
        };
        let mut report = Vec::new();
        write_report(&mut report, &result).unwrap();
        let report = String::from_utf8(report).unwrap();
        assert!(report.contains("frame 1: winner=1 candidates=2"));
        assert!(report.contains("strategy=Full encoded_bytes=100 encoded_frames=1 elapsed=1s\n"));
        assert!(report.contains(
            "strategy=CropPrevious encoded_bytes=70 encoded_frames=2 elapsed=3s winner\n"
        ));
        assert!(report.contains("sequence: inputs=1 output_bytes=123"));
    }

    struct TestDir(PathBuf);
    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            for _ in 0..128 {
                let path = std::env::temp_dir().join(format!(
                    "rs-jxl-frames-test-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                if fs::create_dir(&path).is_ok() {
                    return Self(path);
                }
            }
            panic!("cannot create test directory");
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn file_args(inputs: &[PathBuf], output: &Path, extra: &[&str]) -> EncodeFramesArgs {
        let mut argv = vec![OsString::from("rs-jxl"), OsString::from("encode-frames")];
        argv.extend(inputs.iter().map(|p| p.as_os_str().to_owned()));
        argv.push("--output".into());
        argv.push(output.as_os_str().to_owned());
        argv.extend(extra.iter().map(|arg| OsString::from(*arg)));
        let Command::EncodeFrames(args) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("expected encode-frames");
        };
        args
    }

    fn write_png(path: &Path, depth: png::BitDepth, data: &[u8]) {
        let mut encoder = png::Encoder::new(fs::File::create(path).unwrap(), 2, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(depth);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(data).unwrap();
        writer.finish().unwrap();
    }

    #[test]
    fn output_cannot_alias_any_input_even_with_force() {
        let dir = TestDir::new();
        let inputs = vec![dir.0.join("first.png"), dir.0.join("second.png")];
        for path in &inputs {
            fs::write(path, b"unchanged").unwrap();
        }
        for output in [&inputs[1], &dir.0.join(".").join("second.png")] {
            assert!(
                file_args(&inputs, output, &["--force"])
                    .check_paths()
                    .is_err()
            );
        }
        assert_eq!(fs::read(&inputs[1]).unwrap(), b"unchanged");
    }

    #[test]
    fn failures_never_create_or_replace_output() {
        let dir = TestDir::new();
        let input = dir.0.join("frame.png");
        write_png(&input, png::BitDepth::Eight, &[1, 2, 3, 0, 4, 5, 6, 255]);
        let malformed = dir.0.join("bad.png");
        fs::write(&malformed, b"not an image").unwrap();
        let output = dir.0.join("out.jxl");
        let inputs = vec![input, malformed];
        assert!(run(file_args(&inputs, &output, &[])).is_err());
        assert!(!output.exists());
        fs::write(&output, b"existing output").unwrap();
        assert!(run(file_args(&inputs, &output, &["--force"])).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"existing output");
        assert!(
            run(file_args(
                &inputs,
                &output,
                &["--force", "--durations", "1"]
            ))
            .is_err()
        );
        assert_eq!(fs::read(&output).unwrap(), b"existing output");
    }

    #[test]
    fn metadata_stripping_does_not_erase_color_mismatches() {
        let dir = TestDir::new();
        let inputs = vec![dir.0.join("srgb.png"), dir.0.join("linear.png")];
        let data = [1, 2, 3, 0, 4, 5, 6, 255];
        write_png(&inputs[0], png::BitDepth::Eight, &data);
        let mut encoder = png::Encoder::new(fs::File::create(&inputs[1]).unwrap(), 2, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_gamma(png::ScaledFloat::new(1.0));
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&data).unwrap();
        writer.finish().unwrap();
        let output = dir.0.join("out.jxl");
        for extra in [&[][..], &["--strip-metadata"][..]] {
            assert!(
                load_images(&file_args(&inputs, &output, extra))
                    .unwrap_err()
                    .to_string()
                    .contains("identical metadata")
            );
        }
    }

    #[test]
    fn unsupported_metadata_requires_explicit_stripping_without_sample_changes() {
        let dir = TestDir::new();
        let inputs = vec![dir.0.join("plain.png"), dir.0.join("comment.png")];
        let data = [1, 2, 3, 0, 4, 5, 6, 255];
        write_png(&inputs[0], png::BitDepth::Eight, &data);
        let mut encoder = png::Encoder::new(fs::File::create(&inputs[1]).unwrap(), 2, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&data).unwrap();
        writer
            .write_chunk(
                png::chunk::ChunkType(*b"tEXt"),
                b"Comment\0discard explicitly",
            )
            .unwrap();
        writer.finish().unwrap();
        let output = dir.0.join("out.jxl");
        assert!(
            load_images(&file_args(&inputs, &output, &[]))
                .unwrap_err()
                .to_string()
                .contains("--strip-metadata")
        );
        let images = load_images(&file_args(&inputs, &output, &["--strip-metadata"])).unwrap();
        assert_eq!(images[0].metadata(), images[1].metadata());
        for image in images {
            assert_eq!(image.pixels(), &PixelData::U8(data.to_vec()));
        }
    }

    #[test]
    fn png_sequence_round_trip_preserves_hidden_rgb_depth_and_durations() {
        for depth in [png::BitDepth::Eight, png::BitDepth::Sixteen] {
            let dir = TestDir::new();
            let inputs = vec![
                dir.0.join("b.png"),
                dir.0.join("a.png"),
                dir.0.join("again.png"),
            ];
            let mut expected = Vec::new();
            for (index, path) in inputs.iter().enumerate() {
                let samples = [17 + u16::from(index == 1), 23, 31, 0, 42, 53, 64, 255];
                let (data, pixels) = if depth == png::BitDepth::Sixteen {
                    let values: Vec<u16> = samples.iter().map(|v| v * 257).collect();
                    (
                        values
                            .iter()
                            .flat_map(|v| v.to_be_bytes())
                            .collect::<Vec<_>>(),
                        PixelData::U16(values),
                    )
                } else {
                    let values: Vec<u8> = samples.iter().map(|v| *v as u8).collect();
                    (values.clone(), PixelData::U8(values))
                };
                write_png(path, depth, &data);
                expected.push(pixels);
            }
            let output = dir.0.join("out.jxl");
            run(file_args(
                &inputs,
                &output,
                &[
                    "--durations",
                    "11,13,17",
                    "--tps",
                    "30000/1001",
                    "--loops",
                    "2",
                    "--threads",
                    if depth == png::BitDepth::Eight {
                        "1"
                    } else {
                        "2"
                    },
                ],
            ))
            .unwrap();
            let decoded =
                rs_jxl::decode::decode(&fs::read(&output).unwrap(), &Default::default()).unwrap();
            assert!(decoded.complete);
            assert_eq!(decoded.frames.len(), 3);
            let animation = decoded.info.animation.as_ref().unwrap();
            assert_eq!(
                (animation.tps_numerator, animation.tps_denominator),
                (30000, 1001)
            );
            assert_eq!(animation.num_loops, 2);
            for ((frame, pixels), duration) in
                decoded.frames.iter().zip(&expected).zip([11, 13, 17])
            {
                assert_eq!(
                    (frame.width, frame.height, frame.layout),
                    (2, 1, PixelLayout::Rgba)
                );
                assert_eq!(&frame.pixels, pixels);
                assert_eq!(frame.duration_ticks, duration);
            }
            assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 4);
        }
    }
}
