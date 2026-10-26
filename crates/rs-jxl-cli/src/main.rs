mod decode;
mod input;
mod io;
mod multiframe;
mod options;
mod transcode;

#[cfg(test)]
mod acceptance;

#[cfg(test)]
#[path = "ma_budget_parser_tests.rs"]
mod ma_budget_parser_tests;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use rs_jxl::scheduler::{
    CandidateParams, DriverOptions, NativeBackendOptions, native::encode_native_candidate,
    run_native_pixel_search, run_native_tectonic,
};
use rs_jxl::{EncodeOptions, encode};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const DEFAULT_SOURCE_SAMPLE_LIMIT: usize = 1_048_576;

#[derive(Debug, Parser)]
#[command(
    name = "rs-jxl",
    version,
    about = "Encode, decode, or transcode JPEG XL images",
    long_about = "Encode, decode, or transcode images using Rust JPEG XL codecs.\n\
decode: JPEG XL pixels to PNG/APNG, NPY, PPM, PGM or PFM, or original JPEG reconstruction.\n\
encode: PNG or JPEG decoded pixels to lossless JPEG XL.\n\
encode-frames: ordered PNG/JPEG frames to pixel-lossless animated JPEG XL.\n\
PNG: gray, gray-alpha, RGB and RGBA at 8/16 bits; palette and low-bit gray \
are expanded without losing decoded samples, including RGB under zero alpha.\n\
JPEG: 8-bit grayscale/RGB pixel input via zune-jpeg. Pixel losslessness is \
relative to this decoder's samples, NOT the original JPEG bytes. \
Use transcode-jpeg for byte-reversible JPEG coefficient transcoding instead.\n\
transcode-jpeg: supported 8-bit Huffman DCT JPEG inputs only, with JPEG \
reconstruction data and no pixel fallback. Full optimized source-tool parity \
is not claimed."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Decode JPEG XL pixels or reconstruct original JPEG bytes
    Decode(decode::DecodeArgs),
    /// Encode one still PNG or JPEG to a JXL file
    Encode(EncodeArgs),
    /// Encode ordered PNG/JPEG frames with automatic lossless strategies
    EncodeFrames(multiframe::EncodeFramesArgs),
    /// Transcode JPEG coefficients with original-JPEG reconstruction data
    TranscodeJpeg(transcode::TranscodeArgs),
}

#[cfg(test)]
fn expect_encode(command: Command) -> EncodeArgs {
    let Command::Encode(args) = command else {
        panic!("expected encode subcommand");
    };
    args
}

#[derive(Debug, Args)]
#[command(
    after_long_help = "JPEG input is pixel-lossless relative to zune-jpeg's \
decoded samples only; original JPEG bytes cannot be reconstructed.\n\
PNG sRGB/gAMA/cHRM/iCCP and JPEG ICC color metadata are preserved. \
Untagged input defaults to sRGB gray/RGB. No color conversion or EXIF orientation \
transform is performed; PNG tRNS is decoded pixel data.\n\
Unsupported metadata such as EXIF/XMP/comments requires --strip-metadata. \
Supported color metadata is retained even with --strip-metadata. Stripping \
unsupported metadata can change displayed orientation without changing samples.\n\
APNG, CMYK/YCCK JPEG and non-8-bit JPEG are unsupported. Limits: 512 MiB input, \
16,384 pixels per dimension, 64,000,000 pixels and 512 MiB decoded samples. \
The encoder applies its own resource limits as well.\n\
Source mode is experimental and entirely Rust: --source-effort 1..11 encodes \
one native candidate; --source-search additionally runs the existing native \
pixel search driver (effort 11 uses its separate full-frame tournament). \
At effort 9/10, --source-scout selects an experimental coverage-guided frontier \
with fewer full-frame candidates; it may change compression size. \
--source-pilot selects a separate experimental content-aware pilot-first policy. \
Optional routes are exposed through the Rust API; source-identical bytes or \
performance are not claimed. Unsupported policies and resource exhaustion \
are errors unless --ma-budget-mode bounded explicitly stops optional MA splits.\n\
Effort 11 without --source-search selects the explicit TectonicPlate (-1) \
backend policy, not the frame-level tournament.\n\
Source mode cannot be combined with explicitly specified tool options \
(even their default values). --threads, --strip-metadata and --force remain \
available, as do --ma-budget-mode strict|bounded, --ma-max-leaves N and \
experimental --ma-prune (exact partition reuse and small-node dominance). \
Omitting the MA budget mode keeps source strict and manual bounded. \
Required MA partitions must fit even in bounded mode. \
--source-sample-limit bounds MA training samples independently of \
the source sampling policy; exceeding it fails rather than reducing sampling.\n\
Use '-' for stdin/stdout. Diagnostics always go to stderr. \
Existing output is not replaced unless --force is supplied."
)]
struct EncodeArgs {
    #[command(flatten)]
    tools: options::Tools,
    #[command(flatten)]
    ma_budget: options::MaBudgetOptions,
    #[command(flatten)]
    ma_pruning: options::MaPruningOptions,
    /// Experimental source policy: one native candidate at effort 1..=11
    #[arg(long, value_name = "N", conflicts_with = "manual_tools", value_parser = clap::value_parser!(u8).range(1..=11))]
    source_effort: Option<u8>,
    /// Run native pixel search instead of just one source candidate
    #[arg(long, requires = "source_effort")]
    source_search: bool,
    /// Experimental coverage-guided search at source effort 9/10
    #[arg(long, requires = "source_search")]
    source_scout: bool,
    /// Experimental content-aware pilot-first search at source effort 9/10
    #[arg(long, requires = "source_search", conflicts_with = "source_scout")]
    source_pilot: bool,
    /// Experimental original-group racing and continuation within source pilot
    #[arg(long, requires = "source_pilot")]
    source_progressive_groups: bool,
    /// Hard MA training sample limit, not a source sampling target
    #[arg(long, value_name = "N", default_value_t = DEFAULT_SOURCE_SAMPLE_LIMIT, requires = "source_effort", value_parser = parse_sample_limit)]
    source_sample_limit: usize,
    /// PNG or JPEG input path, or '-' for stdin (detected by file signature)
    input: PathBuf,
    /// JXL output path, or '-' for binary stdout
    output: PathBuf,
    /// Encoder worker count (1..=256)
    #[arg(long, value_name = "N", default_value = "1", value_parser = parse_threads)]
    threads: usize,
    /// Discard unsupported metadata; retain supported color metadata and samples
    #[arg(long)]
    strip_metadata: bool,
    /// Replace an existing output file; input and output must still differ
    #[arg(long)]
    force: bool,
}

fn parse_threads(value: &str) -> std::result::Result<usize, String> {
    let count = value
        .parse::<usize>()
        .map_err(|_| "threads must be an integer in 1..=256")?;
    if !(1..=256).contains(&count) {
        return Err("threads must be in 1..=256".into());
    }
    Ok(count)
}

fn parse_sample_limit(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&limit| limit > 0)
        .ok_or_else(|| "source sample limit must be a positive integer fitting usize".into())
}

// Both file encoding and the real-image timing harness use this dispatch.
fn encode_image(image: &rs_jxl::Image, args: &EncodeArgs) -> Result<Vec<u8>> {
    let mut options = EncodeOptions {
        threads: args.threads,
        ..EncodeOptions::default()
    };
    let Some(effort) = args.source_effort else {
        args.tools.apply(&mut options)?;
        args.ma_budget.apply(&mut options.modular);
        args.ma_pruning.apply(&mut options.modular);
        return Ok(encode(image, &options)?);
    };
    args.ma_budget.apply(&mut options.modular);
    args.ma_pruning.apply(&mut options.modular);
    options.modular.sample_limit = args.source_sample_limit;
    let params = CandidateParams::default().backend(effort)?;
    if args.source_scout && !(9..=10).contains(&effort) {
        return Err("--source-scout requires --source-effort 9 or 10".into());
    }
    if args.source_pilot && !(9..=10).contains(&effort) {
        return Err("--source-pilot requires --source-effort 9 or 10".into());
    }
    let settings = NativeBackendOptions::default();
    eprintln!(
        "note: experimental pure-Rust source mode: effort={effort} search={} \
         sample_limit={}; no byte-identity or performance-parity claim",
        args.source_search, args.source_sample_limit,
    );
    if effort == 11 && !args.source_search {
        eprintln!(
            "note: explicit TectonicPlate (-1) backend; no C++ frame-level effort11 tournament"
        );
    }
    if args.source_search && effort == 11 {
        let result = run_native_tectonic(
            image,
            &params,
            Default::default(),
            Default::default(),
            &options,
            settings,
        )?;
        eprintln!(
            "source effort11 tournament: branch={:?} full_encodes={} winner={} winner_effort={}",
            result.output.report.branch,
            result.output.report.full_encodes,
            result.output.report.winner_index,
            10 - result.output.params.speed_tier
        );
        Ok(result.output.bytes)
    } else if args.source_search {
        let driver = if args.source_scout {
            DriverOptions::scout()
        } else {
            DriverOptions::default()
        };
        let result = if args.source_pilot {
            rs_jxl::scheduler::pilot::encode_with_options(
                image,
                &params,
                &options,
                rs_jxl::scheduler::pilot::PilotOptions {
                    group_race: args.source_progressive_groups.then(Default::default),
                    ..Default::default()
                },
            )?
        } else {
            run_native_pixel_search(image, &params, &driver, &options, settings)?
        };
        eprintln!(
            "source search: full_encodes={} probe_encodes={} winner={} winner_effort={}",
            result.output.report.full_encodes,
            result.output.report.probe_encodes,
            result.output.report.winner_index,
            10 - result.output.params.speed_tier,
        );
        if args.source_scout || args.source_pilot {
            eprintln!("source scout: winner_params={:?}", result.output.params);
        }
        Ok(result.output.bytes)
    } else {
        Ok(encode_native_candidate(&params, image, &options, settings)?.bytes)
    }
}

fn run(args: EncodeArgs) -> Result<()> {
    let _total = rs_jxl::profile::scope("cli.total");
    io::check_paths(&args.input, &args.output, args.force)?;
    let bytes = io::read_input(&args.input)?;
    let decoded = {
        let _input = rs_jxl::profile::scope("cli.input_decode");
        input::decode(&bytes, args.strip_metadata)?
    };
    if decoded.jpeg {
        eprintln!(
            "warning: JPEG -> JXL is PIXEL LOSSLESS relative to zune-jpeg's decoded \
             samples, NOT a byte-reversible JPEG transcode. Original JPEG bytes \
             cannot be reconstructed."
        );
    }
    if args.strip_metadata {
        eprintln!(
            "warning: --strip-metadata discards unsupported metadata without color \
             conversion or EXIF orientation transforms; supported color metadata \
             is retained, but displayed orientation may change."
        );
    }
    eprintln!(
        "note: encoding decoded samples without color or orientation conversion; \
         supported color metadata is retained, including with --strip-metadata."
    );
    let encoded = {
        let _encoding = rs_jxl::profile::scope("cli.encode");
        encode_image(&decoded.image, &args)?
    };
    io::write_output(&args.input, &args.output, args.force, &encoded)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Decode(args) => decode::run(args),
        Command::Encode(args) => run(args),
        Command::EncodeFrames(args) => multiframe::run(args),
        Command::TranscodeJpeg(args) => transcode::run(args),
    };
    if let Some(report) = rs_jxl::profile::report() {
        eprint!("{report}");
    }
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_shape_and_defaults() {
        Cli::command().debug_assert();
        let cli = Cli::try_parse_from(["rs-jxl", "encode", "-", "-"]).unwrap();
        let args = expect_encode(cli.command);
        assert_eq!(args.threads, 1);
        assert!(!args.force);
        assert!(!args.strip_metadata);
        assert_eq!(args.source_effort, None);
        assert!(!args.source_search);
        assert!(!args.source_scout);
        assert_eq!(args.source_sample_limit, DEFAULT_SOURCE_SAMPLE_LIMIT);
    }

    #[test]
    fn source_arguments_and_resources() {
        for effort in 1..=11 {
            let cli = Cli::try_parse_from([
                "rs-jxl",
                "encode",
                "a",
                "b",
                "--source-effort",
                &effort.to_string(),
            ])
            .unwrap();
            let args = expect_encode(cli.command);
            assert_eq!(args.source_effort, Some(effort));
            assert!(!args.source_search);
        }
        let cli = Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            "3",
            "--source-search",
            "--source-sample-limit",
            "1048576",
            "--threads",
            "2",
            "--strip-metadata",
            "--force",
        ])
        .unwrap();
        let args = expect_encode(cli.command);
        assert!(args.source_search);
        assert_eq!(args.source_sample_limit, DEFAULT_SOURCE_SAMPLE_LIMIT);
        assert_eq!(args.threads, 2);
        for effort in ["0", "12", "-1", "abc", "256"] {
            assert!(
                Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--source-effort", effort,])
                    .is_err()
            );
        }
        for limit in ["0", "-1", "abc", "18446744073709551616"] {
            assert!(
                Cli::try_parse_from([
                    "rs-jxl",
                    "encode",
                    "a",
                    "b",
                    "--source-effort",
                    "3",
                    "--source-sample-limit",
                    limit,
                ])
                .is_err()
            );
        }
        for extra in [
            vec!["--source-search"],
            vec!["--source-scout"],
            vec!["--source-pilot"],
            vec!["--source-sample-limit", "1048576"],
        ] {
            let mut argv = vec!["rs-jxl", "encode", "a", "b"];
            argv.extend(extra);
            assert!(Cli::try_parse_from(argv).is_err());
        }
    }

    #[test]
    fn source_scout_is_explicit_and_rejects_unsupported_efforts() {
        use rs_jxl::{Image, PixelData, PixelLayout};
        let image = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![0])).unwrap();
        for effort in ["8", "11"] {
            let cli = Cli::try_parse_from([
                "rs-jxl",
                "encode",
                "a",
                "b",
                "--source-effort",
                effort,
                "--source-search",
                "--source-scout",
            ])
            .unwrap();
            assert!(
                encode_image(&image, &expect_encode(cli.command))
                    .unwrap_err()
                    .to_string()
                    .contains("9 or 10")
            );
        }
        let cli = Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            "9",
            "--source-search",
            "--source-scout",
        ])
        .unwrap();
        assert!(expect_encode(cli.command).source_scout);
    }

    #[test]
    fn content_pilot_is_explicit_and_excludes_other_search_presets() {
        let image = rs_jxl::Image::new(
            1,
            1,
            rs_jxl::PixelLayout::Gray,
            rs_jxl::PixelData::U8(vec![0]),
        )
        .unwrap();
        for effort in ["8", "11"] {
            let cli = Cli::try_parse_from([
                "rs-jxl",
                "encode",
                "a",
                "b",
                "--source-effort",
                effort,
                "--source-search",
                "--source-pilot",
            ])
            .unwrap();
            assert!(
                encode_image(&image, &expect_encode(cli.command))
                    .unwrap_err()
                    .to_string()
                    .contains("9 or 10")
            );
        }
        assert!(
            Cli::try_parse_from([
                "rs-jxl",
                "encode",
                "a",
                "b",
                "--source-effort",
                "9",
                "--source-search",
                "--source-pilot",
                "--source-scout"
            ])
            .is_err()
        );
        let cli = Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            "9",
            "--source-search",
            "--source-pilot",
        ])
        .unwrap();
        assert!(expect_encode(cli.command).source_pilot);
        let cli = Cli::try_parse_from([
            "rs-jxl",
            "encode",
            "a",
            "b",
            "--source-effort",
            "9",
            "--source-search",
            "--source-pilot",
            "--source-progressive-groups",
        ])
        .unwrap();
        assert!(expect_encode(cli.command).source_progressive_groups);
        assert!(
            Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--source-progressive-groups",])
                .is_err()
        );
    }

    #[test]
    fn explicit_effort_eleven_maps_to_tectonic_not_effort_ten() {
        let cli =
            Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--source-effort", "11"]).unwrap();
        let args = expect_encode(cli.command);
        let candidate = CandidateParams::default()
            .backend(args.source_effort.unwrap())
            .unwrap();
        assert_eq!(candidate.speed_tier, -1);
        assert!(!args.source_search);
        assert_ne!(candidate, CandidateParams::default().backend(10).unwrap());
        assert_eq!(candidate.training_policy(3, 22).node_threshold, 61);
    }

    #[test]
    fn source_rejects_every_explicit_manual_tool_including_defaults() {
        let cases: &[&[&str]] = &[
            &["--rct", "6"],
            &["--predictor", "14"],
            &["--weighted-preset", "0"],
            &["--rct-candidates", "1"],
            &["--weighted-candidates", "1"],
            &["--palette-colors", "256"],
            &["--palette-colors", "256", "--palette-predictor", "0"],
            &["--squeeze", "horizontal"],
            &["--entropy", "auto"],
            &["--lz77", "lazy"],
            &["--tree-samples", "1024"],
            &["--tree-leaves", "16"],
            &["--tree-split-threshold", "48"],
            &["--previous-channels", "0"],
            &["--group-size-shift", "1"],
            &["--shared-tree"],
            &["--patches"],
            &["--global-palette", "--palette-colors", "256"],
            &["--responsive"],
            &["--progressive"],
            &["--source-patches"],
        ];
        for &manual in cases {
            // The same tool arguments remain legal without source mode.
            let mut argv = vec!["rs-jxl", "encode", "a", "b"];
            argv.extend_from_slice(manual);
            Cli::try_parse_from(&argv).unwrap();
            for source_first in [false, true] {
                for search in [false, true] {
                    let mut argv = vec!["rs-jxl", "encode", "a", "b"];
                    let mut source = vec!["--source-effort", "3"];
                    if search {
                        source.push("--source-search");
                    }
                    if source_first {
                        argv.extend_from_slice(&source);
                        argv.extend_from_slice(manual);
                    } else {
                        argv.extend_from_slice(manual);
                        argv.extend_from_slice(&source);
                    }
                    let error = Cli::try_parse_from(argv).unwrap_err();
                    assert_eq!(
                        error.kind(),
                        clap::error::ErrorKind::ArgumentConflict,
                        "{manual:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn only_supported_options_are_accepted() {
        for option in ["--effort", "--quality", "--lossy"] {
            assert!(Cli::try_parse_from(["rs-jxl", "encode", "a", "b", option, "1"]).is_err());
        }
        for count in ["0", "-1", "257", "abc", "18446744073709551616"] {
            assert!(
                Cli::try_parse_from(["rs-jxl", "encode", "a", "b", "--threads", count]).is_err()
            );
        }
        assert!(Cli::try_parse_from(["rs-jxl", "decode", "a", "b"]).is_ok());
    }
}
