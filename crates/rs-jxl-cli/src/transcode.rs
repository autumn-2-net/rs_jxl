//! Separate JPEG coefficient CLI route; never decodes pixels or strips input.

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use rs_jxl::codec;
use rs_jxl::jpeg::{
    TranscodeLimits, TranscodeOptions, cfl::CflMode, coefficient_order::OrderPolicy,
    search::SearchLimits,
};

use crate::validators::{parse_sample_limit, parse_threads};
use crate::{Result, io};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Order {
    #[default]
    Source,
    Default,
    Adaptive,
}

#[derive(Debug, Args)]
#[command(
    after_long_help = "Accepts supported 8-bit Huffman DCT JPEG inputs only. \
Coefficients and JPEG reconstruction data are encoded without pixel decoding \
or re-quantization. Unsupported inputs, malformed metadata and exhausted \
budgets fail; no pixel fallback or metadata-stripping mode is provided.\n\
Effort selects the coefficient backend policy unless --source-search is set. \
Source search executes the JPEG-specific effort prefix and retains the smallest \
complete result. It requires --decoding-speed=0.\n\
Brotli quality defaults to min(effort + 1, 11). An explicit --sample-limit is a \
hard selected-sample budget across shared auxiliary tree domains, not a sampling \
target; omission adds no sample-count cap. No extra CLI input/output byte cap is \
imposed. Format bounds, size-overflow checks and allocation failures still apply.\n\
Use '-' for stdin/stdout. Diagnostics go to stderr. Encoding completes before \
output is written. Existing output is not replaced unless --force is supplied."
)]
pub(crate) struct TranscodeArgs {
    /// JPEG input path, or '-' for stdin (contents, not extension, are checked)
    input: PathBuf,
    /// JXL container output path, or '-' for binary stdout
    output: PathBuf,
    /// Source coefficient backend effort (1..=11)
    #[arg(long, value_name = "N", default_value = "3", value_parser = clap::value_parser!(u8).range(1..=11))]
    effort: u8,
    /// Run the optimized source JPEG search prefix (up to six complete trials)
    #[arg(long)]
    source_search: bool,
    /// Encoder worker count (1..=256)
    #[arg(long, value_name = "N", default_value = "1", value_parser = parse_threads)]
    threads: usize,
    /// Source decoding-speed policy (0..=4)
    #[arg(long, value_name = "N", default_value = "0", value_parser = clap::value_parser!(u8).range(0..=4))]
    decoding_speed: u8,
    /// JPEG reconstruction Brotli quality (0..=11; otherwise follows effort)
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u8).range(0..=11))]
    brotli_quality: Option<u8>,
    /// Disable coefficient chroma-from-luma prediction
    #[arg(long)]
    no_cfl: bool,
    /// Use the fixed default coefficient order (same as --order default)
    #[arg(long, conflicts_with = "order")]
    default_order: bool,
    /// Coefficient order policy
    #[arg(
        long,
        value_enum,
        default_value = "source",
        conflicts_with = "default_order"
    )]
    order: Order,
    /// Hard shared MA sample limit (omitted: unlimited); never reduces sampling
    #[arg(long, value_name = "N", default_value_t = TranscodeLimits::default().sample_limit, hide_default_value = true, value_parser = parse_sample_limit)]
    sample_limit: usize,
    /// Replace an existing output file; input and output must still differ
    #[arg(long)]
    force: bool,
}

impl TranscodeArgs {
    fn search_limits(&self) -> Option<SearchLimits> {
        self.source_search.then(SearchLimits::default)
    }

    fn options(&self) -> TranscodeOptions {
        TranscodeOptions {
            effort: self.effort,
            threads: self.threads,
            cooperative_ma: None,
            decoding_speed: self.decoding_speed,
            reconstruction_quality: self.brotli_quality,
            cfl: if self.no_cfl {
                CflMode::Disabled
            } else {
                CflMode::Enabled
            },
            order: if self.default_order {
                OrderPolicy::Default
            } else {
                match self.order {
                    Order::Source => OrderPolicy::Source,
                    Order::Default => OrderPolicy::Default,
                    Order::Adaptive => OrderPolicy::Adaptive,
                }
            },
            limits: TranscodeLimits {
                sample_limit: self.sample_limit,
                ..TranscodeLimits::default()
            },
        }
    }
}

// Delay both stdout writes and atomic file publication until encoding succeeds.
fn write_transcoded(
    input: &[u8],
    options: &TranscodeOptions,
    publish: impl FnOnce(&[u8]) -> Result<()>,
) -> Result<()> {
    let encoded = {
        let _encoding = rs_jxl::profile::scope("cli.jpeg_transcode");
        codec::transcode_jpeg(input, options, None)?
    };
    publish(&encoded.bytes)
}

pub(crate) fn run(args: TranscodeArgs) -> Result<()> {
    let _total = rs_jxl::profile::scope("cli.total");
    io::check_paths(&args.input, &args.output, args.force)?;
    let bytes = io::read_input(&args.input)?;
    if let Some(limits) = args.search_limits() {
        let _encoding = rs_jxl::profile::scope("cli.jpeg_search");
        let result = codec::transcode_jpeg(&bytes, &args.options(), Some(limits))?;
        return io::write_output(&args.input, &args.output, args.force, &result.bytes);
    }
    write_transcoded(&bytes, &args.options(), |encoded| {
        io::write_output(&args.input, &args.output, args.force, encoded)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;
    use rs_jxl::jpeg::transcode;
    use std::{
        ffi::OsString,
        fs,
        io::Write,
        path::Path,
        sync::atomic::{AtomicU64, Ordering},
    };

    fn parse(extra: &[&str]) -> TranscodeArgs {
        let mut argv = vec!["rs-jxl", "transcode-jpeg", "-", "-"];
        argv.extend_from_slice(extra);
        let Command::TranscodeJpeg(args) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("expected transcode-jpeg subcommand");
        };
        args
    }

    fn segment(bytes: &mut Vec<u8>, marker: u8, data: &[u8]) {
        bytes.extend_from_slice(&[0xff, marker]);
        bytes.extend_from_slice(&u16::try_from(data.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(data);
    }

    // One gray DC-zero/EOB block, as in the existing pixel-input unit fixtures.
    // Varied quantization entries exercise the quant-table MA sample budget.
    fn jpeg() -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8];
        segment(&mut bytes, 0xfe, b"retained comment");
        let mut quant = vec![0];
        quant.extend((1..=64).map(|i| (i % 31 + 1) as u8));
        segment(&mut bytes, 0xdb, &quant);
        segment(&mut bytes, 0xc0, &[8, 0, 2, 0, 3, 1, 1, 0x11, 0]);
        let mut tables = Vec::new();
        for selector in [0, 0x10] {
            tables.extend_from_slice(&[selector, 1]);
            tables.extend_from_slice(&[0; 15]);
            tables.push(0);
        }
        segment(&mut bytes, 0xc4, &tables);
        segment(&mut bytes, 0xda, &[1, 1, 0, 0, 63, 0]);
        bytes.extend_from_slice(&[0x3f, 0xff, 0xd9]);
        bytes.extend_from_slice(b"retained trailing bytes");
        bytes
    }

    #[test]
    fn defaults_follow_public_transcode_options_and_limits() {
        let args = parse(&[]);
        assert_eq!(args.input, Path::new("-"));
        assert_eq!(args.output, Path::new("-"));
        assert!(!args.force);
        assert!(!args.source_search);
        assert!(parse(&["--source-search"]).source_search);
        let options = args.options();
        let defaults = TranscodeOptions::default();
        assert_eq!(options.effort, defaults.effort);
        assert_eq!(options.threads, defaults.threads);
        assert_eq!(options.decoding_speed, defaults.decoding_speed);
        assert_eq!(options.reconstruction_quality, None);
        assert_eq!(options.cfl, defaults.cfl);
        assert_eq!(options.order, defaults.order);
        assert_eq!(options.limits.sample_limit, defaults.limits.sample_limit);
        assert_eq!(options.limits.sample_limit, usize::MAX);
        assert_eq!(
            options.limits.max_output_bytes,
            defaults.limits.max_output_bytes
        );
        assert_eq!(
            options.limits.max_coefficients,
            defaults.limits.max_coefficients
        );
        assert_eq!(options.limits.max_tokens, defaults.limits.max_tokens);
        assert_eq!(
            options.limits.input.structural.max_input_bytes,
            defaults.limits.input.structural.max_input_bytes
        );
    }

    #[test]
    fn search_limits_are_opt_in_without_rewriting_backend_options() {
        for effort in 1..=11 {
            let effort = effort.to_string();
            let direct = parse(&["--effort", &effort]);
            assert!(direct.search_limits().is_none());
            let search = parse(&["--effort", &effort, "--source-search"]);
            assert_eq!(search.options().effort, direct.options().effort);
            assert_eq!(search.options().reconstruction_quality, None);
            let limits = search.search_limits().unwrap();
            let defaults = SearchLimits::default();
            assert_eq!(limits.max_full_encodes, defaults.max_full_encodes);
            assert_eq!(limits.max_full_pixels, defaults.max_full_pixels);
            assert_eq!(limits.max_output_bytes, defaults.max_output_bytes);
        }
        let args = parse(&[
            "--source-search",
            "--decoding-speed",
            "1",
            "--threads",
            "2",
            "--sample-limit",
            "17",
            "--brotli-quality",
            "0",
            "--no-cfl",
            "--order",
            "adaptive",
        ]);
        assert!(args.search_limits().is_some());
        let options = args.options();
        // Preserve the unsupported speed for the codec's existing error gate.
        assert_eq!(options.decoding_speed, 1);
        assert_eq!(options.threads, 2);
        assert_eq!(options.limits.sample_limit, 17);
        assert_eq!(options.reconstruction_quality, Some(0));
        assert_eq!(options.cfl, CflMode::Disabled);
        assert_eq!(options.order, OrderPolicy::Adaptive);
    }

    #[test]
    fn explicit_arguments_map_without_changing_unrequested_settings() {
        for effort in 1..=11 {
            let options = parse(&["--effort", &effort.to_string()]).options();
            assert_eq!(options.effort, effort);
            assert_eq!(options.reconstruction_quality, None);
        }
        for decoding_speed in 0..=4 {
            assert_eq!(
                parse(&["--decoding-speed", &decoding_speed.to_string()])
                    .options()
                    .decoding_speed,
                decoding_speed
            );
        }
        for quality in 0..=11 {
            assert_eq!(
                parse(&["--brotli-quality", &quality.to_string()])
                    .options()
                    .reconstruction_quality,
                Some(quality)
            );
        }
        let args = parse(&[
            "--effort",
            "11",
            "--threads",
            "256",
            "--decoding-speed",
            "4",
            "--brotli-quality",
            "0",
            "--no-cfl",
            "--default-order",
            "--sample-limit",
            "1",
            "--force",
        ]);
        let options = args.options();
        assert_eq!(options.threads, 256);
        assert_eq!(options.cfl, CflMode::Disabled);
        assert_eq!(options.order, OrderPolicy::Default);
        assert_eq!(options.limits.sample_limit, 1);
        assert!(args.force);
        for (text, expected) in [
            ("source", OrderPolicy::Source),
            ("default", OrderPolicy::Default),
            ("adaptive", OrderPolicy::Adaptive),
        ] {
            assert_eq!(parse(&["--order", text]).options().order, expected);
        }
        assert_eq!(
            parse(&["--sample-limit", &usize::MAX.to_string()]).sample_limit,
            usize::MAX
        );
    }

    #[test]
    fn rejects_invalid_ranges_and_pixel_encode_flags() {
        let cases: &[(&str, &[&str])] = &[
            ("--effort", &["0", "12", "-1", "256", "abc"]),
            ("--threads", &["0", "257", "-1", "abc"]),
            ("--decoding-speed", &["5", "-1", "256", "abc"]),
            ("--brotli-quality", &["12", "-1", "256", "abc"]),
            (
                "--sample-limit",
                &["0", "-1", "abc", "18446744073709551616"],
            ),
            ("--order", &["unknown", "0"]),
        ];
        for &(flag, values) in cases {
            for value in values {
                assert!(
                    Cli::try_parse_from(["rs-jxl", "transcode-jpeg", "a", "b", flag, value,])
                        .is_err(),
                    "{flag}={value}"
                );
            }
        }
        for flag in [
            "--strip-metadata",
            "--strip",
            "--quality",
            "--lossy",
            "--rct",
            "--palette-colors",
            "--source-effort",
            "--source-sample-limit",
        ] {
            let error =
                Cli::try_parse_from(["rs-jxl", "transcode-jpeg", "a", "b", flag]).unwrap_err();
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::UnknownArgument,
                "{flag}"
            );
        }
        for order in ["source", "default", "adaptive"] {
            for flags in [
                ["--default-order", "--order", order],
                ["--order", order, "--default-order"],
            ] {
                let mut argv = vec!["rs-jxl", "transcode-jpeg", "a", "b"];
                argv.extend(flags);
                assert_eq!(
                    Cli::try_parse_from(argv).unwrap_err().kind(),
                    clap::error::ErrorKind::ArgumentConflict
                );
            }
        }
    }

    #[test]
    fn sink_receives_complete_public_api_output_and_reports_write_errors() {
        let input = jpeg();
        let options = parse(&["--no-cfl", "--default-order", "--brotli-quality", "0"]).options();
        let expected = transcode(&input, &options).unwrap();
        assert!(expected.starts_with(b"\0\0\0\x0cJXL \x0d\x0a\x87\x0a"));
        assert!(expected.windows(4).any(|bytes| bytes == b"jbrd"));
        let mut output = Vec::new();
        write_transcoded(&input, &options, |bytes| {
            output.write_all(bytes)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(output, expected);
        write_transcoded(&input, &options, |bytes| {
            std::io::sink().write_all(bytes)?;
            Ok(())
        })
        .unwrap();
        let error = write_transcoded(&input, &options, |_| {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "closed sink").into())
        })
        .unwrap_err();
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::BrokenPipe
        );
    }

    #[test]
    fn rejects_wrong_or_truncated_input_without_touching_sink() {
        let mut png = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png, 1, 1);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[0])
                .unwrap();
        }
        let valid = jpeg();
        let options = parse(&[]).options();
        for input in [b"".as_slice(), b"not JPEG", &png, &valid[..valid.len() / 2]] {
            let mut output = b"untouched".to_vec();
            assert!(
                write_transcoded(input, &options, |bytes| {
                    output.extend_from_slice(bytes);
                    Ok(())
                })
                .is_err()
            );
            assert_eq!(output, b"untouched");
        }
    }

    #[test]
    fn hard_budgets_fail_without_retry_or_partial_output() {
        let input = jpeg();
        let base = parse(&[]).options();
        let mut cases = vec![base.clone(); 7];
        cases[0].limits.max_output_bytes = 1;
        cases[1].limits.max_coefficients = 3 * 64 - 1;
        cases[2].limits.max_tokens = 1;
        cases[3].limits.input.structural.max_input_bytes = input.len() - 1;
        cases[4].limits.input.max_coefficient_bytes = 1;
        cases[5].limits.input.max_block_visits = 0;
        cases[6] = parse(&["--sample-limit", "1"]).options();
        for (i, options) in cases.iter().enumerate() {
            let mut output = b"untouched".to_vec();
            let error = write_transcoded(&input, options, |bytes| {
                output.extend_from_slice(bytes);
                Ok(())
            })
            .unwrap_err();
            assert!(
                error.downcast_ref::<rs_jxl::Error>().is_some(),
                "budget {i}: {error}"
            );
            assert_eq!(output, b"untouched", "budget {i}");
        }
        let exact = TranscodeOptions {
            limits: TranscodeLimits {
                max_coefficients: 3 * 64,
                ..base.limits
            },
            ..base
        };
        write_transcoded(&input, &exact, |_| Ok(())).unwrap();
    }

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            for _ in 0..128 {
                let id = NEXT.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir()
                    .join(format!("rs-jxl-cli-transcode-{}-{id}", std::process::id()));
                if fs::create_dir(&path).is_ok() {
                    return Self(path);
                }
            }
            panic!("could not create transcode scratch directory");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn file_args(input: &Path, output: &Path, force: bool) -> TranscodeArgs {
        let mut argv = vec![
            OsString::from("rs-jxl"),
            OsString::from("transcode-jpeg"),
            input.as_os_str().to_owned(),
            output.as_os_str().to_owned(),
        ];
        if force {
            argv.push("--force".into());
        }
        let Command::TranscodeJpeg(args) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("expected transcode-jpeg subcommand");
        };
        args
    }

    #[test]
    fn file_route_uses_atomic_output_force_and_content_detection() {
        let dir = Scratch::new();
        let input = dir.0.join("jpeg-with-wrong-extension.png");
        let output = dir.0.join("output.jxl");
        let bytes = jpeg();
        fs::write(&input, &bytes).unwrap();
        let expected = transcode(&bytes, &parse(&[]).options()).unwrap();
        run(file_args(&input, &output, false)).unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);
        fs::write(&output, b"existing output").unwrap();
        assert!(run(file_args(&input, &output, false)).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"existing output");
        run(file_args(&input, &output, true)).unwrap();
        assert_eq!(fs::read(&output).unwrap(), expected);
        assert!(run(file_args(&input, &input, true)).is_err());
        assert!(
            run(file_args(
                &input,
                &dir.0.join(".").join(input.file_name().unwrap()),
                true
            ))
            .is_err()
        );
        assert_eq!(fs::read(&input).unwrap(), bytes);
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
    }

    #[test]
    fn failed_file_transcode_does_not_create_or_replace_output() {
        let dir = Scratch::new();
        let input = dir.0.join("invalid.jpg");
        let output = dir.0.join("output.jxl");
        fs::write(&input, b"not JPEG").unwrap();
        assert!(run(file_args(&input, &output, false)).is_err());
        assert!(!output.exists());
        fs::write(&output, b"existing output").unwrap();
        assert!(run(file_args(&input, &output, true)).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"existing output");
        fs::write(&input, jpeg()).unwrap();
        let mut args = file_args(&input, &output, true);
        args.sample_limit = 1;
        assert!(run(args).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"existing output");
        assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 2);
    }

    #[test]
    fn source_search_has_a_separate_file_route_and_keeps_failure_atomic() {
        let dir = Scratch::new();
        let input = dir.0.join("source.jpg");
        let output = dir.0.join("searched.jxl");
        fs::write(&input, jpeg()).unwrap();
        let mut args = file_args(&input, &output, false);
        args.source_search = true;
        run(args).unwrap();
        assert!(fs::read(&output).unwrap().starts_with(b"\0\0\0\x0cJXL "));
        fs::write(&output, b"preserve").unwrap();
        let mut args = file_args(&input, &output, true);
        args.source_search = true;
        args.decoding_speed = 1;
        assert!(run(args).is_err());
        assert_eq!(fs::read(output).unwrap(), b"preserve");
    }
}
