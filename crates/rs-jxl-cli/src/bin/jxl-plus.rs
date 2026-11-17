//! Experimental JPEG-only frontend; default rs-jxl still selects standard JXL.
use clap::{Args, Parser, Subcommand};
use rs_jxl_plus::{DecodeLimits, EncodeOptions, decode_jpeg, encode_jpeg};
use std::{path::PathBuf, process::ExitCode};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Share the CLI's bounded I/O and atomic/no-clobber publication contracts.
#[allow(dead_code)]
#[path = "../io.rs"]
mod io;

#[derive(Parser)]
#[command(
    name = "jxl-plus",
    version,
    about = "Experimental jxl_plus JPEG byte-exact recompression (not standard JPEG XL)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode a supported JPEG to .jxl_plus, verifying original bytes before writing
    Encode {
        #[command(flatten)]
        files: Files,
        #[arg(long, default_value_t=6, value_parser=clap::value_parser!(u32).range(0..=11))]
        metadata_quality: u32,
    },
    /// Reconstruct the original JPEG from a .jxl_plus file
    Decode(Files),
}

#[derive(Args)]
struct Files {
    /// Input file, or - for binary stdin
    input: PathBuf,
    /// Output file, or - for binary stdout
    output: PathBuf,
    #[arg(long)]
    force: bool,
    #[arg(long, default_value_t=512*1024*1024)]
    max_input_bytes: usize,
    #[arg(long, default_value_t=512*1024*1024)]
    max_output_bytes: usize,
    #[arg(long, default_value_t=256*1024*1024)]
    max_coefficient_bytes: usize,
}

fn execute(command: Command) -> Result<()> {
    let (files, quality) = match command {
        Command::Encode {
            files,
            metadata_quality,
        } => (files, Some(metadata_quality)),
        Command::Decode(files) => (files, None),
    };
    io::check_paths(&files.input, &files.output, files.force)?;
    let input = io::read_input_with_limit(&files.input, files.max_input_bytes)?;
    let limits = DecodeLimits {
        max_input_bytes: files.max_input_bytes,
        max_output_bytes: files.max_output_bytes,
        max_coefficient_bytes: files.max_coefficient_bytes,
        ..DecodeLimits::default()
    };
    let output = if let Some(metadata_quality) = quality {
        encode_jpeg(
            &input,
            &EncodeOptions {
                limits,
                metadata_quality,
            },
        )?
    } else {
        decode_jpeg(&input, &limits)?
    };
    io::write_output(&files.input, &files.output, files.force, &output)
}

fn main() -> ExitCode {
    match execute(Cli::parse().command) {
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
    #[test]
    fn parser_rejects_unimplemented_options_and_bad_metadata_quality() {
        assert!(
            Cli::try_parse_from([
                "jxl-plus",
                "encode",
                "a.jpg",
                "a.jxl_plus",
                "--metadata-quality",
                "12"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "jxl-plus",
                "encode",
                "a.jpg",
                "a.jxl_plus",
                "--threads",
                "4"
            ])
            .is_err()
        );
        assert!(Cli::try_parse_from(["jxl-plus", "decode", "a.jxl_plus", "a.jpg"]).is_ok());
    }
}
