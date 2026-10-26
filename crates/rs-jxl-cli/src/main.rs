mod decode;
mod encode;
mod io;
mod multiframe;
mod options;
mod transcode;
mod validators;

#[cfg(test)]
mod acceptance;

#[cfg(test)]
#[path = "ma_budget_parser_tests.rs"]
mod ma_budget_parser_tests;

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use rs_jxl_image as input;

// Keep the crate-root parser path used by decoder controls.
use validators::parse_threads;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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
    Encode(encode::EncodeArgs),
    /// Encode ordered PNG/JPEG frames with automatic lossless strategies
    EncodeFrames(multiframe::EncodeFramesArgs),
    /// Transcode JPEG coefficients with original-JPEG reconstruction data
    TranscodeJpeg(transcode::TranscodeArgs),
}

#[cfg(test)]
fn expect_encode(command: Command) -> encode::EncodeArgs {
    let Command::Encode(args) = command else {
        panic!("expected encode subcommand");
    };
    args
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Decode(args) => decode::run(args),
        Command::Encode(args) => encode::run(args),
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
#[path = "cli_tests.rs"]
mod tests;
