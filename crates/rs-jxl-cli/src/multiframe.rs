//! Ordered still-image CLI orchestration for automatic lossless multiframe encoding.

use rs_jxl::EncodeOptions;
use rs_jxl::codec::Encoder;

use crate::{Result, io};

mod args;
mod input;
mod report;

pub(crate) use args::EncodeFramesArgs;
use args::StrategyArg;
use input::load_images;
use report::{write_crop_report, write_json_report, write_report};

pub(crate) fn run(args: EncodeFramesArgs) -> Result<()> {
    let _total = rs_jxl::profile::scope("cli.total");
    args.validate_options()?;
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
        "note: encoding ordered decoded samples without sample, alpha, color or orientation conversion."
    );
    let bytes = {
        let _encoding = rs_jxl::profile::scope("cli.encode_frames");
        let mut encoder = Encoder::new(options.threads)?;
        match args.strategy {
            StrategyArg::Auto => {
                let result =
                    encoder.encode_auto(&images, &durations, &options, &args.auto_options())?;
                if args.report_json {
                    write_json_report(std::io::stderr().lock(), &result)?;
                } else {
                    write_report(std::io::stderr().lock(), &result)?;
                }
                result.bytes
            }
            StrategyArg::Crop => {
                let started = std::time::Instant::now();
                let plan = rs_jxl::multiframe::plan(&images, &durations, args.reference())?;
                let frames: Vec<_> = plan.iter().map(|frame| frame.as_frame()).collect();
                let bytes = encoder.encode_frames(&images[0], &frames, &options)?;
                let elapsed = started.elapsed();
                write_crop_report(
                    std::io::stderr().lock(),
                    &plan,
                    bytes.len(),
                    elapsed,
                    args.report_json,
                )?;
                bytes
            }
        }
    };
    // Protect every source again before the existing transactional publisher.
    args.check_paths()?;
    io::write_output(&args.inputs[0], &args.output, args.force, &bytes)
}

#[cfg(test)]
mod tests;
