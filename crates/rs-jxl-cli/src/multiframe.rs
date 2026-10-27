//! Ordered still-image CLI orchestration for automatic lossless multiframe encoding.

use rs_jxl::EncodeOptions;
use rs_jxl::codec::Encoder;

use crate::{Result, io};

mod args;
mod input;
mod report;

pub(crate) use args::EncodeFramesArgs;
use input::load_images;
use report::{write_json_report, write_report};

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
        "note: encoding ordered decoded samples without sample, alpha, color or orientation conversion; automatic candidates are selected by encoded bytes, not elapsed time."
    );
    let result = {
        let _encoding = rs_jxl::profile::scope("cli.encode_frames");
        let mut encoder = Encoder::new(options.threads)?;
        encoder.encode_auto(&images, &durations, &options, &args.auto_options())?
    };
    if args.report_json {
        write_json_report(std::io::stderr().lock(), &result)?;
    } else {
        write_report(std::io::stderr().lock(), &result)?;
    }
    // Protect every source again before the existing transactional publisher.
    args.check_paths()?;
    io::write_output(&args.inputs[0], &args.output, args.force, &result.bytes)
}

#[cfg(test)]
mod tests;
