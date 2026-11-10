//! Isolated JPEG AC head/tail experiment; original JPEG bytes are the oracle.
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

use clap::Parser;
use rs_jxl::jpeg::{self, TranscodeOptions};

#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(1..=11))]
    effort: u8,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    /// Expanded standard AC HybridUint configurations, exact container fallback.
    #[arg(long)]
    head_tail: bool,
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let input = fs::read(&args.input)?;
    let (width, height) = {
        let info = jpeg::reader::read_jpeg(&input)?;
        (info.width, info.height)
    };
    let options = TranscodeOptions {
        effort: args.effort,
        threads: usize::from(args.threads),
        ..Default::default()
    };
    let receipt = Arc::new(jpeg::head_tail::Receipt::default());
    let start = Instant::now();
    let output = if args.head_tail {
        jpeg::head_tail::transcode(&input, &options, &receipt)?
    } else {
        jpeg::transcode(&input, &options)?
    };
    let seconds = start.elapsed().as_secs_f64();
    let reconstructed = rs_jxl::decode::reconstruct_jpeg(&output, &Default::default())?;
    if reconstructed != input {
        return Err("native JPEG reconstruction differs from original bytes".into());
    }
    if let Some(path) = args.output {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&output)?;
    }
    let head_tail = args.head_tail.then(|| {
        let r = receipt.snapshot();
        serde_json::json!({"attempted":r.attempted,"candidate":r.candidate,"selected":r.selected,
            "histograms":r.histograms,"changed_histograms":r.changed_histograms,
            "configurations":r.configurations,"baseline_bytes":r.baseline_bytes,
            "candidate_bytes":r.candidate_bytes,"search_seconds":r.search_seconds,"write_seconds":r.write_seconds})
    });
    let pixels = f64::from(width) * f64::from(height);
    println!(
        "{}",
        serde_json::json!({"bytes":output.len(),"seconds":seconds,"width":width,"height":height,
        "bpp":output.len() as f64*8.0/pixels,"mpix_per_second":pixels/1e6/seconds,
        "jpeg_bytes":input.len(),"native_verified":true,"head_tail":head_tail,
        "timing":"native JPEG parse and encode; excludes disk I/O, dimension inspection and reconstruction"})
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run(Args::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[test]
fn experiment_is_explicit_and_validates_effort_and_threads() {
    assert!(!Args::try_parse_from(["lab", "in.jpg"]).unwrap().head_tail);
    assert!(
        Args::try_parse_from(["lab", "in.jpg", "--head-tail"])
            .unwrap()
            .head_tail
    );
    for args in [
        ["--effort", "0"],
        ["--effort", "12"],
        ["--threads", "0"],
        ["--threads", "257"],
    ] {
        assert!(Args::try_parse_from(["lab", "in.jpg", args[0], args[1]]).is_err());
    }
}
