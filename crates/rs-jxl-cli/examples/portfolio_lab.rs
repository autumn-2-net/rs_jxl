//! Opt-in native portfolio benchmark. No corpus traversal or automatic launch.
use clap::Parser;
use rs_jxl::scheduler::portfolio::{self, ALL, Candidate, Options};
use std::{fs, io::Write, path::PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    input: Option<PathBuf>,
    #[arg(long, required_unless_present_any = ["plan_only", "list_candidates"])]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(8..=10))]
    effort: u8,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    #[arg(long, default_value_t = 48)]
    max_probe_encodes: usize,
    #[arg(long, default_value_t = 2)]
    max_finalists: usize,
    #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u16).range(32..=512))]
    probe_side: u16,
    #[arg(long, default_value_t = 0.01)]
    uncertainty: f64,
    /// Explicit comma-separated subset; baseline is always retained.
    #[arg(long, value_delimiter = ',')]
    candidates: Vec<String>,
    #[arg(long, conflicts_with = "output")]
    plan_only: bool,
    #[arg(long, conflicts_with_all = ["output", "input"])]
    list_candidates: bool,
}

fn candidate(name: &str) -> Result<Candidate> {
    ALL.iter()
        .copied()
        .find(|c| c.name() == name)
        .ok_or_else(|| format!("unknown candidate {name}; use --list-candidates").into())
}

fn run(args: Args) -> Result<()> {
    if args.list_candidates {
        println!(
            "{}",
            serde_json::json!(
                ALL.iter()
                    .map(|c| serde_json::json!({
                        "name": c.name(), "family": format!("{:?}", c.family()),
                    }))
                    .collect::<Vec<_>>()
            )
        );
        return Ok(());
    }
    let input = args.input.ok_or("input is required")?;
    if args.output.as_ref().is_some_and(|p| p.exists()) {
        return Err("output already exists".into());
    }
    let image = rs_jxl_image::decode(&fs::read(input)?, true)?.image;
    let options = Options {
        effort: args.effort,
        threads: usize::from(args.threads),
        max_probe_encodes: args.max_probe_encodes,
        max_finalists: args.max_finalists,
        probe_side: usize::from(args.probe_side),
        uncertainty: args.uncertainty,
        candidates: if args.candidates.is_empty() {
            ALL.to_vec()
        } else {
            args.candidates
                .iter()
                .map(|s| candidate(s))
                .collect::<Result<_>>()?
        },
        ..Default::default()
    };
    let (bytes, report) = if args.plan_only {
        (None, portfolio::plan(&image, &options)?)
    } else {
        let result = portfolio::encode(&image, &options)?;
        (Some(result.bytes), result.report)
    };
    if let Some(bytes) = &bytes {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.output.ok_or("output is required")?)?
            .write_all(bytes)?;
    }
    let pixels = f64::from(image.width()) * f64::from(image.height());
    let baseline = &report.candidates[0];
    let f = &report.features;
    println!(
        "{}",
        serde_json::json!({
            "schema": 1, "plan_only": args.plan_only,
            "effort": args.effort, "threads": args.threads,
            "width": image.width(), "height": image.height(), "bits": image.bits(),
            "bytes": bytes.as_ref().map(Vec::len),
            "bpp": bytes.as_ref().map(|b| b.len() as f64 * 8.0 / pixels),
            "seconds": report.seconds,
            "mpix_per_second": (report.seconds > 0.0).then(|| pixels / 1e6 / report.seconds),
            "baseline": {"bytes": baseline.full_bytes, "seconds": baseline.full_seconds,
                "bpp": baseline.full_bytes.map(|n| n as f64 * 8.0 / pixels)},
            "scout_seconds": report.scout_seconds,
            "probe_encodes": report.probe_encodes, "probe_pixels": report.probe_pixels,
            "full_encodes": report.full_encodes, "winner": report.winner.name(),
            "features": {"samples": f.samples, "unique_fraction": f.unique_fraction,
                "flat_fraction": f.flat_fraction, "gradient_zero_fraction": f.gradient_zero_fraction,
                "residual_bias": f.residual_bias, "cross_channel_gain": f.cross_channel_gain,
                "residual_unique_fraction": f.residual_unique_fraction,
            "residual_zero_fraction": f.residual_zero_fraction,
                "repeated_tile_fraction": f.repeated_tile_fraction, "sampled_tiles": f.sampled_tiles},
            "regions": report.regions.iter().map(|r| serde_json::json!({
                "x": r.x,"y": r.y,"width": r.width,"height": r.height,
            })).collect::<Vec<_>>(),
            "candidates": report.candidates.iter().map(|c| serde_json::json!({
                "name": c.candidate.name(), "family": format!("{:?}", c.candidate.family()),
                "decision": c.decision, "reason": c.reason, "priority": c.priority,
                "probe_bytes": c.probe_bytes, "probe_seconds": c.probe_seconds,
                "full_bytes": c.full_bytes, "full_seconds": c.full_seconds, "error": c.error,
            })).collect::<Vec<_>>(),
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parsing_requires_explicit_output_or_analysis() {
        assert!(Args::try_parse_from(["lab", "input.png"]).is_err());
        assert!(Args::try_parse_from(["lab", "--list-candidates"]).is_ok());
        assert!(Args::try_parse_from(["lab", "input.png", "--plan-only"]).is_ok());
        assert!(
            Args::try_parse_from(["lab", "input.png", "--plan-only", "--output", "out.jxl"])
                .is_err()
        );
        assert!(candidate("made-up-tool").is_err());
    }
}
