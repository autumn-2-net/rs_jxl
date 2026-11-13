//! One complete native e9 encode per process; independent delta-palette ablation.
//! No searchbest, shallow model, patch discovery, or production policy changes.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions,
    modular::{
        encode::receipt::{ReceiptCollector, ReceiptRoute},
        palette::{PaletteMode, PaletteOutcome, PaletteReceipt, PaletteStage},
    },
    scheduler::{CandidateParams, adapter},
};
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Mode {
    Absolute,
    ExactDeltaGradient,
    ExactDeltaLeft,
    ExactDeltaTop,
    ExactZeroGradient,
    ExactZeroLeft,
    ExactZeroTop,
}

impl Mode {
    fn source_mode(self) -> u8 {
        match self {
            Self::Absolute => 0,
            Self::ExactDeltaGradient => 1,
            Self::ExactDeltaLeft => 2,
            Self::ExactDeltaTop => 3,
            Self::ExactZeroGradient => 4,
            Self::ExactZeroLeft => 5,
            Self::ExactZeroTop => 6,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Absolute => "absolute",
            Self::ExactDeltaGradient => "exact-delta-gradient",
            Self::ExactDeltaLeft => "exact-delta-left",
            Self::ExactDeltaTop => "exact-delta-top",
            Self::ExactZeroGradient => "exact-zero-gradient",
            Self::ExactZeroLeft => "exact-zero-left",
            Self::ExactZeroTop => "exact-zero-top",
        }
    }
}

#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(9..=9))]
    effort: u8,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    #[arg(long, value_enum, default_value = "absolute")]
    mode: Mode,
    /// Explicit sensitivity override, identical for all modes in a comparison.
    #[arg(long, value_parser = clap::value_parser!(i32).range(1..=70911))]
    palette_colors: Option<i32>,
    /// Disable the projected groupwise request; local AC palettes still exist.
    #[arg(long)]
    full_frame: bool,
}

fn run(args: Args) -> Result<()> {
    if args.output.exists() {
        return Err("benchmark output already exists".into());
    }
    let image = rs_jxl_image::decode(&fs::read(&args.input)?, true)?.image;
    let params = CandidateParams::default().backend(args.effort)?;
    let base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    let mut options = adapter::project_native(&params, &image, &base)?.exact_options()?;
    options.patches = None;
    options.source_patches = None;
    options.source_patch_reference = None;
    options.tile_patches = None;
    if options.modular.fast_context.is_some() || options.modular.block_context.is_some() {
        return Err("delta ablation requires ordinary native MA".into());
    }
    let receipt = Arc::new(PaletteReceipt::default());
    let policy = options
        .modular
        .palette_policy
        .as_mut()
        .ok_or("missing e9 palette policy")?;
    let projected = serde_json::json!({
        "palette_colors": policy.palette_colors, "groupwise": policy.groupwise,
    });
    policy.mode = PaletteMode::from_source_mode(args.mode.source_mode())?;
    if let Some(colors) = args.palette_colors {
        policy.palette_colors = colors;
    }
    if args.full_frame {
        policy.groupwise = false;
    }
    policy.receipt = Some(Arc::clone(&receipt));
    let configured = serde_json::json!({
        "palette_colors": policy.palette_colors, "groupwise": policy.groupwise,
        "mode": format!("{:?}", policy.mode),
    });
    // Compare every non-mode option across variants, not just selected fields.
    let mut invariants = options.clone();
    let policy = invariants.modular.palette_policy.as_mut().unwrap();
    policy.mode = PaletteMode::Absolute;
    policy.receipt = None;
    let invariant_options = format!("{invariants:?}");
    let representation = Arc::new(ReceiptCollector::default());
    options.modular.representation_receipt = Some(Arc::clone(&representation));
    let start = Instant::now();
    let bytes = rs_jxl::encode(&image, &options)?;
    let seconds = start.elapsed().as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("invalid native encode duration".into());
    }
    let actual = representation
        .get()
        .ok_or("missing representation receipt")?;
    let attempts: Vec<_> = receipt.snapshot().iter().map(|a| serde_json::json!({
        "scope": match a.stage { PaletteStage::PreRct => "global", PaletteStage::PostRct => "local" },
        "requested_mode": format!("{:?}", a.mode),
        "width": a.width, "height": a.height, "begin": a.begin,
        "channels": a.channels, "limit": a.limit,
        "exact_delta_attempted": a.kernel.exact_delta_attempted,
        "exact_delta_produced": a.kernel.exact_delta_produced,
        "exact_zero_attempted": a.kernel.exact_zero_attempted,
        "zero_mask_eligible": a.kernel.zero_mask_eligible,
        "zero_mask_matches": a.kernel.zero_mask_matches,
        "absolute_attempted": a.kernel.absolute_attempted,
        "absolute_produced": a.kernel.absolute_produced,
        "colors": a.colors, "deltas": a.deltas, "predictor": format!("{:?}", a.predictor),
        "delta_pixels": a.delta_pixels, "cost_before": a.cost_before, "cost_after": a.cost_after,
        "cost_bypassed": a.cost_bypassed,
        "outcome": match a.outcome {
            PaletteOutcome::KernelDeclined => "kernel_declined",
            PaletteOutcome::CostRejected => "cost_rejected",
            PaletteOutcome::Accepted => "accepted",
        },
    })).collect();
    let global_tuples: Vec<_> = actual
        .global_tuple_palettes
        .iter()
        .map(|p| {
            serde_json::json!({
                "input_channels": [p.input_channels.start, p.input_channels.end],
                "colors": p.colors, "deltas": p.deltas, "predictor": format!("{:?}", p.predictor),
            })
        })
        .collect();
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?
        .write_all(&bytes)?;
    let pixels = f64::from(image.width()) * f64::from(image.height());
    println!(
        "{}",
        serde_json::json!({
            "mode": args.mode.name(), "effort": args.effort, "threads": args.threads,
            "encode_calls": 1, "search": false, "patches_cleared": true,
            "width": image.width(), "height": image.height(), "bytes": bytes.len(),
            "seconds": seconds, "bpp": bytes.len() as f64 * 8.0 / pixels,
            "mpix_per_second": pixels / 1e6 / seconds,
            "configured": configured, "projected": projected, "invariant_options": invariant_options,
            "overrides": {"palette_colors": args.palette_colors, "full_frame": args.full_frame},
            "palette_evidence": {
                "scope": "invocation_policy_attempts_global_and_local",
                "route": match actual.route { ReceiptRoute::FullImage => "full_image", ReceiptRoute::Groupwise => "groupwise" },
                "attempts": attempts, "global_tuple_palettes": global_tuples,
                "limitations": "Preparation acceptance, not byte savings; local physical channel offsets, no stream IDs; kernel decline combines eligibility/budget; single-channel attempts included; failed encodes may have rolled-back events",
            },
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
    fn seven_modes_and_bounded_arguments() {
        for (index, mode) in [
            Mode::Absolute,
            Mode::ExactDeltaGradient,
            Mode::ExactDeltaLeft,
            Mode::ExactDeltaTop,
            Mode::ExactZeroGradient,
            Mode::ExactZeroLeft,
            Mode::ExactZeroTop,
        ]
        .into_iter()
        .enumerate()
        {
            let args = Args::try_parse_from([
                "lab",
                "in.png",
                "--output",
                "out.jxl",
                "--mode",
                mode.name(),
            ])
            .unwrap();
            assert_eq!(args.mode, mode);
            assert_eq!(mode.source_mode(), index as u8);
            assert_eq!(args.effort, 9);
            assert!(!args.full_frame);
            assert_eq!(args.palette_colors, None);
        }
        for (flag, value) in [
            ("--effort", "8"),
            ("--effort", "10"),
            ("--threads", "0"),
            ("--palette-colors", "0"),
            ("--palette-colors", "70912"),
        ] {
            assert!(
                Args::try_parse_from(["lab", "in.png", "--output", "out.jxl", flag, value])
                    .is_err()
            );
        }
    }
}
