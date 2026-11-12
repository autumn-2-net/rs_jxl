//! Development-only raw MA root-lookahead trials. Exactly one native encode;
//! no greedy fallback, output-size selection, or estimate-to-wire-bit claim.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions, Image,
    modular::ma_train::{self, TreeLearning},
    scheduler::{CandidateParams, adapter, params::Override},
};
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Greedy,
    Lookahead2,
    Lookahead4,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Greedy => "greedy",
            Self::Lookahead2 => "lookahead2",
            Self::Lookahead4 => "lookahead4",
        }
    }

    fn width(self) -> Option<usize> {
        match self {
            Self::Greedy => None,
            Self::Lookahead2 => Some(2),
            Self::Lookahead4 => Some(4),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Learning {
    Legacy,
    Cost,
}

impl Learning {
    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Cost => "cost",
        }
    }
}

fn positive_size(value: &str) -> std::result::Result<usize, String> {
    value.parse::<usize>().ok().filter(|&n| n >= 4)
        .ok_or_else(|| "min-records must be an integer at least 4".into())
}

fn relative_window(value: &str) -> std::result::Result<f32, String> {
    value.parse::<f32>().ok().filter(|n| n.is_finite() && (0.0..=1.0).contains(n))
        .ok_or_else(|| "relative-window must be finite and in 0..=1".into())
}

#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(4..=10))]
    effort: u8,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    #[arg(long, value_enum, default_value = "greedy")]
    mode: Mode,
    #[arg(long, value_enum, default_value = "legacy")]
    tree_learning: Learning,
    #[arg(long, default_value_t = 8192, value_parser = positive_size)]
    min_records: usize,
    #[arg(long, default_value_t = 0.01, value_parser = relative_window)]
    relative_window: f32,
}

fn project(
    args: &Args,
    image: &Image,
    receipt: &Arc<ma_train::LookaheadReceipt>,
) -> Result<EncodeOptions> {
    let mut base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    base.modular.tree_learning = match args.tree_learning {
        Learning::Legacy => TreeLearning::Legacy,
        Learning::Cost => TreeLearning::Cost,
    };
    base.modular.source_ma_lookahead = args.mode.width().map(|width| ma_train::LookaheadOptions {
        width,
        min_records: args.min_records,
        relative_window: args.relative_window,
        receipt: Some(Arc::clone(receipt)),
    });
    let mut params = CandidateParams::default().backend(args.effort)?;
    params.patches = Override::Off;
    let mut options = adapter::project_native(&params, image, &base)?.exact_options()?;
    options.patches = None;
    options.source_patches = None;
    options.source_patch_reference = None;
    options.tile_patches = None;
    Ok(options)
}

fn run(args: Args) -> Result<()> {
    if args.output.exists() {
        return Err("benchmark output already exists".into());
    }
    let decoded = rs_jxl_image::decode(&fs::read(&args.input)?, true)?;
    let image = decoded.image;
    let receipt = Arc::new(ma_train::LookaheadReceipt::default());
    let options = project(&args, &image, &receipt)?;
    let start = Instant::now();
    let bytes = rs_jxl::encode(&image, &options)?;
    let seconds = start.elapsed().as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("invalid native encode duration".into());
    }
    let lookahead = args.mode.width().map(|_| {
        let work = receipt.snapshot();
        serde_json::json!({
            "roots": work.roots, "eligible": work.eligible,
            "probed_candidates": work.probed_candidates, "child_searches": work.child_searches,
            "changed": work.changed, "estimated_saved_bits": work.estimated_saved_bits,
        })
    });
    let pixels = f64::from(image.width()) * f64::from(image.height());
    fs::OpenOptions::new().write(true).create_new(true).open(&args.output)?.write_all(&bytes)?;
    println!("{}", serde_json::json!({
        "bytes": bytes.len(), "seconds": seconds,
        "bpp": bytes.len() as f64 * 8.0 / pixels,
        "mpix_per_second": pixels / 1e6 / seconds,
        "width": image.width(), "height": image.height(),
        "mode": args.mode.name(), "tree_learning": args.tree_learning.name(),
        "effort": args.effort, "threads": args.threads,
        "previous_channels": options.modular.previous_channels,
        "patches": "Off", "patches_cleared": options.patches.is_none()
            && options.source_patches.is_none() && options.source_patch_reference.is_none()
            && options.tile_patches.is_none(),
        "lookahead_options": args.mode.width().map(|width| serde_json::json!({
            "width": width, "min_records": args.min_records, "relative_window": args.relative_window,
        })),
        "lookahead": lookahead,
        "lookahead_score_contract": "estimated local scoring only; not emitted wire bits",
        "native_encode_calls": 1,
    }));
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_and_bounds() {
        let argv = ["ma_lookahead_lab", "in.png", "--output", "out.jxl"];
        let args = Args::try_parse_from(argv).unwrap();
        assert_eq!((args.effort, args.threads, args.min_records), (9, 8, 8192));
        assert_eq!(args.mode, Mode::Greedy);
        assert_eq!(args.tree_learning, Learning::Legacy);
        assert_eq!(args.relative_window, 0.01);
        for (flag, value) in [
            ("--effort", "3"), ("--effort", "11"), ("--threads", "0"),
            ("--threads", "257"), ("--min-records", "0"), ("--min-records", "3"),
            ("--relative-window", "NaN"), ("--relative-window", "inf"),
            ("--relative-window", "1.1"), ("--mode", "lookahead3"),
            ("--tree-learning", "symmetric"),
        ] {
            assert!(Args::try_parse_from(argv.into_iter().chain([flag, value])).is_err());
        }
        assert!(Args::try_parse_from(["ma_lookahead_lab", "in.png"]).is_err());
    }

    #[test]
    fn native_projection_retains_requested_lookahead() {
        let image = Image::new(128, 128, rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 128 * 128 * 3])).unwrap();
        for mode in ["greedy", "lookahead2", "lookahead4"] {
            for learning in ["legacy", "cost"] {
                let args = Args::try_parse_from([
                    "ma_lookahead_lab", "in.png", "--output", "out.jxl", "--mode", mode,
                    "--tree-learning", learning,
                ]).unwrap();
                let receipt = Arc::new(ma_train::LookaheadReceipt::default());
                let options = project(&args, &image, &receipt).unwrap();
                assert!(options.modular.sampling.is_some());
                assert!(options.modular.fixed_tree.is_none());
                assert_eq!(options.modular.previous_channels, 0);
                assert!(options.patches.is_none() && options.source_patches.is_none()
                    && options.source_patch_reference.is_none() && options.tile_patches.is_none());
                assert_eq!(options.modular.tree_learning, match args.tree_learning {
                    Learning::Legacy => TreeLearning::Legacy,
                    Learning::Cost => TreeLearning::Cost,
                });
                assert_eq!(options.modular.source_ma_lookahead.is_some(), mode != "greedy");
                if let Some(lookahead) = options.modular.source_ma_lookahead {
                    assert_eq!(Some(lookahead.width), args.mode.width());
                    assert_eq!(lookahead.min_records, 8192);
                    assert!(Arc::ptr_eq(lookahead.receipt.as_ref().unwrap(), &receipt));
                }
            }
        }
    }
}
