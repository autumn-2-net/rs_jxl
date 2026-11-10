//! Development-only palette ordering probe: one complete native encode, no search.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions,
    modular::{
        encode::receipt::{ReceiptCollector, ReceiptRoute},
        palette::PaletteOrder,
    },
    scheduler::{CandidateParams, adapter},
};
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Default,
    Frequency,
    Runs,
    Screened,
    Hot,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Frequency => "frequency",
            Self::Runs => "runs",
            Self::Screened => "screened",
            Self::Hot => "hot",
        }
    }
    fn order(self, hot_colors: usize) -> PaletteOrder {
        match self {
            Self::Default => PaletteOrder::Default,
            Self::Frequency => PaletteOrder::FrequencyLuma,
            Self::Runs => PaletteOrder::RunFrequency,
            Self::Screened => PaletteOrder::ScreenedRuns,
            Self::Hot => PaletteOrder::HotRuns { hot_colors },
        }
    }
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
    #[arg(long, value_enum, default_value = "default")]
    mode: Mode,
    #[arg(long, default_value_t = 128)]
    hot_colors: usize,
    /// Explicit experimental cap, equally applicable to every ordering mode.
    #[arg(long, value_parser = clap::value_parser!(i32).range(0..=70911))]
    palette_colors: Option<i32>,
    /// Disable the projected groupwise request without forcing palette admission.
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
    let policy = options
        .modular
        .palette_policy
        .as_mut()
        .ok_or("projected native candidate has no palette policy")?;
    let projected_groupwise = policy.groupwise;
    let projected_palette_colors = policy.palette_colors;
    policy.tools.order = args.mode.order(args.hot_colors);
    if let Some(colors) = args.palette_colors {
        policy.palette_colors = colors;
    }
    if args.full_frame {
        policy.groupwise = false;
    }
    let configured = serde_json::json!({
        "order": format!("{:?}", policy.tools.order),
        "hot_colors": (args.mode == Mode::Hot).then_some(args.hot_colors),
        "palette_colors": policy.palette_colors, "groupwise": policy.groupwise,
    });
    let receipt = Arc::new(ReceiptCollector::default());
    options.modular.representation_receipt = Some(Arc::clone(&receipt));
    let start = Instant::now();
    let bytes = rs_jxl::encode(&image, &options)?;
    let seconds = start.elapsed().as_secs_f64();
    let profile = rs_jxl::profile::report();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("invalid encode duration".into());
    }
    let actual = receipt
        .get()
        .ok_or("native encode did not publish representation receipt")?;
    let palettes: Vec<_> = actual
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
            "mode": args.mode.name(), "configured": configured,
            "effort": args.effort, "threads": args.threads, "encode_calls": 1,
            "patches_cleared": true,
            "profile": profile,
            "overrides": {"palette_colors": args.palette_colors, "full_frame": args.full_frame},
            "projected": {"groupwise": projected_groupwise, "palette_colors": projected_palette_colors},
            "bytes": bytes.len(), "seconds": seconds, "bpp": bytes.len() as f64 * 8.0 / pixels,
            "mpix_per_second": pixels / 1e6 / seconds,
            "width": image.width(), "height": image.height(),
            "palette_evidence": {
                "scope": "global_tuple_only; excludes single-channel compaction and local/groupwise palettes",
                "route": match actual.route {ReceiptRoute::FullImage => "full_image", ReceiptRoute::Groupwise => "groupwise"},
                "input_width": actual.input_width, "input_height": actual.input_height,
                "input_channels": actual.input_channels, "spatial_channels": actual.spatial_channels,
                "meta_channels": actual.meta_channels,
                "global_tuple_palette_count": palettes.len(), "global_tuple_palettes": palettes,
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
    fn arguments_and_orders() {
        let parse = |extra: &[&str]| {
            Args::try_parse_from(
                ["palette_lab", "in.png", "--output", "out.jxl"]
                    .into_iter()
                    .chain(extra.iter().copied()),
            )
        };
        let args = parse(&[]).unwrap();
        assert_eq!((args.effort, args.threads, args.hot_colors), (9, 8, 128));
        assert_eq!(args.mode, Mode::Default);
        assert!(!args.full_frame);
        assert_eq!(args.palette_colors, None);
        assert_eq!(Mode::Runs.order(0), PaletteOrder::RunFrequency);
        let screened = parse(&["--mode", "screened", "--hot-colors", "32"]).unwrap();
        assert_eq!(screened.mode, Mode::Screened);
        assert_eq!(screened.mode.name(), "screened");
        assert_eq!(
            screened.mode.order(screened.hot_colors),
            PaletteOrder::ScreenedRuns
        );
        assert_eq!(Mode::Screened.order(0), PaletteOrder::ScreenedRuns);
        assert_eq!(Mode::Hot.order(0), PaletteOrder::HotRuns { hot_colors: 0 });
        assert!(parse(&["--hot-colors", "0", "--palette-colors", "0", "--full-frame"]).is_ok());
        for extra in [
            ["--effort", "3"],
            ["--effort", "11"],
            ["--threads", "0"],
            ["--hot-colors", "-1"],
            ["--palette-colors", "70912"],
            ["--mode", "10"],
        ] {
            assert!(parse(&extra).is_err());
        }
    }
}
