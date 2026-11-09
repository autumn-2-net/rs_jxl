//! Development-only exact patch experiments; not part of the installed CLI.
//! selected_mode=source means the source detector candidate is enabled, not
//! proof that discovery emitted a dictionary. Source gates can still disable it.
//! Screened tiles use cheap admission before encoding; guarded is an explicit
//! dual-encode diagnostic, not the admission policy.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions, Image,
    scheduler::{CandidateParams, adapter, params::Override},
    tools::patches::{TilePatchOptions, TilePatchReceipt},
};
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Off,
    Source,
    Components,
    Tiles,
    Guarded,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Source => "source",
            Self::Components => "components",
            Self::Tiles => "tiles",
            Self::Guarded => "guarded",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum TileAdmission {
    Screened,
    Force,
}

impl TileAdmission {
    fn name(self) -> &'static str {
        match self {
            Self::Screened => "screened",
            Self::Force => "force",
        }
    }
}

fn positive_size(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&size| size > 0)
        .ok_or_else(|| "tile-size must be a positive integer".into())
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
    #[arg(long, value_enum, default_value = "off")]
    mode: Mode,
    #[arg(long, default_value_t = 16, value_parser = positive_size)]
    tile_size: usize,
    #[arg(long, value_enum, default_value = "screened")]
    tile_admission: TileAdmission,
}

fn clear_detectors(options: &mut EncodeOptions) {
    options.patches = None;
    options.source_patches = None;
    options.source_patch_reference = None;
    options.tile_patches = None;
}

fn encode_timed(image: &Image, options: &EncodeOptions) -> Result<(Vec<u8>, f64)> {
    let start = Instant::now();
    let bytes = rs_jxl::encode(image, options)?;
    Ok((bytes, start.elapsed().as_secs_f64()))
}

fn run(args: Args) -> Result<()> {
    if args.output.exists() {
        return Err("benchmark output already exists".into());
    }
    let decoded = rs_jxl_image::decode(&fs::read(&args.input)?, true)?;
    let image = decoded.image;
    let mut params = CandidateParams::default().backend(args.effort)?;
    let base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    // Keep the normal source candidate's non-patch settings across all trials.
    let mut options = adapter::project_native(&params, &image, &base)?.exact_options()?;
    clear_detectors(&mut options);
    let receipt = Arc::new(TilePatchReceipt::default());
    let mut guard_work = None;
    let mut source_detector_enabled = None;
    let (bytes, seconds, mut selected_mode) = match args.mode {
        Mode::Off => {
            let (bytes, seconds) = encode_timed(&image, &options)?;
            (bytes, seconds, Mode::Off)
        }
        Mode::Source => {
            params.patches = Override::On;
            let source = adapter::project_native(&params, &image, &base)?.exact_options()?;
            // Only adopt the forced detector and its deferred atlas policy.
            options.source_patches = source.source_patches;
            options.source_patch_reference = source.source_patch_reference;
            let enabled = options.source_patches.is_some();
            source_detector_enabled = Some(enabled);
            let (bytes, seconds) = encode_timed(&image, &options)?;
            (
                bytes,
                seconds,
                if enabled { Mode::Source } else { Mode::Off },
            )
        }
        Mode::Components => {
            options.patches = Some(Default::default());
            let (bytes, seconds) = encode_timed(&image, &options)?;
            (bytes, seconds, Mode::Components)
        }
        Mode::Tiles | Mode::Guarded => {
            let mut tiles = options.clone();
            tiles.tile_patches = Some(TilePatchOptions {
                tile_size: args.tile_size,
                force: args.tile_admission == TileAdmission::Force,
                receipt: Some(Arc::clone(&receipt)),
                ..Default::default()
            });
            if args.mode == Mode::Guarded {
                let (off_bytes, off_seconds) = encode_timed(&image, &options)?;
                let (tile_bytes, tile_seconds) = encode_timed(&image, &tiles)?;
                guard_work = Some(serde_json::json!({
                    "off_bytes": off_bytes.len(), "tiles_bytes": tile_bytes.len(),
                    "off_seconds": off_seconds, "tiles_seconds": tile_seconds,
                }));
                let seconds = off_seconds + tile_seconds;
                if tile_bytes.len() < off_bytes.len() {
                    (tile_bytes, seconds, Mode::Tiles)
                } else {
                    (off_bytes, seconds, Mode::Off)
                }
            } else {
                let (bytes, seconds) = encode_timed(&image, &tiles)?;
                (bytes, seconds, Mode::Tiles)
            }
        }
    };
    let tile_work = matches!(args.mode, Mode::Tiles | Mode::Guarded).then(|| {
        let work = receipt.snapshot();
        if work.selected_groups == 0 {
            selected_mode = Mode::Off;
        }
        serde_json::json!({
            "tile_size": args.tile_size,
            "calls": work.calls, "tiles_examined": work.tiles_examined,
            "constant_tiles": work.constant_tiles, "repeated_groups": work.repeated_groups,
            "selected_groups": work.selected_groups, "positions": work.positions,
            "covered_pixels": work.covered_pixels, "atlas_pixels": work.atlas_pixels,
            "admission_rejected": work.admission_rejected,
            "estimated_saved_bits": work.estimated_saved_bits,
            "estimated_baseline_bits": work.estimated_baseline_bits,
            "seconds": work.seconds,
            "plan_selected": work.selected_groups != 0,
            "note": if work.admission_rejected != 0 { "rejected_by_estimate" }
                else if work.selected_groups != 0 { "plan_selected" }
                else if work.repeated_groups != 0 { "no_selection" }
                else { "no_repeated_tiles" },
        })
    });
    let pixels = f64::from(image.width()) * f64::from(image.height());
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("invalid native encode duration".into());
    }
    // create_new also closes the race after the initial no-overwrite check.
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?
        .write_all(&bytes)?;
    println!(
        "{}",
        serde_json::json!({
            "bytes": bytes.len(), "seconds": seconds,
            "mpix_per_second": pixels / 1e6 / seconds,
            "bpp": bytes.len() as f64 * 8.0 / pixels,
            "width": image.width(), "height": image.height(),
            "mode": args.mode.name(), "selected_mode": selected_mode.name(),
            "source_detector_enabled": source_detector_enabled,
            "tile_admission": matches!(args.mode, Mode::Tiles | Mode::Guarded)
                .then(|| args.tile_admission.name()),
            "tile_work": tile_work, "guard_work": guard_work,
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
    fn argument_contract() {
        let args = Args::try_parse_from(["patch_lab", "in.png", "--output", "out.jxl"])
            .expect("default arguments");
        assert_eq!((args.effort, args.threads, args.tile_size), (9, 8, 16));
        assert_eq!(args.mode, Mode::Off);
        assert_eq!(args.tile_admission, TileAdmission::Screened);
        for (flag, value) in [
            ("--effort", "3"),
            ("--effort", "11"),
            ("--threads", "0"),
            ("--tile-size", "0"),
            ("--tile-admission", "automatic"),
            ("--mode", "baseline"),
        ] {
            assert!(
                Args::try_parse_from(["patch_lab", "in.png", "--output", "out.jxl", flag, value,])
                    .is_err()
            );
        }
        assert!(Args::try_parse_from(["patch_lab", "in.png"]).is_err());
        let forced = Args::try_parse_from([
            "patch_lab",
            "in.png",
            "--output",
            "out.jxl",
            "--mode",
            "tiles",
            "--tile-admission",
            "force",
        ])
        .expect("forced tile admission");
        assert_eq!(forced.tile_admission, TileAdmission::Force);
    }
}
