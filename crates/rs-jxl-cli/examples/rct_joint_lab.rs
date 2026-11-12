//! Development-only thin wrapper for scheduler::rct_joint. JSON goes to stdout.
//! All trials use native e9 patches-off for RCT/MA isolation. No decoder
//! switches, source-path heuristics, or default CLI changes.
use clap::Parser;
use rs_jxl::{
    EncodeOptions,
    scheduler::{
        CandidateParams,
        rct_joint::{self, RctJointOptions, RctJointTrialKind},
    },
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::PathBuf, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Optional copy of the stdout JSON receipt. Existing files are refused.
    #[arg(long)]
    receipt: Option<PathBuf>,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=8))]
    shortlist: u8,
    #[arg(long, default_value_t = 8192, value_parser = clap::value_parser!(u32).range(9..=65536))]
    sample_pixels: u32,
}

fn params_json(p: &CandidateParams) -> Value {
    json!({
        "speed_tier": p.speed_tier,
        "buffering": format!("{:?}", p.buffering),
        "responsive": p.responsive,
        "group_size_shift": p.group_size_shift,
        "rct": p.rct,
        "palette_colors": p.palette_colors,
        "channel_colors_percent": p.channel_colors_percent,
        "channel_colors_pre_transform_percent": p.channel_colors_pre_transform_percent,
        "patches": format!("{:?}", p.patches),
        "predictor": p.predictor.map(|v| v as u32),
        "tree_mode": format!("{:?}", p.tree_mode),
        "learning_fraction": p.learning_fraction,
        "max_properties": p.max_properties,
        "lz77_careful_first_pass": p.lz77_careful_first_pass,
        "bucket_matcher": p.bucket_matcher,
        "precise_entropy": p.precise_entropy,
        "probe_lz77": p.probe_lz77,
        "delta_palette": format!("{:?}", p.delta_palette),
        "squeeze": p.squeeze,
    })
}

fn run(args: Args) -> Result<()> {
    let started = Instant::now();
    if args.output.exists() || args.receipt.as_ref().is_some_and(|p| p.exists()) {
        return Err("output/receipt already exists".into());
    }
    if args.receipt.as_ref() == Some(&args.output) {
        return Err("output and receipt must be different files".into());
    }
    let decode_started = Instant::now();
    let decoded = rs_jxl_image::decode(&fs::read(&args.input)?, true)?;
    let image = decoded.image;
    let input_decode_seconds = decode_started.elapsed().as_secs_f64();
    let settings = RctJointOptions {
        shortlist: usize::from(args.shortlist),
        max_sample_pixels: args.sample_pixels as usize,
    };
    let base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    let result = rct_joint::run_rct_joint(&image, &base, &settings)?;
    let write_started = Instant::now();
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?
        .write_all(&result.encoded)?;
    let output_write_seconds = write_started.elapsed().as_secs_f64();
    let trials: Vec<_> = result
        .trials
        .iter()
        .enumerate()
        .map(|(index, t)| {
            let o = &t.options;
            let sampling = o.modular.sampling.as_ref();
            json!({
                "index": index,
                "kind": match t.kind {
                    RctJointTrialKind::NativeE9Baseline => "native_e9_baseline",
                    RctJointTrialKind::AutoRctControl => "auto_rct_control",
                    RctJointTrialKind::ShortlistedRct => "shortlisted_rct",
                },
                "params": params_json(&t.params),
                "effective": {
                    "rct": o.modular.rct,
                    "rct_candidates": o.modular.rct_candidates,
                    "previous_channels": o.modular.previous_channels,
                    "training_previous_channels": sampling.map(|s| s.previous_channels),
                    "training_properties": sampling.map(|s| &s.properties),
                    "training_nb_repeats": sampling.map(|s| s.nb_repeats),
                    "predictor": o.modular.predictor as u32,
                    "fixed_tree": format!("{:?}", o.modular.fixed_tree),
                    "threads": o.threads,
                    "shared_tree": o.shared_tree,
                    "group_size_shift": o.group_size_shift,
                    "patches_disabled": o.patches.is_none()
                        && o.tile_patches.is_none()
                        && o.source_patches.is_none()
                        && o.source_patch_reference.is_none(),
                },
                "bytes": t.bytes,
                "seconds": t.seconds,
                "reused_from": t.reused_from,
                "error": t.error,
            })
        })
        .collect();
    let screens: Vec<_> = result
        .screen
        .scores
        .iter()
        .map(|s| {
            json!({
                "rct": s.rct, "native_bits": s.native_bits, "proxy_bits": s.proxy_bits,
                "seconds": s.seconds,
            })
        })
        .collect();
    let tiles: Vec<_> = result
        .screen
        .tiles
        .iter()
        .map(|t| {
            json!({
                "x": t.x, "y": t.y, "width": t.width, "height": t.height, "offset": t.offset,
            })
        })
        .collect();
    let receipt = json!({
        "schema": "rs-jxl.rct-joint-lab.v1",
        "effort": 9,
        "width": image.width(), "height": image.height(), "bits": image.bits(),
        "channels": image.channel_count(),
        "shortlist_limit": settings.shortlist,
        "sample_pixel_limit": settings.max_sample_pixels,
        "threads": base.threads,
        "winner_trial": result.winner_trial,
        "winner_bytes": result.encoded.len(),
        "selection": "complete_standard_jxl_bytes_first_minimum",
        "decoder_policy": "unchanged",
        "patch_policy": "disabled",
        "baseline_policy": "native_e9_patches_off",
        "baseline_trial": 0,
        "auto_rct_control_trials": [1, 2, 3],
        "actual_encodes": result.trials.iter().filter(|t| t.reused_from.is_none()).count(),
        "screen": {
            "method": result.screen.method,
            "real_ma_training": false,
            "sampled_pixels": result.screen.sampled_pixels,
            "sampled_channels": result.screen.sampled_channels,
            "tiles": tiles,
            "shortlisted_rcts": result.screen.shortlisted_rcts,
            "sampling_seconds": result.screen.sampling_seconds,
            "seconds": result.screen.seconds,
            "scores": screens,
        },
        "timing": {
            "total_seconds": result.timing.total_seconds,
            "screen_seconds": result.timing.screen_seconds,
            "encode_seconds": result.timing.encode_seconds,
            "overhead_seconds": result.timing.overhead_seconds,
            "pool_setup_seconds": result.timing.pool_setup_seconds,
            "input_decode_seconds": input_decode_seconds,
            "output_write_seconds": output_write_seconds,
            "lab_seconds_before_receipt_io": started.elapsed().as_secs_f64(),
        },
        "trials": trials,
    });
    let json = serde_json::to_vec(&receipt)?;
    if let Some(path) = args.receipt {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&json)?;
        file.write_all(b"\n")?;
    }
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(&json)?;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("rct_joint_lab: {error}");
        std::process::exit(1);
    }
}
