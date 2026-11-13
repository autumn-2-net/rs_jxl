//! Development-only exact patch experiments; not part of the installed CLI.
//! selected_mode=source means the source detector candidate is enabled, not
//! proof that discovery emitted a dictionary. Source gates can still disable it.
//! Screened tiles use cheap admission before encoding; guarded is an explicit
//! dual-encode diagnostic, not the admission policy.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions, Image,
    modular::block_context,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BlockTarget {
    Reference,
    Residual,
    Both,
}

impl BlockTarget {
    fn name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Residual => "residual",
            Self::Both => "both",
        }
    }

    fn reference(self) -> bool {
        matches!(self, Self::Reference | Self::Both)
    }

    fn residual(self) -> bool {
        matches!(self, Self::Residual | Self::Both)
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
    /// Match grid seeds at arbitrary pixel destinations (tiles/guarded only).
    #[arg(long)]
    off_grid: bool,
    /// Replace MA learning with a standard-predictor spatial tree (off/source).
    #[arg(long, value_parser = ["8", "16"])]
    block_size: Option<String>,
    /// Source atlas, residual/whole frame (default), or both; off is residual only.
    #[arg(long, value_enum, requires = "block_size")]
    block_target: Option<BlockTarget>,
    /// Planning terminal budget; use a larger value to test uncoarsened grids.
    #[arg(long, requires = "block_size", value_parser = positive_size)]
    block_leaves: Option<usize>,
    /// Print profile totals to stderr if RS_JXL_PROFILE enables collection.
    #[arg(long)]
    profile: bool,
    /// Opt into existing parallel sparse MA sample gathering (off/source only).
    #[arg(long)]
    parallel_gather: bool,
    /// Use ordered bounded waves instead of requiring the whole batch to fit.
    #[arg(long, requires = "parallel_gather")]
    gather_waves: bool,
    /// Bound cross-channel MA properties without changing omitted defaults.
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=2))]
    previous_channels: Option<u8>,
    /// Main frame/residual (default), separately resolved reference, or both.
    #[arg(long, value_enum, requires = "previous_channels")]
    channel_target: Option<BlockTarget>,
}

fn configure_channels(
    options: &mut EncodeOptions,
    params: &CandidateParams,
    image: &Image,
    base: &EncodeOptions,
    previous: u8,
    target: BlockTarget,
) -> Result<()> {
    if target.residual() {
        let mut params = params.clone();
        params.max_properties = previous;
        let projected = adapter::project_native(&params, image, base)?.exact_options()?;
        options.modular.previous_channels = projected.modular.previous_channels;
        options.modular.sampling = projected.modular.sampling;
    }
    if target.reference() {
        if let Some(reference) = options.source_patch_reference.as_mut() {
            // resolve() overwrites the modular template from this RAW policy.
            reference.sampling.max_previous_channels = usize::from(previous);
        }
    }
    Ok(())
}

fn channel_receipt(
    options: &EncodeOptions,
    image: &Image,
    previous: Option<u8>,
    target: BlockTarget,
) -> Result<serde_json::Value> {
    let reference = options.source_patch_reference.as_ref();
    Ok(serde_json::json!({
        "requested": previous,
        "target": previous.map(|_| target.name()),
        "main": {
            "configured": previous.is_some() && target.residual(),
            "configured_previous_channels": previous.filter(|_| target.residual()),
            "effective_previous_channels": options.modular.previous_channels,
        },
        "reference": {
            "available": reference.is_some(),
            "configured": previous.is_some() && target.reference() && reference.is_some(),
            "configured_previous_channels": reference.map(|r| r.sampling.max_previous_channels),
            "effective_previous_channels": reference.map(|r|
                r.sampling.max_previous_channels.min(image.channel_count().saturating_sub(1))),
        },
        "scope": "configuration only; reference budget capped using source channel metadata (same channel count as atlas), not actual atlas resolution or proof of emission",
    }))
}

fn configure_parallel_gather(options: &mut EncodeOptions, threads: u16, waves: bool) {
    let gather = rs_jxl::modular::sample::GatherOptions {
        mode: if waves {
            rs_jxl::modular::sample::GatherMode::Waves
        } else {
            rs_jxl::modular::sample::GatherMode::Channels
        },
        workers: usize::from(threads).min(64),
        sparse: true,
        ..Default::default()
    };
    options.modular.source_gather = Some(gather.clone());
    if let Some(reference) = options.source_patch_reference.as_mut() {
        reference.modular.source_gather = Some(gather);
    }
}

fn configure_blocks(
    options: &mut EncodeOptions,
    target: BlockTarget,
    block: &block_context::Options,
    reference_receipt: &Arc<block_context::Receipt>,
) {
    if target.residual() {
        options.modular.block_context = Some(block.clone());
    }
    if target.reference() {
        if let Some(reference) = options.source_patch_reference.as_mut() {
            let mut block = block.clone();
            block.receipt = Some(Arc::clone(reference_receipt));
            reference.modular.block_context = Some(block);
        }
    }
}

fn block_work(receipt: &block_context::Receipt) -> serde_json::Value {
    let work = receipt.snapshot();
    serde_json::json!({
        "trees": work.trees, "pixels": work.pixels, "views": work.views,
        "requested_tiles": work.requested_tiles, "leaves": work.leaves,
        "spatial_splits": work.spatial_splits, "channel_splits": work.channel_splits,
        "stream_splits": work.stream_splits, "coarsened_views": work.coarsened_views,
        "predictor_leaves": work.predictor_leaves,
    })
}

fn targeted_block_work(receipt: &block_context::Receipt, configured: bool) -> serde_json::Value {
    let mut work = block_work(receipt);
    work["configured"] = configured.into();
    work["used"] = (work["trees"].as_u64().unwrap() != 0).into();
    work
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
    let channel_target = args.channel_target.unwrap_or(BlockTarget::Residual);
    if args.previous_channels.is_some() && !matches!(args.mode, Mode::Off | Mode::Source) {
        return Err("--previous-channels requires --mode off or source".into());
    }
    if args.mode != Mode::Source && channel_target != BlockTarget::Residual {
        return Err("--channel-target reference or both requires --mode source".into());
    }
    if args.off_grid && !matches!(args.mode, Mode::Tiles | Mode::Guarded) {
        return Err("--off-grid requires --mode tiles or guarded".into());
    }
    if args.block_size.is_some() && !matches!(args.mode, Mode::Off | Mode::Source) {
        return Err("--block-size requires --mode off or source".into());
    }
    if args.parallel_gather && !matches!(args.mode, Mode::Off | Mode::Source) {
        return Err("--parallel-gather requires --mode off or source".into());
    }
    let block_target = args.block_target.unwrap_or(BlockTarget::Residual);
    if args.mode != Mode::Source && block_target != BlockTarget::Residual {
        return Err("--block-target reference or both requires --mode source".into());
    }
    if args.output.exists() {
        return Err("benchmark output already exists".into());
    }
    let decoded = rs_jxl_image::decode(&fs::read(&args.input)?, true)?;
    let image = decoded.image;
    let mut params = CandidateParams::default().backend(args.effort)?;
    let main_params = params.clone();
    let base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    // Keep the normal source candidate's non-patch settings across all trials.
    let mut options = adapter::project_native(&params, &image, &base)?.exact_options()?;
    clear_detectors(&mut options);
    if args.mode == Mode::Source {
        params.patches = Override::On;
        let source = adapter::project_native(&params, &image, &base)?.exact_options()?;
        // Only adopt the forced detector and its deferred atlas policy.
        options.source_patches = source.source_patches;
        options.source_patch_reference = source.source_patch_reference;
    }
    if args.parallel_gather {
        configure_parallel_gather(&mut options, args.threads, args.gather_waves);
    }
    if let Some(previous) = args.previous_channels {
        configure_channels(
            &mut options,
            &main_params,
            &image,
            &base,
            previous,
            channel_target,
        )?;
    }
    let channels = channel_receipt(&options, &image, args.previous_channels, channel_target)?;
    let block_size = args
        .block_size
        .as_ref()
        .map(|s| s.parse::<usize>().unwrap());
    let block_receipt = Arc::new(block_context::Receipt::default());
    let reference_block_receipt = Arc::new(block_context::Receipt::default());
    let block_plan_leaves = block_size.map(|_| {
        args.block_leaves
            .unwrap_or(block_context::DEFAULT_PLAN_LEAVES)
    });
    if let Some(tile_size) = block_size {
        let block = block_context::Options {
            tile_size,
            max_plan_leaves: block_plan_leaves.unwrap(),
            receipt: Some(Arc::clone(&block_receipt)),
        };
        // Apply AFTER source projection so the atlas policy retains its own collector.
        configure_blocks(&mut options, block_target, &block, &reference_block_receipt);
    }
    let receipt = Arc::new(TilePatchReceipt::default());
    let mut guard_work = None;
    let mut source_detector_enabled = None;
    let (bytes, seconds, mut selected_mode) = match args.mode {
        Mode::Off => {
            let (bytes, seconds) = encode_timed(&image, &options)?;
            (bytes, seconds, Mode::Off)
        }
        Mode::Source => {
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
                off_grid: args.off_grid,
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
    if args.profile {
        if let Some(report) = rs_jxl::profile::report() {
            eprint!("{report}");
        }
    }
    let tile_work = matches!(args.mode, Mode::Tiles | Mode::Guarded).then(|| {
        let work = receipt.snapshot();
        if work.selected_groups == 0 {
            selected_mode = Mode::Off;
        }
        serde_json::json!({
            "tile_size": args.tile_size,
            "off_grid": args.off_grid,
            "windows_examined": work.windows_examined,
            "discovery_limited": work.discovery_limited,
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
    let block_work = block_size.map(|_| {
        if args.mode == Mode::Source {
            serde_json::json!({
                "reference": block_target.reference().then(|| targeted_block_work(
                    &reference_block_receipt,
                    options.source_patch_reference.as_ref()
                        .is_some_and(|r| r.modular.block_context.is_some()),
                )),
                "residual": block_target.residual().then(|| targeted_block_work(
                    &block_receipt, options.modular.block_context.is_some(),
                )),
            })
        } else {
            // Preserve the original flat off-mode block receipt.
            block_work(&block_receipt)
        }
    });
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
            "source_patch_selected": if args.mode == Mode::Off { Some(false) } else { None },
            "selected_mode_contract": "enabled route only; source does not prove dictionary/reference emission",
            "tile_admission": matches!(args.mode, Mode::Tiles | Mode::Guarded)
                .then(|| args.tile_admission.name()),
            "tile_work": tile_work, "guard_work": guard_work,
            "block_size": block_size, "block_work": block_work,
            "block_target": block_size.map(|_| block_target.name()),
            "block_plan_leaves": block_plan_leaves,
            "parallel_gather": args.parallel_gather,
            "gather_mode": if args.gather_waves { "waves" } else { "channels" },
            "channels": channels,
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
        assert!(!args.off_grid);
        assert!(args.block_size.is_none());
        assert!(args.block_target.is_none());
        assert!(!args.profile);
        assert!(!args.parallel_gather);
        assert!(args.previous_channels.is_none());
        assert!(args.channel_target.is_none());
        for (flag, value) in [
            ("--effort", "3"),
            ("--effort", "11"),
            ("--threads", "0"),
            ("--tile-size", "0"),
            ("--block-size", "32"),
            ("--block-leaves", "0"),
            ("--block-target", "whole"),
            ("--tile-admission", "automatic"),
            ("--mode", "baseline"),
            ("--previous-channels", "3"),
            ("--previous-channels", "-1"),
            ("--channel-target", "invalid"),
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
        assert!(
            Args::try_parse_from([
                "patch_lab",
                "in.png",
                "--output",
                "out.jxl",
                "--block-target",
                "residual",
            ])
            .is_err()
        );
        for (name, target) in [
            ("reference", BlockTarget::Reference),
            ("residual", BlockTarget::Residual),
            ("both", BlockTarget::Both),
        ] {
            for size in ["8", "16"] {
                let args = Args::try_parse_from([
                    "patch_lab",
                    "in.png",
                    "--output",
                    "out.jxl",
                    "--mode",
                    "source",
                    "--block-size",
                    size,
                    "--block-target",
                    name,
                    "--profile",
                    "--parallel-gather",
                ])
                .unwrap();
                assert_eq!(args.block_target, Some(target));
                assert!(args.profile);
                assert!(args.parallel_gather);
            }
        }
    }

    #[test]
    fn incompatible_detectors_fail_before_input_io() {
        for flags in [
            vec!["--off-grid"],
            vec!["--mode", "tiles", "--block-size", "16"],
            vec!["--mode", "components", "--block-size", "8"],
            vec!["--mode", "guarded", "--block-size", "16"],
            vec!["--block-size", "8", "--block-target", "reference"],
            vec!["--block-size", "16", "--block-target", "both"],
            vec!["--mode", "tiles", "--parallel-gather"],
            vec!["--mode", "components", "--parallel-gather"],
            vec!["--mode", "guarded", "--parallel-gather"],
            vec!["--mode", "tiles", "--previous-channels", "1"],
            vec!["--mode", "components", "--previous-channels", "0"],
            vec!["--mode", "guarded", "--previous-channels", "2"],
            vec!["--previous-channels", "1", "--channel-target", "reference"],
            vec!["--previous-channels", "2", "--channel-target", "both"],
        ] {
            let mut argv = vec!["patch_lab", "nonexistent.png", "--output", "unused.jxl"];
            argv.extend(flags);
            let message = run(Args::try_parse_from(argv).unwrap())
                .unwrap_err()
                .to_string();
            assert!(message.contains("requires --mode"), "{message}");
        }
    }

    #[test]
    fn channel_targets_preserve_defaults_and_resolve_independently() {
        for layout in [rs_jxl::PixelLayout::Gray, rs_jxl::PixelLayout::Rgb] {
            let channels = if layout == rs_jxl::PixelLayout::Gray {
                1
            } else {
                3
            };
            let image = Image::new(
                2304,
                1,
                layout,
                rs_jxl::PixelData::U8(vec![17; 2304 * channels]),
            )
            .unwrap();
            let params = CandidateParams::default().backend(9).unwrap();
            let base = EncodeOptions::default();
            for target in [
                BlockTarget::Residual,
                BlockTarget::Reference,
                BlockTarget::Both,
            ] {
                for previous in 0..=2 {
                    let mut options = adapter::project_native(&params, &image, &base)
                        .unwrap()
                        .exact_options()
                        .unwrap();
                    options.source_patch_reference = Some(
                        adapter::source_patch_reference_policy(
                            &params,
                            &image,
                            &base,
                            Default::default(),
                        )
                        .unwrap(),
                    );
                    let default_main = options.modular.previous_channels;
                    let original_sampling = options.modular.sampling.clone().unwrap();
                    let default_ref = options
                        .source_patch_reference
                        .as_ref()
                        .unwrap()
                        .resolve(&image)
                        .unwrap()
                        .modular
                        .previous_channels;
                    let before = channel_receipt(&options, &image, None, target).unwrap();
                    assert_eq!(before["main"]["configured"], false);
                    assert_eq!(before["reference"]["configured"], false);
                    configure_channels(&mut options, &params, &image, &base, previous, target)
                        .unwrap();
                    let effective = usize::from(previous).min(channels - 1);
                    let resolved = options
                        .source_patch_reference
                        .as_ref()
                        .unwrap()
                        .resolve(&image)
                        .unwrap();
                    assert_eq!(
                        options.modular.previous_channels,
                        if target.residual() {
                            effective
                        } else {
                            default_main
                        }
                    );
                    assert_eq!(
                        resolved.modular.previous_channels,
                        if target.reference() {
                            effective
                        } else {
                            default_ref
                        }
                    );
                    let sampling = options.modular.sampling.as_ref().unwrap();
                    assert_eq!(
                        sampling.previous_channels,
                        options.modular.previous_channels
                    );
                    assert_eq!(sampling.nb_repeats, original_sampling.nb_repeats);
                    assert_eq!(
                        sampling.max_channel_size,
                        original_sampling.max_channel_size
                    );
                    assert_eq!(sampling.node_threshold, original_sampling.node_threshold);
                    assert_eq!(
                        sampling.max_property_values,
                        original_sampling.max_property_values
                    );
                    assert_eq!(
                        sampling
                            .properties
                            .iter()
                            .filter(|&&p| p < 16)
                            .collect::<Vec<_>>(),
                        original_sampling
                            .properties
                            .iter()
                            .filter(|&&p| p < 16)
                            .collect::<Vec<_>>()
                    );
                    assert!(
                        sampling
                            .properties
                            .iter()
                            .all(|&p| p < 16 + 4 * sampling.previous_channels)
                    );
                    options.modular.validate().unwrap();
                    resolved.modular.validate().unwrap();
                    let receipt =
                        channel_receipt(&options, &image, Some(previous), target).unwrap();
                    assert_eq!(receipt["main"]["configured"], target.residual());
                    assert_eq!(receipt["reference"]["configured"], target.reference());
                    options.source_patch_reference = None;
                    configure_channels(&mut options, &params, &image, &base, previous, target)
                        .unwrap();
                    let gated = channel_receipt(&options, &image, Some(previous), target).unwrap();
                    assert_eq!(gated["reference"]["configured"], false);
                    assert!(gated["reference"]["effective_previous_channels"].is_null());
                }
            }
        }
        assert!(
            Args::try_parse_from([
                "patch_lab",
                "in.png",
                "--output",
                "out.jxl",
                "--channel-target",
                "residual"
            ])
            .is_err()
        );
    }

    #[test]
    fn block_targets_keep_projected_reference_and_residual_independent() {
        let image = Image::new(
            2304,
            1,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 2304 * 3]),
        )
        .unwrap();
        let params = CandidateParams::default().backend(9).unwrap();
        let base = EncodeOptions::default();
        for target in [
            BlockTarget::Reference,
            BlockTarget::Residual,
            BlockTarget::Both,
        ] {
            let mut options = adapter::project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap();
            options.source_patch_reference = Some(
                adapter::source_patch_reference_policy(&params, &image, &base, Default::default())
                    .unwrap(),
            );
            let residual = Arc::new(block_context::Receipt::default());
            let reference = Arc::new(block_context::Receipt::default());
            let block = block_context::Options {
                tile_size: 8,
                max_plan_leaves: 1234,
                receipt: Some(Arc::clone(&residual)),
            };
            configure_blocks(&mut options, target, &block, &reference);
            assert_eq!(options.modular.block_context.is_some(), target.residual());
            if let Some(block) = options.modular.block_context.as_ref() {
                assert!(Arc::ptr_eq(block.receipt.as_ref().unwrap(), &residual));
            }
            let policy = options.source_patch_reference.as_ref().unwrap();
            assert_eq!(policy.modular.block_context.is_some(), target.reference());
            // Deferred resolution must preserve the reference-only collector.
            let resolved = policy.resolve(&image).unwrap();
            assert_eq!(resolved.modular.block_context.is_some(), target.reference());
            if let Some(block) = resolved.modular.block_context.as_ref() {
                assert_eq!((block.tile_size, block.max_plan_leaves), (8, 1234));
                assert!(Arc::ptr_eq(block.receipt.as_ref().unwrap(), &reference));
                assert!(!Arc::ptr_eq(block.receipt.as_ref().unwrap(), &residual));
            }
        }
    }

    #[test]
    fn gated_reference_does_not_configure_whole_frame() {
        let mut options = EncodeOptions::default();
        let reference = Arc::new(block_context::Receipt::default());
        configure_blocks(
            &mut options,
            BlockTarget::Reference,
            &Default::default(),
            &reference,
        );
        assert!(options.modular.block_context.is_none());
        assert!(options.source_patch_reference.is_none());
        let work = targeted_block_work(&reference, false);
        assert_eq!(work["configured"], false);
        assert_eq!(work["used"], false);
        assert_eq!(work["trees"], 0);
        let work = targeted_block_work(&reference, true);
        assert_eq!(work["configured"], true);
        assert_eq!(work["used"], false);
        assert!(block_work(&reference).get("used").is_none());
        assert!(block_work(&reference).get("configured").is_none());
    }

    #[test]
    fn parallel_gather_is_independent_and_survives_reference_resolution() {
        let image = Image::new(
            2304,
            1,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 2304 * 3]),
        )
        .unwrap();
        let params = CandidateParams::default().backend(9).unwrap();
        let base = EncodeOptions::default();
        for (threads, waves) in [1, 8, 64, 256]
            .into_iter()
            .flat_map(|threads| [false, true].map(|waves| (threads, waves)))
        {
            let mut options = adapter::project_native(&params, &image, &base)
                .unwrap()
                .exact_options()
                .unwrap();
            options.source_patch_reference = Some(
                adapter::source_patch_reference_policy(&params, &image, &base, Default::default())
                    .unwrap(),
            );
            assert!(options.modular.source_gather.is_none());
            configure_parallel_gather(&mut options, threads, waves);
            let resolved = options
                .source_patch_reference
                .as_ref()
                .unwrap()
                .resolve(&image)
                .unwrap();
            for modular in [&options.modular, &resolved.modular] {
                let gather = modular.source_gather.as_ref().unwrap();
                assert_eq!(gather.workers, usize::from(threads).min(64));
                assert!(gather.sparse);
                assert_eq!(gather.stripes, 1);
                assert_eq!(
                    gather.mode == rs_jxl::modular::sample::GatherMode::Waves,
                    waves
                );
                assert!(!gather.cache_hashes && !gather.concat_batches);
                assert!(modular.block_context.is_none());
                modular.validate().unwrap();
            }
            options.source_patch_reference = None;
            configure_parallel_gather(&mut options, threads, waves);
            assert!(options.source_patch_reference.is_none());
        }
        let args = Args::try_parse_from([
            "patch_lab",
            "in.png",
            "--output",
            "out.jxl",
            "--parallel-gather",
        ])
        .unwrap();
        assert_eq!(args.mode, Mode::Off);
        assert!(args.block_size.is_none());
        assert!(
            Args::try_parse_from([
                "patch_lab",
                "in.png",
                "--output",
                "out.jxl",
                "--gather-waves",
            ])
            .is_err()
        );
    }
}
