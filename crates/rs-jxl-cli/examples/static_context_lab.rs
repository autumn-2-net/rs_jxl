//! Development-only static-context trials: one direct native encode, no portfolio.
use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions, Image,
    modular::{
        encode::receipt_cost::ModelCostReceipt,
        fast_context,
        ma_train::{
            LookaheadOptions, LookaheadReceipt, SeedMode, SeedOptions, SeedReceipt, SeedTopology,
        },
        refinement,
    },
    scheduler::{CandidateParams, adapter, params::Override},
};
use std::{fs, io::Write, path::PathBuf, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    Baseline,
    Auto12,
    Activity25,
    Activity81,
    Activity25Refine,
    Activity81Refine,
    Activity81Repair,
    OffsetDistribution,
    Auto12Offset,
    Full14Static,
    Full14StaticOffset,
    Lookahead2,
    Lookahead4,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Auto12 => "auto12",
            Self::Activity25 => "activity25",
            Self::Activity81 => "activity81",
            Self::Activity25Refine => "activity25-refine",
            Self::Activity81Refine => "activity81-refine",
            Self::Activity81Repair => "activity81-repair",
            Self::OffsetDistribution => "offset-distribution",
            Self::Auto12Offset => "auto12-offset",
            Self::Full14Static => "full14-static",
            Self::Full14StaticOffset => "full14-static-offset",
            Self::Lookahead2 => "lookahead2",
            Self::Lookahead4 => "lookahead4",
        }
    }

    fn native(self) -> Option<fast_context::Mode> {
        match self {
            Self::Auto12 | Self::Auto12Offset | Self::Full14Static | Self::Full14StaticOffset => {
                Some(fast_context::Mode::Auto12)
            }
            Self::Activity25 => Some(fast_context::Mode::Activity25),
            Self::Activity81 => Some(fast_context::Mode::Activity81),
            _ => None,
        }
    }

    fn seed(self) -> Option<(SeedTopology, SeedMode)> {
        match self {
            Self::Activity25Refine => Some((SeedTopology::Activity25, SeedMode::Refine)),
            Self::Activity81Refine => Some((SeedTopology::Activity81, SeedMode::Refine)),
            Self::Activity81Repair => Some((SeedTopology::Activity81, SeedMode::Repair)),
            _ => None,
        }
    }

    fn offsets(self) -> bool {
        matches!(
            self,
            Self::OffsetDistribution | Self::Auto12Offset | Self::Full14StaticOffset
        )
    }

    fn full_static(self) -> bool {
        matches!(self, Self::Full14Static | Self::Full14StaticOffset)
    }

    fn lookahead(self) -> Option<usize> {
        match self {
            Self::Lookahead2 => Some(2),
            Self::Lookahead4 => Some(4),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Patches {
    Off,
    Auto,
}

impl Patches {
    fn name(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Auto => "Auto",
        }
    }
}

#[derive(Default)]
struct Receipts {
    fast: Arc<fast_context::Receipt>,
    seed: Arc<SeedReceipt>,
    refinement: Arc<refinement::Receipt>,
    lookahead: Arc<LookaheadReceipt>,
}

#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(1..=10))]
    effort: u8,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    #[arg(long, value_enum, default_value = "baseline")]
    mode: Mode,
    #[arg(long, value_enum, default_value = "off")]
    patches: Patches,
    #[arg(long)]
    model_costs: bool,
}

fn project(args: &Args, image: &Image, receipts: &Receipts) -> Result<EncodeOptions> {
    if args.effort <= 3
        && (args.mode == Mode::OffsetDistribution || args.mode.lookahead().is_some())
    {
        return Err(format!(
            "{} requires effort 4..10; effort {} uses a fixed source MA tree, incompatible with this tool",
            args.mode.name(), args.effort
        ).into());
    }
    let mut base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    base.modular.fast_context = args.mode.native().map(|mode| fast_context::Options {
        mode,
        bias: if args.mode.full_static() {
            fast_context::Bias::Static
        } else {
            fast_context::Bias::Off
        },
        predictors: if args.mode.full_static() {
            fast_context::PredictorSet::Full
        } else {
            fast_context::PredictorSet::Basic
        },
        receipt: Some(Arc::clone(&receipts.fast)),
        ..Default::default()
    });
    base.modular.source_ma_seed = args.mode.seed().map(|(topology, mode)| SeedOptions {
        topology,
        mode,
        receipt: Some(Arc::clone(&receipts.seed)),
        ..Default::default()
    });
    base.modular.refinement = args
        .mode
        .offsets()
        .then(|| refinement::Options {
            offsets: Some(refinement::OffsetOptions {
                mode: refinement::OffsetMode::Distribution,
                ..Default::default()
            }),
            receipt: Some(Arc::clone(&receipts.refinement)),
            ..Default::default()
        })
        .unwrap_or_default();
    base.modular.source_ma_lookahead = args.mode.lookahead().map(|width| LookaheadOptions {
        width,
        min_records: 8192,
        relative_window: 0.01,
        receipt: Some(Arc::clone(&receipts.lookahead)),
    });
    if args.model_costs {
        base.modular.model_cost_receipt = Some(Arc::new(ModelCostReceipt::default()));
    }
    let mut params = CandidateParams::default().backend(args.effort)?;
    params.patches = match args.patches {
        Patches::Off => Override::Off,
        Patches::Auto => Override::Default,
    };
    let mut options = adapter::project_native(&params, image, &base)?.exact_options()?;
    if args.patches == Patches::Off {
        options.patches = None;
        options.source_patches = None;
        options.source_patch_reference = None;
        options.tile_patches = None;
    }
    if options.modular.previous_channels != 0 {
        return Err("static-context lab requires previous_channels=0".into());
    }
    Ok(options)
}

fn patches_cleared(options: &EncodeOptions) -> bool {
    options.patches.is_none()
        && options.source_patches.is_none()
        && options.source_patch_reference.is_none()
        && options.tile_patches.is_none()
}

fn fast_parameters(fast: &fast_context::Options) -> serde_json::Value {
    serde_json::json!({
        "mode": match fast.mode {
            fast_context::Mode::Leaf => "leaf",
            fast_context::Mode::Gradient12 => "gradient12",
            fast_context::Mode::Directional12 => "directional12",
            fast_context::Mode::Auto12 => "auto12",
            fast_context::Mode::Activity25 => "activity25",
            fast_context::Mode::Activity81 => "activity81",
        },
        "bias": match fast.bias {
            fast_context::Bias::Off => "off",
            fast_context::Bias::Static => "static",
            fast_context::Bias::Adaptive => "adaptive",
        },
        "predictors": match fast.predictors {
            fast_context::PredictorSet::Basic => "basic",
            fast_context::PredictorSet::Full => "full",
        },
        "scan": match fast.scan {
            fast_context::Scan::Auto => "auto",
            fast_context::Scan::Scalar => "scalar",
            fast_context::Scan::Avx2 => "avx2",
            fast_context::Scan::Avx512 => "avx512",
        },
        "cross_channel": fast.cross_channel,
    })
}

fn measurement(
    args: &Args,
    image: &Image,
    options: &EncodeOptions,
    receipts: &Receipts,
    size: usize,
    seconds: f64,
) -> serde_json::Value {
    let fast_options = options.modular.fast_context.as_ref().map(fast_parameters);
    let fast = fast_options.as_ref().map(|settings| {
        let work = receipts.fast.snapshot();
        serde_json::json!({
            "mode": settings["mode"], "bias": settings["bias"], "scan": settings["scan"],
            "predictors": settings["predictors"], "cross_channel": settings["cross_channel"],
            "trees": work.trees, "leaves": work.leaves, "pixels": work.pixels,
            "offset_leaves": work.offset_leaves, "horizontal": work.horizontal,
            "vertical": work.vertical, "single": work.single, "joint": work.joint,
            "cross_channel_plans": work.cross_channel,
        })
    });
    let model_costs = options.modular.model_cost_receipt.as_ref().map(|cost| {
        serde_json::json!({
            "tree_bits": cost.tree_bits(), "entropy_header_bits": cost.entropy_header_bits(),
            "total_bits": cost.bits(),
        })
    });
    let seed_options = options.modular.source_ma_seed.as_ref().map(|seed| {
        serde_json::json!({
            "topology": match seed.topology {
                SeedTopology::Quantile => "quantile",
                SeedTopology::Activity25 => "activity25",
                SeedTopology::Activity81 => "activity81",
            },
            "mode": match seed.mode {
                SeedMode::Only => "only",
                SeedMode::Refine => "refine",
                SeedMode::Repair => "repair",
            },
            // Fixed activity-grid topologies ignore the quantile seed depth.
            "depth": seed.depth,
        })
    });
    let seed = options.modular.source_ma_seed.as_ref().map(|_| {
        let work = receipts.seed.snapshot();
        serde_json::json!({
            "learners": work.learners, "prepared_records": work.prepared_records,
            "seed_splits": work.seed_splits, "refined_leaves": work.refined_leaves,
            "repaired_branches": work.repaired_branches, "merged_branches": work.merged_branches,
            "predictor_changes": work.predictor_changes, "histogram_reuses": work.histogram_reuses,
        })
    });
    let offset_options = options.modular.refinement.offsets.as_ref().map(|offsets| {
        serde_json::json!({
            "mode": match offsets.mode {
                refinement::OffsetMode::Mean => "mean",
                refinement::OffsetMode::Median => "median",
                refinement::OffsetMode::Entropy => "entropy",
                refinement::OffsetMode::Distribution => "distribution",
                refinement::OffsetMode::DistributionSinglePass => "distribution-single-pass",
            },
            "samples_per_leaf": offsets.samples_per_leaf,
        })
    });
    let refinement = options.modular.refinement.receipt.as_ref().map(|receipt| {
        let work = receipt.snapshot();
        serde_json::json!({
            "entropy": options.modular.refinement.entropy, "leaves": options.modular.refinement.leaves,
            "context_map": options.modular.refinement.context_map.is_some(),
            "offset_mode": offset_options.as_ref().map(|o| &o["mode"]),
            "offset_sample_limit": offset_options.as_ref().map(|o| &o["samples_per_leaf"]),
            "domains": work.domains, "models": work.models,
            "refit_proposals": work.refit_proposals, "refit_wins": work.refit_wins,
            "offset_proposals": work.offset_proposals, "offset_wins": work.offset_wins,
            "offset_leaves": work.offset_leaves, "offset_samples": work.offset_samples,
            "offset_population": work.offset_population, "offset_candidates": work.offset_candidates,
            "offset_seconds": work.offset_seconds, "entropy_wins": work.entropy_wins,
            "saved_bits": work.saved_bits, "refit_seconds": work.refit_seconds,
            "model_seconds": work.model_seconds, "write_seconds": work.write_seconds,
        })
    });
    let lookahead_options = options.modular.source_ma_lookahead.as_ref().map(|look| {
        serde_json::json!({
            "width": look.width, "min_records": look.min_records, "relative_window": look.relative_window,
        })
    });
    let lookahead = options.modular.source_ma_lookahead.as_ref().map(|_| {
        let work = receipts.lookahead.snapshot();
        serde_json::json!({
            "roots": work.roots, "eligible": work.eligible, "probed_candidates": work.probed_candidates,
            "child_searches": work.child_searches, "changed": work.changed,
            "estimated_saved_bits": work.estimated_saved_bits,
        })
    });
    let patch_options = serde_json::json!({
        "legacy": options.patches.is_some(), "source": options.source_patches.is_some(),
        "source_reference": options.source_patch_reference.is_some(), "tiles": options.tile_patches.is_some(),
    });
    let options_signature = serde_json::json!({
        "schema": "static_context_lab.options.v1",
        "mode": args.mode.name(), "effort": args.effort, "threads": args.threads,
        "patches": args.patches.name(), "patch_options": patch_options,
        "previous_channels": options.modular.previous_channels,
        "predictor": format!("{:?}", options.modular.predictor),
        "fixed_tree": options.modular.fixed_tree.map(|tree| format!("{tree:?}")),
        "preset_fixed_tree_overridden": args.effort <= 3
            && (options.modular.fast_context.is_some() || options.modular.source_ma_seed.is_some()),
        "sampling": options.modular.sampling.as_ref().map(|p| serde_json::json!({
            "properties": p.properties, "max_property_values": p.max_property_values,
            "nb_repeats": p.nb_repeats, "tree_mode": format!("{:?}", p.tree_mode),
            "predictor_override": p.predictor_override.map(|v| format!("{v:?}")),
        })),
        "tree_learning": format!("{:?}", options.modular.tree_learning),
        "rct": options.modular.rct, "rct_candidates": options.modular.rct_candidates,
        "max_tree_leaves": options.modular.max_tree_leaves,
        "max_merged_tree_leaves": options.modular.max_merged_tree_leaves,
        "fast_context": fast_options, "seed": seed_options,
        "offsets": offset_options, "lookahead": lookahead_options,
        "refine_entropy": options.modular.refinement.entropy,
        "refine_leaves": options.modular.refinement.leaves,
        "refine_context_map": options.modular.refinement.context_map.is_some(),
    });
    let pixels = f64::from(image.width()) * f64::from(image.height());
    serde_json::json!({
            "bytes": size, "seconds": seconds,
            "bpp": size as f64 * 8.0 / pixels,
            "mpix_per_second": pixels / 1e6 / seconds,
            "width": image.width(), "height": image.height(),
            "mode": args.mode.name(), "effort": args.effort, "threads": args.threads,
            "previous_channels": options.modular.previous_channels,
            "patches": args.patches.name(), "patches_cleared": patches_cleared(options),
            "patch_options": patch_options, "options_signature": options_signature,
            "search": false, "native_encode_calls": 1, "fast_context": fast,
            "seed": seed, "seed_options": seed_options,
            "offset_options": offset_options, "refinement": refinement,
            "lookahead_options": lookahead_options, "lookahead": lookahead,
            "lookahead_score_contract": "estimated local scoring only; not emitted wire bits",
            "refinement_cost_contract": "saved_bits compares serialized Modular domains, not whole-file savings; diagnostic times may overlap",
            "model_costs": model_costs,
            "model_cost_contract": "emitted MA tree and entropy-header bits only; not estimated scores",
            "timing": "native projection+analysis+encode; excludes startup and input/output IO",
    })
}

fn run(args: Args) -> Result<()> {
    if args.output.exists() {
        return Err("benchmark output already exists".into());
    }
    let decoded = rs_jxl_image::decode(&fs::read(&args.input)?, true)?;
    let image = decoded.image;
    let receipts = Receipts::default();
    // Charge projection and all native analysis to this single invocation.
    let start = Instant::now();
    let options = project(&args, &image, &receipts)?;
    let bytes = rs_jxl::encode(&image, &options)?;
    let seconds = start.elapsed().as_secs_f64();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("invalid native encode duration".into());
    }
    let report = measurement(&args, &image, &options, &receipts, bytes.len(), seconds);
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?
        .write_all(&bytes)?;
    println!("{report}");
    if let Some(profile) = rs_jxl::profile::report() {
        eprint!("{profile}");
    }
    Ok(())
}

fn main() -> Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argument_defaults_and_bounds() {
        let argv = ["static_context_lab", "in.png", "--output", "out.jxl"];
        let args = Args::try_parse_from(argv).unwrap();
        assert_eq!(
            (args.effort, args.threads, args.mode),
            (9, 8, Mode::Baseline)
        );
        assert!(!args.model_costs);
        assert_eq!(args.patches, Patches::Off);
        assert_eq!(Mode::value_variants().len(), 13);
        for &mode in Mode::value_variants() {
            let parsed =
                Args::try_parse_from(argv.into_iter().chain(["--mode", mode.name()])).unwrap();
            assert_eq!(parsed.mode, mode);
        }
        for effort in 1..=10 {
            let parsed =
                Args::try_parse_from(argv.into_iter().chain(["--effort", &effort.to_string()]))
                    .unwrap();
            assert_eq!(parsed.effort, effort);
        }
        assert_eq!(
            Args::try_parse_from(argv.into_iter().chain(["--patches", "auto"]))
                .unwrap()
                .patches,
            Patches::Auto
        );
        for (flag, value) in [
            ("--effort", "0"),
            ("--effort", "11"),
            ("--threads", "0"),
            ("--threads", "257"),
            ("--mode", "activity49"),
            ("--mode", "greedy"),
            ("--patches", "on"),
        ] {
            assert!(Args::try_parse_from(argv.into_iter().chain([flag, value])).is_err());
        }
        assert!(Args::try_parse_from(["static_context_lab", "in.png"]).is_err());
    }

    #[test]
    fn direct_projection_retains_modes_defaults_receipts_and_common_policy() {
        let image = Image::new(
            128,
            128,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 128 * 128 * 3]),
        )
        .unwrap();
        for effort in 1..=10 {
            for mode in [
                Mode::Baseline,
                Mode::Auto12,
                Mode::Activity25,
                Mode::Activity81,
            ] {
                for costs in [false, true] {
                    let args = Args {
                        input: "in.png".into(),
                        output: "out.jxl".into(),
                        effort,
                        threads: 256,
                        mode,
                        patches: Patches::Off,
                        model_costs: costs,
                    };
                    let receipts = Receipts::default();
                    let options = project(&args, &image, &receipts).unwrap();
                    assert_eq!(options.threads, 256);
                    assert_eq!(options.modular.previous_channels, 0);
                    assert!(patches_cleared(&options));
                    assert!(options.modular.sampling.is_some());
                    let expected_fixed = if mode != Mode::Baseline {
                        None
                    } else {
                        match effort {
                            1 | 2 => Some(rs_jxl::modular::encode::FixedTree::Gradient),
                            3 => Some(rs_jxl::modular::encode::FixedTree::Weighted),
                            _ => None,
                        }
                    };
                    assert_eq!(options.modular.fixed_tree, expected_fixed);
                    assert!(options.modular.source_ma_lookahead.is_none());
                    assert!(options.modular.source_ma_seed.is_none());
                    assert_eq!(options.modular.model_cost_receipt.is_some(), costs);
                    assert_eq!(
                        options.modular.fast_context.is_some(),
                        mode != Mode::Baseline
                    );
                    if let Some(fast) = options.modular.fast_context {
                        assert_eq!(Some(fast.mode), mode.native());
                        assert_eq!(fast.bias, fast_context::Bias::Off);
                        assert_eq!(fast.predictors, fast_context::PredictorSet::Basic);
                        assert_eq!(fast.scan, fast_context::Scan::Auto);
                        assert!(!fast.cross_channel);
                        assert!(Arc::ptr_eq(fast.receipt.as_ref().unwrap(), &receipts.fast));
                    }
                }
            }
        }
    }

    #[test]
    fn activity_seed_projection_retains_unrestricted_effort_policy() {
        let image = Image::new(
            128,
            128,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 128 * 128 * 3]),
        )
        .unwrap();
        let receipts = Receipts::default();
        for effort in 1..=10 {
            let mut args = Args {
                input: "in.png".into(),
                output: "out.jxl".into(),
                effort,
                threads: 1,
                mode: Mode::Baseline,
                patches: Patches::Off,
                model_costs: false,
            };
            let baseline = project(&args, &image, &receipts).unwrap();
            for mode in [
                Mode::Activity25Refine,
                Mode::Activity81Refine,
                Mode::Activity81Repair,
            ] {
                args.mode = mode;
                let options = project(&args, &image, &receipts).unwrap();
                let modular = &options.modular;
                assert!(modular.fast_context.is_none());
                assert!(modular.fixed_tree.is_none() && modular.custom_tree.is_none());
                assert!(modular.source_ma_lookahead.is_none());
                assert!(patches_cleared(&options));
                assert_eq!(modular.previous_channels, 0);
                assert_eq!(modular.predictor, baseline.modular.predictor);
                assert_eq!(modular.max_tree_leaves, baseline.modular.max_tree_leaves);
                assert_eq!(
                    modular.max_merged_tree_leaves,
                    baseline.modular.max_merged_tree_leaves
                );
                assert_eq!(modular.ma_budget_mode, baseline.modular.ma_budget_mode);
                assert_eq!(modular.source_ma_search, baseline.modular.source_ma_search);
                assert_eq!(modular.tree_learning, baseline.modular.tree_learning);
                let policy = modular.sampling.as_ref().unwrap();
                let baseline_policy = baseline.modular.sampling.as_ref().unwrap();
                assert_eq!(policy.properties, baseline_policy.properties);
                assert_eq!(
                    policy.max_property_values,
                    baseline_policy.max_property_values
                );
                assert_eq!(policy.nb_repeats, baseline_policy.nb_repeats);
                assert_eq!(policy.tree_mode, baseline_policy.tree_mode);
                assert_eq!(
                    policy.predictor_override,
                    baseline_policy.predictor_override
                );
                if effort == 9 {
                    assert_eq!(
                        modular.predictor,
                        rs_jxl::modular::predictor::Predictor::Best
                    );
                    assert_eq!(policy.max_property_values, 256);
                    assert_eq!(policy.properties.len(), 16);
                    assert!((0..16).all(|p| policy.properties.contains(&p)));
                    assert!(policy.predictor_override.is_none());
                    assert_eq!(policy.tree_mode, rs_jxl::modular::sample::TreeMode::Default);
                }
                let seed = modular.source_ma_seed.as_ref().unwrap();
                assert_eq!(Some((seed.topology, seed.mode)), mode.seed());
                assert_eq!(seed.depth, SeedOptions::default().depth);
                assert_eq!(seed.split_bits, SeedOptions::default().split_bits);
                assert!(Arc::ptr_eq(seed.receipt.as_ref().unwrap(), &receipts.seed));
            }
        }
    }

    #[test]
    fn offset_full14_and_lookahead_project_requested_tools_without_portfolio() {
        let image = Image::new(
            128,
            128,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 128 * 128 * 3]),
        )
        .unwrap();
        let receipts = Receipts::default();
        for effort in 1..=10 {
            for mode in [
                Mode::OffsetDistribution,
                Mode::Auto12Offset,
                Mode::Full14Static,
                Mode::Full14StaticOffset,
                Mode::Lookahead2,
                Mode::Lookahead4,
            ] {
                let args = Args {
                    input: "in.png".into(),
                    output: "out.jxl".into(),
                    effort,
                    threads: 1,
                    mode,
                    patches: Patches::Off,
                    model_costs: false,
                };
                if effort <= 3 && (mode == Mode::OffsetDistribution || mode.lookahead().is_some()) {
                    for patches in [Patches::Off, Patches::Auto] {
                        let mut rejected = Args::try_parse_from([
                            "static_context_lab",
                            "in.png",
                            "--output",
                            "out.jxl",
                        ])
                        .unwrap();
                        rejected.mode = mode;
                        rejected.effort = effort;
                        rejected.patches = patches;
                        let error = project(&rejected, &image, &receipts)
                            .unwrap_err()
                            .to_string();
                        assert!(
                            error.contains(mode.name()) && error.contains("requires effort 4..10")
                        );
                        assert!(error.contains("fixed source MA tree"));
                    }
                    continue;
                }
                let options = project(&args, &image, &receipts).unwrap();
                let modular = &options.modular;
                assert!(modular.fixed_tree.is_none());
                assert!(modular.source_ma_seed.is_none());
                assert_eq!(modular.fast_context.is_some(), mode.native().is_some());
                assert_eq!(modular.refinement.offsets.is_some(), mode.offsets());
                assert!(!modular.refinement.entropy && !modular.refinement.leaves);
                assert!(modular.refinement.context_map.is_none());
                if let Some(fast) = &modular.fast_context {
                    assert_eq!(fast.mode, fast_context::Mode::Auto12);
                    assert_eq!(
                        fast.predictors,
                        if mode.full_static() {
                            fast_context::PredictorSet::Full
                        } else {
                            fast_context::PredictorSet::Basic
                        }
                    );
                    assert_eq!(
                        fast.bias,
                        if mode.full_static() {
                            fast_context::Bias::Static
                        } else {
                            fast_context::Bias::Off
                        }
                    );
                    assert!(!fast.cross_channel);
                }
                if let Some(offsets) = &modular.refinement.offsets {
                    assert_eq!(offsets.mode, refinement::OffsetMode::Distribution);
                    assert_eq!(offsets.samples_per_leaf, 1024);
                    assert!(Arc::ptr_eq(
                        modular.refinement.receipt.as_ref().unwrap(),
                        &receipts.refinement
                    ));
                }
                assert_eq!(
                    modular.source_ma_lookahead.is_some(),
                    mode.lookahead().is_some()
                );
                if let Some(look) = &modular.source_ma_lookahead {
                    assert_eq!(Some(look.width), mode.lookahead());
                    assert_eq!(look.min_records, 8192);
                    assert_eq!(look.relative_window, 0.01);
                    assert!(Arc::ptr_eq(
                        look.receipt.as_ref().unwrap(),
                        &receipts.lookahead
                    ));
                }
                assert_eq!(modular.previous_channels, 0);
                assert!(modular.sampling.is_some() && patches_cleared(&options));
                let report = measurement(&args, &image, &options, &receipts, 123, 0.5);
                assert_eq!(
                    report["options_signature"]["fixed_tree"],
                    serde_json::Value::Null
                );
                assert_eq!(
                    report["options_signature"]["preset_fixed_tree_overridden"],
                    effort <= 3
                );
            }
        }
    }

    #[test]
    fn auto_patch_policy_is_not_cleared_after_source_projection() {
        let image = Image::new(
            128,
            128,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 128 * 128 * 3]),
        )
        .unwrap();
        let receipts = Receipts::default();
        for effort in 1..=10 {
            let args = Args {
                input: "in.png".into(),
                output: "out.jxl".into(),
                effort,
                threads: 1,
                mode: Mode::Baseline,
                patches: Patches::Auto,
                model_costs: false,
            };
            let options = project(&args, &image, &receipts).unwrap();
            let params = CandidateParams::default().backend(effort).unwrap();
            assert_eq!(params.patches, Override::Default);
            let projected = adapter::project_native(&params, &image, &EncodeOptions::default())
                .unwrap()
                .exact_options()
                .unwrap();
            assert_eq!(options.patches.is_some(), projected.patches.is_some());
            assert_eq!(
                options.source_patches.is_some(),
                projected.source_patches.is_some()
            );
            assert_eq!(
                options.source_patch_reference.is_some(),
                projected.source_patch_reference.is_some()
            );
            assert_eq!(
                options.tile_patches.is_some(),
                projected.tile_patches.is_some()
            );
            let report = measurement(&args, &image, &options, &receipts, 123, 0.5);
            assert_eq!(report["patches"], "Auto");
            assert_eq!(report["patches_cleared"], patches_cleared(&projected));
            assert_eq!(
                report["patch_options"]["source"],
                options.source_patches.is_some()
            );
            if effort == 9 {
                assert!(
                    options.source_patches.is_some() && options.source_patch_reference.is_some()
                );
                assert!(!patches_cleared(&options));
            }
        }
    }

    #[test]
    fn mode_signatures_and_nullable_work_receipts_are_distinct() {
        let image = Image::new(
            128,
            128,
            rs_jxl::PixelLayout::Rgb,
            rs_jxl::PixelData::U8(vec![17; 128 * 128 * 3]),
        )
        .unwrap();
        let receipts = Receipts::default();
        let mut signatures = std::collections::BTreeSet::new();
        for &mode in Mode::value_variants() {
            let args = Args {
                input: "in.png".into(),
                output: "out.jxl".into(),
                effort: 9,
                threads: 1,
                mode,
                patches: Patches::Off,
                model_costs: true,
            };
            let options = project(&args, &image, &receipts).unwrap();
            let report = measurement(&args, &image, &options, &receipts, 123, 0.5);
            assert_eq!(report["mode"], mode.name());
            assert_eq!(report["native_encode_calls"], 1);
            assert_eq!(report["search"], false);
            assert_eq!(report["bytes"], 123);
            assert_eq!(report["seconds"], 0.5);
            assert_eq!(report["patches"], "Off");
            assert_eq!(report["patches_cleared"], true);
            assert_eq!(report["fast_context"].is_null(), mode.native().is_none());
            assert_eq!(report["seed"].is_null(), mode.seed().is_none());
            assert_eq!(report["seed_options"].is_null(), mode.seed().is_none());
            assert_eq!(report["offset_options"].is_null(), !mode.offsets());
            assert_eq!(report["refinement"].is_null(), !mode.offsets());
            assert_eq!(report["lookahead"].is_null(), mode.lookahead().is_none());
            assert_eq!(
                report["lookahead_options"].is_null(),
                mode.lookahead().is_none()
            );
            assert_eq!(
                report["options_signature"]["schema"],
                "static_context_lab.options.v1"
            );
            assert_eq!(
                report["options_signature"]["preset_fixed_tree_overridden"],
                false
            );
            assert_eq!(report["model_costs"]["total_bits"], 0);
            if mode.offsets() {
                assert_eq!(report["offset_options"]["mode"], "distribution");
                assert_eq!(report["refinement"]["offset_sample_limit"], 1024);
                assert_eq!(report["refinement"]["offset_proposals"], 0);
            }
            if mode.native() == Some(fast_context::Mode::Auto12) {
                assert_eq!(report["fast_context"]["mode"], "auto12");
                assert_eq!(
                    report["fast_context"]["bias"],
                    if mode.full_static() { "static" } else { "off" }
                );
                assert_eq!(
                    report["fast_context"]["predictors"],
                    if mode.full_static() { "full" } else { "basic" }
                );
            }
            assert!(
                signatures.insert(serde_json::to_string(&report["options_signature"]).unwrap())
            );
            let auto_args = Args {
                patches: Patches::Auto,
                ..args
            };
            let auto = project(&auto_args, &image, &receipts).unwrap();
            assert!(auto.source_patches.is_some() && auto.source_patch_reference.is_some());
            let auto_report = measurement(&auto_args, &image, &auto, &receipts, 123, 0.5);
            assert_eq!(auto_report["patches"], "Auto");
            assert_eq!(auto_report["patches_cleared"], false);
            assert_ne!(
                auto_report["options_signature"],
                report["options_signature"]
            );
        }
        assert_eq!(signatures.len(), 13);
    }
}
