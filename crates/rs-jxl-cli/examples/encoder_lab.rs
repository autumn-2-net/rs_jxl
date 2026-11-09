//! Development-only candidate/scout experiments; not part of the installed CLI.
use rs_jxl_image as input;

#[path = "../src/options.rs"]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "This example shares MA budget/pruning parsers, not the manual Tools::apply adapter."
    )
)]
mod options;

#[cfg(test)]
#[path = "../src/encoder_lab_budget_parser_tests.rs"]
mod budget_parser_tests;

use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions,
    modular::{
        ma_train::{DenseHistogram, DescriptionCost, HistogramOptions, PropertyBins},
        predictor::Predictor,
        sample::GatherOptions,
    },
    scheduler::{
        CandidateParams, DriverOptions, NativeBackendOptions,
        native::encode_native_candidate,
        params::{Override, lossless_probe_plan},
        pilot::{self, PilotOptions},
        run_native_pixel_search,
    },
    tools::lz77::{
        SourceBlockMode, SourceBucketSelection, SourceConfig, SourceIndexMode, SourceIndexTags,
        SourceMatchCost, SourceParallelism, SourceSuffixMode,
    },
};
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Family {
    Automatic,
    Neighborhood,
    Literal,
    Cross0,
    Cross6,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Histograms {
    Default,
    Bins,
    Dense,
    DenseBins,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum Description {
    #[default]
    Off,
    Quarter,
    Half,
    Full,
}

impl From<Description> for DescriptionCost {
    fn from(value: Description) -> Self {
        match value {
            Description::Off => Self::Disabled,
            Description::Quarter => Self::Quarter,
            Description::Half => Self::Half,
            Description::Full => Self::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Matcher {
    Source,
    Buckets,
    Indexed,
    Suffix,
    Routed,
}

fn parse_context_side(value: &str) -> std::result::Result<u16, String> {
    match value {
        "256" => Ok(256),
        "512" => Ok(512),
        _ => Err("pilot-context-side must be 256 or 512".into()),
    }
}

fn parse_quantile_splits(value: &str) -> std::result::Result<u8, String> {
    match value {
        "32" => Ok(32),
        "64" => Ok(64),
        _ => Err("quantile-splits must be 32 or 64".into()),
    }
}

#[derive(Parser)]
struct Args {
    input: PathBuf,
    #[command(flatten)]
    ma_budget: options::MaBudgetOptions,
    #[command(flatten)]
    ma_pruning: options::MaPruningOptions,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "automatic")]
    family: Family,
    /// Override the RCT for a fixed-plan measurement, without the outer search.
    #[arg(long, conflicts_with = "search", value_parser = clap::value_parser!(u8).range(0..=41))]
    rct: Option<u8>,
    /// Override the source MA sample fraction for a direct candidate experiment.
    #[arg(long)]
    learning_fraction: Option<f32>,
    /// Override pre-transform channel compaction for a direct candidate.
    #[arg(long)]
    pre_colors: Option<f32>,
    /// Explicit lossless palette permutation (source modes 0..9).
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=9))]
    palette_order: Option<u8>,
    /// Source buffering mode: -1 automatic, 0 full-frame, 1 large, 2 groupwise.
    #[arg(long, allow_hyphen_values = true, value_parser = clap::value_parser!(i8).range(-1..=2))]
    buffering: Option<i8>,
    #[arg(long)]
    force_palette: bool,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..=256))]
    threads: u16,
    #[arg(long, default_value_t = 9, value_parser = clap::value_parser!(u8).range(1..=10))]
    effort: u8,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=2))]
    lz: u8,
    #[arg(long, value_enum, default_value = "default")]
    histograms: Histograms,
    /// Experimental earlier cache construction for 14-predictor MA searches.
    #[arg(long)]
    early_cache_14: bool,
    #[arg(long)]
    search: bool,
    #[arg(long, requires = "search")]
    pilot_first: bool,
    #[arg(long, requires = "search", conflicts_with_all = ["pilot_first", "model_adjusted", "consensus", "staged_probes", "probe_palette", "wide_probes", "context_side", "matcher"])]
    content_pilot: bool,
    /// Use source palette defaults instead of forced Morton literal tools.
    #[arg(long, requires = "content_pilot")]
    pilot_source_palette: bool,
    /// Enable the experimental pilot family tournament and confirmation.
    #[arg(long, requires = "content_pilot")]
    pilot_family_tournament: bool,
    #[arg(long, requires = "content_pilot")]
    pilot_single_stage: bool,
    #[arg(long, requires = "content_pilot")]
    pilot_independent_probes: bool,
    #[arg(long, requires = "content_pilot")]
    pilot_parallel_initial: bool,
    #[arg(long, requires = "content_pilot")]
    pilot_quiet_arena: bool,
    #[arg(long, requires = "content_pilot")]
    pilot_overlap_automatic: bool,
    #[arg(long, requires = "content_pilot")]
    pilot_quarter_samples: bool,
    /// Experimental rejection of consistently weak near-tie pilot alternatives.
    #[arg(long, requires = "content_pilot")]
    pilot_near_tie_consensus: bool,
    /// Experimental square fine probes for large images.
    #[arg(long, requires = "content_pilot", value_parser = parse_context_side)]
    pilot_context_side: Option<u16>,
    /// Race and resume actual original-image groups, with ordinary fallback.
    #[arg(
        long,
        requires = "content_pilot",
        conflicts_with = "pilot_context_side"
    )]
    pilot_progressive_groups: bool,
    /// Disable the cheap content router for controlled group-race experiments.
    #[arg(long, requires = "pilot_progressive_groups")]
    pilot_progressive_all_content: bool,
    /// Re-test every real-group candidate, ignoring cheap-front admission.
    #[arg(long, requires = "pilot_progressive_groups")]
    pilot_progressive_all_candidates: bool,
    /// Print source scan evidence without running any encoder candidate.
    #[arg(long, conflicts_with_all = ["output", "search"])]
    scan_only: bool,
    #[arg(long, requires = "pilot_first", conflicts_with = "wide_probes", value_parser = clap::value_parser!(u16).range(1..=1024))]
    context_side: Option<u16>,
    #[arg(long)]
    wide_probes: bool,
    #[arg(long)]
    count_log: bool,
    #[arg(long)]
    clipped: bool,
    /// Experimental unclipped incremental Shannon scoring, not the source default.
    #[arg(long, conflicts_with = "clipped")]
    shannon: bool,
    /// Experimental histogram model regularization; not final clustered wire bits.
    #[arg(long, value_enum, default_value = "off")]
    description_cost: Description,
    /// Measure emitted tree/model bits for one direct candidate, not search probes.
    #[arg(long, conflicts_with_all = ["search", "scan_only"])]
    model_costs: bool,
    /// Experimental complete shallow MA plan; no exhaustive suffix learning.
    #[arg(long, value_parser = ["leaf", "gradient12", "directional12", "auto12"], conflicts_with_all = ["search", "scan_only"])]
    fast_context: Option<String>,
    /// Encoder-only static offset fitting, with an optional causal bias proposal.
    #[arg(long, value_parser = ["off", "static", "adaptive"], requires = "fast_context")]
    context_bias: Option<String>,
    #[arg(long, value_parser = ["auto", "scalar", "avx2", "avx512"], requires = "fast_context")]
    context_scan: Option<String>,
    /// Score Basic4 or all fourteen standard predictors, including native WP.
    #[arg(long, value_parser = ["basic", "full"], requires = "fast_context")]
    context_predictors: Option<String>,
    /// Shared-preparation shallow seed, optionally refined or re-evaluated.
    #[arg(long, value_parser = ["only", "refine", "repair"], conflicts_with_all = ["fast_context", "search", "scan_only", "ma_histogram_bins"])]
    ma_seed: Option<String>,
    /// Initial depth only; refinement retains the normal final tree budget.
    #[arg(long, requires = "ma_seed", value_parser = clap::value_parser!(u8).range(1..=8))]
    ma_seed_depth: Option<u8>,
    /// Score coarse MA split quantiles, then refine their strongest valleys.
    #[arg(long)]
    refine_quantiles: bool,
    /// Experimental coarse-only MA split thresholds.
    #[arg(long, value_parser = parse_quantile_splits, conflicts_with = "refine_quantiles")]
    quantile_splits: Option<u8>,
    /// Build node-local weighted histograms with fewer bins (experimental).
    #[arg(long, value_parser = clap::value_parser!(u16).range(2..=256), conflicts_with_all = ["refine_quantiles", "quantile_splits"])]
    ma_histogram_bins: Option<u16>,
    /// Share immutable preparation across full candidates, bounded in MiB.
    #[arg(long, requires = "search", conflicts_with = "representations", value_parser = clap::value_parser!(u16).range(1..=1024))]
    reuse_preparation_mib: Option<u16>,
    #[arg(long)]
    chain_batch: bool,
    #[arg(long, requires = "wide_probes", value_parser = clap::value_parser!(u16).range(1..=1024))]
    strip_rows: Option<u16>,
    #[arg(long, value_enum, default_value = "source")]
    matcher: Matcher,
    #[arg(long)]
    events: bool,
    #[arg(long, requires = "search")]
    probe_palette: bool,
    #[arg(long, requires = "search")]
    representations: bool,
    #[arg(long)]
    subtrees: bool,
    /// Experimental dynamic MA subtree scheduling.
    #[arg(long, requires = "subtrees")]
    dynamic_subtrees: bool,
    /// Experimental MA subtree histogram inheritance.
    #[arg(long, requires = "subtrees")]
    subtree_histograms: bool,
    #[arg(long, requires = "subtrees", value_parser = clap::value_parser!(u16).range(1..=64))]
    ma_workers: Option<u16>,
    #[arg(long)]
    dedup: bool,
    #[arg(long)]
    sparse: bool,
    /// Experimental bounded MA collector hash cache; disabled by default.
    #[arg(long)]
    hash_cache: bool,
    /// Experimental low-merge, bounded final sample-column concatenation.
    #[arg(long)]
    concat_batches: bool,
    #[arg(long)]
    parallel_wp: bool,
    #[arg(long)]
    fast_ranges: bool,
    #[arg(long, requires = "search")]
    model_adjusted: bool,
    #[arg(long, requires_all = ["pilot_first", "model_adjusted"])]
    consensus: bool,
    #[arg(long, requires = "pilot_first")]
    staged_probes: bool,
}

impl Args {
    fn effective_matcher(&self) -> Matcher {
        if self.content_pilot {
            Matcher::Routed
        } else {
            self.matcher
        }
    }

    fn search_policy(&self) -> &'static str {
        if self.content_pilot {
            "content-pilot"
        } else if self.pilot_first {
            "pilot-first"
        } else if self.search {
            "scout"
        } else {
            "candidate"
        }
    }

    fn pilot_options(&self) -> PilotOptions {
        PilotOptions {
            group_race: self.pilot_progressive_groups.then(|| {
                rs_jxl::scheduler::progressive::GroupRaceOptions {
                    content_route: !self.pilot_progressive_all_content,
                    shortlist: !self.pilot_progressive_all_candidates,
                    ..Default::default()
                }
            }),
            context_probe_side: self.pilot_context_side.unwrap_or(0),
            preparation_cache_bytes: self
                .reuse_preparation_mib
                .map_or(0, |mib| usize::from(mib) << 20),
            record_representations: self.representations,
            source_palette: self.pilot_source_palette,
            family_tournament: self.pilot_family_tournament,
            single_stage: self.pilot_single_stage,
            independent_probes: self.pilot_independent_probes,
            parallel_initial: self.pilot_parallel_initial,
            quiet_arena: self.pilot_quiet_arena,
            overlap_automatic: self.pilot_overlap_automatic,
            probe_sampling_quarter: self.pilot_quarter_samples,
            near_tie_consensus: self.pilot_near_tie_consensus,
            ..Default::default()
        }
    }
}

fn run(args: Args) -> Result<()> {
    if args.output.as_ref().is_some_and(|path| path.exists()) {
        return Err("benchmark output already exists".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(&args.input)?.read_to_end(&mut bytes)?;
    let decoded = input::decode(&bytes, true)?;
    let image = decoded.image;
    if args.scan_only {
        let scan = rs_jxl::scheduler::scan::scan_lossless_input(&image, Default::default());
        println!(
            "{}",
            serde_json::json!({
                "width": image.width(), "height": image.height(),
                "colors": scan.sampled_colors, "flat": scan.sampled_flat,
                "samples": scan.sampled_pixels, "literal": scan.literal_candidate,
                "continuous_tone": scan.likely_continuous_tone(),
            })
        );
        return Ok(());
    }
    let mut params = CandidateParams::default().backend(args.effort)?;
    match args.family {
        Family::Automatic => {}
        Family::Neighborhood => {
            params = rs_jxl::scheduler::policy::winner_neighborhood(&params, image.channel_count());
        }
        Family::Literal => {
            params.predictor = Some(Predictor::Zero);
            params.group_size_shift = Some(3);
            params.palette_colors = 10000;
            params.channel_colors_percent = 0.0;
            params.learning_fraction = 0.01;
            params.patches = Override::Off;
        }
        Family::Cross0 | Family::Cross6 => {
            let rct = if matches!(args.family, Family::Cross6) {
                6
            } else {
                0
            };
            params = lossless_probe_plan(&params, Default::default())
                .into_iter()
                .find(|candidate| {
                    candidate.params.predictor == Some(Predictor::Variable)
                        && candidate.params.rct == Some(rct)
                })
                .ok_or("cross-channel family missing")?
                .params;
        }
    }
    params.probe_lz77 = args.lz;
    if let Some(rct) = args.rct {
        params.rct = Some(u32::from(rct));
    }
    if let Some(fraction) = args.learning_fraction {
        params.learning_fraction = fraction;
    }
    if let Some(percent) = args.pre_colors {
        params.channel_colors_pre_transform_percent = percent;
    }
    if let Some(mode) = args.buffering {
        params.buffering = match mode {
            0 => rs_jxl::scheduler::params::SourceBuffering::FullFrame,
            -1 => rs_jxl::scheduler::params::SourceBuffering::Auto,
            1 => rs_jxl::scheduler::params::SourceBuffering::StreamingAbove2048,
            2 => rs_jxl::scheduler::params::SourceBuffering::Streaming,
            _ => unreachable!("validated buffering mode"),
        };
    }
    params.validate()?;
    let matcher = args.effective_matcher();
    let mut base = EncodeOptions {
        threads: usize::from(args.threads),
        ..Default::default()
    };
    args.ma_budget.apply(&mut base.modular);
    if args.palette_order.is_some() || args.force_palette {
        base.modular.palette_policy = Some(rs_jxl::modular::palette::PaletteOptions {
            force_palette: args.force_palette,
            tools: rs_jxl::modular::palette::PaletteTools {
                order: rs_jxl::modular::palette::PaletteOrder::from_source_mode(
                    args.palette_order.unwrap_or(0),
                )?,
                mapping: rs_jxl::modular::palette::PaletteMapping::Flat,
                ..Default::default()
            },
            ..Default::default()
        });
    }
    base.modular.parallel_tokens = true;
    base.modular.parallel_weighted = args.parallel_wp;
    base.modular.source_predictor_dedup = args.dedup;
    base.modular.source_property_dedup = args.dedup;
    args.ma_pruning.apply(&mut base.modular);
    if args.subtrees {
        base.modular.source_ma_parallel = rs_jxl::modular::ma_train::ParallelOptions {
            subtrees: true,
            balanced: true,
            small_subtrees: true,
            dynamic_subtrees: args.dynamic_subtrees,
            inherit_histograms: args.subtree_histograms,
            ..Default::default()
        };
        if let Some(workers) = args.ma_workers {
            base.modular.source_ma_parallel.workers.tasks = usize::from(workers);
        }
    }
    base.modular.source_gather = Some(GatherOptions {
        workers: base.threads.min(64),
        sparse: true,
        cache_hashes: args.hash_cache,
        concat_batches: args.concat_batches,
        ..Default::default()
    });
    base.modular.source_lz77 = Some(SourceConfig {
        compact_steps: true,
        chain_batch: args.chain_batch,
        parallelism: Some(SourceParallelism {
            max_workers: base.threads.min(64),
            fast_streams: true,
            ..Default::default()
        }),
        ..Default::default()
    });
    let lz = base.modular.source_lz77.as_mut().unwrap();
    if args.fast_ranges {
        lz.fast_ranges = rs_jxl::tools::lz77::SourceFastRanges::Both;
        lz.fast_prefix = Some(rs_jxl::tools::lz77::SourceFastPrefix::Simd);
    }
    if !matches!(matcher, Matcher::Source | Matcher::Routed) {
        lz.bucket_selection = Some(SourceBucketSelection::ParallelWide);
        lz.match_cost = SourceMatchCost::Frontier;
        lz.bucket_dominance = true;
        lz.bucket_distances = true;
    }
    if matches!(matcher, Matcher::Indexed | Matcher::Suffix) {
        lz.index_mode = SourceIndexMode::All;
        lz.index_tags = SourceIndexTags::Simd;
        lz.parallel_index_build = true;
        lz.index_compact_values = true;
        lz.index_word_match = true;
        lz.block_mode = SourceBlockMode::Intervals;
        lz.block_sparse_cache = true;
        lz.block_shared_frontier = true;
        lz.block_distance_bound = true;
        lz.block_distance_classes = true;
        lz.block_zero_tail = true;
        lz.block_literal_bound = true;
    }
    if matches!(matcher, Matcher::Suffix) {
        lz.suffix_mode = SourceSuffixMode::SlidingLcp;
    }
    base.modular.source_ma_histograms = HistogramOptions {
        early_cache_14: args.early_cache_14,
        events: args.events,
        sparse: args.sparse,
        dense: if matches!(args.histograms, Histograms::Dense | Histograms::DenseBins) {
            DenseHistogram::Extended
        } else {
            DenseHistogram::Disabled
        },
        property_bins: if matches!(args.histograms, Histograms::Bins | Histograms::DenseBins) {
            PropertyBins::SmallOrCompacted
        } else {
            PropertyBins::Disabled
        },
        ..Default::default()
    };
    base.modular.source_ma_scoring.count_log = args.count_log;
    if args.clipped {
        base.modular.source_ma_scoring.clipped =
            rs_jxl::modular::ma_train::ClippedEntropyMode::SlidingAtLeast2048;
    }
    base.modular.source_ma_scoring.fast_entropy = args.shannon;
    base.modular.source_ma_scoring.description = args.description_cost.into();
    if let Some(mode) = &args.fast_context {
        use rs_jxl::modular::fast_context::{Bias, Mode, Options, PredictorSet, Receipt, Scan};
        base.modular.fast_context = Some(Options {
            mode: match mode.as_str() {
                "leaf" => Mode::Leaf,
                "gradient12" => Mode::Gradient12,
                "directional12" => Mode::Directional12,
                "auto12" => Mode::Auto12,
                _ => unreachable!("validated mode"),
            },
            bias: match args.context_bias.as_deref().unwrap_or("off") {
                "off" => Bias::Off,
                "static" => Bias::Static,
                "adaptive" => Bias::Adaptive,
                _ => unreachable!("validated bias"),
            },
            scan: match args.context_scan.as_deref().unwrap_or("auto") {
                "auto" => Scan::Auto,
                "scalar" => Scan::Scalar,
                "avx2" => Scan::Avx2,
                "avx512" => Scan::Avx512,
                _ => unreachable!("validated scan"),
            },
            receipt: Some(std::sync::Arc::new(Receipt::default())),
            predictors: match args.context_predictors.as_deref().unwrap_or("basic") {
                "basic" => PredictorSet::Basic,
                "full" => PredictorSet::Full,
                _ => unreachable!("validated predictors"),
            },
        });
    }
    if let Some(mode) = &args.ma_seed {
        use rs_jxl::modular::ma_train::{SeedMode, SeedOptions, SeedReceipt};
        base.modular.source_ma_seed = Some(SeedOptions {
            mode: match mode.as_str() {
                "only" => SeedMode::Only,
                "refine" => SeedMode::Refine,
                "repair" => SeedMode::Repair,
                _ => unreachable!("validated seed"),
            },
            depth: args.ma_seed_depth.unwrap_or(3),
            receipt: Some(std::sync::Arc::new(SeedReceipt::default())),
            ..Default::default()
        });
    }
    if args.model_costs {
        base.modular.model_cost_receipt = Some(std::sync::Arc::new(
            rs_jxl::modular::encode::receipt_cost::ModelCostReceipt::default(),
        ));
    }
    base.modular.source_ma_search.refine_quantiles = args.refine_quantiles;
    base.modular.source_ma_search.histogram_bins = args.ma_histogram_bins.unwrap_or(0);
    base.modular.source_ma_search.quantile_splits = match args.quantile_splits {
        None => rs_jxl::modular::ma_train::QuantileSplits::Disabled,
        Some(32) => rs_jxl::modular::ma_train::QuantileSplits::ThirtyTwo,
        Some(64) => rs_jxl::modular::ma_train::QuantileSplits::SixtyFour,
        Some(_) => return Err("quantile-splits must be 32 or 64".into()),
    };
    let seconds;
    let mut search_report = serde_json::Value::Null;
    let mut model_costs = serde_json::Value::Null;
    let mut fast_context = serde_json::Value::Null;
    let mut ma_seed = serde_json::Value::Null;
    let start = Instant::now();
    let encoded = if args.search {
        let mut driver = DriverOptions::scout();
        driver.scout_pilot_first = args.pilot_first;
        driver.scout_pilot_consensus = args.consensus;
        driver.scout_staged_probes = args.staged_probes;
        driver.screen.regions.context_side = args.context_side.map_or(0, usize::from);
        driver.screen.regions.full_width_strips = args.wide_probes;
        driver.screen.regions.strip_rows = args.strip_rows.map(usize::from);
        let result = if args.content_pilot {
            pilot::encode_with_options(&image, &params, &base, args.pilot_options())?
        } else {
            run_native_pixel_search(
                &image,
                &params,
                &driver,
                &base,
                NativeBackendOptions {
                    preparation_cache_bytes: args
                        .reuse_preparation_mib
                        .map_or(0, |mib| usize::from(mib) << 20),
                    probe_palette: args.probe_palette,
                    model_adjusted_probes: args.model_adjusted,
                    routed_tools: matches!(matcher, Matcher::Routed),
                    record_representations: args.representations,
                    acceptance: if args.probe_palette || args.model_adjusted {
                        rs_jxl::scheduler::native::NativeAcceptance::AllowReportedGaps
                    } else {
                        Default::default()
                    },
                    ..Default::default()
                },
            )?
        };
        seconds = start.elapsed().as_secs_f64();
        if let Some(stats) = result.preparation_reuse {
            eprintln!("preparation_reuse: {stats:?}");
        }
        if args.representations {
            for record in &result.adaptations {
                eprintln!(
                    "representation: kind={:?} predictor={:?} rct={:?} receipt={:?}",
                    record.kind,
                    record.effective.predictor,
                    record.effective.rct,
                    record
                        .representation
                        .as_ref()
                        .and_then(|collector| collector.get())
                );
            }
        }
        let output = result.output;
        search_report = serde_json::json!({
            "routes": output.report.events.iter().filter_map(|event| match event {
                rs_jxl::scheduler::driver::SearchEvent::Route { name, candidates } =>
                    Some(serde_json::json!({ "name": name, "candidates": candidates })),
                _ => None,
            }).collect::<Vec<_>>(),
            "progressive": result.progressive.as_ref().map(|p| serde_json::json!({
                "group_pixels": p.group_pixels,
                "peak_retained_bytes": p.peak_retained_bytes,
                "regions": p.regions.iter().map(|r| [r.x, r.y, r.width, r.height]).collect::<Vec<_>>(),
                "candidates": p.candidates.iter().map(|c| serde_json::json!({
                    "name": c.name, "params": format!("{:?}", c.params),
                    "state": format!("{:?}", c.state),
                    "total_groups": c.total_groups, "encoded_groups": c.encoded_groups,
                    "reused_groups": c.reused_groups, "encoded_pixels": c.encoded_pixels,
                    "pilot_scores": c.pilot_scores,
                    "sampled_region_bytes": c.sampled_region_bytes,
                    "estimated_bytes": c.estimated_bytes, "seconds": c.seconds,
                    "final_bytes": c.final_bytes,
                })).collect::<Vec<_>>(),
            })),
            "full_encodes": output.report.full_encodes,
            "probe_encodes": output.report.probe_encodes,
            "probe_pixels": output.report.probe_pixels,
            "winner_index": output.report.winner_index,
            "trials": output.report.trials.iter().map(|trial| serde_json::json!({
                "params": format!("{:?}", trial.params),
                "bytes": trial.bytes, "lower_bound": trial.lower_bound,
            })).collect::<Vec<_>>(),
        });
        eprintln!(
            "full={} probes={} winner={:?}",
            output.report.full_encodes, output.report.probe_encodes, output.params
        );
        output.bytes
    } else {
        let result = encode_native_candidate(
            &params,
            &image,
            &base,
            NativeBackendOptions {
                routed_tools: matches!(matcher, Matcher::Routed),
                ..Default::default()
            },
        )?;
        seconds = start.elapsed().as_secs_f64();
        if let Some(receipt) = &result.artifact.options.modular.model_cost_receipt {
            model_costs = serde_json::json!({
                "tree_bits": receipt.tree_bits(),
                "entropy_header_bits": receipt.entropy_header_bits(),
                "total_bits": receipt.bits(),
            });
        }
        if let Some(options) = &result.artifact.options.modular.fast_context {
            let receipt = options
                .receipt
                .as_ref()
                .expect("requested receipt")
                .snapshot();
            fast_context = serde_json::json!({
                "mode": args.fast_context, "bias": args.context_bias.as_deref().unwrap_or("off"),
                "scan": args.context_scan.as_deref().unwrap_or("auto"),
                "predictors": args.context_predictors.as_deref().unwrap_or("basic"),
                "trees": receipt.trees, "leaves": receipt.leaves, "pixels": receipt.pixels,
                "offset_leaves": receipt.offset_leaves, "horizontal": receipt.horizontal,
                "vertical": receipt.vertical, "single": receipt.single,
            });
        }
        if let Some(options) = &result.artifact.options.modular.source_ma_seed {
            let receipt = options
                .receipt
                .as_ref()
                .expect("requested seed receipt")
                .snapshot();
            ma_seed = serde_json::json!({
                "mode": args.ma_seed, "depth": options.depth,
                "learners": receipt.learners, "prepared_records": receipt.prepared_records,
                "seed_splits": receipt.seed_splits, "refined_leaves": receipt.refined_leaves,
                "repaired_branches": receipt.repaired_branches, "merged_branches": receipt.merged_branches,
                "predictor_changes": receipt.predictor_changes, "histogram_reuses": receipt.histogram_reuses,
            });
        }
        eprintln!("candidate={params:?}");
        result.bytes
    };
    let pixels = f64::from(image.width()) * f64::from(image.height());
    println!(
        "{}",
        serde_json::json!({
            "bytes": encoded.len(), "seconds": seconds, "bpp": encoded.len() as f64 * 8.0 / pixels,
            "mpix_per_second": pixels / 1e6 / seconds, "width": image.width(), "height": image.height(),
            "jpeg_pixel_input": decoded.jpeg, "family": format!("{:?}", args.family), "lz": args.lz,
            "matcher": format!("{matcher:?}"), "search_policy": args.search_policy(),
            "search": args.search, "timing": "encoder call only; input decode and output IO excluded",
            "search_report": search_report,
            "description_cost": format!("{:?}", args.description_cost),
            "model_costs": model_costs,
            "fast_context": fast_context,
            "ma_seed": ma_seed,
        })
    );
    if let Some(path) = args.output {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(&encoded)?;
    }
    if let Some(profile) = rs_jxl::profile::report() {
        eprint!("{profile}");
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_and_full_predictor_experiments_are_explicit() {
        let defaults = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert!(defaults.ma_seed.is_none() && defaults.ma_seed_depth.is_none());
        assert!(defaults.context_predictors.is_none());
        for mode in ["only", "refine", "repair"] {
            for depth in ["1", "3", "8"] {
                assert!(
                    Args::try_parse_from([
                        "encoder_lab",
                        "input.png",
                        "--ma-seed",
                        mode,
                        "--ma-seed-depth",
                        depth
                    ])
                    .is_ok()
                );
            }
            for flag in ["--search", "--scan-only"] {
                assert!(
                    Args::try_parse_from(["encoder_lab", "input.png", "--ma-seed", mode, flag])
                        .is_err()
                );
            }
            assert!(
                Args::try_parse_from([
                    "encoder_lab",
                    "input.png",
                    "--ma-seed",
                    mode,
                    "--fast-context",
                    "auto12"
                ])
                .is_err()
            );
        }
        for args in [
            vec!["--ma-seed-depth", "3"],
            vec!["--ma-seed", "only", "--ma-seed-depth", "0"],
            vec!["--ma-seed", "only", "--ma-seed-depth", "9"],
            vec!["--context-predictors", "full"],
        ] {
            let mut argv = vec!["encoder_lab", "input.png"];
            argv.extend(args);
            assert!(Args::try_parse_from(argv).is_err());
        }
        for predictors in ["basic", "full"] {
            assert!(
                Args::try_parse_from([
                    "encoder_lab",
                    "input.png",
                    "--fast-context",
                    "auto12",
                    "--context-predictors",
                    predictors
                ])
                .is_ok()
            );
        }
    }

    #[test]
    fn model_cost_experiments_are_explicit_and_direct_only() {
        let defaults = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert_eq!(
            DescriptionCost::from(defaults.description_cost),
            DescriptionCost::Disabled
        );
        assert!(!defaults.model_costs);
        for (name, expected) in [
            ("off", DescriptionCost::Disabled),
            ("quarter", DescriptionCost::Quarter),
            ("half", DescriptionCost::Half),
            ("full", DescriptionCost::Full),
        ] {
            let args = Args::try_parse_from([
                "encoder_lab",
                "input.png",
                "--description-cost",
                name,
                "--model-costs",
            ])
            .unwrap();
            assert_eq!(DescriptionCost::from(args.description_cost), expected);
            assert!(args.model_costs);
        }
        for flag in ["--search", "--scan-only"] {
            assert!(
                Args::try_parse_from(["encoder_lab", "input.png", "--model-costs", flag,]).is_err()
            );
        }
        assert!(
            Args::try_parse_from(["encoder_lab", "input.png", "--description-cost", "unknown",])
                .is_err()
        );
    }

    #[test]
    fn subtree_experiments_are_independent_opt_ins() {
        let plain = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert!(!plain.subtrees);
        assert!(!plain.dynamic_subtrees);
        assert!(!plain.subtree_histograms);
        for dynamic in [false, true] {
            for histograms in [false, true] {
                let mut argv = vec!["encoder_lab", "input.png", "--subtrees"];
                if dynamic {
                    argv.push("--dynamic-subtrees");
                }
                if histograms {
                    argv.push("--subtree-histograms");
                }
                let args = Args::try_parse_from(argv).unwrap();
                assert!(args.subtrees);
                assert_eq!(args.dynamic_subtrees, dynamic);
                assert_eq!(args.subtree_histograms, histograms);
            }
        }
    }

    #[test]
    fn subtree_experiments_require_subtrees() {
        for flags in [
            &["--dynamic-subtrees"][..],
            &["--subtree-histograms"][..],
            &["--dynamic-subtrees", "--subtree-histograms"][..],
        ] {
            let mut argv = vec!["encoder_lab", "input.png"];
            argv.extend_from_slice(flags);
            let error = Args::try_parse_from(argv)
                .err()
                .expect("subtree experiments must require --subtrees");
            assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
            assert!(error.to_string().contains("--subtrees"));
        }
    }

    #[test]
    fn pilot_palette_and_family_flags_are_independent_opt_ins() {
        let base = ["encoder_lab", "input.png", "--search", "--content-pilot"];
        for source_palette in [false, true] {
            for family_tournament in [false, true] {
                let mut argv = base.to_vec();
                if source_palette {
                    argv.push("--pilot-source-palette");
                }
                if family_tournament {
                    argv.push("--pilot-family-tournament");
                }
                let args = Args::try_parse_from(argv).unwrap();
                assert_eq!(args.pilot_source_palette, source_palette);
                assert_eq!(args.pilot_family_tournament, family_tournament);
                let options = args.pilot_options();
                assert_eq!(options.source_palette, source_palette);
                assert_eq!(options.family_tournament, family_tournament);
            }
        }
    }

    #[test]
    fn pilot_palette_and_family_flags_require_content_pilot() {
        for flag in ["--pilot-source-palette", "--pilot-family-tournament"] {
            for search in [false, true] {
                let mut argv = vec!["encoder_lab", "input.png", flag];
                if search {
                    argv.push("--search");
                }
                assert_eq!(
                    Args::try_parse_from(argv)
                        .err()
                        .expect("expected argument rejection")
                        .kind(),
                    ErrorKind::MissingRequiredArgument
                );
            }
        }
    }

    #[test]
    fn group_race_is_explicit_and_not_combined_with_crop_probe_experiments() {
        let base = ["encoder_lab", "input.png", "--search", "--content-pilot"];
        assert!(
            Args::try_parse_from(base)
                .unwrap()
                .pilot_options()
                .group_race
                .is_none()
        );
        let mut argv = base.to_vec();
        argv.push("--pilot-progressive-groups");
        assert!(
            Args::try_parse_from(&argv)
                .unwrap()
                .pilot_options()
                .group_race
                .is_some()
        );
        argv.extend(["--pilot-context-side", "256"]);
        assert!(Args::try_parse_from(argv).is_err());
        assert!(
            Args::try_parse_from(["encoder_lab", "input.png", "--pilot-progressive-groups"])
                .is_err()
        );
    }

    #[test]
    fn group_front_shortlist_can_be_disabled_for_paired_measurement() {
        let base = [
            "encoder_lab",
            "input.png",
            "--search",
            "--content-pilot",
            "--pilot-progressive-groups",
        ];
        let options = Args::try_parse_from(base).unwrap().pilot_options();
        assert!(options.group_race.unwrap().shortlist);
        let mut argv = base.to_vec();
        argv.push("--pilot-progressive-all-candidates");
        let options = Args::try_parse_from(argv).unwrap().pilot_options();
        let race = options.group_race.unwrap();
        assert!(!race.shortlist);
        assert!(race.content_route);
        assert!(
            Args::try_parse_from([
                "encoder_lab",
                "input.png",
                "--search",
                "--content-pilot",
                "--pilot-progressive-all-candidates",
            ])
            .is_err()
        );
    }

    #[test]
    fn square_probe_parser_is_explicit_and_bounded() {
        let base = ["encoder_lab", "input.png", "--search", "--content-pilot"];
        assert_eq!(
            Args::try_parse_from(base)
                .unwrap()
                .pilot_options()
                .context_probe_side,
            0
        );
        for value in ["256", "512"] {
            let mut argv = base.to_vec();
            argv.extend(["--pilot-context-side", value]);
            assert_eq!(
                Args::try_parse_from(argv)
                    .unwrap()
                    .pilot_options()
                    .context_probe_side,
                value.parse::<u16>().unwrap()
            );
        }
        for value in ["0", "255", "300", "513", "x"] {
            let mut argv = base.to_vec();
            argv.extend(["--pilot-context-side", value]);
            assert!(Args::try_parse_from(argv).is_err());
        }
        assert!(
            Args::try_parse_from(["encoder_lab", "input.png", "--pilot-context-side", "256"])
                .is_err()
        );
    }
    use clap::{CommandFactory, error::ErrorKind};

    #[test]
    fn alternative_scoring_flags_are_explicit_and_validated_before_io() {
        let args = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert!(!args.shannon && !args.refine_quantiles);
        assert_eq!(args.quantile_splits, None);
        assert_eq!(args.ma_histogram_bins, None);
        for value in ["2", "16", "32", "64", "256"] {
            let args =
                Args::try_parse_from(["encoder_lab", "input.png", "--ma-histogram-bins", value])
                    .unwrap();
            assert_eq!(args.ma_histogram_bins, Some(value.parse().unwrap()));
        }
        for value in ["0", "1", "257", "-1", "x"] {
            assert!(
                Args::try_parse_from(["encoder_lab", "input.png", "--ma-histogram-bins", value])
                    .is_err()
            );
        }
        assert!(
            Args::try_parse_from([
                "encoder_lab",
                "input.png",
                "--ma-histogram-bins",
                "32",
                "--refine-quantiles"
            ])
            .is_err()
        );
        for value in ["32", "64"] {
            let args =
                Args::try_parse_from(["encoder_lab", "input.png", "--quantile-splits", value])
                    .unwrap();
            assert_eq!(args.quantile_splits, Some(value.parse().unwrap()));
        }
        for value in ["0", "31", "33", "63", "65", "-1", "x"] {
            assert!(
                Args::try_parse_from(["encoder_lab", "input.png", "--quantile-splits", value,])
                    .is_err()
            );
        }
        assert!(
            Args::try_parse_from([
                "encoder_lab",
                "input.png",
                "--quantile-splits",
                "32",
                "--refine-quantiles",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from(["encoder_lab", "input.png", "--shannon", "--clipped",]).is_err()
        );
    }

    #[test]
    fn single_stage_pilot_is_explicit_and_forwards_its_option() {
        assert!(
            Args::try_parse_from(["encoder_lab", "input.png", "--pilot-single-stage"]).is_err()
        );
        let args = Args::try_parse_from([
            "encoder_lab",
            "input.png",
            "--search",
            "--content-pilot",
            "--pilot-single-stage",
        ])
        .unwrap();
        assert!(args.pilot_options().single_stage);
    }

    #[test]
    fn representation_experiments_are_explicit_and_range_checked() {
        let plain = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert!(plain.palette_order.is_none() && plain.buffering.is_none());
        assert!(!plain.force_palette);
        let args = Args::try_parse_from([
            "encoder_lab",
            "input.png",
            "--palette-order",
            "3",
            "--force-palette",
            "--buffering",
            "-1",
            "--learning-fraction",
            "0.01",
            "--pre-colors",
            "0",
        ])
        .unwrap();
        assert_eq!(args.palette_order, Some(3));
        assert_eq!(args.buffering, Some(-1));
        assert_eq!(args.learning_fraction, Some(0.01));
        assert_eq!(args.pre_colors, Some(0.0));
        assert!(args.force_palette);
        for (flag, value) in [
            ("--palette-order", "10"),
            ("--buffering", "3"),
            ("--buffering", "-2"),
        ] {
            assert!(Args::try_parse_from(["encoder_lab", "input.png", flag, value]).is_err());
        }
    }

    #[test]
    fn probe_and_near_tie_experiments_are_separate_opt_ins() {
        let plain = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert!(!plain.pilot_options().probe_sampling_quarter);
        assert!(!plain.pilot_options().near_tie_consensus);
        for flag in ["--pilot-quarter-samples", "--pilot-near-tie-consensus"] {
            assert!(Args::try_parse_from(["encoder_lab", "input.png", flag]).is_err());
            let args = Args::try_parse_from([
                "encoder_lab",
                "input.png",
                "--search",
                "--content-pilot",
                flag,
            ])
            .unwrap();
            assert_eq!(
                args.pilot_options().probe_sampling_quarter,
                flag == "--pilot-quarter-samples"
            );
            assert_eq!(
                args.pilot_options().near_tie_consensus,
                flag == "--pilot-near-tie-consensus"
            );
        }
    }

    #[test]
    fn overlapped_pilot_is_explicit_and_requires_content_policy() {
        assert!(
            Args::try_parse_from(["encoder_lab", "input.png", "--pilot-overlap-automatic",])
                .is_err()
        );
        let args = Args::try_parse_from([
            "encoder_lab",
            "input.png",
            "--search",
            "--content-pilot",
            "--pilot-overlap-automatic",
        ])
        .unwrap();
        assert!(args.pilot_options().overlap_automatic);
        assert!(
            !Args::try_parse_from(["encoder_lab", "input.png", "--search", "--content-pilot",])
                .unwrap()
                .pilot_options()
                .overlap_automatic
        );
    }

    #[test]
    fn content_pilot_allows_default_matcher_and_forwards_representation_request() {
        Args::command().debug_assert();
        let plain = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert!(matches!(plain.effective_matcher(), Matcher::Source));
        assert_eq!(plain.search_policy(), "candidate");
        assert!(!plain.pilot_options().record_representations);
        let pilot =
            Args::try_parse_from(["encoder_lab", "input.png", "--search", "--content-pilot"])
                .unwrap();
        assert!(matches!(pilot.matcher, Matcher::Source));
        assert!(matches!(pilot.effective_matcher(), Matcher::Routed));
        assert_eq!(pilot.search_policy(), "content-pilot");
        assert!(!pilot.pilot_options().record_representations);
        let recorded = Args::try_parse_from([
            "encoder_lab",
            "input.png",
            "--search",
            "--content-pilot",
            "--representations",
        ])
        .unwrap();
        assert!(recorded.pilot_options().record_representations);
        assert_eq!(recorded.pilot_options().resources, Default::default());
    }

    #[test]
    fn content_pilot_rejects_every_explicit_matcher_in_either_order() {
        for matcher in ["source", "buckets", "indexed", "suffix", "routed"] {
            for pilot_first in [false, true] {
                let mut argv = vec!["encoder_lab", "input.png", "--search"];
                if pilot_first {
                    argv.push("--content-pilot");
                }
                argv.extend(["--matcher", matcher]);
                if !pilot_first {
                    argv.push("--content-pilot");
                }
                let error = Args::try_parse_from(argv)
                    .err()
                    .expect("explicit matcher must conflict");
                assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
            }
        }
    }

    #[test]
    fn ordinary_experiment_matchers_and_policy_labels_remain_explicit() {
        for matcher in ["source", "buckets", "indexed", "suffix", "routed"] {
            let args = Args::try_parse_from([
                "encoder_lab",
                "input.png",
                "--matcher",
                matcher,
                "--search",
            ])
            .unwrap();
            assert_eq!(
                format!("{:?}", args.effective_matcher()).to_lowercase(),
                matcher
            );
            assert_eq!(args.search_policy(), "scout");
        }
        let args = Args::try_parse_from(["encoder_lab", "input.png", "--search", "--pilot-first"])
            .unwrap();
        assert_eq!(args.search_policy(), "pilot-first");
    }
}
#[test]
fn fast_context_modes_are_explicit_and_require_their_primary_switch() {
    let defaults = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
    assert!(defaults.fast_context.is_none());
    assert!(defaults.context_bias.is_none());
    assert!(defaults.context_scan.is_none());
    for mode in ["leaf", "gradient12", "directional12", "auto12"] {
        for bias in ["off", "static", "adaptive"] {
            let args = Args::try_parse_from([
                "encoder_lab",
                "input.png",
                "--fast-context",
                mode,
                "--context-bias",
                bias,
                "--context-scan",
                "scalar",
            ])
            .unwrap();
            assert_eq!(args.fast_context.as_deref(), Some(mode));
            assert_eq!(args.context_bias.as_deref(), Some(bias));
        }
    }
    for (flag, value) in [("--context-bias", "static"), ("--context-scan", "avx2")] {
        assert!(Args::try_parse_from(["encoder_lab", "input.png", flag, value]).is_err());
    }
    for flag in ["--search", "--scan-only"] {
        assert!(
            Args::try_parse_from(["encoder_lab", "input.png", "--fast-context", "auto12", flag])
                .is_err()
        );
    }
}
