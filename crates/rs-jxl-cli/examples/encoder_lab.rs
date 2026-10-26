//! Development-only candidate/scout experiments; not part of the installed CLI.
#[path = "../src/input/mod.rs"]
mod input;

#[path = "../src/options.rs"]
mod options;

#[cfg(test)]
#[path = "../src/encoder_lab_budget_parser_tests.rs"]
mod budget_parser_tests;

use clap::{Parser, ValueEnum};
use rs_jxl::{
    EncodeOptions,
    modular::{
        ma_train::{DenseHistogram, HistogramOptions, PropertyBins},
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Matcher {
    Source,
    Buckets,
    Indexed,
    Suffix,
    Routed,
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
    /// Score coarse MA split quantiles, then refine their strongest valleys.
    #[arg(long)]
    refine_quantiles: bool,
    /// Experimental coarse-only MA split thresholds.
    #[arg(long, value_parser = parse_quantile_splits, conflicts_with = "refine_quantiles")]
    quantile_splits: Option<u8>,
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
            record_representations: self.representations,
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
    fs::File::open(&args.input)?
        .take(input::MAX_INPUT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    let decoded = input::decode(&bytes, true)?;
    let image = decoded.image;
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
    base.modular.sample_limit = 33_554_432;
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
    base.modular.source_ma_search.refine_quantiles = args.refine_quantiles;
    base.modular.source_ma_search.quantile_splits = match args.quantile_splits {
        None => rs_jxl::modular::ma_train::QuantileSplits::Disabled,
        Some(32) => rs_jxl::modular::ma_train::QuantileSplits::ThirtyTwo,
        Some(64) => rs_jxl::modular::ma_train::QuantileSplits::SixtyFour,
        Some(_) => return Err("quantile-splits must be 32 or 64".into()),
    };
    let seconds;
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
    use clap::{CommandFactory, error::ErrorKind};

    #[test]
    fn alternative_scoring_flags_are_explicit_and_validated_before_io() {
        let args = Args::try_parse_from(["encoder_lab", "input.png"]).unwrap();
        assert!(!args.shannon && !args.refine_quantiles);
        assert_eq!(args.quantile_splits, None);
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
