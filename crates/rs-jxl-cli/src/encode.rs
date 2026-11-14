//! Still-image encode arguments, policy dispatch and file orchestration.

use std::path::PathBuf;

use clap::Args;
use rs_jxl::EncodeOptions;
use rs_jxl::codec::{self, EncodeJob, SearchDetails, SourceRequest, SourceSearch};
use rs_jxl::scheduler::{CandidateParams, NativeBackendOptions, pilot::PilotOptions};

use crate::validators::{parse_prev_props, parse_sample_limit, parse_threads};
use crate::{Result, input, io, options};

#[derive(Debug, Args)]
#[command(
    after_long_help = "JPEG input is pixel-lossless relative to zune-jpeg's \
decoded samples only; original JPEG bytes cannot be reconstructed.\n\
PNG sRGB/gAMA/cHRM/iCCP and JPEG ICC color metadata are preserved. \
Untagged input defaults to sRGB gray/RGB. No color conversion or EXIF orientation \
transform is performed; PNG tRNS is decoded pixel data.\n\
Unsupported metadata such as EXIF/XMP/comments requires --strip-metadata. \
Supported color metadata is retained even with --strip-metadata. Stripping \
unsupported metadata can change displayed orientation without changing samples.\n\
APNG, CMYK/YCCK JPEG and non-8-bit JPEG are unsupported. No extra input-size \
or pixel-count cap is imposed; format, allocation and overflow checks still apply. \
Explicit encoder resource budgets are enforced as well.\n\
Source mode is experimental and entirely Rust: --source-effort 1..11 encodes \
one native candidate; --source-search additionally runs the existing native \
pixel search driver (effort 11 uses its separate full-frame tournament). \
At effort 9/10, --source-scout selects an experimental coverage-guided frontier \
with fewer full-frame candidates; it may change compression size. \
--source-pilot selects a separate experimental content-aware pilot-first policy. \
Optional routes are exposed through the Rust API; source-identical bytes or \
performance are not claimed. Unsupported policies and resource exhaustion \
are errors unless --ma-budget-mode bounded explicitly stops optional MA splits.\n\
Effort 11 without --source-search selects the explicit TectonicPlate (-1) \
backend policy, not the frame-level tournament.\n\
Source mode cannot be combined with explicitly specified tool options \
(even their default values). --threads, --strip-metadata and --force remain \
available, as do --ma-budget-mode strict|bounded, --ma-max-leaves N and \
experimental --ma-prune (exact partition reuse and small-node dominance). \
Omitting the MA budget mode keeps source strict and manual bounded. \
Required MA partitions must fit even in bounded mode. \
Source sampling has no extra sample-count cap by default. An explicit \
--source-sample-limit bounds training samples independently of the source \
sampling policy; exceeding it fails rather than reducing sampling.\n\
Source-prefixed transform, refinement, buffering and LZ77 switches override \
the requested source candidate. With --source-search they describe its starting \
candidate, not a promise that every trial or the winner retains those settings. \
--source-prev-props restricts previous-channel MA features only when \
--source-max-properties is nonzero; it does not enable cross-channel MA itself.\n\
Use '-' for stdin/stdout. Diagnostics always go to stderr. \
Existing output is not replaced unless --force is supplied."
)]
pub(crate) struct EncodeArgs {
    #[command(flatten)]
    pub(crate) tools: options::Tools,
    #[command(flatten)]
    pub(crate) ma_budget: options::MaBudgetOptions,
    #[command(flatten)]
    pub(crate) ma_pruning: options::MaPruningOptions,
    /// Experimental source policy: one native candidate at effort 1..=11
    #[arg(long, value_name = "N", conflicts_with = "manual_tools", value_parser = clap::value_parser!(u8).range(1..=11))]
    pub(crate) source_effort: Option<u8>,
    /// Disable automatic e9 MA learner/offsets/scoring; explicit tools still apply.
    #[arg(long, requires = "source_effort")]
    pub(crate) no_source_ma_defaults: bool,
    /// Run native pixel search instead of just one source candidate
    #[arg(long, requires = "source_effort")]
    pub(crate) source_search: bool,
    /// Experimental coverage-guided search at source effort 9/10
    #[arg(long, requires = "source_search")]
    pub(crate) source_scout: bool,
    /// Experimental content-aware pilot-first search at source effort 9/10
    #[arg(long, requires = "source_search", conflicts_with = "source_scout")]
    pub(crate) source_pilot: bool,
    /// Experimental original-group racing and continuation within source pilot
    #[arg(long, requires = "source_pilot")]
    pub(crate) source_progressive_groups: bool,
    /// Optional hard MA training sample limit; omitted means no sample-count cap
    #[arg(long, value_name = "N", requires = "source_effort", value_parser = parse_sample_limit)]
    pub(crate) source_sample_limit: Option<usize>,
    /// Optional source grouping shift override: 0..=3 gives 128/256/512/1024
    /// pixel groups; omitted keeps the source automatic choice.
    #[arg(long, value_name = "N", requires = "source_effort",
          value_parser = clap::value_parser!(u8).range(0..=3))]
    pub(crate) source_group_size_shift: Option<u8>,
    /// Opt-in post-training entropy-mode comparison: the alternate coding
    /// (ANS vs Prefix) is serialized and kept only when actually smaller.
    #[arg(long, requires = "source_effort")]
    pub(crate) source_refine_entropy: bool,
    /// Opt-in post-training leaf refitting: the refitted tree is serialized and
    /// kept only when actually smaller.
    #[arg(long, requires = "source_effort")]
    pub(crate) source_refine_leaves: bool,
    /// Opt-in static leaf offset fitting: candidates are serialized and kept
    /// only when actually smaller.
    #[arg(long, requires = "source_effort")]
    pub(crate) source_refine_offsets: bool,
    /// Opt-in ANS context-map merge proposals: each candidate is serialized and
    /// kept only when actually smaller (multiple trials, highest cost).
    #[arg(long, requires = "source_effort")]
    pub(crate) source_refine_context_map: bool,
    /// Opt-in source responsive mode: the whole-frame default squeeze script and
    /// the responsive sampling/predictor policy (may change the codestream).
    #[arg(long, requires = "source_effort")]
    pub(crate) source_responsive: bool,
    /// Opt-in lossless precise entropy: forces the Tortoise entropy tier.
    #[arg(long, requires = "source_effort")]
    pub(crate) source_precise_entropy: bool,
    /// Opt-in previous-channel MA properties (0..=16; the effective count is
    /// clamped by the transformed channel count).
    #[arg(long, value_name = "N", requires = "source_effort",
          value_parser = clap::value_parser!(u8).range(0..=16))]
    pub(crate) source_max_properties: Option<u8>,
    /// Opt-in previous-channel property subset at source effort 9/10: 4 keeps
    /// the full upstream set per channel, 2 keeps signed sample + signed
    /// residual (17, 19), 1 keeps the signed residual only (19). Effective
    /// only when previous-channel properties are enabled.
    #[arg(long, value_name = "N", requires = "source_effort", value_parser = parse_prev_props)]
    pub(crate) source_prev_props: Option<u8>,
    /// Opt-in careful first LZ77 pass instead of the fast prefix matcher.
    #[arg(long, requires = "source_effort")]
    pub(crate) source_lz77_careful_first_pass: bool,
    /// Opt-in Optc256 bucket matcher (short32/long8 rings + 24 spatial anchors)
    /// instead of the default hash chain.
    #[arg(long, requires = "source_effort")]
    pub(crate) source_bucket_matcher: bool,
    /// Opt-in source buffering override: auto (default), full (one shared
    /// full-frame tree), above2048, streaming.
    #[arg(long, value_name = "MODE", requires = "source_effort",
          value_parser = ["auto", "full", "above2048", "streaming"])]
    pub(crate) source_buffering: Option<String>,
    /// PNG or JPEG input path, or '-' for stdin (detected by file signature)
    pub(crate) input: PathBuf,
    /// JXL output path, or '-' for binary stdout
    pub(crate) output: PathBuf,
    /// Encoder worker count (1..=256)
    #[arg(long, value_name = "N", default_value = "1", value_parser = parse_threads)]
    pub(crate) threads: usize,
    /// Discard unsupported metadata; retain supported color metadata and samples
    #[arg(long)]
    pub(crate) strip_metadata: bool,
    /// Replace an existing output file; input and output must still differ
    #[arg(long)]
    pub(crate) force: bool,
}

impl EncodeArgs {
    pub(crate) fn job(&self) -> Result<EncodeJob> {
        let mut options = EncodeOptions {
            threads: self.threads,
            ..EncodeOptions::default()
        };
        let Some(effort) = self.source_effort else {
            self.tools.apply(&mut options)?;
            self.ma_budget.apply(&mut options.modular);
            self.ma_pruning.apply(&mut options.modular);
            return Ok(EncodeJob {
                options,
                source: None,
            });
        };
        self.ma_budget.apply(&mut options.modular);
        self.ma_pruning.apply(&mut options.modular);
        options.modular.sample_limit = self.source_sample_limit;
        options.modular.source_ma_defaults = !self.no_source_ma_defaults;
        let mut params = CandidateParams::default().backend(effort)?;
        if let Some(shift) = self.source_group_size_shift {
            params.group_size_shift = Some(shift);
        }
        if self.source_refine_entropy {
            options.modular.refinement.entropy = true;
        }
        if self.source_refine_leaves {
            options.modular.refinement.leaves = true;
        }
        if self.source_refine_offsets {
            options.modular.refinement.offsets = Some(Default::default());
        }
        if self.source_refine_context_map {
            options.modular.refinement.context_map = Some(Default::default());
        }
        if self.source_responsive {
            params.responsive = true;
        }
        if self.source_precise_entropy {
            params.precise_entropy = true;
        }
        if let Some(max_properties) = self.source_max_properties {
            params.max_properties = max_properties;
        }
        if let Some(subset) = self.source_prev_props {
            params.previous_property_subset = subset;
        }
        if self.source_lz77_careful_first_pass {
            params.lz77_careful_first_pass = true;
        }
        if self.source_bucket_matcher {
            params.bucket_matcher = true;
        }
        if let Some(mode) = &self.source_buffering {
            params.buffering = match mode.as_str() {
                "auto" => rs_jxl::modular::source_options::SourceBuffering::Auto,
                "full" => rs_jxl::modular::source_options::SourceBuffering::FullFrame,
                "above2048" => rs_jxl::modular::source_options::SourceBuffering::StreamingAbove2048,
                "streaming" => rs_jxl::modular::source_options::SourceBuffering::Streaming,
                other => return Err(format!("unhandled --source-buffering {other}").into()),
            };
        }
        if self.source_scout && !(9..=10).contains(&effort) {
            return Err("--source-scout requires --source-effort 9 or 10".into());
        }
        if self.source_pilot && !(9..=10).contains(&effort) {
            return Err("--source-pilot requires --source-effort 9 or 10".into());
        }
        let search = if !self.source_search {
            SourceSearch::Disabled
        } else if self.source_pilot {
            SourceSearch::Pilot(PilotOptions {
                group_race: self.source_progressive_groups.then(Default::default),
                ..Default::default()
            })
        } else if self.source_scout {
            SourceSearch::Scout
        } else {
            SourceSearch::Auto
        };
        Ok(EncodeJob {
            options,
            source: Some(SourceRequest {
                params,
                search,
                backend: NativeBackendOptions::default(),
            }),
        })
    }
}

// Both file encoding and the real-image timing harness use this dispatch.
pub(crate) fn encode_image(image: &rs_jxl::Image, args: &EncodeArgs) -> Result<Vec<u8>> {
    let job = args.job()?;
    if let Some(effort) = args.source_effort {
        eprintln!(
            "note: experimental pure-Rust source mode: effort={effort} search={} \
         sample_limit={}; no byte-identity or performance-parity claim",
            args.source_search,
            args.source_sample_limit
                .map_or_else(|| "none".to_owned(), |limit| limit.to_string()),
        );
        if effort == 11 && !args.source_search {
            eprintln!(
                "note: explicit TectonicPlate (-1) backend; no C++ frame-level effort11 tournament"
            );
        }
    }
    let result = codec::encode_job(image, &job)?;
    match &result.report.details {
        SearchDetails::Tectonic(report) => {
            let params = &report.trials[report.winner_index].params;
            eprintln!(
                "source effort11 tournament: branch={:?} full_encodes={} winner={} winner_effort={}",
                report.branch,
                report.full_encodes,
                report.winner_index,
                10 - params.speed_tier
            );
        }
        SearchDetails::Pixel(report) => {
            let params = &report.trials[report.winner_index].params;
            eprintln!(
                "source search: full_encodes={} probe_encodes={} winner={} winner_effort={}",
                report.full_encodes,
                report.probe_encodes,
                report.winner_index,
                10 - params.speed_tier,
            );
            if args.source_scout || args.source_pilot {
                eprintln!("source scout: winner_params={params:?}");
            }
        }
        SearchDetails::None => {}
    }
    Ok(result.bytes)
}

pub(crate) fn run(args: EncodeArgs) -> Result<()> {
    let _total = rs_jxl::profile::scope("cli.total");
    io::check_paths(&args.input, &args.output, args.force)?;
    let bytes = io::read_input(&args.input)?;
    let decoded = {
        let _input = rs_jxl::profile::scope("cli.input_decode");
        input::decode(&bytes, args.strip_metadata)?
    };
    if decoded.jpeg {
        eprintln!(
            "warning: JPEG -> JXL is PIXEL LOSSLESS relative to zune-jpeg's decoded \
             samples, NOT a byte-reversible JPEG transcode. Original JPEG bytes \
             cannot be reconstructed."
        );
    }
    if args.strip_metadata {
        eprintln!(
            "warning: --strip-metadata discards unsupported metadata without color \
             conversion or EXIF orientation transforms; supported color metadata \
             is retained, but displayed orientation may change."
        );
    }
    eprintln!(
        "note: encoding decoded samples without color or orientation conversion; \
         supported color metadata is retained, including with --strip-metadata."
    );
    let encoded = {
        let _encoding = rs_jxl::profile::scope("cli.encode");
        encode_image(&decoded.image, &args)?
    };
    io::write_output(&args.input, &args.output, args.force, &encoded)
}
