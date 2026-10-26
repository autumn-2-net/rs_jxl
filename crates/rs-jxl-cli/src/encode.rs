//! Still-image encode arguments, policy dispatch and file orchestration.

use std::path::PathBuf;

use clap::Args;
use rs_jxl::scheduler::{
    CandidateParams, DriverOptions, NativeBackendOptions, native::encode_native_candidate,
    run_native_pixel_search, run_native_tectonic,
};
use rs_jxl::{EncodeOptions, encode};

use crate::validators::{parse_sample_limit, parse_threads};
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

// Both file encoding and the real-image timing harness use this dispatch.
pub(crate) fn encode_image(image: &rs_jxl::Image, args: &EncodeArgs) -> Result<Vec<u8>> {
    let mut options = EncodeOptions {
        threads: args.threads,
        ..EncodeOptions::default()
    };
    let Some(effort) = args.source_effort else {
        args.tools.apply(&mut options)?;
        args.ma_budget.apply(&mut options.modular);
        args.ma_pruning.apply(&mut options.modular);
        return Ok(encode(image, &options)?);
    };
    args.ma_budget.apply(&mut options.modular);
    args.ma_pruning.apply(&mut options.modular);
    options.modular.sample_limit = args.source_sample_limit;
    let params = CandidateParams::default().backend(effort)?;
    if args.source_scout && !(9..=10).contains(&effort) {
        return Err("--source-scout requires --source-effort 9 or 10".into());
    }
    if args.source_pilot && !(9..=10).contains(&effort) {
        return Err("--source-pilot requires --source-effort 9 or 10".into());
    }
    let settings = NativeBackendOptions::default();
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
    if args.source_search && effort == 11 {
        let result = run_native_tectonic(
            image,
            &params,
            Default::default(),
            Default::default(),
            &options,
            settings,
        )?;
        eprintln!(
            "source effort11 tournament: branch={:?} full_encodes={} winner={} winner_effort={}",
            result.output.report.branch,
            result.output.report.full_encodes,
            result.output.report.winner_index,
            10 - result.output.params.speed_tier
        );
        Ok(result.output.bytes)
    } else if args.source_search {
        let driver = if args.source_scout {
            DriverOptions::scout()
        } else {
            DriverOptions::default()
        };
        let result = if args.source_pilot {
            rs_jxl::scheduler::pilot::encode_with_options(
                image,
                &params,
                &options,
                rs_jxl::scheduler::pilot::PilotOptions {
                    group_race: args.source_progressive_groups.then(Default::default),
                    ..Default::default()
                },
            )?
        } else {
            run_native_pixel_search(image, &params, &driver, &options, settings)?
        };
        eprintln!(
            "source search: full_encodes={} probe_encodes={} winner={} winner_effort={}",
            result.output.report.full_encodes,
            result.output.report.probe_encodes,
            result.output.report.winner_index,
            10 - result.output.params.speed_tier,
        );
        if args.source_scout || args.source_pilot {
            eprintln!("source scout: winner_params={:?}", result.output.params);
        }
        Ok(result.output.bytes)
    } else {
        Ok(encode_native_candidate(&params, image, &options, settings)?.bytes)
    }
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
