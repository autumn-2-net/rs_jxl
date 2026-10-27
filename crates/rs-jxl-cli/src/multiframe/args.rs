//! Ordered input arguments, option translation and preflight validation.

use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use rs_jxl::metadata::Animation;
use rs_jxl::multiframe::{AutoOptions, Reference};

use crate::{Result, io, options::Tools, validators::parse_threads};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ReferenceArg {
    Auto,
    Previous,
    First,
    Full,
}

impl ReferenceArg {
    fn core(self) -> Reference {
        match self {
            Self::Auto => Reference::SmallestCrop,
            Self::Previous => Reference::Previous,
            Self::First => Reference::First,
            Self::Full => Reference::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Timebase {
    numerator: u32,
    denominator: u32,
}

pub(super) fn parse_tps(value: &str) -> std::result::Result<Timebase, String> {
    let (numerator, denominator) = value.split_once('/').unwrap_or((value, "1"));
    let parse = |part: &str| {
        part.parse::<u32>()
            .map_err(|_| "tps must be an integer or NUM/DEN".to_owned())
    };
    let timebase = Timebase {
        numerator: parse(numerator)?,
        denominator: parse(denominator)?,
    };
    Animation {
        ticks_per_second_numerator: timebase.numerator,
        ticks_per_second_denominator: timebase.denominator,
        ..Animation::default()
    }
    .validate()
    .map_err(|e| e.to_string())?;
    Ok(timebase)
}

fn positive_usize(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| "limit must be a positive integer fitting usize".into())
}

#[derive(Debug, Args)]
#[command(
    after_long_help = "Inputs are processed in exactly the given order, without sorting. \
All frames must have identical dimensions, decoded sample/alpha types and metadata. \
No bit-depth, alpha, color or orientation conversion is performed. RGB under zero \
alpha is retained. PNG palette/packed gray and tRNS use the existing lossless \
input expansion. APNG and CMYK/YCCK or non-8-bit JPEG are unsupported.\n\
JPEG input is pixel-lossless relative to zune-jpeg, not byte-reversible. Supported \
PNG/JPEG color metadata is retained. Unsupported metadata requires explicit \
--strip-metadata; supported color metadata is retained even when stripping. \
Discarding unsupported metadata may change displayed orientation.\n\
Durations are integer ticks: seconds = ticks * tps denominator / tps numerator. \
--duration applies to every input; --durations requires one comma-separated \
value per input. Zero ticks mean a non-display layer, not a one-tick substitute.\n\
Automatic selection measures encoded bytes, not elapsed time. --reference auto \
permits previous and first references. --retained-reference requires auto and \
rotates two reference slots to also reuse a retained older canvas; the first \
frame is no longer permanently pinned. --reference-lookahead additionally requires \
--retained-reference and uses exact differences against at most the next input to \
choose an old state to retain, without future trial encodes or a global size guarantee. \
Both reference flags are off by default. Mask and multi-rectangle candidates are \
enabled by default when lossless and compatible. These are multiframe strategies, \
not source-search or an effort mapping. Proven-bound pruning and adaptive rectangle \
proposals are enabled by default; --no-prune and --no-adaptive-rectangles disable \
them independently. Manual tool options and --threads apply.\n\
No extra CLI input, decoded-sample or retained-sequence byte cap is imposed. \
Frame, trial and trial-pixel budgets are unlimited unless explicitly supplied; \
zero budgets explicitly refuse work. \
Format bounds, size-overflow checks and allocation failures still apply; errors \
do not silently disable tools or discard frames.\n\
Use '-' for at most one input from stdin, or -o - for binary stdout. Candidate \
reports and warnings go to stderr. Encoding finishes before output is written. \
Output must differ from every input, including with --force."
)]
pub(crate) struct EncodeFramesArgs {
    #[command(flatten)]
    pub(super) tools: Tools,
    /// Ordered PNG/JPEG input paths (format detected by signature)
    #[arg(required = true, num_args = 1.., value_name = "INPUT")]
    pub(super) inputs: Vec<PathBuf>,
    /// JXL output path, or '-' for binary stdout
    #[arg(short, long, value_name = "OUTPUT")]
    pub(super) output: PathBuf,
    /// Uniform duration in ticks, including zero for a non-display layer
    #[arg(
        long,
        default_value_t = 100,
        conflicts_with = "durations",
        value_name = "TICKS"
    )]
    duration: u32,
    /// One tick count per input, comma-separated; may be repeated
    #[arg(long, value_delimiter = ',', num_args = 1, value_name = "TICKS,...")]
    durations: Vec<u32>,
    /// Ticks per second: positive integer or numerator/denominator
    #[arg(long, default_value = "1000", value_parser = parse_tps, value_name = "NUM[/DEN]")]
    tps: Timebase,
    /// Animation loop count; zero repeats indefinitely
    #[arg(long, default_value_t = 0, value_name = "N")]
    loops: u32,
    /// Reference candidates: auto permits both previous and first
    #[arg(long, value_enum, default_value = "auto")]
    reference: ReferenceArg,
    /// Rotate retained reference canvases (requires --reference auto)
    #[arg(long)]
    retained_reference: bool,
    /// Choose retained states using the next input (opt-in; no future encoding)
    #[arg(long, requires = "retained_reference")]
    pub(super) reference_lookahead: bool,
    /// Disable the safe mask candidate
    #[arg(long)]
    no_mask: bool,
    /// Disable the safe multi-rectangle candidate
    #[arg(long)]
    no_multi_rect: bool,
    /// Disable proven-bound pruning; complete every eligible candidate
    #[arg(long)]
    no_prune: bool,
    /// Disable additional cost-guided rectangle proposals; retain tile candidates
    #[arg(long)]
    no_adaptive_rectangles: bool,
    /// Tile side length for multi-rectangle planning
    #[arg(long, default_value_t = AutoOptions::default().tile_size, value_parser = clap::value_parser!(u32).range(1..), value_name = "PIXELS")]
    tile_size: u32,
    /// Extra rectangle grids (comma-separated); actual bytes select the winner
    #[arg(long, value_delimiter = ',', value_parser = clap::value_parser!(u32).range(1..), value_name = "PIXELS,...")]
    extra_tile_sizes: Vec<u32>,
    /// Maximum rectangles in a multi-rectangle candidate (positive integer)
    #[arg(long, default_value_t = AutoOptions::default().max_rectangles, value_parser = positive_usize, value_name = "N")]
    max_rectangles: usize,
    /// Logical input frame budget (omitted: unlimited; generated layers not counted)
    #[arg(long, default_value_t = AutoOptions::default().max_frames, hide_default_value = true, value_name = "N")]
    pub(super) max_frames: usize,
    /// Hard total trial-encode budget (omitted: unlimited)
    #[arg(long, default_value_t = AutoOptions::default().max_total_trials, hide_default_value = true, value_name = "N")]
    max_total_trials: usize,
    /// Hard trial-pixel budget (omitted: unlimited)
    #[arg(long, default_value_t = AutoOptions::default().max_trial_pixels, hide_default_value = true, value_name = "N")]
    max_trial_pixels: u64,
    /// Encoder worker count (1..=256)
    #[arg(long, default_value = "1", value_parser = parse_threads, value_name = "N")]
    pub(super) threads: usize,
    /// Write the candidate report as a single JSON line to stderr
    #[arg(long)]
    pub(super) report_json: bool,
    /// Discard unsupported metadata; retain supported color metadata and samples
    #[arg(long)]
    pub(super) strip_metadata: bool,
    /// Replace an existing output file; all input paths remain protected
    #[arg(long)]
    pub(super) force: bool,
}

impl EncodeFramesArgs {
    pub(super) fn auto_options(&self) -> AutoOptions {
        AutoOptions {
            reference: self.reference.core(),
            retained_reference: self.retained_reference,
            reference_lookahead: self.reference_lookahead,
            allow_mask: !self.no_mask,
            allow_multi_rect: !self.no_multi_rect,
            prune: !self.no_prune,
            adaptive_rectangles: !self.no_adaptive_rectangles,
            tile_size: self.tile_size,
            extra_tile_sizes: self.extra_tile_sizes.clone(),
            max_rectangles: self.max_rectangles,
            max_frames: self.max_frames,
            max_total_trials: self.max_total_trials,
            max_trial_pixels: self.max_trial_pixels,
        }
    }

    pub(super) fn validate_options(&self) -> Result<()> {
        if self.reference_lookahead && !self.retained_reference {
            return Err("--reference-lookahead requires --retained-reference".into());
        }
        if self.retained_reference && self.reference != ReferenceArg::Auto {
            return Err("--retained-reference requires --reference auto".into());
        }
        Ok(())
    }

    pub(super) fn animation(&self) -> Animation {
        Animation {
            ticks_per_second_numerator: self.tps.numerator,
            ticks_per_second_denominator: self.tps.denominator,
            loops: self.loops,
            timecodes: false,
        }
    }

    pub(super) fn frame_durations(&self) -> Result<Vec<u32>> {
        if !self.durations.is_empty() {
            if self.durations.len() != self.inputs.len() {
                return Err(format!(
                    "--durations requires one value per input ({} inputs, {} durations)",
                    self.inputs.len(),
                    self.durations.len()
                )
                .into());
            }
            return Ok(self.durations.clone());
        }
        let mut durations = Vec::new();
        durations.try_reserve_exact(self.inputs.len())?;
        durations.resize(self.inputs.len(), self.duration);
        Ok(durations)
    }

    pub(super) fn check_paths(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err("at least one input frame is required".into());
        }
        if self.inputs.len() > self.max_frames {
            return Err("input frame count exceeds --max-frames".into());
        }
        if self
            .inputs
            .iter()
            .filter(|p| p.as_path() == Path::new("-"))
            .count()
            > 1
        {
            return Err("stdin ('-') may appear at most once in the input sequence".into());
        }
        for path in &self.inputs {
            io::check_paths(path, &self.output, self.force)?;
        }
        Ok(())
    }
}
