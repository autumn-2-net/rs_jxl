// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Executable enc_lz77.cc routes from b866b3defec13494bfbaf705db93581cda118e37.
//! Source f32 RLE, fast 1/3/7/15/31-entry rings, careful 256-link lazy seed,
//! fast 7-entry large seed, 1/3/8/256-link optimal chains, and nominated
//! short32/long8 optimal buckets.
//! Source insertion order, zero-run chains, prefix hash, spatial probe limits,
//! seed gates, first-pass threshold, match-cost modes 0..7, repetition
//! skipping and bounded entropy refit are implemented, not projected onto caps.
//! Optc256 additionally supports history, wide-key, wide-ring, baseline static
//! indexes (including run-key and deep-prefix modes), mode4 shared-budget
//! replay and explicitly bounded stream dispatch on an installed caller pool.
//!
//! Cost tables and FastLog2f coefficients retain source precision (f32). Fused
//! versus non-fused polynomial arithmetic is explicit because C++ target/compiler
//! selection is not a portable bit-identical numerical contract. Sparse contexts
//! replace dense arrays; separate distance histograms permit every u32 context.
//! Default source DP endpoint relaxation uses exact dyadic constant-cost
//! intervals from enc_lz77_parse.h / enc_lz77_cost.h, preserving strict-cost
//! update order by the earliest origin on ties. It does not expand every length
//! in a long run. Range modes use the source heap parser's distinct explicit/
//! delayed tie rules. There is no retained-seed fallback or post-DP gate.
//! The with_state APIs retain ApplyLZ77's outer-vector acceptance separately
//! from commands. Entropy headers must not infer this state from match count.
//!
//! Optional short128/precise-ring, precomputed SIMD index tags, cold/warm seed
//! splitting, seed-boundary block parsers and shared indexed seeds are explicit.
//! Block parsers also support verified seed hints/sparse repair origins and
//! source lazy gain parsing with its distinct pricing and query controls.
//! Optional DP-only zero-tail query cuts and conservative literal-price bounds
//! retain the source float/double operation order and rounding margin.
//! Suffix nominations/window ranks/Kasai LCP use the bounded integer serial
//! API of the vendored pure-Rust libsais utility. Calendar, vector DP,
//! range dominance, literal-only jumps, dyadic trees and block envelopes now
//! have explicit source-gated options. Non-block indexed boundary DP and mutable
//! bucket dominance, chain depth/cache overrides are also selectable. Generic
//! b1/b3/b7/b15/b31 w3f/w3t and Optc1/3 dispatch is implemented. FAST_RANGES
//! and scalar/SIMD prefix choices retain source implicit and activation gates.
//! Optional exact bucket selections distinguish stored/compact aliases and
//! source parameter-first precedence without changing legacy explicit overrides.
//! Optional indexed stream arenas share the caller's cooperative executor;
//! profiling remains incomplete. See
//! docs/LZ77_PORT_STATUS.md. No serializer or encoder is included.
//! Static block indexes support guarded u16 values and word extension; shared
//! seed indexes auto-narrow, while ordinary non-block batches stay u32. Optional
//! u16 short/long bucket counts avoid query-time hash/offset lookups.
//! COMPACT uses 12-byte f32/coded-distance/length nodes in the existing endpoint
//! engine; precise-ring and the separate block/range heaps are not narrowed.
//! Decoder integration still emits length-3
//! and canonical physical distance symbols using the caller's wire configs.

use std::collections::BTreeMap;

use super::dp::Frontier;
#[cfg(test)]
use super::dp::visit_intervals;
use super::{
    Command, DISTANCE_COST, DistanceCodes, FilterPolicy, HybridUintConfig, LENGTH_COST, Literal,
    MIN_MATCH, Match, SourceLz77Method, WINDOW_SIZE, literals, reserved,
};
use crate::{Error, Result};

mod matchers;
use matchers::{Chain, Fast, Optimal};
mod range;
use range::Parser;
pub use range::{SourceBlockDyadic, SourceBlockEnvelope};
mod workers;
use workers::Schedule;
pub use workers::SourceParallelism;
mod index;
use index::StaticIndex;
pub use index::{SourceDeepPrefix, SourceIndexMode, SourceIndexTags, SourceRunKeys};
mod replay;
use replay::Classes;
mod blocks;
pub use blocks::{SourceBlockMode, SourceSeedBlocks, SourceSeedGuided};
mod suffix;
pub use suffix::SourceSuffixMode;
mod chain_batch;
mod distances;
mod histograms;
mod prices;
mod profile;
mod sais;
mod search;

/// enc_lz77.cc's JXL_LOSSLESS_LZ77_MATCH_COST modes 0..7.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum SourceMatchCost {
    /// 0: source default, smallest distance symbol and best-length-minus-two
    /// candidate pruning. This is not distance entropy ordering.
    #[default]
    SymbolId,
    /// 1: order distances by fitted cost, retain the source length pruning.
    Entropy,
    /// 2: retain short candidates and use their fitted-distance frontier.
    Frontier,
    /// 3: mode 2 followed by one additional DP pass fitted to its output.
    Refit,
    /// 4: two range passes with a shared 32 MiB first-pass match replay cache.
    RangeReplay = 4,
    /// 5: chain range parsing with distance-class reduction, no refit.
    Range = 5,
    /// 6: range parsing with one refit and a fresh chain on each pass.
    RangeRefit = 6,
    /// 7: choose mode 5 or 0 from accepted seed coverage across all streams.
    Auto = 7,
}

impl SourceMatchCost {
    fn range(self) -> bool {
        matches!(self, Self::RangeReplay | Self::Range | Self::RangeRefit)
    }
    fn keep_short(self) -> bool {
        matches!(
            self,
            Self::Frontier | Self::Refit | Self::RangeReplay | Self::Range | Self::RangeRefit
        )
    }
    fn passes(self) -> usize {
        if matches!(self, Self::Refit | Self::RangeReplay | Self::RangeRefit) {
            2
        } else {
            1
        }
    }
}

// enc_lz77_parse.h::PreferLZ77RangeParsing uses integer division, including
// ceil(tokens/2) coverage for odd totals. Empty matches must not divide by zero.
fn prefer_range(tokens: usize, matches: usize, covered: usize, long_covered: usize) -> bool {
    tokens >= 65_536
        && matches != 0
        && covered / matches >= 32
        && covered >= tokens - tokens / 2
        && long_covered >= tokens / 4
}

fn select_match_cost(
    requested: SourceMatchCost,
    total: usize,
    seed: &[Vec<Command>],
) -> SourceMatchCost {
    if requested != SourceMatchCost::Auto {
        return requested;
    }
    let (mut matches, mut covered, mut long_covered) = (0, 0, 0);
    for command in seed.iter().flatten() {
        if let Command::Match { length, .. } = *command {
            matches += 1;
            covered += length as usize;
            if length >= 64 {
                long_covered += length as usize;
            }
        }
    }
    if prefer_range(total, matches, covered, long_covered) {
        SourceMatchCost::Range
    } else {
        SourceMatchCost::SymbolId
    }
}

/// Explicit enc_lz77.cc bucket experiments; no process environment is read.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceBucketMode {
    /// Keep the existing bucket_matcher flag (false = chain, true = short32/long8).
    #[default]
    Default,
    /// short32/long8 plus twelve age-sampled long-bucket evictions.
    History,
    /// History, hashing every value of the eight-token long prefix.
    WideKey,
    /// short32/long32 with the full eight-token long hash, without history.
    WideRing,
    /// precise-ring: wide-ring candidates with double cumulative/DP costs.
    /// Histogram prices and length-plus-distance sums remain source f32.
    PreciseRing,
    /// BUCKETS=128: short128/long8, without short-ring fingerprints.
    Short128,
}

/// Exact BUCKETS aliases in the lossless dispatcher (cost spans on, no audit).
/// Unlike the older explicit bucket_mode override, the bucket_matcher parameter
/// takes precedence over every selection here, as in enc_lz77.cc:2452-2490.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceBucketSelection {
    /// Unset/unrecognized BUCKETS: chain, with the separate COMPACT switch.
    #[default]
    Unset,
    /// 32: stored-context nodes, serial, no index/shared-index eligibility.
    Short32,
    /// parallel32: compact short32/long8, eligible for parallel/index routes.
    Parallel32,
    History,
    WideKey,
    /// wide-ring: stored-context wide32 nodes, serial, no indexes.
    WideRing,
    /// parallel-wide: compact wide32, eligible for parallel/index routes.
    ParallelWide,
    PreciseRing,
    Short128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BucketRoute {
    mode: SourceBucketMode,
    enabled: bool,
    compact: bool,
    indexed: bool,
    parallel: bool,
}

/// MATCH_CACHE's independent override for ordinary optimal hash chains.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceChainCache {
    /// Unset: use PreferLZ77MatchCaching's source seed-compression gate.
    #[default]
    Auto,
    /// MATCH_CACHE=0.
    Disabled,
    /// MATCH_CACHE=1, including streams below the automatic size threshold.
    Enabled,
}

/// INTERVAL_DP=0..2, for non-overflowed static-index batches only.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceIntervalDp {
    #[default]
    Disabled,
    /// 1: all endpoints of every constant-price span.
    AllEndpoints,
    /// 2: first/last eight endpoints when a span contains more than sixteen.
    Boundaries,
}

/// FAST_RANGES=0..3: independent fast-ring insertion and match-extension tools.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SourceFastRanges {
    #[default]
    Disabled,
    /// 1: bulk rotation for repeated-value insertions.
    RepeatedInsert,
    /// 2: eight-value exact extension; implicit source prefix mode 2.
    MatchLength,
    /// 3: both tools; implicit source prefix mode 2.
    Both,
}

impl SourceFastRanges {
    fn repeated(self) -> bool {
        matches!(self, Self::RepeatedInsert | Self::Both)
    }
    fn extension(self) -> bool {
        matches!(self, Self::MatchLength | Self::Both)
    }
}

/// Explicit fast-matcher PREFIX_FILTER=0/1/2, separate from cache timing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceFastPrefix {
    Disabled,
    Scalar,
    Simd,
}

#[derive(Clone, Debug)]
pub struct SourceConfig {
    pub method: SourceLz77Method,
    /// Only Optc256 uses the source's nominated short32/long8 bucket override.
    pub bucket_matcher: bool,
    /// Optc256-only override. Range modes use chains regardless of this setting.
    pub bucket_mode: SourceBucketMode,
    /// Some selects exact source aliases and parameter-first precedence.
    /// None preserves the existing explicit bucket_mode API. With Some, the
    /// old bucket_mode is ignored; bucket_matcher=true selects Parallel32.
    pub bucket_selection: Option<SourceBucketSelection>,
    /// JXL_LOSSLESS_BUCKET_DISTANCES=1: bounded i16 special-symbol lookup.
    /// Mutable buckets opt in; static and suffix indexes always try this layout.
    pub bucket_distances: bool,
    /// COMPACT=1: 12-byte f32/u32/u32 DP nodes for chain Optc256. Default off.
    /// Parameter-selected short32 and eligible indexed endpoint routes use
    /// compact nodes automatically. Explicit other bucket modes ignore this
    /// chain switch; precise-ring and block/range heap parsers stay unchanged.
    pub compact_steps: bool,
    /// CHAIN_LIMIT: ordinary optimal chains only, not seeds or range modes.
    /// Source accepts 1..=256 and clamps to the method's maximum; invalid
    /// values fall back to that maximum. None keeps the source default.
    pub chain_limit: Option<u32>,
    /// MATCH_CACHE override; buckets, seeds and range modes ignore this option.
    pub chain_match_cache: SourceChainCache,
    /// Opt-in immutable chain256 SymbolId batches, not the bucket index.
    /// Keeps coarse-chain visit budgets and the global endpoint DP. Requires
    /// at least 65536 tokens and a seed with more than one wire token per four inputs;
    /// highly repetitive streams keep the demand-driven mutable chain.
    pub chain_batch: bool,
    /// Peak index/build/batch scratch allowance per stream, excluding the
    /// existing DP/input/output. Clamped by parallelism.scratch_bytes. A small
    /// budget disables refinement first, then falls back to the mutable chain.
    pub chain_batch_bytes: usize,
    /// DOMINANCE=1: cost-aware mutable-bucket candidates for modes 2/3.
    /// Does not enable indexed search or change the default candidate route.
    pub bucket_dominance: bool,
    /// INTERVAL_DP=1/2: non-block indexed batches with uint000. Overflow
    /// re-queries retain the full endpoint route, as in source.
    pub index_interval_dp: SourceIntervalDp,
    /// Opt-in stream dispatch on an already installed caller pool. None is
    /// serial; shared indexed seeds still use default logical partition limits.
    /// Process-global Rayon and environment variables are never used.
    pub parallelism: Option<SourceParallelism>,
    /// STREAM_ARENA=1: eligible indexed optimal streams share one caller-pool
    /// arena, preserving child index/block dispatch. Requires parallelism and
    /// an installed non-null runner; shared seed-index reuse excludes it.
    pub stream_arena: bool,
    /// ARENA_QUIET=1 for this stream arena; default false like the source.
    pub stream_arena_quiet: bool,
    /// Immutable short32/long8-or-32 counting-sort index. Only the
    /// Frontier/Optc256 non-history bucket route is eligible, as in source.
    pub index_mode: SourceIndexMode,
    /// Precomputed mandatory-prefix tags, independent of the mutable matcher.
    pub index_tags: SourceIndexTags,
    /// ANCHORS_FIRST=1: probe evicted spatial anchors before bucket candidates.
    pub index_anchors_first: bool,
    /// SHARED_INDEX=1 with INDEXED=2/BLOCK_DP=2: build once for the largest
    /// fast seed and retain that same immutable index for its first DP pass.
    pub shared_index: bool,
    /// SMALL_INDEX=1 permits >=8192 tokens without a pool; otherwise >=65536.
    pub small_index: bool,
    /// Source long-bucket zero-run keys; only active for an eligible index.
    pub index_run_keys: SourceRunKeys,
    /// Source deep64/deep16 proposal chains, including their 4/8 visit limits.
    pub index_deep_prefix: SourceDeepPrefix,
    /// PARALLEL_INDEX=1 stable counting sort on the installed caller pool.
    /// Source restrictions: no shared run keys or deep-prefix chains.
    pub parallel_index_build: bool,
    /// INDEX_CACHE=1: retain verified distance endpoints per match-batch task.
    pub index_match_cache: bool,
    /// COMPACT_VALUES=1: pack eligible static block-index values into u16 only
    /// if every value fits. Shared seed indexes select this automatically;
    /// non-block batches and suffix indexes ignore this switch.
    pub index_compact_values: bool,
    /// INDEX_COORDINATES=1: cache short/long bucket counts in one u16 per
    /// position. Applies to ordinary static indexes, not shared seed indexes.
    pub index_coordinates: bool,
    /// WORD_MATCH=1: u32/u16 word extension in eligible static-index blocks only.
    /// Suffix indexes and the shared seed index keep source ordinary extension.
    pub index_word_match: bool,
    /// INDEX_CACHE=2: 4096 direct-mapped source slots (at most 32 KiB).
    /// Overrides index_match_cache in blocks; non-block batches stay uncached.
    pub block_sparse_cache: bool,
    /// SHARED_FRONTIER=1: let the matcher fill the parser's bounded frontier.
    pub block_shared_frontier: bool,
    /// FRONTIER_CARRY=1: retain up to 16 verified endpoints between queries.
    pub block_frontier_carry: bool,
    /// DISTANCE_BOUND=1: source minimum-price bound; suffix search ignores it.
    pub block_distance_bound: bool,
    /// DISTANCE_TABLE=1: per-stream dense f32 prices, bounded by the window.
    pub block_distance_table: bool,
    /// DISTANCE_CLASSES=1: 128 source price classes; takes precedence at lookup
    /// when both caches are requested, except block_lazy which ignores classes.
    pub block_distance_classes: bool,
    /// BLOCK_DP=1/2: indexed parsing with seed-boundary destinations.
    pub block_mode: SourceBlockMode,
    /// BLOCK_GRAIN, bounded to 1..=8; no implicit global worker pool.
    pub block_grain: usize,
    /// WORK_CUTS=1: balance source-estimated work instead of token counts.
    pub block_work_cuts: bool,
    /// SEED_GUIDED=1/2: verified seed-distance hints and optional sparse origins.
    /// Only DP block parsing uses guidance; block_lazy keeps the source bypass.
    pub block_seed_guided: SourceSeedGuided,
    /// LAZY=1: source positive-gain block parser with one-token lookahead.
    /// Uses dense/direct prices; ignores guidance, carry, shared frontier and bound.
    pub block_lazy: bool,
    /// ZERO_TAIL=1: stop DP queries in the block's exact zero-price suffix.
    /// Literal relaxation and earlier copies ending in the suffix are retained.
    pub block_zero_tail: bool,
    /// LITERAL_BOUND=1: conservatively prune DP intervals against literal costs.
    /// The source lazy branch bypasses this bound and the zero-tail query cutoff.
    pub block_literal_bound: bool,
    /// CALENDAR=1: destination event lists with recycled slots. Interval blocks only.
    pub block_calendar: bool,
    /// VECTOR_DP=1: four SoA arrays with native masked f32 endpoint relaxation.
    /// Applies to both block DP modes, not LAZY or non-block optimal parsing.
    pub block_vector_dp: bool,
    /// RANGE_DOMINANCE=1: conservative full-16-endpoint-block cost certificates.
    pub block_range_dominance: bool,
    /// DP_JUMP=1: skip literal-only repetition runs in interval blocks. Source
    /// disables jumps with calendar/vector/dyadic storage and without active offers.
    pub block_dp_jump: bool,
    /// DYADIC_DP=1..3: source future-only tree and cached root path. Interval
    /// blocks only; takes precedence over envelope and calendar when combined.
    pub block_dyadic: SourceBlockDyadic,
    /// BLOCK_ENVELOPE=1..3: cheapest fully covered block offers. Interval blocks
    /// only; takes precedence over calendar, but not dyadic storage.
    pub block_envelope: SourceBlockEnvelope,
    /// SEED_BLOCKS=1/2, only the dominant >=262144-token fast seed is split.
    pub seed_blocks: SourceSeedBlocks,
    /// SUFFIX=1..4, only for eligible indexed block parsing. Uses the bounded
    /// integer serial API of the vendored pure-Rust libsais translation.
    pub suffix_mode: SourceSuffixMode,
    /// Conservative suffix construction scratch ceiling; ineligible inputs
    /// fall back to the static index before allocating suffix storage.
    pub suffix_scratch_bytes: usize,
    /// Otherwise the source switches from lazy chain256 to fast bucket7 when
    /// the combined number of input tokens reaches 65536.
    pub careful_first_pass: bool,
    pub match_cost: SourceMatchCost,
    pub literal_uint_config: HybridUintConfig,
    /// Source default 000; native callers may explicitly supply their wire400.
    /// Zero LSB-in-token bits are required for accelerated contiguous intervals.
    pub length_uint_config: HybridUintConfig,
    /// None uses source ANS=224 or force_huffman=512. This is an entropy symbol
    /// offset, not a matcher minimum length (which remains three).
    pub min_symbol: Option<u32>,
    pub force_huffman: bool,
    /// Independent of physical backward distance. The default covers all
    /// supported inputs; callers with a fixed alphabet must cap accordingly.
    pub max_match_length: usize,
    /// false is a diagnostic override of the source seed/greedy savings gates.
    pub require_global_savings: bool,
    /// Fast spatial probes only: upper-bound pruning without tags, grouped SIMD
    /// with an active prefix cache. Hash chains and bucket anchors do not use it.
    pub range_search: bool,
    /// FAST_RANGES=1..3, default off. Applies to all fast methods and fast
    /// optimal seeds; careful/optimal chains and indexed seeds ignore it.
    pub fast_ranges: SourceFastRanges,
    /// Exact source fast PREFIX_FILTER mode. None preserves the existing
    /// prefix_filter override; with that disabled, modes 2/3 implicitly select
    /// SIMD tags as in source. Some(Disabled) explicitly suppresses that default.
    /// Source builds lazily with range_search, eagerly without it.
    pub fast_prefix: Option<SourceFastPrefix>,
    /// Fast spatial prefix cache and optimal short/long mandatory-prefix tags
    /// reuse native SIMD helpers. Disabled is the source environment default.
    /// This does not filter hash chains. Buckets use tags immediately for either
    /// enabled policy; Lazy controls only the fast spatial cache build.
    pub prefix_filter: FilterPolicy,
    /// Select the source polynomial's MulAdd behavior explicitly. This does not
    /// use f64 log2 or a replacement entropy approximation.
    pub fused_cost_math: bool,
}

impl SourceConfig {
    fn bucket_route(&self) -> BucketRoute {
        use SourceBucketSelection as S;
        if let Some(selection) = self.bucket_selection {
            let selection = if self.bucket_matcher {
                S::Parallel32
            } else {
                selection
            };
            let mode = match selection {
                S::Unset | S::Short32 | S::Parallel32 => SourceBucketMode::Default,
                S::History => SourceBucketMode::History,
                S::WideKey => SourceBucketMode::WideKey,
                S::WideRing | S::ParallelWide => SourceBucketMode::WideRing,
                S::PreciseRing => SourceBucketMode::PreciseRing,
                S::Short128 => SourceBucketMode::Short128,
            };
            let indexed = matches!(selection, S::Parallel32 | S::ParallelWide);
            return BucketRoute {
                mode,
                enabled: selection != S::Unset,
                compact: indexed || (selection == S::Unset && self.compact_steps),
                indexed,
                parallel: indexed || selection == S::Unset,
            };
        }
        // Retain the original API's explicit candidate override and its wide
        // indexed/parallel shorthand. Exact source aliases are opt-in above.
        let mode = self.bucket_mode;
        let enabled = self.bucket_matcher || mode != SourceBucketMode::Default;
        BucketRoute {
            mode,
            enabled,
            compact: match mode {
                SourceBucketMode::Default => self.bucket_matcher || self.compact_steps,
                SourceBucketMode::WideRing => self.index_mode != SourceIndexMode::Disabled,
                _ => false,
            },
            indexed: enabled
                && matches!(mode, SourceBucketMode::Default | SourceBucketMode::WideRing),
            parallel: matches!(mode, SourceBucketMode::Default | SourceBucketMode::WideRing),
        }
    }

    fn uses_compact_steps(&self) -> bool {
        if self.method != SourceLz77Method::Optc256 || self.match_cost.range() {
            return false;
        }
        self.bucket_route().compact
    }

    fn precise(&self) -> bool {
        self.method == SourceLz77Method::Optc256
            && self.bucket_route().mode == SourceBucketMode::PreciseRing
            && !self.match_cost.range()
    }

    pub fn new(method: SourceLz77Method) -> Self {
        Self {
            method,
            ..Self::default()
        }
    }
    pub fn min_symbol(&self) -> u32 {
        self.min_symbol
            .unwrap_or(if self.force_huffman { 512 } else { 224 })
    }
    fn validate(&self) -> Result<()> {
        if !(1..=8).contains(&self.block_grain)
            || matches!(self.seed_blocks, SourceSeedBlocks::Warm { history } if history == 0 || history > 1 << 20)
        {
            return Err(Error::InvalidInput(
                "invalid source LZ77 block limits".into(),
            ));
        }
        if self
            .parallelism
            .is_some_and(|p| p.max_workers == 0 || p.max_workers > 64 || p.scratch_bytes == 0)
        {
            return Err(Error::InvalidInput(
                "invalid source LZ77 worker limits".into(),
            ));
        }
        if !(MIN_MATCH..=u32::MAX as usize).contains(&self.max_match_length)
            || self.length_uint_config.lsb_in_token() != 0
            || self
                .min_symbol()
                .checked_add(
                    self.length_uint_config
                        .encode((self.max_match_length - MIN_MATCH) as u32)
                        .token,
                )
                .is_none()
        {
            return Err(Error::InvalidInput(
                "invalid source LZ77 length configuration".into(),
            ));
        }
        Ok(())
    }
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            method: SourceLz77Method::Lz77b3w3t,
            bucket_matcher: false,
            bucket_mode: SourceBucketMode::Default,
            bucket_selection: None,
            bucket_distances: false,
            compact_steps: false,
            chain_limit: None,
            chain_match_cache: SourceChainCache::Auto,
            chain_batch: false,
            chain_batch_bytes: 128 << 20,
            bucket_dominance: false,
            index_interval_dp: SourceIntervalDp::Disabled,
            parallelism: None,
            stream_arena: false,
            stream_arena_quiet: false,
            index_mode: SourceIndexMode::Disabled,
            index_tags: SourceIndexTags::Default,
            index_anchors_first: false,
            shared_index: false,
            small_index: false,
            index_run_keys: SourceRunKeys::Disabled,
            index_deep_prefix: SourceDeepPrefix::Disabled,
            parallel_index_build: false,
            index_match_cache: false,
            index_word_match: false,
            index_compact_values: false,
            index_coordinates: false,
            block_sparse_cache: false,
            block_shared_frontier: false,
            block_frontier_carry: false,
            block_distance_bound: false,
            block_distance_table: false,
            block_distance_classes: false,
            block_mode: SourceBlockMode::Disabled,
            block_grain: 1,
            block_work_cuts: false,
            block_seed_guided: SourceSeedGuided::Disabled,
            block_lazy: false,
            block_zero_tail: false,
            block_literal_bound: false,
            block_calendar: false,
            block_vector_dp: false,
            block_range_dominance: false,
            block_dp_jump: false,
            block_dyadic: SourceBlockDyadic::Disabled,
            block_envelope: SourceBlockEnvelope::Disabled,
            seed_blocks: SourceSeedBlocks::Disabled,
            suffix_mode: SourceSuffixMode::Disabled,
            suffix_scratch_bytes: 256 << 20,
            careful_first_pass: false,
            match_cost: SourceMatchCost::SymbolId,
            literal_uint_config: HybridUintConfig::default(),
            length_uint_config: HybridUintConfig::new(0, 0, 0).unwrap(),
            min_symbol: None,
            force_huffman: false,
            max_match_length: u32::MAX as usize,
            require_global_savings: true,
            range_search: true,
            fast_ranges: SourceFastRanges::Disabled,
            fast_prefix: None,
            prefix_filter: FilterPolicy::Disabled,
            fused_cost_math: false,
        }
    }
}

/// ApplyLZ77 acceptance, not a scan for Match commands. A rejected call keeps
/// original literals in commands for replay/compatibility, but enabled=false.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceParseOutcome {
    pub commands: Vec<Command>,
    pub enabled: bool,
}

/// One shared histogram/gate decision for all streams. Empty/literal-only
/// streams still use the shared LZ77 entropy header when enabled=true.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceStreamsOutcome {
    pub commands: Vec<Vec<Command>>,
    pub enabled: bool,
}

// base/fast_math-inl.h::FastLog2f and rational_polynomial-inl.h. Inputs are
// positive normal probabilities because validated stream totals fit u32.
#[allow(clippy::excessive_precision)]
fn fast_log2(value: f32, fused: bool) -> f32 {
    let bits = value.to_bits() as i32;
    let exponent = bits.wrapping_sub(0x3f2aaaab) >> 23;
    let x = f32::from_bits(bits.wrapping_sub(exponent.wrapping_shl(23)) as u32) - 1.0;
    let ma = |a: f32, b: f32, c: f32| if fused { a.mul_add(b, c) } else { a * b + c };
    let p = ma(
        ma(7.4245873327820566E-01, x, 1.4287160470083755E+00),
        x,
        -1.8503833400518310E-06,
    );
    let q = ma(
        ma(1.7409343003366853E-01, x, 1.0096718572241148E+00),
        x,
        9.9032814277590719E-01,
    );
    p / q + exponent as f32
}

type Counts = BTreeMap<u32, BTreeMap<u32, usize>>;

struct ContextCost {
    bits: BTreeMap<u32, f32>,
    penalty: f32,
}
struct Costs {
    literals: BTreeMap<u32, ContextCost>,
    distances: ContextCost,
}

fn context_cost(counts: BTreeMap<u32, usize>, config: &SourceConfig) -> ContextCost {
    let total: usize = counts.values().sum();
    let inverse = 1.0f32 / (total as f32 + 1e-8f32);
    let mut total_cost = 0.0f32;
    let bits = counts
        .into_iter()
        .map(|(symbol, count)| {
            let mut cost = if count == total {
                0.0
            } else {
                -fast_log2(count as f32 * inverse, config.fused_cost_math)
            };
            if config.force_huffman {
                cost = cost.ceil();
            }
            total_cost += cost * count as f32;
            (symbol, cost)
        })
        .collect();
    ContextCost {
        bits,
        penalty: (6.0 - total_cost * inverse).max(0.0),
    }
}

impl Costs {
    fn from_counts(counts: Counts, distances: BTreeMap<u32, usize>, config: &SourceConfig) -> Self {
        Self {
            literals: counts
                .into_iter()
                .map(|(ctx, counts)| (ctx, context_cost(counts, config)))
                .collect(),
            distances: context_cost(distances, config),
        }
    }
    fn raw(inputs: &[&[Literal]], config: &SourceConfig) -> Self {
        let _profile = crate::profile::scope("lz77.source.raw_histogram");
        let count_profile = crate::profile::scope("lz77.source.raw_histogram.count");
        let tokens = inputs
            .iter()
            .fold(0usize, |total, input| total.saturating_add(input.len()));
        let mut counts = histograms::RawCounts::for_inputs(inputs, tokens);
        for input in inputs {
            for literal in *input {
                let token = config.literal_uint_config.encode(literal.value).token;
                counts.increment(literal.context, token);
            }
        }
        drop(count_profile);
        let merge_profile = crate::profile::scope("lz77.source.raw_histogram.merge");
        let counts = counts.into_counts();
        drop(merge_profile);
        let _cost_profile = crate::profile::scope("lz77.source.raw_histogram.costs");
        Self::from_counts(counts, BTreeMap::new(), config)
    }
    fn fitted(seed: &[Vec<Command>], widths: &[usize], config: &SourceConfig) -> Self {
        let _profile = crate::profile::scope("lz77.source.fitted_histogram");
        let count_profile = crate::profile::scope("lz77.source.fitted_histogram.count");
        let tokens = seed
            .iter()
            .fold(0usize, |total, stream| total.saturating_add(stream.len()));
        let mut counts = histograms::ContextCounts::new(tokens);
        let mut distances = histograms::SymbolCounts::new();
        for (index, stream) in seed.iter().enumerate() {
            let codes = DistanceCodes::new(widths.get(index).copied().unwrap_or(0), WINDOW_SIZE);
            for &command in stream {
                let (context, symbol) = match command {
                    Command::Literal(literal) => (
                        literal.context,
                        config.literal_uint_config.encode(literal.value).token,
                    ),
                    Command::Match {
                        context,
                        length,
                        distance,
                    } => {
                        // As in C++ SymbolCostEstimator, ordinary distance tokens
                        // are histogrammed with params.UintConfig, NOT length uint.
                        let token = config
                            .literal_uint_config
                            .encode(codes.symbol(distance as usize))
                            .token;
                        distances.increment(token);
                        (
                            context,
                            config.min_symbol()
                                + config
                                    .length_uint_config
                                    .encode(length - MIN_MATCH as u32)
                                    .token,
                        )
                    }
                };
                counts.increment(context, symbol);
            }
        }
        drop(count_profile);
        let merge_profile = crate::profile::scope("lz77.source.fitted_histogram.merge");
        let counts = counts.into_counts();
        let distances = distances.into_counts();
        drop(merge_profile);
        let _cost_profile = crate::profile::scope("lz77.source.fitted_histogram.costs");
        Self::from_counts(counts, distances, config)
    }
    fn bits(&self, context: u32, symbol: u32) -> f32 {
        self.literals
            .get(&context)
            .and_then(|row| row.bits.get(&symbol))
            .copied()
            .unwrap_or(12.0)
    }
    fn penalty(&self, context: u32) -> f32 {
        self.literals.get(&context).map_or(6.0, |row| row.penalty)
    }
    fn cumulative(&self, input: &[Literal], config: &SourceConfig) -> Result<Vec<f32>> {
        let mut sums = reserved(input.len() + 1)?;
        sums.push(0.0);
        for literal in input {
            let encoded = config.literal_uint_config.encode(literal.value);
            sums.push(
                self.bits(literal.context, encoded.token)
                    + encoded.nbits as f32
                    + sums.last().copied().unwrap(),
            );
        }
        Ok(sums)
    }
    #[cfg(test)]
    fn optimal_cumulative(&self, input: &[Literal], config: &SourceConfig) -> Result<Vec<f64>> {
        let mut sums = reserved(input.len() + 1)?;
        sums.push(0.0);
        for literal in input {
            let encoded = config.literal_uint_config.encode(literal.value);
            // C++ adds the two f32 operands before promotion to Cost=double.
            let price = self.bits(literal.context, encoded.token) + encoded.nbits as f32;
            let previous = *sums.last().unwrap();
            sums.push(if config.precise() {
                previous + f64::from(price)
            } else {
                f64::from(price + previous as f32)
            });
        }
        Ok(sums)
    }
    fn length(&self, context: u32, length: usize, config: &SourceConfig) -> f32 {
        let encoded = config
            .length_uint_config
            .encode((length - MIN_MATCH) as u32);
        encoded.nbits as f32 + self.bits(context, config.min_symbol() + encoded.token)
    }
    fn distance(&self, symbol: u32) -> f32 {
        // Source DistCost deliberately uses default420 even if UintConfig for
        // histogram fitting above is overridden. Preserve that distinction.
        let encoded = HybridUintConfig::default().encode(symbol);
        encoded.nbits as f32
            + self
                .distances
                .bits
                .get(&encoded.token)
                .copied()
                .unwrap_or(12.0)
    }
}

fn static_cost(matched: Match, context: u32, costs: &Costs) -> f32 {
    static_cost_with_penalty(matched, costs.penalty(context))
}

#[inline]
fn static_cost_with_penalty(matched: Match, penalty: f32) -> f32 {
    let len = HybridUintConfig::new(1, 0, 0)
        .unwrap()
        .encode((matched.length - MIN_MATCH) as u32);
    let length =
        LENGTH_COST[(len.token as usize).min(LENGTH_COST.len() - 1)] as f32 + len.nbits as f32;
    length + static_distance(matched.symbol) + penalty
}

fn static_distance(symbol: u32) -> f32 {
    let dist = HybridUintConfig::new(7, 0, 0).unwrap().encode(symbol);
    DISTANCE_COST[(dist.token as usize).min(DISTANCE_COST.len() - 1)] as f32 + dist.nbits as f32
}

fn fast_stream<const N: usize>(
    input: &[Literal],
    values: &[u32],
    width: usize,
    config: &SourceConfig,
    penalties: &prices::Penalties<'_>,
    sums: &[f32],
    saving: &mut f32,
) -> Result<Vec<Command>> {
    let _profile = crate::profile::scope("lz77.source.fast_seed");
    let init_profile = crate::profile::scope("lz77.source.seed_matcher_init");
    let mut matcher = Fast::<N>::new(values, width, config)?;
    drop(init_profile);
    let mut output = reserved(input.len())?;
    let mut position = 0;
    while position < input.len() {
        if input.len() - position < MIN_MATCH {
            output.push(Command::Literal(input[position]));
            position += 1;
            continue;
        }
        let matched = matcher.find(position);
        if matched.length < MIN_MATCH {
            output.push(Command::Literal(input[position]));
            position += 1;
            continue;
        }
        let raw = sums[position + matched.length] - sums[position];
        let cost = static_cost_with_penalty(matched, penalties.get(input[position].context));
        let consumed = if config.method.runtime_cost_comparison() && cost > raw {
            let count = if raw < matched.length as f32 && matched.length > MIN_MATCH {
                matched.length - MIN_MATCH + 1
            } else {
                1
            };
            output.extend(
                input[position..position + count]
                    .iter()
                    .copied()
                    .map(Command::Literal),
            );
            count
        } else {
            *saving += raw - cost;
            output.push(matched.command(input[position].context));
            matched.length
        };
        matcher.update_range(position + 1, position + consumed);
        position += consumed;
    }
    Ok(output)
}

fn careful_stream(
    input: &[Literal],
    values: &[u32],
    width: usize,
    config: &SourceConfig,
    costs: &Costs,
    sums: &[f32],
    saving: &mut f32,
) -> Result<Vec<Command>> {
    let _profile = crate::profile::scope("lz77.source.careful_seed");
    let init_profile = crate::profile::scope("lz77.source.seed_matcher_init");
    let mut matcher = Chain::new(values, width, 256, config.max_match_length)?;
    drop(init_profile);
    let mut output = reserved(input.len())?;
    let mut position = 0;
    let mut already_updated = false;
    while position < input.len() {
        if !already_updated {
            matcher.update(position);
        }
        already_updated = false;
        let mut matched = matcher.find(position);
        if matched.length < MIN_MATCH {
            output.push(Command::Literal(input[position]));
            position += 1;
            continue;
        }
        if matched.length < 256 && position + 1 < input.len() {
            matcher.update(position + 1);
            already_updated = true;
            let next = matcher.find(position + 1);
            if next.length > matched.length {
                output.push(Command::Literal(input[position]));
                position += 1;
                already_updated = false;
                matched = next;
            }
        }
        let raw = sums[position + matched.length] - sums[position];
        let cost = static_cost(matched, input[position].context, costs);
        if cost <= raw {
            output.push(matched.command(input[position].context));
            *saving += raw - cost;
        } else {
            output.extend(
                input[position..position + matched.length]
                    .iter()
                    .copied()
                    .map(Command::Literal),
            );
        }
        matcher.update_range(
            position + if already_updated { 2 } else { 1 },
            position + matched.length,
        );
        already_updated = false;
        position += matched.length;
    }
    Ok(output)
}

#[derive(Default)]
struct Work {
    #[cfg(test)]
    queries: usize,
    #[cfg(test)]
    intervals: usize,
    #[cfg(test)]
    passes: usize,
    #[cfg(test)]
    cached_streams: usize,
    #[cfg(test)]
    chain_batch_streams: usize,
    #[cfg(test)]
    careful_seed: bool,
    #[cfg(test)]
    fast_tasks: usize,
    #[cfg(test)]
    optimal_tasks: usize,
    #[cfg(test)]
    arena_passes: usize,
    #[cfg(test)]
    arena_roots: usize,
    #[cfg(test)]
    selected_cost: Option<SourceMatchCost>,
    #[cfg(test)]
    indexed_streams: usize,
    #[cfg(test)]
    compact_index_streams: usize,
    #[cfg(test)]
    coordinate_index_streams: usize,
    #[cfg(test)]
    compact_dp_streams: usize,
    #[cfg(test)]
    block_streams: usize,
    #[cfg(test)]
    suffix_streams: usize,
    #[cfg(test)]
    shared_index_builds: usize,
    #[cfg(test)]
    replayed_positions: usize,
    #[cfg(test)]
    full_replay_streams: usize,
    #[cfg(test)]
    partial_replay_streams: usize,
    #[cfg(test)]
    replay_bytes: usize,
}

impl Work {
    fn merge(&mut self, _other: Self) {
        #[cfg(test)]
        {
            self.queries += _other.queries;
            self.intervals += _other.intervals;
            self.cached_streams += _other.cached_streams;
            self.chain_batch_streams += _other.chain_batch_streams;
            self.indexed_streams += _other.indexed_streams;
            self.compact_index_streams += _other.compact_index_streams;
            self.coordinate_index_streams += _other.coordinate_index_streams;
            self.compact_dp_streams += _other.compact_dp_streams;
            self.block_streams += _other.block_streams;
            self.suffix_streams += _other.suffix_streams;
        }
    }
}

#[cfg(test)]
fn optimal_stream(
    input: &[Literal],
    values: &[u32],
    width: usize,
    config: &SourceConfig,
    costs: &Costs,
    seed_tokens: usize,
    _work: &mut Work,
) -> Result<Vec<Command>> {
    optimal_stream_replay(
        input,
        values,
        width,
        config,
        &prices::Prices::new(costs)?,
        seed_tokens,
        _work,
        None,
    )
}

// enc_lz77_cost.h::VisitLZ77CostRange: source mode 2 intentionally omits
// interior endpoints of long constant-price runs. This is not exact DP.
fn visit_endpoint_ranges(
    begin: usize,
    end: usize,
    boundary_only: bool,
    mut visit: impl FnMut(usize, usize),
) {
    if boundary_only && end - begin > 16 {
        visit(begin, begin + 8);
        visit(end - 8, end);
    } else {
        visit(begin, end);
    }
}

// Preserve the existing per-stream parser contract plus its optional replay state.
#[allow(clippy::too_many_arguments)]
fn optimal_stream_replay(
    input: &[Literal],
    values: &[u32],
    width: usize,
    config: &SourceConfig,
    prices: &prices::Prices<'_>,
    seed_tokens: usize,
    _work: &mut Work,
    mut replay: Option<replay::Pass<'_>>,
) -> Result<Vec<Command>> {
    let _profile = crate::profile::scope("lz77.source.optimal");
    let setup_profile = crate::profile::scope("lz77.source.matcher_init");
    let indexed =
        config.index_mode != SourceIndexMode::Disabled && values.len() >= index::minimum(config);
    let index = if indexed {
        // Source COMPACT_VALUES/WORD_MATCH are passed only to block builders.
        Some(StaticIndex::new(
            values,
            width,
            &SourceConfig {
                index_word_match: false,
                index_compact_values: false,
                ..config.clone()
            },
        )?)
    } else {
        None
    };
    let mut indexed_batch = index
        .as_ref()
        .map(|index| index::Batch::new(index, width, config))
        .transpose()?;
    let full_replay = replay.as_ref().is_some_and(|pass| pass.full_replay());
    let mut chain_batch = chain_batch::Batch::new(values, width, config, seed_tokens)?;
    let mut matcher = if indexed || full_replay || chain_batch.is_some() {
        None
    } else {
        Some(Optimal::new(values, width, config, seed_tokens)?)
    };
    drop(setup_profile);
    #[cfg(test)]
    {
        _work.cached_streams +=
            usize::from(matcher.as_ref().is_some_and(|m| m.match_cache_enabled()));
        _work.indexed_streams += usize::from(indexed);
        _work.chain_batch_streams += usize::from(chain_batch.is_some());
        if let Some(index) = &index {
            let (compact, coordinates) = index.layout();
            _work.compact_index_streams += usize::from(compact);
            _work.coordinate_index_streams += usize::from(coordinates);
        }
        _work.full_replay_streams += usize::from(full_replay);
        _work.partial_replay_streams +=
            usize::from(replay.as_ref().is_some_and(|pass| pass.partial_replay()));
    }
    let range = config.match_cost.range();
    let sums = if range {
        Vec::new()
    } else {
        prices.optimal_cumulative(input, config)?
    };
    let compact = config.uses_compact_steps();
    let parser_profile = crate::profile::scope("lz77.source.dp_init");
    let mut parser = Parser::new(input.len(), range, compact, width)?;
    drop(parser_profile);
    #[cfg(test)]
    {
        _work.compact_dp_streams += usize::from(compact);
    }
    let mut frontier = Frontier::new(256)?;
    let mut classes = Classes::new()?;
    let (mut skip, mut repeats, mut previous_end, mut previous_symbol) = (0, 0, 0, 0);
    let mut profile = profile::Profile::new(input.len());
    if crate::profile::enabled() {
        crate::profile::note(
            "lz77.source.route",
            format_args!(
                "tokens={} method={:?} mode={:?} indexed={} bucket={} chain_limit={:?} seed_tokens={} chain_batch={}",
                input.len(),
                config.method,
                config.match_cost,
                indexed,
                config.bucket_route().enabled,
                config.chain_limit,
                seed_tokens,
                chain_batch.is_some(),
            ),
        );
    }
    for position in 0..input.len() {
        let mut sample = profile.sample(position);
        if let Some(matcher) = &mut matcher {
            matcher.update(position);
        }
        profile.index(&mut sample);
        parser.advance(position);
        let base = parser.cost(position);
        let literal_cost = if range {
            let encoded = config.literal_uint_config.encode(input[position].value);
            f64::from(
                base as f32
                    + (prices.bits(input[position].context, encoded.token) + encoded.nbits as f32),
            )
        } else if config.precise() {
            base + sums[position + 1] - sums[position]
        } else {
            // Default optimal uses left-associative cumulative subtraction.
            f64::from(base as f32 + sums[position + 1] as f32 - sums[position] as f32)
        };
        parser.literal(position, literal_cost);
        profile.dp(&mut sample);
        let replaying = replay.as_ref().is_some_and(|pass| pass.replaying(position));
        if !replaying && skip > 0 {
            skip -= 1;
            continue;
        }
        #[cfg(test)]
        {
            _work.queries += usize::from(!replaying);
        }
        frontier.entries.clear();
        if range {
            classes.clear();
        }
        let mut bounded_index_frontier = false;
        if replaying {
            if !replay.as_mut().unwrap().load(position, width, &mut classes) {
                continue;
            }
            #[cfg(test)]
            {
                _work.replayed_positions += 1;
            }
        } else {
            let mut add = |matched: Match| {
                if range {
                    classes.add(matched);
                    return;
                }
                let order = if config.match_cost == SourceMatchCost::SymbolId {
                    matched.symbol as f64
                } else {
                    f64::from(prices.distance(matched.symbol))
                };
                frontier.add(matched, order);
            };
            if let Some(batch) = &mut chain_batch {
                batch.find_matches(position, &mut add);
            } else if let Some(index) = &index {
                indexed_batch.as_mut().unwrap().find_matches(
                    index,
                    position,
                    |symbol| prices.distance(symbol),
                    &mut add,
                );
                bounded_index_frontier = indexed_batch
                    .as_ref()
                    .unwrap()
                    .has_bounded_frontier(position);
            } else if config.bucket_dominance
                && matches!(
                    config.match_cost,
                    SourceMatchCost::Frontier | SourceMatchCost::Refit
                )
            {
                matcher.as_mut().unwrap().find_cost_matches(
                    position,
                    |symbol| prices.distance(symbol),
                    &mut add,
                );
            } else {
                matcher.as_mut().unwrap().find_matches(
                    position,
                    config.match_cost.keep_short(),
                    &mut add,
                );
            }
            if let Some(replay) = &mut replay {
                replay.record(position, &classes.entries)?;
            }
        }
        if range {
            for &matched in &classes.entries {
                frontier.add(matched, f64::from(prices.distance(matched.symbol)));
            }
        }
        profile.matching(&mut sample);
        let boundary_only = bounded_index_frontier
            && config.index_interval_dp == SourceIntervalDp::Boundaries
            && config.length_uint_config == HybridUintConfig::new(0, 0, 0).unwrap();
        let mut begin = MIN_MATCH;
        let row = prices.row(input[position].context);
        for entry in &frontier.entries {
            let matched = entry.matched;
            let distance_price = prices.distance(matched.symbol);
            let mut relax = |first, end, length_price: f32| {
                let price = length_price + distance_price;
                let cost = if config.precise() {
                    base + f64::from(price)
                } else {
                    f64::from(base as f32 + price)
                };
                visit_endpoint_ranges(first, end, boundary_only, |start, stop| {
                    parser.interval(position, start, stop, matched.distance, cost);
                    #[cfg(test)]
                    {
                        _work.intervals += 1;
                    }
                });
            };
            if range && config.length_uint_config != HybridUintConfig::new(0, 0, 0).unwrap() {
                // ApplyLZ77_RangeParsed only queues length spans for uint000.
                // Other configs write explicit endpoints, including tie order.
                for length in begin..=matched.length {
                    relax(length, length + 1, row.length(length, config));
                }
            } else {
                row.intervals(begin, matched.length + 1, config, &mut relax);
            }
            begin = matched.length + 1;
        }
        profile.dp(&mut sample);
        // Replay preserves the first pass's queried origins. It must neither
        // recompute skip decisions nor seed repetition state for a spilled tail.
        if replaying {
            continue;
        }
        let Some(last) = frontier.entries.last() else {
            repeats = 0;
            continue;
        };
        let matched = last.matched;
        let end = position + matched.length;
        if matched.symbol == 0
            || (matched.symbol == 1 && width != 0)
            || (matched.length >= 256 && end == previous_end && matched.symbol == previous_symbol)
        {
            repeats += 1;
        } else {
            repeats = 0;
        }
        previous_end = end;
        previous_symbol = matched.symbol;
        if repeats >= 8 && matched.length >= 9 {
            skip = matched.length - 9;
            repeats = 0;
        }
    }
    parser.advance(input.len());
    let _backtrack_profile = crate::profile::scope("lz77.source.backtrack");
    parser.backtrack(input)
}

fn range_replayed(
    inputs: &[&[Literal]],
    values: &[Vec<u32>],
    widths: &[usize],
    config: &SourceConfig,
    mut seed: Vec<Vec<Command>>,
    cache_limit: usize,
    work: &mut Work,
) -> Result<SourceStreamsOutcome> {
    let mut caches: Vec<_> = (0..inputs.len())
        .map(|_| replay::Cache::default())
        .collect();
    let mut budget = replay::Budget::new(cache_limit);
    for pass in 0..2 {
        let costs = Costs::fitted(&seed, widths, config);
        let prices = prices::Prices::new(&costs)?;
        let mut parsed = reserved(inputs.len())?;
        #[cfg(test)]
        {
            work.passes += 1;
            work.optimal_tasks = 1;
        }
        for (index, input) in inputs.iter().enumerate() {
            if input.is_empty() {
                parsed.push(Vec::new());
                continue;
            }
            let replay = replay::Pass::new(&mut caches[index], &mut budget, pass != 0);
            parsed.push(optimal_stream_replay(
                input,
                &values[index],
                widths.get(index).copied().unwrap_or(0),
                config,
                &prices,
                0,
                work,
                Some(replay),
            )?);
        }
        seed = parsed;
    }
    #[cfg(test)]
    {
        work.replay_bytes = budget.used;
    }
    Ok(SourceStreamsOutcome {
        commands: seed,
        enabled: true,
    })
}

fn validate_inputs(inputs: &[&[Literal]], config: &SourceConfig) -> Result<usize> {
    config.validate()?;
    let mut total = 0usize;
    for input in inputs {
        if input.len() > i32::MAX as usize {
            return Err(Error::Limit(
                "source LZ77 stream exceeds signed C++ index domain",
            ));
        }
        total = total
            .checked_add(input.len())
            .ok_or(Error::Limit("source LZ77 token total"))?;
        if input.iter().any(|literal| {
            config.literal_uint_config.encode(literal.value).token >= config.min_symbol()
        }) {
            return Err(Error::InvalidInput(
                "source LZ77 literal and length alphabets overlap".into(),
            ));
        }
    }
    if total > u32::MAX as usize {
        return Err(Error::Limit("source LZ77 histogram total"));
    }
    Ok(total)
}

fn wire_token_count(commands: &[Command]) -> usize {
    commands
        .iter()
        .map(|command| 1 + usize::from(matches!(command, Command::Match { .. })))
        .sum()
}

fn run(
    inputs: &[&[Literal]],
    widths: &[usize],
    config: &SourceConfig,
    _work: &mut Work,
) -> Result<SourceStreamsOutcome> {
    let total = validate_inputs(inputs, config)?;
    if total == 0 {
        return Ok(SourceStreamsOutcome {
            commands: inputs
                .iter()
                .map(|input| literals(input))
                .collect::<Result<_>>()?,
            // Empty input is rejected by the normal source gate. The existing
            // diagnostic override bypasses that gate, but no streams still
            // means an empty outer vector in enc_ans.cc::BuildAndEncodeHistograms.
            enabled: !config.require_global_savings && !inputs.is_empty(),
        });
    }
    let optimal = config.method.chain_maximum().is_some();
    let careful = optimal && (config.careful_first_pass || total < 65_536);
    #[cfg(test)]
    {
        _work.careful_seed = careful;
    }
    let values_profile = crate::profile::scope("lz77.source.values");
    let mut values = reserved(inputs.len())?;
    for input in inputs {
        let mut data = reserved(input.len())?;
        data.extend(input.iter().map(|literal| literal.value));
        values.push(data);
    }
    drop(values_profile);
    let mut shared = if !careful {
        blocks::shared_index(&values, widths, config)?
    } else {
        None
    };
    #[cfg(test)]
    {
        _work.shared_index_builds += usize::from(shared.is_some());
    }
    let parts = blocks::seed_parts(inputs, config, optimal && !careful && shared.is_none())?;
    let seed_inputs: Vec<_> = parts
        .iter()
        .map(|part| &inputs[part.owner][part.begin..part.end])
        .collect();
    let seed_total: usize = seed_inputs.iter().map(|s| s.len()).sum();
    let costs = Costs::raw(&seed_inputs, config);
    let seed_prices = prices::Prices::new(&costs)?;
    let seed_penalties = if careful {
        None
    } else {
        Some(prices::Penalties::new(&costs)?)
    };
    let encode_seed = |index: usize, saving: &mut f32| -> Result<(Vec<Command>, f32)> {
        let part = &parts[index];
        let input = seed_inputs[index];
        let data = &values[part.owner][part.begin..part.end];
        let width = widths.get(part.owner).copied().unwrap_or(0);
        if let Some((owner, index)) = &shared
            && *owner == part.owner
        {
            let (commands, decrease, raw) = blocks::indexed_seed(input, index, config, &costs)?;
            *saving += decrease;
            return Ok((commands, raw));
        }
        let sums = seed_prices.cumulative(input, config)?;
        let commands = if careful {
            careful_stream(input, data, width, config, &costs, &sums, saving)?
        } else {
            let penalties = seed_penalties.as_ref().expect("fast-seed penalty table");
            match config.method.fast_capacity() {
                Some(1) => fast_stream::<1>(input, data, width, config, penalties, &sums, saving)?,
                Some(3) => fast_stream::<3>(input, data, width, config, penalties, &sums, saving)?,
                Some(7) | None => {
                    fast_stream::<7>(input, data, width, config, penalties, &sums, saving)?
                }
                Some(15) => {
                    fast_stream::<15>(input, data, width, config, penalties, &sums, saving)?
                }
                Some(31) => {
                    fast_stream::<31>(input, data, width, config, penalties, &sums, saving)?
                }
                _ => unreachable!("source fast bucket capacity"),
            }
        };
        Ok((commands, sums[input.len()]))
    };
    let schedule = Schedule::new(&seed_inputs, config, true);
    // ApplyLZ77_LZ77 disables stream parallelism when an indexed seed owns
    // intra-stream tasks. Do not silently turn that into nested pool work.
    let parallel_seed = !careful && shared.is_none() && schedule.tasks > 1;
    #[cfg(test)]
    {
        _work.fast_tasks = if parallel_seed { schedule.tasks } else { 1 };
    }
    let mut seed = reserved(seed_inputs.len())?;
    let (mut saving, mut raw) = (0.0f32, 0.0f32);
    if parallel_seed {
        let results = schedule.collect(seed_inputs.len(), |index| {
            let mut decrease = 0.0;
            let (commands, raw) = encode_seed(index, &mut decrease)?;
            Ok((commands, raw, decrease))
        })?;
        // enc_lz77.cc reduces per-stream f32 savings in input order. The
        // serial branch instead adds each match into the shared accumulator.
        for result in results {
            let (commands, raw_cost, decrease) = result.unwrap_or_default();
            seed.push(commands);
            raw += raw_cost;
            saving += decrease;
        }
    } else {
        for index in 0..seed_inputs.len() {
            let (commands, raw_cost) = encode_seed(index, &mut saving)?;
            seed.push(commands);
            raw += raw_cost;
        }
    }
    drop(seed_penalties);
    drop(seed_prices);
    let output_tokens: usize = seed.iter().map(|commands| wire_token_count(commands)).sum();
    let mut threshold = (if careful { seed_total } else { output_tokens }) as f32 * 0.2 + 16.0;
    if !careful && raw <= seed_total as f32 * 2.5 {
        threshold = threshold.min(raw * 0.01 + 256.0);
    }
    let rejected = if careful {
        f64::from(saving) <= seed_total as f64 * 0.2 + 16.0
    } else {
        saving <= threshold
    };
    if config.require_global_savings && rejected {
        return Ok(SourceStreamsOutcome {
            commands: inputs
                .iter()
                .map(|input| literals(input))
                .collect::<Result<_>>()?,
            enabled: false,
        });
    }
    let mut seed = blocks::merge_seed(inputs, &parts, seed)?;
    if !optimal {
        return Ok(SourceStreamsOutcome {
            commands: seed,
            enabled: true,
        });
    }
    let mut selected = SourceConfig {
        match_cost: select_match_cost(config.match_cost, total, &seed),
        ..config.clone()
    };
    if !index::enabled(inputs, &selected) {
        selected.index_mode = SourceIndexMode::Disabled;
    }
    let config = &selected;
    #[cfg(test)]
    {
        _work.selected_cost = Some(config.match_cost);
    }
    if config.match_cost == SourceMatchCost::RangeReplay {
        return range_replayed(
            inputs,
            &values,
            widths,
            config,
            seed,
            replay::CACHE_LIMIT,
            _work,
        );
    }
    for pass in 0..config.match_cost.passes() {
        let costs = Costs::fitted(&seed, widths, config);
        let prices = prices::Prices::new(&costs)?;
        let mut parsed = reserved(inputs.len())?;
        #[cfg(test)]
        {
            _work.passes += 1;
        }
        // Chain batches own intra-stream matching jobs. Retain the caller's
        // executor instead of entering Schedule's serial independent-stream
        // closures, which would disable their parallel work.
        let chain_batches = inputs
            .iter()
            .zip(&seed)
            .any(|(input, seed)| chain_batch::enabled(input.len(), wire_token_count(seed), config));
        let schedule = if shared.is_some() || chain_batches {
            Schedule::jobs(inputs.len(), 1)
        } else {
            Schedule::new(inputs, config, false)
        };
        let arena = workers::StreamArena::new(inputs, config, shared.is_some())?;
        #[cfg(test)]
        {
            _work.optimal_tasks = arena.as_ref().map_or(schedule.tasks, |a| a.roots);
            _work.arena_passes += usize::from(arena.is_some());
            _work.arena_roots = arena.as_ref().map_or(0, |a| a.roots);
        }
        let process_stream = |index: usize| {
            let mut work = Work::default();
            let width = widths.get(index).copied().unwrap_or(0);
            let reuse = shared
                .as_ref()
                .filter(|(owner, _)| *owner == index && pass == 0 && !config.match_cost.range());
            let commands = if let Some((_, indexed)) = reuse {
                #[cfg(test)]
                {
                    work.indexed_streams += 1;
                    work.block_streams += 1;
                    let (compact, coordinates) = indexed.layout();
                    work.compact_index_streams += usize::from(compact);
                    work.coordinate_index_streams += usize::from(coordinates);
                }
                blocks::parse(
                    inputs[index],
                    &seed[index],
                    width,
                    config,
                    &costs,
                    |p, query, cache, price, visit| {
                        indexed.find_query(p, query, price, visit, cache)
                    },
                )?
            } else if config.index_mode != SourceIndexMode::Disabled
                && config.block_mode != SourceBlockMode::Disabled
                && config.length_uint_config == HybridUintConfig::new(0, 0, 0).unwrap()
                && inputs[index].len() >= index::minimum(config)
            {
                #[cfg(test)]
                {
                    work.indexed_streams += 1;
                    work.block_streams += 1;
                }
                if let Some(suffix) = suffix::SuffixIndex::new(&values[index], width, config)? {
                    #[cfg(test)]
                    {
                        work.suffix_streams += 1;
                    }
                    // Source suffix dispatch always uses interval blocks,
                    // including when BLOCK_DP=1 requested explicit endpoints.
                    let suffix_config = SourceConfig {
                        block_mode: SourceBlockMode::Intervals,
                        ..config.clone()
                    };
                    blocks::parse(
                        inputs[index],
                        &seed[index],
                        width,
                        &suffix_config,
                        &costs,
                        |p, query, cache, price, visit| {
                            suffix.find_query(p, query, price, visit, cache)
                        },
                    )?
                } else {
                    let indexed = StaticIndex::new(&values[index], width, config)?;
                    #[cfg(test)]
                    {
                        let (compact, coordinates) = indexed.layout();
                        work.compact_index_streams += usize::from(compact);
                        work.coordinate_index_streams += usize::from(coordinates);
                    }
                    blocks::parse(
                        inputs[index],
                        &seed[index],
                        width,
                        config,
                        &costs,
                        |p, query, cache, price, visit| {
                            indexed.find_query(p, query, price, visit, cache)
                        },
                    )?
                }
            } else {
                optimal_stream_replay(
                    inputs[index],
                    &values[index],
                    width,
                    config,
                    &prices,
                    wire_token_count(&seed[index]),
                    &mut work,
                    None,
                )?
            };
            Ok((commands, work))
        };
        let results = if let Some(arena) = arena {
            arena.collect(inputs.len(), process_stream)?
        } else {
            schedule.collect(inputs.len(), process_stream)?
        };
        for result in results {
            let (commands, work) = result.unwrap_or_default();
            parsed.push(commands);
            _work.merge(work);
        }
        seed = parsed;
        shared = None;
    }
    // Source ApplyLZ77_Optimal returns one vector per stream after an accepted
    // seed, without a post-DP savings/match-count gate (including mode 3 refit).
    // enc_ans.cc enables LZ77 from that nonempty OUTER vector, not its contents.
    Ok(SourceStreamsOutcome {
        commands: seed,
        enabled: true,
    })
}

/// Executable source method on one stream. Distances are physical and 1-based;
/// only the first context of each match is retained. SourceConfig is NOT the
/// earlier source_options() projection onto the legacy recent-bucket parser.
pub fn parse_source(
    input: &[Literal],
    distance_multiplier: usize,
    config: &SourceConfig,
) -> Result<Vec<Command>> {
    Ok(parse_source_with_state(input, distance_multiplier, config)?.commands)
}

/// ApplyLZ77's shared-model/shared-gate contract. Histories reset per stream;
/// missing distance multipliers default to zero, as in C++ image_widths.
/// This compatibility wrapper drops enabled. Literal-only output does NOT
/// imply rejection; entropy writers must use parse_source_streams_with_state.
pub fn parse_source_streams(
    inputs: &[&[Literal]],
    distance_multipliers: &[usize],
    config: &SourceConfig,
) -> Result<Vec<Vec<Command>>> {
    Ok(parse_source_streams_with_state(inputs, distance_multipliers, config)?.commands)
}

/// Source commands and the entropy-header acceptance decision for one stream.
pub fn parse_source_with_state(
    input: &[Literal],
    distance_multiplier: usize,
    config: &SourceConfig,
) -> Result<SourceParseOutcome> {
    let mut outcome = parse_source_streams_with_state(&[input], &[distance_multiplier], config)?;
    Ok(SourceParseOutcome {
        commands: outcome.commands.pop().expect("one source stream"),
        enabled: outcome.enabled,
    })
}

/// The global decision corresponds to !ApplyLZ77(...).empty() in enc_ans.cc.
/// Preserve enabled even when some/all final streams have no Match commands.
pub fn parse_source_streams_with_state(
    inputs: &[&[Literal]],
    distance_multipliers: &[usize],
    config: &SourceConfig,
) -> Result<SourceStreamsOutcome> {
    run(inputs, distance_multipliers, config, &mut Work::default())
}

/// Source ApplyLZ77_RLE, not the legacy f64 RLE parser. Method/matcher/DP
/// settings are ignored. C++ RLE deliberately uses default literal uint420
/// even when params.UintConfig() differs. Length/min_symbol remain wire config.
pub fn parse_source_rle_with_state(
    input: &[Literal],
    distance_multiplier: usize,
    config: &SourceConfig,
) -> Result<SourceParseOutcome> {
    let mut outcome =
        parse_source_rle_streams_with_state(&[input], &[distance_multiplier], config)?;
    Ok(SourceParseOutcome {
        commands: outcome.commands.pop().expect("one source RLE stream"),
        enabled: outcome.enabled,
    })
}

/// Frozen enc_lz77.cc:132..201: shared f32 entropy/cumulative/savings model,
/// per-run CeilLog2Nonzero(length-3+1)+1, then a shared DOUBLE threshold.
/// Distances are physical 1; the parent applies its canonical distance_symbol.
/// At width=1 that canonical code may differ from C++'s explicit special 1,
/// which aliases the same physical distance. Patch dictionaries use width=0.
pub fn parse_source_rle_streams_with_state(
    inputs: &[&[Literal]],
    _distance_multipliers: &[usize],
    config: &SourceConfig,
) -> Result<SourceStreamsOutcome> {
    let config = SourceConfig {
        literal_uint_config: HybridUintConfig::default(),
        ..config.clone()
    };
    let total = validate_inputs(inputs, &config)?;
    let costs = Costs::raw(inputs, &config);
    let mut commands = reserved(inputs.len())?;
    let mut saving = 0.0f32;
    for input in inputs {
        let sums = costs.cumulative(input, &config)?;
        let mut output = reserved(input.len())?;
        let mut position = 0;
        while position < input.len() {
            let mut length = 0;
            if position > 0 {
                let limit = (input.len() - position).min(config.max_match_length);
                while length < limit && input[position + length].value == input[position - 1].value
                {
                    length += 1;
                }
            }
            if length == 0 {
                output.push(Command::Literal(input[position]));
                position += 1;
                continue;
            }
            let cost = sums[position + length] - sums[position];
            let match_cost = if length >= MIN_MATCH {
                // ceil(log2(n+1)) equals bit_width(n), including n=0.
                ((usize::BITS - (length - MIN_MATCH).leading_zeros()) + 1) as f32
            } else {
                0.0
            };
            if length < MIN_MATCH || cost <= match_cost {
                output.extend(
                    input[position..position + length]
                        .iter()
                        .copied()
                        .map(Command::Literal),
                );
            } else {
                output.push(Command::Match {
                    context: input[position].context,
                    length: length as u32,
                    distance: 1,
                });
                saving += cost - match_cost;
            }
            position += length;
        }
        commands.push(output);
    }
    let enabled = !inputs.is_empty()
        && (!config.require_global_savings || f64::from(saving) > total as f64 * 0.2 + 16.0);
    if !enabled {
        commands = inputs
            .iter()
            .map(|input| literals(input))
            .collect::<Result<_>>()?;
    }
    Ok(SourceStreamsOutcome { commands, enabled })
}

#[cfg(test)]
mod controls_tests;

#[cfg(test)]
mod alias_tests;

#[cfg(test)]
mod tests;
