// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; the complete upstream notice and license
// are retained in docs/PORTING_KERNELS.md.

//! Source LZ77 policy and explicit native projections, not new matcher engines.
//!
//! Frozen b866b3defec13494bfbaf705db93581cda118e37 sources:
//! * enc_ans_params.h: HistogramParams::LZ77Method and UintConfig defaults.
//! * enc_lz77.cc: ApplyLZ77 dispatch, LZ77HashMap, HashChain,
//!   LZ77BucketMatcher, ApplyLZ77_Optimal's first-pass selection.
//! * enc_lz77_bucket.h: LZ77RecentBuckets' 15-bit tables.
//! * dec_ans.h/.cc: decoder window, default min_length and length uint config.
//! * enc_ans.cc: BuildAndEncodeHistograms' min_symbol and uint selection.
//!
//! Scope is lossless Modular, ANS, default UintConfig, no environment overrides,
//! and no direct audit. The named methods do NOT determine every codec setting.
//! Both b1w3t and b3w3t are GREEDY: b is retained bucket entries, w is the
//! three-token hash length, and t enables per-match runtime cost comparison.
//! Optc8/Optc256 use hash-chain VISIT limits, not bucket capacities. The explicit
//! lossless_bucket_matcher flag only changes Optc256, to short32 + long8 buckets.
//!
//! Source defaults are min_length=3, length uint000, literal cost uint420,
//! distance cost uint420, and ANS min_symbol=224 (512 for force_huffman).
//! DistCost's trained uint700 and LenCost's uint100 tables are STATIC COST
//! MODELS, not serialized length/distance configurations. Actual ordinary
//! entropy configurations, including distance, may be selected per histogram.
//! In particular, the current native encoder's length400/distance000 choices
//! are not defaults taken from these C++ method names. Projection uses source
//! cost defaults; callers retaining other wire configs must update the returned
//! options' cost configs and cap lengths for their actual entropy alphabet.
//!
//! A native projection is usable, but its gaps must remain visible to the
//! scheduler/Goodall adapter. Do not cache its score under a source-equivalent
//! candidate identity. None of the current projections is algorithm-identical.
//! The executable source matcher routes now live in lz77::parse_source and
//! parse_source_streams, configured by SourceConfig. The gaps below describe
//! only this module's legacy Lz77Options projection, not those new entry points.
//! SourceConfig now also routes history/wide buckets, baseline static indexes,
//! match-cost 4..7 and bounded stream workers. These do not repair this legacy
//! projection: callers must opt into parse_source_streams[_with_state]. Remaining
//! executable-route experiments are listed in lz77_source.rs, not erased here.
//! This module never changes Lz77Options::default(), reads environment settings,
//! or changes a caller's entropy serializer. SourceLz77Method deliberately has
//! no None: the scheduler must bypass parse entirely for disabled LZ77.
//!
//! Registered once as tools::lz77::policy, with method/source_options reexports
//! from tools::lz77. This file uses absolute crate paths so it can also be a
//! top-level tools module, but must not be registered through both paths.
//! Propagate the whole returned Lz77Options into group parsing, not only mode.
//! Preserve gaps in Goodall capability reporting and keep disabled-LZ77 calls
//! outside this helper. Register this file only once so its tests are not run
//! through two distinct module paths.

use crate::entropy::HybridUintConfig;
use crate::tools::lz77::{FilterPolicy, Lz77Mode, Lz77Options, MIN_MATCH, WINDOW_SIZE};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceLz77Method {
    Lz77b1w3t,
    Lz77b3w3t,
    Optc8,
    Optc256,
    Lz77b1w3f,
    Lz77b3w3f,
    Lz77b7w3f,
    Lz77b15w3f,
    Lz77b31w3f,
    Lz77b7w3t,
    Lz77b15w3t,
    Lz77b31w3t,
    Optc1,
    Optc3,
}

impl SourceLz77Method {
    pub(super) fn fast_capacity(self) -> Option<usize> {
        match self {
            Self::Lz77b1w3f | Self::Lz77b1w3t => Some(1),
            Self::Lz77b3w3f | Self::Lz77b3w3t => Some(3),
            Self::Lz77b7w3f | Self::Lz77b7w3t => Some(7),
            Self::Lz77b15w3f | Self::Lz77b15w3t => Some(15),
            Self::Lz77b31w3f | Self::Lz77b31w3t => Some(31),
            _ => None,
        }
    }

    pub(super) fn chain_maximum(self) -> Option<usize> {
        match self {
            Self::Optc1 => Some(1),
            Self::Optc3 => Some(3),
            Self::Optc8 => Some(8),
            Self::Optc256 => Some(256),
            _ => None,
        }
    }

    pub(super) fn runtime_cost_comparison(self) -> bool {
        !matches!(
            self,
            Self::Lz77b1w3f
                | Self::Lz77b3w3f
                | Self::Lz77b7w3f
                | Self::Lz77b15w3f
                | Self::Lz77b31w3f
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceMatcher {
    /// The current origin is inserted AFTER querying; all entries can be prior
    /// positions. LZ77HashMap uses 14 hash bits and checks 24 spatial anchors.
    FastHash { bucket_capacity: usize },
    /// HashChain uses 15 hash bits and zero-run chains, with the current origin
    /// inserted BEFORE querying. The limit counts visits, including collisions.
    HashChain { max_chain_length: usize },
    /// The nominated lossless Optc256 path, not its default hash-chain path.
    /// Both 15-bit rings include the current position when searched, so a full
    /// short32 ring retains at most 31 prior positions; the long8 ring adds
    /// older-prefix candidates. There are also 24 independent spatial anchors.
    OptimalBuckets {
        short_capacity: usize,
        long_capacity: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceFirstPass {
    /// These methods are the parse itself; there is no preliminary parse.
    None,
    /// ApplyLZ77_LazyGreedy: chain limit 256, lazy lookahead only for len < 256.
    LazyChain256,
    /// ApplyLZ77_LZ77<7, 3, true>, with 120 spatial anchors and 14 hash bits.
    GreedyBucket7,
}

/// Exact source settings for the dispatch scope stated above. Unlike the
/// native projection, this preserves unrepresentable caps (3 and 256).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LzMethodConfig {
    pub method: SourceLz77Method,
    pub mode: Lz77Mode,
    pub matcher: SourceMatcher,
    pub hash_bits: u32,
    pub hash_length: usize,
    pub min_length: usize,
    /// Decoder storage/window cap, NOT hash length or maximum copy length.
    pub window_size: usize,
    /// FastHash accepts distance == window; HashChain's masked ring cannot
    /// represent it, and OptimalBuckets explicitly rejects distance >= window.
    pub window_end_inclusive: bool,
    /// Independent spatial probes. All modes can encode all 120 special codes;
    /// HashChain maps discovered distances but has no separate spatial probes.
    pub special_distances_to_check: usize,
    /// `t` in the greedy method name. Optimal has DP costs instead; its seed
    /// still performs the per-match cost check.
    pub runtime_cost_comparison: bool,
    pub min_symbol: u32,
    pub length_uint_config: HybridUintConfig,
    pub literal_cost_uint_config: HybridUintConfig,
    /// SymbolCostEstimator::DistCost always uses HybridUintConfig(), even when
    /// other histogram uint choices have been requested by a caller.
    pub distance_cost_uint_config: HybridUintConfig,
    /// Default source dispatch makes one DP pass; match-cost=3/5/6 experiments
    /// may refit, but are not implied by Optc8 or Optc256.
    pub extra_refit_passes: usize,
}

impl LzMethodConfig {
    /// ApplyLZ77_Optimal selects using total tokens across ALL streams sharing
    /// the histogram model, not the size of an individual group/channel.
    pub fn first_pass(&self, total_tokens: usize, careful_first_pass: bool) -> SourceFirstPass {
        if self.mode != Lz77Mode::Optimal {
            SourceFirstPass::None
        } else if careful_first_pass || total_tokens < (1 << 16) {
            SourceFirstPass::LazyChain256
        } else {
            SourceFirstPass::GreedyBucket7
        }
    }

    /// HashChain/OptimalBuckets allocate the next power of two of the stream
    /// length, capped at 1<<20. The empty-stream case has a one-entry ring.
    pub fn matcher_window(&self, input_len: usize) -> usize {
        if matches!(self.matcher, SourceMatcher::FastHash { .. }) {
            self.window_size
        } else {
            input_len.clamp(1, self.window_size).next_power_of_two()
        }
    }

    /// Maximum physically available backwards distance in this stream. Copy
    /// length is independent: the C++ matcher permits the entire remaining
    /// input, including overlapping copies longer than the distance window.
    pub fn max_backward_distance(&self, input_len: usize) -> usize {
        let window = self.matcher_window(input_len);
        let cap = window - usize::from(!self.window_end_inclusive);
        cap.min(input_len.saturating_sub(1))
    }
}

pub fn source_config(method: SourceLz77Method) -> LzMethodConfig {
    source_config_with_bucket_matcher(method, false)
}

/// The nominated source override is only active for Optc256. This function
/// models lossless_lz77_cost_spans=true and direct_audit=false; environment
/// alternatives (history, wide-ring, suffix/index modes) remain outside scope.
pub fn source_config_with_bucket_matcher(
    method: SourceLz77Method,
    bucket_matcher: bool,
) -> LzMethodConfig {
    let (mode, matcher, hash_bits, special_distances_to_check) =
        if let Some(capacity) = method.fast_capacity() {
            (
                Lz77Mode::Greedy,
                SourceMatcher::FastHash {
                    bucket_capacity: capacity,
                },
                14,
                if capacity <= 3 { 24 } else { 120 },
            )
        } else if method == SourceLz77Method::Optc256 && bucket_matcher {
            (
                Lz77Mode::Optimal,
                SourceMatcher::OptimalBuckets {
                    short_capacity: 32,
                    long_capacity: 8,
                },
                15,
                24,
            )
        } else {
            (
                Lz77Mode::Optimal,
                SourceMatcher::HashChain {
                    max_chain_length: method.chain_maximum().expect("optimal source method"),
                },
                15,
                0,
            )
        };
    LzMethodConfig {
        method,
        mode,
        matcher,
        hash_bits,
        hash_length: 3,
        min_length: 3,
        window_size: WINDOW_SIZE,
        window_end_inclusive: mode == Lz77Mode::Greedy,
        special_distances_to_check,
        runtime_cost_comparison: mode == Lz77Mode::Greedy && method.runtime_cost_comparison(),
        min_symbol: 224,
        length_uint_config: HybridUintConfig::new(0, 0, 0).unwrap(),
        literal_cost_uint_config: HybridUintConfig::default(),
        distance_cost_uint_config: HybridUintConfig::default(),
        extra_refit_passes: 0,
    }
}

/// Behavioral differences that must remain in parent capability reporting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lz77PolicyGap {
    /// The legacy greedy projection always applies a per-match cost veto.
    RuntimeCostComparison,
    /// Rust evaluates one input stream at a time; C++ pools entropy statistics
    /// and the global savings decision over every stream in ApplyLZ77's call.
    SingleStreamModelAndSavings,
    /// Rust uses f64/log2, C++ float/FastLog2f. Near-tie decisions can differ.
    FloatingPointCosts,
    SpatialProbeCap {
        source: usize,
        native: usize,
    },
    FastBucketCapacity {
        source: usize,
        native: usize,
    },
    /// Equal numeric limits would still not make recent buckets hash/zero
    /// chains. Optc256's 256 visits are never silently called 64 visits here.
    HashChainReplaced {
        source_chain_limit: usize,
        native_bucket_capacity: usize,
    },
    /// Rust has only the short-prefix ring, queried before current insertion;
    /// it lacks the source long-prefix/zero-run proposals and insertion timing.
    OptimalBucketMatcherReplaced,
    /// Rust always uses its own lazy recent-bucket seed, without the source's
    /// 65536-total-token switch, 256-link careful seed or seed rejection gate.
    OptimalSeedSelection,
    /// Rust uses distance frontiers, f64 dyadic intervals, retained seed edges
    /// and different pruning/tie rules, not the source's default chain DP.
    OptimalCandidateAndTiePolicy,
    /// Rust makes a post-DP savings check; source default gates on its seed.
    OptimalSavingsGate,
}

#[derive(Clone, Debug)]
#[must_use = "inspect gaps before treating these as source-equivalent options"]
pub struct SourceLz77Options {
    pub source: LzMethodConfig,
    /// Explicit usable native proxy, NOT an exact implementation of source.
    pub options: Lz77Options,
    pub gaps: Vec<Lz77PolicyGap>,
}

impl SourceLz77Options {
    pub fn is_source_equivalent(&self) -> bool {
        self.gaps.is_empty()
    }

    /// Fail closed for scheduler paths that require actual source equivalence.
    pub fn exact_options(self) -> Result<Lz77Options> {
        if !self.gaps.is_empty() {
            return Err(Error::Encoding(format!(
                "unported LZ77 policy capabilities: {:?}",
                self.gaps
            )));
        }
        Ok(self.options)
    }

    /// Tighten a native proxy to the source's per-stream physical window and
    /// full-input copy bound, without creating invalid zero-length options for
    /// tiny streams. This does NOT remove the capability gaps.
    pub fn options_for_stream(&self, input_len: usize) -> Lz77Options {
        let mut options = self.options.clone();
        options.window_size = options
            .window_size
            .min(self.source.max_backward_distance(input_len).max(1));
        options.max_match_length = options
            .max_match_length
            .min(input_len.clamp(MIN_MATCH, u32::MAX as usize));
        options
    }
}

/// Source settings plus a clearly marked projection onto the current native
/// parser. Parent can re-export this at tools::lz77::source_options.
pub fn source_options(method: SourceLz77Method) -> SourceLz77Options {
    source_options_with_bucket_matcher(method, false)
}

pub fn source_options_with_bucket_matcher(
    method: SourceLz77Method,
    bucket_matcher: bool,
) -> SourceLz77Options {
    let source = source_config_with_bucket_matcher(method, bucket_matcher);
    let mut gaps = vec![
        Lz77PolicyGap::SingleStreamModelAndSavings,
        Lz77PolicyGap::FloatingPointCosts,
    ];
    // These are explicitly native proxy choices. The source's exact limits are
    // preserved in source.matcher and in the gap, never overwritten by these.
    let bucket_capacity = match source.matcher {
        SourceMatcher::FastHash { bucket_capacity: 1 } => 1,
        SourceMatcher::FastHash { bucket_capacity } => {
            gaps.push(Lz77PolicyGap::FastBucketCapacity {
                source: bucket_capacity,
                native: 4,
            });
            4
        }
        SourceMatcher::HashChain { max_chain_length } => {
            let native = max_chain_length.min(64).next_power_of_two();
            gaps.push(Lz77PolicyGap::HashChainReplaced {
                source_chain_limit: max_chain_length,
                native_bucket_capacity: native,
            });
            native
        }
        SourceMatcher::OptimalBuckets { short_capacity, .. } => {
            gaps.push(Lz77PolicyGap::OptimalBucketMatcherReplaced);
            short_capacity
        }
    };
    gaps.push(Lz77PolicyGap::SpatialProbeCap {
        source: source.special_distances_to_check,
        native: 120,
    });
    if source.mode == Lz77Mode::Greedy && !source.runtime_cost_comparison {
        gaps.push(Lz77PolicyGap::RuntimeCostComparison);
    }
    if source.mode == Lz77Mode::Optimal {
        gaps.extend([
            Lz77PolicyGap::OptimalSeedSelection,
            Lz77PolicyGap::OptimalCandidateAndTiePolicy,
            Lz77PolicyGap::OptimalSavingsGate,
        ]);
    }
    let options = Lz77Options {
        mode: source.mode,
        // Optimal source rings exclude the window-end position even though the
        // decoder itself permits a copy at physical distance exactly 1<<20.
        window_size: source.window_size - usize::from(!source.window_end_inclusive),
        max_match_length: u32::MAX as usize,
        bucket_bits: source.hash_bits,
        bucket_capacity,
        range_search: true,
        // Source default has no prefix-filter environment override. Repeated
        // range insertion and native exact comparison remain available.
        prefix_filter: FilterPolicy::Disabled,
        literal_uint_config: source.literal_cost_uint_config,
        force_huffman: false,
        require_global_savings: true,
        optimal_refit: false,
        optimal_skip_repetitions: true,
        optimal_min_symbol: source.min_symbol,
        optimal_length_uint_config: source.length_uint_config,
        optimal_distance_uint_config: source.distance_cost_uint_config,
    };
    SourceLz77Options {
        source,
        options,
        gaps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::lz77::{Command, Literal, parse};

    const METHODS: [SourceLz77Method; 4] = [
        SourceLz77Method::Lz77b1w3t,
        SourceLz77Method::Lz77b3w3t,
        SourceLz77Method::Optc8,
        SourceLz77Method::Optc256,
    ];

    #[test]
    fn cpp_dispatch_caps_are_not_lazy_modes_or_window_sizes() {
        for (method, capacity) in [(METHODS[0], 1), (METHODS[1], 3)] {
            let source = source_config(method);
            assert_eq!(source.mode, Lz77Mode::Greedy);
            assert_eq!(
                source.matcher,
                SourceMatcher::FastHash {
                    bucket_capacity: capacity
                }
            );
            assert_eq!(
                (source.hash_bits, source.hash_length, source.min_length),
                (14, 3, 3)
            );
            assert_eq!(source.special_distances_to_check, 24);
            assert!(source.runtime_cost_comparison && source.window_end_inclusive);
        }
        for (method, limit) in [(METHODS[2], 8), (METHODS[3], 256)] {
            let source = source_config(method);
            assert_eq!(source.mode, Lz77Mode::Optimal);
            assert_eq!(
                source.matcher,
                SourceMatcher::HashChain {
                    max_chain_length: limit
                }
            );
            assert_eq!(
                (source.hash_bits, source.hash_length, source.min_length),
                (15, 3, 3)
            );
            assert_eq!(source.special_distances_to_check, 0);
            assert!(!source.runtime_cost_comparison && !source.window_end_inclusive);
        }
    }

    #[test]
    fn source_coding_defaults_are_not_native_wire_defaults() {
        for method in METHODS {
            let policy = source_options(method);
            let source = policy.source;
            assert_eq!(source.window_size, 1 << 20);
            assert_eq!(
                source.length_uint_config,
                HybridUintConfig::new(0, 0, 0).unwrap()
            );
            assert_eq!(
                source.distance_cost_uint_config,
                HybridUintConfig::new(4, 2, 0).unwrap()
            );
            assert_eq!(
                source.literal_cost_uint_config,
                HybridUintConfig::new(4, 2, 0).unwrap()
            );
            assert_eq!((source.min_symbol, source.extra_refit_passes), (224, 0));
            assert_eq!(
                policy.options.optimal_length_uint_config,
                source.length_uint_config
            );
            assert_eq!(
                policy.options.optimal_distance_uint_config,
                source.distance_cost_uint_config
            );
            assert!(!policy.options.optimal_refit);
        }
        let default = Lz77Options::default();
        assert_eq!(default.mode, Lz77Mode::Lazy);
        assert_eq!(
            default.optimal_length_uint_config,
            HybridUintConfig::new(4, 0, 0).unwrap()
        );
        assert_eq!(
            default.optimal_distance_uint_config,
            HybridUintConfig::new(0, 0, 0).unwrap()
        );
    }

    #[test]
    fn first_pass_boundary_is_total_tokens_and_careful_override() {
        for method in METHODS {
            let source = source_config(method);
            if source.mode == Lz77Mode::Greedy {
                assert_eq!(source.first_pass(0, true), SourceFirstPass::None);
                continue;
            }
            assert_eq!(
                source.first_pass(65_535, false),
                SourceFirstPass::LazyChain256
            );
            assert_eq!(
                source.first_pass(65_536, false),
                SourceFirstPass::GreedyBucket7
            );
            assert_eq!(
                source.first_pass(usize::MAX, true),
                SourceFirstPass::LazyChain256
            );
        }
    }

    #[test]
    fn window_boundaries_and_overlaps_do_not_cap_copy_length() {
        let fast = source_config(METHODS[0]);
        let optimal = source_config(METHODS[3]);
        for (n, ring) in [
            (0, 1),
            (1, 1),
            (3, 4),
            (4, 4),
            (5, 8),
            (WINDOW_SIZE, WINDOW_SIZE),
            (WINDOW_SIZE + 1, WINDOW_SIZE),
            (usize::MAX, WINDOW_SIZE),
        ] {
            assert_eq!(optimal.matcher_window(n), ring);
            assert_eq!(fast.matcher_window(n), WINDOW_SIZE);
            assert_eq!(
                optimal.max_backward_distance(n),
                (ring - 1).min(n.saturating_sub(1))
            );
        }
        assert_eq!(fast.max_backward_distance(WINDOW_SIZE + 1), WINDOW_SIZE);
        assert_eq!(
            optimal.max_backward_distance(WINDOW_SIZE + 1),
            WINDOW_SIZE - 1
        );
        let options = source_options(METHODS[3]).options_for_stream(2 * WINDOW_SIZE);
        assert_eq!(options.window_size, WINDOW_SIZE - 1);
        assert_eq!(options.max_match_length, 2 * WINDOW_SIZE);
    }

    #[test]
    fn bucket_override_is_optc256_only_and_not_short_ring_equivalence() {
        for method in &METHODS[..3] {
            assert_eq!(
                source_config(*method),
                source_config_with_bucket_matcher(*method, true)
            );
        }
        let projection = source_options_with_bucket_matcher(METHODS[3], true);
        assert_eq!(
            projection.source.matcher,
            SourceMatcher::OptimalBuckets {
                short_capacity: 32,
                long_capacity: 8
            }
        );
        assert_eq!(projection.source.hash_bits, 15);
        assert_eq!(projection.source.special_distances_to_check, 24);
        assert_eq!(projection.options.bucket_capacity, 32);
        assert!(
            projection
                .gaps
                .contains(&Lz77PolicyGap::OptimalBucketMatcherReplaced)
        );
        assert!(projection.exact_options().is_err());
    }

    #[test]
    fn proxies_never_silently_claim_source_caps() {
        for method in METHODS {
            let projection = source_options(method);
            assert!(!projection.is_source_equivalent());
            assert!(projection.exact_options().is_err());
        }
        let b3 = source_options(METHODS[1]);
        assert_eq!(b3.options.mode, Lz77Mode::Greedy);
        assert!(b3.gaps.contains(&Lz77PolicyGap::FastBucketCapacity {
            source: 3,
            native: 4
        }));
        let c256 = source_options(METHODS[3]);
        assert!(c256.gaps.contains(&Lz77PolicyGap::HashChainReplaced {
            source_chain_limit: 256,
            native_bucket_capacity: 64
        }));
    }

    fn expand(commands: &[Command], input: &[Literal], options: &Lz77Options) {
        let mut values = Vec::new();
        for &command in commands {
            let origin = values.len();
            match command {
                Command::Literal(literal) => {
                    assert_eq!(literal, input[origin]);
                    values.push(literal.value);
                }
                Command::Match {
                    context,
                    length,
                    distance,
                } => {
                    assert_eq!(context, input[origin].context);
                    assert!((3..=options.max_match_length).contains(&(length as usize)));
                    assert!((1..=origin.min(options.window_size)).contains(&(distance as usize)));
                    for _ in 0..length {
                        values.push(values[values.len() - distance as usize]);
                    }
                }
            }
        }
        assert_eq!(values, input.iter().map(|l| l.value).collect::<Vec<_>>());
    }

    #[test]
    fn projected_options_parse_empty_tails_collisions_and_full_u32_values() {
        for method in METHODS {
            for buckets in [false, true] {
                let policy = source_options_with_bucket_matcher(method, buckets);
                for len in [0, 1, 2, 3, 4, 257, 513] {
                    let input: Vec<_> = (0..len)
                        .map(|i| Literal {
                            context: if i % 3 == 0 { u32::MAX } else { (i % 7) as u32 },
                            value: u32::MAX - (i % 17) as u32,
                        })
                        .collect();
                    for width in [0, 1, 2, 31] {
                        let mut options = policy.options_for_stream(len);
                        options.bucket_bits = 0;
                        options.require_global_savings = false;
                        expand(&parse(&input, width, &options).unwrap(), &input, &options);
                    }
                }
            }
        }
    }
}
