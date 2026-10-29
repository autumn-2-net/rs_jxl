// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; the complete upstream notice and license
// are retained in docs/PORTING_KERNELS.md.

//! Token-stream LZ77 port from libjxl b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Sources: lib/jxl/enc_lz77.cc (SymbolCostEstimator, ApplyLZ77_RLE,
//! LZ77HashMap, ApplyLZ77_LZ77, ApplyLZ77_LazyGreedy, LenCost, DistCost),
//! enc_lz77_bucket.h (recent buckets and repeated-range updates), dec_ans.h
//! (the 120 special distances and the decoder's 1 << 20 window), and the
//! optimal/range-parser sources recorded in the private lz77::dp module.
//!
//! Adaptations: one stream per call, physical rather than coded distances,
//! fixed minimum length three, sparse context IDs, f64 cumulative costs, and
//! the existing power-of-two RecentBucket instead of the fast C++ 2^n-1 ring.
//! Lazy parsing uses that same recent-bucket matcher instead of HashChain.
//! PrefixFilter/RecentBucket use the existing native Rust SIMD and scalar
//! fallback; exact match extension is the C++ scalar comparison loop.
//!
//! Optimal mode adds entropy refitting, distance-cost frontiers, verified match
//! caching, and exact dyadic interval relaxation. Its candidate search remains
//! bounded by the selected recent buckets, not an exhaustive dictionary search.
//! The separate parse_source entry point ports exact fast-ring/hash-chain and
//! short32/long8 matcher routes, history/wide variants, baseline static indexes,
//! run-key/deep proposals, range modes 4..7 and bounded stream dispatch;
//! see lz77/source.rs. Mode4 has source shared-budget cross-pass replay.
//! Source routes also include parallel index builds/match batches, precomputed
//! SIMD tags, block/seed splitting, shared seed-index reuse, suffix proposals,
//! u32/packed-u16 word extension, cached bucket coordinates, dense/sparse
//! endpoint caches, shared/carried frontiers,
//! distance-bound exits, table/class prices, seed-guided repair origins and
//! lazy block parsing, zero-tail query cuts and conservative literal bounds.
//! Block DP also has calendar events, native vector relaxation, conservative
//! range dominance, literal-only jumps, dyadic trees and block envelopes.
//! Bounded dense special-distance symbols are automatic for static/suffix
//! indexes and opt-in for mutable buckets.
//! Ordinary source optimal DP can use the source's 12-byte compact nodes;
//! block/range heaps and precise-ring keep their distinct storage and costs.
//! Suffix construction uses a bounded serial wrapper of vendored libsais-rs,
//! the pure-Rust translation of the external libsais integer engine.
//! Non-block indexed DP exposes boundary-only intervalDP2, and optimal routes
//! expose independent bucket dominance and chain cache/depth controls.
//! Generic source methods also expose b1/b3/b7/b15/b31 w3f/w3t and Optc1/3.
//! FAST_RANGES exposes separate repeated-insert/extension bits with source
//! implicit prefix policy and explicit scalar/SIMD prefix overrides.
//! Exact bucket selections distinguish stored/compact aliases and source
//! parameter precedence while retaining the older explicit override API.
//! Indexed optimal streams can opt into the caller's cooperative arena,
//! including nested index/build/block work and explicit quiet wake policy.
//! NOT PORTED here: profiling/audits and environment dispatch. See
//! docs/LZ77_PORT_STATUS.md for executor ownership and verification status.
//! Global LZ77 parameters, module registration, entropy wiring, and bitstream
//! serialization remain the caller's responsibility. This is not an encoder.

use std::collections::BTreeMap;

use crate::entropy::HybridUintConfig;
use crate::{Error, Result};

pub use super::lz77_prefix::FilterPolicy;
use super::lz77_prefix::{
    PrefixFilter, RecentBucket, SpecialPrefixGroup, group_distances, prefix_hash,
};

mod dp;
pub mod policy;
mod source;

pub use policy::{SourceLz77Method, source_config, source_options};
pub use source::{
    SourceBlockDyadic, SourceBlockEnvelope, SourceBlockMode, SourceBucketMode,
    SourceBucketSelection, SourceChainCache, SourceConfig, SourceDeepPrefix, SourceFastPrefix,
    SourceFastRanges, SourceIndexMode, SourceIndexTags, SourceIntervalDp, SourceMatchCost,
    SourceParallelism, SourceParseOutcome, SourceRunKeys, SourceSeedBlocks, SourceSeedGuided,
    SourceStreamsOutcome, SourceSuffixMode, parse_source, parse_source_rle_streams_with_state,
    parse_source_rle_with_state, parse_source_streams, parse_source_streams_with_state,
    parse_source_with_state,
};

pub const MIN_MATCH: usize = 3;
pub const WINDOW_SIZE: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Literal {
    pub context: u32,
    pub value: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Command {
    Literal(Literal),
    Match {
        /// Context of the first replaced literal only.
        context: u32,
        /// Actual number of values to copy, not length minus MIN_MATCH.
        length: u32,
        /// Physical, one-based backward distance; overlap is legal.
        distance: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Lz77Mode {
    Rle,
    Greedy,
    Lazy,
    /// Shortest path over the retained candidates, with a seeded entropy model.
    Optimal,
}

#[derive(Clone, Debug)]
pub struct Lz77Options {
    pub mode: Lz77Mode,
    /// Inclusive backward-distance limit, in 1..=WINDOW_SIZE.
    pub window_size: usize,
    /// In MIN_MATCH..=u32::MAX, independent of the distance window. The caller
    /// must also respect its entropy alphabet: base 224 plus hybrid (4,0,0)
    /// in a 256-symbol alphabet permits lengths only through (1 << 20) + 2.
    pub max_match_length: usize,
    /// Number of low hash bits; 0..=20. Zero deliberately uses one bucket.
    pub bucket_bits: u32,
    /// Actual retained candidates per bucket: a power of two in 1..=64.
    pub bucket_capacity: usize,
    /// Use grouped, range-pruned spatial candidates rather than scalar visits.
    pub range_search: bool,
    /// Policy for the full-stream spatial prefix cache. Bucket tags are cheap
    /// and enabled immediately unless this is Disabled.
    pub prefix_filter: FilterPolicy,
    /// Tokenization used to estimate literal entropy; not a serialized config.
    pub literal_uint_config: HybridUintConfig,
    pub force_huffman: bool,
    /// Apply the C++ whole-stream savings gate. False disables only this gate,
    /// not the per-match entropy-cost check. Rejection returns all literals.
    pub require_global_savings: bool,
    /// Optimal only: one additional model refit and DP pass, never an unbounded
    /// fixed-point search. The first DP pass already fits the lazy seed.
    pub optimal_refit: bool,
    /// Optimal only: the C++ eight-query repetition skip. It leaves the final
    /// eight origins and all seed-match origins available, but is heuristic.
    /// False searches every origin; interval relaxation remains accelerated.
    pub optimal_skip_repetitions: bool,
    /// Optimal model's length-symbol offset. Match/literal alphabets must not
    /// overlap. These settings estimate costs; the caller still writes them.
    pub optimal_min_symbol: u32,
    /// Optimal length model. LSB-in-token must be zero for contiguous intervals.
    pub optimal_length_uint_config: HybridUintConfig,
    pub optimal_distance_uint_config: HybridUintConfig,
}

impl Default for Lz77Options {
    fn default() -> Self {
        Self {
            mode: Lz77Mode::Lazy,
            window_size: WINDOW_SIZE,
            max_match_length: u32::MAX as usize,
            bucket_bits: 14,
            bucket_capacity: 32,
            range_search: true,
            prefix_filter: FilterPolicy::Lazy,
            literal_uint_config: HybridUintConfig::default(),
            force_huffman: false,
            require_global_savings: true,
            optimal_refit: true,
            optimal_skip_repetitions: true,
            optimal_min_symbol: 224,
            optimal_length_uint_config: HybridUintConfig::new(4, 0, 0).unwrap(),
            optimal_distance_uint_config: HybridUintConfig::new(0, 0, 0).unwrap(),
        }
    }
}

impl Lz77Options {
    fn validate(&self) -> Result<()> {
        if !(1..=WINDOW_SIZE).contains(&self.window_size) {
            return Err(Error::InvalidInput(
                "LZ77 window must be in 1..=1048576".into(),
            ));
        }
        if !(MIN_MATCH..=u32::MAX as usize).contains(&self.max_match_length) {
            return Err(Error::InvalidInput(
                "LZ77 maximum length must be in 3..=u32::MAX".into(),
            ));
        }
        if self.bucket_bits > 20 {
            return Err(Error::InvalidInput(
                "LZ77 bucket bits must be in 0..=20".into(),
            ));
        }
        if !self.bucket_capacity.is_power_of_two() || self.bucket_capacity > 64 {
            return Err(Error::InvalidInput(
                "LZ77 bucket capacity must be a power of two in 1..=64".into(),
            ));
        }
        if self.mode == Lz77Mode::Optimal {
            if self.optimal_length_uint_config.lsb_in_token() != 0 {
                return Err(Error::InvalidInput(
                    "optimal LZ77 length intervals require zero LSB-in-token bits".into(),
                ));
            }
            let maximum = self
                .optimal_length_uint_config
                .encode((self.max_match_length - MIN_MATCH) as u32)
                .token;
            if self.optimal_min_symbol.checked_add(maximum).is_none() {
                return Err(Error::InvalidInput(
                    "optimal LZ77 length symbol overflows u32".into(),
                ));
            }
        }
        Ok(())
    }
}

// dec_ans.h::kSpecialDistances, in wire-symbol order, not physical order.
#[rustfmt::skip]
const SPECIAL_DISTANCES: [(i8, i8); 120] = [
    (0, 1), (1, 0), (1, 1), (-1, 1), (0, 2), (2, 0), (1, 2), (-1, 2),
    (2, 1), (-2, 1), (2, 2), (-2, 2), (0, 3), (3, 0), (1, 3), (-1, 3),
    (3, 1), (-3, 1), (2, 3), (-2, 3), (3, 2), (-3, 2), (0, 4), (4, 0),
    (1, 4), (-1, 4), (4, 1), (-4, 1), (3, 3), (-3, 3), (2, 4), (-2, 4),
    (4, 2), (-4, 2), (0, 5), (3, 4), (-3, 4), (4, 3), (-4, 3), (5, 0),
    (1, 5), (-1, 5), (5, 1), (-5, 1), (2, 5), (-2, 5), (5, 2), (-5, 2),
    (4, 4), (-4, 4), (3, 5), (-3, 5), (5, 3), (-5, 3), (0, 6), (6, 0),
    (1, 6), (-1, 6), (6, 1), (-6, 1), (2, 6), (-2, 6), (6, 2), (-6, 2),
    (4, 5), (-4, 5), (5, 4), (-5, 4), (3, 6), (-3, 6), (6, 3), (-6, 3),
    (0, 7), (7, 0), (1, 7), (-1, 7), (5, 5), (-5, 5), (7, 1), (-7, 1),
    (4, 6), (-4, 6), (6, 4), (-6, 4), (2, 7), (-2, 7), (7, 2), (-7, 2),
    (3, 7), (-3, 7), (7, 3), (-7, 3), (5, 6), (-5, 6), (6, 5), (-6, 5),
    (8, 0), (4, 7), (-4, 7), (7, 4), (-7, 4), (8, 1), (8, 2), (6, 6),
    (-6, 6), (8, 3), (5, 7), (-5, 7), (7, 5), (-7, 5), (8, 4), (6, 7),
    (-6, 7), (7, 6), (-7, 6), (8, 5), (7, 7), (-7, 7), (8, 6), (8, 7),
];

fn special_distance(symbol: usize, width: usize) -> i128 {
    let (x, y) = SPECIAL_DISTANCES[symbol];
    // Wide arithmetic preserves the decoder's signed offset/clamp semantics
    // even at usize::MAX widths, without allocating a width-sized table.
    (i128::from(x) + width as i128 * i128::from(y)).max(1)
}

/// Canonical distance token for dec_ans.h. Width zero disables special codes.
/// Narrow-width aliases (including clamped nonpositive offsets) use the lowest
/// symbol. Zero and distances beyond the decoder's window are rejected.
pub fn distance_symbol(distance: usize, width: usize) -> Result<u32> {
    if !(1..=WINDOW_SIZE).contains(&distance) {
        return Err(Error::InvalidInput(
            "LZ77 distance must be in 1..=1048576".into(),
        ));
    }
    if width != 0 {
        for symbol in 0..SPECIAL_DISTANCES.len() {
            if special_distance(symbol, width) == distance as i128 {
                return Ok(symbol as u32);
            }
        }
    }
    Ok((distance - 1 + if width == 0 { 0 } else { 120 }) as u32)
}

struct DistanceCodes {
    entries: Vec<(usize, u32)>,
    groups: Vec<SpecialPrefixGroup>,
    special_count: usize,
}

impl DistanceCodes {
    fn new(width: usize, window: usize) -> Self {
        let mut canonical = BTreeMap::new();
        if width != 0 {
            for symbol in 0..SPECIAL_DISTANCES.len() {
                let distance = special_distance(symbol, width);
                if distance <= window as i128 {
                    canonical.entry(distance as usize).or_insert(symbol as u32);
                }
            }
        }
        let entries: Vec<_> = canonical.into_iter().collect();
        let distances: Vec<_> = entries.iter().map(|&(distance, _)| distance).collect();
        let groups = group_distances(&distances).expect("positive, sorted, unique distances");
        Self {
            entries,
            groups,
            special_count: if width == 0 { 0 } else { 120 },
        }
    }

    fn symbol(&self, distance: usize) -> u32 {
        match self.entries.binary_search_by_key(&distance, |&(d, _)| d) {
            Ok(index) => self.entries[index].1,
            Err(_) => (self.special_count + distance - 1) as u32,
        }
    }
}

struct ContextCosts {
    bits: BTreeMap<u32, f64>,
    add_symbol: f64,
}

struct SymbolCosts {
    contexts: BTreeMap<u32, ContextCosts>,
}

impl SymbolCosts {
    // SymbolCostEstimator, using sparse context IDs instead of allocating up
    // to max(context). Counts concern hybrid tokens, not full input values.
    fn new(input: &[Literal], options: &Lz77Options) -> Self {
        let mut counts = BTreeMap::<u32, BTreeMap<u32, usize>>::new();
        for literal in input {
            let token = options.literal_uint_config.encode(literal.value).token;
            *counts
                .entry(literal.context)
                .or_default()
                .entry(token)
                .or_default() += 1;
        }
        Self::from_counts(counts, options.force_huffman)
    }

    fn from_counts(counts: BTreeMap<u32, BTreeMap<u32, usize>>, force_huffman: bool) -> Self {
        let contexts = counts
            .into_iter()
            .map(|(context, histogram)| {
                let total: usize = histogram.values().sum();
                let mut total_cost = 0.0;
                let bits = histogram
                    .into_iter()
                    .map(|(symbol, count)| {
                        let mut cost = if count == total {
                            0.0
                        } else {
                            -(count as f64 / total as f64).log2()
                        };
                        if force_huffman {
                            cost = cost.ceil();
                        }
                        total_cost += cost * count as f64;
                        (symbol, cost)
                    })
                    .collect();
                (
                    context,
                    ContextCosts {
                        bits,
                        add_symbol: (6.0 - total_cost / total as f64).max(0.0),
                    },
                )
            })
            .collect();
        Self { contexts }
    }

    fn bits(&self, context: u32, symbol: u32) -> f64 {
        self.contexts
            .get(&context)
            .and_then(|c| c.bits.get(&symbol))
            .copied()
            .unwrap_or(12.0)
    }

    fn add_symbol(&self, context: u32) -> f64 {
        self.contexts.get(&context).map_or(6.0, |c| c.add_symbol)
    }

    fn cumulative(&self, input: &[Literal], config: HybridUintConfig) -> Result<Vec<f64>> {
        let size = input
            .len()
            .checked_add(1)
            .ok_or(Error::Limit("LZ77 input length"))?;
        let mut sums = reserved(size)?;
        sums.push(0.0);
        for literal in input {
            let encoded = config.encode(literal.value);
            sums.push(
                sums.last().copied().unwrap()
                    + self.bits(literal.context, encoded.token)
                    + f64::from(encoded.nbits),
            );
        }
        Ok(sums)
    }
}

// enc_lz77.cc::LenCost and DistCost, including their original trained tables.
#[rustfmt::skip]
const LENGTH_COST: [f64; 17] = [
        2.797667318563126, 3.213177690381199, 2.5706009246743737,
        2.408392498667534, 2.829649191872326, 3.3923087753324577,
        4.029267451554331, 4.415576699706408, 4.509357574741465,
        9.21481543803004, 10.020590190114898, 11.858671627804766,
        12.45853300490526, 11.713105831990857, 12.561996324849314,
        13.775477692278367, 13.174027068768641,
    ];
fn length_cost(length_minus_three: usize) -> f64 {
    let encoded = HybridUintConfig::new(1, 0, 0)
        .unwrap()
        .encode(length_minus_three as u32);
    LENGTH_COST[(encoded.token as usize).min(LENGTH_COST.len() - 1)] + f64::from(encoded.nbits)
}

#[rustfmt::skip]
const DISTANCE_COST: &[f64] = &[
        6.368282626312716, 5.680793277090298, 8.347404197105247,
        7.641619201599141, 6.914328374119438, 7.959808291537444,
        8.70023120759855, 8.71378518934703, 9.379132523982769,
        9.110472749092708, 9.159029569270908, 9.430936766731973,
        7.278284055315169, 7.8278514904267755, 10.026641158289236,
        9.976049229827066, 9.64351607048908, 9.563403863480442,
        10.171474111762747, 10.45950155077234, 9.994813912104219,
        10.322524683741156, 8.465808729388186, 8.756254166066853,
        10.160930174662234, 10.247329273413435, 10.04090403724809,
        10.129398517544082, 9.342311691539546, 9.07608009102374,
        10.104799540677513, 10.378079384990906, 10.165828974075072,
        10.337595322341553, 7.940557464567944, 10.575665823319431,
        11.023344321751955, 10.736144698831827, 11.118277044595054,
        7.468468230648442, 10.738305230932939, 10.906980780216568,
        10.163468216353817, 10.17805759656433, 11.167283670483565,
        11.147050200274544, 10.517921919244333, 10.651764778156886,
        10.17074446448919, 11.217636876224745, 11.261630721139484,
        11.403140815247259, 10.892472096873417, 11.1859607804481,
        8.017346947551262, 7.895143720278828, 11.036577113822025,
        11.170562110315794, 10.326988722591086, 10.40872184751056,
        11.213498225466386, 11.30580635516863, 10.672272515665442,
        10.768069466228063, 11.145257364153565, 11.64668307145549,
        10.593156194627339, 11.207499484844943, 10.767517766396908,
        10.826629811407042, 10.737764794499988, 10.6200448518045,
        10.191315385198092, 8.468384171390085, 11.731295299170432,
        11.824619886654398, 10.41518844301179, 10.16310536548649,
        10.539423685097576, 10.495136599328031, 10.469112847728267,
        11.72057686174922, 10.910326337834674, 11.378921834673758,
        11.847759036098536, 11.92071647623854, 10.810628276345282,
        11.008601085273893, 11.910326337834674, 11.949212023423133,
        11.298614839104337, 11.611603659010392, 10.472930394619985,
        11.835564720850282, 11.523267392285337, 12.01055816679611,
        8.413029688994023, 11.895784139536406, 11.984679534970505,
        11.220654278717394, 11.716311684833672, 10.61036646226114,
        10.89849965960364, 10.203762898863669, 10.997560826267238,
        11.484217379438984, 11.792836176993665, 12.24310468755171,
        11.464858097919262, 12.212747017409377, 11.425595666074955,
        11.572048533398757, 12.742093965163013, 11.381874288645637,
        12.191870445817015, 11.683156920035426, 11.152442115262197,
        11.90303691580457, 11.653292787169159, 11.938615382266098,
        16.970641701570223, 16.853602280380002, 17.26240782594733,
        16.644655390108507, 17.14310889757499, 16.910935455445955,
        17.505678976959697, 17.213498225466388, 2.4162310293553024,
        3.494587244462329, 3.5258600986408344, 3.4959806589517095,
        3.098390886949687, 3.343454654302911, 3.588847442290287,
        4.14614790111827, 5.152948641990529, 7.433696808092598,
        9.716311684833672,
    ];
fn distance_cost(symbol: u32) -> f64 {
    let encoded = HybridUintConfig::new(7, 0, 0).unwrap().encode(symbol);
    DISTANCE_COST[(encoded.token as usize).min(DISTANCE_COST.len() - 1)] + f64::from(encoded.nbits)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Match {
    length: usize,
    distance: usize,
    symbol: u32,
}

impl Match {
    fn command(self, context: u32) -> Command {
        Command::Match {
            context,
            length: self.length as u32,
            distance: self.distance as u32,
        }
    }

    fn cost(self, context: u32, costs: &SymbolCosts) -> f64 {
        length_cost(self.length - MIN_MATCH)
            + distance_cost(self.symbol)
            + costs.add_symbol(context)
    }
}

fn consider_candidate(
    values: &[u32],
    position: usize,
    previous: usize,
    options: &Lz77Options,
    distances: &DistanceCodes,
    best: &mut Match,
) {
    if previous >= position || position - previous > options.window_size {
        return;
    }
    let distance = position - previous;
    let symbol = distances.symbol(distance);
    let required = if best.length == 0 {
        MIN_MATCH
    } else if symbol < best.symbol {
        best.length
    } else {
        let Some(required) = best.length.checked_add(1) else {
            return;
        };
        required
    };
    let limit = options.max_match_length.min(values.len() - position);
    if required > limit
        || values[previous] != values[position]
        || values[previous + required - 1] != values[position + required - 1]
    {
        return;
    }
    let mut length = 0;
    // Compare the immutable original values, allowing a match to overlap its
    // source. Fingerprints and first/last checks are only rejection filters.
    while length < limit && values[previous + length] == values[position + length] {
        length += 1;
    }
    if length >= required {
        *best = Match {
            length,
            distance,
            symbol,
        };
    }
}

struct Matcher<'a, const N: usize> {
    values: &'a [u32],
    options: &'a Lz77Options,
    buckets: Vec<RecentBucket<N>>,
    distances: DistanceCodes,
    prefix: PrefixFilter<'a>,
    next_position: usize,
}

impl<'a, const N: usize> Matcher<'a, N> {
    fn new(values: &'a [u32], width: usize, options: &'a Lz77Options) -> Result<Self> {
        let count = 1usize << options.bucket_bits;
        let mut buckets = reserved(count)?;
        for _ in 0..count {
            buckets.push(RecentBucket::new().expect("validated bucket capacity"));
        }
        Ok(Self {
            values,
            options,
            buckets,
            distances: DistanceCodes::new(width, options.window_size),
            prefix: PrefixFilter::new(values, options.prefix_filter),
            next_position: 0,
        })
    }

    fn insert(&mut self, position: usize) {
        if let Some(hash) = prefix_hash(self.values, position) {
            let bucket = hash as usize & (self.buckets.len() - 1);
            self.buckets[bucket].insert(position, hash);
        }
    }

    // LZ77HashMap::UpdateRange/LZ77InsertRepeatedBucket. After a constant run,
    // only its last N equal-prefix positions survive. Inserting those retains
    // the exact logical queue even though the physical ring rotation differs.
    fn update_range(&mut self, end: usize) {
        debug_assert!(end >= self.next_position && end <= self.values.len());
        while self.next_position < end {
            let begin = self.next_position;
            let limit = end + (self.values.len() - end).min(MIN_MATCH - 1);
            let mut run_end = begin + 1;
            while run_end < limit && self.values[run_end] == self.values[begin] {
                run_end += 1;
            }
            let count = (run_end - begin).saturating_sub(MIN_MATCH - 1);
            if count >= N {
                for position in begin + count - N..begin + count {
                    self.insert(position);
                }
                self.next_position += count;
            } else {
                self.insert(begin);
                self.next_position += 1;
            }
        }
    }

    // Query before inserting, as in LZ77HashMap::FindMatch. Every position is
    // inserted exactly once, including lookahead and skipped match interiors.
    fn find(&mut self, position: usize) -> Match {
        let mut best = Match::default();
        let values = self.values;
        let options = self.options;
        self.visit_candidates(position, |previous, distances| {
            consider_candidate(values, position, previous, options, distances, &mut best);
        });
        best
    }

    fn visit_candidates(&mut self, position: usize, mut visit: impl FnMut(usize, &DistanceCodes)) {
        debug_assert_eq!(position, self.next_position);
        self.next_position += 1;
        let Some(hash) = prefix_hash(self.values, position) else {
            return;
        };
        let bucket_index = hash as usize & (self.buckets.len() - 1);
        let options = self.options;
        let distances = &self.distances;
        self.prefix
            .begin_query(position)
            .expect("query inside input");

        // Matching three-token prefixes necessarily share a bucket. Spatial
        // candidates are needed only beyond its oldest still-retained entry.
        if let Some(oldest) = self.buckets[bucket_index].oldest_if_full() {
            let safe_distance = position - oldest;
            if options.range_search {
                self.prefix
                    .visit_special(position, safe_distance, MIN_MATCH, &distances.groups, |d| {
                        visit(position - d, distances);
                    })
                    .expect("validated spatial groups");
            } else {
                for &(distance, _) in &distances.entries {
                    if distance <= safe_distance || distance > position {
                        continue;
                    }
                    self.prefix.record_candidate_work(1);
                    if self
                        .prefix
                        .may_match(position, position - distance, MIN_MATCH)
                        .expect("previous in-bounds candidate")
                    {
                        visit(position - distance, distances);
                    }
                }
            }
        }
        let query_hash = (options.prefix_filter != FilterPolicy::Disabled).then_some(hash);
        self.buckets[bucket_index].visit(query_hash, MIN_MATCH, |previous| {
            visit(previous, distances);
        });
        self.buckets[bucket_index].insert(position, hash);
    }
}

fn reserved<T>(capacity: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Limit("LZ77 scratch allocation"))?;
    Ok(result)
}

fn literals(input: &[Literal]) -> Result<Vec<Command>> {
    let mut output = reserved(input.len())?;
    output.extend(input.iter().copied().map(Command::Literal));
    Ok(output)
}

fn finish(
    input: &[Literal],
    output: Vec<Command>,
    savings: f64,
    raw_cost: f64,
    options: &Lz77Options,
) -> Result<Vec<Command>> {
    if !options.require_global_savings {
        return Ok(output);
    }
    let symbols = if options.mode == Lz77Mode::Greedy {
        // One match command becomes two entropy tokens, not one.
        output.len() as f64
            + output
                .iter()
                .filter(|c| matches!(c, Command::Match { .. }))
                .count() as f64
    } else {
        input.len() as f64
    };
    let mut minimum = symbols * 0.2 + 16.0;
    if options.mode == Lz77Mode::Greedy && raw_cost <= input.len() as f64 * 2.5 {
        minimum = minimum.min(raw_cost * 0.01 + 256.0);
    }
    if savings > minimum {
        Ok(output)
    } else {
        literals(input)
    }
}

fn parse_rle(input: &[Literal], sums: &[f64], options: &Lz77Options) -> Result<Vec<Command>> {
    let mut output = reserved(input.len())?;
    let mut position = 0;
    let mut savings = 0.0;
    while position < input.len() {
        let mut length = 0;
        if position != 0 {
            let limit = options.max_match_length.min(input.len() - position);
            while length < limit && input[position + length].value == input[position - 1].value {
                length += 1;
            }
        }
        if length == 0 {
            output.push(Command::Literal(input[position]));
            position += 1;
            continue;
        }
        let raw = sums[position + length] - sums[position];
        // ApplyLZ77_RLE: ceil(log2(length - min_length + 1)) + 1.
        let cost = if length >= MIN_MATCH {
            f64::from(usize::BITS - (length - MIN_MATCH).leading_zeros()) + 1.0
        } else {
            0.0
        };
        if length >= MIN_MATCH && raw > cost {
            output.push(Command::Match {
                context: input[position].context,
                length: length as u32,
                distance: 1,
            });
            savings += raw - cost;
        } else {
            output.extend(
                input[position..position + length]
                    .iter()
                    .copied()
                    .map(Command::Literal),
            );
        }
        position += length;
    }
    finish(input, output, savings, sums[input.len()], options)
}

fn parse_matches<const N: usize>(
    input: &[Literal],
    width: usize,
    options: &Lz77Options,
    costs: &SymbolCosts,
    sums: &[f64],
) -> Result<Vec<Command>> {
    let mut values = reserved(input.len())?;
    values.extend(input.iter().map(|literal| literal.value));
    let mut matcher = Matcher::<N>::new(&values, width, options)?;
    let mut output = reserved(input.len())?;
    let mut position = 0;
    let mut savings = 0.0;
    while position < input.len() {
        let mut found = matcher.find(position);
        if found.length < MIN_MATCH {
            output.push(Command::Literal(input[position]));
            position += 1;
            continue;
        }
        if options.mode == Lz77Mode::Lazy && found.length < 256 {
            let next = matcher.find(position + 1);
            if next.length > found.length {
                output.push(Command::Literal(input[position]));
                position += 1;
                found = next;
            }
        }
        let raw = sums[position + found.length] - sums[position];
        let cost = found.cost(input[position].context, costs);
        let consumed = if cost <= raw {
            output.push(found.command(input[position].context));
            savings += raw - cost;
            found.length
        } else {
            // Both skips are from enc_lz77.cc: lazy rejects the whole match;
            // greedy skips a low-entropy interior but retains its two-token tail.
            let count = if options.mode == Lz77Mode::Lazy {
                found.length
            } else if raw < found.length as f64 && found.length > MIN_MATCH {
                found.length - MIN_MATCH + 1
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
        };
        position += consumed;
        matcher.update_range(position);
    }
    finish(input, output, savings, sums[input.len()], options)
}

/// Parse values losslessly. Literal commands retain their original contexts;
/// matches retain only the first replaced context. Expansion reproduces values,
/// not the unused contexts of copied positions. Each call starts a new history.
/// Width zero means no spatial distance coding, not an empty image.
pub fn parse(input: &[Literal], width: usize, options: &Lz77Options) -> Result<Vec<Command>> {
    options.validate()?;
    if options.mode == Lz77Mode::Optimal {
        if input.iter().any(|literal| {
            options.literal_uint_config.encode(literal.value).token >= options.optimal_min_symbol
        }) {
            return Err(Error::InvalidInput(
                "optimal LZ77 literal and length alphabets overlap".into(),
            ));
        }
        if input.len() > MIN_MATCH {
            return dp::parse(input, width, options);
        }
    }
    if input.len() <= MIN_MATCH {
        return literals(input);
    }
    let costs = SymbolCosts::new(input, options);
    let sums = costs.cumulative(input, options.literal_uint_config)?;
    if options.mode == Lz77Mode::Rle {
        return parse_rle(input, &sums, options);
    }
    match options.bucket_capacity {
        1 => parse_matches::<1>(input, width, options, &costs, &sums),
        2 => parse_matches::<2>(input, width, options, &costs, &sums),
        4 => parse_matches::<4>(input, width, options, &costs, &sums),
        8 => parse_matches::<8>(input, width, options, &costs, &sums),
        16 => parse_matches::<16>(input, width, options, &costs, &sums),
        32 => parse_matches::<32>(input, width, options, &costs, &sums),
        64 => parse_matches::<64>(input, width, options, &costs, &sums),
        _ => unreachable!("validated bucket capacity"),
    }
}

#[cfg(test)]
mod tests;
