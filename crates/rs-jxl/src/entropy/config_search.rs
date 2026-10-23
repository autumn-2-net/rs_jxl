// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Port of EntropyEncodingData::ChooseUintConfigs in lib/jxl/enc_ans.cc,
//! baseline b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Retains the source best/fast candidate order, per-histogram value transpose,
//! conservative max-token/LSB-mask validity check, extra-bit accounting, and
//! ANSPopulationCost using Fast normalization and real histogram serialization.
//! LZ77 length tokens retain the caller's separate config and reservation, but
//! are ignored when choosing ordinary configs, as in the source's LZ77 TODO.
//!
//! choose_config_plan ports source Fixed/Fast/Best and streaming policy, the
//! source signaling estimate, and final joint config/alphabet selection. ANS
//! uses log 5..8; prefix uses 15. Like source, adaptive population trials remain
//! limited to 256 symbols even in prefix mode; explicit fixed prefix configs
//! can use all 32768 symbols. Length tokens are added back for alphabet sizing.
//! Fixed/streaming policies validate and size directly from the input streams;
//! only adaptive search allocates a transposed copy of the literal values.
//! Adaptive histograms with at least 4096 values use enc_uint_histogram.h's
//! refining (12,2,5) bins when every candidate is supported. This exact-work
//! optimization is active without the source's experimental environment gate.
//! It does not sample, prune configurations, or change candidate tie order.
//! Extra-bit sums retain Rust's u64 arithmetic rather than the source's u32
//! wraparound above 2^32 bits, identically to the unweighted Rust path.
//! The older choose_configs entry points retain their fixed-log-8 cost policy.
//!
//! Independent adaptive searches use the caller's Rayon pool for at least 4096
//! literals and multiple histograms. Work is ordered largest-first as in source;
//! configs and errors are collected in histogram order, with candidate ties
//! unchanged. No production pool is created or experimental env gate required.
//!
//! Native EstimateTokenCost dispatch is shared by raw and weighted candidates.
//! The internal model entry point retains winning histograms, adds lengths, and
//! passes them directly to table construction. Public config-only APIs remain
//! unchanged. Not ported: fuzzer modes, final-plan replay and audit hooks.

use super::ans_histogram::{AnsHistogramStrategy, NormalizedHistogram};
use super::hybrid_uint::HybridUintConfig;
use super::options::{EntropyOptions, PREFIX_LOG_ALPHA_SIZE, UintConfigStrategy};
use super::stream::{Token, write_uint_config};
use crate::bit_writer::BitWriter;
use crate::{Error, Result};

const LOG_ALPHA_SIZE: u8 = 8;
const ALPHABET_SIZE: usize = 1 << LOG_ALPHA_SIZE;
const LZ77_MIN_SYMBOL: u32 = 224;

const fn config(split: u32, msb: u32, lsb: u32) -> HybridUintConfig {
    match HybridUintConfig::new(split, msb, lsb) {
        Ok(config) => config,
        Err(_) => panic!("invalid source HybridUint configuration"),
    }
}

const LENGTH_CONFIG: HybridUintConfig = config(4, 0, 0);
const BEST_CONFIGS: [HybridUintConfig; 28] = [
    config(4, 2, 0),
    config(4, 1, 0),
    config(4, 2, 1),
    config(4, 2, 2),
    config(4, 1, 2),
    config(5, 2, 0),
    config(5, 1, 0),
    config(5, 2, 1),
    config(5, 2, 2),
    config(5, 1, 2),
    config(3, 2, 0),
    config(3, 1, 0),
    config(3, 2, 1),
    config(3, 1, 2),
    config(4, 1, 3),
    config(5, 1, 4),
    config(5, 2, 3),
    config(6, 1, 5),
    config(6, 2, 4),
    config(6, 0, 0),
    config(0, 0, 0),
    config(2, 0, 1),
    config(7, 0, 0),
    config(8, 0, 0),
    config(9, 0, 0),
    config(10, 0, 0),
    config(11, 0, 0),
    config(12, 0, 0),
];
const FAST_CONFIGS: [HybridUintConfig; 4] = [
    config(4, 2, 0),
    config(4, 1, 2),
    config(0, 0, 0),
    config(2, 0, 1),
];

// enc_uint_histogram.h: one representative preserves every supported config's
// token and extra-bit width; payload bits are still read from original tokens.
struct WeightedUintHistogram {
    values: Vec<(u32, u32)>,
}

impl WeightedUintHistogram {
    const BINS: usize = (1 << 12) + (32 - 12) * (1 << 7);

    fn supports(cfg: HybridUintConfig) -> bool {
        cfg.split_exponent() <= 12
            && cfg.msb_in_token() <= 2
            && cfg.lsb_in_token() <= 5
            && cfg.msb_in_token() + cfg.lsb_in_token() <= cfg.split_exponent()
    }

    fn build(data: &[u32]) -> Result<Option<Self>> {
        if data.len() < 4096 || data.len() > i32::MAX as usize {
            return Ok(None);
        }
        let mut slots = vec![0u32; Self::BINS];
        let mut values: Vec<(u32, u32)> = Vec::new();
        values
            .try_reserve_exact(data.len().min(Self::BINS))
            .map_err(|_| Error::Limit("weighted uint histogram"))?;
        let refining = config(12, 2, 5);
        let mut tokens = [0; super::simd::TOKEN_BATCH];
        let mut widths = [0; super::simd::TOKEN_BATCH];
        for batch in data.chunks(super::simd::TOKEN_BATCH) {
            super::simd::tokens(batch, refining, &mut tokens, &mut widths);
            for (&value, &token) in batch.iter().zip(&tokens) {
                let slot = &mut slots[token as usize];
                if *slot == 0 {
                    values.push((value, 1));
                    *slot = values.len() as u32;
                } else {
                    values[*slot as usize - 1].1 += 1;
                }
            }
        }
        Ok(Some(Self { values }))
    }

    fn add_to(&self, cfg: HybridUintConfig, counts: &mut [u64]) -> u64 {
        debug_assert!(Self::supports(cfg));
        let mut extra_bits = 0u64;
        let mut values = [0; super::simd::TOKEN_BATCH];
        let mut tokens = [0; super::simd::TOKEN_BATCH];
        let mut widths = [0; super::simd::TOKEN_BATCH];
        for batch in self.values.chunks(super::simd::TOKEN_BATCH) {
            for (value, &(representative, _)) in values.iter_mut().zip(batch) {
                *value = representative;
            }
            super::simd::tokens(&values[..batch.len()], cfg, &mut tokens, &mut widths);
            for (i, &(_, count)) in batch.iter().enumerate() {
                counts[tokens[i] as usize] += u64::from(count);
                extra_bits += u64::from(widths[i]) * u64::from(count);
            }
        }
        extra_bits
    }
}

enum HistogramValues<'a> {
    Raw(&'a [u32]),
    Weighted(WeightedUintHistogram),
}

impl<'a> HistogramValues<'a> {
    fn new(values: &'a [u32], candidates: &[HybridUintConfig]) -> Result<Self> {
        if candidates
            .iter()
            .copied()
            .all(WeightedUintHistogram::supports)
            && let Some(weighted) = WeightedUintHistogram::build(values)?
        {
            return Ok(Self::Weighted(weighted));
        }
        Ok(Self::Raw(values))
    }

    fn add_to(&self, cfg: HybridUintConfig, counts: &mut [u64]) -> u64 {
        match self {
            Self::Weighted(weighted) => weighted.add_to(cfg, counts),
            Self::Raw(values) => super::simd::add_tokens(values, cfg, counts),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConfigSearchStrategy {
    Fast,
    #[default]
    Best,
}

impl ConfigSearchStrategy {
    fn candidates(self) -> &'static [HybridUintConfig] {
        match self {
            Self::Fast => &FAST_CONFIGS,
            Self::Best => &BEST_CONFIGS,
        }
    }
}

struct Transposed {
    values: Vec<u32>,
    // First num_histograms ranges are literals; the second half is lengths.
    offsets: Vec<usize>,
    maxima: Vec<u32>,
    max_length_symbol: Option<usize>,
}

struct InputStatistics {
    offsets: Vec<usize>,
    maxima: Vec<u32>,
    max_length_symbol: Option<usize>,
}

fn transpose(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
) -> Result<Transposed> {
    transpose_with_length_config(
        streams,
        context_map,
        num_histograms,
        lz77,
        LENGTH_CONFIG,
        LZ77_MIN_SYMBOL,
        ALPHABET_SIZE,
    )
}

fn input_statistics(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    length_config: HybridUintConfig,
    min_symbol: u32,
    alphabet_size: usize,
) -> Result<InputStatistics> {
    if !(1..=256).contains(&num_histograms) {
        return Err(Error::InvalidInput(
            "config search needs 1..=256 histograms".into(),
        ));
    }
    if context_map
        .iter()
        .any(|&histogram| usize::from(histogram) >= num_histograms)
    {
        return Err(Error::InvalidInput(
            "config search context map exceeds histogram count".into(),
        ));
    }
    if lz77 && context_map.len() < 2 {
        return Err(Error::InvalidInput(
            "LZ77 needs literal and distance contexts".into(),
        ));
    }
    let mut volumes = vec![0usize; 2 * num_histograms];
    let mut totals = vec![0u64; num_histograms];
    let mut maxima = vec![0; 2 * num_histograms];
    let mut max_length_symbol = None;
    for token in streams.iter().flatten() {
        let histogram = usize::from(*context_map.get(token.context).ok_or_else(|| {
            Error::InvalidInput("config search token context is out of range".into())
        })?);
        totals[histogram] += 1;
        if totals[histogram] > i32::MAX as u64 {
            return Err(Error::Limit(
                "config search histogram weight exceeds i32::MAX",
            ));
        }
        if token.is_length {
            if !lz77 || token.context == context_map.len() - 1 {
                return Err(Error::InvalidInput(
                    "invalid length token in config search".into(),
                ));
            }
            let symbol = length_config.encode(token.value).token as usize + min_symbol as usize;
            if token.value > u32::MAX - 3 || symbol >= alphabet_size {
                return Err(Error::InvalidInput(
                    "length token exceeds the selected entropy alphabet".into(),
                ));
            }
            max_length_symbol =
                Some(max_length_symbol.map_or(symbol, |old: usize| old.max(symbol)));
            volumes[histogram + num_histograms] += 1;
            maxima[histogram + num_histograms] =
                maxima[histogram + num_histograms].max(token.value);
        } else {
            volumes[histogram] += 1;
            maxima[histogram] = maxima[histogram].max(token.value);
        }
    }
    let mut offsets = vec![0usize; 2 * num_histograms + 1];
    for histogram in 0..2 * num_histograms {
        offsets[histogram + 1] = offsets[histogram]
            .checked_add(volumes[histogram])
            .ok_or(Error::Limit("config search token count"))?;
    }
    Ok(InputStatistics {
        offsets,
        maxima,
        max_length_symbol,
    })
}

fn transpose_with_length_config(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    length_config: HybridUintConfig,
    min_symbol: u32,
    alphabet_size: usize,
) -> Result<Transposed> {
    let _profile = crate::profile::scope("entropy.uint_config_search.transpose");
    let InputStatistics {
        offsets,
        maxima,
        max_length_symbol,
    } = input_statistics(
        streams,
        context_map,
        num_histograms,
        lz77,
        length_config,
        min_symbol,
        alphabet_size,
    )?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(offsets[2 * num_histograms])
        .map_err(|_| Error::Limit("config search transposed values"))?;
    values.resize(offsets[2 * num_histograms], 0);
    let mut next = offsets[..2 * num_histograms].to_vec();
    for token in streams.iter().flatten() {
        let histogram = usize::from(context_map[token.context])
            + if token.is_length { num_histograms } else { 0 };
        values[next[histogram]] = token.value;
        next[histogram] += 1;
    }
    if crate::profile::enabled() {
        crate::profile::note(
            "entropy.uint_config_search.transpose",
            format_args!(
                "histograms={num_histograms} values={} bytes={} largest_histogram={}",
                values.len(),
                values.len() * std::mem::size_of::<u32>(),
                offsets
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .max()
                    .unwrap_or(0),
            ),
        );
    }
    Ok(Transposed {
        values,
        offsets,
        maxima,
        max_length_symbol,
    })
}

fn signaling_bits(config: HybridUintConfig) -> Result<usize> {
    let mut writer = BitWriter::new(8);
    write_uint_config(&mut writer, config, LOG_ALPHA_SIZE)?;
    Ok(writer.bits_written())
}

fn candidate_cost_data(
    values: &HistogramValues<'_>,
    max_value: u32,
    config: HybridUintConfig,
    lz77: bool,
) -> Result<Option<f32>> {
    if config.split_exponent() > u32::from(LOG_ALPHA_SIZE) {
        return Ok(None);
    }
    let max_token = config.encode(max_value).token | config.lsb_mask();
    // Apply the source's conservative LZ77 ceiling to distance histograms too.
    // A histogram shared by literal and distance contexts is therefore safe.
    if max_token >= ALPHABET_SIZE as u32 || (lz77 && max_token >= LZ77_MIN_SYMBOL) {
        return Ok(None);
    }
    let mut counts = [0u64; ALPHABET_SIZE];
    let extra_bits = values.add_to(config, &mut counts);
    let histogram = NormalizedHistogram::compute(
        &counts[..max_token as usize + 1],
        AnsHistogramStrategy::Fast,
    )?;
    // Preserve source f32 arithmetic and strict comparison for candidate ties.
    let mut cost = histogram.cost;
    cost += extra_bits as f32;
    cost += signaling_bits(config)? as f32;
    Ok(Some(cost))
}

#[cfg(test)]
fn candidate_cost(
    values: &[u32],
    max_value: u32,
    config: HybridUintConfig,
    lz77: bool,
) -> Result<Option<f32>> {
    candidate_cost_data(&HistogramValues::Raw(values), max_value, config, lz77)
}

/// Choose a separate HybridUint configuration for every mapped histogram using
/// the source's best candidate list. The caller's alphabet logarithm must be 8.
/// With LZ77, the final raw context is the distance context and length tokens
/// already contain length-minus-min_length, as in entropy::stream::Token.
///
/// Lengths keep config (4,0,0). Non-length symbols are below 224 with LZ77 and
/// below 256 otherwise. Distance histograms are searched independently rather
/// than inheriting a literal config; if the map shares them, one safe common
/// config is selected. Unused histograms deterministically choose (0,0,0).
pub fn choose_configs(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
) -> Result<Vec<HybridUintConfig>> {
    choose_configs_with_strategy(
        streams,
        context_map,
        num_histograms,
        lz77,
        ConfigSearchStrategy::Best,
    )
}

pub fn choose_configs_with_strategy(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    strategy: ConfigSearchStrategy,
) -> Result<Vec<HybridUintConfig>> {
    let _profile = crate::profile::scope("entropy.uint_config_search.legacy");
    let transposed = transpose(streams, context_map, num_histograms, lz77)?;
    let mut selected = Vec::with_capacity(num_histograms);
    for histogram in 0..num_histograms {
        let values =
            &transposed.values[transposed.offsets[histogram]..transposed.offsets[histogram + 1]];
        let values = HistogramValues::new(values, strategy.candidates())?;
        let mut best = None;
        let mut best_cost = f32::MAX;
        for &config in strategy.candidates() {
            if let Some(cost) =
                candidate_cost_data(&values, transposed.maxima[histogram], config, lz77)?
                && cost < best_cost
            {
                best_cost = cost;
                best = Some(config);
            }
        }
        selected
            .push(best.ok_or_else(|| Error::Encoding("no valid HybridUint configuration".into()))?);
    }
    Ok(selected)
}

/// Configurations and the shared wire alphabet chosen together. Prefix always
/// uses log 15; ANS stores log-5 in its two-bit header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigPlan {
    pub configs: Vec<HybridUintConfig>,
    pub log_alpha_size: u8,
}

#[cfg(test)]
fn source_candidate_cost_data(
    values: &HistogramValues<'_>,
    max_value: u32,
    config: HybridUintConfig,
    alphabet_size: usize,
    literal_limit: usize,
) -> Result<Option<f32>> {
    source_candidate_with_counts(
        values,
        max_value,
        config,
        alphabet_size,
        literal_limit,
        &mut [0; 256],
    )
}

fn source_candidate_with_counts(
    values: &HistogramValues<'_>,
    max_value: u32,
    config: HybridUintConfig,
    alphabet_size: usize,
    literal_limit: usize,
    counts: &mut [u64],
) -> Result<Option<f32>> {
    let maximum = (config.encode(max_value).token | config.lsb_mask()) as usize;
    if maximum >= alphabet_size.min(256) || maximum >= literal_limit {
        return Ok(None);
    }
    counts.fill(0);
    let extra_bits = values.add_to(config, counts);
    let (_, length) = super::simd::condition(&counts[..maximum + 1]).ok_or(Error::Limit(
        "config search histogram weight exceeds i32::MAX",
    ))?;
    let histogram =
        NormalizedHistogram::compute(&counts[..length.max(1)], AnsHistogramStrategy::Fast)?;
    let ceil_log = |n: u32| {
        if n <= 1 {
            0
        } else {
            u32::BITS - (n - 1).leading_zeros()
        }
    };
    let mut cost = histogram.cost;
    cost += extra_bits as f32;
    cost += ceil_log(config.split_exponent() + 1) as f32;
    cost += ceil_log(config.split_exponent() - config.msb_in_token() + 1) as f32;
    Ok(Some(cost))
}

#[cfg(test)]
fn source_candidate_cost(
    values: &[u32],
    max_value: u32,
    config: HybridUintConfig,
    alphabet_size: usize,
    literal_limit: usize,
) -> Result<Option<f32>> {
    source_candidate_cost_data(
        &HistogramValues::Raw(values),
        max_value,
        config,
        alphabet_size,
        literal_limit,
    )
}

fn fixed_config_plan(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    prefix_codes: bool,
    options: &EntropyOptions,
) -> Result<ConfigPlan> {
    let _profile = crate::profile::scope("entropy.uint_config_search.fixed_scan");
    let maximum_log = if prefix_codes {
        PREFIX_LOG_ALPHA_SIZE
    } else {
        options.max_log_alpha_size
    };
    let alphabet_size = 1usize << maximum_log;
    // Retain the original validation order: map/weights/lengths before the
    // fixed config and literal symbols. No token-sized scratch is needed.
    let statistics = input_statistics(
        streams,
        context_map,
        num_histograms,
        lz77,
        options.lz77_length_config,
        options.lz77_min_symbol,
        alphabet_size,
    )?;
    let selected = match options.uint_config {
        UintConfigStrategy::Fixed(config) => config,
        _ => HybridUintConfig::default(),
    };
    if selected.split_exponent() > u32::from(maximum_log) {
        return Err(Error::InvalidInput(
            "fixed uint config exceeds the requested entropy alphabet".into(),
        ));
    }
    let literal_limit = if lz77 {
        options.lz77_min_symbol as usize
    } else {
        alphabet_size
    };
    let mut max_symbol = statistics.max_length_symbol.unwrap_or(0);
    for token in streams.iter().flatten().filter(|token| !token.is_length) {
        // LSB-in-token configurations are not monotonic in the raw value.
        let symbol = selected.encode(token.value).token as usize;
        if symbol >= alphabet_size || symbol >= literal_limit {
            return Err(Error::InvalidInput(
                "uint symbol exceeds its alphabet or LZ77 reservation".into(),
            ));
        }
        max_symbol = max_symbol.max(symbol);
    }
    crate::profile::note(
        "entropy.uint_config_search.fixed_scan",
        format_args!(
            "histograms={num_histograms} values={} transposed_bytes=0",
            statistics.offsets[num_histograms],
        ),
    );
    finish_config_plan(
        vec![selected; num_histograms],
        max_symbol,
        lz77,
        prefix_codes,
        options,
    )
}

/// Source ChooseUintConfigs policy with configurable ANS bounds. Adaptive
/// trials ignore lengths for cost, then include them when sizing the alphabet.
/// Ordinary configs remain per histogram, including the distance histogram.
pub fn choose_config_plan(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    prefix_codes: bool,
    options: &EntropyOptions,
) -> Result<ConfigPlan> {
    let use_workers = !crate::execution::is_serial()
        && rayon::current_thread_index().is_some()
        && crate::execution::worker_count() > 1;
    choose_config_plan_with_workers(
        streams,
        context_map,
        num_histograms,
        lz77,
        prefix_codes,
        options,
        use_workers,
    )
}

fn choose_config_plan_with_workers(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    prefix_codes: bool,
    options: &EntropyOptions,
    use_workers: bool,
) -> Result<ConfigPlan> {
    search_config_plan_with_workers(
        streams,
        context_map,
        num_histograms,
        lz77,
        prefix_codes,
        options,
        use_workers,
    )
    .map(|search| search.plan)
}

pub(super) struct SearchPlan {
    pub(super) plan: ConfigPlan,
    /// Adaptive trials already counted literals. Fixed/streaming policies keep
    /// their non-transposing scan; the caller retains the clustered raw counts.
    pub(super) counts: Option<Vec<Vec<u64>>>,
}

pub(super) fn choose_config_plan_and_histograms(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    prefix_codes: bool,
    options: &EntropyOptions,
) -> Result<SearchPlan> {
    let use_workers = !crate::execution::is_serial()
        && rayon::current_thread_index().is_some()
        && crate::execution::worker_count() > 1;
    search_config_plan_with_workers(
        streams,
        context_map,
        num_histograms,
        lz77,
        prefix_codes,
        options,
        use_workers,
    )
}

fn search_config_plan_with_workers(
    streams: &[Vec<Token>],
    context_map: &[u8],
    num_histograms: usize,
    lz77: bool,
    prefix_codes: bool,
    options: &EntropyOptions,
    use_workers: bool,
) -> Result<SearchPlan> {
    let _profile = crate::profile::scope("entropy.uint_config_search");
    crate::profile::note(
        "entropy.uint_config_search",
        format_args!(
            "histograms={num_histograms} prefix={prefix_codes} uint={:?} streaming={}",
            options.uint_config, options.streaming_mode,
        ),
    );
    options.validate()?;
    let adaptive =
        !options.streaming_mode && !matches!(options.uint_config, UintConfigStrategy::Fixed(_));
    if !adaptive {
        return fixed_config_plan(
            streams,
            context_map,
            num_histograms,
            lz77,
            prefix_codes,
            options,
        )
        .map(|plan| SearchPlan { plan, counts: None });
    }
    let maximum_log = if prefix_codes {
        PREFIX_LOG_ALPHA_SIZE
    } else {
        options.max_log_alpha_size
    };
    let alphabet_size = 1usize << maximum_log;
    let literal_limit = if lz77 {
        options.lz77_min_symbol as usize
    } else {
        alphabet_size
    };
    let grouped = transpose_with_length_config(
        streams,
        context_map,
        num_histograms,
        lz77,
        options.lz77_length_config,
        options.lz77_min_symbol,
        alphabet_size,
    )?;
    let search_profile = crate::profile::scope("entropy.uint_config_search.candidates");
    let mut configs = Vec::with_capacity(num_histograms);
    let mut histograms = Vec::with_capacity(num_histograms);
    let mut max_symbol = grouped.max_length_symbol.unwrap_or(0);
    let mut weighted_histograms = 0usize;
    let mut weighted_values = 0usize;
    let mut weighted_bins = 0usize;
    let candidates = match options.uint_config {
        UintConfigStrategy::Fast => &FAST_CONFIGS[..],
        UintConfigStrategy::Best => &BEST_CONFIGS[..],
        UintConfigStrategy::Fixed(_) => unreachable!(),
    };
    #[cfg(test)]
    let probe = super::worker_test::current();
    let choose = |histogram: usize| {
        #[cfg(test)]
        probe.record("uint.search");
        let values = &grouped.values[grouped.offsets[histogram]..grouped.offsets[histogram + 1]];
        search_histogram(
            values,
            grouped.maxima[histogram],
            candidates,
            alphabet_size,
            literal_limit,
        )
    };
    let parallel = !crate::execution::is_serial()
        && use_workers
        && num_histograms > 1
        && grouped.offsets[num_histograms] >= 4096;
    let results: Vec<_> = if parallel {
        #[cfg(test)]
        probe.record("uint.parallel_dispatch");
        let mut order: Vec<_> = (0..num_histograms).collect();
        order.sort_by_key(|&histogram| {
            std::cmp::Reverse(grouped.offsets[histogram + 1] - grouped.offsets[histogram])
        });
        let largest = grouped
            .offsets
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0);
        // Retain the source's conservative per-root estimate even though token
        // SIMD now uses bounded batches instead of a volume-sized scratch Vec.
        // Source MaxVectorSize returns bytes (32 for AVX2), even though this
        // allocation expression adds it to a uint32 element count.
        let scratch = largest
            .saturating_add(32)
            .saturating_mul(4)
            .saturating_add(1 << 20);
        let mut selected = super::jobs::map(
            &order,
            crate::execution::worker_count(),
            scratch,
            |histogram| (histogram, choose(histogram)),
        );
        selected.sort_by_key(|(histogram, _)| *histogram);
        selected.into_iter().map(|(_, result)| result).collect()
    } else {
        (0..num_histograms).map(choose).collect()
    };
    for result in results {
        let result = result?;
        max_symbol = max_symbol.max(result.max_symbol);
        weighted_histograms += usize::from(result.weighted_bins != 0);
        weighted_values += result.weighted_values;
        weighted_bins += result.weighted_bins;
        configs.push(result.config);
        histograms.push(result.counts);
    }
    crate::profile::note(
        "entropy.uint_config_search.weighted",
        format_args!(
            "histograms={weighted_histograms} values={weighted_values} bins={weighted_bins} parallel={parallel}"
        ),
    );
    drop(search_profile);
    // Source reuses each winning literal histogram and adds the separately
    // configured LZ77 lengths, never re-encodes the winning literal population.
    if lz77 {
        for (h, histogram) in histograms.iter_mut().enumerate() {
            let index = num_histograms + h;
            let values = &grouped.values[grouped.offsets[index]..grouped.offsets[index + 1]];
            if values.is_empty() {
                continue;
            }
            let cfg = options.lz77_length_config;
            let min_symbol = options.lz77_min_symbol as usize;
            let maximum = (cfg.encode(grouped.maxima[index]).token | cfg.lsb_mask()) as usize;
            // Validation checked actual symbols; a conservative LSB mask may
            // overestimate the capacity beyond the requested wire alphabet.
            let capacity = (maximum + min_symbol + 1).min(alphabet_size);
            histogram.resize(histogram.len().max(capacity), 0);
            super::simd::add_tokens(values, cfg, &mut histogram[min_symbol..]);
            let (_, length) = super::simd::condition(histogram).ok_or(Error::Limit(
                "config search histogram weight exceeds i32::MAX",
            ))?;
            histogram.truncate(length.max(1));
        }
    }
    let plan = finish_config_plan(configs, max_symbol, lz77, prefix_codes, options)?;
    Ok(SearchPlan {
        plan,
        counts: Some(histograms),
    })
}

struct HistogramChoice {
    config: HybridUintConfig,
    counts: Vec<u64>,
    max_symbol: usize,
    weighted_values: usize,
    weighted_bins: usize,
}

fn search_histogram(
    values: &[u32],
    maximum: u32,
    candidates: &[HybridUintConfig],
    alphabet_size: usize,
    literal_limit: usize,
) -> Result<HistogramChoice> {
    let data = HistogramValues::new(values, candidates)?;
    let (weighted_values, weighted_bins) = match &data {
        HistogramValues::Weighted(weighted) => (values.len(), weighted.values.len()),
        HistogramValues::Raw(_) => (0, 0),
    };
    let mut best = None;
    let mut best_cost = f32::MAX;
    let mut counts = vec![0; 256];
    let mut best_counts = vec![0; 256];
    for &cfg in candidates {
        if let Some(cost) = source_candidate_with_counts(
            &data,
            maximum,
            cfg,
            alphabet_size,
            literal_limit,
            &mut counts,
        )? && cost < best_cost
        {
            best = Some(cfg);
            best_cost = cost;
            // Source swaps the winning histogram. Recycle the two allocations
            // across candidates; a win swaps buffer ownership, not 256 bins.
            std::mem::swap(&mut counts, &mut best_counts);
        }
    }
    let selected = best.ok_or_else(|| {
        Error::InvalidInput("no uint config fits the requested entropy alphabet".into())
    })?;
    let (_, length) = super::simd::condition(&best_counts).ok_or(Error::Limit(
        "config search histogram weight exceeds i32::MAX",
    ))?;
    let max_symbol = length.saturating_sub(1);
    best_counts.truncate(length.max(1));
    Ok(HistogramChoice {
        config: selected,
        counts: best_counts,
        max_symbol,
        weighted_values,
        weighted_bins,
    })
}

fn finish_config_plan(
    mut configs: Vec<HybridUintConfig>,
    max_symbol: usize,
    lz77: bool,
    prefix_codes: bool,
    options: &EntropyOptions,
) -> Result<ConfigPlan> {
    let adaptive =
        !options.streaming_mode && !matches!(options.uint_config, UintConfigStrategy::Fixed(_));
    let log_alpha_size = if prefix_codes {
        PREFIX_LOG_ALPHA_SIZE
    } else {
        let required = (usize::BITS - max_symbol.leading_zeros()) as u8;
        let log = if options.streaming_mode {
            8
        } else if !adaptive {
            let source_default = if lz77 { 8 } else { 7 };
            let split = configs
                .iter()
                .map(|c| c.split_exponent() as u8)
                .max()
                .unwrap_or(0);
            source_default.min(options.max_log_alpha_size).max(split)
        } else {
            options.min_log_alpha_size
        };
        let log = log.max(required).max(options.min_log_alpha_size);
        if log > options.max_log_alpha_size {
            return Err(Error::InvalidInput(
                "selected uint configs exceed the ANS log-alpha range".into(),
            ));
        }
        // Direct-only configs can outlive the alphabet reduction (e.g. split6
        // with symbols <32). Canonicalize the equivalent direct mapping so the
        // config header never signals a split above the selected logarithm.
        for config in &mut configs {
            if config.split_exponent() >= u32::from(log) {
                *config = HybridUintConfig::new(u32::from(log), 0, 0).expect("supported log");
            }
        }
        log
    };
    crate::profile::note(
        "entropy.uint_config_search.result",
        format_args!("adaptive={adaptive} max_symbol={max_symbol} log_alpha={log_alpha_size}"),
    );
    Ok(ConfigPlan {
        configs,
        log_alpha_size,
    })
}

#[cfg(test)]
mod tests {
    use super::super::ans::AnsCode;
    use super::*;

    #[test]
    fn adaptive_winners_reuse_exact_counts_with_shared_contexts_and_lengths() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            for size in [0, 31, 4095, 4096, 8501] {
                for prefix in [false, true] {
                    for lz77 in [false, true] {
                        for strategy in [UintConfigStrategy::Fast, UintConfigStrategy::Best] {
                            let mut streams = vec![Vec::new(), Vec::new()];
                            for i in 0..size {
                                let value = if i % 37 == 0 {
                                    u32::MAX
                                } else {
                                    (i % 71) as u32
                                };
                                streams[i % 2].push(literal(if i % 5 == 0 { 2 } else { 0 }, value));
                            }
                            // Histogram zero is raw, one crosses the weighted
                            // threshold, two is distance, and three stays empty.
                            streams[0].extend((0..19).map(|i| literal(1, i * 333)));
                            streams[1].extend((0..7).map(|i| literal(3, i * 17)));
                            if lz77 {
                                for context in [0, 1, 2] {
                                    streams[1].push(Token {
                                        context,
                                        value: 1 << 30,
                                        is_length: true,
                                    });
                                }
                            }
                            let map = [1, 0, 1, 2];
                            let options = EntropyOptions {
                                uint_config: strategy,
                                lz77_length_config: config(0, 0, 0),
                                ..EntropyOptions::default()
                            };
                            let search = search_config_plan_with_workers(
                                &streams, &map, 4, lz77, prefix, &options, true,
                            )
                            .unwrap();
                            let serial = search_config_plan_with_workers(
                                &streams, &map, 4, lz77, prefix, &options, false,
                            )
                            .unwrap();
                            assert_eq!(search.plan, serial.plan);
                            assert_eq!(search.counts, serial.counts);
                            let mut expected = vec![vec![0u64]; 4];
                            for token in streams.iter().flatten() {
                                let h = usize::from(map[token.context]);
                                let cfg = if token.is_length {
                                    options.lz77_length_config
                                } else {
                                    search.plan.configs[h]
                                };
                                let symbol = cfg.encode(token.value).token as usize
                                    + if token.is_length {
                                        options.lz77_min_symbol as usize
                                    } else {
                                        0
                                    };
                                let histogram = &mut expected[h];
                                histogram.resize(histogram.len().max(symbol + 1), 0);
                                histogram[symbol] += 1;
                            }
                            assert_eq!(search.counts.unwrap(), expected);
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn fixed_and_streaming_plans_do_not_allocate_adaptive_histograms() {
        let streams = [vec![literal(0, 7); 33]];
        for options in [
            EntropyOptions {
                uint_config: UintConfigStrategy::Fixed(config(4, 2, 0)),
                ..EntropyOptions::default()
            },
            EntropyOptions {
                streaming_mode: true,
                ..EntropyOptions::default()
            },
        ] {
            let search =
                choose_config_plan_and_histograms(&streams, &[0], 1, false, false, &options)
                    .unwrap();
            assert!(search.counts.is_none());
            assert_eq!(
                search.plan,
                choose_config_plan(&streams, &[0], 1, false, false, &options).unwrap()
            );
        }
    }

    fn literal(context: usize, value: u32) -> Token {
        Token {
            context,
            value,
            is_length: false,
        }
    }

    #[test]
    fn serial_scope_inside_pool_keeps_config_trials_on_caller_thread() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let streams = [
            (0..4096).map(|i| literal(0, i % 31)).collect(),
            (0..4096)
                .map(|i| literal(1 + i as usize % 3, i % 127))
                .collect(),
        ];
        let options = EntropyOptions::default();
        pool.install(|| {
            let caller = std::thread::current().id();
            assert_eq!(rayon::current_num_threads(), 4);
            crate::execution::serial(|| {
                let (plan, events) = super::super::worker_test::observe(|| {
                    choose_config_plan(&streams, &[0, 1, 2, 3], 4, false, false, &options).unwrap()
                });
                assert_eq!(plan.configs.len(), 4);
                assert_eq!(events.len(), 4);
                assert!(
                    events
                        .iter()
                        .all(|e| { e.stage == "uint.search" && e.thread == caller && e.serial }),
                    "{events:?}"
                );
                let (_, events) = super::super::worker_test::observe(|| {
                    choose_config_plan_with_workers(
                        &streams,
                        &[0, 1, 2, 3],
                        4,
                        false,
                        false,
                        &options,
                        true,
                    )
                    .unwrap()
                });
                assert_eq!(events.len(), 4);
                assert!(
                    events
                        .iter()
                        .all(|e| { e.stage == "uint.search" && e.thread == caller && e.serial }),
                    "{events:?}"
                );
            });
            assert!(!crate::execution::is_serial());
        });
    }

    #[test]
    fn worker_search_preserves_raw_candidate_order_configs_and_alphabet() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let mut values = Vec::new();
            for (context, length) in [(0, 14000), (1, 8000), (2, 3071), (3, 3), (4, 2400)] {
                for i in 0..length {
                    let value = match context {
                        0 | 3 => (i % 17) as u32,
                        1 => {
                            if i % 23 == 0 {
                                u32::MAX
                            } else {
                                (i % 511) as u32
                            }
                        }
                        2 => (i as u32).wrapping_mul(1_000_003),
                        _ => (i % 127) as u32,
                    };
                    values.push(literal(context, value));
                }
            }
            let map = [3, 1, 0, 3, 2];
            for lz77 in [false, true] {
                let mut streams = vec![values[..16000].to_vec(), values[16000..].to_vec()];
                if lz77 {
                    streams[1].push(Token {
                        context: 0,
                        value: 1 << 30,
                        is_length: true,
                    });
                }
                for strategy in [UintConfigStrategy::Fast, UintConfigStrategy::Best] {
                    for prefix in [false, true] {
                        for maximum in [5, 8] {
                            let options = EntropyOptions {
                                uint_config: strategy,
                                max_log_alpha_size: maximum,
                                lz77_length_config: config(0, 0, 0),
                                ..EntropyOptions::default()
                            };
                            let serial = choose_config_plan_with_workers(
                                &streams, &map, 6, lz77, prefix, &options, false,
                            )
                            .map_err(|e| e.to_string());
                            let parallel = choose_config_plan_with_workers(
                                &streams, &map, 6, lz77, prefix, &options, true,
                            )
                            .map_err(|e| e.to_string());
                            assert_eq!(parallel, serial);
                            assert_eq!(
                                parallel,
                                raw_adaptive_reference(&streams, &map, 6, lz77, prefix, &options,)
                                    .map_err(|e| e.to_string())
                            );
                            assert_eq!(
                                parallel,
                                choose_config_plan(&streams, &map, 6, lz77, prefix, &options,)
                                    .map_err(|e| e.to_string())
                            );
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn worker_search_keeps_empty_ties_small_input_and_validation_errors() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        pool.install(|| {
            for length in [0, 4095, 4096] {
                let streams = [vec![literal(1, 0); length]];
                let options = EntropyOptions::default();
                let expected = choose_config_plan_with_workers(
                    &streams,
                    &[0, 1, 2],
                    3,
                    false,
                    false,
                    &options,
                    false,
                )
                .unwrap();
                let actual = choose_config_plan_with_workers(
                    &streams,
                    &[0, 1, 2],
                    3,
                    false,
                    false,
                    &options,
                    true,
                )
                .unwrap();
                assert_eq!(actual, expected);
                assert_eq!(actual.configs, vec![config(0, 0, 0); 3]);
            }
            let options = EntropyOptions::default();
            for tokens in [
                vec![literal(9, 0)],
                vec![Token {
                    context: 1,
                    value: 4,
                    is_length: true,
                }],
                vec![Token {
                    context: 0,
                    value: u32::MAX,
                    is_length: true,
                }],
            ] {
                let streams = [vec![literal(0, 7); 4096], tokens];
                let serial = choose_config_plan_with_workers(
                    &streams,
                    &[0, 1],
                    2,
                    true,
                    false,
                    &options,
                    false,
                )
                .unwrap_err();
                let parallel = choose_config_plan_with_workers(
                    &streams,
                    &[0, 1],
                    2,
                    true,
                    false,
                    &options,
                    true,
                )
                .unwrap_err();
                assert_eq!(serial.to_string(), parallel.to_string());
            }
        });
    }

    #[test]
    fn refining_histogram_preserves_all_supported_tokens_and_extra_bit_counts() {
        let mut data: Vec<u32> = (0..65536).collect();
        let mut random = 1729u32;
        for _ in 0..4096 {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            data.push(random);
        }
        for bits in 0..32 {
            let value = 1u32 << bits;
            data.extend([value - 1, value, value + 1, u32::MAX]);
        }
        let weighted = WeightedUintHistogram::build(&data).unwrap().unwrap();
        assert!(weighted.values.len() <= WeightedUintHistogram::BINS);
        assert_eq!(
            weighted.values.iter().map(|v| u64::from(v.1)).sum::<u64>(),
            data.len() as u64
        );
        for split in 0..=12u32 {
            for msb in 0..=2u32.min(split) {
                for lsb in 0..=5u32.min(split - msb) {
                    let cfg = config(split, msb, lsb);
                    assert!(WeightedUintHistogram::supports(cfg));
                    let maximum = (cfg.encode(u32::MAX).token | cfg.lsb_mask()) as usize;
                    let mut expected = vec![0u64; maximum + 1];
                    let mut actual = vec![0u64; maximum + 1];
                    let extra = HistogramValues::Raw(&data).add_to(cfg, &mut expected);
                    assert_eq!(weighted.add_to(cfg, &mut actual), extra, "{cfg:?}");
                    assert_eq!(actual, expected, "{cfg:?}");
                }
            }
        }
    }

    #[test]
    fn weighted_search_threshold_fallback_and_large_extra_bit_totals() {
        let small = vec![0; 4095];
        assert!(WeightedUintHistogram::build(&small).unwrap().is_none());
        assert!(WeightedUintHistogram::build(&[]).unwrap().is_none());
        assert!(matches!(
            HistogramValues::new(&small, &BEST_CONFIGS).unwrap(),
            HistogramValues::Raw(_)
        ));
        let large = vec![u32::MAX; 4096];
        let HistogramValues::Weighted(weighted) =
            HistogramValues::new(&large, &BEST_CONFIGS).unwrap()
        else {
            panic!("source threshold must activate weighted search");
        };
        assert_eq!(weighted.values, [(u32::MAX, 4096)]);
        for cfg in [config(13, 0, 0), config(4, 3, 0), config(6, 0, 6)] {
            assert!(!WeightedUintHistogram::supports(cfg));
            assert!(matches!(
                HistogramValues::new(&large, &[cfg]).unwrap(),
                HistogramValues::Raw(_)
            ));
        }
        // Avoid a multi-billion-value fixture: exercise the same bounded bin
        // weight directly, including an extra-bit sum above the C++ u32 limit.
        let weighted = WeightedUintHistogram {
            values: vec![(u32::MAX, i32::MAX as u32)],
        };
        let mut counts = [0u64; 33];
        assert_eq!(
            weighted.add_to(config(0, 0, 0), &mut counts),
            31 * i32::MAX as u64
        );
        assert_eq!(counts[32], i32::MAX as u64);
    }

    #[test]
    fn weighted_and_raw_candidate_costs_match_exactly() {
        let data: Vec<u32> = (0..8192u32)
            .map(|i| if i % 31 == 0 { u32::MAX } else { i % 511 })
            .collect();
        let weighted = HistogramValues::new(&data, &BEST_CONFIGS).unwrap();
        assert!(matches!(weighted, HistogramValues::Weighted(_)));
        for &cfg in &BEST_CONFIGS {
            for lz77 in [false, true] {
                assert_eq!(
                    candidate_cost_data(&weighted, u32::MAX, cfg, lz77)
                        .unwrap()
                        .map(f32::to_bits),
                    candidate_cost(&data, u32::MAX, cfg, lz77)
                        .unwrap()
                        .map(f32::to_bits),
                );
            }
            for alphabet in [32, 128, 256, 32768] {
                for reservation in [8, 224, 512, alphabet] {
                    assert_eq!(
                        source_candidate_cost_data(&weighted, u32::MAX, cfg, alphabet, reservation)
                            .unwrap()
                            .map(f32::to_bits),
                        source_candidate_cost(&data, u32::MAX, cfg, alphabet, reservation)
                            .unwrap()
                            .map(f32::to_bits),
                    );
                }
            }
        }
    }

    fn raw_adaptive_reference(
        streams: &[Vec<Token>],
        map: &[u8],
        histograms: usize,
        lz77: bool,
        prefix: bool,
        options: &EntropyOptions,
    ) -> Result<ConfigPlan> {
        let alphabet = 1usize
            << if prefix {
                15
            } else {
                options.max_log_alpha_size
            };
        let limit = if lz77 {
            options.lz77_min_symbol as usize
        } else {
            alphabet
        };
        let grouped = transpose_with_length_config(
            streams,
            map,
            histograms,
            lz77,
            options.lz77_length_config,
            options.lz77_min_symbol,
            alphabet,
        )?;
        let candidates: &[_] = match options.uint_config {
            UintConfigStrategy::Best => &BEST_CONFIGS,
            UintConfigStrategy::Fast => &FAST_CONFIGS,
            _ => panic!("adaptive reference only"),
        };
        let mut configs = Vec::new();
        let mut maximum = grouped.max_length_symbol.unwrap_or(0);
        for h in 0..histograms {
            let values = &grouped.values[grouped.offsets[h]..grouped.offsets[h + 1]];
            let mut best = None;
            let mut best_cost = f32::MAX;
            for &cfg in candidates {
                if let Some(cost) =
                    source_candidate_cost(values, grouped.maxima[h], cfg, alphabet, limit)?
                    && cost < best_cost
                {
                    best = Some(cfg);
                    best_cost = cost;
                }
            }
            let cfg = best.ok_or_else(|| {
                Error::InvalidInput("no uint config fits the requested entropy alphabet".into())
            })?;
            for &value in values {
                maximum = maximum.max(cfg.encode(value).token as usize);
            }
            configs.push(cfg);
        }
        finish_config_plan(configs, maximum, lz77, prefix, options)
    }

    #[test]
    fn active_weighted_plans_match_raw_search_with_lengths_and_shared_contexts() {
        let mut streams = vec![Vec::new(), Vec::new()];
        for i in 0..24576usize {
            streams[i % 2].push(literal(
                i % 3,
                if i % 19 == 0 {
                    u32::MAX
                } else {
                    (i % 1023) as u32
                },
            ));
        }
        for lz77 in [false, true] {
            let mut input = streams.clone();
            if lz77 {
                input[0].push(Token {
                    context: 1,
                    value: 1 << 30,
                    is_length: true,
                });
            }
            for map in [[0, 1, 0], [2, 0, 1]] {
                for uint_config in [UintConfigStrategy::Fast, UintConfigStrategy::Best] {
                    for prefix in [false, true] {
                        let options = EntropyOptions {
                            uint_config,
                            lz77_length_config: config(0, 0, 0),
                            ..EntropyOptions::default()
                        };
                        let actual =
                            choose_config_plan(&input, &map, 4, lz77, prefix, &options).unwrap();
                        let expected =
                            raw_adaptive_reference(&input, &map, 4, lz77, prefix, &options)
                                .unwrap();
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }

    // Retain the pre-fast-path algorithm as an oracle: materialize histogram-
    // ordered values, then apply the fixed config to every actual value.
    fn transposed_fixed_reference(
        streams: &[Vec<Token>],
        context_map: &[u8],
        num_histograms: usize,
        lz77: bool,
        prefix_codes: bool,
        options: &EntropyOptions,
    ) -> Result<ConfigPlan> {
        options.validate()?;
        assert!(
            options.streaming_mode || matches!(options.uint_config, UintConfigStrategy::Fixed(_))
        );
        let maximum_log = if prefix_codes {
            15
        } else {
            options.max_log_alpha_size
        };
        let alphabet_size = 1usize << maximum_log;
        let literal_limit = if lz77 {
            options.lz77_min_symbol as usize
        } else {
            alphabet_size
        };
        let grouped = transpose_with_length_config(
            streams,
            context_map,
            num_histograms,
            lz77,
            options.lz77_length_config,
            options.lz77_min_symbol,
            alphabet_size,
        )?;
        let selected = match options.uint_config {
            UintConfigStrategy::Fixed(config) => config,
            _ => HybridUintConfig::default(),
        };
        let mut configs = Vec::new();
        let mut max_symbol = grouped.max_length_symbol.unwrap_or(0);
        for histogram in 0..num_histograms {
            if selected.split_exponent() > u32::from(maximum_log) {
                return Err(Error::InvalidInput(
                    "fixed uint config exceeds the requested entropy alphabet".into(),
                ));
            }
            for &value in
                &grouped.values[grouped.offsets[histogram]..grouped.offsets[histogram + 1]]
            {
                let symbol = selected.encode(value).token as usize;
                if symbol >= alphabet_size || symbol >= literal_limit {
                    return Err(Error::InvalidInput(
                        "uint symbol exceeds its alphabet or LZ77 reservation".into(),
                    ));
                }
                max_symbol = max_symbol.max(symbol);
            }
            configs.push(selected);
        }
        finish_config_plan(configs, max_symbol, lz77, prefix_codes, options)
    }

    #[test]
    fn fixed_scan_matches_transposed_plans_and_errors() {
        let values = [0, 1, 7, 15, 16, 47, 48, 223, 224, 255, 511, 32767, u32::MAX];
        let literals: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| literal(i % 3, v))
            .collect();
        let inputs = [
            (vec![], false),
            (vec![vec![], vec![literal(0, 0), literal(2, 7)]], false),
            (vec![literals.clone(), vec![literal(1, 48)]], false),
            (
                vec![
                    literals,
                    vec![Token {
                        context: 1,
                        value: u32::MAX - 3,
                        is_length: true,
                    }],
                ],
                true,
            ),
        ];
        for (streams, lz77) in &inputs {
            for map in [[0, 0, 0], [2, 0, 2]] {
                for strategy in [
                    UintConfigStrategy::Fixed(config(0, 0, 0)),
                    UintConfigStrategy::Fixed(config(2, 0, 1)),
                    UintConfigStrategy::Fixed(config(4, 2, 0)),
                    UintConfigStrategy::Fixed(config(4, 1, 2)),
                    UintConfigStrategy::Fixed(config(8, 0, 0)),
                    UintConfigStrategy::Fixed(config(15, 0, 0)),
                    UintConfigStrategy::Fast,
                    UintConfigStrategy::Best,
                ] {
                    for streaming_mode in [false, true] {
                        if !streaming_mode && !matches!(strategy, UintConfigStrategy::Fixed(_)) {
                            continue;
                        }
                        for prefix in [false, true] {
                            for maximum in [5, 7, 8] {
                                for minimum_symbol in [224, 512] {
                                    let options = EntropyOptions {
                                        uint_config: strategy,
                                        streaming_mode,
                                        max_log_alpha_size: maximum,
                                        lz77_min_symbol: minimum_symbol,
                                        lz77_length_config: config(0, 0, 0),
                                        ..EntropyOptions::default()
                                    };
                                    let actual = choose_config_plan(
                                        streams, &map, 4, *lz77, prefix, &options,
                                    );
                                    let expected = transposed_fixed_reference(
                                        streams, &map, 4, *lz77, prefix, &options,
                                    );
                                    assert_eq!(
                                        actual.map_err(|error| format!("{error:?}")),
                                        expected.map_err(|error| format!("{error:?}")),
                                        "{strategy:?}, streaming={streaming_mode}, prefix={prefix}, log={maximum}, min={minimum_symbol}, lz77={lz77}, map={map:?}",
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_scan_uses_observed_symbols_not_raw_maximum_or_lsb_ceiling() {
        let cfg = config(4, 0, 4);
        assert_eq!(cfg.encode(47).token, 47);
        assert_eq!(cfg.encode(48).token, 32);
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(cfg),
            lz77_min_symbol: 40,
            ..EntropyOptions::default()
        };
        for context in 0..2 {
            let accepted = [vec![literal(context, 48)]];
            let plan = choose_config_plan(&accepted, &[0, 0], 1, true, false, &options).unwrap();
            assert_eq!(plan.configs, [cfg]);
            assert_eq!(plan.log_alpha_size, 8);
            let rejected = [vec![literal(context, 47), literal(context, 48)]];
            let error =
                choose_config_plan(&rejected, &[0, 0], 1, true, false, &options).unwrap_err();
            assert!(matches!(error, Error::InvalidInput(message)
                if message == "uint symbol exceeds its alphabet or LZ77 reservation"));
        }
    }

    #[test]
    fn fixed_scan_preserves_validation_errors_before_config_errors() {
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(config(15, 0, 0)),
            lz77_length_config: config(0, 0, 0),
            ..EntropyOptions::default()
        };
        for (map, histograms, token, expected_message) in [
            (
                vec![0, 0],
                0,
                literal(0, 0),
                "config search needs 1..=256 histograms",
            ),
            (
                vec![0, 0],
                257,
                literal(0, 0),
                "config search needs 1..=256 histograms",
            ),
            (
                vec![0, 1],
                1,
                literal(0, 0),
                "config search context map exceeds histogram count",
            ),
            (
                vec![0],
                1,
                literal(0, 0),
                "LZ77 needs literal and distance contexts",
            ),
            (
                vec![0, 0],
                1,
                literal(2, 0),
                "config search token context is out of range",
            ),
            (
                vec![0, 0],
                1,
                Token {
                    context: 1,
                    value: 0,
                    is_length: true,
                },
                "invalid length token in config search",
            ),
            (
                vec![0, 0],
                1,
                Token {
                    context: 0,
                    value: u32::MAX,
                    is_length: true,
                },
                "length token exceeds the selected entropy alphabet",
            ),
            (
                vec![0, 0],
                1,
                literal(0, 0),
                "fixed uint config exceeds the requested entropy alphabet",
            ),
        ] {
            let streams = [vec![literal(0, u32::MAX), token]];
            let actual =
                choose_config_plan(&streams, &map, histograms, true, false, &options).unwrap_err();
            let expected =
                transposed_fixed_reference(&streams, &map, histograms, true, false, &options)
                    .unwrap_err();
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            assert!(matches!(actual, Error::InvalidInput(message) if message == expected_message));
        }
        let reserved = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(config(8, 0, 0)),
            ..options
        };
        let streams = [vec![
            literal(0, 224),
            Token {
                context: 0,
                value: u32::MAX,
                is_length: true,
            },
        ]];
        let error = choose_config_plan(&streams, &[0, 0], 1, true, false, &reserved).unwrap_err();
        assert!(matches!(error, Error::InvalidInput(message)
            if message == "length token exceeds the selected entropy alphabet"));
    }

    fn independent_cost(values: &[u32], config: HybridUintConfig) -> f64 {
        let mut counts = [0u64; 256];
        let mut extra = 0usize;
        for &value in values {
            let encoded = config.encode(value);
            counts[encoded.token as usize] += 1;
            extra += encoded.nbits as usize;
        }
        let code =
            AnsCode::from_counts_with_strategy(&counts, 8, AnsHistogramStrategy::Fast).unwrap();
        let mut writer = BitWriter::new(4096);
        write_uint_config(&mut writer, config, 8).unwrap();
        code.write_histogram(&mut writer).unwrap();
        let data: f64 = counts
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count != 0)
            .map(|(symbol, &count)| count as f64 * code.symbol_cost(symbol).unwrap())
            .sum();
        writer.bits_written() as f64 + data + extra as f64
    }

    #[test]
    fn source_candidate_order_and_exact_config_signaling() {
        assert_eq!(BEST_CONFIGS.len(), 28);
        assert_eq!(BEST_CONFIGS[0], HybridUintConfig::default());
        assert_eq!(
            FAST_CONFIGS,
            [
                config(4, 2, 0),
                config(4, 1, 2),
                config(0, 0, 0),
                config(2, 0, 1)
            ]
        );
        assert_eq!(signaling_bits(config(4, 2, 0)).unwrap(), 9);
        assert_eq!(signaling_bits(config(0, 0, 0)).unwrap(), 4);
        assert_eq!(signaling_bits(config(8, 0, 0)).unwrap(), 4);
        assert!(
            candidate_cost(&[0], 0, config(9, 0, 0), false)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn transpose_respects_context_map_and_separates_lengths() {
        let streams = vec![
            vec![
                literal(0, 7),
                literal(2, 90),
                Token {
                    context: 1,
                    value: 17,
                    is_length: true,
                },
            ],
            vec![literal(1, 11), literal(0, 5), literal(2, 4)],
        ];
        let grouped = transpose(&streams, &[1, 1, 0], 3, true).unwrap();
        assert_eq!(grouped.offsets, [0, 2, 5, 5, 5, 6, 6]);
        assert_eq!(grouped.values, [90, 4, 7, 11, 5, 17]);
        assert_eq!(grouped.maxima, [90, 11, 0, 0, 17, 0]);
    }

    #[test]
    fn empty_histograms_and_distinct_distance_configuration() {
        assert_eq!(
            choose_configs(&[], &[0, 1], 2, true).unwrap(),
            [config(0, 0, 0); 2]
        );
        let mut stream: Vec<Token> = (0..2048).map(|i| literal(0, i % 64)).collect();
        stream.extend((0..2048).map(|_| literal(1, 0)));
        let selected = choose_configs(&[stream], &[0, 1], 2, true).unwrap();
        assert_eq!(selected, [config(8, 0, 0), config(0, 0, 0)]);
    }

    #[test]
    fn includes_extra_bits_and_actual_normalized_histogram_headers() {
        let values: Vec<u32> = (0..1000).map(|i| (i * 73 + 19) % 8191).collect();
        for &config in &BEST_CONFIGS {
            if let Some(cost) =
                candidate_cost(&values, *values.iter().max().unwrap(), config, false).unwrap()
            {
                assert!((f64::from(cost) - independent_cost(&values, config)).abs() < 0.03);
            }
        }
        let streams = [values.iter().map(|&value| literal(0, value)).collect()];
        let selected = choose_configs(&streams, &[0], 1, false).unwrap()[0];
        let selected_cost = independent_cost(&values, selected);
        for &config in &BEST_CONFIGS {
            if candidate_cost(&values, *values.iter().max().unwrap(), config, false)
                .unwrap()
                .is_some()
            {
                assert!(selected_cost <= independent_cost(&values, config) + 0.03);
            }
        }
    }

    #[test]
    fn lengths_are_fixed_and_do_not_bias_literal_search() {
        let stream: Vec<Token> = (0..512).map(|i| literal(0, i * 11)).collect();
        let baseline = choose_configs(std::slice::from_ref(&stream), &[0, 1], 2, true).unwrap();
        let mut with_lengths = stream;
        with_lengths.extend((0..400).map(|i| Token {
            context: 0,
            value: i * 997,
            is_length: true,
        }));
        let selected = choose_configs(&[with_lengths], &[0, 1], 2, true).unwrap();
        assert_eq!(selected, baseline);
        assert_eq!(
            LENGTH_CONFIG.encode((1 << 20) - 1).token + LZ77_MIN_SYMBOL,
            255
        );
        assert_eq!(LENGTH_CONFIG.encode(1 << 20).token + LZ77_MIN_SYMBOL, 256);
    }

    #[test]
    fn full_u32_domain_and_shared_literal_distance_histograms_stay_safe() {
        let mut values = vec![0, 1, u32::MAX];
        for exponent in 0..32 {
            values.push((1u32 << exponent) - 1);
            values.push(1u32 << exponent);
        }
        let streams = [values
            .iter()
            .enumerate()
            .map(|(i, &value)| literal(i % 2, value))
            .collect()];
        for map in [[0, 0], [1, 0]] {
            for strategy in [ConfigSearchStrategy::Fast, ConfigSearchStrategy::Best] {
                let selected =
                    choose_configs_with_strategy(&streams, &map, 2, true, strategy).unwrap();
                for token in &streams[0] {
                    let config = selected[usize::from(map[token.context])];
                    assert!(config.encode(token.value).token < 224);
                    assert!(config.split_exponent() <= 8);
                }
                assert_eq!(
                    selected,
                    choose_configs_with_strategy(&streams, &map, 2, true, strategy).unwrap()
                );
            }
        }
        assert!(
            candidate_cost(&[u32::MAX], u32::MAX, config(4, 1, 2), true)
                .unwrap()
                .is_none()
        );
        assert!(
            candidate_cost(&[u32::MAX], u32::MAX, config(4, 1, 2), false)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn joint_plan_sizes_lengths_and_keeps_their_config_separate() {
        let streams = [vec![literal(0, 0), literal(1, 0)]];
        let options = EntropyOptions {
            lz77_length_config: config(0, 0, 0),
            ..EntropyOptions::default()
        };
        let baseline = choose_config_plan(&streams, &[0, 1], 2, true, false, &options).unwrap();
        assert_eq!(baseline.log_alpha_size, 5);
        assert_eq!(baseline.configs, [config(0, 0, 0); 2]);
        let mut with_lengths = streams[0].clone();
        with_lengths.push(Token {
            context: 0,
            value: 1 << 30,
            is_length: true,
        });
        let with_lengths = [with_lengths];
        let selected =
            choose_config_plan(&with_lengths, &[0, 1], 2, true, false, &options).unwrap();
        assert_eq!(selected.configs, baseline.configs);
        assert_eq!(selected.log_alpha_size, 8); // 224 + token31 = 255
        let prefix_options = EntropyOptions {
            lz77_min_symbol: 512,
            ..options
        };
        assert!(
            choose_config_plan(&with_lengths, &[0, 1], 2, true, false, &prefix_options).is_err()
        );
        let prefix =
            choose_config_plan(&with_lengths, &[0, 1], 2, true, true, &prefix_options).unwrap();
        assert_eq!(prefix.log_alpha_size, 15);
        assert_eq!(prefix.configs, baseline.configs);
    }

    #[test]
    fn joint_plan_fixed_streaming_and_restricted_alphabets() {
        let mut options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(config(0, 0, 0)),
            ..EntropyOptions::default()
        };
        assert_eq!(
            choose_config_plan(&[], &[0], 1, false, false, &options)
                .unwrap()
                .log_alpha_size,
            7
        );
        assert_eq!(
            choose_config_plan(&[], &[0, 0], 1, true, false, &options)
                .unwrap()
                .log_alpha_size,
            8
        );
        options.streaming_mode = true;
        assert_eq!(
            choose_config_plan(&[], &[0], 1, false, false, &options)
                .unwrap()
                .log_alpha_size,
            8
        );
        options.max_log_alpha_size = 7;
        assert!(choose_config_plan(&[], &[0], 1, false, false, &options).is_err());
        options.streaming_mode = false;
        options.uint_config = UintConfigStrategy::Fixed(config(15, 0, 0));
        let streams = [vec![literal(0, 32767)]];
        let wide = choose_config_plan(&streams, &[0], 1, false, true, &options).unwrap();
        assert_eq!(wide.configs, [config(15, 0, 0)]);
        assert_eq!(wide.log_alpha_size, 15);
        assert!(choose_config_plan(&streams, &[0], 1, false, false, &options).is_err());
        options.uint_config = UintConfigStrategy::Best;
        options.max_log_alpha_size = 5;
        let small =
            choose_config_plan(&[vec![literal(0, 7)]], &[0], 1, false, false, &options).unwrap();
        assert_eq!(small.log_alpha_size, 5);
        assert!(small.configs.iter().all(|c| c.split_exponent() <= 5));
    }

    #[test]
    fn source_search_cost_includes_histogram_data_extra_and_source_config_estimate() {
        let values: Vec<u32> = (0..1000).map(|i| (i * 73 + 19) % 8191).collect();
        let maximum = *values.iter().max().unwrap();
        for &cfg in &BEST_CONFIGS {
            if cfg.split_exponent() > 8 {
                continue;
            }
            if let Some(cost) = source_candidate_cost(&values, maximum, cfg, 256, 256).unwrap() {
                let source_fields = (u32::BITS - cfg.split_exponent().leading_zeros())
                    + (u32::BITS - (cfg.split_exponent() - cfg.msb_in_token()).leading_zeros());
                let expected = independent_cost(&values, cfg) - signaling_bits(cfg).unwrap() as f64
                    + f64::from(source_fields);
                assert!((f64::from(cost) - expected).abs() < 0.03);
            }
        }
    }

    #[test]
    fn malformed_maps_and_lengths_are_rejected() {
        assert!(choose_configs(&[], &[], 0, false).is_err());
        assert!(choose_configs(&[], &[], 257, false).is_err());
        assert!(choose_configs(&[], &[2], 2, false).is_err());
        assert!(choose_configs(&[], &[0], 1, true).is_err());
        assert!(choose_configs(&[vec![literal(usize::MAX, 1)]], &[0], 1, false).is_err());
        for (context, value, lz77) in [(0, 3, false), (1, 3, true), (0, 1 << 20, true)] {
            let token = Token {
                context,
                value,
                is_length: true,
            };
            assert!(choose_configs(&[vec![token]], &[0, 1], 2, lz77).is_err());
        }
    }
}
