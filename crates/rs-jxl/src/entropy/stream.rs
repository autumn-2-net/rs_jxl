// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! General entropy stream wiring translated from enc_ans.cc::EncodeHistograms,
//! WriteTokens, EncodeUintConfig and enc_context_map.cc. Model selection is
//! separated from emission; clustering and hybrid-uint search are explicit tools.
use std::ops::Range;

use super::ans::{AnsCode, AnsToken, write_mapped_tokens};
#[cfg(test)]
use super::config_search::choose_config_plan;
use super::config_search::choose_config_plan_and_histograms;
use super::options::{ClusterStrategy, EntropyOptions, PREFIX_LOG_ALPHA_SIZE, UintConfigStrategy};
use super::{hybrid_uint::HybridUintConfig, prefix_code::PrefixCode};
use crate::{Error, Result, bit_writer::BitWriter};

/// The global MA tree's maximum leaf count plus an optional LZ77 distance
/// context. Raw contexts are not the at-most-256 clustered histogram IDs.
pub const MAX_RAW_CONTEXTS: usize = (1 << 21) + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub context: usize,
    pub value: u32,
    pub is_length: bool,
}
impl Token {
    pub fn literal(context: usize, value: u32) -> Self {
        Self {
            context,
            value,
            is_length: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coding {
    Prefix,
    Ans,
    /// Compare separately optimized prefix/ANS models using their header and
    /// data costs. This is an encoder extension, not the source early heuristic.
    Auto,
    /// Source BuildAndEncodeHistograms: prefix for <100 tokens, Fastest
    /// clustering, or all-singleton raw histograms; otherwise ANS.
    Source,
}

enum Tables {
    Prefix(Vec<PrefixCode>),
    Ans(Vec<AnsCode>),
}

struct StreamingState {
    config: HybridUintConfig,
    context_ranges: Vec<Range<usize>>,
    encoded_histograms: Vec<BitWriter>,
}

pub struct Model {
    map: Vec<u8>,
    configs: Vec<HybridUintConfig>,
    tables: Tables,
    lz77: bool,
    length_config: HybridUintConfig,
    min_symbol: u32,
    log_alpha_size: u8,
    streaming: Option<StreamingState>,
}

fn ceil_log2(value: usize) -> u8 {
    if value <= 1 {
        0
    } else {
        (usize::BITS - (value - 1).leading_zeros()) as u8
    }
}

pub(crate) fn write_uint_config(
    w: &mut BitWriter,
    cfg: HybridUintConfig,
    log_alpha: u8,
) -> Result<()> {
    if cfg.split_exponent() > u32::from(log_alpha) {
        return Err(Error::Encoding(
            "hybrid configuration exceeds alphabet".into(),
        ));
    }
    w.write(
        ceil_log2(usize::from(log_alpha) + 1),
        u64::from(cfg.split_exponent()),
    )?;
    if cfg.split_exponent() == u32::from(log_alpha) {
        return Ok(());
    }
    w.write(
        ceil_log2(cfg.split_exponent() as usize + 1),
        u64::from(cfg.msb_in_token()),
    )?;
    w.write(
        ceil_log2((cfg.split_exponent() - cfg.msb_in_token()) as usize + 1),
        u64::from(cfg.lsb_in_token()),
    )?;
    Ok(())
}

fn var_len16(w: &mut BitWriter, value: usize) -> Result<()> {
    if value > 65535 {
        return Err(Error::Encoding("entropy alphabet too large".into()));
    }
    w.write(1, u64::from(value != 0))?;
    if value != 0 {
        let n = (usize::BITS - 1 - value.leading_zeros()) as u8;
        w.write(4, u64::from(n))?;
        w.write(n, (value - (1usize << n)) as u64)?;
    }
    Ok(())
}

fn add_count(counts: &mut Vec<u64>, symbol: usize) -> Result<()> {
    if counts.len() <= symbol {
        counts
            .try_reserve(symbol + 1 - counts.len())
            .map_err(|_| Error::Limit("histogram allocation"))?;
        counts.resize(symbol + 1, 0);
    }
    counts[symbol] = counts[symbol]
        .checked_add(1)
        .ok_or(Error::Limit("histogram count"))?;
    Ok(())
}

fn finish_counts(counts: &mut [Vec<u64>]) {
    for histogram in counts {
        if histogram.is_empty() {
            histogram.push(1);
        }
        while histogram.len() > 1 && histogram.last() == Some(&0) {
            histogram.pop();
        }
        if histogram.len() == 1 && histogram[0] == 0 {
            // An adaptive empty winner has an explicit zero bin, unlike the
            // empty Vec produced by a fresh count pass. Both code as symbol 0.
            histogram[0] = 1;
        }
    }
}

fn source_all_singleton(counts: &[Vec<u64>]) -> Result<bool> {
    // BuildAndEncodeHistograms asks ShannonEntropy, not a symbol-count proxy.
    for histogram in counts {
        let (total, _) = super::simd::condition(histogram)
            .ok_or(Error::Limit("cluster histogram weight exceeds i32::MAX"))?;
        if super::simd::entropy(histogram, total) >= 1e-5 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn initial_config(options: &EntropyOptions) -> HybridUintConfig {
    match options.uint_config {
        UintConfigStrategy::Fixed(config) => config,
        UintConfigStrategy::Fast | UintConfigStrategy::Best => HybridUintConfig::default(),
    }
}

fn raw_histograms(
    streams: &[Vec<Token>],
    contexts: usize,
    lz77: bool,
    options: &EntropyOptions,
) -> Result<(Vec<Vec<u64>>, usize)> {
    options.validate()?;
    if contexts == 0 || contexts > MAX_RAW_CONTEXTS {
        return Err(Error::InvalidInput(format!(
            "entropy raw contexts must be 1..={MAX_RAW_CONTEXTS}",
        )));
    }
    if lz77 && contexts < 2 {
        return Err(Error::InvalidInput(
            "LZ77 needs literal and distance contexts".into(),
        ));
    }
    let count_profile = crate::profile::scope("entropy.raw_histograms");
    let config = initial_config(options);
    let mut counts = vec![Vec::new(); contexts];
    let mut token_count = 0usize;
    for token in streams.iter().flatten() {
        if token.context >= contexts
            || (token.is_length
                && (!lz77 || token.context + 1 == contexts || token.value > u32::MAX - 3))
        {
            return Err(Error::InvalidInput(
                "invalid entropy token context/length".into(),
            ));
        }
        let cfg = if token.is_length {
            options.lz77_length_config
        } else {
            config
        };
        let symbol = cfg.encode(token.value).token as usize
            + if token.is_length {
                options.lz77_min_symbol as usize
            } else {
                0
            };
        if symbol >= (1 << PREFIX_LOG_ALPHA_SIZE)
            || (lz77
                && !token.is_length
                && (options.streaming_mode
                    || matches!(options.uint_config, UintConfigStrategy::Fixed(_)))
                && symbol >= options.lz77_min_symbol as usize)
        {
            return Err(Error::Encoding(
                "entropy symbol exceeds active alphabet".into(),
            ));
        }
        add_count(&mut counts[token.context], symbol)?;
        token_count = token_count
            .checked_add(1)
            .ok_or(Error::Limit("entropy token count"))?;
    }
    if options.add_missing_symbols {
        // enc_ans.cc adds one to every bin, not just those absent in this batch.
        for histogram in &mut counts {
            histogram.resize(histogram.len().max(256), 0);
            for count in &mut histogram[..256] {
                *count = count
                    .checked_add(1)
                    .ok_or(Error::Limit("histogram count"))?;
            }
        }
    }
    if crate::profile::enabled() {
        crate::profile::note(
            "entropy.raw_histograms",
            format_args!(
                "nonempty={} bins={} injected={}",
                counts.iter().filter(|h| !h.is_empty()).count(),
                counts.iter().map(Vec::len).sum::<usize>(),
                options.add_missing_symbols,
            ),
        );
    }
    drop(count_profile);
    Ok((counts, token_count))
}

fn build_tables(
    counts: &[Vec<u64>],
    prefix: bool,
    log_alpha: u8,
    options: &EntropyOptions,
) -> Result<Tables> {
    if prefix {
        Ok(Tables::Prefix(
            counts
                .iter()
                .map(|h| PrefixCode::from_counts(h).map_err(|e| Error::Encoding(e.to_string())))
                .collect::<Result<Vec<_>>>()?,
        ))
    } else {
        Ok(Tables::Ans(
            counts
                .iter()
                .map(|h| AnsCode::from_counts_with_strategy(h, log_alpha, options.ans_histogram))
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

fn cache_histograms(tables: &Tables) -> Result<Vec<BitWriter>> {
    let count = match tables {
        Tables::Ans(t) => t.len(),
        Tables::Prefix(t) => t.len(),
    };
    (0..count)
        .map(|i| {
            let alphabet = match tables {
                Tables::Ans(t) => t[i].normalized_counts().len(),
                Tables::Prefix(t) => t[i].alphabet_size(),
            };
            // Source BuildAndStoreEntropyCodes histogram allotment, in bits.
            let mut writer = BitWriter::new((256 + alphabet * 24).div_ceil(8));
            match tables {
                Tables::Ans(t) => t[i].write_histogram(&mut writer)?,
                Tables::Prefix(t) => t[i]
                    .write_histogram(&mut writer)
                    .map_err(|e| Error::Encoding(e.to_string()))?,
            }
            Ok(writer)
        })
        .collect()
}

impl Model {
    /// Retains Best uint search/clustering and Precise ANS normalization.
    pub fn build(
        streams: &[Vec<Token>],
        contexts: usize,
        lz77: bool,
        coding: Coding,
    ) -> Result<Self> {
        Self::build_with_options(streams, contexts, lz77, coding, &EntropyOptions::default())
    }

    /// Build from raw contexts, then cluster down to at most 256 histograms.
    /// With LZ77, the last raw context holds distances. Length configuration
    /// and reserved symbol start come from options and are serialized verbatim.
    pub fn build_with_options(
        streams: &[Vec<Token>],
        contexts: usize,
        lz77: bool,
        coding: Coding,
        options: &EntropyOptions,
    ) -> Result<Self> {
        let _profile = crate::profile::scope("entropy.model_build");
        if crate::profile::enabled() {
            crate::profile::note(
                "entropy.model_build",
                format_args!(
                    "streams={} tokens={} contexts={contexts} lz77={lz77} coding={coding:?} clustering={:?} uint={:?} streaming={}",
                    streams.len(),
                    streams.iter().map(Vec::len).sum::<usize>(),
                    options.clustering,
                    options.uint_config,
                    options.streaming_mode,
                ),
            );
        }
        let (counts, token_count) = raw_histograms(streams, contexts, lz77, options)?;
        if options.max_histograms == 0 && !options.add_fixed_histograms {
            return Err(Error::InvalidInput(
                "initial model needs a histogram budget or a fixed table".into(),
            ));
        }
        // Empty source contexts have zero population, not invented observations.
        // Final completely empty coding tables get the symbol-zero fallback.
        if coding == Coding::Auto {
            let prefix = Self::build_candidate(streams, &counts, lz77, true, options)?;
            let ans = match Self::build_candidate(streams, &counts, lz77, false, options) {
                Ok(model) => model,
                // A wide fixed prefix config/length reservation or a restricted
                // ANS range may have no legal ANS representation.
                Err(Error::InvalidInput(_)) => return Ok(prefix),
                Err(error) => return Err(error),
            };
            return if prefix.estimated_bits(streams)? <= ans.estimated_bits(streams)? {
                Ok(prefix)
            } else {
                Ok(ans)
            };
        }
        let prefix = coding == Coding::Prefix
            || (coding == Coding::Source
                && (token_count < 100
                    || options.clustering == ClusterStrategy::Fastest
                    || source_all_singleton(&counts)?
                    || counts.iter().any(|h| h.len() > 256)
                    || matches!(options.uint_config, UintConfigStrategy::Fixed(c) if c.split_exponent() > 8)));
        Self::build_candidate(streams, &counts, lz77, prefix, options)
    }

    fn build_candidate(
        streams: &[Vec<Token>],
        counts: &[Vec<u64>],
        lz77: bool,
        prefix: bool,
        options: &EntropyOptions,
    ) -> Result<Self> {
        let _profile = crate::profile::scope("entropy.model_candidate");
        crate::profile::note("entropy.model_candidate", format_args!("prefix={prefix}"));
        let (clustered, map) = if options.add_fixed_histograms {
            super::cluster::cluster_with_preexisting(
                counts,
                &[vec![if prefix { 128 } else { 16 }; 256]],
                options.max_histograms,
                options.clustering,
                prefix && options.prefix_singleton_guard,
            )?
        } else if counts.len() == 1 && counts[0].len() <= super::prefix_code::MAX_ALPHABET_SIZE {
            // Source BuildAndStoreEntropyCodes skips clustering one context.
            // Preserve the histogram weight contract without building a center.
            counts[0].iter().try_fold(0u64, |total, &count| {
                total
                    .checked_add(count)
                    .filter(|&n| n <= i32::MAX as u64)
                    .ok_or(Error::Limit("cluster histogram weight exceeds i32::MAX"))
            })?;
            (vec![counts[0].clone()], vec![0])
        } else {
            let (clustered, map) = super::cluster::cluster_with_options(
                counts,
                options.max_histograms,
                options.clustering,
                prefix && options.prefix_singleton_guard,
            )?;
            (clustered, map)
        };
        let num_histograms = clustered.len();
        let search = choose_config_plan_and_histograms(
            streams,
            &map,
            num_histograms,
            lz77,
            prefix,
            options,
        )?;
        let mut plan = search.plan;
        // Adaptive search already counted the winning configuration and lengths.
        // Fixed/streaming policies retain the clustered raw counts, including
        // missing-symbol injection and immutable preexisting fixed centers.
        let count_profile = crate::profile::scope("entropy.final_histograms");
        let reused = search.counts.is_some();
        let mut unique = search.counts.unwrap_or(clustered);
        finish_counts(&mut unique);
        if !prefix {
            let alphabet = unique.iter().map(Vec::len).max().unwrap_or(1);
            plan.log_alpha_size = plan.log_alpha_size.max(ceil_log2(alphabet));
            if plan.log_alpha_size > options.max_log_alpha_size {
                return Err(Error::InvalidInput(
                    "injected histogram exceeds the ANS log-alpha range".into(),
                ));
            }
        }
        crate::profile::note("entropy.final_histograms", format_args!("reused={reused}"));
        drop(count_profile);
        let table_profile = crate::profile::scope("entropy.encoding_tables");
        let tables = build_tables(&unique, prefix, plan.log_alpha_size, options)?;
        let streaming = if options.streaming_mode {
            Some(StreamingState {
                config: initial_config(options),
                context_ranges: std::iter::once(0..counts.len()).collect(),
                encoded_histograms: cache_histograms(&tables)?,
            })
        } else {
            None
        };
        drop(table_profile);
        crate::profile::note(
            "entropy.model_candidate.result",
            format_args!(
                "histograms={} log_alpha={}",
                unique.len(),
                plan.log_alpha_size
            ),
        );
        Ok(Self {
            map,
            configs: plan.configs,
            tables,
            lz77,
            length_config: options.lz77_length_config,
            min_symbol: options.lz77_min_symbol,
            log_alpha_size: plan.log_alpha_size,
            streaming,
        })
    }

    fn estimated_bits(&self, streams: &[Vec<Token>]) -> Result<f64> {
        let alphabet_total: usize = match &self.tables {
            Tables::Prefix(tables) => tables.iter().map(PrefixCode::alphabet_size).sum(),
            Tables::Ans(tables) => tables.iter().map(|t| t.normalized_counts().len()).sum(),
        };
        let mut header = BitWriter::new(alphabet_total * 8 + self.map.len() * 8 + 65536);
        self.write_header(&mut header)?;
        let mut bits = header.bits_written() as f64;
        if let Tables::Prefix(tables) = &self.tables {
            for t in streams.iter().flatten() {
                let t = self.map_token(*t)?;
                bits += f64::from(
                    tables[t.context]
                        .symbol_bits(t.symbol)
                        .map_err(|e| Error::Encoding(e.to_string()))?,
                ) + f64::from(t.nbits);
            }
        }
        if let Tables::Ans(tables) = &self.tables {
            bits += 32.0 * streams.len() as f64;
            for stream in streams {
                for t in stream {
                    let t = self.map_token(*t)?;
                    bits += tables[t.context].symbol_cost(t.symbol)? + f64::from(t.nbits);
                }
            }
        }
        Ok(bits)
    }

    pub fn log_alpha_size(&self) -> u8 {
        self.log_alpha_size
    }

    pub fn uses_ans(&self) -> bool {
        matches!(self.tables, Tables::Ans(_))
    }

    pub fn histogram_count(&self) -> usize {
        self.configs.len()
    }

    pub fn context_count(&self) -> usize {
        self.map.len()
    }

    /// Source initialize_global_state=false: append a new local context range,
    /// retaining previous coding tables, wire mode and HybridUint mapping.
    /// `contexts` includes this batch's distance context when LZ77 is enabled.
    /// `max_histograms` is the budget of NEW tables (zero permits refit only).
    /// Failure leaves the model's logical state unchanged. Emit the combined
    /// header after all appends, then use the returned range's start when
    /// writing this batch. Previously written token payloads stay valid.
    pub fn append_with_options(
        &mut self,
        streams: &[Vec<Token>],
        contexts: usize,
        lz77: bool,
        options: &EntropyOptions,
    ) -> Result<Range<usize>> {
        options.validate()?;
        let state = self.streaming.as_ref().ok_or_else(|| {
            Error::InvalidInput("incremental entropy append requires a streaming model".into())
        })?;
        if !options.streaming_mode || options.add_fixed_histograms {
            return Err(Error::InvalidInput(
                "entropy append requires streaming mode without repeated fixed injection".into(),
            ));
        }
        if initial_config(options) != state.config
            || lz77 != self.lz77
            || options.lz77_length_config != self.length_config
            || options.lz77_min_symbol != self.min_symbol
        {
            return Err(Error::InvalidInput(
                "entropy append cannot change the stored symbol or LZ77 mapping".into(),
            ));
        }
        let start = self.map.len();
        let end = start
            .checked_add(contexts)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or(Error::Limit("incremental entropy context map"))?;
        let (counts, _) = raw_histograms(streams, contexts, lz77, options)?;
        let previous: Vec<Vec<u64>> = match &self.tables {
            Tables::Ans(tables) => tables
                .iter()
                .map(|table| {
                    table
                        .normalized_counts()
                        .iter()
                        .map(|&n| u64::from(n))
                        .collect()
                })
                .collect(),
            Tables::Prefix(tables) => tables
                .iter()
                .map(|table| {
                    (0..table.alphabet_size())
                        .map(|symbol| {
                            match table.symbol_bits(symbol) {
                                Ok(depth) if depth != 0 => 1u64 << (PREFIX_LOG_ALPHA_SIZE - depth),
                                // Source HistogramFromSymbolInfo assigns zero weight
                                // even to a singleton's zero-depth present symbol.
                                _ => 0,
                            }
                        })
                        .collect()
                })
                .collect(),
        };
        let old_count = previous.len();
        let prefix = !self.uses_ans();
        let (mut clustered, map) = super::cluster::cluster_with_preexisting(
            &counts,
            &previous,
            options.max_histograms,
            options.clustering,
            prefix && options.prefix_singleton_guard,
        )?;
        let plan = super::config_search::choose_config_plan(
            streams,
            &map,
            clustered.len(),
            lz77,
            prefix,
            options,
        )?;
        if plan.log_alpha_size != self.log_alpha_size
            || &plan.configs[..old_count] != self.configs.as_slice()
        {
            return Err(Error::InvalidInput(
                "entropy append cannot change existing table configuration".into(),
            ));
        }
        finish_counts(&mut clustered[old_count..]);
        if clustered[old_count..]
            .iter()
            .any(|h| h.len() > (1usize << self.log_alpha_size))
        {
            return Err(Error::InvalidInput(
                "appended histogram exceeds the stored alphabet".into(),
            ));
        }
        let added = build_tables(
            &clustered[old_count..],
            prefix,
            self.log_alpha_size,
            options,
        )?;
        let encoded = cache_histograms(&added)?;
        let additional = plan.configs.len() - old_count;
        // Reserve all destinations before committing any append. Temporary
        // coding tables and bit caches are dropped intact on an earlier error.
        self.map
            .try_reserve(map.len())
            .map_err(|_| Error::Limit("incremental entropy map"))?;
        self.configs
            .try_reserve(additional)
            .map_err(|_| Error::Limit("incremental entropy configs"))?;
        let state = self.streaming.as_mut().expect("validated streaming model");
        state
            .encoded_histograms
            .try_reserve(additional)
            .map_err(|_| Error::Limit("incremental histogram cache"))?;
        state
            .context_ranges
            .try_reserve(1)
            .map_err(|_| Error::Limit("incremental context ranges"))?;
        match (&mut self.tables, added) {
            (Tables::Ans(tables), Tables::Ans(added)) => {
                tables
                    .try_reserve(additional)
                    .map_err(|_| Error::Limit("incremental ANS tables"))?;
                tables.extend(added);
            }
            (Tables::Prefix(tables), Tables::Prefix(added)) => {
                tables
                    .try_reserve(additional)
                    .map_err(|_| Error::Limit("incremental prefix tables"))?;
                tables.extend(added);
            }
            _ => unreachable!("coding mode is immutable"),
        }
        state.encoded_histograms.extend(encoded);
        state.context_ranges.push(start..end);
        self.configs.extend_from_slice(&plan.configs[old_count..]);
        self.map.extend(map);
        Ok(start..end)
    }

    fn compact_map(&self) -> (Vec<u8>, Vec<usize>) {
        let mut remap = [usize::MAX; 256];
        let mut order = Vec::new();
        let map = self
            .map
            .iter()
            .map(|&id| {
                let old = usize::from(id);
                if remap[old] == usize::MAX {
                    remap[old] = order.len();
                    order.push(old);
                }
                remap[old] as u8
            })
            .collect();
        (map, order)
    }

    /// Source enc_frame.cc::RemoveUnusedHistograms. Context offsets and token
    /// payloads remain valid; IDs and stored tables are reordered together.
    /// Do this after the last append if unused fixed tables should remain
    /// available as coding centers during subsequent batches.
    pub fn remove_unused_histograms(&mut self) -> usize {
        fn reorder<T>(values: &mut Vec<T>, order: &[usize]) {
            let mut old: Vec<_> = std::mem::take(values).into_iter().map(Some).collect();
            *values = order
                .iter()
                .map(|&i| old[i].take().expect("unique histogram ID"))
                .collect();
        }
        let (map, order) = self.compact_map();
        let removed = self.configs.len() - order.len();
        reorder(&mut self.configs, &order);
        match &mut self.tables {
            Tables::Ans(tables) => reorder(tables, &order),
            Tables::Prefix(tables) => reorder(tables, &order),
        }
        if let Some(state) = &mut self.streaming {
            reorder(&mut state.encoded_histograms, &order);
        }
        self.map = map;
        removed
    }

    /// Serialize the complete current model, including all appended contexts.
    /// Streaming histograms are copied from their cached exact bit lengths.
    /// Unused fixed centers are omitted from the wire view without mutating
    /// the model; explicit compaction can reclaim them after the last append.
    pub fn write_header(&self, w: &mut BitWriter) -> Result<()> {
        w.write(1, u64::from(self.lz77))?;
        if self.lz77 {
            match self.min_symbol {
                224 => w.write(2, 0)?,
                512 => w.write(2, 1)?,
                4096 => w.write(2, 2)?,
                value => {
                    w.write(2, 3)?;
                    w.write(15, u64::from(value - 8))?;
                }
            }
            w.write(2, 0)?; // min_length=3.
            write_uint_config(w, self.length_config, 8)?;
        }
        let (map, order) = if self.streaming.is_some() {
            let (map, order) = self.compact_map();
            (std::borrow::Cow::Owned(map), order)
        } else {
            (
                std::borrow::Cow::Borrowed(self.map.as_slice()),
                (0..self.configs.len()).collect(),
            )
        };
        if map.len() > 1 {
            super::context_map::write_context_map(&map, order.len(), w)?;
        }
        let prefix = !self.uses_ans();
        w.write(1, u64::from(prefix))?;
        let log_alpha = self.log_alpha_size;
        if !prefix {
            w.write(2, u64::from(log_alpha - 5))?;
        }
        for &i in &order {
            write_uint_config(w, self.configs[i], log_alpha)?;
        }
        if let Tables::Prefix(tables) = &self.tables {
            for &i in &order {
                var_len16(w, tables[i].alphabet_size() - 1)?;
            }
        }
        for &i in &order {
            if let Some(state) = &self.streaming {
                w.append_unaligned(&state.encoded_histograms[i])?;
            } else {
                match &self.tables {
                    Tables::Prefix(tables) => tables[i]
                        .write_histogram(w)
                        .map_err(|e| Error::Encoding(e.to_string()))?,
                    Tables::Ans(tables) => tables[i].write_histogram(w)?,
                }
            }
        }
        Ok(())
    }

    fn context_range(&self, offset: usize) -> Result<Range<usize>> {
        if let Some(state) = &self.streaming {
            state
                .context_ranges
                .iter()
                .find(|range| range.start == offset)
                .cloned()
                .ok_or_else(|| {
                    Error::Encoding("context offset is not a stored batch boundary".into())
                })
        } else if offset == 0 {
            Ok(0..self.map.len())
        } else {
            Err(Error::Encoding(
                "context offset requires an appended model".into(),
            ))
        }
    }

    fn map_token(&self, t: Token) -> Result<AnsToken> {
        self.map_token_in(t, &self.context_range(0)?)
    }

    fn map_token_in(&self, t: Token, range: &Range<usize>) -> Result<AnsToken> {
        if t.context >= range.len() {
            return Err(Error::Encoding("token context not in model batch".into()));
        }
        let context = usize::from(
            *self
                .map
                .get(range.start + t.context)
                .ok_or_else(|| Error::Encoding("token context not in model".into()))?,
        );
        if t.is_length && (!self.lz77 || t.context + 1 == range.len() || t.value > u32::MAX - 3) {
            return Err(Error::Encoding("invalid length token context".into()));
        }
        let cfg = if t.is_length {
            self.length_config
        } else {
            self.configs[context]
        };
        let encoded = cfg.encode(t.value);
        let symbol = encoded.token as usize
            + if t.is_length {
                self.min_symbol as usize
            } else {
                0
            };
        if symbol >= (1 << self.log_alpha_size)
            || (self.lz77 && !t.is_length && symbol >= self.min_symbol as usize)
        {
            return Err(Error::Encoding(
                "token exceeds active entropy alphabet".into(),
            ));
        }
        Ok(AnsToken {
            context,
            symbol,
            nbits: encoded.nbits as u8,
            bits: encoded.bits,
        })
    }

    pub fn write_stream(&self, tokens: &[Token], w: &mut BitWriter) -> Result<()> {
        self.write_stream_at(tokens, 0, w)
    }

    /// Tokens carry batch-local context IDs. `context_offset` is zero for the
    /// initial batch, or the start of a range returned by append_with_options.
    pub fn write_stream_at(
        &self,
        tokens: &[Token],
        context_offset: usize,
        w: &mut BitWriter,
    ) -> Result<()> {
        let _profile = crate::profile::scope("entropy.write_tokens");
        let range = self.context_range(context_offset)?;
        match &self.tables {
            Tables::Prefix(tables) => {
                for t in tokens {
                    let t = self.map_token_in(*t, &range)?;
                    tables[t.context]
                        .write_symbol_with_extra(t.symbol, t.nbits, t.bits, w)
                        .map_err(|e| Error::Encoding(e.to_string()))?;
                }
            }
            Tables::Ans(tables) => {
                if !self.lz77 && self.map.as_slice() == [0] {
                    let config = self.configs[0];
                    let alphabet = 1usize << self.log_alpha_size;
                    write_mapped_tokens(
                        tables,
                        tokens.len(),
                        |index| {
                            let token = tokens[index];
                            if token.context != 0 || token.is_length {
                                return Err(Error::Encoding(
                                    "invalid single-context literal token".into(),
                                ));
                            }
                            let encoded = config.encode(token.value);
                            if encoded.token as usize >= alphabet {
                                return Err(Error::Encoding(
                                    "token exceeds active entropy alphabet".into(),
                                ));
                            }
                            Ok(AnsToken {
                                context: 0,
                                symbol: encoded.token as usize,
                                nbits: encoded.nbits as u8,
                                bits: encoded.bits,
                            })
                        },
                        w,
                    )?;
                } else {
                    write_mapped_tokens(
                        tables,
                        tokens.len(),
                        |index| self.map_token_in(tokens[index], &range),
                        w,
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod streaming_tests;

#[cfg(test)]
mod tests {
    use super::super::options::{AnsHistogramStrategy, ClusterStrategy};
    use super::*;

    #[test]
    fn source_singleton_entropy_and_empty_winner_fallback() {
        assert!(source_all_singleton(&[vec![], vec![0, 19], vec![1 << 25]]).unwrap());
        assert!(!source_all_singleton(&[vec![0, 19], vec![(1 << 25) - 1, 1]]).unwrap());
        assert!(source_all_singleton(&[vec![i32::MAX as u64, 1]]).is_err());
        let mut counts = vec![vec![], vec![0], vec![0; 9], vec![0, 2, 0]];
        finish_counts(&mut counts);
        assert_eq!(counts, [vec![1], vec![1], vec![1], vec![0, 2]]);
        for histogram in counts {
            assert!(PrefixCode::from_counts(&histogram).is_ok());
        }
    }

    #[test]
    fn single_context_bypass_preserves_empty_models_and_weight_limits() {
        for coding in [Coding::Prefix, Coding::Ans, Coding::Source] {
            let model = Model::build(&[vec![]], 1, false, coding).unwrap();
            assert_eq!(model.map, [0]);
            assert_eq!(model.configs.len(), 1);
            encoded(&model, &[vec![]]);
        }
        for histogram in [vec![i32::MAX as u64 + 1], vec![u64::MAX, 1]] {
            assert!(matches!(
                Model::build_candidate(&[], &[histogram], false, false, &EntropyOptions::default()),
                Err(Error::Limit(_))
            ));
        }
    }

    #[test]
    fn direct_stream_mapping_matches_expanded_wire_for_all_routes() {
        for contexts in [1, 3] {
            for lz77 in [false, true] {
                if contexts == 1 && lz77 {
                    continue;
                }
                let tokens: Vec<_> = (0..4097)
                    .map(|i| {
                        if lz77 && i % 7 == 0 {
                            Token {
                                context: 0,
                                value: (i % 100) as u32,
                                is_length: true,
                            }
                        } else {
                            Token::literal(i % contexts, ((i * 31) % 4096) as u32)
                        }
                    })
                    .collect();
                for coding in [Coding::Prefix, Coding::Ans] {
                    let model = Model::build(std::slice::from_ref(&tokens), contexts, lz77, coding)
                        .unwrap();
                    let expanded: Vec<_> = tokens
                        .iter()
                        .map(|&t| model.map_token(t).unwrap())
                        .collect();
                    for prefix in [0, 1, 7] {
                        let mut actual = BitWriter::new(1 << 20);
                        let mut expected = BitWriter::new(1 << 20);
                        actual.write(prefix, (1 << prefix) - 1).unwrap();
                        expected.write(prefix, (1 << prefix) - 1).unwrap();
                        model.write_stream(&tokens, &mut actual).unwrap();
                        match &model.tables {
                            Tables::Ans(tables) => {
                                super::super::ans::write_tokens(tables, &expanded, &mut expected)
                                    .unwrap()
                            }
                            Tables::Prefix(tables) => {
                                for t in &expanded {
                                    tables[t.context]
                                        .write_symbol(t.symbol, &mut expected)
                                        .unwrap();
                                    expected.write(t.nbits, u64::from(t.bits)).unwrap();
                                }
                            }
                        }
                        assert_eq!(actual.bits_written(), expected.bits_written());
                        actual.zero_pad_to_byte().unwrap();
                        expected.zero_pad_to_byte().unwrap();
                        assert_eq!(actual.into_bytes().unwrap(), expected.into_bytes().unwrap());
                    }
                }
            }
        }
    }

    #[test]
    fn mapped_ans_errors_do_not_modify_destination_after_valid_reverse_prefix() {
        for contexts in [1, 2] {
            let tokens = vec![Token::literal(0, 0), Token::literal(contexts - 1, 1)];
            let model =
                Model::build(std::slice::from_ref(&tokens), contexts, false, Coding::Ans).unwrap();
            for invalid in [
                Token::literal(contexts, 0),
                Token {
                    context: 0,
                    value: 0,
                    is_length: true,
                },
            ] {
                let mut writer = BitWriter::new(1 << 20);
                writer.write(5, 19).unwrap();
                assert!(
                    model
                        .write_stream(&[invalid, tokens[1]], &mut writer)
                        .is_err()
                );
                assert_eq!(writer.bits_written(), 5);
                writer.zero_pad_to_byte().unwrap();
                assert_eq!(writer.into_bytes().unwrap(), [19]);
            }
        }
    }

    fn encoded(model: &Model, streams: &[Vec<Token>]) -> (usize, Vec<u8>) {
        let mut writer = BitWriter::new(1 << 20);
        model.write_header(&mut writer).unwrap();
        for stream in streams {
            model.write_stream(stream, &mut writer).unwrap();
        }
        let bits = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        (bits, writer.into_bytes().unwrap())
    }

    fn read_bits(bytes: &[u8], position: &mut usize, width: u8) -> u32 {
        let mut value = 0;
        for bit in 0..width {
            value |= u32::from((bytes[*position / 8] >> (*position % 8)) & 1) << bit;
            *position += 1;
        }
        value
    }

    fn read_config(bytes: &[u8], position: &mut usize, log: u8) -> HybridUintConfig {
        let split = read_bits(bytes, position, (u8::BITS - log.leading_zeros()) as u8);
        if split == u32::from(log) {
            return HybridUintConfig::new(split, 0, 0).unwrap();
        }
        let msb = read_bits(bytes, position, (u32::BITS - split.leading_zeros()) as u8);
        let lsb = read_bits(
            bytes,
            position,
            (u32::BITS - (split - msb).leading_zeros()) as u8,
        );
        HybridUintConfig::new(split, msb, lsb).unwrap()
    }

    #[test]
    fn ans_alphabet_range_is_signaled_and_used_by_tables() {
        for log in 5..=8 {
            let streams = [(0..1u32 << log).map(|v| Token::literal(0, v)).collect()];
            let config = HybridUintConfig::new(log.into(), 0, 0).unwrap();
            let options = EntropyOptions {
                uint_config: UintConfigStrategy::Fixed(config),
                min_log_alpha_size: log,
                max_log_alpha_size: log,
                ..EntropyOptions::default()
            };
            let model =
                Model::build_with_options(&streams, 1, false, Coding::Ans, &options).unwrap();
            assert_eq!(model.log_alpha_size(), log);
            let Tables::Ans(tables) = &model.tables else {
                panic!("expected ANS");
            };
            assert_eq!(tables[0].log_alpha_size(), log);
            let (_, bytes) = encoded(&model, &streams);
            let mut position = 0;
            assert_eq!(read_bits(&bytes, &mut position, 1), 0);
            assert_eq!(read_bits(&bytes, &mut position, 1), 0);
            assert_eq!(read_bits(&bytes, &mut position, 2), u32::from(log - 5));
            assert_eq!(read_config(&bytes, &mut position, log), config);
        }
        let model = Model::build(&[vec![Token::literal(0, 0)]], 1, false, Coding::Ans).unwrap();
        assert_eq!(model.log_alpha_size(), 5);
    }

    #[test]
    fn lz77_header_presets_offset_and_length_configs_round_trip() {
        for (minimum, selector) in [(224, 0), (512, 1), (4096, 2), (8, 3), (300, 3), (32775, 3)] {
            for split in [0, 4] {
                let length_config = HybridUintConfig::new(split, 0, 0).unwrap();
                let options = EntropyOptions {
                    uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap()),
                    lz77_length_config: length_config,
                    lz77_min_symbol: minimum,
                    max_histograms: 1,
                    ..EntropyOptions::default()
                };
                let mut tokens = vec![Token::literal(0, 0), Token::literal(1, 0)];
                if minimum < 32768 {
                    tokens.insert(
                        1,
                        Token {
                            context: 0,
                            value: 16,
                            is_length: true,
                        },
                    );
                }
                let streams = [tokens];
                let model =
                    Model::build_with_options(&streams, 2, true, Coding::Prefix, &options).unwrap();
                assert_eq!(model.length_config, length_config);
                assert_eq!(model.min_symbol, minimum);
                let (_, bytes) = encoded(&model, &streams);
                let mut position = 0;
                assert_eq!(read_bits(&bytes, &mut position, 1), 1);
                assert_eq!(read_bits(&bytes, &mut position, 2), selector);
                if selector == 3 {
                    assert_eq!(read_bits(&bytes, &mut position, 15) + 8, minimum);
                }
                assert_eq!(read_bits(&bytes, &mut position, 2), 0); // min_length=3
                assert_eq!(read_config(&bytes, &mut position, 8), length_config);
                if minimum < 32768 {
                    let actual = model.map_token(streams[0][1]).unwrap();
                    let expected = length_config.encode(16);
                    assert_eq!(actual.symbol as u32, minimum + expected.token);
                    assert_eq!(u32::from(actual.nbits), expected.nbits);
                    assert_eq!(actual.bits, expected.bits);
                }
            }
        }
    }

    #[test]
    fn wide_prefix_auto_and_lz77_reservations() {
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(15, 0, 0).unwrap()),
            max_histograms: 1,
            ..EntropyOptions::default()
        };
        let streams = [vec![Token::literal(0, 256), Token::literal(0, 32767)]];
        for coding in [Coding::Prefix, Coding::Auto, Coding::Source] {
            let model = Model::build_with_options(&streams, 1, false, coding, &options).unwrap();
            assert!(!model.uses_ans());
            assert_eq!(model.map_token(streams[0][1]).unwrap().symbol, 32767);
            encoded(&model, &streams);
        }
        assert!(Model::build_with_options(&streams, 1, false, Coding::Ans, &options).is_err());

        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(9, 0, 0).unwrap()),
            lz77_length_config: HybridUintConfig::new(0, 0, 0).unwrap(),
            lz77_min_symbol: 512,
            max_histograms: 1,
            ..EntropyOptions::default()
        };
        let streams = [vec![
            Token::literal(0, 511),
            Token {
                context: 0,
                value: u32::MAX - 3,
                is_length: true,
            },
            Token::literal(1, 0),
        ]];
        for coding in [Coding::Prefix, Coding::Auto] {
            let model = Model::build_with_options(&streams, 2, true, coding, &options).unwrap();
            assert_eq!(model.map_token(streams[0][1]).unwrap().symbol, 544);
            encoded(&model, &streams);
            assert!(model.map_token(Token::literal(0, 512)).is_err());
        }
        assert!(Model::build_with_options(&streams, 2, true, Coding::Ans, &options).is_err());
    }

    #[test]
    fn source_coding_heuristic_and_streaming_policy() {
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap()),
            clustering: ClusterStrategy::Fast,
            ..EntropyOptions::default()
        };
        for (count, distinct, ans) in [(99, true, false), (100, true, true), (100, false, false)] {
            let streams = [(0..count)
                .map(|i| Token::literal(0, if distinct { i % 2 } else { 0 }))
                .collect()];
            let model =
                Model::build_with_options(&streams, 1, false, Coding::Source, &options).unwrap();
            assert_eq!(model.uses_ans(), ans);
        }
        let streams = [(0..100).map(|i| Token::literal(0, i % 2)).collect()];
        let fastest = EntropyOptions {
            clustering: ClusterStrategy::Fastest,
            ..options
        };
        assert!(
            !Model::build_with_options(&streams, 1, false, Coding::Source, &fastest)
                .unwrap()
                .uses_ans()
        );
        let streaming = EntropyOptions {
            streaming_mode: true,
            ..EntropyOptions::default()
        };
        let model = Model::build_with_options(&streams, 1, false, Coding::Ans, &streaming).unwrap();
        assert_eq!(model.log_alpha_size(), 8);
        assert!(
            model
                .configs
                .iter()
                .all(|&c| c == HybridUintConfig::default())
        );
    }

    #[test]
    fn default_options_preserve_model_build() {
        let streams = vec![vec![
            Token::literal(0, 0),
            Token::literal(1, 213),
            Token::literal(0, 17),
            Token::literal(1, 19),
            Token::literal(2, u32::MAX),
        ]];
        for coding in [Coding::Prefix, Coding::Ans, Coding::Auto] {
            let default = Model::build(&streams, 3, false, coding).unwrap();
            let explicit =
                Model::build_with_options(&streams, 3, false, coding, &EntropyOptions::default())
                    .unwrap();
            assert_eq!(default.map, explicit.map);
            assert_eq!(default.configs, explicit.configs);
            assert_eq!(encoded(&default, &streams), encoded(&explicit, &streams));
        }
    }

    #[test]
    fn fixed_fast_and_best_reach_actual_tables() {
        let streams = vec![
            (0..2048)
                .map(|i| Token::literal(i % 3, [i as u32 % 5, (i as u32 * 701) & 65535, 0][i % 3]))
                .collect(),
        ];
        for strategy in [
            UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap()),
            UintConfigStrategy::Fixed(HybridUintConfig::new(2, 0, 1).unwrap()),
            UintConfigStrategy::Fixed(HybridUintConfig::default()),
            UintConfigStrategy::Fast,
            UintConfigStrategy::Best,
        ] {
            let options = EntropyOptions {
                uint_config: strategy,
                ..EntropyOptions::default()
            };
            let model =
                Model::build_with_options(&streams, 3, false, Coding::Ans, &options).unwrap();
            let expected = choose_config_plan(
                &streams,
                &model.map,
                model.configs.len(),
                false,
                false,
                &options,
            )
            .unwrap();
            assert_eq!(model.configs, expected.configs);
            assert_eq!(model.log_alpha_size, expected.log_alpha_size);
            assert!(encoded(&model, &streams).0 > 32);
        }
    }

    #[test]
    fn every_ans_strategy_controls_normalization_and_histogram_bits() {
        let counts: Vec<u64> = (0..64)
            .map(|i| ((i * 701 + 37) % 1013 + 1) as u64)
            .collect();
        let tokens: Vec<Token> = counts
            .iter()
            .enumerate()
            .flat_map(|(symbol, &count)| {
                std::iter::repeat_n(Token::literal(0, symbol as u32), count as usize)
            })
            .collect();
        let streams = [tokens];
        let mut representations = Vec::new();
        for strategy in [
            AnsHistogramStrategy::Fast,
            AnsHistogramStrategy::Approximate,
            AnsHistogramStrategy::Precise,
            AnsHistogramStrategy::Direct0,
            AnsHistogramStrategy::Direct6,
        ] {
            let options = EntropyOptions {
                uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(8, 0, 0).unwrap()),
                ans_histogram: strategy,
                ..EntropyOptions::default()
            };
            let model =
                Model::build_with_options(&streams, 1, false, Coding::Ans, &options).unwrap();
            let Tables::Ans(tables) = &model.tables else {
                panic!("expected ANS tables")
            };
            let expected = AnsCode::from_counts_with_strategy(&counts, 8, strategy).unwrap();
            assert_eq!(tables[0].normalized_counts(), expected.normalized_counts());
            let mut actual_bits = BitWriter::new(8192);
            let mut expected_bits = BitWriter::new(8192);
            tables[0].write_histogram(&mut actual_bits).unwrap();
            expected.write_histogram(&mut expected_bits).unwrap();
            assert_eq!(actual_bits.bits_written(), expected_bits.bits_written());
            actual_bits.zero_pad_to_byte().unwrap();
            expected_bits.zero_pad_to_byte().unwrap();
            let bytes = actual_bits.into_bytes().unwrap();
            assert_eq!(bytes, expected_bits.into_bytes().unwrap());
            representations.push(bytes);
        }
        assert!(representations.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn clustering_strategy_and_histogram_cap_are_not_ignored() {
        let streams = [(0..12)
            .flat_map(|context| std::iter::repeat_n(Token::literal(context, context as u32), 100))
            .collect()];
        for (strategy, maximum, expected) in [
            (ClusterStrategy::Fast, 256, 12),
            (ClusterStrategy::Best, 256, 12),
            (ClusterStrategy::Fastest, 256, 4),
            (ClusterStrategy::Fast, 3, 3),
            (ClusterStrategy::Fastest, 2, 2),
        ] {
            let options = EntropyOptions {
                uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(8, 0, 0).unwrap()),
                clustering: strategy,
                max_histograms: maximum,
                ..EntropyOptions::default()
            };
            let model =
                Model::build_with_options(&streams, 12, false, Coding::Ans, &options).unwrap();
            assert_eq!(
                model.configs.len(),
                expected,
                "{strategy:?}, maximum {maximum}"
            );
            assert!(model.map.iter().all(|&id| usize::from(id) < expected));
            encoded(&model, &streams);
        }
    }

    #[test]
    fn sparse_bins_and_large_raw_context_maps() {
        const CONTEXTS: usize = 32_769;
        let mut sparse = vec![Vec::new(); CONTEXTS];
        add_count(&mut sparse[CONTEXTS - 1], 7).unwrap();
        assert_eq!(sparse.iter().map(Vec::len).sum::<usize>(), 8);
        finish_counts(&mut sparse);
        assert_eq!(sparse.iter().map(Vec::len).sum::<usize>(), CONTEXTS + 7);

        let streams = [vec![Token::literal(0, 1), Token::literal(CONTEXTS - 1, 17)]];
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(HybridUintConfig::default()),
            clustering: ClusterStrategy::Fast,
            max_histograms: 4,
            ..EntropyOptions::default()
        };
        let model =
            Model::build_with_options(&streams, CONTEXTS, false, Coding::Ans, &options).unwrap();
        assert_eq!(model.map.len(), CONTEXTS);
        assert!(model.configs.len() <= 4);
        assert_eq!(
            model.map_token(streams[0][1]).unwrap().context,
            usize::from(model.map[CONTEXTS - 1])
        );
        encoded(&model, &streams);

        // Exercise all 256 possible IDs in a long map, independently of the
        // clustering outcome; the nested models must still have one context.
        let map: Vec<u8> = (0..=255).cycle().take(CONTEXTS).collect();
        let mut writer = BitWriter::new(1 << 20);
        super::super::context_map::write_context_map(&map, 256, &mut writer).unwrap();
        assert!(writer.bits_written() > 3);
    }

    #[test]
    fn long_single_histogram_map_uses_three_bits() {
        const CONTEXTS: usize = 32_769;
        let options = EntropyOptions {
            max_histograms: 1,
            ..EntropyOptions::default()
        };
        let model =
            Model::build_with_options(&[], CONTEXTS, false, Coding::Prefix, &options).unwrap();
        assert_eq!(model.map, vec![0; CONTEXTS]);
        let (_, bytes) = encoded(&model, &[]);
        // Independently read LZ77=0, simple context map=1, width=0, prefix=1.
        let first: Vec<u8> = (0..5).map(|bit| (bytes[0] >> bit) & 1).collect();
        assert_eq!(first, [0, 1, 0, 0, 1]);
    }

    #[test]
    fn lz77_fixed_configs_keep_lengths_separate_and_reject_collisions() {
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(8, 0, 0).unwrap()),
            max_histograms: 1,
            ..EntropyOptions::default()
        };
        let streams = [vec![
            Token::literal(0, 223),
            Token {
                context: 0,
                value: 16,
                is_length: true,
            },
            Token::literal(1, 0),
        ]];
        for coding in [Coding::Prefix, Coding::Ans] {
            let model = Model::build_with_options(&streams, 2, true, coding, &options).unwrap();
            let length = model.map_token(streams[0][1]).unwrap();
            assert_eq!((length.symbol, length.nbits, length.bits), (240, 4, 0));
            encoded(&model, &streams);
            for token in [
                Token::literal(0, 224),
                Token::literal(1, 224),
                Token {
                    context: 1,
                    value: 0,
                    is_length: true,
                },
                Token {
                    context: 0,
                    value: u32::MAX,
                    is_length: true,
                },
            ] {
                let mut writer = BitWriter::new(8192);
                assert!(model.write_stream(&[token], &mut writer).is_err());
                assert_eq!(writer.bits_written(), 0);
                assert!(
                    Model::build_with_options(&[vec![token]], 2, true, coding, &options).is_err()
                );
            }
        }
    }

    #[test]
    fn merged_ma_leaf_count_leaves_room_for_lz77_distance() {
        const LEAVES: usize = 32_768;
        assert_eq!(
            MAX_RAW_CONTEXTS,
            crate::modular::ma_train::MAX_TREE_LEAVES + 1
        );
        let streams = [vec![
            Token::literal(LEAVES - 1, 5),
            Token {
                context: LEAVES - 1,
                value: 0,
                is_length: true,
            },
            Token::literal(LEAVES, 0),
        ]];
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(0, 0, 0).unwrap()),
            clustering: ClusterStrategy::Fast,
            max_histograms: 1,
            ..EntropyOptions::default()
        };
        let model = Model::build_with_options(&streams, LEAVES + 1, true, Coding::Prefix, &options)
            .unwrap();
        assert_eq!(model.map.len(), LEAVES + 1);
        assert_eq!(model.configs.len(), 1);
        encoded(&model, &streams);
    }

    #[test]
    fn raw_context_and_option_bounds_are_checked() {
        for contexts in [0, MAX_RAW_CONTEXTS + 1, usize::MAX] {
            assert!(Model::build(&[], contexts, false, Coding::Ans).is_err());
        }
        assert!(Model::build(&[], 1, true, Coding::Ans).is_err());
        assert!(Model::build(&[vec![Token::literal(512, 0)]], 512, false, Coding::Ans).is_err());
        let options = EntropyOptions {
            max_histograms: 257,
            ..EntropyOptions::default()
        };
        assert!(Model::build_with_options(&[], 512, false, Coding::Ans, &options).is_err());
        let options = EntropyOptions {
            uint_config: UintConfigStrategy::Fixed(HybridUintConfig::new(9, 0, 0).unwrap()),
            ..EntropyOptions::default()
        };
        assert!(Model::build_with_options(&[], 512, false, Coding::Ans, &options).is_err());
        let model = Model::build_with_options(&[], 512, false, Coding::Prefix, &options).unwrap();
        assert_eq!(model.log_alpha_size(), 15);
    }
}
