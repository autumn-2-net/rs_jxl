//! Bounded compiled views of a fitted model; sparse u32 contexts/symbols remain valid.

use super::{Costs, HybridUintConfig, Literal, MIN_MATCH, SourceConfig, reserved};
use crate::Result;

const SYMBOLS: usize = 256;
const CONTEXTS: usize = 65536;
const ROWS: usize = 8192;
const MISSING: u16 = u16::MAX;
#[cfg(feature = "experimental-lz77-hotspots")]
const LENGTH_TOKENS: usize = 128;

#[cfg(feature = "experimental-lz77-hotspots")]
struct LengthPrices {
    config: HybridUintConfig,
    min_symbol: u32,
    ends: Vec<u64>,
    bits: Vec<f32>,
}

#[cfg(feature = "experimental-lz77-hotspots")]
impl LengthPrices {
    fn new(
        rows: &[([f32; SYMBOLS], &super::ContextCost)],
        config: &SourceConfig,
    ) -> Result<Option<Self>> {
        let uint = config.length_uint_config;
        let count = uint.encode(u32::MAX).token as usize + 1;
        // This bounds a cache, not supported match lengths or configurations.
        if uint.lsb_in_token() != 0
            || count > LENGTH_TOKENS
            || config.min_symbol().checked_add(count as u32 - 1).is_none()
        {
            return Ok(None);
        }
        let mut ends = reserved(count)?;
        let mut extra = reserved(count)?;
        let mut value = 0u64;
        while value <= u64::from(u32::MAX) {
            let encoded = uint.encode(value as u32);
            debug_assert_eq!(encoded.token as usize, ends.len());
            value = ((value >> encoded.nbits) + 1) << encoded.nbits;
            ends.push(value);
            extra.push(encoded.nbits);
        }
        let mut bits = reserved(rows.len() * count)?;
        for (dense, sparse) in rows {
            let row = Row {
                dense: Some(dense),
                sparse: Some(sparse),
                lengths: None,
            };
            for (token, &nbits) in extra.iter().enumerate() {
                bits.push(nbits as f32 + row.bits(config.min_symbol() + token as u32));
            }
        }
        Ok(Some(Self {
            config: uint,
            min_symbol: config.min_symbol(),
            ends,
            bits,
        }))
    }
}

// Compiled once per fast-seed group, independently of the wider DP price rows.
pub(super) struct Penalties<'a> {
    costs: &'a Costs,
    dense: Vec<f32>,
}

impl<'a> Penalties<'a> {
    pub(super) fn new(costs: &'a Costs) -> Result<Self> {
        let _profile = crate::profile::scope("lz77.source.compile_penalties");
        let size = costs
            .literals
            .range(..CONTEXTS as u32)
            .next_back()
            .map_or(0, |(&context, _)| context as usize + 1);
        let mut dense = reserved(size)?;
        dense.resize(size, 6.0);
        for (&context, row) in costs.literals.range(..CONTEXTS as u32) {
            dense[context as usize] = row.penalty;
        }
        Ok(Self { costs, dense })
    }

    #[inline]
    pub(super) fn get(&self, context: u32) -> f32 {
        self.dense
            .get(context as usize)
            .copied()
            .unwrap_or_else(|| self.costs.penalty(context))
    }
}

pub(super) struct Prices<'a> {
    costs: &'a Costs,
    contexts: Vec<u16>,
    rows: Vec<([f32; SYMBOLS], &'a super::ContextCost)>,
    distance_tokens: [f32; 128],
    small_distances: [f32; 512],
    #[cfg(feature = "experimental-lz77-hotspots")]
    lengths: Option<LengthPrices>,
}

#[derive(Clone, Copy)]
pub(super) struct Row<'a> {
    dense: Option<&'a [f32; SYMBOLS]>,
    sparse: Option<&'a super::ContextCost>,
    #[cfg(feature = "experimental-lz77-hotspots")]
    lengths: Option<(&'a LengthPrices, &'a [f32])>,
}

impl Row<'_> {
    #[inline]
    fn bits(self, symbol: u32) -> f32 {
        if let Some(bits) = self.dense.and_then(|row| row.get(symbol as usize)) {
            *bits
        } else {
            self.sparse
                .and_then(|row| row.bits.get(&symbol))
                .copied()
                .unwrap_or(12.0)
        }
    }

    pub(super) fn length(self, length: usize, config: &SourceConfig) -> f32 {
        let encoded = config
            .length_uint_config
            .encode((length - MIN_MATCH) as u32);
        encoded.nbits as f32 + self.bits(config.min_symbol() + encoded.token)
    }

    pub(super) fn intervals(
        self,
        mut begin: usize,
        end: usize,
        config: &SourceConfig,
        mut visit: impl FnMut(usize, usize, f32),
    ) {
        debug_assert_eq!(config.length_uint_config.lsb_in_token(), 0);
        #[cfg(feature = "experimental-lz77-hotspots")]
        if begin >= end {
            return;
        }
        #[cfg(feature = "experimental-lz77-hotspots")]
        if let Some((table, prices)) = self.lengths {
            debug_assert_eq!(table.config, config.length_uint_config);
            debug_assert_eq!(table.min_symbol, config.min_symbol());
            let first = table.config.encode((begin - MIN_MATCH) as u32).token as usize;
            for (&boundary, &price) in table.ends[first..].iter().zip(&prices[first..]) {
                let stop = MIN_MATCH + boundary.min((end - MIN_MATCH) as u64) as usize;
                visit(begin, stop, price);
                if stop == end {
                    return;
                }
                begin = stop;
            }
            unreachable!("compiled length intervals cover all u32 values");
        }
        while begin < end {
            let value = (begin - MIN_MATCH) as u32;
            let encoded = config.length_uint_config.encode(value);
            let next = ((u64::from(value) >> encoded.nbits) + 1) << encoded.nbits;
            let stop = MIN_MATCH + next.min((end - MIN_MATCH) as u64) as usize;
            let price = encoded.nbits as f32 + self.bits(config.min_symbol() + encoded.token);
            visit(begin, stop, price);
            begin = stop;
        }
    }
}

impl<'a> Prices<'a> {
    #[cfg(not(feature = "experimental-lz77-hotspots"))]
    #[inline]
    pub(super) fn with_lengths(costs: &'a Costs, _config: &SourceConfig) -> Result<Self> {
        Self::new(costs)
    }

    #[cfg(feature = "experimental-lz77-hotspots")]
    pub(super) fn with_lengths(costs: &'a Costs, config: &SourceConfig) -> Result<Self> {
        let mut prices = Self::new(costs)?;
        // Indexed blocks use their own cost view, and non-000 range parsing
        // enumerates explicit endpoints instead of calling Row::intervals.
        let indexed_blocks = config.block_mode != super::SourceBlockMode::Disabled
            && (config.shared_index || config.index_mode != super::SourceIndexMode::Disabled);
        if indexed_blocks
            || (config.match_cost.range()
                && config.length_uint_config != HybridUintConfig::new(0, 0, 0).unwrap())
        {
            return Ok(prices);
        }
        let _profile = crate::profile::scope("lz77.source.compile_lengths");
        prices.lengths = LengthPrices::new(&prices.rows, config)?;
        Ok(prices)
    }

    pub(super) fn new(costs: &'a Costs) -> Result<Self> {
        let _profile = crate::profile::scope("lz77.source.compile_prices");
        let count = costs.literals.range(..CONTEXTS as u32).count().min(ROWS);
        let size = costs
            .literals
            .range(..CONTEXTS as u32)
            .take(ROWS)
            .last()
            .map_or(0, |(&context, _)| context as usize + 1);
        let mut contexts = reserved(size)?;
        contexts.resize(size, MISSING);
        let mut rows = reserved(count)?;
        for (&context, row) in costs.literals.range(..CONTEXTS as u32).take(ROWS) {
            contexts[context as usize] = rows.len() as u16;
            let mut bits = [12.0; SYMBOLS];
            for (&symbol, &cost) in row.bits.range(..SYMBOLS as u32) {
                bits[symbol as usize] = cost;
            }
            rows.push((bits, row));
        }
        let distance_tokens = std::array::from_fn(|token| {
            costs
                .distances
                .bits
                .get(&(token as u32))
                .copied()
                .unwrap_or(12.0)
        });
        let small_distances = std::array::from_fn(|symbol| costs.distance(symbol as u32));
        Ok(Self {
            costs,
            contexts,
            rows,
            distance_tokens,
            small_distances,
            #[cfg(feature = "experimental-lz77-hotspots")]
            lengths: None,
        })
    }

    #[inline]
    #[cfg(feature = "experimental-lz77-hotspots")]
    pub(super) fn row(&self, context: u32) -> Row<'_> {
        if let Some(&index) = self
            .contexts
            .get(context as usize)
            .filter(|&&index| index != MISSING)
        {
            let (dense, sparse) = &self.rows[index as usize];
            Row {
                dense: Some(dense),
                sparse: Some(sparse),
                lengths: self.lengths.as_ref().map(|table| {
                    let begin = index as usize * table.ends.len();
                    (table, &table.bits[begin..begin + table.ends.len()])
                }),
            }
        } else {
            Row {
                dense: None,
                sparse: self.costs.literals.get(&context),
                lengths: None,
            }
        }
    }

    #[inline]
    #[cfg(not(feature = "experimental-lz77-hotspots"))]
    pub(super) fn row(&self, context: u32) -> Row<'_> {
        if let Some((dense, sparse)) = self
            .contexts
            .get(context as usize)
            .and_then(|&row| self.rows.get(row as usize))
        {
            Row {
                dense: Some(dense),
                sparse: Some(sparse),
            }
        } else {
            Row {
                dense: None,
                sparse: self.costs.literals.get(&context),
            }
        }
    }

    #[inline]
    pub(super) fn distance(&self, symbol: u32) -> f32 {
        if let Some(&price) = self.small_distances.get(symbol as usize) {
            return price;
        }
        let encoded = HybridUintConfig::default().encode(symbol);
        encoded.nbits as f32 + self.distance_tokens[encoded.token as usize]
    }

    pub(super) fn cumulative(&self, input: &[Literal], config: &SourceConfig) -> Result<Vec<f32>> {
        let _profile = crate::profile::scope("lz77.source.literal_costs");
        let mut sums = reserved(input.len() + 1)?;
        sums.push(0.0);
        for literal in input {
            let encoded = config.literal_uint_config.encode(literal.value);
            let bits = self.bits(literal.context, encoded.token);
            sums.push(bits + encoded.nbits as f32 + sums.last().copied().unwrap());
        }
        Ok(sums)
    }

    pub(super) fn optimal_cumulative(
        &self,
        input: &[Literal],
        config: &SourceConfig,
    ) -> Result<Vec<f64>> {
        let _profile = crate::profile::scope("lz77.source.literal_costs");
        let mut sums = reserved(input.len() + 1)?;
        sums.push(0.0);
        for literal in input {
            let encoded = config.literal_uint_config.encode(literal.value);
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

    #[inline]
    pub(super) fn bits(&self, context: u32, symbol: u32) -> f32 {
        if let Some(&price) = self
            .contexts
            .get(context as usize)
            .and_then(|&row| self.rows.get(row as usize))
            .and_then(|(row, _)| row.get(symbol as usize))
        {
            price
        } else {
            self.costs.bits(context, symbol)
        }
    }
}

#[cfg(test)]
mod penalty_tests;

#[cfg(test)]
mod tests {
    use super::super::{Command, ContextCost, SourceBucketMode, SourceLz77Method};
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    #[cfg(feature = "experimental-lz77-hotspots")]
    fn routes_without_compiled_intervals_keep_original_prices() {
        let costs = Costs {
            literals: BTreeMap::from([(
                0,
                ContextCost {
                    bits: BTreeMap::from([(224, 1.25)]),
                    penalty: 0.0,
                },
            )]),
            distances: ContextCost {
                bits: BTreeMap::new(),
                penalty: 0.0,
            },
        };
        for config in [
            SourceConfig {
                block_mode: super::super::SourceBlockMode::Intervals,
                shared_index: true,
                ..SourceConfig::default()
            },
            SourceConfig {
                match_cost: super::super::SourceMatchCost::RangeRefit,
                length_uint_config: HybridUintConfig::default(),
                ..SourceConfig::default()
            },
        ] {
            let prices = Prices::with_lengths(&costs, &config).unwrap();
            assert!(prices.lengths.is_none());
            for length in MIN_MATCH..1000 {
                assert_eq!(
                    prices.row(0).length(length, &config),
                    costs.length(0, length, &config)
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "experimental-lz77-hotspots")]
    fn capped_lengths_allow_high_symbol_offsets_without_cache_overflow() {
        let config = SourceConfig {
            max_match_length: MIN_MATCH,
            min_symbol: Some(u32::MAX),
            ..SourceConfig::default()
        };
        config.validate().unwrap();
        let costs = Costs {
            literals: BTreeMap::from([(
                0,
                ContextCost {
                    bits: BTreeMap::from([(u32::MAX, 1.25)]),
                    penalty: 0.0,
                },
            )]),
            distances: ContextCost {
                bits: BTreeMap::new(),
                penalty: 0.0,
            },
        };
        let prices = Prices::with_lengths(&costs, &config).unwrap();
        assert!(prices.lengths.is_none());
        let mut seen = Vec::new();
        prices
            .row(0)
            .intervals(MIN_MATCH, MIN_MATCH + 1, &config, |a, b, cost| {
                seen.push((a, b, cost));
            });
        assert_eq!(seen, [(MIN_MATCH, MIN_MATCH + 1, 1.25)]);
    }

    #[test]
    #[cfg(feature = "experimental-lz77-hotspots")]
    fn compiled_length_intervals_cover_boundaries_and_sparse_fallback() {
        let costs = Costs {
            literals: [0, 17, u32::MAX]
                .into_iter()
                .map(|context| {
                    (
                        context,
                        ContextCost {
                            bits: (0..1000)
                                .map(|symbol| (symbol, (symbol % 13) as f32 / 7.0))
                                .collect(),
                            penalty: 0.0,
                        },
                    )
                })
                .collect(),
            distances: ContextCost {
                bits: BTreeMap::new(),
                penalty: 0.0,
            },
        };
        for (split, msb) in [(0, 0), (4, 0), (4, 2), (7, 3), (15, 15)] {
            for offset in [224, 512, 900] {
                let config = SourceConfig {
                    length_uint_config: HybridUintConfig::new(split, msb, 0).unwrap(),
                    min_symbol: Some(offset),
                    ..SourceConfig::default()
                };
                let compiled = Prices::with_lengths(&costs, &config).unwrap();
                let plain = Prices::new(&costs).unwrap();
                assert_eq!(compiled.lengths.is_some(), (split, msb) <= (4, 2));
                for context in [0, 17, 99, u32::MAX] {
                    for value in [
                        0,
                        1,
                        2,
                        15,
                        16,
                        17,
                        255,
                        1023,
                        65535,
                        1 << 20,
                        u32::MAX - 2048,
                    ] {
                        for span in [0, 1, 17, 2048] {
                            if u64::from(value) + MIN_MATCH as u64 + span as u64 > usize::MAX as u64
                            {
                                continue;
                            }
                            let begin = value as usize + MIN_MATCH;
                            let end = begin + span;
                            let mut expected = Vec::new();
                            plain
                                .row(context)
                                .intervals(begin, end, &config, |a, b, bits| {
                                    expected.push((a, b, bits.to_bits()));
                                });
                            let mut actual = Vec::new();
                            compiled
                                .row(context)
                                .intervals(begin, end, &config, |a, b, bits| {
                                    actual.push((a, b, bits.to_bits()));
                                });
                            assert_eq!(
                                actual, expected,
                                "{split}/{msb} {offset} {context} {begin} {end}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compiled_prices_preserve_sparse_contexts_symbols_and_float_bits() {
        let contexts = [0, 1, 37, 65535, 65536, u32::MAX];
        let symbols = [0, 1, 7, 224, 255, 256, 65536, u32::MAX];
        let row = || ContextCost {
            bits: symbols
                .into_iter()
                .enumerate()
                .map(|(i, symbol)| (symbol, if i == 0 { -0.0 } else { i as f32 / 7.0 }))
                .collect(),
            penalty: 0.0,
        };
        let costs = Costs {
            literals: contexts
                .into_iter()
                .map(|context| (context, row()))
                .collect(),
            distances: row(),
        };
        let prices = Prices::new(&costs).unwrap();
        for context in contexts.into_iter().chain([2, 65534, u32::MAX - 1]) {
            for symbol in symbols.into_iter().chain([2, 254, 257]) {
                let expected = costs.bits(context, symbol).to_bits();
                assert_eq!(prices.bits(context, symbol).to_bits(), expected);
                assert_eq!(prices.row(context).bits(symbol).to_bits(), expected);
            }
        }
        for symbol in (0..65536).chain([1 << 24, u32::MAX]) {
            assert_eq!(
                prices.distance(symbol).to_bits(),
                costs.distance(symbol).to_bits()
            );
        }
        assert!(prices.contexts.len() <= CONTEXTS);
        assert!(prices.rows.len() <= ROWS);
    }

    #[test]
    fn compiled_price_row_budget_falls_back_without_losing_contexts() {
        let costs = Costs {
            literals: (0..ROWS as u32 + 3)
                .map(|context| {
                    (
                        context,
                        ContextCost {
                            bits: BTreeMap::from([(7, context as f32 * 0.25)]),
                            penalty: 0.0,
                        },
                    )
                })
                .collect(),
            distances: ContextCost {
                bits: BTreeMap::new(),
                penalty: 0.0,
            },
        };
        let prices = Prices::new(&costs).unwrap();
        assert_eq!(prices.rows.len(), ROWS);
        for context in 0..ROWS as u32 + 3 {
            assert_eq!(prices.bits(context, 7), costs.bits(context, 7));
            assert_eq!(prices.row(context).bits(7), costs.bits(context, 7));
        }
    }

    #[test]
    fn compiled_cumulative_and_intervals_equal_source_arithmetic() {
        let input: Vec<_> = (0..777u32)
            .map(|p| Literal {
                context: [0, 7, u32::MAX][p as usize % 3],
                value: if p % 41 == 0 { u32::MAX } else { p % 23 },
            })
            .collect();
        for literal_config in [
            HybridUintConfig::default(),
            HybridUintConfig::new(0, 0, 0).unwrap(),
        ] {
            for length_config in [
                HybridUintConfig::new(0, 0, 0).unwrap(),
                HybridUintConfig::new(4, 2, 0).unwrap(),
            ] {
                for bucket_mode in [SourceBucketMode::Default, SourceBucketMode::PreciseRing] {
                    let config = SourceConfig {
                        method: SourceLz77Method::Optc256,
                        bucket_mode,
                        literal_uint_config: literal_config,
                        length_uint_config: length_config,
                        ..SourceConfig::default()
                    };
                    let seed = vec![
                        input
                            .iter()
                            .copied()
                            .map(Command::Literal)
                            .chain([
                                Command::Match {
                                    context: 0,
                                    length: 33,
                                    distance: 37,
                                },
                                Command::Match {
                                    context: u32::MAX,
                                    length: 259,
                                    distance: 1024,
                                },
                            ])
                            .collect(),
                    ];
                    let costs = Costs::fitted(&seed, &[37], &config);
                    let prices = Prices::with_lengths(&costs, &config).unwrap();
                    assert_eq!(
                        prices.cumulative(&input, &config).unwrap(),
                        costs.cumulative(&input, &config).unwrap()
                    );
                    assert_eq!(
                        prices.optimal_cumulative(&input, &config).unwrap(),
                        costs.optimal_cumulative(&input, &config).unwrap()
                    );
                    for context in [0, 7, u32::MAX, 65535] {
                        for begin in [3, 4, 7, 18, 255, 258] {
                            let mut expected = Vec::new();
                            super::super::visit_intervals(begin, 1027, length_config, |a, b| {
                                expected.push((a, b, costs.length(context, a, &config).to_bits()));
                            });
                            let mut actual = Vec::new();
                            prices
                                .row(context)
                                .intervals(begin, 1027, &config, |a, b, cost| {
                                    actual.push((a, b, cost.to_bits()));
                                });
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }
}
