//! Bounded compiled views of a fitted model; sparse u32 contexts/symbols remain valid.

use super::{Costs, HybridUintConfig, Literal, MIN_MATCH, SourceConfig, reserved};
use crate::Result;

const SYMBOLS: usize = 256;
const CONTEXTS: usize = 65536;
const ROWS: usize = 8192;
const MISSING: u16 = u16::MAX;

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
}

#[derive(Clone, Copy)]
pub(super) struct Row<'a> {
    dense: Option<&'a [f32; SYMBOLS]>,
    sparse: Option<&'a super::ContextCost>,
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
        })
    }

    #[inline]
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
#[path = "lz77_source_penalty_tests.rs"]
mod penalty_tests;

#[cfg(test)]
mod tests {
    use super::super::{Command, ContextCost, SourceBucketMode, SourceLz77Method};
    use super::*;
    use std::collections::BTreeMap;

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
                    let prices = Prices::new(&costs).unwrap();
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
