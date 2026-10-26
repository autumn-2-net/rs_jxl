// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Port of ANSEncodingHistogram in lib/jxl/enc_ans.cc and the population
//! precision/flat-histogram helpers in ans_common.h, baseline
//! b866b3defec13494bfbaf705db93581cda118e37.
//!
//! All five source normalization strategies, fixed-point greedy rebalancing,
//! flat/simple/general representations, omitted counts, and RLE are implemented.
//! Final-plan replay and diagnostic audit/profiling hooks are not implemented.
//! Strategy selection by effort belongs to the caller. Input weight totals are
//! checked against i32::MAX, the source algorithm's documented arithmetic bound.

use std::sync::OnceLock;

use crate::bit_writer::BitWriter;
use crate::{Error, Result};

pub(super) const TABLE_SIZE: usize = 4096;
pub(super) const LOG_TABLE_SIZE: u8 = 12;
pub(super) const MAX_ALPHABET_SIZE: usize = 256;

/// Precision trials used by the source ANSEncodingHistogram::ComputeBest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnsHistogramStrategy {
    Fast,
    Approximate,
    #[default]
    Precise,
    Direct0,
    Direct6,
}

#[derive(Clone, Debug)]
pub(super) struct NormalizedHistogram<Counts = Vec<u16>> {
    pub(super) counts: Counts,
    pub(super) cost: f32,
    method: u8,
    omit_pos: usize,
    alphabet_size: usize,
    num_symbols: usize,
    symbols: [usize; 2],
}

#[derive(Clone, Copy, Default)]
struct Bin {
    freq: i64,
    count_index: usize,
    bin_index: usize,
}

pub(super) fn population_precision(logcount: u8, shift: u8) -> u8 {
    logcount.min(shift.saturating_sub((LOG_TABLE_SIZE - logcount) >> 1))
}

fn floor_log2(n: u32) -> u8 {
    debug_assert_ne!(n, 0);
    (31 - n.leading_zeros()) as u8
}

fn smallest_increment_log(count: u32, shift: u8) -> u8 {
    if count == 0 {
        return 0;
    }
    let log = floor_log2(count);
    log - population_precision(log, shift)
}

fn log_table() -> &'static [u32; TABLE_SIZE + 1] {
    static TABLE: OnceLock<[u32; TABLE_SIZE + 1]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut logs = [0; TABLE_SIZE + 1];
        for (i, value) in logs.iter_mut().enumerate().skip(1) {
            *value = ((i as f64).log2() / 12.0 * 2147483648.0).round() as u32;
        }
        logs
    })
}

#[derive(Clone, Copy, Debug)]
struct AllowedCount {
    count: u16,
    step_log: u8,
    delta_log: i32,
}

struct AllowedCounts {
    entries: Vec<AllowedCount>,
    indices: [u16; TABLE_SIZE],
}

fn allowed_counts(shift: u8) -> &'static AllowedCounts {
    static TABLES: [OnceLock<AllowedCounts>; 12] = [const { OnceLock::new() }; 12];
    TABLES[usize::from(shift)].get_or_init(|| {
        let mut entries: Vec<AllowedCount> = Vec::with_capacity(TABLE_SIZE);
        let mut indices = [0; TABLE_SIZE];
        for i in (0..TABLE_SIZE as u32).rev() {
            let count = i & !((1 << smallest_increment_log(i, shift)) - 1);
            if entries
                .last()
                .is_some_and(|last| u32::from(last.count) == count)
            {
                continue;
            }
            let (step_log, delta_log) = if count == 0 {
                (0, i32::MAX)
            } else if let Some(last) = entries.last() {
                let previous = u32::from(last.count);
                (
                    floor_log2(previous - count),
                    ((f64::from(previous) / f64::from(count)).log2() / 12.0 * 2147483648.0).round()
                        as i32,
                )
            } else {
                (0, 0)
            };
            indices[count as usize] = entries.len() as u16;
            entries.push(AllowedCount {
                count: count as u16,
                step_log,
                delta_log,
            });
        }
        AllowedCounts { entries, indices }
    })
}

fn write_varlen_u8(n: usize, emit: &mut impl FnMut(u8, u64) -> Result<()>) -> Result<()> {
    debug_assert!(n <= 255);
    if n == 0 {
        emit(1, 0)
    } else {
        let bits = floor_log2(n as u32);
        emit(1, 1)?;
        emit(3, u64::from(bits))?;
        emit(bits, (n - (1 << bits)) as u64)
    }
}

fn input_shape(counts: &[u64]) -> Result<(u64, usize)> {
    if counts.len() > MAX_ALPHABET_SIZE {
        return Err(Error::InvalidInput(
            "ANS histogram has more than 256 bins".into(),
        ));
    }
    let total = counts.iter().try_fold(0u64, |sum, &count| {
        sum.checked_add(count)
            .filter(|&n| n <= i32::MAX as u64)
            .ok_or(Error::Limit("ANS histogram weight total exceeds i32::MAX"))
    })?;
    let alphabet_size = counts.iter().rposition(|&n| n != 0).map_or(0, |n| n + 1);
    Ok((total, alphabet_size))
}

/// Exact Fast population cost without retaining a normalized coding table.
pub(super) fn fast_population_cost(counts: &[u64]) -> Result<f32> {
    let (total, alphabet_size) = input_shape(counts)?;
    if alphabet_size == 0 {
        return Ok(3.0);
    }
    let mut storage = [0u16; MAX_ALPHABET_SIZE];
    let mut normalized = NormalizedHistogram {
        counts: &mut storage[..alphabet_size],
        cost: 0.0,
        method: 0,
        omit_pos: 0,
        alphabet_size,
        num_symbols: 0,
        symbols: [0; 2],
    };
    for (symbol, &count) in counts[..alphabet_size].iter().enumerate() {
        if count != 0 {
            if normalized.num_symbols < 2 {
                normalized.symbols[normalized.num_symbols] = symbol;
            }
            normalized.num_symbols += 1;
        }
    }
    if normalized.num_symbols == 1 {
        normalized.method = 1;
        normalized.counts[normalized.symbols[0]] = TABLE_SIZE as u16;
        return Ok(normalized.histogram_bits()? as f32);
    }
    // Flat header bits do not read counts; each rebalance overwrites every bin.
    let flat_sum = u128::from(total) * u128::from(log_table()[alphabet_size]) * 12;
    let mut cost = normalized.histogram_bits()? as f32 + flat_sum as f32 * (1.0 / 2147483648.0);
    let mut bins = [Bin::default(); MAX_ALPHABET_SIZE];
    for shift in [0, 6, 11] {
        normalized.method = shift + 1;
        normalized.rebalance_with_scratch(&counts[..alphabet_size], total, &mut bins)?;
        let sum: i64 = counts[..alphabet_size]
            .iter()
            .zip(normalized.counts.iter())
            .map(|(&weight, &count)| weight as i64 * i64::from(log_table()[usize::from(count)]))
            .sum();
        let data_bits = (total as f32 - sum as f32 * (1.0 / 2147483648.0)) * 12.0;
        let candidate = normalized.histogram_bits()? as f32 + data_bits;
        if candidate < cost {
            cost = candidate;
        }
    }
    Ok(cost)
}

impl NormalizedHistogram {
    pub(super) fn compute(counts: &[u64], strategy: AnsHistogramStrategy) -> Result<Self> {
        let (total, alphabet_size) = input_shape(counts)?;
        let mut result = Self {
            counts: vec![0; alphabet_size.max(1)],
            cost: 0.0,
            method: 0,
            omit_pos: 0,
            alphabet_size,
            num_symbols: 0,
            symbols: [0; 2],
        };
        if alphabet_size == 0 {
            result.method = 1;
            result.cost = 3.0;
            // The serialized empty histogram and InitAliasTable both use symbol 0.
            result.counts[0] = TABLE_SIZE as u16;
            return Ok(result);
        }
        for (symbol, &count) in counts[..alphabet_size].iter().enumerate() {
            if count != 0 {
                if result.num_symbols < 2 {
                    result.symbols[result.num_symbols] = symbol;
                }
                result.num_symbols += 1;
            }
        }
        if result.num_symbols == 1 {
            result.method = 1;
            result.counts[result.symbols[0]] = TABLE_SIZE as u16;
            result.cost = result.histogram_bits()? as f32;
            return Ok(result);
        }

        for (i, count) in result.counts.iter_mut().enumerate() {
            *count =
                (TABLE_SIZE / alphabet_size + usize::from(i < TABLE_SIZE % alphabet_size)) as u16;
        }
        let direct = matches!(
            strategy,
            AnsHistogramStrategy::Direct0 | AnsHistogramStrategy::Direct6
        );
        if !direct {
            // u128 avoids overflowing the source flat-cost intermediate for large
            // but otherwise supported weight totals. The final cost stays f32.
            let flat_sum = u128::from(total) * u128::from(log_table()[alphabet_size]) * 12;
            result.cost = result.histogram_bits()? as f32 + flat_sum as f32 * (1.0 / 2147483648.0);
        }
        let shifts: &[u8] = match strategy {
            AnsHistogramStrategy::Fast => &[0, 6, 11],
            AnsHistogramStrategy::Approximate => &[0, 2, 4, 6, 8, 10, 11],
            AnsHistogramStrategy::Precise => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            AnsHistogramStrategy::Direct0 => &[0],
            AnsHistogramStrategy::Direct6 => &[6],
        };
        let mut normalized = result.clone();
        for &shift in shifts {
            normalized.method = shift + 1;
            normalized.rebalance(&counts[..alphabet_size], total)?;
            let sum: i64 = counts[..alphabet_size]
                .iter()
                .zip(&normalized.counts)
                .map(|(&weight, &count)| weight as i64 * i64::from(log_table()[usize::from(count)]))
                .sum();
            let data_bits = (total as f32 - sum as f32 * (1.0 / 2147483648.0)) * 12.0;
            normalized.cost = normalized.histogram_bits()? as f32 + data_bits;
            if direct || normalized.cost < result.cost {
                result = normalized.clone();
            }
        }
        Ok(result)
    }

    pub(super) fn write(&self, writer: &mut BitWriter) -> Result<()> {
        self.encode(&mut |bits, value| Ok(writer.write(bits, value)?))
    }

    fn rebalance(&mut self, weights: &[u64], total: u64) -> Result<()> {
        self.rebalance_with_scratch(weights, total, &mut [Bin::default(); MAX_ALPHABET_SIZE])
    }
}

impl<Counts: AsRef<[u16]>> NormalizedHistogram<Counts> {
    pub(super) fn histogram_bits(&self) -> Result<usize> {
        let mut size = 0;
        self.encode(&mut |bits, _| {
            size += usize::from(bits);
            Ok(())
        })?;
        Ok(size)
    }

    fn encode(&self, emit: &mut impl FnMut(u8, u64) -> Result<()>) -> Result<()> {
        let counts = self.counts.as_ref();
        if self.method == 0 {
            emit(1, 0)?;
            emit(1, 1)?;
            return write_varlen_u8(self.alphabet_size - 1, emit);
        }
        if self.num_symbols <= 2 {
            emit(1, 1)?;
            if self.num_symbols == 0 {
                emit(1, 0)?;
                write_varlen_u8(0, emit)?;
            } else {
                emit(1, (self.num_symbols - 1) as u64)?;
                for &symbol in &self.symbols[..self.num_symbols] {
                    write_varlen_u8(symbol, emit)?;
                }
            }
            if self.num_symbols == 2 {
                emit(LOG_TABLE_SIZE, u64::from(counts[self.symbols[0]]))?;
            }
            return Ok(());
        }
        emit(2, 0)?;
        let log = floor_log2(u32::from(self.method));
        emit(log, (1 << log) - 1)?;
        if log != 3 {
            emit(1, 0)?;
        }
        emit(log, u64::from(self.method) & ((1 << log) - 1))?;
        write_varlen_u8(self.alphabet_size - 3, emit)?;

        let mut same = [0usize; MAX_ALPHABET_SIZE];
        let mut last = 0;
        for i in 1..=self.alphabet_size {
            if i == self.alphabet_size
                || i == self.omit_pos
                || i == self.omit_pos + 1
                || counts[i] != counts[last]
            {
                same[last] = i - last;
                last = i;
            }
        }
        let mut widths = [0u8; MAX_ALPHABET_SIZE];
        let mut omit_width = 10;
        for (i, &count) in counts[..self.alphabet_size].iter().enumerate() {
            if i != self.omit_pos && count > 0 {
                widths[i] = floor_log2(u32::from(count)) + 1;
                omit_width = omit_width.max(widths[i] + u8::from(i < self.omit_pos));
            }
        }
        widths[self.omit_pos] = omit_width;
        const LENGTHS: [u8; 14] = [5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7, 7];
        const SYMBOLS: [u8; 14] = [17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65];
        let mut i = 0;
        while i < self.alphabet_size {
            let width = usize::from(widths[i]);
            emit(LENGTHS[width], u64::from(SYMBOLS[width]))?;
            if same[i] >= 5 {
                emit(LENGTHS[13], u64::from(SYMBOLS[13]))?;
                write_varlen_u8(same[i] - 5, emit)?;
                i += same[i];
            } else {
                i += 1;
            }
        }
        let shift = self.method - 1;
        if shift != 0 {
            let mut i = 0;
            while i < self.alphabet_size {
                if widths[i] > 1 && i != self.omit_pos {
                    let bitcount = population_precision(widths[i] - 1, shift);
                    let drop_bits = widths[i] - 1 - bitcount;
                    debug_assert_eq!(counts[i] & ((1 << drop_bits) - 1), 0);
                    emit(
                        bitcount,
                        u64::from((counts[i] >> drop_bits) - (1 << bitcount)),
                    )?;
                }
                i += if same[i] >= 5 { same[i] } else { 1 };
            }
        }
        Ok(())
    }
}

impl<Counts: AsRef<[u16]> + AsMut<[u16]>> NormalizedHistogram<Counts> {
    fn rebalance_with_scratch(
        &mut self,
        weights: &[u64],
        total: u64,
        bins: &mut [Bin; MAX_ALPHABET_SIZE],
    ) -> Result<()> {
        let counts = self.counts.as_mut();
        let shift = self.method - 1;
        let allowed = allowed_counts(shift);
        let ac = &allowed.entries;
        let logs = log_table();
        let norm = TABLE_SIZE as f64 / total as f64;
        let mut num_bins = 0;
        let mut remainder_pos = 0;
        let mut max_freq = 0i64;
        let mut rest = TABLE_SIZE as i32;
        for (n, &weight) in weights.iter().enumerate() {
            let freq = weight as i64;
            if freq > max_freq {
                remainder_pos = n;
                max_freq = freq;
            }
            let target = weight as f64 * norm;
            let mut count = (target.round() as u32)
                .max(u32::from(weight > 0))
                .min(TABLE_SIZE as u32 - 1);
            count &= !((1 << smallest_increment_log(count, shift)) - 1);
            counts[n] = count as u16;
            rest -= count as i32;
            if target > 1.0 {
                bins[num_bins] = Bin {
                    freq,
                    count_index: usize::from(allowed.indices[count as usize]),
                    bin_index: n,
                };
                num_bins += 1;
            }
        }
        // Stable compaction preserves first-equal increment/decrement choices.
        let mut retained = 0;
        for i in 0..num_bins {
            if bins[i].bin_index != remainder_pos {
                bins[retained] = bins[i];
                retained += 1;
            }
        }
        let bins = &mut bins[..retained];
        rest += i32::from(counts[remainder_pos]);
        if !bins.is_empty() {
            let max_log = ac[1].step_log;
            let mut balance_inc = [0i64; 11];
            let mut balance_dec = [0i64; 11];
            loop {
                for log in 0..=max_log {
                    let delta = 1i32 << log;
                    let slot = usize::from(log);
                    if rest >= TABLE_SIZE as i32 {
                        balance_inc[slot] = 0;
                        balance_dec[slot] = 0;
                    } else if rest > 1 {
                        balance_inc[slot] = if rest > delta {
                            max_freq
                                * i64::from(logs[rest as usize] - logs[(rest - delta) as usize])
                        } else {
                            i64::MAX
                        };
                        balance_dec[slot] = if rest + delta < TABLE_SIZE as i32 {
                            max_freq
                                * i64::from(logs[(rest + delta) as usize] - logs[rest as usize])
                        } else {
                            0
                        };
                    } else {
                        balance_inc[slot] = i64::MAX;
                        balance_dec[slot] = i64::MAX;
                    }
                }
                let inc = |bin: &Bin| {
                    let entry = ac[bin.count_index];
                    bin.freq * i64::from(entry.delta_log) - balance_inc[usize::from(entry.step_log)]
                };
                let dec = |bin: &Bin| {
                    // Source uses an INT32_MAX delta at count zero as a guard.
                    // When the balancing count reaches one, its INT64_MAX
                    // repair penalty can overpower that finite sentinel. Never
                    // remove the last occurrence of a present symbol or step
                    // beyond the compact allowed-count table.
                    match ac.get(bin.count_index + 1).filter(|entry| entry.count != 0) {
                        Some(entry) => (
                            bin.freq * i64::from(entry.delta_log)
                                - balance_dec[usize::from(entry.step_log)],
                            entry.step_log,
                        ),
                        None => (i64::MAX, 0),
                    }
                };
                // Keep the first equal candidate, as std::max/min_element do.
                let mut best_inc = 0;
                for i in 1..bins.len() {
                    if inc(&bins[i]) >> ac[bins[i].count_index].step_log
                        > inc(&bins[best_inc]) >> ac[bins[best_inc].count_index].step_log
                    {
                        best_inc = i;
                    }
                }
                if inc(&bins[best_inc]) > 0 {
                    let bin = &mut bins[best_inc];
                    rest -= 1 << ac[bin.count_index].step_log;
                    bin.count_index -= 1;
                } else {
                    let mut best_dec = 0;
                    for i in 1..bins.len() {
                        let candidate = dec(&bins[i]);
                        let best = dec(&bins[best_dec]);
                        if candidate.0 >> candidate.1 < best.0 >> best.1 {
                            best_dec = i;
                        }
                    }
                    if dec(&bins[best_dec]).0 >= 0 {
                        break;
                    }
                    let bin = &mut bins[best_dec];
                    bin.count_index += 1;
                    rest += 1 << ac[bin.count_index].step_log;
                }
            }
            for bin in bins {
                counts[bin.bin_index] = ac[bin.count_index].count;
            }
            for n in 0..remainder_pos {
                if counts[n] >= 2048 {
                    counts[remainder_pos] = counts[n];
                    remainder_pos = n;
                    break;
                }
            }
        }
        if !(1..TABLE_SIZE as i32).contains(&rest) {
            return Err(Error::Encoding("ANS histogram rebalancing failed".into()));
        }
        counts[remainder_pos] = rest as u16;
        self.omit_pos = remainder_pos;
        if counts
            .iter()
            .map(|&count| usize::from(count))
            .sum::<usize>()
            != TABLE_SIZE
        {
            return Err(Error::Encoding(
                "ANS normalized counts do not sum to 4096".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod test_decode {
    pub struct Reader<'a> {
        pub bytes: &'a [u8],
        pub position: usize,
    }

    impl Reader<'_> {
        pub fn read(&mut self, width: u8) -> u32 {
            let mut value = 0;
            for bit in 0..width {
                assert!(
                    self.position < self.bytes.len() * 8,
                    "truncated test stream"
                );
                value |=
                    u32::from((self.bytes[self.position / 8] >> (self.position % 8)) & 1) << bit;
                self.position += 1;
            }
            value
        }

        fn varlen(&mut self) -> usize {
            if self.read(1) == 0 {
                0
            } else {
                let width = self.read(3) as u8;
                (1 << width) + self.read(width) as usize
            }
        }

        fn count_width(&mut self) -> u8 {
            // Independent bit-at-a-time decoding of the format's static code.
            const CODES: [(u8, u32); 14] = [
                (5, 17),
                (4, 11),
                (4, 15),
                (4, 3),
                (4, 9),
                (4, 7),
                (3, 4),
                (3, 2),
                (3, 5),
                (3, 6),
                (3, 0),
                (6, 33),
                (7, 1),
                (7, 65),
            ];
            let mut code = 0;
            for width in 1..=7 {
                code |= self.read(1) << (width - 1);
                if let Some(symbol) = CODES.iter().position(|&entry| entry == (width, code)) {
                    return symbol as u8;
                }
            }
            panic!("invalid count-width code");
        }

        pub fn histogram(&mut self) -> Vec<u16> {
            if self.read(1) != 0 {
                let symbols = self.read(1) + 1;
                let first = self.varlen();
                if symbols == 1 {
                    let mut counts = vec![0; first + 1];
                    counts[first] = 4096;
                    return counts;
                }
                let second = self.varlen();
                assert_ne!(first, second);
                let mut counts = vec![0; first.max(second) + 1];
                counts[first] = self.read(12) as u16;
                counts[second] = 4096 - counts[first];
                return counts;
            }
            if self.read(1) != 0 {
                let length = self.varlen() + 1;
                return (0..length)
                    .map(|i| (4096 / length + usize::from(i < 4096 % length)) as u16)
                    .collect();
            }
            let mut log = 0;
            while log < 3 && self.read(1) != 0 {
                log += 1;
            }
            let shift = (1 << log) + self.read(log) - 1;
            let length = self.varlen() + 3;
            let mut groups: Vec<(usize, u8, usize)> = Vec::new();
            let mut position = 0;
            let mut omitted = 0;
            let mut largest = 0;
            while position < length {
                let width = self.count_width();
                if width == 13 {
                    let repeated = self.varlen() + 5;
                    let previous = groups.last_mut().expect("repeat needs a preceding count");
                    assert_eq!(previous.2, 1);
                    previous.2 = repeated;
                    position += repeated - 1;
                } else {
                    if width > largest {
                        largest = width;
                        omitted = position;
                    }
                    groups.push((position, width, 1));
                    position += 1;
                }
            }
            assert_eq!(position, length);
            let mut counts = vec![0u16; length];
            for (start, width, repeated) in groups {
                if start == omitted {
                    assert_eq!(repeated, 1);
                    continue;
                }
                let value = if width == 0 {
                    0
                } else {
                    let logcount = width - 1;
                    let precision = (i32::try_from(shift).unwrap() - i32::from((12 - logcount) / 2))
                        .clamp(0, i32::from(logcount)) as u8;
                    (1 << logcount) + (self.read(precision) << (logcount - precision))
                };
                counts[start..start + repeated].fill(value as u16);
            }
            let sum: u32 = counts.iter().map(|&n| u32::from(n)).sum();
            assert!(sum < 4096);
            counts[omitted] = (4096 - sum) as u16;
            counts
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_decode::Reader;

    const STRATEGIES: [AnsHistogramStrategy; 5] = [
        AnsHistogramStrategy::Fast,
        AnsHistogramStrategy::Approximate,
        AnsHistogramStrategy::Precise,
        AnsHistogramStrategy::Direct0,
        AnsHistogramStrategy::Direct6,
    ];

    fn assert_fast_population_cost(counts: &[u64]) {
        let expected = NormalizedHistogram::compute(counts, AnsHistogramStrategy::Fast).unwrap();
        assert_eq!(
            fast_population_cost(counts).unwrap().to_bits(),
            expected.cost.to_bits(),
            "counts={counts:?} method={}",
            expected.method
        );
    }

    fn population_fixtures() -> Vec<Vec<u64>> {
        let mut fixtures = vec![
            vec![2048, 2047, 1],
            vec![1, 2, 3, 5, 8, 13, 16384],
            vec![i32::MAX as u64 - 255, 1, 254],
            vec![1_000_000; 256],
            (0..256).map(|i| if i == 0 { 1 } else { 257 }).collect(),
        ];
        for size in [
            3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128, 255, 256,
        ] {
            fixtures.push(vec![1; size]);
            fixtures.push(
                (0..size)
                    .map(|i| if i % 7 == 0 { 0 } else { (i * i + 1) as u64 })
                    .collect(),
            );
            for omit in [0, size / 2, size - 1] {
                let mut counts = vec![1; size];
                counts[omit] = i32::MAX as u64 - size as u64 + 1;
                fixtures.push(counts);
            }
        }
        fixtures
    }

    #[test]
    fn fast_population_cost_empty_singletons_and_pairs_match_compute_bits() {
        for size in 0..=256 {
            assert_fast_population_cost(&vec![0; size]);
        }
        for symbol in 0..256 {
            for weight in [1, 4096, i32::MAX as u64] {
                let mut counts = vec![0; 256];
                counts[symbol] = weight;
                assert_fast_population_cost(&counts);
            }
        }
        for a in [0, 1, 15, 31, 127, 254] {
            for b in [1, 16, 32, 128, 255] {
                if a >= b {
                    continue;
                }
                for weight in [1, 2048, i32::MAX as u64 - 1] {
                    let mut counts = vec![0; 256];
                    counts[a] = weight;
                    counts[b] = 1;
                    assert_fast_population_cost(&counts);
                }
            }
        }
    }

    #[test]
    fn fast_population_cost_dense_sparse_rle_and_large_weights_match_compute_bits() {
        for counts in population_fixtures() {
            assert_fast_population_cost(&counts);
            if counts.len() < 256 {
                let mut padded = counts;
                padded.resize(256, 0);
                assert_fast_population_cost(&padded);
            }
        }
        let mut state = 0x297a_3c41u32;
        for case in 0..384 {
            let length = case % 256 + 1;
            let counts: Vec<_> = (0..length)
                .map(|_| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    if state & 7 == 0 {
                        0
                    } else {
                        u64::from((state >> 8) % (1 << (case % 17)))
                    }
                })
                .collect();
            assert_fast_population_cost(&counts);
        }
    }

    #[test]
    fn fast_population_cost_preserves_validation_errors_and_order() {
        for counts in [
            vec![0; 257],
            vec![u64::MAX; 257],
            vec![u64::MAX],
            vec![u64::MAX, 1],
            vec![i32::MAX as u64, 1],
            vec![0, i32::MAX as u64 + 1, 0],
        ] {
            let expected =
                NormalizedHistogram::compute(&counts, AnsHistogramStrategy::Fast).unwrap_err();
            let actual = fast_population_cost(&counts).unwrap_err();
            assert_eq!(actual.to_string(), expected.to_string());
        }
    }

    #[test]
    fn stack_scratch_preserves_all_fast_rebalances_counts_omission_and_bits() {
        let mut bins = [Bin {
            freq: i64::MIN,
            count_index: usize::MAX,
            bin_index: usize::MAX,
        }; MAX_ALPHABET_SIZE];
        let mut storage = [u16::MAX; MAX_ALPHABET_SIZE];
        for counts in population_fixtures().into_iter().rev() {
            let prototype =
                NormalizedHistogram::compute(&counts, AnsHistogramStrategy::Direct0).unwrap();
            let size = prototype.alphabet_size;
            let weights = &counts[..size];
            let total = counts.iter().sum();
            for shift in [0, 6, 11] {
                let mut expected = prototype.clone();
                expected.method = shift + 1;
                expected.rebalance(weights, total).unwrap();
                storage.fill(u16::MAX);
                let mut actual = NormalizedHistogram {
                    counts: &mut storage[..size],
                    cost: 0.0,
                    method: shift + 1,
                    omit_pos: usize::MAX,
                    alphabet_size: size,
                    num_symbols: prototype.num_symbols,
                    symbols: prototype.symbols,
                };
                actual
                    .rebalance_with_scratch(weights, total, &mut bins)
                    .unwrap();
                assert_eq!(&*actual.counts, expected.counts.as_slice());
                assert_eq!(actual.omit_pos, expected.omit_pos);
                assert_eq!(
                    actual.histogram_bits().unwrap(),
                    expected.histogram_bits().unwrap()
                );
                let mut writer = BitWriter::new(4096);
                actual
                    .encode(&mut |bits, value| Ok(writer.write(bits, value)?))
                    .unwrap();
                writer.zero_pad_to_byte().unwrap();
                let mut oracle = BitWriter::new(4096);
                expected.write(&mut oracle).unwrap();
                oracle.zero_pad_to_byte().unwrap();
                assert_eq!(writer.into_bytes().unwrap(), oracle.into_bytes().unwrap());
                assert!(storage[size..].iter().all(|&count| count == u16::MAX));
            }
        }
    }

    fn roundtrip(histogram: &NormalizedHistogram) {
        let mut writer = BitWriter::new(4096);
        writer.write(3, 5).unwrap();
        histogram.write(&mut writer).unwrap();
        let end = writer.bits_written();
        assert_eq!(end - 3, histogram.histogram_bits().unwrap());
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        let mut reader = Reader {
            bytes: &bytes,
            position: 3,
        };
        assert_eq!(reader.histogram(), histogram.counts);
        assert_eq!(reader.position, end);
        assert_eq!(
            histogram.counts.iter().map(|&n| u32::from(n)).sum::<u32>(),
            4096
        );
    }

    #[test]
    fn rebalancing_never_decrements_present_symbols_to_zero() {
        let cases: Vec<Vec<u64>> = vec![
            (0..256).map(|i| if i == 0 { 1 } else { 257 }).collect(),
            (0..256)
                .map(|i| ((i * 701 + 37) % 1013 + 1) as u64)
                .collect(),
            (0..256).map(|i| if i % 5 == 0 { 1 } else { 64 }).collect(),
            vec![1, 2, 3, 5, 8, 13, 16384],
        ];
        for counts in cases {
            for strategy in STRATEGIES {
                let histogram = NormalizedHistogram::compute(&counts, strategy).unwrap();
                assert!(histogram.counts.iter().all(|&count| count > 0));
                roundtrip(&histogram);
            }
        }
    }

    #[test]
    fn histogram_known_bit_vectors() {
        // Hand-derived LSB-first format vectors, independent of the encoder.
        let cases: &[(&[u64], AnsHistogramStrategy, usize, &[u8])] = &[
            (&[], AnsHistogramStrategy::Precise, 3, &[0x01]),
            (
                &[0, 0, 0, 0, 0, 7],
                AnsHistogramStrategy::Precise,
                8,
                &[0x55],
            ),
            (
                &[3072, 1024],
                AnsHistogramStrategy::Direct0,
                19,
                &[0x0b, 0x00, 0x06],
            ),
            (&[1, 1, 1, 1], AnsHistogramStrategy::Precise, 7, &[0x4e]),
            (
                &[2048, 1024, 1024],
                AnsHistogramStrategy::Direct0,
                22,
                &[0x10, 0x86, 0x21],
            ),
        ];
        for &(counts, strategy, bits, expected) in cases {
            let histogram = NormalizedHistogram::compute(counts, strategy).unwrap();
            let mut writer = BitWriter::new(4096);
            histogram.write(&mut writer).unwrap();
            assert_eq!(writer.bits_written(), bits);
            writer.zero_pad_to_byte().unwrap();
            assert_eq!(writer.into_bytes().unwrap(), expected);
            roundtrip(&histogram);
        }
    }

    #[test]
    fn empty_singletons_and_sparse_histograms() {
        for strategy in STRATEGIES {
            for size in [0, 1, 32, 256] {
                roundtrip(&NormalizedHistogram::compute(&vec![0; size], strategy).unwrap());
            }
            for symbol in [0, 1, 31, 32, 127, 255] {
                let mut counts = vec![0; 256];
                counts[symbol] = 17;
                let histogram = NormalizedHistogram::compute(&counts, strategy).unwrap();
                assert_eq!(histogram.counts[symbol], 4096);
                roundtrip(&histogram);
            }
            let mut counts = vec![0; 256];
            counts[0] = 100_000;
            counts[9] = 3;
            counts[150] = 2;
            counts[255] = 1;
            let histogram = NormalizedHistogram::compute(&counts, strategy).unwrap();
            assert_ne!(histogram.method, 0);
            for (i, &count) in counts.iter().enumerate() {
                assert_eq!(histogram.counts[i] > 0, count > 0);
            }
            roundtrip(&histogram);
        }
    }

    #[test]
    fn every_precision_and_rle_boundaries_decode() {
        let mut cases = vec![vec![1; 256], vec![100; 256], vec![2048, 2047, 1]];
        for omit in [0, 1, 4, 5, 127, 250, 255] {
            let mut counts = vec![1; 256];
            counts[omit] = 100_000;
            cases.push(counts);
        }
        for size in [3, 4, 5, 6, 7, 32, 128, 256] {
            cases.push(
                (0..size)
                    .map(|i| if i % 7 == 0 { 0 } else { (i * i + 1) as u64 })
                    .collect(),
            );
        }
        for counts in cases {
            let mut histogram =
                NormalizedHistogram::compute(&counts, AnsHistogramStrategy::Direct0).unwrap();
            for shift in 0..12 {
                histogram.method = shift + 1;
                histogram
                    .rebalance(&counts[..histogram.alphabet_size], counts.iter().sum())
                    .unwrap();
                for (i, &count) in histogram.counts.iter().enumerate() {
                    assert_eq!(count != 0, counts[i] != 0);
                    if count != 0 && i != histogram.omit_pos {
                        let drop = smallest_increment_log(u32::from(count), shift);
                        assert_eq!(count & ((1 << drop) - 1), 0);
                    }
                }
                roundtrip(&histogram);
            }
        }
    }

    #[test]
    fn fixed_point_tables_match_representability_and_step_sizes() {
        assert_eq!(log_table()[1], 0);
        assert_eq!(log_table()[4096], 1 << 31);
        for shift in 0..12 {
            let allowed = allowed_counts(shift);
            assert_eq!(allowed.entries.last().unwrap().count, 0);
            for (i, entry) in allowed.entries.iter().enumerate() {
                assert_eq!(usize::from(allowed.indices[usize::from(entry.count)]), i);
                let drop = smallest_increment_log(u32::from(entry.count), shift);
                assert_eq!(entry.count & ((1 << drop) - 1), 0);
                if i > 0 && entry.count != 0 {
                    assert_eq!(
                        allowed.entries[i - 1].count - entry.count,
                        1 << entry.step_log
                    );
                    assert!(entry.delta_log > 0);
                }
            }
        }
    }

    #[test]
    fn bounded_weights_and_strategy_costs() {
        assert!(
            NormalizedHistogram::compute(&[u64::MAX, 1], AnsHistogramStrategy::Precise).is_err()
        );
        assert!(
            NormalizedHistogram::compute(&[i32::MAX as u64, 1], AnsHistogramStrategy::Precise)
                .is_err()
        );
        assert!(NormalizedHistogram::compute(&[0; 257], AnsHistogramStrategy::Precise).is_err());
        for counts in [
            vec![i32::MAX as u64],
            vec![i32::MAX as u64 - 1, 1],
            vec![i32::MAX as u64 - 255, 1, 254],
            vec![1_000_000; 256],
        ] {
            for strategy in STRATEGIES {
                let histogram = NormalizedHistogram::compute(&counts, strategy).unwrap();
                assert!(histogram.cost.is_finite());
                assert!(histogram.cost >= 0.0);
                roundtrip(&histogram);
            }
        }
        let counts: Vec<u64> = (0..256).map(|i| ((i * 701 + 37) % 1013) as u64).collect();
        let precise = NormalizedHistogram::compute(&counts, AnsHistogramStrategy::Precise).unwrap();
        for strategy in [
            AnsHistogramStrategy::Fast,
            AnsHistogramStrategy::Approximate,
        ] {
            let other = NormalizedHistogram::compute(&counts, strategy).unwrap();
            assert!(precise.cost <= other.cost);
        }
    }
}
