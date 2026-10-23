// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from the BSD-3-Clause sources identified in docs/PORTING_KERNELS.md.
// The complete upstream notice and license are retained in that document.

//! Exact integer sample weighting and incremental probability-floored entropy.
//!
//! Optional learner scoring and standalone weighted primitives. The default
//! modular::ma_train path still uses ma_simd::DefaultScorer's f32 expression.
//! Ported from
//! libjxl b866b3defec13494bfbaf705db93581cda118e37:
//! `lib/jxl/modular/encoding/enc_ma_entropy.h::MaClippedEntropy` and
//! `enc_ma.cc::{CountEntropyTable, CountEntropy, HistogramCountEntropy}`.
//! The entropy model is `sum(count * min(log2(total / count), 12))`.
//! Counts and weights are exact integers. Floating-point entropy is compared to
//! that model within tolerance, not promised bit-identical across math libraries.
//! A shared 0..=65536 f64 c*log2(c) cache avoids repeated logarithms. Native
//! SSE2/AVX2 kernels initialize sums and update both sides of split transfers;
//! non-x86 targets use scalar code. Counts above the cache retain f64 log2 and
//! the full u32 domain. The Fenwick correction is ALWAYS applied: this is not
//! the source's unclipped FastEntropy/Shannon profile or ma_simd's FastLog2f
//! approximation. No histogram rescan is added to incremental bits()/updates.
//! SourceClippedEntropy supplies reusable count-frequency state for explicit
//! learner Fenwick/sliding modes; raw histogram columns remain learner-owned.
//! These tools do not implicitly change the trainer's scoring profile.

use super::ma_simd::CountEntropyTable;
use std::fmt;

pub const ANS_LOG_TAB_SIZE: u32 = 12;
pub const ANS_TAB_SIZE: u32 = 1 << ANS_LOG_TAB_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EntropyError {
    SymbolOutOfBounds,
    CountOverflow,
    CountUnderflow,
    MaximumTotalExceeded,
    ExtraBitsOverflow,
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SymbolOutOfBounds => "histogram symbol is out of bounds",
            Self::CountOverflow => "histogram count exceeds u32",
            Self::CountUnderflow => "sample weight exceeds the source count",
            Self::MaximumTotalExceeded => "histogram exceeds its declared maximum total",
            Self::ExtraBitsOverflow => "weighted extra-bit total exceeds u64",
        })
    }
}

impl std::error::Error for EntropyError {}

/// One deduplicated sample. Weight is multiplicity, not a probability or an
/// estimate; zero weight is permitted and still requires a valid symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightedSample {
    pub symbol: usize,
    pub weight: u16,
    pub extra_bits: u8,
}

/// Add one sample's full weight to its count and its extra-bit cost. Errors
/// leave both outputs unchanged. A caller looping over samples gets per-sample,
/// not whole-batch, atomicity.
pub fn add_weighted_sample(
    counts: &mut [u32],
    extra_bits: &mut u64,
    sample: WeightedSample,
) -> Result<(), EntropyError> {
    let old = *counts
        .get(sample.symbol)
        .ok_or(EntropyError::SymbolOutOfBounds)?;
    let count = old
        .checked_add(u32::from(sample.weight))
        .ok_or(EntropyError::CountOverflow)?;
    let extra = extra_bits
        .checked_add(u64::from(sample.weight) * u64::from(sample.extra_bits))
        .ok_or(EntropyError::ExtraBitsOverflow)?;
    counts[sample.symbol] = count;
    *extra_bits = extra;
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct RareCount {
    mass: i64,
    entropy: f64,
}

/// Source count-frequency state. Learner-owned count columns remain the sole
/// count storage; resets reuse private scratch across predictors and nodes.
#[derive(Default)]
pub(crate) struct SourceClippedEntropy {
    sum: f64,
    rare: Vec<RareCount>,
    sliding: bool,
    cursor: usize,
    tail: RareCount,
}

impl SourceClippedEntropy {
    pub(crate) fn reset(
        &mut self,
        counts: &[u32],
        maximum_total: u32,
        sliding: bool,
    ) -> crate::Result<()> {
        let size = if maximum_total > ANS_TAB_SIZE {
            ((maximum_total - 1) / ANS_TAB_SIZE + 1) as usize
        } else {
            0
        };
        self.rare
            .try_reserve(size.saturating_sub(self.rare.len()))
            .map_err(|_| crate::Error::Limit("MA clipped entropy scratch allocation"))?;
        self.rare.clear();
        self.rare.resize(size, RareCount::default());
        self.sum = 0.0;
        self.sliding = sliding;
        self.cursor = 0;
        self.tail = RareCount::default();
        for &count in counts {
            self.replace(0, count);
        }
        Ok(())
    }

    pub(crate) fn replace(&mut self, before: u32, after: u32) {
        if before == after {
            return;
        }
        let table = CountEntropyTable::shared();
        let old = table.get(before);
        let new = table.get(after);
        self.sum += new - old;
        self.update(before, -i64::from(before), -old);
        self.update(after, i64::from(after), new);
    }

    fn update(&mut self, count: u32, mass: i64, entropy: f64) {
        let mut index = count as usize;
        if index == 0 {
            return;
        }
        if self.sliding {
            if index >= self.rare.len() {
                return;
            }
            self.rare[index].mass += mass;
            self.rare[index].entropy += entropy;
            if index <= self.cursor {
                self.tail.mass += mass;
                self.tail.entropy += entropy;
            }
        } else {
            while index < self.rare.len() {
                self.rare[index].mass += mass;
                self.rare[index].entropy += entropy;
                index += index & index.wrapping_neg();
            }
        }
    }

    pub(crate) fn bits(&mut self, total: u32) -> f32 {
        if total == 0 {
            return 0.0;
        }
        let mut clipped = RareCount::default();
        if !self.rare.is_empty() {
            let mut index = (((total - 1) / ANS_TAB_SIZE) as usize).min(self.rare.len() - 1);
            if self.sliding {
                while self.cursor < index {
                    self.cursor += 1;
                    self.tail.mass += self.rare[self.cursor].mass;
                    self.tail.entropy += self.rare[self.cursor].entropy;
                }
                while self.cursor > index {
                    self.tail.mass -= self.rare[self.cursor].mass;
                    self.tail.entropy -= self.rare[self.cursor].entropy;
                    self.cursor -= 1;
                }
                clipped = self.tail;
            } else {
                while index != 0 {
                    clipped.mass += self.rare[index].mass;
                    clipped.entropy += self.rare[index].entropy;
                    index &= index - 1;
                }
            }
        }
        let count_entropy = CountEntropyTable::shared().get(total);
        let log_total = count_entropy / f64::from(total);
        (count_entropy - self.sum
            + clipped.entropy
            + clipped.mass as f64 * (f64::from(ANS_LOG_TAB_SIZE) - log_total))
            .max(0.0) as f32
    }
}

/// Safe, owning equivalent of `MaClippedEntropy`'s Fenwick-tree mode.
/// Owning the counts prevents invalid before/after pairs and stale caller totals.
/// `maximum_total` bounds both valid updates and rare-count scratch allocation.
/// This does not include token extra bits, tree signaling, or predictor cost.
#[derive(Clone, Debug)]
pub struct ClippedEntropy {
    counts: Vec<u32>,
    total: u32,
    maximum_total: u32,
    sum: f64,
    rare: Vec<RareCount>,
}

impl ClippedEntropy {
    pub fn new(counts: &[u32], maximum_total: u32) -> Result<Self, EntropyError> {
        let total = counts.iter().try_fold(0u32, |total, &count| {
            total.checked_add(count).ok_or(EntropyError::CountOverflow)
        })?;
        if total > maximum_total {
            return Err(EntropyError::MaximumTotalExceeded);
        }
        let rare_len = if maximum_total > ANS_TAB_SIZE {
            ((maximum_total - 1) / ANS_TAB_SIZE + 1) as usize
        } else {
            0
        };
        let mut result = Self {
            counts: counts.to_vec(),
            total,
            maximum_total,
            sum: CountEntropyTable::shared().sum(counts),
            rare: vec![RareCount::default(); rare_len],
        };
        if rare_len > 1 {
            let table = CountEntropyTable::shared();
            for &count in counts {
                if count != 0 && (count as usize) < rare_len {
                    result.update_rare(count, i64::from(count), table.get(count));
                }
            }
        }
        Ok(result)
    }

    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn maximum_total(&self) -> u32 {
        self.maximum_total
    }

    /// Replace one count, checking all bounds before changing any state.
    pub fn replace(&mut self, symbol: usize, after: u32) -> Result<(), EntropyError> {
        let total = self.checked_replacement(symbol, after)?;
        self.replace_validated(symbol, after, total);
        Ok(())
    }

    /// Add a full sample multiplicity, never one per distinct sample record.
    pub fn add_weight(&mut self, symbol: usize, weight: u32) -> Result<(), EntropyError> {
        let before = *self
            .counts
            .get(symbol)
            .ok_or(EntropyError::SymbolOutOfBounds)?;
        let after = before
            .checked_add(weight)
            .ok_or(EntropyError::CountOverflow)?;
        self.replace(symbol, after)
    }

    pub fn remove_weight(&mut self, symbol: usize, weight: u32) -> Result<(), EntropyError> {
        let before = *self
            .counts
            .get(symbol)
            .ok_or(EntropyError::SymbolOutOfBounds)?;
        let after = before
            .checked_sub(weight)
            .ok_or(EntropyError::CountUnderflow)?;
        self.replace(symbol, after)
    }

    /// Transfer weighted counts during a split sweep. Either both histograms
    /// change, or neither changes. The same symbol index is used on each side.
    pub fn move_weight_to(
        &mut self,
        destination: &mut Self,
        symbol: usize,
        weight: u32,
    ) -> Result<(), EntropyError> {
        let source = *self
            .counts
            .get(symbol)
            .ok_or(EntropyError::SymbolOutOfBounds)?;
        let target = *destination
            .counts
            .get(symbol)
            .ok_or(EntropyError::SymbolOutOfBounds)?;
        let source_after = source
            .checked_sub(weight)
            .ok_or(EntropyError::CountUnderflow)?;
        let target_after = target
            .checked_add(weight)
            .ok_or(EntropyError::CountOverflow)?;
        let source_total = self.checked_replacement(symbol, source_after)?;
        let target_total = destination.checked_replacement(symbol, target_after)?;
        if weight == 0 {
            return Ok(());
        }
        let update = CountEntropyTable::shared().replace_pair(
            [source, target],
            [source_after, target_after],
            [self.sum, destination.sum],
        );
        self.sum = update.sums[0];
        destination.sum = update.sums[1];
        self.record_replacement(
            symbol,
            source_after,
            source_total,
            update.before[0],
            update.after[0],
        );
        destination.record_replacement(
            symbol,
            target_after,
            target_total,
            update.before[1],
            update.after[1],
        );
        Ok(())
    }

    /// Incremental form of the exact probability-floor model, rounded to f32
    /// like the source kernel. The strict rare threshold is `(total - 1) / 4096`.
    pub fn bits(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let mut clipped = RareCount::default();
        if !self.rare.is_empty() {
            let mut index = ((self.total - 1) / ANS_TAB_SIZE) as usize;
            while index != 0 {
                clipped.mass += self.rare[index].mass;
                clipped.entropy += self.rare[index].entropy;
                index &= index - 1;
            }
        }
        let total_entropy = CountEntropyTable::shared().get(self.total);
        let log_total = total_entropy / f64::from(self.total);
        let bits = total_entropy - self.sum
            + clipped.entropy
            + clipped.mass as f64 * (f64::from(ANS_LOG_TAB_SIZE) - log_total);
        bits.max(0.0) as f32
    }

    fn checked_replacement(&self, symbol: usize, after: u32) -> Result<u32, EntropyError> {
        let before = *self
            .counts
            .get(symbol)
            .ok_or(EntropyError::SymbolOutOfBounds)?;
        let total = (self.total - before)
            .checked_add(after)
            .ok_or(EntropyError::CountOverflow)?;
        if total > self.maximum_total {
            return Err(EntropyError::MaximumTotalExceeded);
        }
        Ok(total)
    }

    fn replace_validated(&mut self, symbol: usize, after: u32, total: u32) {
        let before = self.counts[symbol];
        if before == after {
            return;
        }
        let table = CountEntropyTable::shared();
        let old_entropy = table.get(before);
        let new_entropy = table.get(after);
        self.sum += new_entropy - old_entropy;
        self.record_replacement(symbol, after, total, old_entropy, new_entropy);
    }

    fn record_replacement(
        &mut self,
        symbol: usize,
        after: u32,
        total: u32,
        old_entropy: f64,
        new_entropy: f64,
    ) {
        let before = self.counts[symbol];
        self.update_rare(before, -i64::from(before), -old_entropy);
        self.update_rare(after, i64::from(after), new_entropy);
        self.counts[symbol] = after;
        self.total = total;
    }

    fn update_rare(&mut self, count: u32, mass: i64, entropy: f64) {
        if count == 0 {
            return;
        }
        let mut index = count as usize;
        while index < self.rare.len() {
            self.rare[index].mass += mass;
            self.rare[index].entropy += entropy;
            // The tree has at most 2^20 entries, so advancing cannot overflow.
            index += index & index.wrapping_neg();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_clipped_sliding_and_fenwick_reuse_strict_tail_boundaries() {
        let mut model = SourceClippedEntropy::default();
        for sliding in [false, true, false] {
            for maximum in [4095u32, 4096, 4097, 8192, 8193, 65536, 65537, 131072] {
                let mut counts = [1, 2, maximum - 3];
                model.reset(&counts, maximum, sliding).unwrap();
                for total in [
                    maximum,
                    maximum - 1,
                    maximum - 2,
                    maximum - 3,
                    maximum - 2,
                    maximum,
                ] {
                    let next = total - 3;
                    model.replace(counts[2], next);
                    counts[2] = next;
                    let expected = direct_bits(&counts);
                    assert!((f64::from(model.bits(total)) - expected).abs() < 0.01);
                }
                // Replacements on either side of an already moved cursor update
                // the retained tail; mode/reset changes must not reuse that tail.
                model.replace(1, 2);
                model.replace(counts[2], counts[2] - 1);
                counts[0] = 2;
                counts[2] -= 1;
                assert!((f64::from(model.bits(maximum)) - direct_bits(&counts)).abs() < 0.01);
            }
            model.reset(&[], 0, sliding).unwrap();
            assert_eq!(model.bits(0), 0.0);
            assert_eq!(model.cursor, 0);
            assert_eq!(model.tail.mass, 0);
        }
    }

    fn direct_bits(counts: &[u32]) -> f64 {
        let total: f64 = counts.iter().map(|&n| f64::from(n)).sum();
        counts
            .iter()
            .filter(|&&n| n != 0)
            .map(|&n| f64::from(n) * (total / f64::from(n)).log2().min(12.0))
            .sum()
    }

    fn assert_model(histogram: &ClippedEntropy) {
        let expected = direct_bits(histogram.counts());
        let actual = f64::from(histogram.bits());
        let tolerance = 0.0001 + expected * 2.0 * f64::from(f32::EPSILON);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
        assert_eq!(histogram.total(), histogram.counts().iter().sum::<u32>());
    }

    #[test]
    fn native_split_updates_match_scalar_replacements_across_cache_boundary() {
        let counts = [65537, 1, 2, 65536, 0, 65535, 4097, 19, 120_000];
        let total: u32 = counts.iter().sum();
        let mut above = ClippedEntropy::new(&counts, total).unwrap();
        let mut below = ClippedEntropy::new(&[0; 9], total).unwrap();
        let mut scalar_above = above.clone();
        scalar_above.sum = counts
            .iter()
            .map(|&count| {
                if count == 0 {
                    0.0
                } else {
                    f64::from(count) * f64::from(count).log2()
                }
            })
            .sum();
        let mut scalar_below = below.clone();
        for direction in [true, false] {
            let (source, target, scalar_source, scalar_target) = if direction {
                (&mut above, &mut below, &mut scalar_above, &mut scalar_below)
            } else {
                (&mut below, &mut above, &mut scalar_below, &mut scalar_above)
            };
            for symbol in 0..counts.len() {
                while source.counts()[symbol] != 0 {
                    let weight = source.counts()[symbol].min(997);
                    source.move_weight_to(target, symbol, weight).unwrap();
                    scalar_source
                        .replace(symbol, scalar_source.counts()[symbol] - weight)
                        .unwrap();
                    scalar_target
                        .replace(symbol, scalar_target.counts()[symbol] + weight)
                        .unwrap();
                    for (native, scalar) in
                        [(&*source, &*scalar_source), (&*target, &*scalar_target)]
                    {
                        assert_eq!(native.counts(), scalar.counts());
                        assert_model(native);
                        assert_model(scalar);
                        let tolerance = 0.0001 + scalar.bits().abs() * 2.0 * f32::EPSILON;
                        assert!((native.bits() - scalar.bits()).abs() <= tolerance);
                    }
                }
            }
        }
        assert_eq!(above.counts(), counts);
        assert_eq!(below.total(), 0);
    }

    #[test]
    fn native_split_zero_weight_still_validates_both_symbols() {
        let mut source = ClippedEntropy::new(&[7, 0], 10).unwrap();
        let mut destination = ClippedEntropy::new(&[0, 2], 10).unwrap();
        let costs = (source.bits(), destination.bits());
        source.move_weight_to(&mut destination, 0, 0).unwrap();
        assert_eq!(source.counts(), [7, 0]);
        assert_eq!(destination.counts(), [0, 2]);
        assert_eq!((source.bits(), destination.bits()), costs);
        let mut empty = ClippedEntropy::new(&[], 0).unwrap();
        assert_eq!(
            source.move_weight_to(&mut empty, 0, 0),
            Err(EntropyError::SymbolOutOfBounds)
        );
        assert_eq!(
            source.move_weight_to(&mut destination, 2, 0),
            Err(EntropyError::SymbolOutOfBounds)
        );
        assert_eq!(source.counts(), [7, 0]);
        assert_eq!((source.bits(), destination.bits()), costs);
    }

    #[test]
    fn empty_single_symbol_and_zero_counts() {
        for counts in [&[][..], &[0, 0], &[17], &[0, 8000, 0]] {
            let histogram = ClippedEntropy::new(counts, 8000).unwrap();
            assert_eq!(histogram.bits(), 0.0);
            assert_model(&histogram);
        }
        let mut empty = ClippedEntropy::new(&[], 0).unwrap();
        assert_eq!(empty.add_weight(0, 0), Err(EntropyError::SymbolOutOfBounds));
        assert_eq!(
            empty.replace(usize::MAX, 0),
            Err(EntropyError::SymbolOutOfBounds)
        );
    }

    #[test]
    fn clipping_thresholds_match_direct_model() {
        for total in [
            4095, 4096, 4097, 8191, 8192, 8193, 65535, 65536, 65537, 1_000_000,
        ] {
            for rare in [1, 2, 3, 15, 16, 17, 100] {
                let histogram = ClippedEntropy::new(&[rare, 0, total - rare], total).unwrap();
                assert_model(&histogram);
            }
        }
        let histogram = ClippedEntropy::new(&[1, 999_999], 1_000_000).unwrap();
        assert!(histogram.bits() < 14.0);
        assert_model(&histogram);
    }

    #[test]
    fn split_sweep_preserves_weight_and_entropy() {
        let counts = [65535, 1, 2, 31, 8193, 0, 90_000];
        let total = counts.iter().sum();
        let mut above = ClippedEntropy::new(&counts, total).unwrap();
        let mut below = ClippedEntropy::new(&[0; 7], total).unwrap();
        for (symbol, &count) in counts.iter().enumerate() {
            let mut remaining = count;
            while remaining != 0 {
                let weight = remaining.min(997);
                above.move_weight_to(&mut below, symbol, weight).unwrap();
                remaining -= weight;
                assert_eq!(above.total() + below.total(), total);
                assert_model(&above);
                assert_model(&below);
            }
        }
        assert_eq!(below.counts(), counts);
        assert_eq!(above.bits(), 0.0);
        for (symbol, &count) in counts.iter().enumerate().rev() {
            below.move_weight_to(&mut above, symbol, count).unwrap();
            assert_model(&above);
            assert_model(&below);
        }
        assert_eq!(above.counts(), counts);
    }

    #[test]
    fn arbitrary_updates_follow_direct_recomputation() {
        let mut histogram = ClippedEntropy::new(&[0; 37], 1_000_000).unwrap();
        let mut random = 19u32;
        for step in 0..5000 {
            random = random.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let symbol = random as usize % 37;
            let count = if step % 3 == 0 {
                random % 7
            } else {
                (random >> 9) % 20_000
            };
            histogram.replace(symbol, count).unwrap();
            assert_model(&histogram);
        }
    }

    #[test]
    fn weighted_samples_equal_literal_expansion_including_extra_bits() {
        let samples = [
            WeightedSample {
                symbol: 0,
                weight: 0,
                extra_bits: 255,
            },
            WeightedSample {
                symbol: 1,
                weight: 1,
                extra_bits: 2,
            },
            WeightedSample {
                symbol: 2,
                weight: u16::MAX,
                extra_bits: 31,
            },
            WeightedSample {
                symbol: 1,
                weight: 123,
                extra_bits: 7,
            },
        ];
        let mut counts = [0; 3];
        let mut extra = 0;
        let mut expanded = [0; 3];
        let mut expanded_extra = 0u64;
        let mut incremental = ClippedEntropy::new(&[0; 3], 100_000).unwrap();
        for sample in samples {
            add_weighted_sample(&mut counts, &mut extra, sample).unwrap();
            incremental
                .add_weight(sample.symbol, u32::from(sample.weight))
                .unwrap();
            for _ in 0..sample.weight {
                expanded[sample.symbol] += 1;
                expanded_extra += u64::from(sample.extra_bits);
            }
        }
        assert_eq!(counts, expanded);
        assert_eq!(extra, expanded_extra);
        assert_eq!(incremental.counts(), counts);
        assert_eq!(incremental.total(), 65_659);
        assert_model(&incremental);
        incremental.remove_weight(2, u32::from(u16::MAX)).unwrap();
        assert_eq!(incremental.counts(), [0, 124, 0]);
        assert_model(&incremental);
    }

    #[test]
    fn checked_errors_do_not_mutate_histograms() {
        assert!(matches!(
            ClippedEntropy::new(&[u32::MAX, 1], u32::MAX),
            Err(EntropyError::CountOverflow)
        ));
        assert!(matches!(
            ClippedEntropy::new(&[2], 1),
            Err(EntropyError::MaximumTotalExceeded)
        ));
        let mut source = ClippedEntropy::new(&[8, 2], 10).unwrap();
        let mut destination = ClippedEntropy::new(&[0, 1], 1).unwrap();
        let old_source = source.bits();
        let old_target = destination.bits();
        assert_eq!(
            source.move_weight_to(&mut destination, 0, 1),
            Err(EntropyError::MaximumTotalExceeded)
        );
        assert_eq!(
            source.move_weight_to(&mut destination, 0, 9),
            Err(EntropyError::CountUnderflow)
        );
        assert_eq!(
            source.replace(usize::MAX, 0),
            Err(EntropyError::SymbolOutOfBounds)
        );
        assert_eq!(
            source.add_weight(0, u32::MAX),
            Err(EntropyError::CountOverflow)
        );
        assert_eq!(
            source.replace(0, u32::MAX),
            Err(EntropyError::CountOverflow)
        );
        assert_eq!(
            source.remove_weight(1, 3),
            Err(EntropyError::CountUnderflow)
        );
        assert_eq!(
            source.add_weight(1, 1),
            Err(EntropyError::MaximumTotalExceeded)
        );
        assert_eq!(source.counts(), [8, 2]);
        assert_eq!(destination.counts(), [0, 1]);
        assert_eq!(source.bits(), old_source);
        assert_eq!(destination.bits(), old_target);
        assert_model(&source);
        assert_model(&destination);
    }

    #[test]
    fn weighted_count_and_extra_bit_overflow_are_atomic() {
        let sample = WeightedSample {
            symbol: 0,
            weight: 1,
            extra_bits: 1,
        };
        let mut counts = [u32::MAX];
        let mut extra = 17;
        assert_eq!(
            add_weighted_sample(&mut counts, &mut extra, sample),
            Err(EntropyError::CountOverflow)
        );
        assert_eq!(counts, [u32::MAX]);
        assert_eq!(extra, 17);
        counts[0] = 5;
        extra = u64::MAX;
        assert_eq!(
            add_weighted_sample(&mut counts, &mut extra, sample),
            Err(EntropyError::ExtraBitsOverflow)
        );
        assert_eq!(counts, [5]);
        assert_eq!(extra, u64::MAX);
        assert_eq!(
            add_weighted_sample(&mut [], &mut extra, sample),
            Err(EntropyError::SymbolOutOfBounds)
        );
    }

    #[test]
    fn maximum_u32_total_does_not_wrap_rare_tree_or_mass() {
        let mut histogram = ClippedEntropy::new(&[u32::MAX - 1, 1], u32::MAX).unwrap();
        assert_model(&histogram);
        assert_eq!(histogram.add_weight(1, 1), Err(EntropyError::CountOverflow));
        histogram.replace(0, u32::MAX - 2).unwrap();
        histogram.add_weight(1, 1).unwrap();
        assert_eq!(histogram.total(), u32::MAX);
        assert_model(&histogram);
    }
}
