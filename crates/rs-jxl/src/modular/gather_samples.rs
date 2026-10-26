// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; the complete upstream notice is in docs/PORTING_KERNELS.md.

//! Gather-owned TreeSamples storage, from enc_ma.cc::{PrepareForSamples,
//! InitTable, AddToTableAndMerge, AddWeightedToTableAndMerge, AppendSamples}
//! at b866b3defec13494bfbaf705db93581cda118e37. No row keys or raw dynamic
//! properties/residuals survive insertion. Production transfers prepared
//! columns directly to the learner. PreparedSamples::into_samples is a
//! test-only compatibility oracle, not a second production collector.

use super::super::ma_train::PropertyQuantization;
#[cfg(test)]
use super::super::ma_train::Sample;
use super::super::predictor::Predictor;
use super::super::tree::pack_signed;
#[cfg(test)]
use crate::entropy::hybrid_uint::HybridUintConfig;
use crate::{Error, Result};

const UNUSED: u32 = u32::MAX;
const A: u64 = 0x1e35_a7bd;
const B: u64 = 0x1e35_a7bd_1e35_a7bd;
const MAX_CACHED_HASHES: usize = 1 << 20;
const MAX_CONCAT_PAYLOAD_BYTES: usize = 64 << 20;

pub(crate) enum GatherColumn {
    Packed(Vec<u8>),
    Static(Vec<u32>),
}

impl GatherColumn {
    fn capacity(&self) -> usize {
        match self {
            Self::Packed(values) => values.capacity(),
            Self::Static(values) => values.capacity(),
        }
    }

    fn value(&self, row: usize) -> u32 {
        match self {
            Self::Packed(values) => u32::from(values[row]),
            Self::Static(values) => values[row],
        }
    }

    fn reserve(&mut self, extra: usize) -> Result<()> {
        match self {
            Self::Packed(values) => reserve(values, extra),
            Self::Static(values) => reserve(values, extra),
        }
    }

    fn push(&mut self, value: u32) {
        match self {
            Self::Packed(values) => values.push(value as u8),
            Self::Static(values) => values.push(value),
        }
    }
}

/// Ownership-transfer interface for ma_train::train_prepared.
/// Residual elements are [token, extra-bit-count], never extra-bit payloads.
/// All columns have counts.len() rows; quantization/property order is shared.
pub(crate) struct PreparedSamples {
    pub(crate) predictors: Vec<Predictor>,
    pub(crate) quantization: PropertyQuantization,
    pub(crate) properties: Vec<GatherColumn>,
    pub(crate) residuals: Vec<Vec<[u8; 2]>>,
    pub(crate) counts: Vec<u16>,
    pub(crate) total: u64,
    // Raw IDs are retained only for test comparisons against the public API.
    #[cfg(test)]
    static_ids: Vec<[i32; 2]>,
}

impl PreparedSamples {
    pub(crate) fn len(&self) -> usize {
        self.counts.len()
    }

    fn validate_concat(&self) -> Result<usize> {
        let ids = &self.quantization.properties;
        if self.predictors.is_empty()
            || self.predictors.len() > 14
            || ids.is_empty()
            || ids.len() >= 64
            || ids.len() != self.quantization.thresholds.len()
            || self.properties.len() != ids.len()
            || self.residuals.len() != self.predictors.len()
            || ids.iter().enumerate().any(|(p, &id)| {
                id >= 256
                    || ids[..p].contains(&id)
                    || self.quantization.thresholds[p]
                        .windows(2)
                        .any(|w| w[0] >= w[1])
                    || (id >= 2 && self.quantization.thresholds[p].len() >= 256)
            })
        {
            return Err(Error::InvalidInput("invalid MA concat schema".into()));
        }
        let rows = self.len();
        if rows > UNUSED as usize {
            return Err(Error::Limit("MA gather index exceeds source u32"));
        }
        if self.total > i32::MAX as u64 {
            return Err(Error::Limit("MA sample weight exceeds source i32::MAX"));
        }
        let mut total = 0u64;
        for &count in &self.counts {
            if count == 0 {
                return Err(Error::InvalidInput("zero MA concat multiplicity".into()));
            }
            total = total
                .checked_add(u64::from(count))
                .filter(|&n| n <= i32::MAX as u64)
                .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
        }
        if total != self.total {
            return Err(Error::InvalidInput("MA concat total mismatch".into()));
        }
        let mut row_bytes = self
            .residuals
            .len()
            .checked_mul(std::mem::size_of::<[u8; 2]>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u16>()))
            .ok_or(Error::Limit("MA concat payload overflow"))?;
        for (&id, column) in ids.iter().zip(&self.properties) {
            let bytes = match column {
                GatherColumn::Packed(values) if id >= 2 && values.len() == rows => 1,
                GatherColumn::Static(values) if id < 2 && values.len() == rows => 4,
                _ => {
                    return Err(Error::InvalidInput(
                        "invalid MA concat property column".into(),
                    ));
                }
            };
            row_bytes = row_bytes
                .checked_add(bytes)
                .ok_or(Error::Limit("MA concat payload overflow"))?;
        }
        if self.residuals.iter().any(|column| column.len() != rows) {
            return Err(Error::InvalidInput(
                "invalid MA concat residual column".into(),
            ));
        }
        #[cfg(test)]
        if self.static_ids.len() != rows {
            return Err(Error::InvalidInput("invalid MA concat static IDs".into()));
        }
        Ok(row_bytes)
    }

    /// Preserve the complete quantized objective for the existing train API.
    /// Canonical residuals carry the same token/nbits, not original pixels.
    /// Never use this adapter to validate raw multiplier/constant evidence.
    #[cfg(test)]
    pub(super) fn into_samples(self) -> Result<Vec<Sample>> {
        let mut output = Vec::new();
        reserve(&mut output, self.len())?;
        let width = self
            .quantization
            .properties
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .max(1)
            + 1;
        for (row, &count) in self.counts.iter().enumerate() {
            let mut properties = Vec::new();
            reserve(&mut properties, width)?;
            properties.resize(width, 0);
            properties[..2].copy_from_slice(&self.static_ids[row]);
            for (p, &id) in self.quantization.properties.iter().enumerate() {
                if id < 2 {
                    continue;
                }
                let bin = self.properties[p].value(row) as usize;
                let thresholds = &self.quantization.thresholds[p];
                properties[id] = if bin == 0 {
                    thresholds.first().copied().unwrap_or(0)
                } else {
                    thresholds[bin - 1]
                        .checked_add(1)
                        .ok_or(Error::Limit("MA canonical property overflow"))?
                };
            }
            let mut residuals = Vec::new();
            reserve(&mut residuals, self.residuals.len())?;
            for column in &self.residuals {
                let [token, nbits] = column[row];
                let token = u32::from(token);
                let packed = if token < 16 {
                    token
                } else {
                    let exponent = (token - 16) / 8 + 4;
                    debug_assert_eq!(u32::from(nbits), exponent - 3);
                    (1u32 << exponent) | (((token - 16) & 4) << (exponent - 3)) | (token & 3)
                };
                residuals.push(((packed >> 1) as i32) ^ -((packed & 1) as i32));
            }
            output.push(Sample {
                properties,
                residuals,
                weight: u32::from(count),
            });
        }
        Ok(output)
    }
}

fn reserve<T>(values: &mut Vec<T>, extra: usize) -> Result<()> {
    let required = values
        .len()
        .checked_add(extra)
        .ok_or(Error::Limit("MA gather column capacity overflow"))?;
    if required > values.capacity() {
        let grown = values
            .capacity()
            .checked_add(values.capacity() / 2)
            .unwrap_or(required);
        values
            .try_reserve_exact(required.max(grown) - values.len())
            .map_err(|_| Error::Limit("MA gather column allocation"))?;
    }
    Ok(())
}

fn table_slots(records: usize) -> Result<usize> {
    if records == 0 {
        return Ok(0);
    }
    if records > UNUSED as usize {
        return Err(Error::Limit("MA gather index exceeds source u32"));
    }
    records
        .checked_mul(3)
        .map(|n| n / 2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(Error::Limit("MA gather table capacity overflow"))
}

fn concat_allowed(rows: usize, total: u64, row_bytes: usize, payload_limit: usize) -> bool {
    // ceil(99 * total / 100), without multiplying a potentially large total.
    rows as u64 >= total - total / 100
        && rows
            .checked_mul(row_bytes)
            .is_some_and(|bytes| bytes <= payload_limit.min(MAX_CONCAT_PAYLOAD_BYTES))
}

/// Source double expression converted to size_t only after adding 64.
pub(super) fn expected_samples(fraction: f64, width: usize, rows: usize) -> Result<usize> {
    let value = fraction * rows as f64 * width as f64 + 64.0;
    if !value.is_finite() || value < 0.0 || value >= usize::MAX as f64 {
        return Err(Error::Limit("MA gather expected sample overflow"));
    }
    Ok(value as usize)
}

fn hashes(
    residuals: impl Iterator<Item = [u8; 2]> + Clone,
    properties: impl Iterator<Item = u32>,
) -> [u64; 2] {
    let (mut first, mut second) = (A, B);
    for [token, nbits] in residuals.clone() {
        first = first
            .wrapping_mul(A)
            .wrapping_add(u64::from(token))
            .wrapping_mul(A)
            .wrapping_add(u64::from(nbits));
    }
    for value in properties {
        first = first.wrapping_mul(A).wrapping_add(u64::from(value));
        second = second.wrapping_mul(B) ^ u64::from(value);
    }
    for [token, nbits] in residuals {
        second = (second.wrapping_mul(B) ^ u64::from(token)).wrapping_mul(B) ^ u64::from(nbits);
    }
    [first, second]
}

/// Only the token and bit count of HybridUintConfig(4, 1, 2) are stored.
#[inline]
fn residual_token(value: i32) -> [u8; 2] {
    let packed = pack_signed(value);
    if packed < 16 {
        [packed as u8, 0]
    } else {
        // nbits = floor(log2(packed)) - 3, in 1..=28. The base token
        // 16 + (exponent - 4) * 8 is (nbits + 1) * 8; its low three
        // bits hold the next-highest bit and the two low bits of packed.
        let nbits = 28 - packed.leading_zeros();
        let token = ((nbits + 1) << 3) | ((packed >> nbits) & 4) | (packed & 3);
        [token as u8, nbits as u8]
    }
}

pub(super) struct SourceSampleCollector {
    prepared: PreparedSamples,
    table: Vec<u32>,
    property_order: Vec<usize>,
    property_luts: Vec<Vec<u8>>,
    required_properties: usize,
    // The schema is fixed; channel/group IDs usually repeat for a whole stripe.
    static_bins: [Option<(i32, u32)>; 2],
    // Keys do not change when weights merge; saturated rows still need hashes
    // during append. Only this prefix is cached, never the learner's columns.
    cached_hashes: Vec<[u64; 2]>,
    hash_limit: usize,
    reserved_rows: usize,
}

impl SourceSampleCollector {
    pub(super) fn new(
        predictors: &[Predictor],
        quantization: &PropertyQuantization,
        cache_hashes: bool,
    ) -> Result<Self> {
        let ids = &quantization.properties;
        if predictors.is_empty()
            || predictors.len() > 14
            || ids.is_empty()
            || ids.len() >= 64
            || ids.len() != quantization.thresholds.len()
            || ids.iter().enumerate().any(|(p, &id)| {
                id >= 256
                    || ids[..p].contains(&id)
                    || quantization.thresholds[p].windows(2).any(|w| w[0] >= w[1])
                    || (id >= 2 && quantization.thresholds[p].len() >= 256)
            })
        {
            return Err(Error::InvalidInput(
                "invalid MA prepared gather schema".into(),
            ));
        }
        let property_order = [0, 1]
            .into_iter()
            .filter_map(|id| ids.iter().position(|&p| p == id))
            .chain(
                ids.iter()
                    .enumerate()
                    .filter(|&(_, &id)| id >= 2)
                    .map(|(p, _)| p),
            )
            .collect();
        // Dynamic properties have only 1023 possible clamped values. Build
        // their exact bins once; empty tables retain the wide/static search.
        let mut property_luts = Vec::new();
        property_luts
            .try_reserve_exact(ids.len())
            .map_err(|_| Error::Limit("MA gather quantization allocation"))?;
        for (&id, thresholds) in ids.iter().zip(&quantization.thresholds) {
            let mut lut = Vec::new();
            if id >= 2 && thresholds.len() <= usize::from(u8::MAX) {
                lut.try_reserve_exact(1023)
                    .map_err(|_| Error::Limit("MA gather quantization allocation"))?;
                let mut bin = 0u8;
                for value in -511..=511 {
                    while usize::from(bin) < thresholds.len()
                        && value > thresholds[usize::from(bin)]
                    {
                        bin += 1;
                    }
                    lut.push(bin);
                }
            }
            property_luts.push(lut);
        }
        Ok(Self {
            prepared: PreparedSamples {
                predictors: predictors.to_vec(),
                quantization: quantization.clone(),
                properties: ids
                    .iter()
                    .map(|&id| {
                        if id < 2 {
                            GatherColumn::Static(Vec::new())
                        } else {
                            GatherColumn::Packed(Vec::new())
                        }
                    })
                    .collect(),
                residuals: vec![Vec::new(); predictors.len()],
                counts: Vec::new(),
                #[cfg(test)]
                static_ids: Vec::new(),
                total: 0,
            },
            table: Vec::new(),
            property_order,
            property_luts,
            required_properties: ids.iter().copied().max().unwrap_or(1).max(1) + 1,
            static_bins: [None; 2],
            cached_hashes: Vec::new(),
            hash_limit: if cache_hashes { MAX_CACHED_HASHES } else { 0 },
            reserved_rows: 0,
        })
    }

    fn reserve_hashes(&mut self, records: usize) -> Result<()> {
        let required = records.min(self.hash_limit);
        let capacity = self.cached_hashes.capacity();
        if required > capacity {
            let target = required.max(capacity + capacity / 2).min(self.hash_limit);
            self.cached_hashes
                .try_reserve_exact(target - self.cached_hashes.len())
                .map_err(|_| Error::Limit("MA gather hash allocation"))?;
        }
        Ok(())
    }

    fn reserve_columns(&mut self, extra: usize) -> Result<()> {
        for column in &mut self.prepared.properties {
            column.reserve(extra)?;
        }
        for column in &mut self.prepared.residuals {
            reserve(column, extra)?;
        }
        reserve(&mut self.prepared.counts, extra)?;
        self.reserve_hashes(self.prepared.len().saturating_add(extra))?;
        self.reserved_rows = self
            .prepared
            .properties
            .iter()
            .map(GatherColumn::capacity)
            .chain(self.prepared.residuals.iter().map(Vec::capacity))
            .chain([self.prepared.counts.capacity()])
            .min()
            .unwrap_or(0);
        // Only the cached prefix needs hash capacity. At the cap, subsequent
        // records use the original hashing/comparison path without allocation.
        if self.cached_hashes.capacity() < self.hash_limit {
            self.reserved_rows = self.reserved_rows.min(self.cached_hashes.capacity());
        }
        #[cfg(test)]
        {
            reserve(&mut self.prepared.static_ids, extra)?;
            self.reserved_rows = self.reserved_rows.min(self.prepared.static_ids.capacity());
        }
        Ok(())
    }

    fn row_hash(&self, row: usize) -> [u64; 2] {
        self.cached_hashes.get(row).copied().unwrap_or_else(|| {
            hashes(
                self.prepared.residuals.iter().map(|column| column[row]),
                self.property_order
                    .iter()
                    .map(|&p| self.prepared.properties[p].value(row)),
            )
        })
    }

    /// Called once per channel/stripe, or once for a complete appended wave.
    /// Equal size retains slots; every changed size rebuilds from empty slots.
    pub(super) fn prepare(&mut self, extra: usize) -> Result<()> {
        let records = self
            .prepared
            .len()
            .checked_add(extra)
            .ok_or(Error::Limit("MA gather sample capacity overflow"))?;
        let slots = table_slots(records)?;
        self.reserve_columns(extra)?;
        if slots == self.table.len() {
            return Ok(());
        }
        let mut table = Vec::new();
        table
            .try_reserve_exact(slots)
            .map_err(|_| Error::Limit("MA gather table allocation"))?;
        table.resize(slots, UNUSED);
        self.table = table;
        for row in 0..self.prepared.len() {
            if self.prepared.counts[row] != u16::MAX {
                let hash = self.row_hash(row);
                self.register(row, self.positions(hash));
            }
        }
        Ok(())
    }

    fn positions(&self, hash: [u64; 2]) -> [usize; 2] {
        hash.map(|h| (h >> 16) as usize & (self.table.len() - 1))
    }

    fn register(&mut self, row: usize, positions: [usize; 2]) {
        for position in positions {
            if self.table[position] == UNUSED {
                self.table[position] = row as u32;
                break;
            }
        }
    }

    fn insert(
        &mut self,
        properties: &[u32],
        residuals: &[[u8; 2]],
        ids: [i32; 2],
        weight: u32,
    ) -> Result<()> {
        let hash = hashes(
            residuals.iter().copied(),
            self.property_order.iter().map(|&p| properties[p]),
        );
        self.insert_hashed(properties, residuals, ids, weight, hash)
    }

    fn insert_hashed(
        &mut self,
        properties: &[u32],
        residuals: &[[u8; 2]],
        ids: [i32; 2],
        mut weight: u32,
        hash: [u64; 2],
    ) -> Result<()> {
        #[cfg(not(test))]
        let _ = ids;
        let positions = self.positions(hash);
        while weight != 0 {
            let mut remaining = weight.min(u32::from(u16::MAX)) as u16;
            weight -= u32::from(remaining);
            for position in positions {
                let previous = self.table[position];
                if previous == UNUSED {
                    continue;
                }
                let row = previous as usize;
                if self
                    .cached_hashes
                    .get(row)
                    .is_some_and(|&cached| cached != hash)
                {
                    continue;
                }
                if !self
                    .prepared
                    .residuals
                    .iter()
                    .zip(residuals)
                    .all(|(column, value)| column[row] == *value)
                    || !self
                        .prepared
                        .properties
                        .iter()
                        .zip(properties)
                        .all(|(column, &value)| column.value(row) == value)
                {
                    continue;
                }
                let take = remaining.min(u16::MAX - self.prepared.counts[row]);
                self.prepared.counts[row] += take;
                remaining -= take;
                if self.prepared.counts[row] == u16::MAX {
                    self.table[position] = UNUSED;
                }
                if remaining == 0 {
                    break;
                }
            }
            if remaining == 0 {
                continue;
            }
            let row = self.prepared.len();
            if row >= UNUSED as usize {
                return Err(Error::Limit("MA gather index exceeds source u32"));
            }
            // Expected sample counts are reservations, never truncation limits.
            // A sampling overshoot grows columns but does not resize the table.
            if row == self.reserved_rows {
                self.reserve_columns(1)?;
            }
            for (column, &value) in self.prepared.properties.iter_mut().zip(properties) {
                column.push(value);
            }
            for (column, &value) in self.prepared.residuals.iter_mut().zip(residuals) {
                column.push(value);
            }
            self.prepared.counts.push(remaining);
            if row < self.hash_limit {
                debug_assert_eq!(self.cached_hashes.len(), row);
                debug_assert!(self.cached_hashes.len() < self.cached_hashes.capacity());
                self.cached_hashes.push(hash);
            }
            #[cfg(test)]
            self.prepared.static_ids.push(ids);
            if remaining != u16::MAX {
                self.register(row, positions);
            }
        }
        Ok(())
    }

    pub(super) fn push(
        &mut self,
        properties: &[i32],
        residuals: &[i32],
        weight: u32,
    ) -> Result<()> {
        if properties.len() < self.required_properties
            || residuals.len() != self.prepared.predictors.len()
        {
            return Err(Error::InvalidInput(
                "invalid MA gather sample dimensions".into(),
            ));
        }
        if weight == 0 {
            return Ok(());
        }
        if self.table.is_empty() {
            return Err(Error::InvalidInput(
                "MA gather requires prepare before samples".into(),
            ));
        }
        let total = self
            .prepared
            .total
            .checked_add(u64::from(weight))
            .filter(|&n| n <= i32::MAX as u64)
            .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
        let mut bins = [0; 64];
        for (p, &id) in self.prepared.quantization.properties.iter().enumerate() {
            if id < 2 {
                let value = properties[id];
                bins[p] = match self.static_bins[id] {
                    Some((previous, bin)) if previous == value => bin,
                    _ => {
                        let bin = self.prepared.quantization.thresholds[p]
                            .partition_point(|&t| value > t)
                            as u32;
                        self.static_bins[id] = Some((value, bin));
                        bin
                    }
                };
                continue;
            }
            let value = properties[id].clamp(-511, 511);
            bins[p] = match self.property_luts[p].get((i64::from(value) + 511) as usize) {
                Some(&bin) => u32::from(bin),
                None => {
                    self.prepared.quantization.thresholds[p].partition_point(|&t| value > t) as u32
                }
            };
        }
        let mut tokens = [[0; 2]; 14];
        for (token, &value) in tokens.iter_mut().zip(residuals) {
            *token = residual_token(value);
        }
        self.insert(
            &bins[..self.prepared.properties.len()],
            &tokens[..residuals.len()],
            [properties[0], properties[1]],
            weight,
        )?;
        self.prepared.total = total;
        Ok(())
    }

    pub(super) fn append(&mut self, batches: Vec<Self>) -> Result<()> {
        let mut extra = 0usize;
        let mut total = self.prepared.total;
        for batch in &batches {
            if self.prepared.predictors != batch.prepared.predictors
                || self.prepared.quantization.properties != batch.prepared.quantization.properties
                || self.prepared.quantization.thresholds != batch.prepared.quantization.thresholds
            {
                return Err(Error::InvalidInput("incompatible MA gather batches".into()));
            }
            extra = extra
                .checked_add(batch.prepared.len())
                .ok_or(Error::Limit("MA gather append capacity overflow"))?;
            total = total
                .checked_add(batch.prepared.total)
                .filter(|&n| n <= i32::MAX as u64)
                .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
        }
        if extra == 0 {
            return Ok(());
        }
        self.prepare(extra)?;
        for batch in batches {
            for row in 0..batch.prepared.len() {
                let mut bins = [0; 64];
                let mut tokens = [[0; 2]; 14];
                for (bin, column) in bins.iter_mut().zip(&batch.prepared.properties) {
                    *bin = column.value(row);
                }
                for (token, column) in tokens.iter_mut().zip(&batch.prepared.residuals) {
                    *token = column[row];
                }
                #[cfg(test)]
                let ids = batch.prepared.static_ids[row];
                #[cfg(not(test))]
                let ids = [0; 2];
                let hash = batch.cached_hashes.get(row).copied().unwrap_or_else(|| {
                    hashes(
                        tokens[..batch.prepared.residuals.len()].iter().copied(),
                        batch.property_order.iter().map(|&p| bins[p]),
                    )
                });
                self.insert_hashed(
                    &bins[..batch.prepared.properties.len()],
                    &tokens[..batch.prepared.residuals.len()],
                    ids,
                    u32::from(batch.prepared.counts[row]),
                    hash,
                )?;
            }
            // Drop each consumed batch and its table before the next one.
        }
        self.prepared.total = total;
        Ok(())
    }

    /// Terminal, opt-in single-wave merge. Local weights are retained exactly;
    /// extra cross-batch records may change record-count-based search heuristics.
    pub(super) fn finish_appended(self, batches: Vec<Self>) -> Result<PreparedSamples> {
        self.finish_appended_with_limit(batches, MAX_CONCAT_PAYLOAD_BYTES)
    }

    // A smaller private limit lets tests exercise the real fallback cheaply.
    fn finish_appended_with_limit(
        mut self,
        batches: Vec<Self>,
        payload_limit: usize,
    ) -> Result<PreparedSamples> {
        let row_bytes = self.prepared.validate_concat()?;
        let mut rows = self.prepared.len();
        let mut total = self.prepared.total;
        for batch in &batches {
            if self.prepared.predictors != batch.prepared.predictors
                || self.prepared.quantization != batch.prepared.quantization
            {
                return Err(Error::InvalidInput("incompatible MA gather batches".into()));
            }
            batch.prepared.validate_concat()?;
            rows = rows
                .checked_add(batch.prepared.len())
                .filter(|&n| n <= UNUSED as usize)
                .ok_or(Error::Limit("MA gather index exceeds source u32"))?;
            total = total
                .checked_add(batch.prepared.total)
                .filter(|&n| n <= i32::MAX as u64)
                .ok_or(Error::Limit("MA sample weight exceeds source i32::MAX"))?;
        }
        if !concat_allowed(rows, total, row_bytes, payload_limit) {
            self.append(batches)?;
            return Ok(self.finish());
        }

        // Drop every index/hash cache before allocating destination columns.
        // Consuming collectors prevents later insertion with a stale index.
        let mut prepared_batches = Vec::new();
        prepared_batches
            .try_reserve_exact(batches.len())
            .map_err(|_| Error::Limit("MA concat batch allocation"))?;
        for batch in batches {
            prepared_batches.push(batch.finish());
        }
        let mut output = self.finish();
        let extra = rows - output.len();
        for column in &mut output.properties {
            column.reserve(extra)?;
        }
        for column in &mut output.residuals {
            reserve(column, extra)?;
        }
        reserve(&mut output.counts, extra)?;
        #[cfg(test)]
        reserve(&mut output.static_ids, extra)?;
        for mut batch in prepared_batches {
            for (destination, source) in output.properties.iter_mut().zip(batch.properties) {
                match (destination, source) {
                    (GatherColumn::Packed(destination), GatherColumn::Packed(mut source)) => {
                        destination.append(&mut source);
                    }
                    (GatherColumn::Static(destination), GatherColumn::Static(mut source)) => {
                        destination.append(&mut source);
                    }
                    _ => unreachable!("concat property layout validated before allocation"),
                }
            }
            for (destination, mut source) in output.residuals.iter_mut().zip(batch.residuals) {
                destination.append(&mut source);
            }
            output.counts.append(&mut batch.counts);
            #[cfg(test)]
            output.static_ids.append(&mut batch.static_ids);
        }
        output.total = total;
        debug_assert_eq!(output.len(), rows);
        Ok(output)
    }

    /// AllSamplesDone: the invocation-local index table is not handed to search.
    pub(super) fn finish(self) -> PreparedSamples {
        self.prepared
    }
}

#[cfg(test)]
#[path = "gather_concat_tests.rs"]
mod concat_tests;

#[cfg(test)]
#[path = "gather_push_tests.rs"]
mod push_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn collector() -> SourceSampleCollector {
        collector_with_hashes(false)
    }

    fn collector_with_hashes(cache_hashes: bool) -> SourceSampleCollector {
        SourceSampleCollector::new(
            &[Predictor::Zero],
            &PropertyQuantization {
                properties: vec![0, 1, 9],
                thresholds: vec![vec![0], vec![3], vec![-2, 0, 2]],
            },
            cache_hashes,
        )
        .unwrap()
    }

    fn properties(value: i32) -> [i32; 10] {
        let mut p = [0; 10];
        p[1] = 3;
        p[9] = value;
        p
    }

    #[test]
    fn source_quantization_lut_matches_every_clamped_value_and_bin_width() {
        let mut cases: Vec<Vec<i32>> = [0, 1, 2, 17, 127, 254, 255]
            .into_iter()
            .map(|count| (0..count).map(|i| -511 + i * 4).collect())
            .collect();
        cases.push(vec![i32::MIN, -512, -511, -1, 0, 1, 511, 512, i32::MAX]);
        cases.push((-1000..-745).collect());
        cases.push((1000..1255).collect());
        for thresholds in cases {
            let c = SourceSampleCollector::new(
                &[Predictor::Zero],
                &PropertyQuantization {
                    properties: vec![9, 1, 0],
                    thresholds: vec![thresholds.clone(), vec![i32::MIN, i32::MAX], vec![]],
                },
                false,
            )
            .unwrap();
            assert_eq!(c.property_luts[0].len(), 1023);
            assert!(c.property_luts[1].is_empty());
            assert!(c.property_luts[2].is_empty());
            for value in -511..=511 {
                assert_eq!(
                    usize::from(c.property_luts[0][(value + 511) as usize]),
                    thresholds.partition_point(|&t| value > t),
                    "value={value} thresholds={thresholds:?}",
                );
            }
        }
    }

    fn assert_same_collector(actual: &SourceSampleCollector, expected: &SourceSampleCollector) {
        assert_eq!(actual.table, expected.table);
        assert_eq!(actual.prepared.total, expected.prepared.total);
        assert_eq!(actual.prepared.counts, expected.prepared.counts);
        assert_eq!(actual.prepared.residuals, expected.prepared.residuals);
        assert_eq!(actual.prepared.static_ids, expected.prepared.static_ids);
        assert_eq!(
            actual.prepared.properties.len(),
            expected.prepared.properties.len()
        );
        for (a, b) in actual
            .prepared
            .properties
            .iter()
            .zip(&expected.prepared.properties)
        {
            match (a, b) {
                (GatherColumn::Packed(a), GatherColumn::Packed(b)) => assert_eq!(a, b),
                (GatherColumn::Static(a), GatherColumn::Static(b)) => assert_eq!(a, b),
                _ => panic!("property storage changed"),
            }
        }
        for collector in [actual, expected] {
            assert_eq!(
                collector.cached_hashes.len(),
                collector.prepared.len().min(collector.hash_limit),
            );
            assert!(collector.cached_hashes.capacity() <= collector.hash_limit);
            for (row, &cached) in collector.cached_hashes.iter().enumerate() {
                assert_eq!(
                    cached,
                    hashes(
                        collector
                            .prepared
                            .residuals
                            .iter()
                            .map(|column| column[row]),
                        collector
                            .property_order
                            .iter()
                            .map(|&p| collector.prepared.properties[p].value(row)),
                    ),
                );
            }
        }
    }

    fn hash_collector(limit: usize) -> SourceSampleCollector {
        assert!(limit <= MAX_CACHED_HASHES);
        let mut result = collector_with_hashes(limit != 0);
        result.hash_limit = limit;
        result
    }

    #[test]
    fn source_hash_cache_switch_disables_allocations_without_changing_records() {
        let mut cached = collector_with_hashes(true);
        let mut uncached = collector_with_hashes(false);
        assert_eq!(cached.hash_limit, MAX_CACHED_HASHES);
        assert_eq!(uncached.hash_limit, 0);
        for extra in [1, 128, 0] {
            cached.prepare(extra).unwrap();
            uncached.prepare(extra).unwrap();
            for i in 0..32 {
                cached.push(&properties(i), &[i], 1).unwrap();
                uncached.push(&properties(i), &[i], 1).unwrap();
            }
            assert!(!cached.cached_hashes.is_empty());
            assert_eq!(uncached.cached_hashes.capacity(), 0);
            assert_same_collector(&cached, &uncached);
        }
    }

    #[test]
    fn source_hash_cache_matches_uncached_overshoot_rebuilds_and_saturation() {
        for limit in [0, 1, 2, 7, 64] {
            let mut actual = hash_collector(limit);
            let mut expected = hash_collector(0);
            for extra in [1, 0, 128, 0] {
                actual.prepare(extra).unwrap();
                expected.prepare(extra).unwrap();
                let table_size = actual.table.len();
                for i in 0..128 {
                    let value = [i32::MIN, i32::MAX, -511, -1, 0, 511][i % 6];
                    let mut p = properties(value);
                    p[0] = (i % 3) as i32;
                    let residual = value.wrapping_add(i as i32);
                    for weight in [1, 0, 65_534, 1, 2, 65_536] {
                        actual.push(&p, &[residual], weight).unwrap();
                        expected.push(&p, &[residual], weight).unwrap();
                    }
                }
                assert_eq!(
                    actual.table.len(),
                    table_size,
                    "overshoot cannot resize slots"
                );
                assert_same_collector(&actual, &expected);
            }
            assert!(actual.prepared.len() > limit);
            assert!(actual.prepared.counts.contains(&u16::MAX));
            assert!(matches!(
                actual.push(&properties(0), &[0], u32::MAX),
                Err(Error::Limit(_))
            ));
            assert!(matches!(
                expected.push(&properties(0), &[0], u32::MAX),
                Err(Error::Limit(_))
            ));
            assert_same_collector(&actual, &expected);
        }
    }

    #[test]
    fn source_hash_cache_mixed_append_keeps_records_and_preflight_errors() {
        for master_limit in [0, 3, 64] {
            let mut actual = hash_collector(master_limit);
            let mut expected = hash_collector(0);
            actual.prepare(1).unwrap();
            expected.prepare(1).unwrap();
            actual.push(&properties(-7), &[i32::MIN], 65_534).unwrap();
            expected.push(&properties(-7), &[i32::MIN], 65_534).unwrap();
            for wave in 0..3 {
                let mut batches = Vec::new();
                let mut reference_batches = Vec::new();
                for batch_limit in [64, 0, 2] {
                    let mut batch = hash_collector(batch_limit);
                    let mut reference = hash_collector(0);
                    batch.prepare(1).unwrap();
                    reference.prepare(1).unwrap();
                    for i in 0..17 {
                        let p = properties(-7 + i);
                        let residual = if i % 3 == 0 { i32::MIN } else { i + wave };
                        let weight = if i % 4 == 0 { 65_537 } else { 1 };
                        batch.push(&p, &[residual], weight).unwrap();
                        reference.push(&p, &[residual], weight).unwrap();
                    }
                    assert_same_collector(&batch, &reference);
                    batches.push(batch);
                    reference_batches.push(reference);
                }
                batches.push(hash_collector(0));
                reference_batches.push(hash_collector(0));
                actual.append(batches).unwrap();
                expected.append(reference_batches).unwrap();
                assert_same_collector(&actual, &expected);
                actual.prepare(0).unwrap();
                expected.prepare(0).unwrap();
                assert_same_collector(&actual, &expected);
            }
            let mut incompatible = hash_collector(2);
            incompatible.prepared.quantization.thresholds[2][0] -= 1;
            assert!(matches!(
                actual.append(vec![incompatible]),
                Err(Error::InvalidInput(_))
            ));
            assert_same_collector(&actual, &expected);
            let mut overweight = hash_collector(0);
            overweight.prepared.total = i32::MAX as u64;
            assert!(matches!(
                actual.append(vec![overweight]),
                Err(Error::Limit(_))
            ));
            assert_same_collector(&actual, &expected);
        }
    }

    #[test]
    fn source_hash_cache_full_collision_still_checks_keys_and_retains_misses() {
        let mut c = hash_collector(4);
        c.prepare(2).unwrap();
        // Deliberately force equal full hashes for unequal keys. Hash equality
        // may only select the old exact comparison, never prove a match.
        let hash = [0, 1 << 16];
        let first = [0, 0, 0];
        let second = [1, 0, 0];
        c.insert_hashed(&first, &[[1, 0]], [0, 3], 1, hash).unwrap();
        c.insert_hashed(&second, &[[1, 0]], [0, 3], 1, hash)
            .unwrap();
        let slots = c.table.clone();
        assert_eq!(&slots[..2], &[0, 1]);
        for _ in 0..2 {
            c.insert_hashed(&first, &[[2, 0]], [0, 3], 1, hash).unwrap();
        }
        c.insert_hashed(&first, &[[1, 0]], [0, 3], 1, hash).unwrap();
        assert_eq!(c.table, slots);
        assert_eq!(c.prepared.counts, [2, 1, 1, 1]);
        assert_eq!(c.prepared.residuals[0], [[1, 0], [1, 0], [2, 0], [2, 0]]);
        assert_eq!(c.cached_hashes, [hash; 4]);
    }

    #[test]
    fn source_hash_cache_reservation_caps_at_sixteen_mib() {
        let mut c = hash_collector(MAX_CACHED_HASHES);
        c.reserve_hashes(usize::MAX).unwrap();
        assert_eq!(c.cached_hashes.capacity(), MAX_CACHED_HASHES);
        assert_eq!(
            c.cached_hashes.capacity() * std::mem::size_of::<[u64; 2]>(),
            16 << 20
        );
        assert!(c.cached_hashes.is_empty());
        let pointer = c.cached_hashes.as_ptr();
        c.reserve_hashes(0).unwrap();
        c.reserve_hashes(MAX_CACHED_HASHES + 1).unwrap();
        assert_eq!(c.cached_hashes.as_ptr(), pointer);
        assert_eq!(c.cached_hashes.capacity(), MAX_CACHED_HASHES);
        assert!(c.prepare(usize::MAX).is_err());
        assert!(c.prepared.counts.is_empty());
    }

    #[test]
    fn source_quantization_lut_preserves_hash_merge_saturation_and_table_lifetime() {
        let quantization = PropertyQuantization {
            properties: vec![9, 1, 0, 2],
            thresholds: vec![
                (-511..509).step_by(4).collect(),
                (-600..600).collect(),
                vec![i32::MIN, -1, 0, i32::MAX],
                vec![i32::MIN, -512, -511, 0, 511, 512, i32::MAX],
            ],
        };
        let make = || {
            SourceSampleCollector::new(
                &[Predictor::Weighted, Predictor::Gradient],
                &quantization,
                false,
            )
            .unwrap()
        };
        let mut actual = make();
        let mut reference = make();
        for lut in &mut reference.property_luts {
            lut.clear();
        }
        let pointers: Vec<_> = actual.property_luts.iter().map(Vec::as_ptr).collect();
        let values: Vec<_> = [i32::MIN, i32::MAX, -512, -511, 511, 512]
            .into_iter()
            .chain(-520..=520)
            .collect();
        for extra in [1, 0, 4096, 0] {
            actual.prepare(extra).unwrap();
            reference.prepare(extra).unwrap();
            for (i, &value) in values.iter().enumerate() {
                let mut p = properties(value);
                p[0] = [i32::MIN, -1, 0, 1, i32::MAX][i % 5];
                p[1] = value;
                p[2] = value;
                let residuals = [value, value.wrapping_neg()];
                for weight in [1, 0, u32::from(u16::MAX), 2] {
                    actual.push(&p, &residuals, weight).unwrap();
                    reference.push(&p, &residuals, weight).unwrap();
                }
            }
            assert_same_collector(&actual, &reference);
            assert_eq!(
                actual
                    .property_luts
                    .iter()
                    .map(Vec::as_ptr)
                    .collect::<Vec<_>>(),
                pointers,
            );
        }
        let mut batch = make();
        let mut reference_batch = make();
        for lut in &mut reference_batch.property_luts {
            lut.clear();
        }
        batch.prepare(1).unwrap();
        reference_batch.prepare(1).unwrap();
        for value in [i32::MIN, 0, i32::MAX] {
            batch.push(&properties(value), &[0, 0], 65_537).unwrap();
            reference_batch
                .push(&properties(value), &[0, 0], 65_537)
                .unwrap();
        }
        actual.append(vec![batch]).unwrap();
        reference.append(vec![reference_batch]).unwrap();
        assert_same_collector(&actual, &reference);
        assert_eq!(
            actual
                .property_luts
                .iter()
                .map(Vec::as_ptr)
                .collect::<Vec<_>>(),
            pointers,
        );
    }

    #[test]
    fn source_quantization_lut_keeps_schema_and_sample_errors() {
        for thresholds in [(0..256).collect(), vec![1, 1], vec![1, 0]] {
            assert!(matches!(
                SourceSampleCollector::new(
                    &[Predictor::Zero],
                    &PropertyQuantization {
                        properties: vec![9],
                        thresholds: vec![thresholds]
                    },
                    false,
                ),
                Err(Error::InvalidInput(_)),
            ));
        }
        let mut c = SourceSampleCollector::new(
            &[Predictor::Zero],
            &PropertyQuantization {
                properties: vec![0],
                thresholds: vec![(0..1024).collect()],
            },
            false,
        )
        .unwrap();
        assert!(c.property_luts[0].is_empty());
        assert!(matches!(
            c.push(&[i32::MAX, 0], &[0], 1),
            Err(Error::InvalidInput(_))
        ));
        assert!(c.push(&[i32::MAX, 0], &[0], 0).is_ok());
        c.prepare(1).unwrap();
        c.push(&[i32::MAX, 0], &[0], 1).unwrap();
        assert_eq!(c.prepared.properties[0].value(0), 1024);
        assert!(matches!(c.push(&[0], &[0], 0), Err(Error::InvalidInput(_))));
        assert!(matches!(
            c.push(&[0, 0], &[], 1),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            c.push(&[0, 0], &[0], u32::MAX),
            Err(Error::Limit(_))
        ));
        assert_eq!(c.prepared.total, 1);
        assert_eq!(c.prepared.counts, [1]);
    }

    #[test]
    fn source_table_floor_caps_and_prepare_lifetime() {
        for (records, slots) in [
            (0, 0),
            (1, 1),
            (2, 4),
            (3, 4),
            (64, 128),
            (699_052, 1 << 21),
        ] {
            assert_eq!(table_slots(records).unwrap(), slots);
        }
        assert!(table_slots(usize::MAX).is_err());
        assert_eq!(expected_samples(0.1, 17, 19).unwrap(), 96);
        let mut c = collector();
        c.prepare(1).unwrap();
        assert_eq!(c.table.len(), 1);
        c.push(&properties(0), &[13], 1).unwrap();
        let slots = c.table.clone();
        c.prepare(0).unwrap();
        assert_eq!(c.table, slots);
        for value in 0..300 {
            c.push(&properties(value), &[value], 1).unwrap();
        }
        assert_eq!(c.table.len(), 1, "overshoot must not grow source table");
        c.prepare(1000).unwrap();
        let grown = c.table.len();
        c.prepare(0).unwrap();
        assert!(c.table.len() < grown);
        assert_eq!(c.prepared.total, 301);
    }

    #[test]
    fn source_columns_saturation_append_and_adapter_preserve_objective() {
        let mut master = collector();
        master.prepare(1).unwrap();
        master.push(&properties(-7), &[i32::MIN], 65_534).unwrap();
        let mut batch = collector();
        batch.prepare(1).unwrap();
        batch.push(&properties(-3), &[i32::MIN], 65_537).unwrap();
        master.append(vec![batch]).unwrap();
        assert_eq!(master.prepared.total, 131_071);
        assert_eq!(master.prepared.counts, [65_535, 65_535, 1]);
        assert!(matches!(
            master.prepared.properties[0],
            GatherColumn::Static(_)
        ));
        assert!(matches!(
            master.prepared.properties[2],
            GatherColumn::Packed(_)
        ));
        let prepared = master.finish();
        let expected = prepared.residuals[0].clone();
        let samples = prepared.into_samples().unwrap();
        for (sample, token) in samples.iter().zip(expected) {
            assert_eq!(&sample.properties[..2], &[0, 3]);
            assert_eq!(sample.properties[9], -2);
            let encoded = HybridUintConfig::new(4, 1, 2)
                .unwrap()
                .encode(pack_signed(sample.residuals[0]));
            assert_eq!([encoded.token as u8, encoded.nbits as u8], token);
        }
    }

    #[test]
    fn source_adapter_roundtrips_all_residual_token_shapes() {
        for value in [i32::MIN, i32::MAX, -1, 0, 1]
            .into_iter()
            .chain((-8192..8192).step_by(7))
        {
            let mut c = collector();
            c.prepare(1).unwrap();
            c.push(&properties(0), &[value], 1).unwrap();
            let actual = c.finish().into_samples().unwrap();
            let config = HybridUintConfig::new(4, 1, 2).unwrap();
            let a = config.encode(pack_signed(value));
            let b = config.encode(pack_signed(actual[0].residuals[0]));
            assert_eq!((a.token, a.nbits), (b.token, b.nbits));
        }
    }

    #[test]
    fn source_append_rejects_mismatched_quantizers_and_total_limits() {
        let mut master = collector();
        let mut batch = collector();
        batch.prepared.quantization.thresholds[2][0] = -3;
        assert!(master.append(vec![batch]).is_err());
        master.prepare(1).unwrap();
        assert!(master.push(&properties(0), &[0], u32::MAX).is_err());
        assert_eq!(master.prepared.total, 0);
        assert!(master.prepare(usize::MAX).is_err());
    }

    #[test]
    fn source_hash_order_and_collision_misses_do_not_evict_or_probe_further() {
        assert_eq!(
            hashes([[7, 3], [128, 16]].into_iter(), [0, 800, 255].into_iter()),
            [0x610a_7a6b_7605_8ae6, 0xce04_e503_cb3e_0cee]
        );
        let reordered = SourceSampleCollector::new(
            &[Predictor::Zero],
            &PropertyQuantization {
                properties: vec![9, 1, 0],
                thresholds: vec![vec![0], vec![3], vec![0]],
            },
            false,
        )
        .unwrap();
        assert_eq!(reordered.property_order, [2, 1, 0]);

        let mut c = collector();
        c.prepare(2).unwrap();
        let bins = [0, 0, 0];
        let mut candidates = std::collections::BTreeMap::<[usize; 2], Vec<[u8; 2]>>::new();
        for token in 0..=239 {
            let residual = [token, 0];
            let positions = c.positions(hashes([residual].into_iter(), bins.into_iter()));
            if positions[0] != positions[1] {
                candidates.entry(positions).or_default().push(residual);
            }
        }
        let (positions, tokens) = candidates
            .into_iter()
            .find(|(_, tokens)| tokens.len() >= 3)
            .unwrap();
        c.insert(&bins, &tokens[0..1], [0, 3], 1).unwrap();
        c.insert(&bins, &tokens[1..2], [0, 3], 1).unwrap();
        let table = c.table.clone();
        assert_ne!(table[positions[0]], UNUSED);
        assert_ne!(table[positions[1]], UNUSED);
        c.insert(&bins, &tokens[2..3], [0, 3], 1).unwrap();
        c.insert(&bins, &tokens[2..3], [0, 3], 1).unwrap();
        assert_eq!(c.table, table);
        assert_eq!(c.prepared.counts, [1; 4]);
        assert_eq!(
            c.prepared.residuals[0],
            [tokens[0], tokens[1], tokens[2], tokens[2]]
        );
        c.prepare(16).unwrap();
        let mut registered = std::collections::BTreeSet::new();
        for &row in c.table.iter().filter(|&&row| row != UNUSED) {
            assert!(
                registered.insert(row),
                "no stale table aliases after rebuild"
            );
        }
    }
}
