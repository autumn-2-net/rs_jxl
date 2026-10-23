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
use crate::entropy::hybrid_uint::HybridUintConfig;
use crate::{Error, Result};

const UNUSED: u32 = u32::MAX;
const A: u64 = 0x1e35_a7bd;
const B: u64 = 0x1e35_a7bd_1e35_a7bd;

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

pub(super) struct SourceSampleCollector {
    prepared: PreparedSamples,
    table: Vec<u32>,
    property_order: Vec<usize>,
    config: HybridUintConfig,
    reserved_rows: usize,
}

impl SourceSampleCollector {
    pub(super) fn new(
        predictors: &[Predictor],
        quantization: &PropertyQuantization,
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
            config: HybridUintConfig::new(4, 1, 2).map_err(|e| Error::Encoding(e.to_string()))?,
            reserved_rows: 0,
        })
    }

    fn reserve_columns(&mut self, extra: usize) -> Result<()> {
        for column in &mut self.prepared.properties {
            column.reserve(extra)?;
        }
        for column in &mut self.prepared.residuals {
            reserve(column, extra)?;
        }
        reserve(&mut self.prepared.counts, extra)?;
        self.reserved_rows = self
            .prepared
            .properties
            .iter()
            .map(GatherColumn::capacity)
            .chain(self.prepared.residuals.iter().map(Vec::capacity))
            .chain([self.prepared.counts.capacity()])
            .min()
            .unwrap_or(0);
        #[cfg(test)]
        {
            reserve(&mut self.prepared.static_ids, extra)?;
            self.reserved_rows = self.reserved_rows.min(self.prepared.static_ids.capacity());
        }
        Ok(())
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
                let hash = hashes(
                    self.prepared.residuals.iter().map(|column| column[row]),
                    self.property_order
                        .iter()
                        .map(|&p| self.prepared.properties[p].value(row)),
                );
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
        mut weight: u32,
    ) -> Result<()> {
        #[cfg(not(test))]
        let _ = ids;
        let positions = self.positions(hashes(
            residuals.iter().copied(),
            self.property_order.iter().map(|&p| properties[p]),
        ));
        while weight != 0 {
            let mut remaining = weight.min(u32::from(u16::MAX)) as u16;
            weight -= u32::from(remaining);
            for position in positions {
                let previous = self.table[position];
                if previous == UNUSED {
                    continue;
                }
                let row = previous as usize;
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
        if properties.len() < 2
            || residuals.len() != self.prepared.predictors.len()
            || self
                .prepared
                .quantization
                .properties
                .iter()
                .any(|&p| p >= properties.len())
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
            let value = if id < 2 {
                properties[id]
            } else {
                properties[id].clamp(-511, 511)
            };
            bins[p] =
                self.prepared.quantization.thresholds[p].partition_point(|&t| value > t) as u32;
        }
        let mut tokens = [[0; 2]; 14];
        for (token, &value) in tokens.iter_mut().zip(residuals) {
            let encoded = self.config.encode(pack_signed(value));
            *token = [encoded.token as u8, encoded.nbits as u8];
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
                self.insert(
                    &bins[..batch.prepared.properties.len()],
                    &tokens[..batch.prepared.residuals.len()],
                    ids,
                    u32::from(batch.prepared.counts[row]),
                )?;
            }
            // Drop each consumed batch and its table before the next one.
        }
        self.prepared.total = total;
        Ok(())
    }

    /// AllSamplesDone: the invocation-local index table is not handed to search.
    pub(super) fn finish(self) -> PreparedSamples {
        self.prepared
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector() -> SourceSampleCollector {
        SourceSampleCollector::new(
            &[Predictor::Zero],
            &PropertyQuantization {
                properties: vec![0, 1, 9],
                thresholds: vec![vec![0], vec![3], vec![-2, 0, 2]],
            },
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
