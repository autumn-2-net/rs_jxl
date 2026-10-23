// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause sources; the full upstream notice is retained in
// docs/PORTING_KERNELS.md.

//! Default TreeSamples::{PrepareForSamples, InitTable, Hash1, Hash2,
//! AddWeightedToTableAndMerge} from enc_ma.h/cc at frozen libjxl
//! b866b3defec13494bfbaf705db93581cda118e37. Collisions retain records; saturated
//! records leave the table. External wide weights are split, never clamped.

use super::{PropertyColumn, ResidualToken, Sample, SampleKey, TreeSamples};
use crate::{Error, Result};

const UNUSED: u32 = u32::MAX;
const MAX_COUNT: u16 = u16::MAX;

pub(super) fn split_record_capacity<'a>(
    mut samples: impl Iterator<Item = &'a Sample>,
) -> Result<usize> {
    samples.try_fold(0usize, |records, sample| {
        records
            .checked_add(sample.weight.div_ceil(u32::from(MAX_COUNT)) as usize)
            .ok_or(Error::Limit("MA split sample capacity overflow"))
    })
}

fn table_slots(records: usize) -> Result<usize> {
    if records == 0 {
        return Ok(0);
    }
    if records > UNUSED as usize {
        return Err(Error::Limit("MA sample index exceeds source u32"));
    }
    // Source uses floor(3*N/2), then CeilLog2Nonzero, not a load-factor cap
    // or a fixed maximum number of slots. Allocation failure is explicit.
    records
        .checked_mul(3)
        .map(|n| n / 2)
        .and_then(usize::checked_next_power_of_two)
        .ok_or(Error::Limit("MA dedup table capacity overflow"))
}

fn reserve_samples<T>(values: &mut Vec<T>, extra: usize) -> Result<()> {
    let required = values
        .len()
        .checked_add(extra)
        .ok_or(Error::Limit("MA sample column capacity overflow"))?;
    if required <= values.capacity() {
        return Ok(());
    }
    let grown = values
        .capacity()
        .checked_add(values.capacity() / 2)
        .unwrap_or(required);
    values
        .try_reserve_exact(required.max(grown) - values.len())
        .map_err(|_| Error::Limit("MA sample column allocation"))
}

fn property_value(column: &PropertyColumn, index: usize) -> u32 {
    match column {
        PropertyColumn::Packed(values) => u32::from(values[index]),
        PropertyColumn::Wide(values) => values[index],
    }
}

fn hashes(
    residuals: impl Iterator<Item = ResidualToken> + Clone,
    properties: impl Iterator<Item = u32>,
) -> [u64; 2] {
    const A: u64 = 0x1e35_a7bd;
    const B: u64 = 0x1e35_a7bd_1e35_a7bd;
    let (mut first, mut second) = (A, B);
    for r in residuals.clone() {
        first = first
            .wrapping_mul(A)
            .wrapping_add(u64::from(r.token))
            .wrapping_mul(A)
            .wrapping_add(u64::from(r.nbits));
    }
    for value in properties {
        first = first.wrapping_mul(A).wrapping_add(u64::from(value));
        second = second.wrapping_mul(B) ^ u64::from(value);
    }
    for r in residuals {
        second = (second.wrapping_mul(B) ^ u64::from(r.token)).wrapping_mul(B) ^ u64::from(r.nbits);
    }
    [first, second]
}

pub(super) struct DedupTable {
    slots: Vec<u32>,
    property_order: Vec<usize>,
}

impl DedupTable {
    pub(super) fn new(properties: &[usize]) -> Self {
        // Source configurations place static IDs first. Preserve that hash
        // order even for Rust's supported interleaved property configurations.
        let property_order = [0, 1]
            .into_iter()
            .filter_map(|id| properties.iter().position(|&p| p == id))
            .chain(
                properties
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| p >= 2)
                    .map(|(i, _)| i),
            )
            .collect();
        Self {
            slots: Vec::new(),
            property_order,
        }
    }

    fn positions(&self, hashes: [u64; 2]) -> [usize; 2] {
        let mask = self.slots.len() - 1;
        hashes.map(|hash| (hash >> 16) as usize & mask)
    }

    fn key_hashes(&self, key: &SampleKey) -> [u64; 2] {
        hashes(
            key.residuals.iter().copied(),
            self.property_order.iter().map(|&p| key.properties[p]),
        )
    }

    fn row_hashes(&self, samples: &TreeSamples, index: usize) -> [u64; 2] {
        hashes(
            samples.residuals.iter().map(|column| column[index]),
            self.property_order
                .iter()
                .map(|&p| property_value(&samples.properties[p], index)),
        )
    }

    pub(super) fn prepare(&mut self, samples: &mut TreeSamples, extra: usize) -> Result<()> {
        let records = samples
            .counts
            .len()
            .checked_add(extra)
            .ok_or(Error::Limit("MA sample capacity overflow"))?;
        let slots = table_slots(records)?;
        for column in &mut samples.residuals {
            reserve_samples(column, extra)?;
        }
        for column in &mut samples.properties {
            match column {
                PropertyColumn::Packed(values) => reserve_samples(values, extra)?,
                PropertyColumn::Wide(values) => reserve_samples(values, extra)?,
            }
        }
        reserve_samples(&mut samples.counts, extra)?;
        if slots == self.slots.len() {
            return Ok(());
        }
        let mut fresh = Vec::new();
        fresh
            .try_reserve_exact(slots)
            .map_err(|_| Error::Limit("MA dedup table allocation"))?;
        fresh.resize(slots, UNUSED);
        self.slots = fresh;
        // FastDedup is enabled by default in the frozen source. Never retain
        // old slot aliases across a resize; do not merge records on rehash.
        for (index, &count) in samples.counts.iter().enumerate() {
            if count != MAX_COUNT {
                self.register(index, self.positions(self.row_hashes(samples, index)));
            }
        }
        Ok(())
    }

    fn register(&mut self, index: usize, positions: [usize; 2]) {
        for position in positions {
            if self.slots[position] == UNUSED {
                self.slots[position] = index as u32;
                break;
            }
        }
    }

    fn same_sample(samples: &TreeSamples, key: &SampleKey, index: usize) -> bool {
        samples
            .residuals
            .iter()
            .zip(&key.residuals)
            .all(|(column, &value)| column[index] == value)
            && samples
                .properties
                .iter()
                .zip(&key.properties)
                .all(|(column, &value)| property_value(column, index) == value)
    }

    pub(super) fn insert(
        &mut self,
        samples: &mut TreeSamples,
        key: &SampleKey,
        mut weight: u32,
    ) -> Result<()> {
        if weight == 0 {
            return Ok(());
        }
        let positions = self.positions(self.key_hashes(key));
        while weight != 0 {
            let chunk = weight.min(u32::from(MAX_COUNT)) as u16;
            weight -= u32::from(chunk);
            let mut remaining = chunk;
            for position in positions {
                let previous = self.slots[position];
                if previous == UNUSED || !Self::same_sample(samples, key, previous as usize) {
                    continue;
                }
                let previous = previous as usize;
                let take = remaining.min(MAX_COUNT - samples.counts[previous]);
                samples.counts[previous] += take;
                remaining -= take;
                if samples.counts[previous] == MAX_COUNT {
                    self.slots[position] = UNUSED;
                }
                if remaining == 0 {
                    break;
                }
            }
            if remaining == 0 {
                continue;
            }
            let index = samples.counts.len();
            if index >= UNUSED as usize {
                return Err(Error::Limit("MA sample index exceeds source u32"));
            }
            for (column, &value) in samples.properties.iter_mut().zip(&key.properties) {
                column.push(value);
            }
            for (column, &value) in samples.residuals.iter_mut().zip(&key.residuals) {
                column.push(value);
            }
            samples.counts.push(remaining);
            if remaining != MAX_COUNT {
                self.register(index, positions);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn empty_samples(properties: &[usize], predictors: usize) -> TreeSamples {
        TreeSamples {
            properties: properties
                .iter()
                .map(|&p| PropertyColumn::new(p, 255))
                .collect(),
            residuals: vec![Vec::new(); predictors],
            counts: Vec::new(),
            thresholds: vec![Vec::new(); properties.len()],
        }
    }

    fn key(value: u32) -> SampleKey {
        SampleKey {
            properties: vec![value],
            residuals: vec![ResidualToken {
                token: 239,
                nbits: 28,
            }],
        }
    }

    fn assert_live_slots(table: &DedupTable, samples: &TreeSamples) {
        let mut seen = vec![false; samples.counts.len()];
        for (slot, &index) in table.slots.iter().enumerate() {
            if index == UNUSED {
                continue;
            }
            let index = index as usize;
            assert!(!seen[index], "stale alias for record {index}");
            seen[index] = true;
            assert!(samples.counts[index] != 0 && samples.counts[index] < MAX_COUNT);
            assert!(
                table
                    .positions(table.row_hashes(samples, index))
                    .contains(&slot)
            );
        }
    }

    #[test]
    fn sample_table_capacity_matches_source_floor_rounding_and_checked_limits() {
        for (records, slots) in [
            (0, 0),
            (1, 1),
            (2, 4),
            (3, 4),
            (5, 8),
            (6, 16),
            (341, 512),
            (342, 1024),
            (699_051, 1 << 20),
            (699_052, 1 << 21),
        ] {
            assert_eq!(table_slots(records).unwrap(), slots);
        }
        assert!(matches!(table_slots(usize::MAX), Err(Error::Limit(_))));
        if let Some(too_many) = (UNUSED as usize).checked_add(1) {
            assert!(matches!(table_slots(too_many), Err(Error::Limit(_))));
        }
        let mut samples = empty_samples(&[0], 1);
        let mut table = DedupTable::new(&[0]);
        table.prepare(&mut samples, 0).unwrap();
        table.insert(&mut samples, &key(0), 0).unwrap();
        assert!(table.slots.is_empty());
        assert_eq!(samples.counts.capacity(), 0);
        assert!(matches!(
            table.prepare(&mut samples, usize::MAX),
            Err(Error::Limit(_))
        ));
        assert_eq!(samples.counts.capacity(), 0);
        assert!(table.slots.is_empty());
        let mut values = Vec::<u16>::new();
        reserve_samples(&mut values, 4).unwrap();
        let old_capacity = values.capacity();
        reserve_samples(&mut values, old_capacity + 1).unwrap();
        assert!(values.capacity() >= old_capacity + old_capacity / 2);
    }

    #[test]
    fn sample_hashes_match_source_golden_static_order_and_exact_identity() {
        let key = SampleKey {
            properties: vec![0, 800, 255],
            residuals: vec![
                ResidualToken { token: 7, nbits: 3 },
                ResidualToken {
                    token: 128,
                    nbits: 16,
                },
            ],
        };
        let mut table = DedupTable::new(&[0, 1, 6]);
        let expected = [0x610a_7a6b_7605_8ae6, 0xce04_e503_cb3e_0cee];
        assert_eq!(table.key_hashes(&key), expected);
        let mut reordered = key.clone();
        reordered.properties = vec![255, 800, 0];
        assert_eq!(DedupTable::new(&[6, 1, 0]).key_hashes(&reordered), expected);
        let mut samples = empty_samples(&[0, 1, 6], 2);
        table.prepare(&mut samples, 8).unwrap();
        table.insert(&mut samples, &key, 1).unwrap();
        assert_eq!(table.row_hashes(&samples, 0), expected);
        assert!(DedupTable::same_sample(&samples, &key, 0));
        for field in 0..7 {
            let mut different = key.clone();
            match field {
                0..=2 => different.properties[field] ^= 1,
                3..=4 => different.residuals[field - 3].token ^= 1,
                _ => different.residuals[field - 5].nbits ^= 1,
            }
            assert!(!DedupTable::same_sample(&samples, &different, 0));
            table.insert(&mut samples, &different, 1).unwrap();
        }
        assert_eq!(samples.counts, [1; 8]);
        assert_live_slots(&table, &samples);
    }

    #[test]
    fn sample_u16_records_split_full_u32_weights_without_clamping() {
        for weight in [0, 1, 65_534, 65_535, 65_536, 131_070, u32::MAX] {
            let raw = Sample {
                properties: vec![0],
                residuals: vec![i32::MIN],
                weight,
            };
            let capacity = split_record_capacity(std::iter::once(&raw)).unwrap();
            assert_eq!(capacity, weight.div_ceil(65_535) as usize);
            let mut samples = empty_samples(&[0], 1);
            let mut table = DedupTable::new(&[0]);
            table.prepare(&mut samples, capacity).unwrap();
            table.insert(&mut samples, &key(u32::MAX), weight).unwrap();
            assert_eq!(samples.counts.len(), capacity);
            assert!(samples.counts.iter().all(|&count| count != 0));
            assert_eq!(
                samples.counts.iter().map(|&n| u64::from(n)).sum::<u64>(),
                u64::from(weight)
            );
            assert_eq!(
                std::mem::size_of_val(samples.counts.as_slice()),
                capacity * 2
            );
            assert!(
                samples.properties[0]
                    .to_wide()
                    .iter()
                    .all(|&v| v == u32::MAX)
            );
            assert_live_slots(&table, &samples);
        }
    }

    #[test]
    fn sample_saturation_rebuilds_clear_aliases_and_preserve_weight() {
        let mut samples = empty_samples(&[0], 1);
        let mut table = DedupTable::new(&[0]);
        table.prepare(&mut samples, 1).unwrap();
        table.insert(&mut samples, &key(7), 65_534).unwrap();
        assert_eq!(table.slots.len(), 1);
        table.prepare(&mut samples, 2).unwrap();
        assert_eq!(table.slots.len(), 4);
        assert_live_slots(&table, &samples);
        table.insert(&mut samples, &key(7), 2).unwrap();
        assert_eq!(samples.counts, [65_535, 1]);
        assert_live_slots(&table, &samples);
        table.prepare(&mut samples, 512).unwrap();
        assert_eq!(table.slots.len(), 1024);
        table.insert(&mut samples, &key(7), 65_535).unwrap();
        assert_eq!(samples.counts, [65_535, 65_535, 1]);
        assert_live_slots(&table, &samples);
        let pointers: Vec<_> = samples.properties.iter().map(|p| p.as_ptr()).collect();
        table.prepare(&mut samples, 0).unwrap();
        assert_eq!(table.slots.len(), 4);
        assert_eq!(
            pointers,
            samples
                .properties
                .iter()
                .map(|p| p.as_ptr())
                .collect::<Vec<_>>()
        );
        assert_live_slots(&table, &samples);
        let slots = table.slots.clone();
        table.prepare(&mut samples, 0).unwrap();
        assert_eq!(table.slots, slots);
        table.insert(&mut samples, &key(7), 9).unwrap();
        assert_eq!(samples.counts, [65_535, 65_535, 10]);
        assert_eq!(
            samples.counts.iter().map(|&n| u32::from(n)).sum::<u32>(),
            131_080
        );
        assert_live_slots(&table, &samples);
    }

    #[test]
    fn sample_two_probe_collisions_retain_records_without_eviction_or_extra_probes() {
        let mut samples = empty_samples(&[0], 1);
        let mut table = DedupTable::new(&[0]);
        table.prepare(&mut samples, 4).unwrap();
        let mut groups = HashMap::<[usize; 2], Vec<SampleKey>>::new();
        let colliding = (0..4096u32)
            .find_map(|value| {
                let key = key(value.wrapping_mul(0x1_0001));
                let positions = table.positions(table.key_hashes(&key));
                if positions[0] == positions[1] {
                    return None;
                }
                let group = groups.entry(positions).or_default();
                group.push(key);
                (group.len() == 3).then(|| group.clone())
            })
            .expect("three different keys sharing two slots");
        table.insert(&mut samples, &colliding[0], 1).unwrap();
        table.insert(&mut samples, &colliding[1], 1).unwrap();
        let slots = table.slots.clone();
        table.insert(&mut samples, &colliding[2], 1).unwrap();
        table.insert(&mut samples, &colliding[2], 9).unwrap();
        assert_eq!(table.slots, slots);
        assert_eq!(samples.counts, [1, 1, 1, 9]);
        assert_eq!(
            samples.properties[0].value(2),
            samples.properties[0].value(3)
        );
        assert_eq!(table.slots.iter().filter(|&&i| i != UNUSED).count(), 2);
        assert_live_slots(&table, &samples);
    }

    #[test]
    fn sample_weighted_merge_fills_both_matching_probe_records_in_order() {
        let mut samples = empty_samples(&[0], 1);
        let mut table = DedupTable::new(&[0]);
        table.prepare(&mut samples, 4).unwrap();
        let key = (0..4096)
            .map(|value| key(value * 0x1_0001))
            .find(|key| {
                let p = table.positions(table.key_hashes(key));
                p[0] != p[1]
            })
            .unwrap();
        // Collision misses can leave identical live rows. A subsequent resize
        // re-registers both, without merging them, in their original order.
        for count in [65_530, 65_532] {
            samples.properties[0].push(key.properties[0]);
            samples.residuals[0].push(key.residuals[0]);
            samples.counts.push(count);
        }
        table.prepare(&mut samples, 1).unwrap();
        let positions = table.positions(table.key_hashes(&key));
        // The new four-slot table might map the hashes to one slot. Retain the
        // original eight-slot mask for this explicitly two-position fixture.
        if positions[0] == positions[1] {
            table.prepare(&mut samples, 2).unwrap();
        }
        let positions = table.positions(table.key_hashes(&key));
        assert_ne!(positions[0], positions[1]);
        assert_eq!(
            [table.slots[positions[0]], table.slots[positions[1]]],
            [0, 1]
        );
        table.insert(&mut samples, &key, 10).unwrap();
        assert_eq!(samples.counts, [65_535, 65_535, 2]);
        assert_live_slots(&table, &samples);
    }
}
