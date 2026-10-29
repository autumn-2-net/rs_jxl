// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_index.h::BuildParallel at b866b3defec13494bfbaf705db93581cda118e37.
//! Per-chunk counting and disjoint stable scatter, without atomics or unsafe.

use super::super::{reserved, workers};
use super::{Slots, SourceConfig, SourceDeepPrefix, SourceRunKeys, filled, hash, run_hash_count};
use crate::Result;

const HASHES: usize = 1 << 15;

pub(super) fn tasks(size: usize, config: &SourceConfig) -> usize {
    if !config.parallel_index_build
        || config.index_run_keys == SourceRunKeys::SharedSuffix
        || config.index_deep_prefix != SourceDeepPrefix::Disabled
    {
        return 1;
    }
    // In addition to source's two count arrays, safe scatter owns two arrays
    // of disjoint bucket slices. Charge their actual Rust representation.
    let scratch = 2 * HASHES * (size_of::<u32>() + size_of::<&mut [u32]>());
    workers::index_tasks(config, (size / 16384).max(1), scratch)
}

struct Part<'a> {
    begin: usize,
    zeros: &'a mut [u32],
    short: &'a mut [u32],
    long: &'a mut [u32],
    short_counts: Vec<u32>,
    long_counts: Vec<u32>,
    short_output: Vec<&'a mut [u32]>,
    long_output: Vec<&'a mut [u32]>,
    tail: usize,
    carry: usize,
}

pub(super) fn build(
    values: &[u32],
    window: usize,
    full: bool,
    run_keys: SourceRunKeys,
    tasks: usize,
) -> Result<(Slots, Slots, Vec<u32>)> {
    debug_assert_ne!(run_keys, SourceRunKeys::SharedSuffix);
    debug_assert!(tasks > 0);
    let size = values.len();
    let mut zeros = filled(size, 0u32)?;
    let mut short_slots = filled(size, 0u32)?;
    let mut long_slots = filled(size, 0u32)?;
    let mut short_positions = filled(size, 0u32)?;
    let mut long_positions = filled(size.saturating_sub(7), 0u32)?;
    let mut parts = reserved(tasks)?;
    let (mut zero_tail, mut short_tail, mut long_tail) = (
        zeros.as_mut_slice(),
        short_slots.as_mut_slice(),
        long_slots.as_mut_slice(),
    );
    for task in 0..tasks {
        let begin = size * task / tasks;
        let count = size * (task + 1) / tasks - begin;
        let (local_zeros, rest) = zero_tail.split_at_mut(count);
        zero_tail = rest;
        let (short, rest) = short_tail.split_at_mut(count);
        short_tail = rest;
        let (long, rest) = long_tail.split_at_mut(count);
        long_tail = rest;
        parts.push(Part {
            begin,
            zeros: local_zeros,
            short,
            long,
            short_counts: filled(HASHES, 0u32)?,
            long_counts: filled(HASHES, 0u32)?,
            short_output: reserved(HASHES)?,
            long_output: reserved(HASHES)?,
            tail: 0,
            carry: 0,
        });
    }
    workers::each(&mut parts, |_, part| {
        let mut count = 0u32;
        for (offset, zero) in part.zeros.iter_mut().enumerate().rev() {
            count = if values[part.begin + offset] == 0 {
                (count + 1).min(window as u32)
            } else {
                0
            };
            *zero = count;
        }
        part.tail = values[part.begin..part.begin + part.zeros.len()]
            .iter()
            .rev()
            .take_while(|&&value| value == 0)
            .count();
    });
    let mut trailing = 0;
    for part in parts.iter_mut().rev() {
        part.carry = trailing;
        let prefix = part.zeros.first().copied().unwrap_or(0) as usize;
        trailing = window.min(if prefix == part.zeros.len() {
            trailing + part.zeros.len()
        } else {
            prefix
        });
    }
    workers::each(&mut parts, |_, part| {
        let tail_begin = part.zeros.len() - part.tail;
        for zero in &mut part.zeros[tail_begin..] {
            *zero = window.min(*zero as usize + part.carry) as u32;
        }
        for (offset, short) in part.short.iter_mut().enumerate() {
            let position = part.begin + offset;
            *short = hash(values, position);
            part.short_counts[*short as usize & (HASHES - 1)] += 1;
            if position + 8 <= size {
                let key =
                    run_hash_count(values, part.zeros[offset], position, window, full, run_keys);
                part.long[offset] = key;
                part.long_counts[key as usize & (HASHES - 1)] += 1;
            }
        }
    });
    let mut short_offsets = filled(HASHES + 1, 0u32)?;
    let mut long_offsets = filled(HASHES + 1, 0u32)?;
    for part in &parts {
        for (offset, count) in short_offsets[1..].iter_mut().zip(&part.short_counts) {
            *offset += count;
        }
        for (offset, count) in long_offsets[1..].iter_mut().zip(&part.long_counts) {
            *offset += count;
        }
    }
    for hash in 1..=HASHES {
        short_offsets[hash] += short_offsets[hash - 1];
        long_offsets[hash] += long_offsets[hash - 1];
    }
    // Source-order chunks receive disjoint ranges within EVERY bucket. Owning
    // slices lets scatter write concurrently without shared mutable pointers.
    let mut short_tail = short_positions.as_mut_slice();
    let mut long_tail = long_positions.as_mut_slice();
    for (hash, (&short_start, &long_start)) in short_offsets[..HASHES]
        .iter()
        .zip(&long_offsets[..HASHES])
        .enumerate()
    {
        let mut short_next = short_start;
        let mut long_next = long_start;
        for part in &mut parts {
            let short_count = part.short_counts[hash];
            let long_count = part.long_counts[hash];
            let (output, rest) = short_tail.split_at_mut(short_count as usize);
            short_tail = rest;
            part.short_output.push(output);
            let (output, rest) = long_tail.split_at_mut(long_count as usize);
            long_tail = rest;
            part.long_output.push(output);
            part.short_counts[hash] = short_next;
            part.long_counts[hash] = long_next;
            short_next += short_count;
            long_next += long_count;
        }
    }
    workers::each(&mut parts, |_, part| {
        for (offset, short) in part.short.iter_mut().enumerate() {
            let position = part.begin + offset;
            let hash = *short as usize & (HASHES - 1);
            *short = scatter(
                &mut part.short_output[hash],
                &mut part.short_counts[hash],
                position,
            );
            if position + 8 <= size {
                let hash = part.long[offset] as usize & (HASHES - 1);
                part.long[offset] = scatter(
                    &mut part.long_output[hash],
                    &mut part.long_counts[hash],
                    position,
                );
            }
        }
    });
    drop(parts);
    Ok((
        Slots {
            offsets: short_offsets,
            slots: short_slots,
            positions: short_positions,
        },
        Slots {
            offsets: long_offsets,
            slots: long_slots,
            positions: long_positions,
        },
        zeros,
    ))
}

fn scatter(output: &mut &mut [u32], next: &mut u32, position: usize) -> u32 {
    let (slot, rest) = std::mem::take(output)
        .split_first_mut()
        .expect("counted index slot");
    *slot = position as u32;
    *output = rest;
    let result = *next;
    *next += 1;
    result
}

#[cfg(test)]
mod tests {
    use super::super::run_hash;
    use super::*;

    #[test]
    fn chunk_carry_and_stable_scatter_equal_serial_even_when_runs_saturate() {
        let mut values: Vec<u32> = (0..49157).map(|i| (i % 173) + 1).collect();
        values[7000..45000].fill(0);
        values[49140..].fill(0);
        let short = Slots::build(values.len(), values.len(), |p| hash(&values, p)).unwrap();
        for window in [32, 65536] {
            let mut zeros = vec![0u32; values.len()];
            let mut count = 0;
            for (i, zero) in zeros.iter_mut().enumerate().rev() {
                count = if values[i] == 0 {
                    (count + 1).min(window as u32)
                } else {
                    0
                };
                *zero = count;
            }
            for run_keys in [SourceRunKeys::Disabled, SourceRunKeys::LengthAndSuffix] {
                for full in [false, true] {
                    let long = Slots::build(values.len(), values.len() - 7, |p| {
                        run_hash(&values, &zeros, p, window, full, run_keys)
                    })
                    .unwrap();
                    for tasks in [1, 2, 3, 7] {
                        let (actual_short, actual_long, actual_zeros) =
                            build(&values, window, full, run_keys, tasks).unwrap();
                        assert_eq!(actual_zeros, zeros, "window={window}, tasks={tasks}");
                        assert_eq!(actual_short.offsets, short.offsets);
                        assert_eq!(actual_short.slots, short.slots);
                        assert_eq!(actual_short.positions, short.positions);
                        assert_eq!(actual_long.offsets, long.offsets);
                        assert_eq!(actual_long.slots, long.slots);
                        assert_eq!(actual_long.positions, long.positions);
                    }
                }
            }
        }
    }
}
