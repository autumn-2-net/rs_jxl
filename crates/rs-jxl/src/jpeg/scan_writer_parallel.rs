// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Parallel JPEG reconstruction. Only validated scan geometry reaches this module.
//! Independent sequential fragments retain DC history and join BEFORE stuffing;
//! independent progressive scans are eligible only with implicit one padding.

use super::*;
use rayon::prelude::*;

const MIN_MCUS: usize = 1024;

struct Fragment {
    bytes: Vec<u8>,
    tail_bits: u8,
    padding_bits: usize,
}

fn partition_end(total: usize, part: usize, parts: usize) -> usize {
    // A valid total can fit usize while the intermediate product does not.
    (total as u128 * part as u128 / parts as u128) as usize
}

pub(super) fn sequential(
    jpeg: &JpegData,
    index: usize,
    padding: &mut usize,
    limits: &WriteLimits,
    max_bytes: usize,
    pool: &rayon::ThreadPool,
) -> Result<Option<Vec<u8>>, ReconstructionError> {
    let scan = &jpeg.scan_info[index];
    let mcus = scan.mcu_rows as usize * scan.mcu_cols as usize;
    let interval = scan.restart_interval as usize;
    let units = if interval == 0 {
        mcus
    } else {
        mcus.div_ceil(interval)
    };
    let jobs = pool.current_num_threads().min(mcus / MIN_MCUS).min(units);
    if jpeg.coding_process == Some(CodingProcess::Progressive)
        || (interval != 0 && !implicit_one_padding(jpeg))
        || jobs < 2
    {
        return Ok(None);
    }
    let _stage = crate::profile::scope("decode.jpeg.parallel_sequential");
    let mut dc: [Option<HuffmanTable>; MAX_HUFFMAN_TABLES] = std::array::from_fn(|_| None);
    let mut ac: [Option<HuffmanTable>; MAX_HUFFMAN_TABLES] = std::array::from_fn(|_| None);
    for component in &scan.components {
        for (slot, indices, tables) in [
            (component.dc_table_index, &scan.dc_huffman_indices, &mut dc),
            (component.ac_table_index, &scan.ac_huffman_indices, &mut ac),
        ] {
            let slot = slot as usize;
            if tables[slot].is_none() {
                tables[slot] = Some(table(&jpeg.huffman_code[indices[slot].unwrap()])?);
            }
        }
    }
    // Temporary fragments together cannot exceed the remaining output budget.
    // A skewed or very tight budget falls back to the bounded serial writer.
    let fragments: Vec<_> = pool.install(|| {
        (0..jobs)
            .into_par_iter()
            .map(|job| {
                fragment(
                    jpeg,
                    index,
                    &dc,
                    &ac,
                    if interval == 0 {
                        partition_end(mcus, job, jobs)..partition_end(mcus, job + 1, jobs)
                    } else {
                        partition_end(units, job, jobs) * interval
                            ..(partition_end(units, job + 1, jobs) * interval).min(mcus)
                    },
                    max_bytes / jobs,
                )
            })
            .collect()
    });
    if fragments.iter().any(|result| {
        matches!(
            result,
            Err(ReconstructionError::Entropy(WriteError::OutputLimit { .. }))
                | Err(ReconstructionError::Scan { .. })
        )
    }) {
        // The serial path preserves detailed scan errors and their ordering too.
        return Ok(None);
    }
    if interval != 0 {
        let mut output = Vec::new();
        for fragment in fragments {
            let fragment = fragment?;
            *padding = padding
                .checked_add(fragment.padding_bits)
                .ok_or(ReconstructionError::Limit("padding offset"))?;
            if *padding > limits.max_padding_bits {
                return Err(ReconstructionError::Limit("padding bits"));
            }
            reserve(&mut output, fragment.bytes.len(), max_bytes)?;
            output.extend_from_slice(&fragment.bytes);
        }
        return Ok(Some(output));
    }
    let mut writer = EntropyBitWriter::new(max_bytes);
    for fragment in fragments {
        append_fragment(&mut writer, &fragment?)?;
    }
    align(&mut writer, jpeg, padding, limits.max_padding_bits)?;
    Ok(Some(writer.into_bytes()?))
}

fn fragment(
    jpeg: &JpegData,
    index: usize,
    dc: &[Option<HuffmanTable>; MAX_HUFFMAN_TABLES],
    ac: &[Option<HuffmanTable>; MAX_HUFFMAN_TABLES],
    mcus: std::ops::Range<usize>,
    max_bytes: usize,
) -> Result<Fragment, ReconstructionError> {
    let scan = &jpeg.scan_info[index];
    let interleaved = scan.components.len() > 1;
    let interval = scan.restart_interval as usize;
    debug_assert!(interval == 0 || mcus.start.is_multiple_of(interval));
    let mut predictors = [0i16; MAX_COMPONENTS];
    let mut blocks_per_mcu = 0;
    for component in &scan.components {
        let c = &jpeg.components[component.component_index];
        let (nx, ny) = if interleaved {
            (c.h_samp_factor as usize, c.v_samp_factor as usize)
        } else {
            (1, 1)
        };
        blocks_per_mcu += nx * ny;
        if mcus.start != 0 && interval == 0 {
            let previous = mcus.start - 1;
            let x = (previous % scan.mcu_cols as usize + 1) * nx - 1;
            let y = (previous / scan.mcu_cols as usize + 1) * ny - 1;
            predictors[component.component_index] =
                c.coefficients[(y * c.width_in_blocks as usize + x) * BLOCK_SIZE];
        }
    }
    let mut block_index = (mcus.start * blocks_per_mcu) as u32;
    let mut hint = scan
        .extra_zero_runs
        .partition_point(|hint| hint.block_index < block_index);
    let mut writer = EntropyBitWriter::new(max_bytes);
    let start = mcus.start;
    let mut padding_bits = 0;
    let entropy = jpeg.entropy_data(index).unwrap();
    for mcu in mcus {
        if mcu != 0 && interval != 0 && mcu.is_multiple_of(interval) {
            if mcu != start {
                padding_bits += usize::from(writer.padding_bits_needed());
                writer.align_ones()?;
            }
            let index = mcu / interval - 1;
            writer.write_restart_with_fill(
                (index % 8) as u8,
                entropy.restart_markers[index].fill_count,
            )?;
            predictors.fill(0);
        }
        let mx = mcu % scan.mcu_cols as usize;
        let my = mcu / scan.mcu_cols as usize;
        for component in &scan.components {
            let c = &jpeg.components[component.component_index];
            let (nx, ny) = if interleaved {
                (c.h_samp_factor as usize, c.v_samp_factor as usize)
            } else {
                (1, 1)
            };
            for iy in 0..ny {
                for ix in 0..nx {
                    let base =
                        ((my * ny + iy) * c.width_in_blocks as usize + mx * nx + ix) * BLOCK_SIZE;
                    let block = c.coefficients[base..base + BLOCK_SIZE].try_into().unwrap();
                    let zeros = if scan
                        .extra_zero_runs
                        .get(hint)
                        .is_some_and(|entry| entry.block_index == block_index)
                    {
                        let count = scan.extra_zero_runs[hint].count;
                        hint += 1;
                        count
                    } else {
                        0
                    };
                    encode_sequential_block(
                        &mut writer,
                        dc[component.dc_table_index as usize].as_ref().unwrap(),
                        ac[component.ac_table_index as usize].as_ref().unwrap(),
                        zeros,
                        &mut predictors[component.component_index],
                        block,
                    )
                    .map_err(|error| ReconstructionError::Scan {
                        scan: index,
                        block: block_index,
                        message: error.to_string(),
                    })?;
                    block_index += 1;
                }
            }
        }
    }
    let tail_bits = if interval == 0 {
        writer.pending_bits()
    } else {
        0
    };
    if interval == 0 {
        writer.align_with(0, writer.padding_bits_needed())?;
    } else {
        padding_bits += usize::from(writer.padding_bits_needed());
        writer.align_ones()?;
    }
    Ok(Fragment {
        bytes: writer.into_bytes()?,
        tail_bits,
        padding_bits,
    })
}

fn append_fragment(writer: &mut EntropyBitWriter, fragment: &Fragment) -> Result<(), WriteError> {
    let mut offset = 0;
    let complete_end = fragment.bytes.len() - usize::from(fragment.tail_bits != 0);
    while offset < fragment.bytes.len() {
        if complete_end.saturating_sub(offset) >= 8 {
            let word = u64::from_be_bytes(fragment.bytes[offset..offset + 8].try_into().unwrap());
            // Stuffing-free runs can be shifted and re-stuffed word-wise. The
            // writer still handles any 0xff formed across a fragment boundary.
            if ((!word).wrapping_sub(0x0101_0101_0101_0101) & word & 0x8080_8080_8080_8080) == 0 {
                writer.write_bits((word >> 32) as u32, 32)?;
                writer.write_bits(word as u32, 32)?;
                offset += 8;
                continue;
            }
        }
        let byte = fragment.bytes[offset];
        offset += 1;
        if byte == 255 {
            debug_assert_eq!(fragment.bytes.get(offset), Some(&0));
            offset += 1;
        }
        let take = if offset == fragment.bytes.len() && fragment.tail_bits != 0 {
            fragment.tail_bits
        } else {
            8
        };
        writer.write_bits(u32::from(byte >> (8 - take)), take)?;
    }
    Ok(())
}

pub(super) fn progressive(
    jpeg: &JpegData,
    limits: &WriteLimits,
    pool: &rayon::ThreadPool,
) -> Result<Option<Vec<u8>>, ReconstructionError> {
    if jpeg.coding_process != Some(CodingProcess::Progressive)
        || !implicit_one_padding(jpeg)
        || jpeg.scan_info.len() < 2
        || pool.current_num_threads() < 2
        || u64::from(jpeg.width) * u64::from(jpeg.height) < 256 * 1024
    {
        return Ok(None);
    }
    progressive_scans(jpeg, limits, pool)
}

fn progressive_scans(
    jpeg: &JpegData,
    limits: &WriteLimits,
    pool: &rayon::ThreadPool,
) -> Result<Option<Vec<u8>>, ReconstructionError> {
    let _stage = crate::profile::scope("decode.jpeg.parallel_progressive");
    let mut output = Vec::new();
    let mut record_index = 0;
    let mut padding = 0usize;
    // One work queue avoids a barrier between the last long refinement scan
    // and earlier scans. Scratch byte budgets still sum to one output budget.
    let budget = limits.max_output_bytes / jpeg.scan_info.len();
    let results: Vec<_> = pool.install(|| {
        (0..jpeg.scan_info.len())
            .into_par_iter()
            .with_max_len(1)
            .map(|index| {
                let mut used_padding = 0;
                let bytes = encode_scan(jpeg, index, &mut used_padding, limits, budget)?;
                Ok::<_, ReconstructionError>((bytes, used_padding))
            })
            .collect()
    });
    if results.iter().any(Result::is_err) {
        return Ok(None);
    }
    for result in results {
        let (bytes, used_padding) = result?;
        padding = padding
            .checked_add(used_padding)
            .ok_or(ReconstructionError::Limit("padding bits"))?;
        if padding > limits.max_padding_bits {
            return Err(ReconstructionError::Limit("padding bits"));
        }
        loop {
            let record = &jpeg.records[record_index];
            record_index += 1;
            if matches!(record, JpegRecord::Entropy(_)) {
                reserve(&mut output, bytes.len(), limits.max_output_bytes)?;
                output.extend_from_slice(&bytes);
                break;
            }
            append_record(&mut output, record, jpeg, limits.max_output_bytes)?;
        }
    }
    for record in &jpeg.records[record_index..] {
        append_record(&mut output, record, jpeg, limits.max_output_bytes)?;
    }
    Ok(Some(output))
}

#[cfg(test)]
#[path = "scan_writer_parallel_tests.rs"]
mod tests;
