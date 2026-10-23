// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Bounded ProcessScan/ProcessRestart composition from
//! lib/jxl/jpeg/enc_jpeg_data_reader.cc, frozen at
//! b866b3defec13494bfbaf705db93581cda118e37.
//!
//! This extracts quantized coefficients, not pixels or a JXL codestream.
//! Archived entropy remains intact; `JpegData::reconstruct` still replays it.

use std::fmt;

use super::bitstream::{EntropyBitReader, PaddingBits, PaddingPolicy};
use super::coefficients::{
    BlockState, CoefficientError, ScanBand, decode_initial_block, refine_block,
};
use super::data::{
    BLOCK_SIZE, CodingProcess, ExtraZeroRun, HuffmanCode, JpegData, JpegError, JpegRecord,
    MAX_COMPONENTS, MAX_HUFFMAN_TABLES,
};
use super::huffman::HuffmanTable;
use super::reader::{ReadLimits, read_jpeg_with_limits};
use super::scan_progression::ScanProgression;

/// Structural limits also apply. Storage and total scan work are checked before
/// ANY coefficient allocation, including padded blocks of every component.
#[derive(Clone, Debug)]
pub struct CoefficientReadLimits {
    pub structural: ReadLimits,
    /// Sum of padded component coefficient buffers, in bytes (i16 elements).
    pub max_coefficient_bytes: usize,
    /// Sum of decoded blocks over all scans, including refinement passes.
    pub max_block_visits: usize,
    /// Combined reset-point and extra-zero-run entries over all scans.
    pub max_hint_entries: usize,
    /// Stored alignment bits over all restart and scan boundaries.
    pub max_padding_bits: usize,
    pub padding_policy: PaddingPolicy,
}

impl Default for CoefficientReadLimits {
    fn default() -> Self {
        Self {
            structural: ReadLimits::default(),
            max_coefficient_bytes: 256 * 1024 * 1024,
            max_block_visits: 64 * 1024 * 1024,
            max_hint_entries: 1_048_576,
            max_padding_bits: 8 * 1024 * 1024,
            padding_policy: PaddingPolicy::Preserve,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoefficientReadError {
    Jpeg(JpegError),
    /// Block index is scan traversal order, NOT the component storage index.
    /// At a boundary it identifies the next block (or the scan block count).
    /// Offset is the file-relative reader position at the failed operation.
    Scan {
        scan_index: usize,
        block_index: u32,
        offset: usize,
        error: CoefficientError,
    },
}

impl fmt::Display for CoefficientReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jpeg(error) => error.fmt(f),
            Self::Scan {
                scan_index,
                block_index,
                offset,
                error,
            } => write!(
                f,
                "JPEG scan {scan_index}, block {block_index}, byte {offset}: {error}"
            ),
        }
    }
}

impl std::error::Error for CoefficientReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Jpeg(error) => Some(error),
            Self::Scan { error, .. } => Some(error),
        }
    }
}

impl From<JpegError> for CoefficientReadError {
    fn from(error: JpegError) -> Self {
        Self::Jpeg(error)
    }
}

/// Decode 8-bit Huffman SOF0/1/2 JPEG into block-major, natural-order i16
/// coefficients. No dequantization, IDCT, pixel encoding or JXL encoding occurs.
/// Noninterleaved scans leave untransmitted padded blocks zero; interleaved
/// scans decode their actual padding blocks. Table indices remain SOS snapshots.
pub fn read_coefficients(input: &[u8]) -> Result<JpegData, CoefficientReadError> {
    read_coefficients_with_limits(input, &CoefficientReadLimits::default())
}

/// Like `read_coefficients`, with explicit storage/work/metadata budgets.
/// On error no partial coefficient image is returned. Progressive Al is limited
/// to 10 as in the frozen source. Skipped refinement levels remain tolerated,
/// but refinement before initialization and missing component DC are rejected.
/// Untransmitted AC bands and low planes are not required to be present.
/// Exact alignment bits are appended MSB-first to `padding_bits`, including
/// all-one padding; `has_zero_padding_bit` marks nonstandard padding.
pub fn read_coefficients_with_limits(
    input: &[u8],
    limits: &CoefficientReadLimits,
) -> Result<JpegData, CoefficientReadError> {
    let mut jpeg = read_jpeg_with_limits(input, &limits.structural)?;
    let lengths = preflight(&jpeg, limits)?;
    for (component, &length) in jpeg.components.iter_mut().zip(&lengths) {
        component
            .coefficients
            .try_reserve_exact(length)
            .map_err(|_| limit(0, "coefficient allocation"))?;
        component.coefficients.resize(length, 0);
    }
    let mut hints = 0;
    for index in 0..jpeg.records.len() {
        // Read original input directly, including the terminating marker. No
        // synthetic sentinel or copied scan-sized temporary is needed.
        if let JpegRecord::Entropy(entropy) = &jpeg.records[index] {
            let (scan, start, end) = (
                entropy.scan_index,
                entropy.offset,
                entropy.offset + entropy.bytes.len(),
            );
            decode_scan(input, &mut jpeg, scan, start, end, limits, &mut hints)?;
        }
    }
    Ok(jpeg)
}

fn invalid(offset: usize, reason: &'static str) -> JpegError {
    JpegError::Invalid { offset, reason }
}

fn limit(offset: usize, resource: &'static str) -> JpegError {
    JpegError::LimitExceeded { offset, resource }
}

fn scan_error(
    scan_index: usize,
    block_index: u32,
    offset: usize,
    error: impl Into<CoefficientError>,
) -> CoefficientReadError {
    CoefficientReadError::Scan {
        scan_index,
        block_index,
        offset,
        error: error.into(),
    }
}

fn preflight(
    jpeg: &JpegData,
    limits: &CoefficientReadLimits,
) -> Result<[usize; MAX_COMPONENTS], JpegError> {
    let mut lengths = [0; MAX_COMPONENTS];
    let mut storage = 0usize;
    for (index, component) in jpeg.components.iter().enumerate() {
        let length = u64::from(component.width_in_blocks)
            .checked_mul(u64::from(component.height_in_blocks))
            .and_then(|blocks| blocks.checked_mul(BLOCK_SIZE as u64))
            .and_then(|length| usize::try_from(length).ok())
            .ok_or_else(|| limit(0, "coefficient dimensions"))?;
        storage = length
            .checked_mul(std::mem::size_of::<i16>())
            .and_then(|bytes| storage.checked_add(bytes))
            .ok_or_else(|| limit(0, "coefficient bytes"))?;
        lengths[index] = length;
    }
    if storage > limits.max_coefficient_bytes {
        return Err(limit(0, "coefficient bytes"));
    }
    let progressive = jpeg.coding_process == Some(CodingProcess::Progressive);
    let mut progression =
        ScanProgression::new(jpeg.components.len()).map_err(|reason| invalid(0, reason))?;
    let mut visits = 0usize;
    for record in &jpeg.records {
        let JpegRecord::Entropy(entropy) = record else {
            continue;
        };
        let scan = &jpeg.scan_info[entropy.scan_index];
        let offset = entropy.offset;
        if scan.al > 10 {
            return Err(JpegError::Unsupported {
                offset,
                reason: "coefficient decoding supports Al <= 10",
            });
        }
        for component in &scan.components {
            let coverage = progression
                .coverage(component.component_index)
                .expect("structural reader validated component index");
            if scan.ss > 0 && coverage[0] == 0 {
                return Err(invalid(offset, "JPEG AC scan precedes component DC"));
            }
            if scan.ah != 0 && coverage[scan.ss as usize..=scan.se as usize].contains(&0) {
                return Err(invalid(
                    offset,
                    "JPEG refinement precedes initial coefficient scan",
                ));
            }
        }
        progression
            .apply(scan, progressive)
            .map_err(|reason| invalid(offset, reason))?;
        let mcus = u64::from(scan.mcu_cols) * u64::from(scan.mcu_rows);
        let blocks_per_mcu: u64 = if scan.components.len() == 1 {
            1
        } else {
            scan.components
                .iter()
                .map(|sc| {
                    let c = &jpeg.components[sc.component_index];
                    u64::from(c.h_samp_factor) * u64::from(c.v_samp_factor)
                })
                .sum()
        };
        let blocks = mcus
            .checked_mul(blocks_per_mcu)
            .and_then(|blocks| usize::try_from(blocks).ok())
            .ok_or_else(|| limit(offset, "scan block visits"))?;
        // Source EOB counters and scan block indices are signed 32-bit values.
        if blocks > i32::MAX as usize {
            return Err(limit(offset, "scan block indices"));
        }
        visits = visits
            .checked_add(blocks)
            .ok_or_else(|| limit(offset, "scan block visits"))?;
        if visits > limits.max_block_visits {
            return Err(limit(offset, "scan block visits"));
        }
        let restarts = if scan.restart_interval == 0 {
            0
        } else {
            (mcus - 1) / u64::from(scan.restart_interval)
        };
        if restarts != entropy.restart_markers.len() as u64 {
            return Err(invalid(
                offset,
                "restart marker count does not match scan MCU interval",
            ));
        }
    }
    for index in 0..jpeg.components.len() {
        if progression.coverage(index).expect("valid component")[0] == 0 {
            return Err(invalid(0, "JPEG component has no DC scan"));
        }
    }
    Ok(lengths)
}

fn build_table(table: &HuffmanCode) -> Result<HuffmanTable, CoefficientError> {
    // The structural reader has validated each count, symbol and total.
    let counts: [u8; 16] = std::array::from_fn(|i| table.counts[i + 1] as u8);
    let count: usize = counts.iter().map(|&value| usize::from(value)).sum();
    let mut symbols = [0u8; 256];
    for (symbol, &value) in symbols[..count].iter_mut().zip(&table.values) {
        *symbol = value as u8;
    }
    Ok(HuffmanTable::new(&counts, &symbols[..count])?)
}

fn save_padding(
    bits: PaddingBits,
    padding: &mut Vec<u8>,
    has_zero: &mut bool,
    limits: &CoefficientReadLimits,
    offset: usize,
) -> Result<(), JpegError> {
    let count = usize::from(bits.bit_count());
    if padding
        .len()
        .checked_add(count)
        .is_none_or(|n| n > limits.max_padding_bits)
    {
        return Err(limit(offset, "padding bits"));
    }
    padding
        .try_reserve(count)
        .map_err(|_| limit(offset, "padding allocation"))?;
    *has_zero |= !bits.is_all_ones();
    for bit in (0..bits.bit_count()).rev() {
        padding.push((bits.value() >> bit) & 1);
    }
    Ok(())
}

fn save_hint<T>(
    hints: &mut Vec<T>,
    value: T,
    total: &mut usize,
    limits: &CoefficientReadLimits,
    offset: usize,
) -> Result<(), JpegError> {
    if *total >= limits.max_hint_entries {
        return Err(limit(offset, "reconstruction hint entries"));
    }
    hints
        .try_reserve(1)
        .map_err(|_| limit(offset, "reconstruction hint allocation"))?;
    hints.push(value);
    *total += 1;
    Ok(())
}

fn decode_scan(
    input: &[u8],
    jpeg: &mut JpegData,
    scan_index: usize,
    start: usize,
    end: usize,
    limits: &CoefficientReadLimits,
    hints: &mut usize,
) -> Result<(), CoefficientReadError> {
    let scan = &jpeg.scan_info[scan_index];
    let mut dc: [Option<HuffmanTable>; MAX_HUFFMAN_TABLES] = std::array::from_fn(|_| None);
    let mut ac: [Option<HuffmanTable>; MAX_HUFFMAN_TABLES] = std::array::from_fn(|_| None);
    for component in &scan.components {
        for (needed, slot, indices, tables) in [
            (
                scan.ss == 0 && scan.ah == 0,
                component.dc_table_index,
                &scan.dc_huffman_indices,
                &mut dc,
            ),
            (
                scan.se != 0,
                component.ac_table_index,
                &scan.ac_huffman_indices,
                &mut ac,
            ),
        ] {
            let slot = slot as usize;
            if needed && tables[slot].is_none() {
                let index = indices[slot].expect("structural reader validated active table");
                tables[slot] = Some(
                    build_table(&jpeg.huffman_code[index])
                        .map_err(|error| scan_error(scan_index, 0, start, error))?,
                );
            }
        }
    }
    let JpegData {
        components,
        scan_info,
        padding_bits,
        has_zero_padding_bit,
        ..
    } = jpeg;
    let scan = &mut scan_info[scan_index];
    let mut reader = EntropyBitReader::with_offset(input, start)
        .map_err(|error| scan_error(scan_index, 0, start, error))?;
    let mut predictors = [0i16; MAX_COMPONENTS];
    let mut eob_run = -1;
    let mut block_index = 0u32;
    let mut restart_index = 0u8;
    let interleaved = scan.components.len() > 1;
    let band = ScanBand {
        start: scan.ss as u8,
        end: scan.se as u8,
        low_bit: scan.al as u8,
    };
    for my in 0..scan.mcu_rows {
        for mx in 0..scan.mcu_cols {
            let mcu = u64::from(my) * u64::from(scan.mcu_cols) + u64::from(mx);
            if mcu != 0 && scan.restart_interval != 0 && mcu % u64::from(scan.restart_interval) == 0
            {
                if eob_run > 0 {
                    return Err(invalid(
                        reader.position(),
                        "JPEG EOB run crosses restart boundary",
                    )
                    .into());
                }
                let boundary = reader
                    .consume_restart(restart_index, limits.padding_policy)
                    .map_err(|error| {
                        scan_error(scan_index, block_index, reader.position(), error)
                    })?;
                save_padding(
                    boundary.padding,
                    padding_bits,
                    has_zero_padding_bit,
                    limits,
                    boundary.marker.offset,
                )?;
                predictors.fill(0);
                eob_run = -1;
                restart_index = (restart_index + 1) & 7;
            }
            for sc in &scan.components {
                let component = &mut components[sc.component_index];
                let (nx, ny) = if interleaved {
                    (component.h_samp_factor, component.v_samp_factor)
                } else {
                    (1, 1)
                };
                for iy in 0..ny {
                    for ix in 0..nx {
                        let x = (mx * nx + ix) as usize;
                        let y = (my * ny + iy) as usize;
                        let base = (y * component.width_in_blocks as usize + x) * BLOCK_SIZE;
                        let block: &mut [i16; BLOCK_SIZE] = (&mut component.coefficients
                            [base..base + BLOCK_SIZE])
                            .try_into()
                            .expect("preflight validated padded geometry");
                        let mut state = BlockState {
                            last_dc: predictors[sc.component_index],
                            eob_run,
                        };
                        let info = if scan.ah == 0 {
                            decode_initial_block(
                                &mut reader,
                                dc[sc.dc_table_index as usize].as_ref(),
                                ac[sc.ac_table_index as usize].as_ref(),
                                band,
                                &mut state,
                                block,
                            )
                        } else {
                            refine_block(
                                &mut reader,
                                ac[sc.ac_table_index as usize].as_ref(),
                                band,
                                &mut state,
                                block,
                            )
                        }
                        .map_err(|error| {
                            scan_error(scan_index, block_index, reader.position(), error)
                        })?;
                        predictors[sc.component_index] = state.last_dc;
                        eob_run = state.eob_run;
                        if info.reset_eob_state {
                            save_hint(
                                &mut scan.reset_points,
                                block_index,
                                hints,
                                limits,
                                reader.position(),
                            )?;
                        }
                        if info.extra_zero_runs != 0 {
                            save_hint(
                                &mut scan.extra_zero_runs,
                                ExtraZeroRun {
                                    block_index,
                                    count: info.extra_zero_runs,
                                },
                                hints,
                                limits,
                                reader.position(),
                            )?;
                        }
                        block_index += 1;
                    }
                }
            }
        }
    }
    if eob_run > 0 {
        return Err(invalid(reader.position(), "JPEG EOB run exceeds scan block count").into());
    }
    let boundary = reader
        .finish_scan(limits.padding_policy)
        .map_err(|error| scan_error(scan_index, block_index, reader.position(), error))?;
    if boundary.marker.offset != end || boundary.marker.restart_index().is_some() {
        return Err(invalid(
            reader.position(),
            "unexpected entropy or restart after final scan MCU",
        )
        .into());
    }
    save_padding(
        boundary.padding,
        padding_bits,
        has_zero_padding_bit,
        limits,
        boundary.marker.offset,
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "coefficient_reader_tests.rs"]
mod tests;
