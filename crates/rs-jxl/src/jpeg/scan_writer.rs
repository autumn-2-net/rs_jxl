// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Coefficient-based JPEG reconstruction from dec_jpeg_data_writer.cc, b866b3d.
//! Original entropy payloads are neither read nor copied. This is not JXL coding.

use std::fmt;

use super::coefficient_writer::{
    DctCodingState, encode_initial_block, encode_refinement_block, encode_sequential_block,
};
use super::coefficients::ScanBand;
use super::data::{
    BLOCK_SIZE, CodingProcess, HuffmanCode, JpegData, JpegError, JpegRecord, MAX_COMPONENTS,
    MAX_HUFFMAN_TABLES,
};
use super::entropy_writer::{EntropyBitWriter, WriteError};
use super::huffman::HuffmanTable;
use super::reader::ReadLimits;
use super::reconstruction_validation::{append_record, reserve, validate};

#[derive(Clone, Debug)]
pub struct WriteLimits {
    pub structural: ReadLimits,
    pub max_output_bytes: usize,
    pub max_block_visits: usize,
    pub max_hint_entries: usize,
    pub max_padding_bits: usize,
    pub max_refinement_bits: usize,
}

impl Default for WriteLimits {
    fn default() -> Self {
        Self {
            structural: ReadLimits::default(),
            max_output_bytes: 256 * 1024 * 1024,
            max_block_visits: 64 * 1024 * 1024,
            max_hint_entries: 1_048_576,
            max_padding_bits: 8 * 1024 * 1024,
            max_refinement_bits: 8 * 1024 * 1024,
        }
    }
}

#[derive(Debug)]
pub enum ReconstructionError {
    Invalid(&'static str),
    Limit(&'static str),
    Structural(JpegError),
    Entropy(WriteError),
    Scan {
        scan: usize,
        block: u32,
        message: String,
    },
}

impl fmt::Display for ReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid JPEG reconstruction: {reason}"),
            Self::Limit(resource) => write!(f, "JPEG reconstruction limit: {resource}"),
            Self::Structural(error) => error.fmt(f),
            Self::Entropy(error) => error.fmt(f),
            Self::Scan {
                scan,
                block,
                message,
            } => write!(f, "JPEG scan {scan}, block {block}: {message}"),
        }
    }
}

impl std::error::Error for ReconstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Structural(error) => Some(error),
            Self::Entropy(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WriteError> for ReconstructionError {
    fn from(error: WriteError) -> Self {
        Self::Entropy(error)
    }
}

/// Re-encode all scans from quantized integer coefficients and reconstruction
/// hints, retaining original marker/metadata bytes. Opaque entropy payloads may
/// be cleared before calling. Edited descriptive headers/tables are rejected if
/// they disagree with their marker records. No partial JPEG is returned on error.
/// This is not JPEG-to-JXL compression and does not contain a JXL decoder.
pub fn reconstruct_from_coefficients(
    jpeg: &JpegData,
    limits: &WriteLimits,
) -> Result<Vec<u8>, ReconstructionError> {
    validate(jpeg, limits)?;
    let mut output = Vec::new();
    let mut padding = 0;
    for record in &jpeg.records {
        if let JpegRecord::Entropy(entropy) = record {
            let bytes = encode_scan(
                jpeg,
                entropy.scan_index,
                &mut padding,
                limits,
                limits.max_output_bytes - output.len(),
            )?;
            reserve(&mut output, bytes.len(), limits.max_output_bytes)?;
            output.extend_from_slice(&bytes);
        } else {
            append_record(&mut output, record, jpeg, limits.max_output_bytes)?;
        }
    }
    if !implicit_one_padding(jpeg) && padding != jpeg.padding_bits.len() {
        return Err(ReconstructionError::Invalid("unused JPEG padding bits"));
    }
    Ok(output)
}

fn table(definition: &HuffmanCode) -> Result<HuffmanTable, ReconstructionError> {
    // Definitions have been matched against the independently reparsed DHTs.
    let counts = std::array::from_fn(|i| definition.counts[i + 1] as u8);
    let count: usize = counts.iter().map(|&n| usize::from(n)).sum();
    let symbols: Vec<_> = definition.values[..count]
        .iter()
        .map(|&n| n as u8)
        .collect();
    HuffmanTable::new(&counts, &symbols)
        .map_err(|_| ReconstructionError::Invalid("JPEG Huffman definition"))
}

fn implicit_one_padding(jpeg: &JpegData) -> bool {
    // jbrd omits padding entirely when all alignment bits are one.
    !jpeg.has_zero_padding_bit && jpeg.padding_bits.is_empty()
}

fn align(
    writer: &mut EntropyBitWriter,
    jpeg: &JpegData,
    position: &mut usize,
    limit: usize,
) -> Result<(), ReconstructionError> {
    let count = writer.padding_bits_needed();
    let end = position
        .checked_add(usize::from(count))
        .ok_or(ReconstructionError::Limit("padding offset"))?;
    if end > limit {
        return Err(ReconstructionError::Limit("padding bits"));
    }
    let value = if implicit_one_padding(jpeg) {
        ((1u16 << count) - 1) as u8
    } else {
        let bits = jpeg
            .padding_bits
            .get(*position..end)
            .ok_or(ReconstructionError::Invalid("missing JPEG padding bits"))?;
        bits.iter().fold(0, |value, bit| (value << 1) | bit)
    };
    writer.align_with(value, count)?;
    *position = end;
    Ok(())
}

fn encode_scan(
    jpeg: &JpegData,
    index: usize,
    padding: &mut usize,
    limits: &WriteLimits,
    max_bytes: usize,
) -> Result<Vec<u8>, ReconstructionError> {
    let scan = &jpeg.scan_info[index];
    let entropy = jpeg
        .entropy_data(index)
        .ok_or(ReconstructionError::Invalid("entropy descriptor"))?;
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
                let definition =
                    indices[slot].ok_or(ReconstructionError::Invalid("missing Huffman table"))?;
                tables[slot] = Some(table(&jpeg.huffman_code[definition])?);
            }
        }
    }
    let mut writer = EntropyBitWriter::new(max_bytes);
    let mut coding = DctCodingState::new(limits.max_refinement_bits);
    let mut predictors = [0i16; MAX_COMPONENTS];
    let mut block_index = 0u32;
    let mut restart_index = 0;
    let mut reset = scan.reset_points.iter().peekable();
    let mut zeros = scan.extra_zero_runs.iter().peekable();
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
                coding
                    .flush(&mut writer)
                    .map_err(|error| ReconstructionError::Scan {
                        scan: index,
                        block: block_index,
                        message: error.to_string(),
                    })?;
                align(&mut writer, jpeg, padding, limits.max_padding_bits)?;
                let restart = &entropy.restart_markers[restart_index];
                writer.write_restart_with_fill((restart_index % 8) as u8, restart.fill_count)?;
                restart_index += 1;
                predictors.fill(0);
            }
            for component in &scan.components {
                let c = &jpeg.components[component.component_index];
                let (nx, ny) = if scan.components.len() > 1 {
                    (c.h_samp_factor, c.v_samp_factor)
                } else {
                    (1, 1)
                };
                for iy in 0..ny {
                    for ix in 0..nx {
                        let base = ((my * ny + iy) as usize * c.width_in_blocks as usize
                            + (mx * nx + ix) as usize)
                            * BLOCK_SIZE;
                        let block: &[i16; BLOCK_SIZE] = c.coefficients[base..base + BLOCK_SIZE]
                            .try_into()
                            .expect("validated coefficient geometry");
                        let contextual =
                            |error: super::coefficient_writer::CoefficientWriteError| {
                                ReconstructionError::Scan {
                                    scan: index,
                                    block: block_index,
                                    message: error.to_string(),
                                }
                            };
                        if reset.peek().is_some_and(|&&at| at == block_index) {
                            coding.flush(&mut writer).map_err(contextual)?;
                            reset.next();
                        }
                        let zero_runs = if zeros
                            .peek()
                            .is_some_and(|hint| hint.block_index == block_index)
                        {
                            zeros.next().unwrap().count
                        } else {
                            0
                        };
                        let dc = dc[component.dc_table_index as usize].as_ref();
                        let ac = ac[component.ac_table_index as usize].as_ref();
                        if jpeg.coding_process != Some(CodingProcess::Progressive) {
                            encode_sequential_block(
                                &mut writer,
                                dc.unwrap(),
                                ac.unwrap(),
                                zero_runs,
                                &mut predictors[component.component_index],
                                block,
                            )
                            .map_err(contextual)?;
                        } else if scan.ah == 0 {
                            encode_initial_block(
                                &mut writer,
                                dc,
                                ac,
                                band,
                                zero_runs,
                                &mut coding,
                                &mut predictors[component.component_index],
                                block,
                            )
                            .map_err(contextual)?;
                        } else {
                            encode_refinement_block(&mut writer, ac, band, &mut coding, block)
                                .map_err(contextual)?;
                        }
                        block_index += 1;
                    }
                }
            }
        }
    }
    coding
        .flush(&mut writer)
        .map_err(|error| ReconstructionError::Scan {
            scan: index,
            block: block_index,
            message: error.to_string(),
        })?;
    align(&mut writer, jpeg, padding, limits.max_padding_bits)?;
    Ok(writer.into_bytes()?)
}

#[cfg(test)]
#[path = "scan_writer_tests.rs"]
mod tests;
