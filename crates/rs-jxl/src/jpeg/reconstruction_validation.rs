// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Preconditions for coefficient-based JPEG writing. Original entropy bytes are
//! deliberately never read, even when checking public mutable structures.

use super::data::{BLOCK_SIZE, CodingProcess, JpegData, JpegRecord, ScanInfo};
use super::reader::read_jpeg_with_limits;
use super::scan_progression::ScanProgression;
use super::scan_writer::{ReconstructionError as Error, WriteLimits};

pub(super) fn reserve(out: &mut Vec<u8>, additional: usize, limit: usize) -> Result<(), Error> {
    if out.len().checked_add(additional).is_none_or(|n| n > limit) {
        return Err(Error::Limit("JPEG output bytes"));
    }
    out.try_reserve(additional)
        .map_err(|_| Error::Limit("JPEG output allocation"))
}

pub(super) fn append_record(
    out: &mut Vec<u8>,
    record: &JpegRecord,
    jpeg: &JpegData,
    limit: usize,
) -> Result<(), Error> {
    match record {
        JpegRecord::Marker(segment) => {
            reserve(
                out,
                segment
                    .data
                    .len()
                    .checked_add(2)
                    .ok_or(Error::Limit("marker bytes"))?,
                limit,
            )?;
            out.extend([0xff, segment.marker]);
            out.extend_from_slice(&segment.data);
        }
        JpegRecord::Fill { count, .. } => {
            reserve(out, *count, limit)?;
            out.resize(out.len() + count, 0xff);
        }
        JpegRecord::InterMarker { index, .. } => {
            let data = jpeg
                .inter_marker_data
                .get(*index)
                .ok_or(Error::Invalid("inter-marker index"))?;
            reserve(out, data.len(), limit)?;
            out.extend_from_slice(data);
        }
        JpegRecord::Tail { .. } => {
            reserve(out, jpeg.tail_data.len(), limit)?;
            out.extend_from_slice(&jpeg.tail_data);
        }
        JpegRecord::Entropy(_) => {}
    }
    Ok(())
}

fn same_scan_header(a: &ScanInfo, b: &ScanInfo) -> bool {
    (
        a.ss,
        a.se,
        a.ah,
        a.al,
        a.restart_interval,
        a.mcu_cols,
        a.mcu_rows,
    ) == (
        b.ss,
        b.se,
        b.ah,
        b.al,
        b.restart_interval,
        b.mcu_cols,
        b.mcu_rows,
    ) && a.components == b.components
        && a.quant_table_indices == b.quant_table_indices
        && a.dc_huffman_indices == b.dc_huffman_indices
        && a.ac_huffman_indices == b.ac_huffman_indices
}

pub(super) fn validate(jpeg: &JpegData, limits: &WriteLimits) -> Result<(), Error> {
    if jpeg.records.len() > limits.structural.max_records
        || jpeg.scan_info.len() > limits.structural.max_scans
        || jpeg.padding_bits.len() > limits.max_padding_bits
    {
        return Err(Error::Limit("JPEG reconstruction metadata"));
    }
    if jpeg.padding_bits.iter().any(|&bit| bit > 1) {
        return Err(Error::Invalid("JPEG padding must contain only bits"));
    }
    // Parse only marker/metadata bytes. Rechecking against their authoritative
    // originals prevents edited geometry/tables from silently mismatching SOS.
    let mut template = Vec::new();
    let mut scan_count = 0;
    let metadata_limit = limits
        .max_output_bytes
        .min(limits.structural.max_input_bytes);
    for record in &jpeg.records {
        if let JpegRecord::Entropy(entropy) = record {
            if entropy.scan_index != scan_count {
                return Err(Error::Invalid("JPEG scan record order"));
            }
            scan_count += 1;
        } else {
            append_record(&mut template, record, jpeg, metadata_limit)?;
        }
    }
    let header = read_jpeg_with_limits(&template, &limits.structural).map_err(Error::Structural)?;
    if scan_count != jpeg.scan_info.len()
        || scan_count != header.scan_info.len()
        || jpeg.width != header.width
        || jpeg.height != header.height
        || jpeg.precision != header.precision
        || jpeg.coding_process != header.coding_process
        || jpeg.quant != header.quant
        || jpeg.huffman_code != header.huffman_code
        || jpeg.components.len() != header.components.len()
    {
        return Err(Error::Invalid(
            "JPEG fields disagree with original marker records",
        ));
    }
    for (actual, expected) in jpeg.components.iter().zip(&header.components) {
        if (
            actual.id,
            actual.h_samp_factor,
            actual.v_samp_factor,
            actual.quant_index,
            actual.width_in_blocks,
            actual.height_in_blocks,
        ) != (
            expected.id,
            expected.h_samp_factor,
            expected.v_samp_factor,
            expected.quant_index,
            expected.width_in_blocks,
            expected.height_in_blocks,
        ) {
            return Err(Error::Invalid("JPEG component fields disagree with SOF"));
        }
        let count = u64::from(expected.width_in_blocks)
            .checked_mul(u64::from(expected.height_in_blocks))
            .and_then(|n| n.checked_mul(BLOCK_SIZE as u64))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Error::Limit("coefficient geometry"))?;
        if actual.coefficients.len() != count {
            return Err(Error::Invalid("JPEG coefficient storage length"));
        }
    }
    let mut visits = 0usize;
    let mut restarts = 0usize;
    let mut hints = 0usize;
    let progressive = jpeg.coding_process == Some(CodingProcess::Progressive);
    let mut progression = ScanProgression::new(jpeg.components.len()).map_err(Error::Invalid)?;
    for (index, (scan, expected)) in jpeg.scan_info.iter().zip(&header.scan_info).enumerate() {
        if !same_scan_header(scan, expected) {
            return Err(Error::Invalid("JPEG scan fields disagree with SOS/tables"));
        }
        for component in &scan.components {
            let coverage = progression
                .coverage(component.component_index)
                .ok_or(Error::Invalid("scan component"))?;
            if scan.ss > 0 && coverage[0] == 0 {
                return Err(Error::Invalid("AC scan precedes DC"));
            }
            if scan.ah != 0 && coverage[scan.ss as usize..=scan.se as usize].contains(&0) {
                return Err(Error::Invalid("refinement precedes initial scan"));
            }
        }
        progression
            .apply(scan, progressive)
            .map_err(Error::Invalid)?;
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
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(Error::Limit("scan block visits"))?;
        visits = visits
            .checked_add(blocks)
            .ok_or(Error::Limit("scan block visits"))?;
        if blocks > i32::MAX as usize || visits > limits.max_block_visits {
            return Err(Error::Limit("scan block visits"));
        }
        let entropy = jpeg
            .entropy_data(index)
            .ok_or(Error::Invalid("missing entropy descriptor"))?;
        let count = if scan.restart_interval == 0 {
            0
        } else {
            (mcus - 1) / u64::from(scan.restart_interval)
        };
        if count != entropy.restart_markers.len() as u64 {
            return Err(Error::Invalid("restart marker count"));
        }
        restarts = restarts
            .checked_add(entropy.restart_markers.len())
            .ok_or(Error::Limit("restart count"))?;
        if restarts > limits.structural.max_restart_markers {
            return Err(Error::Limit("restart count"));
        }
        for (i, restart) in entropy.restart_markers.iter().enumerate() {
            if restart.marker != 0xd0 + (i % 8) as u8 {
                return Err(Error::Invalid("restart marker order"));
            }
            if restart.fill_count > limits.max_output_bytes.saturating_sub(2) {
                return Err(Error::Limit("restart fill bytes"));
            }
        }
        hints = hints
            .checked_add(scan.reset_points.len())
            .and_then(|n| n.checked_add(scan.extra_zero_runs.len()))
            .ok_or(Error::Limit("reconstruction hints"))?;
        if hints > limits.max_hint_entries {
            return Err(Error::Limit("reconstruction hints"));
        }
        if scan.reset_points.iter().any(|&b| b as usize >= blocks)
            || scan.reset_points.windows(2).any(|pair| pair[0] >= pair[1])
            || scan
                .extra_zero_runs
                .iter()
                .any(|hint| hint.block_index as usize >= blocks || !(1..=4).contains(&hint.count))
            || scan
                .extra_zero_runs
                .windows(2)
                .any(|pair| pair[0].block_index >= pair[1].block_index)
            || (scan.ah != 0 && !scan.extra_zero_runs.is_empty())
        {
            return Err(Error::Invalid("JPEG reconstruction hint order/range"));
        }
    }
    for component in 0..jpeg.components.len() {
        if progression
            .coverage(component)
            .ok_or(Error::Invalid("component"))?[0]
            == 0
        {
            return Err(Error::Invalid("JPEG component has no DC scan"));
        }
    }
    Ok(())
}
