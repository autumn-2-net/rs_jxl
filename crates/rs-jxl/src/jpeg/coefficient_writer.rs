// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Block emission and DCTCodingState from lib/jxl/jpeg/dec_jpeg_data_writer.cc
//! and dec_jpeg_serialization_state.h, frozen at
//! b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Quantized integer coefficients only: no IDCT, quantization, marker traversal
//! or JXL encoding. Callers own scan progression, component DC predictors,
//! reset-point hints, restart markers and alignment. Flush at each reset point,
//! restart and scan end BEFORE padding or emitting a marker.
//!
//! Operations can partially advance the supplied writer/predictor on error.
//! The outer reconstruction API must discard them and return no output. State
//! is poisoned after an error to prevent accidental retries of partial flushes.
//! No accumulated-output clone is made; all additional buffering is bounded.

use std::fmt;

use super::coefficients::ScanBand;
use super::data::{BLOCK_SIZE, NATURAL_ORDER};
use super::entropy_writer::{EntropyBatch, EntropyBitWriter, WriteError, magnitude_bits};
use super::huffman::{HuffmanCode, HuffmanTable};
use super::refinement::{self, Masks};
use super::sequential;

const MAX_EOB_RUN: u16 = 0x7fff;
const MAX_BUFFERED_BITS: usize = MAX_EOB_RUN as usize * (BLOCK_SIZE - 1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoefficientWriteError {
    Entropy(WriteError),
    Invalid(&'static str),
    RefinementLimit { limit: usize },
    AllocationFailed,
    FailedState,
}

impl fmt::Display for CoefficientWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entropy(error) => error.fmt(f),
            Self::Invalid(reason) => write!(f, "invalid JPEG coefficient emission: {reason}"),
            Self::RefinementLimit { limit } => {
                write!(f, "JPEG refinement buffer exceeds {limit} bits")
            }
            Self::AllocationFailed => f.write_str("cannot allocate JPEG refinement buffer"),
            Self::FailedState => f.write_str("JPEG coding state cannot be reused after an error"),
        }
    }
}

impl std::error::Error for CoefficientWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Entropy(error) => Some(error),
            _ => None,
        }
    }
}

impl From<WriteError> for CoefficientWriteError {
    fn from(error: WriteError) -> Self {
        Self::Entropy(error)
    }
}

/// One scan's EOB run and packed, MSB-first AC correction bits. The first block
/// of a pending run snapshots only its fifteen possible EOB Huffman codes, not
/// an entire table or an external table reference. No allocation occurs in new.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DctCodingState {
    eob_run: u16,
    eob_codes: [Option<HuffmanCode>; 15],
    refinement_words: Vec<u16>,
    refinement_bits: usize,
    max_refinement_bits: usize,
    failed: bool,
}

impl DctCodingState {
    pub fn new(max_refinement_bits: usize) -> Self {
        Self {
            eob_run: 0,
            eob_codes: [None; 15],
            refinement_words: Vec::new(),
            refinement_bits: 0,
            max_refinement_bits,
            failed: false,
        }
    }

    pub fn pending_eob_run(&self) -> u16 {
        self.eob_run
    }

    pub fn pending_refinement_bits(&self) -> usize {
        self.refinement_bits
    }

    fn healthy(&self) -> Result<(), CoefficientWriteError> {
        if self.failed {
            Err(CoefficientWriteError::FailedState)
        } else {
            Ok(())
        }
    }

    fn finish_operation(
        &mut self,
        result: Result<(), CoefficientWriteError>,
    ) -> Result<(), CoefficientWriteError> {
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Emit the pending EOB symbol/suffix followed by its correction bits.
    /// Does not align, emit markers or reset the caller's DC predictors.
    /// Successful flush leaves both buffers empty and retains bounded capacity.
    pub fn flush(&mut self, writer: &mut EntropyBitWriter) -> Result<(), CoefficientWriteError> {
        self.healthy()?;
        let result = self.flush_inner(writer);
        self.finish_operation(result)
    }

    fn flush_inner(&mut self, writer: &mut EntropyBitWriter) -> Result<(), CoefficientWriteError> {
        if self.eob_run == 0 {
            // Only buffer_end_of_band appends corrections, after starting a run;
            // successful flushes clear the run and corrections together.
            debug_assert_eq!(self.refinement_bits, 0);
            debug_assert!(self.refinement_words.is_empty());
            return Ok(());
        }
        let count = (15 - self.eob_run.leading_zeros()) as u8;
        let symbol = count << 4;
        let code =
            self.eob_codes[usize::from(count)].ok_or(WriteError::MissingHuffmanSymbol(symbol))?;
        let suffix = u32::from(self.eob_run) & ((1u32 << count) - 1);
        let value = (u32::from(code.bits) << count) | suffix;
        let count = code.bit_length + count;
        // Keep large EOB runs on the streaming path; short runs and correction
        // sequences can share one reservation and several word-wide emissions.
        let batched = (2..=64).contains(&self.refinement_words.len())
            && writer
                .try_batch(|batch| {
                    batch.write_bits(value, count)?;
                    let mut remaining = self.refinement_bits;
                    for &word in &self.refinement_words {
                        let count = remaining.min(16) as u8;
                        batch.write_bits(u32::from(word), count)?;
                        remaining -= usize::from(count);
                    }
                    Some(())
                })
                .is_some();
        if !batched {
            writer.write_bits(value, count)?;
            let mut remaining = self.refinement_bits;
            for &word in &self.refinement_words {
                let count = remaining.min(16) as u8;
                writer.write_bits(u32::from(word), count)?;
                remaining -= usize::from(count);
            }
        }
        self.eob_run = 0;
        self.eob_codes = [None; 15];
        self.refinement_words.clear();
        self.refinement_bits = 0;
        Ok(())
    }

    fn buffer_end_of_band(
        &mut self,
        writer: &mut EntropyBitWriter,
        ac: &HuffmanTable,
        bits: u64,
        count: u8,
    ) -> Result<(), CoefficientWriteError> {
        debug_assert!(count <= 63);
        // A table can be replaced at the same address between calls. Compare
        // contents, but only build a snapshot when starting a new run.
        if self.eob_run != 0
            && self
                .eob_codes
                .iter()
                .enumerate()
                .any(|(i, &code)| ac.code((i as u8) << 4) != code)
        {
            return Err(CoefficientWriteError::Invalid(
                "AC table changed inside a pending EOB run",
            ));
        }
        let total = self.refinement_bits.checked_add(usize::from(count)).ok_or(
            CoefficientWriteError::RefinementLimit {
                limit: self.max_refinement_bits,
            },
        )?;
        if total > self.max_refinement_bits {
            return Err(CoefficientWriteError::RefinementLimit {
                limit: self.max_refinement_bits,
            });
        }
        let words = total.div_ceil(16);
        if words > self.refinement_words.capacity() {
            // Geometric growth, capped by both the configured budget and the
            // largest source EOB run. Never flush early to make room: doing so
            // would change the original entropy representation.
            let ceiling = self.max_refinement_bits.min(MAX_BUFFERED_BITS).div_ceil(16);
            let target = words
                .max(self.refinement_words.capacity().saturating_mul(2))
                .min(ceiling);
            self.refinement_words
                .try_reserve_exact(target - self.refinement_words.len())
                .map_err(|_| CoefficientWriteError::AllocationFailed)?;
        }
        if self.eob_run == 0 {
            self.eob_codes = std::array::from_fn(|i| ac.code((i as u8) << 4));
        }
        self.eob_run += 1;
        let mut remaining = usize::from(count);
        let used = self.refinement_bits & 15;
        if used != 0 && remaining != 0 {
            let take = (16 - used).min(remaining);
            let chunk = ((bits >> (remaining - take)) & ((1u64 << take) - 1)) as u16;
            let last = self
                .refinement_words
                .last_mut()
                .expect("packed word exists");
            *last = ((u32::from(*last) << take) as u16) | chunk;
            remaining -= take;
        }
        while remaining >= 16 {
            remaining -= 16;
            self.refinement_words.push((bits >> remaining) as u16);
        }
        if remaining != 0 {
            self.refinement_words
                .push((bits & ((1u64 << remaining) - 1)) as u16);
        }
        self.refinement_bits = total;
        if self.eob_run == MAX_EOB_RUN {
            self.flush_inner(writer)?;
        }
        Ok(())
    }
}

fn validate_band(band: ScanBand) -> Result<(), CoefficientWriteError> {
    if band.start > band.end || band.end >= 64 || band.low_bit > 10 {
        return Err(CoefficientWriteError::Invalid(
            "spectral band or approximation level",
        ));
    }
    Ok(())
}

fn shifted_ac(coefficient: i16, low_bit: u8) -> i32 {
    let value = i32::from(coefficient);
    let magnitude = value.abs() >> low_bit;
    if value < 0 { -magnitude } else { magnitude }
}

fn validate_ac_band(
    coefficients: &[i16; BLOCK_SIZE],
    band: ScanBand,
) -> Result<usize, CoefficientWriteError> {
    #[cfg(feature = "decoder-profile")]
    let _profile = jxl_decoder::decoder_profile::sample_scope("sample.jpeg.validate_ac_band", 5);
    let mut trailing_zeros = 0;
    for &natural in &NATURAL_ORDER[usize::from(band.start.max(1))..=usize::from(band.end)] {
        let magnitude = i32::from(coefficients[natural]).abs();
        // For low_bit <= 10 the original category test is exactly this bound,
        // including values that disappear under the point transform.
        if magnitude >= 2048 {
            return Err(CoefficientWriteError::Invalid(
                "AC coefficient exceeds supported precision",
            ));
        }
        trailing_zeros = if magnitude >> band.low_bit == 0 {
            trailing_zeros + 1
        } else {
            0
        };
    }
    Ok(trailing_zeros)
}

/// Emit a sequential block, including explicitly preserved trailing ZRLs.
/// The predictor belongs to this component and stores its previous DC value.
/// Errors require discarding the output and predictor; see module contract.
pub fn encode_sequential_block(
    writer: &mut EntropyBitWriter,
    dc: &HuffmanTable,
    ac: &HuffmanTable,
    extra_zero_runs: u32,
    last_dc: &mut i16,
    coefficients: &[i16; BLOCK_SIZE],
) -> Result<(), CoefficientWriteError> {
    let (dc_count, dc_bits) = magnitude_bits(i32::from(coefficients[0]) - i32::from(*last_dc))?;
    if dc_count >= 12 {
        return Err(CoefficientWriteError::Invalid(
            "DC difference exceeds 11 bits",
        ));
    }
    // Validate before any emission or predictor update, retaining nonzero
    // positions so entropy emission need not scan zero coefficients again.
    let nonzero = sequential::nonzero_ac(coefficients).ok_or(CoefficientWriteError::Invalid(
        "AC coefficient exceeds supported precision",
    ))?;
    if extra_zero_runs != 0 {
        let trailing = nonzero.leading_zeros().min(63) as usize;
        // Preserve the frozen reader's single final ZRL overshoot.
        if u64::from(extra_zero_runs) > trailing.div_ceil(16) as u64 {
            return Err(CoefficientWriteError::Invalid(
                "extra zero runs exceed trailing spectral band",
            ));
        }
    }

    if writer
        .try_batch(|batch| {
            batch.write_symbol_bits(dc, dc_count, u32::from(dc_bits), dc_count)?;
            let mut remaining = nonzero;
            let mut next = 1;
            while remaining != 0 {
                let k = remaining.trailing_zeros() as usize;
                let mut zeros = k - next;
                while zeros > 15 {
                    batch.write_symbol(ac, 0xf0)?;
                    zeros -= 16;
                }
                let (count, bits) =
                    magnitude_bits(i32::from(coefficients[NATURAL_ORDER[k]])).ok()?;
                batch.write_symbol_bits(
                    ac,
                    ((zeros as u8) << 4) | count,
                    u32::from(bits),
                    count,
                )?;
                next = k + 1;
                remaining &= remaining - 1;
            }
            let mut zeros = (BLOCK_SIZE - next) as i32;
            for _ in 0..extra_zero_runs {
                batch.write_symbol(ac, 0xf0)?;
                zeros -= 16;
            }
            if zeros > 0 {
                batch.write_symbol(ac, 0)?;
            }
            Some(())
        })
        .is_some()
    {
        *last_dc = coefficients[0];
        return Ok(());
    }

    // Budget or table failures replay the original symbol-at-a-time path,
    // including its partially advanced writer and DC predictor on error.
    writer.write_symbol_bits(dc, dc_count, u32::from(dc_bits), dc_count)?;
    *last_dc = coefficients[0];
    let mut zeros = 0i32;
    for &natural in &NATURAL_ORDER[1..BLOCK_SIZE] {
        let value = i32::from(coefficients[natural]);
        if value == 0 {
            zeros += 1;
            continue;
        }
        while zeros > 15 {
            writer.write_symbol(ac, 0xf0)?;
            zeros -= 16;
        }
        let (count, bits) = magnitude_bits(value)?;
        writer.write_symbol_bits(ac, ((zeros as u8) << 4) | count, u32::from(bits), count)?;
        zeros = 0;
    }
    for _ in 0..extra_zero_runs {
        writer.write_symbol(ac, 0xf0)?;
        zeros -= 16;
    }
    if zeros > 0 {
        writer.write_symbol(ac, 0)?;
    }
    Ok(())
}

#[inline]
fn batch_initial_ac(
    batch: &mut EntropyBatch,
    ac: &HuffmanTable,
    order: &[usize],
    low_bit: u8,
    extra_zero_runs: u32,
    coefficients: &[i16; BLOCK_SIZE],
) -> Option<i32> {
    #[cfg(feature = "decoder-profile")]
    let _profile = jxl_decoder::decoder_profile::sample_scope("sample.jpeg.batch_initial_ac", 6);
    let mut zeros = 0i32;
    for &natural in order {
        let value = shifted_ac(coefficients[natural], low_bit);
        if value == 0 {
            zeros += 1;
            continue;
        }
        while zeros > 15 {
            batch.write_symbol(ac, 0xf0)?;
            zeros -= 16;
        }
        let (count, bits) = magnitude_bits(value).ok()?;
        batch.write_symbol_bits(ac, ((zeros as u8) << 4) | count, u32::from(bits), count)?;
        zeros = 0;
    }
    for _ in 0..extra_zero_runs {
        batch.write_symbol(ac, 0xf0)?;
        zeros -= 16;
    }
    Some(zeros)
}

/// Emit a first progressive pass, or a sequential band. Negative DC uses an
/// arithmetic shift; negative AC shifts its magnitude, truncating toward zero.
/// `last_dc` stores the unscaled predictor, as in the decoder and frozen writer.
/// Reset-point hints are applied by calling state.flush BEFORE this block.
#[allow(clippy::too_many_arguments)]
pub fn encode_initial_block(
    writer: &mut EntropyBitWriter,
    dc: Option<&HuffmanTable>,
    ac: Option<&HuffmanTable>,
    band: ScanBand,
    extra_zero_runs: u32,
    state: &mut DctCodingState,
    last_dc: &mut i16,
    coefficients: &[i16; BLOCK_SIZE],
) -> Result<(), CoefficientWriteError> {
    state.healthy()?;
    let result = (|| {
        validate_band(band)?;
        if band.start == 0 && state.eob_run != 0 {
            return Err(CoefficientWriteError::Invalid(
                "pending AC state before DC block",
            ));
        }
        let dc_value = i32::from(coefficients[0]) >> band.low_bit;
        let dc_bits = if band.start == 0 {
            if dc.is_none() {
                return Err(CoefficientWriteError::Invalid("missing DC table"));
            }
            let magnitude = magnitude_bits(dc_value - i32::from(*last_dc))?;
            if magnitude.0 >= 12 {
                return Err(CoefficientWriteError::Invalid(
                    "DC difference exceeds 11 bits",
                ));
            }
            Some(magnitude)
        } else {
            None
        };
        let trailing = if band.end != 0 {
            if ac.is_none() {
                return Err(CoefficientWriteError::Invalid("missing AC table"));
            }
            validate_ac_band(coefficients, band)?
        } else {
            0
        };
        // The frozen reader permits the last ZRL to pass the band end, but a
        // further ZRL could never have been consumed. Preserve that one overshoot.
        if u64::from(extra_zero_runs) > trailing.div_ceil(16) as u64 {
            return Err(CoefficientWriteError::Invalid(
                "extra zero runs exceed trailing spectral band",
            ));
        }
        if let Some((count, bits)) = dc_bits {
            writer.write_symbol_bits(
                dc.expect("validated DC table"),
                count,
                u32::from(bits),
                count,
            )?;
            *last_dc = dc_value as i16;
        }
        if band.end == 0 {
            return Ok(());
        }
        let ac = ac.expect("validated AC table");
        let order = &NATURAL_ORDER[usize::from(band.start.max(1))..=usize::from(band.end)];
        let has_symbols = trailing != order.len() || extra_zero_runs != 0;
        if has_symbols {
            state.flush_inner(writer)?;
        }
        let batched = !has_symbols
            || writer
                .try_batch(|batch| {
                    batch_initial_ac(
                        batch,
                        ac,
                        order,
                        band.low_bit,
                        extra_zero_runs,
                        coefficients,
                    )?;
                    Some(())
                })
                .is_some();
        let mut zeros = trailing as i32 - 16 * extra_zero_runs as i32;
        if !batched {
            zeros = 0;
            for &natural in order {
                let value = shifted_ac(coefficients[natural], band.low_bit);
                if value == 0 {
                    zeros += 1;
                    continue;
                }
                while zeros > 15 {
                    writer.write_symbol(ac, 0xf0)?;
                    zeros -= 16;
                }
                let (count, bits) = magnitude_bits(value)?;
                writer.write_symbol_bits(
                    ac,
                    ((zeros as u8) << 4) | count,
                    u32::from(bits),
                    count,
                )?;
                zeros = 0;
            }
            for _ in 0..extra_zero_runs {
                writer.write_symbol(ac, 0xf0)?;
                zeros -= 16;
            }
        }
        if zeros > 0 {
            state.buffer_end_of_band(writer, ac, 0, 0)?;
            if band.start == 0 {
                state.flush_inner(writer)?;
            }
        }
        Ok(())
    })();
    state.finish_operation(result)
}

/// Emit one refinement pass from the final coefficients. Existing nonzeros
/// produce magnitude correction bits; newly nonzero AC values produce a sign
/// bit before those corrections. DC refinement instead emits a two's-complement
/// bit. Caller validates Ah/progression and applies reset-point hints first.
pub fn encode_refinement_block(
    writer: &mut EntropyBitWriter,
    ac: Option<&HuffmanTable>,
    band: ScanBand,
    state: &mut DctCodingState,
    coefficients: &[i16; BLOCK_SIZE],
) -> Result<(), CoefficientWriteError> {
    state.healthy()?;
    let result = (|| {
        validate_band(band)?;
        if band.start == 0 && state.eob_run != 0 {
            return Err(CoefficientWriteError::Invalid(
                "pending AC state before DC refinement",
            ));
        }
        let (masks, last_new, tail) = if band.end != 0 {
            if ac.is_none() {
                return Err(CoefficientWriteError::Invalid(
                    "missing refinement AC table",
                ));
            }
            prepare_refinement_band(coefficients, band)?
        } else {
            (Masks::default(), 0, RefinementTail::default())
        };
        if band.start == 0 {
            writer.write_bits(((i32::from(coefficients[0]) >> band.low_bit) & 1) as u32, 1)?;
        }
        if band.end == 0 {
            return Ok(());
        }
        let ac = ac.expect("validated refinement AC table");
        let start = usize::from(band.start.max(1));
        let end = usize::from(band.end);
        let batched = if last_new == 0 {
            // Validation already collected the whole band's corrections. No
            // symbols, scratch batch or second coefficient walk are needed.
            true
        } else {
            state.flush_inner(writer)?;
            writer
                .try_batch(|batch| batch_refinement_ac(batch, ac, start, last_new, masks))
                .is_some()
        };
        if batched {
            if tail.zeros != 0 || tail.count != 0 {
                state.buffer_end_of_band(writer, ac, tail.bits, tail.count)?;
                if band.start == 0 {
                    state.flush_inner(writer)?;
                }
            }
            return Ok(());
        }
        let mut zeros = 0u8;
        let mut correction_bits = 0u64;
        let mut correction_count = 0u8;
        for k in start..=end {
            let magnitude = i32::from(coefficients[NATURAL_ORDER[k]]).abs() >> band.low_bit;
            if magnitude == 0 {
                zeros += 1;
                continue;
            }
            while zeros > 15 && k <= last_new {
                state.flush_inner(writer)?;
                writer.write_symbol(ac, 0xf0)?;
                zeros -= 16;
                write_corrections(writer, correction_bits, correction_count)?;
                correction_bits = 0;
                correction_count = 0;
            }
            if magnitude > 1 {
                correction_bits = (correction_bits << 1) | (magnitude as u64 & 1);
                correction_count += 1;
                continue;
            }
            state.flush_inner(writer)?;
            let sign = u32::from(coefficients[NATURAL_ORDER[k]] > 0);
            writer.write_symbol_bits(ac, (zeros << 4) | 1, sign, 1)?;
            write_corrections(writer, correction_bits, correction_count)?;
            correction_bits = 0;
            correction_count = 0;
            zeros = 0;
        }
        if zeros != 0 || correction_count != 0 {
            state.buffer_end_of_band(writer, ac, correction_bits, correction_count)?;
            if band.start == 0 {
                state.flush_inner(writer)?;
            }
        }
        Ok(())
    })();
    state.finish_operation(result)
}

fn batch_refinement_ac(
    batch: &mut EntropyBatch,
    ac: &HuffmanTable,
    start: usize,
    last_new: usize,
    masks: Masks,
) -> Option<()> {
    #[cfg(feature = "decoder-profile")]
    let _profile = jxl_decoder::decoder_profile::sample_scope("sample.jpeg.batch_refinement_ac", 7);
    let mut zeros = 0u8;
    let mut bits = 0u64;
    let mut count = 0u8;
    let mut remaining = masks.nonzero & (u64::MAX >> (63 - last_new));
    let mut cursor = start;
    while remaining != 0 {
        let k = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        zeros += (k - cursor) as u8;
        cursor = k + 1;
        while zeros > 15 {
            batch.write_symbol(ac, 0xf0)?;
            zeros -= 16;
            batch_corrections(batch, bits, count)?;
            bits = 0;
            count = 0;
        }
        if masks.newly_nonzero & (1 << k) == 0 {
            bits = (bits << 1) | ((masks.correction >> k) & 1);
            count += 1;
            continue;
        }
        let sign = ((masks.positive >> k) & 1) as u32;
        batch.write_symbol_bits(ac, (zeros << 4) | 1, sign, 1)?;
        batch_corrections(batch, bits, count)?;
        bits = 0;
        count = 0;
        zeros = 0;
    }
    Some(())
}

#[inline]
fn batch_corrections(batch: &mut EntropyBatch, bits: u64, count: u8) -> Option<()> {
    batch.write_bits64(bits, count)
}

#[derive(Default)]
struct RefinementTail {
    zeros: u8,
    bits: u64,
    count: u8,
}

fn prepare_refinement_band(
    coefficients: &[i16; BLOCK_SIZE],
    band: ScanBand,
) -> Result<(Masks, usize, RefinementTail), CoefficientWriteError> {
    #[cfg(feature = "decoder-profile")]
    let _profile = jxl_decoder::decoder_profile::sample_scope("sample.jpeg.prepare_refinement", 4);
    let masks = refinement::classify(coefficients, band).ok_or(CoefficientWriteError::Invalid(
        "AC coefficient exceeds supported precision",
    ))?;
    let last_new = 63usize.saturating_sub(masks.newly_nonzero.leading_zeros() as usize);
    // Shift by last_new, then by one, so a new coefficient at index 63
    // produces an empty suffix without a shift-by-64.
    let suffix = (u64::MAX << last_new) << 1;
    let selected = suffix & (u64::MAX << band.start.max(1)) & (u64::MAX >> (63 - band.end));
    let existing = selected & masks.nonzero;
    let (bits, count) = refinement::correction_bits(masks.correction, existing);
    let zeros = (selected & !masks.nonzero).count_ones() as u8;
    Ok((masks, last_new, RefinementTail { zeros, bits, count }))
}

fn write_corrections(
    writer: &mut EntropyBitWriter,
    bits: u64,
    count: u8,
) -> Result<(), WriteError> {
    if count > 32 {
        writer.write_bits((bits >> 32) as u32, count - 32)?;
        writer.write_bits(bits as u32, 32)
    } else {
        writer.write_bits(bits as u32, count)
    }
}

#[cfg(test)]
#[path = "coefficient_writer_tests.rs"]
mod tests;
