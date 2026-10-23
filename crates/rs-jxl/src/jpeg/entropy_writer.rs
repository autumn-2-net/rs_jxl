// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! MSB-first JPEG entropy emission, from dec_jpeg_data_writer.cc at b866b3d.
//!
//! This is a bounded in-memory counterpart of JpegBitWriter, not the LSB-first
//! JXL writer. Stuffing applies to entropy bytes only, never restart markers.
//! Each operation validates and reserves before committing output or bit state.
//! Scan traversal, EOB/refinement buffering and restart cadence belong to callers.

use std::fmt;

use super::huffman::HuffmanTable;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteError {
    InvalidBitCount(u8),
    BitsOutOfRange { value: u32, count: u8 },
    OutputLimit { limit: usize },
    AllocationFailed,
    Unaligned { pending_bits: u8 },
    WrongPaddingLength { expected: u8, actual: u8 },
    InvalidRestartIndex(u8),
    MissingHuffmanSymbol(u8),
    InvalidMagnitude(i32),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidBitCount(count) => write!(f, "cannot write {count} bits (maximum 32)"),
            Self::BitsOutOfRange { value, count } => {
                write!(f, "JPEG bits {value} do not fit {count} bits")
            }
            Self::OutputLimit { limit } => write!(f, "JPEG output exceeds {limit} bytes"),
            Self::AllocationFailed => f.write_str("cannot allocate JPEG output"),
            Self::Unaligned { pending_bits } => {
                write!(f, "{pending_bits} entropy bits await byte alignment")
            }
            Self::WrongPaddingLength { expected, actual } => {
                write!(f, "JPEG alignment needs {expected} bits, received {actual}")
            }
            Self::InvalidRestartIndex(index) => write!(f, "invalid JPEG restart index {index}"),
            Self::MissingHuffmanSymbol(symbol) => {
                write!(f, "JPEG Huffman table has no symbol {symbol}")
            }
            Self::InvalidMagnitude(value) => {
                write!(f, "JPEG magnitude {value} exceeds 16 bits")
            }
        }
    }
}

impl std::error::Error for WriteError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntropyBitWriter {
    bytes: Vec<u8>,
    pending: u8,
    used: u8,
    max_bytes: usize,
}

impl EntropyBitWriter {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            pending: 0,
            used: 0,
            max_bytes,
        }
    }

    /// Complete output bytes, including stuffing; pending bits are not included.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn pending_bits(&self) -> u8 {
        self.used
    }

    pub fn padding_bits_needed(&self) -> u8 {
        (8 - self.used) & 7
    }

    fn append(&mut self, bytes: &[u8]) -> Result<(), WriteError> {
        self.reserve_output(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn reserve_output(&mut self, additional: usize) -> Result<(), WriteError> {
        if self
            .bytes
            .len()
            .checked_add(additional)
            .is_none_or(|len| len > self.max_bytes)
        {
            return Err(WriteError::OutputLimit {
                limit: self.max_bytes,
            });
        }
        self.bytes
            .try_reserve(additional)
            .map_err(|_| WriteError::AllocationFailed)?;
        Ok(())
    }

    /// Append right-aligned bits, most significant first. Zero bits require zero.
    pub fn write_bits(&mut self, value: u32, count: u8) -> Result<(), WriteError> {
        validate_bits(value, count)?;
        let buffer = (u64::from(self.pending) << count) | u64::from(value);
        let mut remaining = self.used + count;
        // At most 39 buffered bits produce four complete, possibly stuffed bytes.
        let mut output = [0u8; 8];
        let mut len = 0;
        while remaining >= 8 {
            remaining -= 8;
            let byte = (buffer >> remaining) as u8;
            output[len] = byte;
            len += 1;
            if byte == 0xff {
                output[len] = 0;
                len += 1;
            }
        }
        self.append(&output[..len])?;
        self.pending = (buffer & ((1u64 << remaining) - 1)) as u8;
        self.used = remaining;
        Ok(())
    }

    /// Preserve the original zero/mixed/all-one alignment bits exactly.
    pub fn align_with(&mut self, value: u8, count: u8) -> Result<(), WriteError> {
        let expected = self.padding_bits_needed();
        if count != expected {
            return Err(WriteError::WrongPaddingLength {
                expected,
                actual: count,
            });
        }
        self.write_bits(u32::from(value), count)
    }

    pub fn align_ones(&mut self) -> Result<(), WriteError> {
        let count = self.padding_bits_needed();
        self.align_with(((1u16 << count) - 1) as u8, count)
    }

    /// Emit an unstuffed RST marker. Caller must first flush EOB and align bits.
    pub fn write_restart(&mut self, index: u8) -> Result<(), WriteError> {
        self.write_restart_with_fill(index, 0)
    }

    /// Preserve additional 0xff marker fill without consulting original entropy.
    pub fn write_restart_with_fill(&mut self, index: u8, fill: usize) -> Result<(), WriteError> {
        if index > 7 {
            return Err(WriteError::InvalidRestartIndex(index));
        }
        self.require_aligned()?;
        let additional = fill.checked_add(2).ok_or(WriteError::OutputLimit {
            limit: self.max_bytes,
        })?;
        self.reserve_output(additional)?;
        self.bytes.resize(self.bytes.len() + fill, 0xff);
        self.bytes.extend([0xff, 0xd0 + index]);
        Ok(())
    }

    pub fn write_symbol(&mut self, table: &HuffmanTable, symbol: u8) -> Result<(), WriteError> {
        self.write_symbol_bits(table, symbol, 0, 0)
    }

    /// Combine a canonical code and its magnitude bits in one atomic emission.
    pub fn write_symbol_bits(
        &mut self,
        table: &HuffmanTable,
        symbol: u8,
        value: u32,
        count: u8,
    ) -> Result<(), WriteError> {
        validate_bits(value, count)?;
        let code = table
            .code(symbol)
            .ok_or(WriteError::MissingHuffmanSymbol(symbol))?;
        let total = count + code.bit_length;
        if total > 32 {
            return Err(WriteError::InvalidBitCount(total));
        }
        self.write_bits((u32::from(code.bits) << count) | value, total)
    }

    fn require_aligned(&self) -> Result<(), WriteError> {
        if self.used != 0 {
            Err(WriteError::Unaligned {
                pending_bits: self.used,
            })
        } else {
            Ok(())
        }
    }

    /// No implicit padding or marker insertion; incomplete entropy is an error.
    pub fn into_bytes(self) -> Result<Vec<u8>, WriteError> {
        self.require_aligned()?;
        Ok(self.bytes)
    }
}

fn validate_bits(value: u32, count: u8) -> Result<(), WriteError> {
    if count > 32 {
        return Err(WriteError::InvalidBitCount(count));
    }
    if count < 32 && value >= 1u32 << count {
        return Err(WriteError::BitsOutOfRange { value, count });
    }
    Ok(())
}

/// JPEG sign-magnitude complement coding, inverse of HuffExtend, not two's complement.
/// Process-specific DC/AC category restrictions must still be checked by callers.
pub fn magnitude_bits(value: i32) -> Result<(u8, u16), WriteError> {
    let category = (32 - value.unsigned_abs().leading_zeros()) as u8;
    if category > 16 {
        return Err(WriteError::InvalidMagnitude(value));
    }
    let mask = ((1u32 << category) - 1) as u16;
    let bits = if value < 0 { value - 1 } else { value } as u16 & mask;
    Ok((category, bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::bitstream::{EntropyBitReader, PaddingPolicy, extend};

    #[test]
    fn bit_packing_matches_independent_bit_vector_and_reader() {
        for initial in 0..8 {
            let mut writer = EntropyBitWriter::new(8192);
            let mut expected = Vec::new();
            let mut fields = Vec::new();
            let mut rng = 0x42ec_9765u32;
            for count in std::iter::once(initial).chain((0..=32).cycle().take(300)) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let mask = if count == 32 {
                    u32::MAX
                } else {
                    (1 << count) - 1
                };
                let value = rng & mask;
                writer.write_bits(value, count).unwrap();
                fields.push((value, count));
                for bit in (0..count).rev() {
                    expected.push((value >> bit) as u8 & 1);
                }
            }
            while !expected.len().is_multiple_of(8) {
                expected.push(1);
            }
            let mut packed = Vec::new();
            for chunk in expected.chunks_exact(8) {
                let byte = chunk.iter().fold(0, |acc, bit| (acc << 1) | bit);
                packed.push(byte);
                if byte == 255 {
                    packed.push(0);
                }
            }
            writer.align_ones().unwrap();
            assert_eq!(writer.bytes(), packed);
            writer.write_restart(3).unwrap();
            let bytes = writer.into_bytes().unwrap();
            let mut reader = EntropyBitReader::new(&bytes);
            for (value, count) in fields {
                assert_eq!(reader.read_bits(count).unwrap(), value);
            }
            reader
                .consume_restart(3, PaddingPolicy::RequireOnes)
                .unwrap();
            assert_eq!(reader.position(), bytes.len());
        }
    }

    #[test]
    fn stuffing_padding_and_unstuffed_markers_are_exact() {
        for count in 0..8 {
            for padding in 0..1u8 << count {
                let mut writer = EntropyBitWriter::new(16);
                writer.write_bits(0xff, 8).unwrap();
                if count != 0 {
                    writer
                        .write_bits((1 << (8 - count)) - 1, 8 - count)
                        .unwrap();
                }
                writer.align_with(padding, count).unwrap();
                writer.write_restart(7).unwrap();
                let mut expected = vec![0xff, 0];
                if count != 0 {
                    let byte = (0xff << count) | padding;
                    expected.push(byte);
                    if byte == 255 {
                        expected.push(0);
                    }
                }
                expected.extend([0xff, 0xd7]);
                assert_eq!(writer.into_bytes().unwrap(), expected);
            }
        }
    }

    #[test]
    fn limit_and_argument_errors_do_not_mutate_writer() {
        let mut writer = EntropyBitWriter::new(7);
        let before = writer.clone();
        assert!(matches!(
            writer.write_bits(u32::MAX, 32),
            Err(WriteError::OutputLimit { .. })
        ));
        assert_eq!(writer, before);
        writer.write_bits(1, 1).unwrap();
        let before = writer.clone();
        assert!(writer.write_bits(1, 0).is_err());
        assert!(writer.write_bits(0, 33).is_err());
        assert!(writer.write_bits(2, 1).is_err());
        assert!(writer.align_with(0, 6).is_err());
        assert!(writer.align_with(128, 7).is_err());
        assert!(writer.write_restart(0).is_err());
        assert!(writer.write_restart(8).is_err());
        assert!(writer.clone().into_bytes().is_err());
        assert_eq!(writer, before);
        let mut writer = EntropyBitWriter::new(8);
        writer.write_bits(u32::MAX, 32).unwrap();
        assert_eq!(writer.bytes(), [255, 0, 255, 0, 255, 0, 255, 0]);
        let before = writer.clone();
        assert!(writer.write_restart(0).is_err());
        assert_eq!(writer, before);
        let mut empty = EntropyBitWriter::new(0);
        empty.write_bits(0, 0).unwrap();
        empty.align_ones().unwrap();
        assert_eq!(empty.into_bytes().unwrap(), []);
    }

    #[test]
    fn huffman_symbols_and_payloads_are_atomic() {
        let mut counts = [0; 16];
        counts[1] = 3;
        let table = HuffmanTable::new(&counts, &[0, 1, 2]).unwrap();
        let mut writer = EntropyBitWriter::new(16);
        writer.write_symbol_bits(&table, 2, 5, 3).unwrap();
        writer.write_symbol(&table, 1).unwrap();
        writer.align_ones().unwrap();
        assert_eq!(writer.bytes(), [0b1010_1011]);
        let before = writer.clone();
        assert!(matches!(
            writer.write_symbol(&table, 3),
            Err(WriteError::MissingHuffmanSymbol(3))
        ));
        assert!(writer.write_symbol_bits(&table, 2, 0, 32).is_err());
        assert_eq!(writer, before);
        let mut limited = EntropyBitWriter::new(0);
        let before = limited.clone();
        assert!(limited.write_symbol_bits(&table, 2, 0, 16).is_err());
        assert_eq!(limited, before);
    }

    #[test]
    fn signed_magnitudes_invert_extension_including_dc_differences() {
        for value in -65535..=65535 {
            let (category, bits) = magnitude_bits(value).unwrap();
            assert_eq!(extend(u32::from(bits), category).unwrap(), value);
        }
        for value in [i32::MIN, -65536, 65536, i32::MAX] {
            assert!(magnitude_bits(value).is_err());
        }
    }

    #[test]
    fn restart_fill_is_separate_from_entropy_and_budgeted_before_allocation() {
        let mut writer = EntropyBitWriter::new(6);
        writer.write_bits(255, 8).unwrap();
        writer.write_restart_with_fill(4, 2).unwrap();
        assert_eq!(writer.bytes(), [255, 0, 255, 255, 255, 0xd4]);
        let before = writer.clone();
        assert!(writer.write_restart_with_fill(0, usize::MAX).is_err());
        assert!(writer.write_restart_with_fill(0, 0).is_err());
        assert_eq!(writer, before);
    }
}
