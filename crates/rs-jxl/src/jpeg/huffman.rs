// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG Huffman lookup ported from lib/jxl/jpeg/enc_jpeg_huffman_decode.{h,cc}
//! and ProcessDHT/ReadSymbol in enc_jpeg_data_reader.cc, frozen at b866b3d.
//!
//! Uses JPEG's MSB-first canonical ordering, an eight-bit root and variable-width
//! second-level tables. Incomplete trees are legal; unused entries stay invalid.
//! Unlike the frozen internal builder, this API accepts only real DHT symbols,
//! not its synthetic all-ones sentinel. A single symbol consumes its actual code
//! length, never zero bits. All-one codes are forbidden, including complete trees.
//!
//! DC/AC symbol semantics and process-specific category limits belong to the
//! coefficient parser. This module does not parse DHT marker segments.

use std::fmt;

use super::bitstream::{EntropyBitReader, EntropyError};

pub const ROOT_BITS: u8 = 8;
pub const MAX_CODE_LENGTH: u8 = 16;
const ROOT_SIZE: usize = 1 << ROOT_BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HuffmanError {
    EmptyTable,
    TooManySymbols(usize),
    SymbolCountMismatch { expected: usize, actual: usize },
    DuplicateSymbol(u8),
    Oversubscribed { bit_length: u8 },
    ForbiddenAllOnes,
    InvalidCode,
    Entropy(EntropyError),
}

impl fmt::Display for HuffmanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTable => f.write_str("empty JPEG Huffman table cannot decode symbols"),
            Self::TooManySymbols(count) => {
                write!(f, "JPEG Huffman table has {count} symbols (maximum 256)")
            }
            Self::SymbolCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Huffman counts require {expected} symbols, received {actual}"
                )
            }
            Self::DuplicateSymbol(symbol) => write!(f, "duplicate JPEG Huffman symbol {symbol}"),
            Self::Oversubscribed { bit_length } => {
                write!(f, "oversubscribed Huffman tree at length {bit_length}")
            }
            Self::ForbiddenAllOnes => f.write_str("JPEG forbids an all-ones Huffman code"),
            Self::InvalidCode => f.write_str("unused code in incomplete JPEG Huffman tree"),
            Self::Entropy(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for HuffmanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Entropy(error) => Some(error),
            _ => None,
        }
    }
}

impl From<EntropyError> for HuffmanError {
    fn from(error: EntropyError) -> Self {
        Self::Entropy(error)
    }
}

/// Right-aligned canonical bits. Emit `bit_length` bits, most significant first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HuffmanCode {
    pub bits: u16,
    pub bit_length: u8,
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    Invalid,
    Symbol { symbol: u8, bit_length: u8 },
    Subtable { offset: usize, bits: u8 },
}

#[derive(Clone, Debug)]
pub struct HuffmanTable {
    entries: Vec<Entry>,
    counts: [u8; 16],
    first_code: [u32; 17],
    first_symbol: [usize; 17],
    symbols: Vec<u8>,
    codes: [Option<HuffmanCode>; 256],
    // Canonical codes occupy one contiguous interval of the 16-bit code space.
    code_space_end: u32,
}

impl HuffmanTable {
    /// Build from the 16 DHT count bytes and exactly their sum of symbol bytes.
    /// `counts[0]` describes length one. Do not append libjxl's fake sentinel.
    /// Symbols must be distinct. Sparse/incomplete trees are accepted, but empty,
    /// oversubscribed, and complete (all-ones-code) trees are rejected.
    pub fn new(counts: &[u8; 16], symbols: &[u8]) -> Result<Self, HuffmanError> {
        let total: usize = counts.iter().map(|&count| usize::from(count)).sum();
        if total == 0 {
            return Err(HuffmanError::EmptyTable);
        }
        if total > 256 {
            return Err(HuffmanError::TooManySymbols(total));
        }
        if symbols.len() != total {
            return Err(HuffmanError::SymbolCountMismatch {
                expected: total,
                actual: symbols.len(),
            });
        }
        let mut left = 1u32;
        for (index, &count) in counts.iter().enumerate() {
            left <<= 1;
            if u32::from(count) > left {
                return Err(HuffmanError::Oversubscribed {
                    bit_length: index as u8 + 1,
                });
            }
            left -= u32::from(count);
        }
        if left == 0 {
            return Err(HuffmanError::ForbiddenAllOnes);
        }
        let mut seen = [false; 256];
        for &symbol in symbols {
            if seen[usize::from(symbol)] {
                return Err(HuffmanError::DuplicateSymbol(symbol));
            }
            seen[usize::from(symbol)] = true;
        }

        let mut table = Self {
            entries: vec![Entry::Invalid; ROOT_SIZE],
            counts: *counts,
            first_code: [0; 17],
            first_symbol: [0; 17],
            symbols: symbols.to_vec(),
            codes: [None; 256],
            code_space_end: (1 << MAX_CODE_LENGTH) - left,
        };
        let mut code = 0u32;
        let mut symbol_index = 0;
        let mut widths = [0u8; ROOT_SIZE];
        for bit_length in 1..=MAX_CODE_LENGTH {
            table.first_code[usize::from(bit_length)] = code;
            table.first_symbol[usize::from(bit_length)] = symbol_index;
            for _ in 0..counts[usize::from(bit_length - 1)] {
                let symbol = symbols[symbol_index];
                table.codes[usize::from(symbol)] = Some(HuffmanCode {
                    bits: code as u16,
                    bit_length,
                });
                if bit_length > ROOT_BITS {
                    let width = bit_length - ROOT_BITS;
                    let prefix = (code >> width) as usize;
                    widths[prefix] = widths[prefix].max(width);
                }
                code += 1;
                symbol_index += 1;
            }
            code <<= 1;
        }

        // Size each subtree for its deepest real code. Dynamic offsets avoid
        // relying on the frozen builder's fixed scratch-size/sentinel contract.
        for (prefix, &bits) in widths.iter().enumerate() {
            if bits != 0 {
                let offset = table.entries.len();
                table
                    .entries
                    .resize(offset + (1usize << bits), Entry::Invalid);
                table.entries[prefix] = Entry::Subtable { offset, bits };
            }
        }
        for &symbol in symbols {
            let code = table.codes[usize::from(symbol)].expect("validated symbol has a code");
            let entry = Entry::Symbol {
                symbol,
                bit_length: code.bit_length,
            };
            if code.bit_length <= ROOT_BITS {
                let repeat_bits = ROOT_BITS - code.bit_length;
                let start = usize::from(code.bits) << repeat_bits;
                table.entries[start..start + (1 << repeat_bits)].fill(entry);
            } else {
                let tail_bits = code.bit_length - ROOT_BITS;
                let prefix = usize::from(code.bits >> tail_bits);
                let Entry::Subtable { offset, bits } = table.entries[prefix] else {
                    unreachable!("validated long code has a subtree");
                };
                let suffix = usize::from(code.bits) & ((1 << tail_bits) - 1);
                let repeat_bits = bits - tail_bits;
                let start = offset + (suffix << repeat_bits);
                table.entries[start..start + (1 << repeat_bits)].fill(entry);
            }
        }
        Ok(table)
    }

    /// Encoding-side metadata for entropy reconstruction, not a JPEG writer.
    pub fn code(&self, symbol: u8) -> Option<HuffmanCode> {
        self.codes[usize::from(symbol)]
    }

    /// Decode one symbol; errors never advance the reader.
    pub fn decode(&self, reader: &mut EntropyBitReader<'_>) -> Result<u8, HuffmanError> {
        let mut trial = *reader;
        let symbol = self.decode_inner(&mut trial)?;
        *reader = trial;
        Ok(symbol)
    }

    fn decode_inner(&self, reader: &mut EntropyBitReader<'_>) -> Result<u8, HuffmanError> {
        let Ok(prefix) = reader.peek_bits(ROOT_BITS) else {
            return self.decode_canonical(reader);
        };
        let entry = match self.entries[prefix as usize] {
            Entry::Subtable { offset, bits } => {
                let Ok(window) = reader.peek_bits(ROOT_BITS + bits) else {
                    return self.decode_canonical(reader);
                };
                self.entries[offset + (window as usize & ((1 << bits) - 1))]
            }
            entry => entry,
        };
        match entry {
            Entry::Symbol { symbol, bit_length } => {
                reader.read_bits(bit_length)?;
                Ok(symbol)
            }
            _ => Err(HuffmanError::InvalidCode),
        }
    }

    // A short symbol may still fit when the full lookup window hits a marker,
    // EOF, or malformed following escape. Consume only the actual code bits.
    fn decode_canonical(&self, reader: &mut EntropyBitReader<'_>) -> Result<u8, HuffmanError> {
        let mut code = 0u32;
        for bit_length in 1..=MAX_CODE_LENGTH {
            code = (code << 1) | reader.read_bits(1)?;
            let index = usize::from(bit_length);
            let first = self.first_code[index];
            let count = u32::from(self.counts[index - 1]);
            if code >= first && code - first < count {
                return Ok(self.symbols[self.first_symbol[index] + (code - first) as usize]);
            }
            if code << (MAX_CODE_LENGTH - bit_length) >= self.code_space_end {
                return Err(HuffmanError::InvalidCode);
            }
        }
        Err(HuffmanError::InvalidCode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::bitstream::PaddingPolicy;

    // Only an entropy fixture builder: MSB-first bits, one-padding and stuffing.
    fn pack(words: &[(u32, u8)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut byte = 0u8;
        let mut used = 0;
        for &(word, length) in words {
            for bit in (0..length).rev() {
                byte = (byte << 1) | ((word >> bit) & 1) as u8;
                used += 1;
                if used == 8 {
                    bytes.push(byte);
                    if byte == 0xff {
                        bytes.push(0);
                    }
                    byte = 0;
                    used = 0;
                }
            }
        }
        if used != 0 {
            byte = (byte << (8 - used)) | ((1u16 << (8 - used)) - 1) as u8;
            bytes.push(byte);
            if byte == 0xff {
                bytes.push(0);
            }
        }
        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }

    fn packed_symbols(table: &HuffmanTable, symbols: &[u8]) -> Vec<u8> {
        let words: Vec<_> = symbols
            .iter()
            .map(|&symbol| {
                let code = table.code(symbol).unwrap();
                (u32::from(code.bits), code.bit_length)
            })
            .collect();
        pack(&words)
    }

    #[test]
    fn canonical_order_is_msb_first_and_keeps_symbol_order() {
        let mut counts = [0; 16];
        counts[1..5].copy_from_slice(&[2, 1, 1, 1]);
        let symbols = [50, 10, 90, 20, 80];
        let table = HuffmanTable::new(&counts, &symbols).unwrap();
        for (symbol, bits, bit_length) in [
            (50, 0b00, 2),
            (10, 0b01, 2),
            (90, 0b100, 3),
            (20, 0b1010, 4),
            (80, 0b10110, 5),
        ] {
            assert_eq!(table.code(symbol), Some(HuffmanCode { bits, bit_length }));
        }
        assert_eq!(table.code(0), None);
        let data = packed_symbols(&table, &symbols);
        let mut reader = EntropyBitReader::new(&data);
        for symbol in symbols {
            assert_eq!(table.decode(&mut reader), Ok(symbol));
        }
        assert_eq!(
            reader
                .finish_scan(PaddingPolicy::RequireOnes)
                .unwrap()
                .marker
                .code,
            0xd9
        );
    }

    #[test]
    fn all_lengths_through_sixteen_decode_with_stuffed_bytes() {
        let symbols: Vec<u8> = (0..16).collect();
        let table = HuffmanTable::new(&[1; 16], &symbols).unwrap();
        for length in 1..=16 {
            assert_eq!(
                table.code(length - 1),
                Some(HuffmanCode {
                    bits: ((1u32 << length) - 2) as u16,
                    bit_length: length
                })
            );
        }
        assert!(table.entries.len() > ROOT_SIZE);
        let sequence = [15, 0, 8, 1, 14, 7, 9, 15, 2, 10, 11, 3, 12, 13, 4, 5, 6];
        let data = packed_symbols(&table, &sequence);
        assert!(data.windows(2).any(|pair| pair == [0xff, 0]));
        let mut reader = EntropyBitReader::new(&data);
        for symbol in sequence {
            assert_eq!(table.decode(&mut reader), Ok(symbol));
        }
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }

    #[test]
    fn multiple_second_level_tables_and_repeated_short_leaves() {
        let mut counts = [0; 16];
        counts[7] = 252;
        counts[8] = 3;
        counts[15] = 1;
        let symbols: Vec<u8> = (0..=255).collect();
        let table = HuffmanTable::new(&counts, &symbols).unwrap();
        assert!(matches!(
            table.entries[252],
            Entry::Subtable { bits: 1, .. }
        ));
        assert!(matches!(
            table.entries[253],
            Entry::Subtable { bits: 8, .. }
        ));
        let data = packed_symbols(&table, &symbols);
        let mut reader = EntropyBitReader::new(&data);
        for symbol in symbols {
            assert_eq!(table.decode(&mut reader), Ok(symbol));
        }
        reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
    }

    #[test]
    fn single_symbol_always_consumes_its_declared_length() {
        for bit_length in 1..=16 {
            let mut counts = [0; 16];
            counts[usize::from(bit_length - 1)] = 1;
            let table = HuffmanTable::new(&counts, &[42]).unwrap();
            assert_eq!(
                table.code(42),
                Some(HuffmanCode {
                    bits: 0,
                    bit_length
                })
            );
            let data = packed_symbols(&table, &[42, 42, 42]);
            let mut reader = EntropyBitReader::new(&data);
            for consumed in 1..=3 {
                assert_eq!(table.decode(&mut reader), Ok(42));
                let bits = usize::from(bit_length) * consumed;
                assert_eq!(reader.position(), bits.div_ceil(8));
                assert_eq!(usize::from(reader.bits_left()), (8 - bits % 8) % 8);
            }
            reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
            let mut invalid = EntropyBitReader::new(&[0xff, 0x00]);
            assert_eq!(table.decode(&mut invalid), Err(HuffmanError::InvalidCode));
            assert_eq!(invalid.position(), 0);
        }
    }

    #[test]
    fn sparse_trees_reject_unused_root_and_subtable_entries() {
        let mut counts = [0; 16];
        counts[1] = 1;
        counts[15] = 1;
        let table = HuffmanTable::new(&counts, &[3, 4]).unwrap();
        assert_eq!(
            table.code(4),
            Some(HuffmanCode {
                bits: 0x4000,
                bit_length: 16
            })
        );
        let data = packed_symbols(&table, &[4, 3]);
        let mut reader = EntropyBitReader::new(&data);
        assert_eq!(table.decode(&mut reader), Ok(4));
        assert_eq!(table.decode(&mut reader), Ok(3));
        for word in [0x8000, 0x4001, 0xffff] {
            let data = pack(&[(word, 16)]);
            let mut reader = EntropyBitReader::new(&data);
            assert_eq!(table.decode(&mut reader), Err(HuffmanError::InvalidCode));
            assert_eq!(reader.position(), 0);
        }
    }

    #[test]
    fn root_and_subtable_fallbacks_do_not_require_lookahead_bytes() {
        let mut counts = [0; 16];
        counts[0] = 1;
        let short = HuffmanTable::new(&counts, &[7]).unwrap();
        for data in [&[0x00, 0xff, 0xd9][..], &[0x00][..], &[0x00, 0xff][..]] {
            let mut reader = EntropyBitReader::new(data);
            reader.read_bits(7).unwrap();
            assert_eq!(short.decode(&mut reader), Ok(7));
            assert_eq!((reader.position(), reader.bits_left()), (1, 0));
        }
        counts[0] = 0;
        counts[8] = 1;
        counts[15] = 1;
        let long = HuffmanTable::new(&counts, &[8, 9]).unwrap();
        let mut reader = EntropyBitReader::new(&[0x00, 0x00, 0xff, 0xd9]);
        reader.read_bits(7).unwrap();
        assert_eq!(long.decode(&mut reader), Ok(8));
        assert_eq!((reader.position(), reader.bits_left()), (2, 0));
        assert_eq!(reader.peek_marker().unwrap().code, 0xd9);
    }

    #[test]
    fn truncation_and_restart_do_not_supply_missing_code_bits() {
        let mut counts = [0; 16];
        counts[15] = 1;
        let table = HuffmanTable::new(&counts, &[7]).unwrap();
        let mut reader = EntropyBitReader::new(&[0x00]);
        assert_eq!(
            table.decode(&mut reader),
            Err(HuffmanError::Entropy(EntropyError::UnexpectedEnd {
                position: 1
            }))
        );
        assert_eq!((reader.position(), reader.bits_left()), (0, 0));
        let mut reader = EntropyBitReader::new(&[0x00, 0xff, 0xd0, 0x00]);
        assert!(matches!(
            table.decode(&mut reader),
            Err(HuffmanError::Entropy(EntropyError::MarkerEncountered(_)))
        ));
        assert_eq!(reader.position(), 0);
        let mut reader = EntropyBitReader::new(&[0x00, 0xff]);
        assert_eq!(
            table.decode(&mut reader),
            Err(HuffmanError::Entropy(EntropyError::TruncatedMarker {
                position: 1
            }))
        );
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn symbols_restart_fill_and_nonstandard_padding_reconstruct_exactly() {
        let mut counts = [0; 16];
        counts[1] = 2;
        let table = HuffmanTable::new(&counts, &[10, 20]).unwrap();
        // Codes 00 and 01, mixed padding, marker fill, then 01 and one-padding.
        let data = [0b0001_0101, 0xff, 0xff, 0xd0, 0b0111_1111, 0xff, 0xd9];
        let mut reader = EntropyBitReader::new(&data);
        let mut rebuilt = Vec::new();
        let mut byte = 0u8;
        for symbol in [10, 20] {
            assert_eq!(table.decode(&mut reader), Ok(symbol));
            let code = table.code(symbol).unwrap();
            byte = (byte << code.bit_length) | code.bits as u8;
        }
        let boundary = reader.consume_restart(0, PaddingPolicy::Preserve).unwrap();
        rebuilt.push((byte << boundary.padding.bit_count()) | boundary.padding.value());
        rebuilt.extend_from_slice(&data[boundary.marker.offset..boundary.marker.end]);
        assert_eq!(table.decode(&mut reader), Ok(20));
        let boundary = reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
        let code = table.code(20).unwrap();
        rebuilt
            .push(((code.bits as u8) << boundary.padding.bit_count()) | boundary.padding.value());
        rebuilt.extend_from_slice(&data[boundary.marker.offset..boundary.marker.end]);
        assert_eq!(rebuilt, data);
        assert_eq!(reader.position(), 5);
    }

    #[test]
    fn validates_counts_symbols_oversubscription_and_all_ones() {
        assert!(matches!(
            HuffmanTable::new(&[0; 16], &[]),
            Err(HuffmanError::EmptyTable)
        ));
        let mut counts = [0; 16];
        counts[1] = 1;
        assert!(matches!(
            HuffmanTable::new(&counts, &[]),
            Err(HuffmanError::SymbolCountMismatch {
                expected: 1,
                actual: 0
            })
        ));
        assert!(matches!(
            HuffmanTable::new(&counts, &[0, 1]),
            Err(HuffmanError::SymbolCountMismatch {
                expected: 1,
                actual: 2
            })
        ));
        counts[1] = 2;
        assert!(matches!(
            HuffmanTable::new(&counts, &[1, 1]),
            Err(HuffmanError::DuplicateSymbol(1))
        ));
        counts[0] = 1;
        counts[1] = 3;
        assert!(matches!(
            HuffmanTable::new(&counts, &[0, 1, 2, 3]),
            Err(HuffmanError::Oversubscribed { bit_length: 2 })
        ));
        counts[0] = 2;
        counts[1] = 0;
        assert!(matches!(
            HuffmanTable::new(&counts, &[0, 1]),
            Err(HuffmanError::ForbiddenAllOnes)
        ));
        counts[0] = 0;
        counts[1] = 4;
        assert!(matches!(
            HuffmanTable::new(&counts, &[0, 1, 2, 3]),
            Err(HuffmanError::ForbiddenAllOnes)
        ));
        counts[0] = 255;
        counts[1] = 2;
        assert!(matches!(
            HuffmanTable::new(&counts, &[]),
            Err(HuffmanError::TooManySymbols(257))
        ));
    }

    #[test]
    fn standard_luminance_dc_table_has_expected_codes() {
        let counts = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let symbols: Vec<u8> = (0..12).collect();
        let table = HuffmanTable::new(&counts, &symbols).unwrap();
        assert_eq!(
            table.code(0),
            Some(HuffmanCode {
                bits: 0,
                bit_length: 2
            })
        );
        assert_eq!(
            table.code(5),
            Some(HuffmanCode {
                bits: 0b110,
                bit_length: 3
            })
        );
        assert_eq!(
            table.code(11),
            Some(HuffmanCode {
                bits: 0b111111110,
                bit_length: 9
            })
        );
        let data = packed_symbols(&table, &symbols);
        let mut reader = EntropyBitReader::new(&data);
        for symbol in symbols {
            assert_eq!(table.decode(&mut reader), Ok(symbol));
        }
    }

    #[test]
    fn exhaustive_sixteen_bit_windows_match_independent_canonical_reference() {
        let mut sparse = [0; 16];
        sparse[1] = 1;
        sparse[15] = 1;
        let mut dense = [0; 16];
        dense[7] = 252;
        dense[8] = 3;
        dense[15] = 1;
        for counts in [[1; 16], sparse, dense] {
            let count: usize = counts.iter().map(|&n| usize::from(n)).sum();
            let symbols: Vec<u8> = (0..count).map(|n| n as u8).collect();
            let table = HuffmanTable::new(&counts, &symbols).unwrap();
            // Build a separate, expanded 16-bit reference without table.code().
            let mut reference = vec![None; 1 << 16];
            let mut cursor = 0;
            let mut symbol = 0u16;
            for (index, &count) in counts.iter().enumerate() {
                let span = 1usize << (15 - index);
                for _ in 0..count {
                    reference[cursor..cursor + span].fill(Some(symbol as u8));
                    cursor += span;
                    symbol += 1;
                }
            }
            for (word, expected) in reference.into_iter().enumerate() {
                let data = pack(&[(word as u32, 16)]);
                let mut reader = EntropyBitReader::new(&data);
                let actual = table.decode(&mut reader);
                assert_eq!(
                    actual,
                    expected.ok_or(HuffmanError::InvalidCode),
                    "word {word:04x}"
                );
                if expected.is_none() {
                    assert_eq!(reader.position(), 0);
                }
            }
        }
    }
}
