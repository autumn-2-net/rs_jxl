// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; the complete upstream notice and license
// are retained in docs/PORTING_KERNELS.md.

//! Direct prefix-code port from libjxl b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Source mapping:
//! - `enc_huffman_tree.{h,cc}`: `HuffmanTree`, `CreateHuffmanTree`, `SetDepth`,
//!   `ConvertBitDepthsToSymbols`, `WriteHuffmanTree`, `DecideOverRleUse`, and
//!   both zero/nonzero repetition encoders.
//! - `enc_huffman.{h,cc}`: `BuildAndStoreHuffmanTree`, `StoreSimpleHuffmanTree`,
//!   and the two-level code-length histogram serialization.
//! - `enc_ans.cc`: `BuildAndStoreANSEncodingData`'s prefix branch, including
//!   the implicit histogram for an externally declared alphabet of size one.
//! - `enc_fast_lossless.cc`: `PrefixCode::WriteTo` uses the same code-length
//!   order and static code; its specialized LZ77 layout is not needed here.
//!
//! Alphabets cover the full format domain, 1..=32768 entries. The caller writes exactly
//! `alphabet_size()` externally, plus the prefix mode and HybridUint config.
//! The source RLE policy for alphabets above 50 entries is retained, including
//! chained repeat codes 16/17 and their two/three extra bits. No source Huffman
//! histogram-serialization branch is omitted within the format's alphabet bound.
//! No LZ77 layout or ANS coding is added.

use crate::bit_writer::{BitWriter, BitWriterError};
use std::fmt;

pub const MAX_ALPHABET_SIZE: usize = 1 << 15;
const MAX_CODE_LENGTH: u8 = 15;
const CODE_LENGTH_CODES: usize = 18;
const CODE_LENGTH_ORDER: [usize; CODE_LENGTH_CODES] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const CODE_LENGTH_LENGTH_BITS: [u64; 6] = [0, 7, 3, 2, 1, 15];
const CODE_LENGTH_LENGTH_NBITS: [u8; 6] = [2, 4, 3, 2, 2, 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCodeError {
    InvalidAlphabetSize,
    EmptyHistogram,
    InvalidCodeLengths,
    SymbolOutOfRange,
    SymbolAbsent,
    BitWriter(BitWriterError),
}

impl fmt::Display for PrefixCodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlphabetSize => f.write_str("prefix alphabet size must be in 1..=32768"),
            Self::EmptyHistogram => f.write_str("prefix histogram has no nonzero counts"),
            Self::InvalidCodeLengths => f.write_str("invalid or overlong prefix code lengths"),
            Self::SymbolOutOfRange => f.write_str("prefix symbol is outside the declared alphabet"),
            Self::SymbolAbsent => f.write_str("prefix symbol has a zero histogram count"),
            Self::BitWriter(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for PrefixCodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BitWriter(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BitWriterError> for PrefixCodeError {
    fn from(error: BitWriterError) -> Self {
        Self::BitWriter(error)
    }
}

/// Canonical, LSB-first symbol codes and their JXL/Brotli histogram.
///
/// Zero-count symbols are absent, not zero-bit codes. Only the sole present
/// symbol in a singleton histogram has a zero-bit code.
#[derive(Clone, Debug)]
pub struct PrefixCode {
    depths: Vec<u8>,
    bits: Vec<u16>,
    singleton: Option<usize>,
}

impl PrefixCode {
    /// Builds the upstream length-limited Huffman tree without trimming the
    /// declared alphabet. Empty slices, all-zero counts, and more than 32768
    /// entries are rejected. Individual counts may use the full `u64` range.
    pub fn from_counts(counts: &[u64]) -> Result<Self, PrefixCodeError> {
        if counts.is_empty() || counts.len() > MAX_ALPHABET_SIZE {
            return Err(PrefixCodeError::InvalidAlphabetSize);
        }
        let mut present = counts
            .iter()
            .enumerate()
            .filter_map(|(symbol, &count)| (count != 0).then_some(symbol));
        let first = present.next().ok_or(PrefixCodeError::EmptyHistogram)?;
        if present.next().is_none() {
            return Ok(Self {
                depths: vec![0; counts.len()],
                bits: vec![0; counts.len()],
                singleton: Some(first),
            });
        }
        let depths = create_huffman_tree(counts, MAX_CODE_LENGTH);
        let bits = convert_bit_depths_to_symbols(&depths)?;
        Ok(Self {
            depths,
            bits,
            singleton: None,
        })
    }

    /// The exact input slice length, including trailing zero counts. The
    /// caller must serialize this size, not the number of present symbols.
    pub fn alphabet_size(&self) -> usize {
        self.depths.len()
    }

    /// Writes only the histogram, with no alphabet size or entropy/config
    /// header. A size-one alphabet has an implicit histogram and writes zero
    /// bits, following `enc_ans.cc`. A singleton in a larger alphabet writes
    /// the simple form. Writer errors propagate; earlier writes are retained.
    pub fn write_histogram(&self, writer: &mut BitWriter) -> Result<(), PrefixCodeError> {
        if self.alphabet_size() == 1 {
            return Ok(());
        }
        let symbol_nbits = (usize::BITS - (self.alphabet_size() - 1).leading_zeros()) as u8;
        if let Some(symbol) = self.singleton {
            writer.write(4, 1)?;
            writer.write(symbol_nbits, symbol as u64)?;
            return Ok(());
        }

        let mut symbols = [0usize; 4];
        let mut num_symbols = 0;
        for (symbol, &depth) in self.depths.iter().enumerate() {
            if depth != 0 {
                if num_symbols < symbols.len() {
                    symbols[num_symbols] = symbol;
                }
                num_symbols += 1;
            }
        }
        if num_symbols <= 4 {
            writer.write(2, 1)?;
            writer.write(2, (num_symbols - 1) as u64)?;
            // Preserve StoreSimpleHuffmanTree's exact swap order, including ties.
            for i in 0..num_symbols {
                for j in i + 1..num_symbols {
                    if self.depths[symbols[j]] < self.depths[symbols[i]] {
                        symbols.swap(i, j);
                    }
                }
            }
            for &symbol in &symbols[..num_symbols] {
                writer.write(symbol_nbits, symbol as u64)?;
            }
            if num_symbols == 4 {
                writer.write(1, u64::from(self.depths[symbols[0]] == 1))?;
            }
            return Ok(());
        }
        self.write_complex_histogram(writer)
    }

    /// Number of encoded bits for a present symbol, excluding HybridUint extra
    /// bits. The singleton symbol costs zero; absent symbols remain errors.
    #[inline]
    pub fn symbol_bits(&self, symbol: usize) -> Result<u8, PrefixCodeError> {
        let &depth = self
            .depths
            .get(symbol)
            .ok_or(PrefixCodeError::SymbolOutOfRange)?;
        if depth == 0 && self.singleton != Some(symbol) {
            return Err(PrefixCodeError::SymbolAbsent);
        }
        Ok(depth)
    }

    /// Writes a present symbol's canonical code. HybridUint extra bits are
    /// separate and must be written by the caller after this operation.
    pub fn write_symbol(
        &self,
        symbol: usize,
        writer: &mut BitWriter,
    ) -> Result<(), PrefixCodeError> {
        let depth = self.symbol_bits(symbol)?;
        writer.write(depth, u64::from(self.bits[symbol]))?;
        Ok(())
    }

    /// enc_ans.cc::WriteTokens combines the canonical code and HybridUint bits.
    pub(crate) fn write_symbol_with_extra(
        &self,
        symbol: usize,
        extra_width: u8,
        extra_bits: u32,
        writer: &mut BitWriter,
    ) -> Result<(), PrefixCodeError> {
        let depth = self.symbol_bits(symbol)?;
        if extra_width > 32 {
            return Err(BitWriterError::TooManyBits.into());
        }
        if u64::from(extra_bits) >> extra_width != 0 {
            return Err(BitWriterError::ValueOutOfRange.into());
        }
        writer.write(
            depth + extra_width,
            u64::from(self.bits[symbol]) | (u64::from(extra_bits) << depth),
        )?;
        Ok(())
    }

    fn write_complex_histogram(&self, writer: &mut BitWriter) -> Result<(), PrefixCodeError> {
        let tree = write_huffman_tree(&self.depths);
        let mut histogram = [0u64; CODE_LENGTH_CODES];
        for token in &tree {
            histogram[usize::from(token.symbol)] += 1;
        }
        let num_codes = histogram.iter().filter(|&&count| count != 0).count();
        let mut code_length_depths = create_huffman_tree(&histogram, 5);
        let code_length_bits = convert_bit_depths_to_symbols(&code_length_depths)?;

        // StoreHuffmanTreeOfHuffmanTreeToBitMask. For a singleton code-length
        // alphabet, all remaining entries must be stored: its Kraft sum is 1/2.
        let mut codes_to_store = CODE_LENGTH_CODES;
        if num_codes > 1 {
            while code_length_depths[CODE_LENGTH_ORDER[codes_to_store - 1]] == 0 {
                codes_to_store -= 1;
            }
        }
        let mut skip_some = 0;
        if code_length_depths[CODE_LENGTH_ORDER[0]] == 0
            && code_length_depths[CODE_LENGTH_ORDER[1]] == 0
        {
            skip_some = 2;
            if code_length_depths[CODE_LENGTH_ORDER[2]] == 0 {
                skip_some = 3;
            }
        }
        writer.write(2, skip_some as u64)?;
        for &code in &CODE_LENGTH_ORDER[skip_some..codes_to_store] {
            let depth = usize::from(code_length_depths[code]);
            writer.write(
                CODE_LENGTH_LENGTH_NBITS[depth],
                CODE_LENGTH_LENGTH_BITS[depth],
            )?;
        }

        if num_codes == 1 {
            let code = histogram.iter().position(|&count| count != 0).unwrap();
            code_length_depths[code] = 0;
        }
        // StoreHuffmanTreeToBitMask writes repeat payloads after their codes.
        for token in tree {
            let code = usize::from(token.symbol);
            writer.write(code_length_depths[code], u64::from(code_length_bits[code]))?;
            match token.symbol {
                16 => writer.write(2, u64::from(token.extra_bits))?,
                17 => writer.write(3, u64::from(token.extra_bits))?,
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CodeLengthToken {
    symbol: u8,
    extra_bits: u8,
}

fn literal_depth(symbol: u8) -> CodeLengthToken {
    CodeLengthToken {
        symbol,
        extra_bits: 0,
    }
}

fn write_repetitions(
    previous: u8,
    value: u8,
    mut repetitions: usize,
    tree: &mut Vec<CodeLengthToken>,
) {
    debug_assert!(value != 0 && repetitions > 0);
    if previous != value {
        tree.push(literal_depth(value));
        repetitions -= 1;
    }
    if repetitions == 7 {
        tree.push(literal_depth(value));
        repetitions -= 1;
    }
    if repetitions < 3 {
        tree.extend(std::iter::repeat_n(literal_depth(value), repetitions));
    } else {
        repetitions -= 3;
        let start = tree.len();
        loop {
            tree.push(CodeLengthToken {
                symbol: 16,
                extra_bits: (repetitions & 3) as u8,
            });
            repetitions >>= 2;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        tree[start..].reverse();
    }
}

fn write_zero_repetitions(mut repetitions: usize, tree: &mut Vec<CodeLengthToken>) {
    debug_assert!(repetitions > 0);
    if repetitions == 11 {
        tree.push(literal_depth(0));
        repetitions -= 1;
    }
    if repetitions < 3 {
        tree.extend(std::iter::repeat_n(literal_depth(0), repetitions));
    } else {
        repetitions -= 3;
        let start = tree.len();
        loop {
            tree.push(CodeLengthToken {
                symbol: 17,
                extra_bits: (repetitions & 7) as u8,
            });
            repetitions >>= 3;
            if repetitions == 0 {
                break;
            }
            repetitions -= 1;
        }
        tree[start..].reverse();
    }
}

fn decide_over_rle_use(depths: &[u8]) -> (bool, bool) {
    let mut total_zero = 0;
    let mut total_nonzero = 0;
    let mut runs_zero = 1;
    let mut runs_nonzero = 1;
    let mut i = 0;
    while i < depths.len() {
        let value = depths[i];
        let mut repetitions = 1;
        while i + repetitions < depths.len() && depths[i + repetitions] == value {
            repetitions += 1;
        }
        if value == 0 && repetitions >= 3 {
            total_zero += repetitions;
            runs_zero += 1;
        }
        if value != 0 && repetitions >= 4 {
            total_nonzero += repetitions;
            runs_nonzero += 1;
        }
        i += repetitions;
    }
    (total_nonzero > runs_nonzero * 2, total_zero > runs_zero * 2)
}

fn write_huffman_tree(depths: &[u8]) -> Vec<CodeLengthToken> {
    let length = depths
        .iter()
        .rposition(|&depth| depth != 0)
        .map_or(0, |i| i + 1);
    let (use_nonzero_rle, use_zero_rle) = if depths.len() > 50 {
        decide_over_rle_use(&depths[..length])
    } else {
        (false, false)
    };
    let mut tree = Vec::with_capacity(length);
    let mut previous = 8;
    let mut i = 0;
    while i < length {
        let value = depths[i];
        let mut repetitions = 1;
        if (value == 0 && use_zero_rle) || (value != 0 && use_nonzero_rle) {
            while i + repetitions < length && depths[i + repetitions] == value {
                repetitions += 1;
            }
        }
        if value == 0 {
            write_zero_repetitions(repetitions, &mut tree);
        } else {
            write_repetitions(previous, value, repetitions, &mut tree);
            previous = value;
        }
        i += repetitions;
    }
    tree
}

#[derive(Clone, Copy)]
struct HuffmanTree {
    // Widen upstream uint32_t weights to support u64 counts without overflow.
    total_count: u128,
    index_left: Option<usize>,
    index_right_or_value: usize,
}

fn set_depth(tree: &[HuffmanTree], index: usize, depths: &mut [u8], level: u8) {
    let node = tree[index];
    if let Some(left) = node.index_left {
        set_depth(tree, left, depths, level + 1);
        set_depth(tree, node.index_right_or_value, depths, level + 1);
    } else {
        depths[node.index_right_or_value] = level;
    }
}

/// Direct CreateHuffmanTree port: sorted leaves, two queues with sentinels,
/// and doubled minimum counts until every leaf satisfies the depth limit.
fn create_huffman_tree(counts: &[u64], tree_limit: u8) -> Vec<u8> {
    debug_assert!(counts.iter().any(|&count| count != 0));
    debug_assert!(counts.len() <= MAX_ALPHABET_SIZE);
    let mut depths = vec![0; counts.len()];
    let mut count_limit = 1u128;
    loop {
        let mut tree = Vec::with_capacity(2 * counts.len() + 1);
        for symbol in (0..counts.len()).rev() {
            if counts[symbol] != 0 {
                tree.push(HuffmanTree {
                    total_count: u128::from(counts[symbol]).max(count_limit - 1),
                    index_left: None,
                    index_right_or_value: symbol,
                });
            }
        }
        let n = tree.len();
        if n == 1 {
            depths[tree[0].index_right_or_value] = 1;
            return depths;
        }
        tree.sort_unstable_by(|a, b| {
            a.total_count
                .cmp(&b.total_count)
                .then_with(|| b.index_right_or_value.cmp(&a.index_right_or_value))
        });
        let sentinel = HuffmanTree {
            total_count: u128::MAX,
            index_left: None,
            index_right_or_value: 0,
        };
        tree.push(sentinel);
        tree.push(sentinel);
        let mut i = 0;
        let mut j = n + 1;
        for _ in 0..n - 1 {
            let left = if tree[i].total_count <= tree[j].total_count {
                i += 1;
                i - 1
            } else {
                j += 1;
                j - 1
            };
            let right = if tree[i].total_count <= tree[j].total_count {
                i += 1;
                i - 1
            } else {
                j += 1;
                j - 1
            };
            let j_end = tree.len() - 1;
            tree[j_end] = HuffmanTree {
                total_count: tree[left].total_count + tree[right].total_count,
                index_left: Some(left),
                index_right_or_value: right,
            };
            tree.push(sentinel);
        }
        debug_assert_eq!(tree.len(), 2 * n + 1);
        set_depth(&tree, 2 * n - 1, &mut depths, 0);
        if depths.iter().all(|&depth| depth <= tree_limit) {
            return depths;
        }
        // At 2^64 all supported u64 weights are equal, so depth is at most 15
        // (at most 5 for the 18-entry code-length alphabet). u128 cannot wrap.
        count_limit *= 2;
    }
}

fn convert_bit_depths_to_symbols(depths: &[u8]) -> Result<Vec<u16>, PrefixCodeError> {
    let mut bl_count = [0u32; 16];
    let mut space = 1u32 << MAX_CODE_LENGTH;
    let mut present = 0;
    for &depth in depths {
        if depth > MAX_CODE_LENGTH {
            return Err(PrefixCodeError::InvalidCodeLengths);
        }
        if depth != 0 {
            space = space
                .checked_sub(1 << (MAX_CODE_LENGTH - depth))
                .ok_or(PrefixCodeError::InvalidCodeLengths)?;
            present += 1;
            bl_count[usize::from(depth)] += 1;
        }
    }
    if present == 0 || (space != 0 && !(present == 1 && bl_count[1] == 1)) {
        return Err(PrefixCodeError::InvalidCodeLengths);
    }

    let mut next_code = [0u32; 16];
    let mut code = 0;
    for i in 1..16 {
        code = (code + bl_count[i - 1]) << 1;
        next_code[i] = code;
    }
    let mut bits = vec![0; depths.len()];
    for (symbol, &depth) in depths.iter().enumerate() {
        if depth != 0 {
            let next = &mut next_code[usize::from(depth)];
            // Equivalent to upstream ReverseBits' reversed-nibble lookup.
            bits[symbol] = (*next as u16).reverse_bits() >> (16 - depth);
            *next += 1;
        }
    }
    Ok(bits)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn combined_symbol_and_extra_bits_cover_singletons_boundaries_and_errors() {
        for counts in [&[0, 7][..], &[9, 2, 1, 4, 0, 1][..]] {
            let code = PrefixCode::from_counts(counts).unwrap();
            for (symbol, &count) in counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                for width in [0, 1, 7, 16, 31, 32] {
                    let bits = (u32::MAX as u64 >> (32 - width)) as u32;
                    for offset in 0..8 {
                        let mut actual = BitWriter::new(32);
                        let mut expected = BitWriter::new(32);
                        actual.write(offset, (1 << offset) - 1).unwrap();
                        expected.write(offset, (1 << offset) - 1).unwrap();
                        code.write_symbol_with_extra(symbol, width, bits, &mut actual)
                            .unwrap();
                        code.write_symbol(symbol, &mut expected).unwrap();
                        expected.write(width, u64::from(bits)).unwrap();
                        assert_eq!(finish(actual), finish(expected));
                    }
                }
            }
        }
        let code = PrefixCode::from_counts(&[0, 7]).unwrap();
        for (symbol, width, bits) in [(0, 0, 0), (2, 0, 0), (1, 33, 0), (1, 0, 1), (1, 2, 4)] {
            let mut writer = BitWriter::new(32);
            writer.write(3, 5).unwrap();
            assert!(
                code.write_symbol_with_extra(symbol, width, bits, &mut writer)
                    .is_err()
            );
            assert_eq!(writer.bits_written(), 3);
        }
    }

    fn finish(mut writer: BitWriter) -> (usize, Vec<u8>) {
        let nbits = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        (nbits, writer.into_bytes().unwrap())
    }

    fn histogram(counts: &[u64]) -> (usize, Vec<u8>) {
        let code = PrefixCode::from_counts(counts).unwrap();
        let mut writer = BitWriter::new(1024);
        code.write_histogram(&mut writer).unwrap();
        finish(writer)
    }

    // Small independent reader for the emitted format, following dec_huffman.cc.
    // It deliberately does not reuse the encoder's canonical-code construction.
    pub(crate) struct TestBits<'a> {
        pub(crate) bytes: &'a [u8],
        pub(crate) position: usize,
        pub(crate) nbits: usize,
        pub(crate) repeat_codes: [usize; 2],
    }

    impl TestBits<'_> {
        fn read(&mut self, width: u8) -> usize {
            assert!(self.position + usize::from(width) <= self.nbits);
            let mut value = 0;
            for bit in 0..width {
                value |=
                    usize::from((self.bytes[self.position / 8] >> (self.position % 8)) & 1) << bit;
                self.position += 1;
            }
            value
        }

        pub(crate) fn read_canonical(&mut self, depths: &[u8]) -> usize {
            let present = depths.iter().filter(|&&depth| depth != 0).count();
            if present == 1 {
                return depths.iter().position(|&depth| depth != 0).unwrap();
            }
            let mut code = 0;
            let mut first = 0;
            for width in 1..=15 {
                code = (code << 1) | self.read(1);
                let count = depths.iter().filter(|&&depth| depth == width).count();
                if code >= first && code - first < count {
                    return depths
                        .iter()
                        .enumerate()
                        .filter(|&(_, &depth)| depth == width)
                        .nth(code - first)
                        .unwrap()
                        .0;
                }
                first = (first + count) << 1;
            }
            panic!("invalid canonical symbol");
        }

        pub(crate) fn read_histogram(&mut self, alphabet_size: usize) -> (Vec<u8>, Option<usize>) {
            let mut depths = vec![0; alphabet_size];
            if alphabet_size == 1 {
                return (depths, Some(0));
            }
            let simple_or_skip = self.read(2);
            if simple_or_skip == 1 {
                let num_symbols = self.read(2) + 1;
                let width = (usize::BITS - (alphabet_size - 1).leading_zeros()) as u8;
                let symbols: Vec<_> = (0..num_symbols).map(|_| self.read(width)).collect();
                if num_symbols == 1 {
                    return (depths, Some(symbols[0]));
                }
                let lengths: &[u8] = match num_symbols {
                    2 => &[1, 1],
                    3 => &[1, 2, 2],
                    4 if self.read(1) == 0 => &[2, 2, 2, 2],
                    4 => &[1, 2, 3, 3],
                    _ => unreachable!(),
                };
                for (&symbol, &depth) in symbols.iter().zip(lengths) {
                    assert!(symbol < alphabet_size);
                    assert_eq!(depths[symbol], 0);
                    depths[symbol] = depth;
                }
                return (depths, None);
            }

            let mut meta = [0; 18];
            let mut space = 32;
            // Order and static code are written out independently of the encoder.
            let order = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
            for &symbol in &order[simple_or_skip..] {
                let depth = match self.read(2) {
                    0 => 0,
                    1 => 4,
                    2 => 3,
                    3 if self.read(1) == 0 => 2,
                    3 if self.read(1) == 0 => 1,
                    3 => 5,
                    _ => unreachable!(),
                };
                meta[symbol] = depth;
                if depth != 0 {
                    space -= 32 >> depth;
                    assert!(space >= 0);
                    if space == 0 {
                        break;
                    }
                }
            }
            assert!(space == 0 || meta.iter().filter(|&&depth| depth != 0).count() == 1);
            let mut space = 32768i32;
            let mut symbol = 0;
            let mut previous = 8;
            let mut repeated_length = 0;
            let mut repeat_count = 0;
            while symbol < alphabet_size && space > 0 {
                let code = self.read_canonical(&meta) as u8;
                let (value, count) = if code < 16 {
                    repeat_count = 0;
                    if code != 0 {
                        previous = code;
                    }
                    (code, 1)
                } else {
                    assert!(code <= 17);
                    self.repeat_codes[usize::from(code - 16)] += 1;
                    let value = if code == 16 { previous } else { 0 };
                    if value != repeated_length {
                        repeat_count = 0;
                        repeated_length = value;
                    }
                    let previous_count = repeat_count;
                    let radix = if code == 16 { 4 } else { 8 };
                    let digit = self.read(code - 14);
                    repeat_count = if previous_count == 0 {
                        digit + 3
                    } else {
                        (previous_count - 2) * radix + digit + 3
                    };
                    assert!(repeat_count > previous_count);
                    (value, repeat_count - previous_count)
                };
                assert!(symbol + count <= alphabet_size);
                depths[symbol..symbol + count].fill(value);
                symbol += count;
                if value != 0 {
                    space -= (count as i32) << (15 - value);
                }
                assert!(space >= 0);
            }
            assert_eq!(space, 0);
            (depths, None)
        }
    }

    fn assert_prefix_code(counts: &[u64]) {
        let code = PrefixCode::from_counts(counts).unwrap();
        assert_eq!(code.alphabet_size(), counts.len());
        let mut writer = BitWriter::new(1024);
        writer.write(3, 5).unwrap();
        code.write_histogram(&mut writer).unwrap();
        let (nbits, bytes) = finish(writer);
        let mut reader = TestBits {
            bytes: &bytes,
            position: 0,
            nbits,
            repeat_codes: [0; 2],
        };
        assert_eq!(reader.read(3), 5);
        let (depths, singleton) = reader.read_histogram(counts.len());
        assert_eq!(depths, code.depths);
        assert_eq!(singleton, code.singleton);
        assert_eq!(reader.position, nbits);
        let present: Vec<_> = (0..counts.len()).filter(|&i| counts[i] != 0).collect();
        if present.len() == 1 {
            assert_eq!(code.singleton, Some(present[0]));
            assert!(code.depths.iter().all(|&depth| depth == 0));
            return;
        }
        let mut kraft = 0u32;
        for (symbol, &count) in counts.iter().enumerate() {
            let depth = code.depths[symbol];
            if count == 0 {
                assert_eq!(depth, 0);
                assert_eq!(code.bits[symbol], 0);
                continue;
            }
            assert!((1..=15).contains(&depth));
            assert!(code.bits[symbol] < 1 << depth);
            kraft += 1 << (15 - depth);
            for &other in &present {
                if symbol != other && depth <= code.depths[other] {
                    let mask = (1u16 << depth) - 1;
                    assert_ne!(code.bits[symbol], code.bits[other] & mask);
                }
            }
        }
        assert_eq!(kraft, 1 << 15);

        let mut writer = BitWriter::new(1024);
        for &symbol in &present {
            code.write_symbol(symbol, &mut writer).unwrap();
        }
        let (nbits, bytes) = finish(writer);
        let mut reader = TestBits {
            bytes: &bytes,
            position: 0,
            nbits,
            repeat_codes: [0; 2],
        };
        for &expected in &present {
            assert_eq!(reader.read_canonical(&depths), expected);
        }
        assert_eq!(reader.position, nbits);
    }

    #[test]
    fn wide_alphabets_decode_histograms_and_payloads_independently() {
        let mut sparse = vec![0; MAX_ALPHABET_SIZE];
        for (symbol, count) in [(0, 20), (256, 4), (512, 3), (4096, 2), (32767, 1)] {
            sparse[symbol] = count;
        }
        let mut singleton = vec![0; MAX_ALPHABET_SIZE];
        singleton[32767] = 7;
        for counts in [vec![1; 257], vec![1; MAX_ALPHABET_SIZE], sparse, singleton] {
            let code = PrefixCode::from_counts(&counts).unwrap();
            let present: Vec<_> = counts
                .iter()
                .enumerate()
                .filter(|&(_, &n)| n != 0)
                .map(|(s, _)| s)
                .collect();
            let samples = [
                present[0],
                present[present.len() / 2],
                *present.last().unwrap(),
            ];
            let mut writer = BitWriter::new(1 << 20);
            writer.write(3, 5).unwrap();
            code.write_histogram(&mut writer).unwrap();
            let header_end = writer.bits_written();
            for symbol in samples {
                code.write_symbol(symbol, &mut writer).unwrap();
            }
            let (nbits, bytes) = finish(writer);
            let mut reader = TestBits {
                bytes: &bytes,
                position: 0,
                nbits,
                repeat_codes: [0; 2],
            };
            assert_eq!(reader.read(3), 5);
            let (depths, singleton) = reader.read_histogram(counts.len());
            assert_eq!(reader.position, header_end);
            assert_eq!(depths, code.depths);
            assert_eq!(singleton, code.singleton);
            for expected in samples {
                let actual = singleton.unwrap_or_else(|| reader.read_canonical(&depths));
                assert_eq!(actual, expected);
            }
            assert_eq!(reader.position, nbits);
            if counts.iter().all(|&n| n == 1) && counts.len() == MAX_ALPHABET_SIZE {
                assert!(depths.iter().all(|&depth| depth == 15));
                assert!(reader.repeat_codes[0] > 0);
            }
        }
    }

    #[test]
    fn rejects_invalid_histograms_and_absent_symbols() {
        assert!(matches!(
            PrefixCode::from_counts(&[]),
            Err(PrefixCodeError::InvalidAlphabetSize)
        ));
        assert!(matches!(
            PrefixCode::from_counts(&vec![1; MAX_ALPHABET_SIZE + 1]),
            Err(PrefixCodeError::InvalidAlphabetSize)
        ));
        assert!(matches!(
            PrefixCode::from_counts(&[0; 19]),
            Err(PrefixCodeError::EmptyHistogram)
        ));
        for counts in [&[0, 7, 0][..], &[0, 7, 2][..]] {
            let code = PrefixCode::from_counts(counts).unwrap();
            let mut writer = BitWriter::new(8);
            writer.write(3, 5).unwrap();
            assert_eq!(
                code.write_symbol(0, &mut writer),
                Err(PrefixCodeError::SymbolAbsent)
            );
            assert_eq!(
                code.write_symbol(3, &mut writer),
                Err(PrefixCodeError::SymbolOutOfRange)
            );
            assert_eq!(writer.bits_written(), 3);
        }
        for depths in [&[0, 0][..], &[16, 1][..], &[1, 1, 1][..], &[2, 2][..]] {
            assert_eq!(
                convert_bit_depths_to_symbols(depths),
                Err(PrefixCodeError::InvalidCodeLengths)
            );
        }
    }

    #[test]
    fn singleton_zero_and_nonzero_known_forms() {
        assert_eq!(histogram(&[8]), (0, vec![]));
        let mut counts = [0; 19];
        counts[0] = 1;
        assert_eq!(histogram(&counts), (9, vec![0x01, 0x00]));
        counts[0] = 0;
        counts[18] = 5;
        assert_eq!(histogram(&counts), (9, vec![0x21, 0x01]));
        let code = PrefixCode::from_counts(&counts).unwrap();
        let mut writer = BitWriter::new(0);
        for _ in 0..100 {
            code.write_symbol(18, &mut writer).unwrap();
        }
        assert_eq!(writer.bits_written(), 0);
        assert_eq!(
            code.write_symbol(0, &mut writer),
            Err(PrefixCodeError::SymbolAbsent)
        );
    }

    #[test]
    fn simple_forms_and_canonical_symbols_match_upstream() {
        // Fields are written LSB first: simple=1, NSYM-1, symbols, tree-select.
        assert_eq!(histogram(&[1, 1]), (6, vec![0x25]));
        // CreateHuffmanTree merges equal-weight symbols 2 and 1 first.
        // StoreSimpleHuffmanTree writes five 2-bit fields: 1, 2, 0, 1, 2.
        // Packed LSB first: 1 | (2 << 2) | (1 << 6) | (2 << 8) = 0x249.
        let three = PrefixCode::from_counts(&[1, 1, 1]).unwrap();
        assert_eq!(three.depths, [1, 2, 2]);
        assert_eq!(three.bits, [0, 1, 3]);
        assert_eq!(histogram(&[1, 1, 1]), (10, vec![0x49, 0x02]));
        assert_eq!(histogram(&[1, 1, 1, 1]), (13, vec![0x4d, 0x0e]));
        assert_eq!(histogram(&[10, 1, 1, 1]), (13, vec![0x4d, 0x1e]));
        let code = PrefixCode::from_counts(&[10, 1, 1, 1]).unwrap();
        assert_eq!(code.depths, [1, 2, 3, 3]);
        assert_eq!(code.bits, [0, 1, 3, 7]);
        let mut writer = BitWriter::new(8);
        for symbol in 0..4 {
            code.write_symbol(symbol, &mut writer).unwrap();
        }
        assert_eq!(finish(writer), (9, vec![0xda, 0x01]));
    }

    #[test]
    fn sparse_simple_form_keeps_the_declared_alphabet() {
        let mut counts = [0; 19];
        counts[0] = 4;
        counts[18] = 1;
        let code = PrefixCode::from_counts(&counts).unwrap();
        assert_eq!(code.alphabet_size(), 19);
        assert_eq!(histogram(&counts), (14, vec![0x05, 0x24]));
        counts[18] = 0;
        counts[2] = 1;
        assert_eq!(histogram(&counts), (14, vec![0x05, 0x04]));
    }

    #[test]
    fn complex_known_forms_cover_code_length_singletons_and_skips() {
        // Equal counts: upstream's descending-index tie break gives 2,2,2,3,3.
        let code = PrefixCode::from_counts(&[1; 5]).unwrap();
        assert_eq!(code.depths, [2, 2, 2, 3, 3]);
        assert_eq!(code.bits, [0, 2, 1, 3, 7]);
        assert_eq!(histogram(&[1; 5]), (17, vec![0x70, 0x87, 0x01]));
        // Eight depth-3 codes: HSKIP=2, then all 16 remaining code-length
        // alphabet entries; the singleton length symbol itself uses no bits.
        assert_eq!(histogram(&[1; 8]), (36, vec![0x1e, 0, 0, 0, 0]));
        // Sixteen depth-4 codes use HSKIP=3 instead.
        assert_eq!(histogram(&[1; 16]), (34, vec![0x1f, 0, 0, 0, 0]));
        assert_eq!(histogram(&[1; 19]), (31, vec![0x1f, 0x07, 0x00, 0x7e]));
        let mut counts = [0; 19];
        counts[..5].fill(1);
        assert_eq!(histogram(&counts), histogram(&[1; 5]));
    }

    #[test]
    fn source_repetition_special_cases_and_chained_digits() {
        let pairs = |tree: Vec<CodeLengthToken>| -> Vec<(u8, u8)> {
            tree.into_iter()
                .map(|token| (token.symbol, token.extra_bits))
                .collect()
        };
        for (previous, value, repetitions, expected) in [
            (8, 8, 3, vec![(16, 0)]),
            (8, 8, 6, vec![(16, 3)]),
            (8, 8, 7, vec![(8, 0), (16, 3)]),
            (8, 8, 8, vec![(16, 0), (16, 1)]),
            (5, 6, 8, vec![(6, 0), (6, 0), (16, 3)]),
            (8, 8, 256, vec![(16, 2), (16, 2), (16, 2), (16, 1)]),
        ] {
            let mut tree = Vec::new();
            write_repetitions(previous, value, repetitions, &mut tree);
            assert_eq!(pairs(tree), expected);
        }
        for (repetitions, expected) in [
            (1, vec![(0, 0)]),
            (3, vec![(17, 0)]),
            (10, vec![(17, 7)]),
            (11, vec![(0, 0), (17, 7)]),
            (12, vec![(17, 0), (17, 1)]),
        ] {
            let mut tree = Vec::new();
            write_zero_repetitions(repetitions, &mut tree);
            assert_eq!(pairs(tree), expected);
        }
    }

    #[test]
    fn source_rle_decisions_use_original_alphabet_length_and_separate_statistics() {
        assert_eq!(decide_over_rle_use(&[3; 4]), (false, false));
        assert_eq!(decide_over_rle_use(&[3; 5]), (true, false));
        assert_eq!(decide_over_rle_use(&[0; 4]), (false, false));
        assert_eq!(decide_over_rle_use(&[0; 5]), (false, true));
        assert_eq!(decide_over_rle_use(&[0, 0, 0, 1, 0, 0, 0]), (false, false));
        assert_eq!(
            decide_over_rle_use(&[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
            (false, true)
        );
        assert!(
            write_huffman_tree(&[8; 50])
                .iter()
                .all(|token| token.symbol == 8)
        );
        assert!(
            write_huffman_tree(&[8; 51])
                .iter()
                .all(|token| token.symbol == 16)
        );

        let mut padded_depths = vec![0; 51];
        padded_depths[..16].fill(4);
        let short = write_huffman_tree(&padded_depths[..16]);
        let padded = write_huffman_tree(&padded_depths);
        assert!(short.iter().all(|token| token.symbol == 4));
        assert!(padded.iter().any(|token| token.symbol == 16));
        assert!(padded.iter().all(|token| token.symbol != 17));
        let mut counts = vec![0; 51];
        counts[..16].fill(1);
        assert_prefix_code(&counts);
    }

    #[test]
    fn emitted_rle_histograms_are_parsed_by_the_independent_decoder() {
        let mut sparse = vec![0; 256];
        for symbol in [0, 30, 60, 90, 120, 255] {
            sparse[symbol] = 1;
        }
        let mut mixed = vec![0; 256];
        mixed[..32].fill(1);
        mixed[160..192].fill(1);
        let mut previous_across_zero_run = vec![0; 256];
        previous_across_zero_run[..4].fill(1);
        previous_across_zero_run[252..].fill(1);
        for (counts, needs_nonzero_repeat, needs_zero_repeat) in [
            (vec![1; 64], true, false),
            (vec![1; 256], true, false),
            (sparse, false, true),
            (mixed, true, true),
            (previous_across_zero_run, true, true),
        ] {
            let code = PrefixCode::from_counts(&counts).unwrap();
            for offset in 0..8u8 {
                let mut writer = BitWriter::new(4096);
                writer.write(offset, (1 << offset) - 1).unwrap();
                code.write_histogram(&mut writer).unwrap();
                let histogram_end = writer.bits_written();
                let present: Vec<usize> = counts
                    .iter()
                    .enumerate()
                    .filter_map(|(symbol, &count)| (count != 0).then_some(symbol))
                    .collect();
                for &symbol in present.iter().rev() {
                    code.write_symbol(symbol, &mut writer).unwrap();
                }
                let (nbits, bytes) = finish(writer);
                let mut reader = TestBits {
                    bytes: &bytes,
                    position: usize::from(offset),
                    nbits,
                    repeat_codes: [0; 2],
                };
                let (decoded_depths, singleton) = reader.read_histogram(counts.len());
                assert_eq!(singleton, None);
                assert_eq!(decoded_depths, code.depths);
                assert_eq!(reader.position, histogram_end);
                if needs_nonzero_repeat {
                    assert!(reader.repeat_codes[0] > 0);
                }
                if needs_zero_repeat {
                    assert!(reader.repeat_codes[1] > 0);
                }
                for &symbol in present.iter().rev() {
                    assert_eq!(reader.read_canonical(&decoded_depths), symbol);
                }
                assert_eq!(reader.position, nbits);
            }
        }
    }

    #[test]
    fn every_zero_run_length_decodes_without_using_encoder_repeat_tokens() {
        for gap in 1..=251 {
            let mut counts = vec![0; gap + 5];
            counts[..4].fill(1);
            counts[gap + 4] = 1;
            assert_prefix_code(&counts);
        }
    }

    #[test]
    fn symbol_bit_cost_matches_emission_and_rejects_absent_symbols() {
        for counts in [vec![9], vec![0, 9, 0], vec![8, 1, 0, 5], vec![1; 256]] {
            let code = PrefixCode::from_counts(&counts).unwrap();
            let mut writer = BitWriter::new(4096);
            writer.write(3, 5).unwrap();
            let mut expected_bits = 3;
            for (symbol, &count) in counts.iter().enumerate() {
                if count == 0 {
                    assert_eq!(code.symbol_bits(symbol), Err(PrefixCodeError::SymbolAbsent));
                    continue;
                }
                let bits = code.symbol_bits(symbol).unwrap();
                expected_bits += usize::from(bits);
                code.write_symbol(symbol, &mut writer).unwrap();
                assert_eq!(writer.bits_written(), expected_bits);
            }
            assert_eq!(
                code.symbol_bits(counts.len()),
                Err(PrefixCodeError::SymbolOutOfRange)
            );
            assert_eq!(
                code.symbol_bits(usize::MAX),
                Err(PrefixCodeError::SymbolOutOfRange)
            );
        }
        assert_eq!(
            PrefixCode::from_counts(&[0, 9, 0]).unwrap().symbol_bits(1),
            Ok(0)
        );
    }

    #[test]
    fn construction_obeys_kraft_prefixes_and_limit_for_u64_counts() {
        for size in 1..=256 {
            assert_prefix_code(&vec![1; size]);
        }
        let mut state = 0x49af_62c0_31db_875eu64;
        for _ in 0..128 {
            let mut counts = [0; 19];
            for count in &mut counts {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *count = if state & 3 == 0 { 0 } else { state >> 32 };
            }
            counts[0] = counts[0].max(1);
            assert_prefix_code(&counts);
        }
        let mut fibonacci = [1; 40];
        for i in 2..fibonacci.len() {
            fibonacci[i] = fibonacci[i - 1] + fibonacci[i - 2];
        }
        assert!(create_huffman_tree(&fibonacci, 63).iter().any(|&d| d > 15));
        assert_prefix_code(&fibonacci);
        assert_prefix_code(&[u64::MAX; 256]);
        let mut extreme = [1; 256];
        extreme[0] = u64::MAX;
        assert_prefix_code(&extreme);
    }

    #[test]
    fn propagates_writer_limits() {
        let code = PrefixCode::from_counts(&[1, 1]).unwrap();
        let mut writer = BitWriter::new(0);
        assert_eq!(
            code.write_histogram(&mut writer),
            Err(PrefixCodeError::BitWriter(BitWriterError::SizeLimit))
        );
        assert_eq!(
            code.write_symbol(0, &mut writer),
            Err(PrefixCodeError::BitWriter(BitWriterError::SizeLimit))
        );
        assert_eq!(writer.bits_written(), 0);
    }
}
