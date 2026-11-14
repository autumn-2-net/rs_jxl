// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Direct port of the ANS branch of BuildAndStoreANSEncodingData and WriteTokens
//! (lib/jxl/enc_ans.cc), ANSCoder (enc_ans.h), and InitAliasTable/Lookup
//! (ans_common.{cc,h}), baseline b866b3defec13494bfbaf705db93581cda118e37.
//!
//! Context mapping, HybridUint/LZ77 tokenization, prefix selection, effort policy,
//! and stream headers belong to the caller. No external codec is used. The source
//! reciprocal multiply, LIFO alias construction, and reversed 56-bit emission
//! chunks are retained. Profiling, final-plan replay, and prefetch are omitted.

use crate::bit_writer::BitWriter;
use crate::{Error, Result};

pub use super::ans_histogram::AnsHistogramStrategy;
use super::ans_histogram::{LOG_TABLE_SIZE, MAX_ALPHABET_SIZE, NormalizedHistogram, TABLE_SIZE};

const INITIAL_STATE: u32 = 0x13 << 16;
const RECIPROCAL_PRECISION: u32 = 32 + LOG_TABLE_SIZE as u32;

/// A HybridUint-encoded token; context is already a histogram index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnsToken {
    pub context: usize,
    pub symbol: usize,
    pub nbits: u8,
    pub bits: u32,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct AliasEntry {
    cutoff: u8,
    right_value: u8,
    freq0: u16,
    offsets1: u16,
    freq1_xor_freq0: u16,
}

impl AliasEntry {
    fn lookup(self, index: usize, position: usize) -> (usize, usize, u16) {
        if position >= usize::from(self.cutoff) {
            (
                usize::from(self.right_value),
                position + usize::from(self.offsets1),
                self.freq0 ^ self.freq1_xor_freq0,
            )
        } else {
            (index, position, self.freq0)
        }
    }
}

fn init_alias_table(counts: &[u16], log_alpha_size: u8) -> Result<Vec<AliasEntry>> {
    let table_size = 1usize << log_alpha_size;
    let entry_size = TABLE_SIZE >> log_alpha_size;
    if counts.len() > table_size
        || counts.iter().map(|&n| usize::from(n)).sum::<usize>() != TABLE_SIZE
    {
        return Err(Error::Encoding(
            "Invalid normalized ANS distribution".into(),
        ));
    }
    let mut table = vec![AliasEntry::default(); table_size];
    if let Some(symbol) = counts.iter().position(|&n| usize::from(n) == TABLE_SIZE) {
        for (i, entry) in table.iter_mut().enumerate() {
            entry.right_value = symbol as u8;
            entry.offsets1 = (entry_size * i) as u16;
            entry.freq1_xor_freq0 = TABLE_SIZE as u16;
        }
        return Ok(table);
    }
    let mut underfull = Vec::with_capacity(table_size);
    let mut overfull = Vec::with_capacity(table_size);
    let mut cutoffs = [0usize; MAX_ALPHABET_SIZE];
    for (i, cutoff) in cutoffs[..table_size].iter_mut().enumerate() {
        *cutoff = usize::from(counts.get(i).copied().unwrap_or(0));
        if *cutoff > entry_size {
            overfull.push(i);
        } else if *cutoff < entry_size {
            underfull.push(i);
        }
    }
    while let Some(over) = overfull.pop() {
        let under = underfull
            .pop()
            .ok_or_else(|| Error::Encoding("Unbalanced ANS alias table".into()))?;
        cutoffs[over] -= entry_size - cutoffs[under];
        table[under].right_value = over as u8;
        table[under].offsets1 = cutoffs[over] as u16;
        if cutoffs[over] < entry_size {
            underfull.push(over);
        } else if cutoffs[over] > entry_size {
            overfull.push(over);
        }
    }
    for (i, entry) in table.iter_mut().enumerate() {
        if cutoffs[i] == entry_size {
            entry.right_value = i as u8;
            entry.offsets1 = 0;
            entry.cutoff = 0;
        } else {
            entry.offsets1 = entry
                .offsets1
                .checked_sub(cutoffs[i] as u16)
                .ok_or_else(|| Error::Encoding("Invalid ANS alias offset".into()))?;
            entry.cutoff = cutoffs[i] as u8;
        }
        entry.freq0 = counts.get(i).copied().unwrap_or(0);
        let freq1 = counts
            .get(usize::from(entry.right_value))
            .copied()
            .unwrap_or(0);
        entry.freq1_xor_freq0 = freq1 ^ entry.freq0;
    }
    Ok(table)
}

#[derive(Clone, Copy, Debug)]
struct SymbolInfo {
    freq: u16,
    reverse_start: u16,
    reciprocal: u64,
}

/// Normalized histogram and the source alias table's inverse encoding map.
///
/// The inverse maps are stored in one 4096-entry allocation, rather than one
/// allocation per symbol; their offsets/order are identical to the source.
#[derive(Clone, Debug)]
pub struct AnsCode {
    histogram: NormalizedHistogram,
    symbols: Vec<SymbolInfo>,
    reverse_map: Vec<u16>,
    log_alpha_size: u8,
}

impl AnsCode {
    /// Uses the source's precise strategy (all twelve precision trials plus flat).
    /// The checked sum of weights must not exceed i32::MAX. Empty/all-zero input
    /// produces the standard degenerate symbol-zero histogram.
    pub fn from_counts(counts: &[u64], log_alpha_size: u8) -> Result<Self> {
        Self::from_counts_with_strategy(counts, log_alpha_size, AnsHistogramStrategy::Precise)
    }

    pub fn from_counts_with_strategy(
        counts: &[u64],
        log_alpha_size: u8,
        strategy: AnsHistogramStrategy,
    ) -> Result<Self> {
        if !(5..=8).contains(&log_alpha_size) {
            return Err(Error::InvalidInput(
                "ANS alphabet logarithm must be in 5..=8".into(),
            ));
        }
        if counts.len() > MAX_ALPHABET_SIZE
            || counts
                .iter()
                .rposition(|&n| n != 0)
                .is_some_and(|n| n >= 1 << log_alpha_size)
        {
            return Err(Error::InvalidInput(
                "ANS histogram exceeds its alphabet".into(),
            ));
        }
        let histogram = NormalizedHistogram::compute(counts, strategy)?;
        let table = init_alias_table(&histogram.counts, log_alpha_size)?;
        let mut symbols = Vec::with_capacity(histogram.counts.len());
        let mut start = 0usize;
        for &freq in &histogram.counts {
            symbols.push(SymbolInfo {
                freq,
                reverse_start: start as u16,
                reciprocal: if freq == 0 {
                    1
                } else {
                    (1u64 << RECIPROCAL_PRECISION).div_ceil(u64::from(freq))
                },
            });
            start += usize::from(freq);
        }
        let mut reverse_map = vec![0; TABLE_SIZE];
        let log_entry_size = LOG_TABLE_SIZE - log_alpha_size;
        let entry_mask = (1usize << log_entry_size) - 1;
        for state in 0..TABLE_SIZE {
            let index = state >> log_entry_size;
            let (symbol, offset, _) = table[index].lookup(index, state & entry_mask);
            let info = symbols
                .get(symbol)
                .ok_or_else(|| Error::Encoding("ANS alias symbol outside histogram".into()))?;
            if offset >= usize::from(info.freq) {
                return Err(Error::Encoding(
                    "ANS alias offset outside symbol frequency".into(),
                ));
            }
            reverse_map[usize::from(info.reverse_start) + offset] = state as u16;
        }
        Ok(Self {
            histogram,
            symbols,
            reverse_map,
            log_alpha_size,
        })
    }

    /// Writes only the histogram representation, without stream-level headers.
    pub fn write_histogram(&self, writer: &mut BitWriter) -> Result<()> {
        self.histogram.write(writer)
    }

    /// Cost under the actual selected distribution, including flat candidates.
    pub fn symbol_cost(&self, symbol: usize) -> Result<f64> {
        let info = self.symbol_info(symbol)?;
        Ok(12.0 - f64::from(info.freq).log2())
    }

    pub fn normalized_counts(&self) -> &[u16] {
        &self.histogram.counts
    }

    pub fn log_alpha_size(&self) -> u8 {
        self.log_alpha_size
    }

    /// Source f32 histogram-plus-data estimate; not the exact rANS stream length.
    pub fn estimated_cost(&self) -> f64 {
        f64::from(self.histogram.cost)
    }

    pub fn histogram_bits(&self) -> Result<usize> {
        self.histogram.histogram_bits()
    }

    fn symbol_info(&self, symbol: usize) -> Result<&SymbolInfo> {
        self.symbols
            .get(symbol)
            .filter(|info| info.freq != 0)
            .ok_or_else(|| Error::InvalidInput("ANS symbol has no encoding frequency".into()))
    }
}

struct AnsCoder {
    state: u32,
}

impl AnsCoder {
    fn put_symbol(&mut self, info: &SymbolInfo, reverse_map: &[u16]) -> (u8, u32) {
        let (nbits, bits) = if self.state >> (32 - LOG_TABLE_SIZE) >= u32::from(info.freq) {
            let bits = self.state & 0xffff;
            self.state >>= 16;
            (16, bits)
        } else {
            (0, 0)
        };
        let quotient = ((u64::from(self.state) * info.reciprocal) >> RECIPROCAL_PRECISION) as u32;
        let remainder = self.state - quotient * u32::from(info.freq);
        let offset = reverse_map[usize::from(info.reverse_start) + remainder as usize];
        self.state = (quotient << LOG_TABLE_SIZE) + u32::from(offset);
        (nbits, bits)
    }
}

struct ReverseBits {
    // Full 56-bit chunks; only the final, in-register tail needs a width.
    chunks: Vec<u64>,
    bits: u64,
    width: u8,
}

impl ReverseBits {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            bits: 0,
            width: 0,
        }
    }

    fn add(&mut self, width: u8, bits: u64) -> Result<()> {
        debug_assert!(width <= 48);
        debug_assert_eq!(bits >> width, 0);
        if width == 0 {
            return Ok(());
        }
        let combined_width = self.width + width;
        if combined_width > BitWriter::MAX_BITS_PER_CALL {
            let remainder_width = combined_width - BitWriter::MAX_BITS_PER_CALL;
            // Split the fused field so every stored chunk is exactly 56 bits.
            // Its low bits stay in the tail and are emitted before this chunk.
            let chunk = (self.bits << (width - remainder_width)) | (bits >> remainder_width);
            self.chunks
                .try_reserve(1)
                .map_err(|_| Error::Limit("ANS token chunks"))?;
            self.chunks.push(chunk);
            self.bits = bits & ((1u64 << remainder_width) - 1);
            self.width = remainder_width;
        } else {
            self.bits = (self.bits << width) | bits;
            self.width = combined_width;
        }
        Ok(())
    }

    fn write(self, writer: &mut BitWriter) -> Result<()> {
        writer.write(self.width, self.bits)?;
        for &bits in self.chunks.iter().rev() {
            writer.write(BitWriter::MAX_BITS_PER_CALL, bits)?;
        }
        Ok(())
    }
}

/// Emits one complete rANS state and token payload, even for an empty stream.
/// Extra bits follow each decoded symbol's possible 16-bit renormalization word.
/// Invalid tokens are rejected before modifying the writer; output-limit errors
/// may leave a partial stream, as with other BitWriter operations.
pub fn write_tokens(codes: &[AnsCode], tokens: &[AnsToken], writer: &mut BitWriter) -> Result<()> {
    write_mapped_tokens(codes, tokens.len(), |index| Ok(tokens[index]), writer)
}

/// Source WriteTokens maps original tokens while visiting them in reverse.
/// Keep only the reverse bit chunks, not a second array of expanded tokens.
/// Mapping/validation failures cannot modify the destination writer.
pub(crate) fn write_mapped_tokens(
    codes: &[AnsCode],
    token_count: usize,
    mut map: impl FnMut(usize) -> Result<AnsToken>,
    writer: &mut BitWriter,
) -> Result<()> {
    if let Some(first) = codes.first()
        && codes
            .iter()
            .any(|code| code.log_alpha_size != first.log_alpha_size)
    {
        return Err(Error::InvalidInput(
            "ANS codes in one stream need a shared alphabet logarithm".into(),
        ));
    }
    let mut out = ReverseBits::new();
    let mut ans = AnsCoder {
        state: INITIAL_STATE,
    };
    for index in (0..token_count).rev() {
        let token = map(index)?;
        let code = codes
            .get(token.context)
            .ok_or_else(|| Error::InvalidInput("ANS token context outside code table".into()))?;
        let info = code.symbol_info(token.symbol)?;
        if token.nbits > 32 || u64::from(token.bits) >> token.nbits != 0 {
            return Err(Error::InvalidInput(
                "ANS token extra bits do not fit their width".into(),
            ));
        }
        let (nbits, bits) = ans.put_symbol(info, &code.reverse_map);
        // Reverse emission places renormalization before the token's extra bits.
        out.add(
            token.nbits + nbits,
            (u64::from(token.bits) << nbits) | u64::from(bits),
        )?;
    }
    writer.write(32, u64::from(ans.state))?;
    out.write(writer)
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::ans_histogram::test_decode::Reader;
    use super::*;

    // Expanded reference table: physically move symbol/ordinal pairs between
    // buckets. This does not use AliasEntry, Lookup, or the encoder inverse map.
    pub(crate) fn decoder_table(counts: &[u16], log_alpha: u8) -> Vec<(usize, u32)> {
        if let Some(symbol) = counts.iter().position(|&n| n == 4096) {
            return (0..4096).map(|offset| (symbol, offset)).collect();
        }
        let buckets = 1usize << log_alpha;
        let width = 4096 / buckets;
        let mut entries: Vec<Vec<(usize, u32)>> = (0..buckets)
            .map(|symbol| {
                (0..u32::from(counts.get(symbol).copied().unwrap_or(0)))
                    .map(|offset| (symbol, offset))
                    .collect()
            })
            .collect();
        let mut small: Vec<usize> = (0..buckets).filter(|&i| entries[i].len() < width).collect();
        let mut large: Vec<usize> = (0..buckets).filter(|&i| entries[i].len() > width).collect();
        while let Some(donor) = large.pop() {
            let receiver = small.pop().unwrap();
            let split = entries[donor].len() - (width - entries[receiver].len());
            let tail = entries[donor].split_off(split);
            entries[receiver].extend(tail);
            match entries[donor].len().cmp(&width) {
                std::cmp::Ordering::Less => small.push(donor),
                std::cmp::Ordering::Greater => large.push(donor),
                std::cmp::Ordering::Equal => {}
            }
        }
        assert!(entries.iter().all(|entry| entry.len() == width));
        entries.into_iter().flatten().collect()
    }

    fn roundtrip(codes: &[AnsCode], tokens: &[AnsToken], initial_bits: u8) {
        let mut writer = BitWriter::new(tokens.len() * 8 + 8192);
        writer
            .write(initial_bits, (1u64 << initial_bits) - 1)
            .unwrap();
        for code in codes {
            code.write_histogram(&mut writer).unwrap();
        }
        write_tokens(codes, tokens, &mut writer).unwrap();
        let end = writer.bits_written();
        writer.zero_pad_to_byte().unwrap();
        let bytes = writer.into_bytes().unwrap();
        let mut reader = Reader {
            bytes: &bytes,
            position: usize::from(initial_bits),
        };
        let decoded_counts: Vec<Vec<u16>> = codes.iter().map(|_| reader.histogram()).collect();
        let tables: Vec<_> = codes
            .iter()
            .zip(&decoded_counts)
            .map(|(code, counts)| {
                assert_eq!(counts, code.normalized_counts());
                decoder_table(counts, code.log_alpha_size())
            })
            .collect();
        let mut state = reader.read(32);
        for expected in tokens {
            let (symbol, offset) = tables[expected.context][(state & 4095) as usize];
            assert_eq!(symbol, expected.symbol);
            state = u32::from(decoded_counts[expected.context][symbol]) * (state >> 12) + offset;
            if state < 65536 {
                state = (state << 16) | reader.read(16);
            }
            assert_eq!(reader.read(expected.nbits), expected.bits);
        }
        assert_eq!(state, 0x13_0000);
        assert_eq!(reader.position, end);
    }

    #[test]
    fn fused_reverse_chunks_match_bitwise_reference_at_every_boundary() {
        for pending in 0..=56u8 {
            for extra_width in 0..=32u8 {
                for renorm_width in [0, 16u8] {
                    let mut out = ReverseBits::new();
                    let mut fields = Vec::new();
                    for width in [pending.min(48), pending.saturating_sub(48)] {
                        let bits = 0x9753_a15c_03d7u64 & ((1u64 << width) - 1);
                        out.add(width, bits).unwrap();
                        fields.push((width, bits));
                    }
                    assert_eq!(out.width, pending);
                    let extra_bits = 0xc195_a73du64 & ((1u64 << extra_width) - 1);
                    let renorm_bits = 0x935bu64 & ((1u64 << renorm_width) - 1);
                    out.add(
                        extra_width + renorm_width,
                        (extra_bits << renorm_width) | renorm_bits,
                    )
                    .unwrap();
                    // Keep the unfused fields for an independent bit-at-a-time oracle.
                    fields.push((extra_width, extra_bits));
                    fields.push((renorm_width, renorm_bits));
                    for width in [48, 0, 1, 48, 8] {
                        let bits = 0x815c_03d7_a96bu64 & ((1u64 << width) - 1);
                        out.add(width, bits).unwrap();
                        fields.push((width, bits));
                    }
                    let payload_bits: usize = fields.iter().map(|&(n, _)| usize::from(n)).sum();
                    assert_eq!(out.chunks.len(), (payload_bits - 1) / 56);
                    assert_eq!(usize::from(out.width), (payload_bits - 1) % 56 + 1);
                    assert!(out.chunks.iter().all(|bits| bits >> 56 == 0));
                    // Across the boundary sweep, every extra/renorm width sees
                    // every destination bit offset, including an unaligned state.
                    let prefix = (pending + extra_width) % 8;
                    let total_bits = payload_bits + usize::from(prefix);
                    let mut expected = vec![0u8; total_bits.div_ceil(8)];
                    let mut position = 0;
                    for (width, bits) in std::iter::once((prefix, (1u64 << prefix) - 1))
                        .chain(fields.iter().rev().copied())
                    {
                        for bit in 0..width {
                            expected[position / 8] |= (((bits >> bit) & 1) as u8) << (position % 8);
                            position += 1;
                        }
                    }
                    let mut writer = BitWriter::new(expected.len());
                    writer.write(prefix, (1u64 << prefix) - 1).unwrap();
                    out.write(&mut writer).unwrap();
                    assert_eq!(writer.bits_written(), total_bits);
                    writer.zero_pad_to_byte().unwrap();
                    assert_eq!(writer.into_bytes().unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn reverse_chunks_allocate_only_when_payload_exceeds_one_chunk() {
        let mut out = ReverseBits::new();
        for _ in 0..4096 {
            out.add(0, 0).unwrap();
        }
        out.add(48, (1u64 << 48) - 1).unwrap();
        out.add(8, 255).unwrap();
        assert_eq!(out.chunks.capacity(), 0);
        assert_eq!(out.width, 56);
        out.add(1, 1).unwrap();
        assert_eq!(out.chunks, [(1u64 << 56) - 1]);
        assert_eq!(out.width, 1);
        assert_eq!(out.bits, 1);
    }

    #[test]
    fn fused_extra_widths_decode_with_and_without_renormalization() {
        let rare = AnsCode::from_counts_with_strategy(&[4095, 1], 8, AnsHistogramStrategy::Direct0)
            .unwrap();
        for nbits in 0..=32 {
            let tokens = [AnsToken {
                context: 0,
                symbol: 1,
                nbits,
                bits: (0xa15c_03d7u64 & ((1u64 << nbits) - 1)) as u32,
            }; 64];
            // Repeated frequency-one symbols include steps with and without
            // renormalization, regardless of the extra-bit width.
            for prefix in [0, 7] {
                roundtrip(std::slice::from_ref(&rare), &tokens, prefix);
            }
        }
    }

    #[test]
    fn mapped_failures_after_buffered_chunks_leave_destination_unchanged() {
        let code =
            AnsCode::from_counts_with_strategy(&[4095, 0, 1], 8, AnsHistogramStrategy::Direct0)
                .unwrap();
        let valid = AnsToken {
            context: 0,
            symbol: 2,
            nbits: 32,
            bits: 0xa15c_03d7,
        };
        for invalid in [
            Some(AnsToken {
                context: 1,
                ..valid
            }),
            Some(AnsToken { symbol: 1, ..valid }),
            Some(AnsToken {
                symbol: usize::MAX,
                ..valid
            }),
            Some(AnsToken { nbits: 33, ..valid }),
            Some(AnsToken {
                nbits: 255,
                ..valid
            }),
            Some(AnsToken {
                nbits: 31,
                bits: u32::MAX,
                ..valid
            }),
            None,
        ] {
            // No room even for the final ANS state: validation must still win.
            let mut writer = BitWriter::new(1);
            writer.write(3, 5).unwrap();
            let mut calls = 0;
            let result = write_mapped_tokens(
                std::slice::from_ref(&code),
                128,
                |index| {
                    assert_eq!(index, 127 - calls);
                    calls += 1;
                    if index == 0 {
                        invalid.ok_or_else(|| Error::InvalidInput("mapping failed".into()))
                    } else {
                        Ok(valid)
                    }
                },
                &mut writer,
            );
            assert!(matches!(result, Err(Error::InvalidInput(_))));
            assert_eq!(calls, 128);
            assert_eq!(writer.bits_written(), 3);
            writer.zero_pad_to_byte().unwrap();
            assert_eq!(writer.into_bytes().unwrap(), [5]);
        }
    }

    #[test]
    fn mapping_does_not_preallocate_from_token_count() {
        let mut writer = BitWriter::new(1);
        writer.write(8, 0xa5).unwrap();
        let mut calls = 0;
        let result = write_mapped_tokens(
            &[],
            usize::MAX,
            |index| {
                assert_eq!(index, usize::MAX - 1);
                calls += 1;
                Err(Error::InvalidInput("mapping failed".into()))
            },
            &mut writer,
        );
        assert!(matches!(result, Err(Error::InvalidInput(_))));
        assert_eq!(calls, 1);
        assert_eq!(writer.bits_written(), 8);
        assert_eq!(writer.into_bytes().unwrap(), [0xa5]);
    }

    #[test]
    fn packed_payload_obeys_exact_and_short_output_limits() {
        let code = AnsCode::from_counts_with_strategy(&[4095, 1], 8, AnsHistogramStrategy::Direct0)
            .unwrap();
        let tokens: Vec<_> = (0..132)
            .map(|i| {
                let nbits = (i % 33) as u8;
                AnsToken {
                    context: 0,
                    symbol: 1,
                    nbits,
                    bits: (0xc195_a73du64 & ((1u64 << nbits) - 1)) as u32,
                }
            })
            .collect();
        for prefix in 0..8 {
            let mut reference = BitWriter::new(2048);
            reference.write(prefix, (1u64 << prefix) - 1).unwrap();
            write_tokens(std::slice::from_ref(&code), &tokens, &mut reference).unwrap();
            let end = reference.bits_written();
            reference.zero_pad_to_byte().unwrap();
            let expected = reference.into_bytes().unwrap();
            for limit in [expected.len() - 1, expected.len()] {
                let mut writer = BitWriter::new(limit);
                writer.write(prefix, (1u64 << prefix) - 1).unwrap();
                let mut calls = 0;
                let result = write_mapped_tokens(
                    std::slice::from_ref(&code),
                    tokens.len(),
                    |index| {
                        assert_eq!(index, tokens.len() - 1 - calls);
                        calls += 1;
                        Ok(tokens[index])
                    },
                    &mut writer,
                );
                assert_eq!(calls, tokens.len());
                assert!(writer.bits_written() <= limit * 8);
                if limit == expected.len() {
                    result.unwrap();
                    assert_eq!(writer.bits_written(), end);
                    writer.zero_pad_to_byte().unwrap();
                    assert_eq!(writer.into_bytes().unwrap(), expected);
                } else {
                    assert!(result.is_err());
                }
            }
        }
    }

    #[test]
    fn rans_known_state_and_renormalization_vectors() {
        let uniform = AnsCode::from_counts(&[1; 256], 8).unwrap();
        let mut writer = BitWriter::new(64);
        write_tokens(
            &[uniform],
            &[AnsToken {
                context: 0,
                symbol: 42,
                nbits: 0,
                bits: 0,
            }],
            &mut writer,
        )
        .unwrap();
        assert_eq!(writer.into_bytes().unwrap(), [0xa0, 0x02, 0x00, 0x13]);

        let rare = AnsCode::from_counts_with_strategy(&[4095, 1], 8, AnsHistogramStrategy::Direct0)
            .unwrap();
        assert_eq!(rare.normalized_counts(), [4095, 1]);
        let tokens = [AnsToken {
            context: 0,
            symbol: 1,
            nbits: 0,
            bits: 0,
        }; 3];
        let mut writer = BitWriter::new(64);
        write_tokens(std::slice::from_ref(&rare), &tokens, &mut writer).unwrap();
        assert_eq!(
            writer.into_bytes().unwrap(),
            [0x10, 0x10, 0x30, 0x01, 0x10, 0x00, 0x00, 0x00]
        );
        roundtrip(&[rare], &tokens, 0);

        let singleton = AnsCode::from_counts(&[0, 0, 1], 8).unwrap();
        let tokens = [
            AnsToken {
                context: 0,
                symbol: 2,
                nbits: 3,
                bits: 5,
            },
            AnsToken {
                context: 0,
                symbol: 2,
                nbits: 5,
                bits: 19,
            },
        ];
        let mut writer = BitWriter::new(64);
        write_tokens(&[singleton], &tokens, &mut writer).unwrap();
        assert_eq!(writer.into_bytes().unwrap(), [0x00, 0x00, 0x13, 0x00, 0x9d]);
    }

    #[test]
    fn alias_mapping_is_a_bijection_and_matches_expanded_decoder() {
        assert_eq!(std::mem::size_of::<AliasEntry>(), 8);
        for log_alpha in 5..=8 {
            let size = 1usize << log_alpha;
            for strategy in [
                AnsHistogramStrategy::Precise,
                AnsHistogramStrategy::Direct0,
                AnsHistogramStrategy::Direct6,
            ] {
                for shape in 0..4 {
                    let weights: Vec<u64> = (0..size)
                        .map(|i| match shape {
                            0 => 1,
                            1 => {
                                if i == size - 1 {
                                    100_000
                                } else {
                                    1
                                }
                            }
                            2 => {
                                if i % 9 == 0 {
                                    (i + 17) as u64
                                } else {
                                    0
                                }
                            }
                            _ => ((i * i * 503 + 711) % 1559) as u64,
                        })
                        .collect();
                    let code =
                        AnsCode::from_counts_with_strategy(&weights, log_alpha, strategy).unwrap();
                    let expected = decoder_table(code.normalized_counts(), log_alpha);
                    let alias = init_alias_table(code.normalized_counts(), log_alpha).unwrap();
                    let width = 4096 >> log_alpha;
                    let mut seen = [false; 4096];
                    for (slot, &(symbol, offset)) in expected.iter().enumerate() {
                        let info = &code.symbols[symbol];
                        assert_eq!(
                            alias[slot / width].lookup(slot / width, slot % width),
                            (symbol, offset as usize, info.freq)
                        );
                        assert_eq!(
                            usize::from(
                                code.reverse_map[usize::from(info.reverse_start) + offset as usize]
                            ),
                            slot
                        );
                        let reverse_index = usize::from(info.reverse_start) + offset as usize;
                        assert!(!seen[reverse_index]);
                        seen[reverse_index] = true;
                    }
                    assert!(seen.into_iter().all(|n| n));
                }
            }
        }
    }

    #[test]
    fn empty_and_singleton_streams_preserve_signature() {
        let mut writer = BitWriter::new(4);
        write_tokens(&[], &[], &mut writer).unwrap();
        assert_eq!(writer.into_bytes().unwrap(), [0, 0, 0x13, 0]);
        for log_alpha in 5..=8 {
            let empty = AnsCode::from_counts(&[0; 256], log_alpha).unwrap();
            assert_eq!(empty.normalized_counts(), [4096]);
            roundtrip(std::slice::from_ref(&empty), &[], 5);
            roundtrip(
                &[empty],
                &[AnsToken {
                    context: 0,
                    symbol: 0,
                    nbits: 32,
                    bits: u32::MAX,
                }; 40],
                3,
            );
            for symbol in [0, (1 << log_alpha) - 1] {
                let mut weights = vec![0; symbol + 1];
                weights[symbol] = 1;
                let code = AnsCode::from_counts(&weights, log_alpha).unwrap();
                for (slot, &value) in code.reverse_map.iter().enumerate() {
                    assert_eq!(usize::from(value), slot);
                }
                let tokens: Vec<_> = (0..100)
                    .map(|i| AnsToken {
                        context: 0,
                        symbol,
                        nbits: (i % 33) as u8,
                        bits: (0xa15c_03d7u64 & ((1u64 << (i % 33)) - 1)) as u32,
                    })
                    .collect();
                roundtrip(&[code], &tokens, 7);
            }
        }
    }

    #[test]
    fn mixed_contexts_extra_bits_and_chunk_boundaries_decode() {
        for log_alpha in 5..=8 {
            let size = 1usize << log_alpha;
            let weights: Vec<u64> = (0..size)
                .map(|i| if i % 5 == 0 { (i * i + 11) as u64 } else { 0 })
                .collect();
            let codes = vec![
                AnsCode::from_counts(&vec![1; size], log_alpha).unwrap(),
                AnsCode::from_counts_with_strategy(
                    &weights,
                    log_alpha,
                    AnsHistogramStrategy::Direct6,
                )
                .unwrap(),
                AnsCode::from_counts(&[4095, 1], log_alpha).unwrap(),
            ];
            let mut random = 0x719e_006b_a18f_3105u64;
            let mut tokens = Vec::new();
            for i in 0..3000 {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let context = i % 3;
                let symbol = match context {
                    0 => random as usize % size,
                    1 => (random as usize % ((size - 1) / 5 + 1)) * 5,
                    _ => 1,
                };
                let nbits = (i % 33) as u8;
                let bits = (random & ((1u64 << nbits) - 1)) as u32;
                tokens.push(AnsToken {
                    context,
                    symbol,
                    nbits,
                    bits,
                });
            }
            for prefix in 0..8 {
                roundtrip(&codes, &tokens, prefix);
            }
        }
    }

    #[test]
    fn reciprocal_multiply_matches_division_after_renormalization() {
        for freq in 1..=4096u32 {
            let reciprocal = (1u64 << 44).div_ceil(u64::from(freq));
            let threshold = u64::from(freq) << 20;
            for state in [
                0u64,
                1,
                65535,
                65536,
                0x13_0000,
                threshold - 1,
                threshold,
                u64::from(u32::MAX),
            ] {
                let normalized = if state >= threshold {
                    state >> 16
                } else {
                    state
                };
                assert_eq!(
                    (normalized * reciprocal) >> 44,
                    normalized / u64::from(freq)
                );
            }
        }
    }

    #[test]
    fn invalid_inputs_costs_and_output_limits() {
        for log_alpha in [0, 4, 9, 255] {
            assert!(AnsCode::from_counts(&[1], log_alpha).is_err());
        }
        assert!(AnsCode::from_counts(&[1; 33], 5).is_err());
        assert!(AnsCode::from_counts(&[0; 257], 8).is_err());
        assert!(AnsCode::from_counts(&[u64::MAX], 8).is_err());
        let code =
            AnsCode::from_counts_with_strategy(&[4095, 0, 1], 8, AnsHistogramStrategy::Direct0)
                .unwrap();
        assert!((code.symbol_cost(0).unwrap() - (12.0 - 4095f64.log2())).abs() < 1e-12);
        assert_eq!(code.symbol_cost(2).unwrap(), 12.0);
        assert!(code.symbol_cost(1).is_err());
        assert!(code.symbol_cost(3).is_err());
        let uniform = AnsCode::from_counts(&[1; 32], 5).unwrap();
        assert_eq!(uniform.symbol_cost(0).unwrap(), 5.0);
        let singleton = AnsCode::from_counts(&[1], 8).unwrap();
        assert_eq!(singleton.symbol_cost(0).unwrap(), 0.0);

        let valid = AnsToken {
            context: 0,
            symbol: 0,
            nbits: 0,
            bits: 0,
        };
        for invalid in [
            AnsToken {
                context: 1,
                ..valid
            },
            AnsToken { symbol: 1, ..valid },
            AnsToken {
                symbol: usize::MAX,
                ..valid
            },
            AnsToken { nbits: 33, ..valid },
            AnsToken { bits: 1, ..valid },
            AnsToken {
                nbits: 4,
                bits: 16,
                ..valid
            },
        ] {
            let mut writer = BitWriter::new(64);
            writer.write(3, 5).unwrap();
            assert!(
                write_tokens(std::slice::from_ref(&code), &[valid, invalid], &mut writer).is_err()
            );
            assert_eq!(writer.bits_written(), 3);
        }
        assert!(write_tokens(&[uniform, singleton], &[], &mut BitWriter::new(64)).is_err());
        assert!(write_tokens(std::slice::from_ref(&code), &[], &mut BitWriter::new(3)).is_err());
        assert!(code.write_histogram(&mut BitWriter::new(0)).is_err());
    }
}
