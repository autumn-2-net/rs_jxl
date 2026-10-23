// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! b866b3d enc_fast_lossless.cc AVX2 TokenizeSIMD, HuffmanSIMD*, StoreSIMD*,
//! Bits16/32::Merge and StoreToWriter. No approximate integer logarithms.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::entropy::Lookup;
use crate::{Error, Result, bit_writer::BitWriter};

pub(super) struct Packed {
    lengths: [u64; 8],
    bits: [u64; 8],
    count: usize,
}

impl Packed {
    pub fn write(&self, writer: &mut BitWriter) -> Result<()> {
        for (&length, &bits) in self.lengths[..self.count].iter().zip(&self.bits) {
            debug_assert!(length <= 64);
            // Source WriteMultiple handles up to 64 bits. The shared bounded
            // writer permits 56, so split only the few wider merged words.
            if length > 56 {
                writer.write(56, bits & ((1u64 << 56) - 1))?;
                writer.write((length - 56) as u8, bits >> 56)?;
            } else {
                writer.write(length as u8, bits)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Bits {
    lengths: __m256i,
    bits: __m256i,
}

const POW2_LOW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0];
const POW2_HIGH: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, 128];

#[target_feature(enable = "avx2")]
unsafe fn lookup(table: &[u8; 16], index: __m256i) -> __m256i {
    // SAFETY: exactly sixteen initialized table bytes; caller enables AVX2.
    unsafe {
        _mm256_shuffle_epi8(
            _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().cast())),
            _mm256_or_si256(index, _mm256_set1_epi16(-256)),
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn tokenize16(residuals: __m256i) -> (__m256i, Bits) {
    const LUT: [[u8; 16]; 4] = [
        [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4],
        [0, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 9, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
        [
            0, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16,
        ],
    ];
    // SAFETY: AVX2 caller; lookup accesses only its fixed sixteen-byte table.
    unsafe {
        let mask = _mm256_set1_epi16(15);
        let t0 = lookup(&LUT[0], _mm256_and_si256(residuals, mask));
        let t1 = lookup(
            &LUT[1],
            _mm256_and_si256(_mm256_srli_epi16::<4>(residuals), mask),
        );
        let t2 = lookup(
            &LUT[2],
            _mm256_and_si256(_mm256_srli_epi16::<8>(residuals), mask),
        );
        let t3 = lookup(&LUT[3], _mm256_srli_epi16::<12>(residuals));
        let tokens = _mm256_max_epi16(_mm256_max_epi16(t0, t1), _mm256_max_epi16(t2, t3));
        let lengths = _mm256_subs_epu16(tokens, _mm256_set1_epi16(1));
        let power = _mm256_or_si256(
            lookup(&POW2_LOW, lengths),
            _mm256_slli_epi16::<8>(lookup(&POW2_HIGH, lengths)),
        );
        let bits = _mm256_subs_epu16(residuals, power);
        (tokens, Bits { lengths, bits })
    }
}

#[target_feature(enable = "avx2")]
fn tokenize32(residuals: __m256i) -> (__m256i, Bits) {
    // All supported residuals are <= 262143: conversion to f32 is exact.
    // Extracting the exponent is therefore exactly source floor(log2)+1.
    let floats = _mm256_castps_si256(_mm256_cvtepi32_ps(residuals));
    let tokens = _mm256_max_epi32(
        _mm256_setzero_si256(),
        _mm256_sub_epi32(_mm256_srli_epi32::<23>(floats), _mm256_set1_epi32(126)),
    );
    let one = _mm256_set1_epi32(1);
    let lengths = _mm256_sub_epi32(_mm256_max_epu32(tokens, one), one);
    let power = _mm256_sllv_epi32(one, lengths);
    let bits = _mm256_sub_epi32(_mm256_max_epu32(residuals, power), power);
    (tokens, Bits { lengths, bits })
}

#[target_feature(enable = "avx2")]
fn narrow(low: __m256i, high: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0b11011000>(_mm256_packus_epi32(low, high))
}

#[target_feature(enable = "avx2")]
fn interleave(high: __m256i, low: __m256i) -> [__m256i; 2] {
    let v02 = _mm256_unpacklo_epi16(low, high);
    let v13 = _mm256_unpackhi_epi16(low, high);
    [
        _mm256_permute2x128_si256::<0x20>(v02, v13),
        _mm256_permute2x128_si256::<0x31>(v02, v13),
    ]
}

#[target_feature(enable = "avx2")]
unsafe fn huffman(tokens: __m256i, table: &Lookup) -> Bits {
    let mut index = tokens;
    let mut high_bit = _mm256_setzero_si256();
    if table.depth == 14 {
        index = _mm256_min_epu16(tokens, _mm256_set1_epi16(15));
        high_bit = _mm256_and_si256(
            _mm256_cmpeq_epi16(tokens, _mm256_set1_epi16(16)),
            _mm256_set1_epi16(128),
        );
    } else if table.depth > 14 {
        let above = _mm256_cmpgt_epi16(tokens, _mm256_set1_epi16(12));
        index = _mm256_blendv_epi8(
            tokens,
            _mm256_srai_epi16::<1>(_mm256_add_epi16(tokens, _mm256_set1_epi16(13))),
            above,
        );
        let even = _mm256_cmpeq_epi16(tokens, _mm256_and_si256(tokens, _mm256_set1_epi16(-2)));
        high_bit = _mm256_and_si256(_mm256_and_si256(above, even), _mm256_set1_epi16(128));
    }
    // SAFETY: fixed tables and AVX2 caller. PrefixCode enforces paired-code
    // lengths/bits at depth 14 and >14 when constructing these lookup tables.
    unsafe {
        Bits {
            lengths: lookup(&table.lengths, index),
            bits: _mm256_or_si256(lookup(&table.bits, index), high_bit),
        }
    }
}

#[target_feature(enable = "avx2")]
fn clip16(input: Bits, count: usize) -> Bits {
    let index = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let mask = _mm256_cmpgt_epi16(_mm256_set1_epi16(count.min(16) as i16), index);
    Bits {
        lengths: _mm256_and_si256(input.lengths, mask),
        bits: _mm256_and_si256(input.bits, mask),
    }
}

#[target_feature(enable = "avx2")]
fn clip32(input: Bits, count: usize) -> Bits {
    let index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(count.min(8) as i32), index);
    Bits {
        lengths: _mm256_and_si256(input.lengths, mask),
        bits: _mm256_and_si256(input.bits, mask),
    }
}

#[target_feature(enable = "avx2")]
fn merge16(input: Bits) -> Bits {
    let low_lengths = _mm256_and_si256(input.lengths, _mm256_set1_epi32(65535));
    let high_bits = _mm256_srli_epi32::<16>(input.bits);
    Bits {
        lengths: _mm256_add_epi32(low_lengths, _mm256_srli_epi32::<16>(input.lengths)),
        bits: _mm256_or_si256(
            _mm256_and_si256(input.bits, _mm256_set1_epi32(65535)),
            _mm256_sllv_epi32(high_bits, low_lengths),
        ),
    }
}

#[target_feature(enable = "avx2")]
unsafe fn merge32_store(input: Bits, output: &mut Packed, offset: usize) {
    let mask = _mm256_set1_epi64x(0xffff_ffff);
    let low_lengths = _mm256_and_si256(input.lengths, mask);
    let lengths = _mm256_add_epi64(low_lengths, _mm256_srli_epi64::<32>(input.lengths));
    let bits = _mm256_or_si256(
        _mm256_and_si256(input.bits, mask),
        _mm256_sllv_epi64(_mm256_srli_epi64::<32>(input.bits), low_lengths),
    );
    debug_assert!(offset == 0 || offset == 4);
    // SAFETY: offsets 0/4 each cover exactly four elements of eight-element
    // output arrays. Stores are unaligned and the caller enables AVX2.
    unsafe {
        _mm256_storeu_si256(output.lengths.as_mut_ptr().add(offset).cast(), lengths);
        _mm256_storeu_si256(output.bits.as_mut_ptr().add(offset).cast(), bits);
    }
}

/// Caller must have detected AVX2. Source chunks have at most sixteen values.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn pack(residuals: &[u32], table: &Lookup) -> Result<Packed> {
    if residuals.len() > 16 {
        return Err(Error::Encoding(
            "FJXL SIMD chunk exceeds sixteen samples".into(),
        ));
    }
    let mut padded = [0u32; 16];
    padded[..residuals.len()].copy_from_slice(residuals);
    let mut output = Packed {
        lengths: [0; 8],
        bits: [0; 8],
        count: if table.depth <= 8 { 4 } else { 8 },
    };
    // SAFETY: all loads use the initialized local sixteen-element buffer or
    // fixed tables; output stores use offsets 0/4. Runtime dispatch checks AVX2.
    unsafe {
        let lo = _mm256_loadu_si256(padded.as_ptr().cast());
        let hi = _mm256_loadu_si256(padded.as_ptr().add(8).cast());
        let max = _mm256_set1_epi32(table.max_residual as i32);
        let valid = _mm256_and_si256(
            _mm256_cmpeq_epi32(_mm256_min_epu32(lo, max), lo),
            _mm256_cmpeq_epi32(_mm256_min_epu32(hi, max), hi),
        );
        if _mm256_movemask_epi8(valid) != -1 {
            return Err(Error::Encoding(
                "FJXL residual outside prefix alphabet".into(),
            ));
        }
        if table.depth <= 14 {
            let (tokens, raw) = tokenize16(narrow(lo, hi));
            let huff = huffman(tokens, table);
            if table.depth <= 8 {
                // Source Bits16::Interleave replaces variable 16-bit shifts by
                // multiplication. This profile constrains Huffman widths <8.
                let bits = _mm256_or_si256(
                    _mm256_mullo_epi16(raw.bits, lookup(&POW2_LOW, huff.lengths)),
                    huff.bits,
                );
                let lengths = _mm256_add_epi16(raw.lengths, huff.lengths);
                merge32_store(
                    merge16(clip16(Bits { lengths, bits }, residuals.len())),
                    &mut output,
                    0,
                );
            } else {
                let lengths = interleave(raw.lengths, huff.lengths);
                let bits = interleave(raw.bits, huff.bits);
                for i in 0..2 {
                    let count = (2 * residuals.len()).saturating_sub(i * 16);
                    merge32_store(
                        merge16(clip16(
                            Bits {
                                lengths: lengths[i],
                                bits: bits[i],
                            },
                            count,
                        )),
                        &mut output,
                        i * 4,
                    );
                }
            }
        } else {
            let (tokens_lo, raw_lo) = tokenize32(lo);
            let (tokens_hi, raw_hi) = tokenize32(hi);
            let huff = huffman(narrow(tokens_lo, tokens_hi), table);
            let huff_lengths = interleave(_mm256_setzero_si256(), huff.lengths);
            let huff_bits = interleave(_mm256_setzero_si256(), huff.bits);
            for (i, raw) in [raw_lo, raw_hi].into_iter().enumerate() {
                let bits =
                    _mm256_or_si256(_mm256_sllv_epi32(raw.bits, huff_lengths[i]), huff_bits[i]);
                let lengths = _mm256_add_epi32(raw.lengths, huff_lengths[i]);
                merge32_store(
                    clip32(
                        Bits { lengths, bits },
                        residuals.len().saturating_sub(i * 8),
                    ),
                    &mut output,
                    i * 4,
                );
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_full_u64_words_preserve_carry_at_every_initial_offset() {
        let packed = Packed {
            lengths: [64, 0, 57, 7, 32, 56, 1, 64],
            bits: [
                u64::MAX,
                0,
                (1 << 57) - 3,
                91,
                0x9182_3456,
                (1 << 56) - 9,
                1,
                0x8192_a3b4_c5d6_e7f8,
            ],
            count: 8,
        };
        for offset in 0..8 {
            let mut writer = BitWriter::new(128);
            writer.write(offset, (1 << offset) - 1).unwrap();
            packed.write(&mut writer).unwrap();
            let mut expected = vec![true; offset as usize];
            for (&length, &bits) in packed.lengths.iter().zip(&packed.bits) {
                expected.extend((0..length).map(|i| bits & (1 << i) != 0));
            }
            assert_eq!(writer.bits_written(), expected.len());
            writer.zero_pad_to_byte().unwrap();
            let bytes = writer.bytes().unwrap();
            for (i, bit) in expected.into_iter().enumerate() {
                assert_eq!(bytes[i / 8] >> (i % 8) & 1 != 0, bit);
            }
        }
    }

    #[test]
    fn native_tokenization_covers_every_supported_residual_exactly() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // SAFETY: runtime feature check above; helper uses fixed local arrays.
        unsafe {
            check_tokens();
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn check_tokens() {
        let mut input = [0u32; 16];
        let mut tokens = [0u16; 16];
        let mut lengths16 = [0u16; 16];
        let mut bits16 = [0u16; 16];
        let mut lengths32 = [0u32; 16];
        let mut bits32 = [0u32; 16];
        // SAFETY: every pointer accesses one complete initialized local vector;
        // the caller has detected AVX2. Values stay below 2^18.
        unsafe {
            for begin in (0..=262_143u32).step_by(16) {
                for (i, value) in input.iter_mut().enumerate() {
                    *value = begin + i as u32;
                }
                let lo = _mm256_loadu_si256(input.as_ptr().cast());
                let hi = _mm256_loadu_si256(input.as_ptr().add(8).cast());
                let (tlo, blo) = tokenize32(lo);
                let (thi, bhi) = tokenize32(hi);
                _mm256_storeu_si256(tokens.as_mut_ptr().cast(), narrow(tlo, thi));
                _mm256_storeu_si256(lengths32.as_mut_ptr().cast(), blo.lengths);
                _mm256_storeu_si256(lengths32.as_mut_ptr().add(8).cast(), bhi.lengths);
                _mm256_storeu_si256(bits32.as_mut_ptr().cast(), blo.bits);
                _mm256_storeu_si256(bits32.as_mut_ptr().add(8).cast(), bhi.bits);
                for i in 0..16 {
                    let token = u32::BITS - input[i].leading_zeros();
                    let length = token.saturating_sub(1);
                    assert_eq!(u32::from(tokens[i]), token);
                    assert_eq!(lengths32[i], length);
                    assert_eq!(bits32[i], input[i].saturating_sub(1 << length));
                }
                if begin <= 65535 {
                    let (tok, bits) = tokenize16(narrow(lo, hi));
                    _mm256_storeu_si256(tokens.as_mut_ptr().cast(), tok);
                    _mm256_storeu_si256(lengths16.as_mut_ptr().cast(), bits.lengths);
                    _mm256_storeu_si256(bits16.as_mut_ptr().cast(), bits.bits);
                    for i in 0..16 {
                        assert_eq!(u32::from(tokens[i]), u32::BITS - input[i].leading_zeros());
                        assert_eq!(u32::from(lengths16[i]), lengths32[i]);
                        assert_eq!(u32::from(bits16[i]), bits32[i]);
                    }
                }
            }
        }
    }
}
