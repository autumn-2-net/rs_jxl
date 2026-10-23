// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! b866b3d enc_fast_lossless.cc: hybrid integers and constrained prefix codes.

use crate::bit_writer::BitWriter;
use crate::{Error, Result};

pub(super) const RAW: usize = 19;
pub(super) const LZ: usize = 33;
const RLE_CACHE: usize = 32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) struct Lookup {
    pub depth: u8,
    pub lengths: [u8; 16],
    pub bits: [u8; 16],
    pub max_residual: u32,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Lookup {
    fn new(depth: u8, lengths: &[u8; RAW], bits: &[u16; RAW], numraw: usize) -> Self {
        let mut result = Self {
            depth,
            lengths: [0; 16],
            bits: [0; 16],
            max_residual: (1 << (numraw - 1)) - 1,
        };
        for i in 0..16 {
            let source = if depth > 14 && i >= 14 { 2 * i - 13 } else { i };
            result.lengths[i] = lengths[source];
            result.bits[i] = bits[source] as u8;
            debug_assert!(bits[source] <= 255);
        }
        // Source PrepareForSimd/CheckHuffmanBitsSIMD rely on these pairs.
        let pairs: &[usize] = if depth > 14 {
            &[13, 15, 17]
        } else if depth == 14 {
            &[15]
        } else {
            &[]
        };
        for &i in pairs {
            debug_assert_eq!((lengths[i], lengths[i + 1]), (8, 8));
            debug_assert_eq!(bits[i] & 128, 0);
            debug_assert_eq!(bits[i] | 128, bits[i + 1]);
        }
        result
    }
}

pub(super) fn uint000(value: u32) -> (usize, u8, u32) {
    if value == 0 {
        (0, 0, 0)
    } else {
        let n = 31 - value.leading_zeros();
        (n as usize + 1, n as u8, value - (1 << n))
    }
}

pub(super) fn uint400(value: u32) -> (usize, u8, u32) {
    if value < 16 {
        (value as usize, 0, 0)
    } else {
        let n = 31 - value.leading_zeros();
        (12 + n as usize, n as u8, value - (1 << n))
    }
}

// Source DP iteration and reverse traceback order determine equal-cost ties.
fn lengths(freq: &[u64], min: &[u8], max: &[u8]) -> Result<Vec<u8>> {
    let symbols: Vec<_> = (0..freq.len()).filter(|&i| freq[i] != 0).collect();
    if symbols.is_empty() || freq.len() > LZ || min.len() < freq.len() || max.len() < freq.len() {
        return Err(Error::Encoding("invalid FJXL code frequencies".into()));
    }
    let precision = symbols.iter().map(|&i| max[i]).max().unwrap();
    if precision >= 15 || symbols.iter().any(|&i| min[i].max(1) > max[i]) {
        return Err(Error::Encoding("invalid FJXL code constraints".into()));
    }
    let sum = freq
        .iter()
        .try_fold(0u64, |a, &b| a.checked_add(b))
        .ok_or(Error::Limit("FJXL frequencies"))?;
    let infinity = sum
        .checked_mul(u64::from(precision) + 1)
        .ok_or(Error::Limit("FJXL prefix costs"))?;
    let size = 1usize << precision;
    let stride = size + 1;
    let mut dp = vec![infinity; (symbols.len() + 1) * stride];
    dp[0] = 0;
    for (sym, &i) in symbols.iter().enumerate() {
        for bits in min[i].max(1)..=max[i] {
            let delta = 1 << (precision - bits);
            for off in 0..=size - delta {
                let cost = dp[sym * stride + off].saturating_add(freq[i] * u64::from(bits));
                let target = &mut dp[(sym + 1) * stride + off + delta];
                *target = (*target).min(cost);
            }
        }
    }
    if dp[symbols.len() * stride + size] == infinity {
        return Err(Error::Encoding("infeasible FJXL prefix code".into()));
    }
    let mut result = vec![0; freq.len()];
    let mut off = size;
    for (sym, &i) in symbols.iter().enumerate().rev() {
        for bits in min[i].max(1)..=max[i] {
            let delta = 1 << (precision - bits);
            if delta <= off
                && dp[(sym + 1) * stride + off]
                    == dp[sym * stride + off - delta].saturating_add(freq[i] * u64::from(bits))
            {
                result[i] = bits;
                off -= delta;
                break;
            }
        }
    }
    debug_assert_eq!(off, 0);
    Ok(result)
}

fn canonical(lengths: &[u8]) -> Vec<u16> {
    let mut counts = [0u16; 16];
    for &len in lengths {
        if len != 0 {
            counts[len as usize] += 1;
        }
    }
    let mut next = [0u16; 16];
    for i in 1..16 {
        next[i] = (next[i - 1] + counts[i - 1]) << 1;
    }
    lengths
        .iter()
        .map(|&len| {
            if len == 0 {
                return 0;
            }
            let code = &mut next[len as usize];
            let reversed = code.reverse_bits() >> (16 - len);
            *code += 1;
            reversed
        })
        .collect()
}

pub(super) struct PrefixCode {
    raw_len: [u8; RAW],
    raw_bits: [u16; RAW],
    lz_len: [u8; LZ],
    lz_bits: [u16; LZ],
    rle_len: [u8; RLE_CACHE],
    rle_bits: [u64; RLE_CACHE],
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub lookup: Lookup,
}

impl PrefixCode {
    pub(super) fn new(raw: &[u64; RAW], lz: &[u64; LZ], depth: u8) -> Result<Self> {
        let numraw = raw.iter().rposition(|&n| n != 0).map_or(0, |i| i + 1);
        let mut freq = raw[..numraw].to_vec();
        freq.push(
            lz.iter()
                .try_fold(0u64, |a, &b| a.checked_add(b))
                .ok_or(Error::Limit("FJXL frequencies"))?,
        );
        let (min, max): (Vec<u8>, Vec<u8>) = match depth {
            0..=8 => (vec![0; 12], [vec![7; 11], vec![10]].concat()),
            9..=13 => (vec![0; 17], [vec![8; 16], vec![10]].concat()),
            14 => (
                [vec![0; 15], vec![8, 8, 7]].concat(),
                [vec![7; 15], vec![8, 8, 10]].concat(),
            ),
            _ => (
                [vec![0; 13], vec![8; 6], vec![7]].concat(),
                [vec![7; 13], vec![8; 6], vec![10]].concat(),
            ),
        };
        let first = lengths(&freq, &min, &max)?;
        let second = lengths(lz, &[0; LZ], &[15 - first[numraw]; LZ])?;
        let mut code = Self {
            raw_len: [0; RAW],
            raw_bits: [0; RAW],
            lz_len: [0; LZ],
            lz_bits: [0; LZ],
            rle_len: [0; RLE_CACHE],
            rle_bits: [0; RLE_CACHE],
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            lookup: Lookup {
                depth,
                lengths: [0; 16],
                bits: [0; 16],
                max_residual: 0,
            },
        };
        code.raw_len[..numraw].copy_from_slice(&first[..numraw]);
        for (out, &len) in code.lz_len.iter_mut().zip(&second) {
            *out = if len == 0 { 0 } else { len + first[numraw] };
        }
        let codes = canonical(&[&code.raw_len[..numraw], &code.lz_len].concat());
        code.raw_bits[..numraw].copy_from_slice(&codes[..numraw]);
        code.lz_bits.copy_from_slice(&codes[numraw..]);
        for count in 0..RLE_CACHE {
            let (token, extra, bits) = uint400(count as u32);
            code.rle_len[count] = code.raw_len[0] + code.lz_len[token] + extra;
            code.rle_bits[count] = (((u64::from(bits) << code.lz_len[token])
                | u64::from(code.lz_bits[token]))
                << code.raw_len[0])
                | u64::from(code.raw_bits[0]);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            code.lookup = Lookup::new(depth, &code.raw_len, &code.raw_bits, numraw);
        }
        Ok(code)
    }

    pub(super) fn write(&self, writer: &mut BitWriter) -> Result<()> {
        let mut freq = [0u64; 18];
        freq[17] = 3 + 2 * (LZ as u64 - 1);
        for &len in self.raw_len.iter().chain(&self.lz_len) {
            freq[len as usize] += 1;
        }
        let len = lengths(&freq, &[0; 18], &[5; 18])?;
        let bits = canonical(&len);
        let order = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let last = order.iter().rposition(|&i| len[i] != 0).unwrap();
        writer.write(2, 0)?;
        for &i in &order[..=last] {
            let v = len[i] as usize;
            writer.write([2, 4, 3, 2, 2, 4][v], [0, 7, 3, 2, 1, 15][v])?;
        }
        for &v in &self.raw_len {
            writer.write(len[v as usize], u64::from(bits[v as usize]))?;
        }
        for extra in [2, 0, 2] {
            writer.write(len[17], u64::from(bits[17]))?;
            writer.write(3, extra)?;
        }
        let last = self.lz_len.iter().rposition(|&n| n != 0).unwrap();
        for &v in &self.lz_len[..=last] {
            writer.write(len[v as usize], u64::from(bits[v as usize]))?;
        }
        Ok(())
    }

    pub(super) fn literal(&self, value: u32, writer: &mut BitWriter) -> Result<()> {
        let (token, extra, bits) = uint000(value);
        if token >= RAW || self.raw_len[token] == 0 {
            return Err(Error::Encoding(
                "FJXL residual outside prefix alphabet".into(),
            ));
        }
        writer.write(
            self.raw_len[token] + extra,
            (u64::from(bits) << self.raw_len[token]) | u64::from(self.raw_bits[token]),
        )?;
        Ok(())
    }

    pub(super) fn run(&self, count: usize, writer: &mut BitWriter) -> Result<()> {
        let count = count
            .checked_sub(8)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| Error::Encoding("invalid FJXL run length".into()))?;
        if count < RLE_CACHE as u32 {
            writer.write(self.rle_len[count as usize], self.rle_bits[count as usize])?;
            return Ok(());
        }
        let (token, extra, bits) = uint400(count);
        if token >= LZ || self.lz_len[token] == 0 {
            return Err(Error::Encoding("FJXL run outside prefix alphabet".into()));
        }
        let length = self.raw_len[0] + self.lz_len[token] + extra;
        let packed = (((u64::from(bits) << self.lz_len[token]) | u64::from(self.lz_bits[token]))
            << self.raw_len[0])
            | u64::from(self.raw_bits[0]);
        writer.write(length, packed)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_integer_boundaries() {
        for value in 0..=131_071 {
            let (token, extra, bits) = uint000(value);
            assert_eq!(if token == 0 { 0 } else { (1 << extra) + bits }, value);
            let (token, extra, bits) = uint400(value);
            assert_eq!(
                if token < 16 {
                    token as u32
                } else {
                    (1 << extra) + bits
                },
                value
            );
        }
    }

    #[test]
    fn dynamic_program_tie_order_and_zero_symbols() {
        assert_eq!(lengths(&[1, 1, 1], &[0; 3], &[3; 3]).unwrap(), [2, 2, 1]);
        assert_eq!(
            lengths(&[1, 0, 1, 1], &[0; 4], &[3; 4]).unwrap(),
            [2, 0, 2, 1]
        );
        assert_eq!(canonical(&[2, 2, 1]), [1, 3, 0]);
    }

    fn code(depth: u8, distribution: usize) -> PrefixCode {
        let symbols = match depth {
            1..=13 => usize::from(depth) + 3,
            14 => 17,
            _ => 19,
        };
        let mut raw = [0; RAW];
        for (i, n) in raw[..symbols].iter_mut().enumerate() {
            *n = 1 + (i * 37 % 101) as u64;
        }
        if distribution == 1 {
            raw[0] = 100_000;
        }
        if distribution == 2 {
            raw[symbols - 1] = 100_000;
        }
        let mut lz = [0; LZ];
        for (i, n) in lz[..28].iter_mut().enumerate() {
            *n = 1 + (i * 17 % 53) as u64;
        }
        PrefixCode::new(&raw, &lz, depth).unwrap()
    }

    // Independent bit-by-bit assembly: no production HybridUint helper,
    // literal writer, RLE writer, native packing or shared BitWriter.
    fn append_bits(bits: &mut Vec<bool>, length: u8, word: u64) {
        assert!(length <= 64);
        if length < 64 {
            assert_eq!(word >> length, 0);
        }
        bits.extend((0..length).map(|i| word & (1u64 << i) != 0));
    }

    fn literal_bits(code: &PrefixCode, value: u32, bits: &mut Vec<bool>) {
        let token = (u32::BITS - value.leading_zeros()) as usize;
        append_bits(bits, code.raw_len[token], u64::from(code.raw_bits[token]));
        if token != 0 {
            append_bits(
                bits,
                (token - 1) as u8,
                u64::from(value - (1 << (token - 1))),
            );
        }
    }

    fn check_bits(mut actual: BitWriter, expected: &[bool]) {
        assert_eq!(actual.bits_written(), expected.len());
        actual.zero_pad_to_byte().unwrap();
        let bytes = actual.bytes().unwrap();
        for (i, &bit) in expected.iter().enumerate() {
            assert_eq!(bytes[i / 8] >> (i % 8) & 1 != 0, bit, "bit {i}");
        }
        for i in expected.len()..bytes.len() * 8 {
            assert_eq!(bytes[i / 8] >> (i % 8) & 1, 0);
        }
    }

    #[test]
    fn native_huffman_packing_matches_independent_bits_all_profiles_tails_offsets() {
        use crate::fast_lossless::{Backend, simd::Kernel};
        for depth in [1, 8, 9, 13, 14, 15, 16] {
            for distribution in 0..3 {
                let code = code(depth, distribution);
                let highest = code.raw_len.iter().rposition(|&n| n != 0).unwrap();
                let max = (1u32 << highest) - 1;
                let mut boundaries = vec![0, 1];
                for bit in 1..highest {
                    boundaries.extend([(1 << bit) - 1, 1 << bit, ((1 << bit) + 1).min(max)]);
                }
                boundaries.push(max);
                for offset in 0..8 {
                    let input: [u32; 16] =
                        std::array::from_fn(|i| boundaries[(i + offset * 11) % boundaries.len()]);
                    for end in 0..=16 {
                        for skip in 0..=end {
                            let values = &input[skip..end];
                            let mut expected = Vec::new();
                            append_bits(&mut expected, offset as u8, (1 << offset) - 1);
                            for &value in values {
                                literal_bits(&code, value, &mut expected);
                            }
                            for backend in [Backend::Scalar, Backend::Auto] {
                                let mut writer = BitWriter::new(128);
                                writer.write(offset as u8, (1 << offset) - 1).unwrap();
                                Kernel::choose(backend)
                                    .literals(&code, values, &mut writer)
                                    .unwrap();
                                check_bits(writer, &expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn short_rle_cache_and_long_runs_match_source_token_bits() {
        for depth in [8, 13, 14, 16] {
            let code = code(depth, 1);
            for count in (8..=80).chain([127, 128, 255, 256, 511, 512, 1024, 32768, 65536]) {
                let value = count - 8;
                let (token, extra, payload) = if value < 16 {
                    (value, 0, 0)
                } else {
                    let highest = 31 - (value as u32).leading_zeros();
                    (12 + highest as usize, highest as u8, value - (1 << highest))
                };
                for offset in 0..8 {
                    let mut expected = Vec::new();
                    append_bits(&mut expected, offset, (1 << offset) - 1);
                    literal_bits(&code, 0, &mut expected);
                    append_bits(
                        &mut expected,
                        code.lz_len[token],
                        u64::from(code.lz_bits[token]),
                    );
                    append_bits(&mut expected, extra, payload as u64);
                    let mut writer = BitWriter::new(32);
                    writer.write(offset, (1 << offset) - 1).unwrap();
                    code.run(count, &mut writer).unwrap();
                    check_bits(writer, &expected);
                }
            }
            assert!(code.run(7, &mut BitWriter::new(32)).is_err());
        }
    }

    #[test]
    fn packed_entropy_remains_bounded_and_rejects_out_of_profile_values() {
        use crate::fast_lossless::{Backend, simd::Kernel};
        let code = code(8, 1);
        for backend in [Backend::Scalar, Backend::Auto] {
            let kernel = Kernel::choose(backend);
            let mut writer = BitWriter::new(0);
            assert!(kernel.literals(&code, &[1023; 16], &mut writer).is_err());
            let mut writer = BitWriter::new(128);
            assert!(kernel.literals(&code, &[1024], &mut writer).is_err());
            assert_eq!(writer.bits_written(), 0);
            assert!(kernel.literals(&code, &[u32::MAX], &mut writer).is_err());
            assert_eq!(writer.bits_written(), 0);
        }
        assert!(code.run(8, &mut BitWriter::new(0)).is_err());
    }

    #[test]
    fn encoder_sink_native_literals_and_cached_runs_match_scalar_same_chunks() {
        use crate::fast_lossless::{
            Backend,
            pixels::{Encoder, Processor},
            simd::Kernel,
        };
        for depth in [8, 13, 14, 16] {
            let code = code(depth, 1);
            let mut scalar = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(16384),
                kernel: Kernel::choose(Backend::Scalar),
            });
            let mut native = Processor::new(Encoder {
                code: &code,
                writer: BitWriter::new(16384),
                kernel: Kernel::choose(Backend::Auto),
            });
            for zeros in [7, 8, 16, 31, 39, 40, 128, 256] {
                let mut row = vec![0; zeros];
                row.extend([3, 0, 1023, 2, 0, 0, 1]);
                row.extend([0; 64]);
                // Hold source chunk boundaries fixed to isolate entropy SIMD.
                scalar.row(&row, 16).unwrap();
                native.row(&row, 16).unwrap();
            }
            scalar.finish().unwrap();
            native.finish().unwrap();
            assert_eq!(
                native.sink.writer.bits_written(),
                scalar.sink.writer.bits_written()
            );
            scalar.sink.writer.zero_pad_to_byte().unwrap();
            native.sink.writer.zero_pad_to_byte().unwrap();
            assert_eq!(
                native.sink.writer.bytes().unwrap(),
                scalar.sink.writer.bytes().unwrap()
            );
        }
    }
}
