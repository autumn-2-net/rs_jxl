//! Exercise the maintained ANS, HybridUint and BitReader source without compiling
//! the upstream test crate, whose unrelated binary fixtures are not bundled.
//! The byteorder adapter below uses only std; bit buffering and EOF handling
//! still come from the actual decoder, not a test double.

extern crate self as byteorder;

pub use jxl_decoder::error;

pub trait ByteOrder {
    fn read_u64(bytes: &[u8]) -> u64;
}

pub struct LittleEndian;

impl ByteOrder for LittleEndian {
    fn read_u64(bytes: &[u8]) -> u64 {
        u64::from_le_bytes(bytes[..8].try_into().unwrap())
    }
}

#[allow(dead_code)]
#[path = "../../../vendor/jxl-rs/jxl/src/entropy_coding/ans.rs"]
mod ans;
#[allow(dead_code)]
#[path = "../../../vendor/jxl-rs/jxl/src/bit_reader.rs"]
mod bit_reader;
#[path = "../../../vendor/jxl-rs/jxl/src/entropy_coding/hybrid_uint.rs"]
mod hybrid_uint;
#[allow(dead_code)]
#[path = "../../../vendor/jxl-rs/jxl/src/util/log2.rs"]
mod util;

use bit_reader::BitReader;
use error::Error;
use hybrid_uint::HybridUint;
use util::CeilLog2;

// Reference formulas retain the pre-optimization JPEG XL Project code.
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Governed by the BSD-style license in vendor/jxl-rs.
#[derive(Clone, Copy, Debug)]
struct ReferenceConfig {
    split: u32,
    msb: u32,
    lsb: u32,
}

impl ReferenceConfig {
    fn read(self, symbol: u32, br: &mut BitReader) -> u32 {
        let split_token = 1u32 << self.split;
        if symbol < split_token {
            return symbol;
        }
        let bits_in_token = self.lsb + self.msb;
        let nbits = self.split - bits_in_token + ((symbol - split_token) >> bits_in_token);
        let nbits = nbits & 31;
        let low = symbol & ((1 << self.lsb) - 1);
        let symbol_nolow = symbol >> self.lsb;
        let bits = br.read_optimistic(nbits as usize) as u32;
        let hi = (symbol_nolow & ((1 << self.msb) - 1)) | (1 << self.msb);
        (((hi << nbits) | bits) << self.lsb) | low
    }

    fn max_bits(self, symbol: u32) -> usize {
        let split_token = 1u32 << self.split;
        if symbol < split_token {
            return if symbol == 0 {
                0
            } else {
                symbol.ilog2() as usize + 1
            };
        }
        let bits_in_token = self.lsb + self.msb;
        let token_offset = (symbol - split_token) >> bits_in_token;
        let nbits = self.split - bits_in_token + token_offset;
        if nbits >= 32 {
            return 32;
        }
        (self.split + token_offset + 1) as usize
    }

    fn decode(log_alpha: usize, br: &mut BitReader) -> Result<Self, Error> {
        let split = br.read((log_alpha + 1).ceil_log2())? as u32;
        let (msb, lsb) = if split != log_alpha as u32 {
            let msb = br.read((split + 1).ceil_log2() as usize)? as u32;
            if msb > split {
                return Err(Error::InvalidUintConfig(split, msb, None));
            }
            let lsb = br.read((split - msb + 1).ceil_log2() as usize)? as u32;
            (msb, lsb)
        } else {
            (0, 0)
        };
        if lsb + msb > split {
            return Err(Error::InvalidUintConfig(split, msb, Some(lsb)));
        }
        Ok(Self { split, msb, lsb })
    }
}

fn overread(br: &BitReader) -> Option<usize> {
    match br.check_for_error() {
        Ok(()) => None,
        Err(Error::OutOfBounds(bits)) => Some(bits),
        Err(error) => panic!("unexpected reader error: {error:?}"),
    }
}

fn compare_readers(actual: &BitReader, expected: &BitReader) {
    assert_eq!(actual.total_bits_read(), expected.total_bits_read());
    assert_eq!(
        actual.total_bits_available(),
        expected.total_bits_available()
    );
    assert_eq!(overread(actual), overread(expected));
}

fn compare_read(
    reference: ReferenceConfig,
    config: &HybridUint,
    symbol: u32,
    bytes: &[u8],
    offset: usize,
) {
    let mut expected_br = BitReader::new(bytes);
    expected_br.skip_bits(offset).unwrap();
    let mut actual_br = expected_br.clone();
    assert_eq!(
        config.read(symbol, &mut actual_br),
        reference.read(symbol, &mut expected_br),
        "config={reference:?}, symbol={symbol}, offset={offset}, bytes={}",
        bytes.len(),
    );
    assert_eq!(
        config.max_bits_for_symbol(symbol),
        reference.max_bits(symbol)
    );
    compare_readers(&actual_br, &expected_br);
    assert_eq!(
        actual_br.read_optimistic(17),
        expected_br.read_optimistic(17)
    );
    compare_readers(&actual_br, &expected_br);
}

const PAYLOAD: [u8; 24] = [
    0xd3, 0, 0xff, 0x7f, 0x80, 0x12, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xff, 0, 0xaa, 0x55, 0x01,
    0x80, 0xfe, 0x5a, 0xa5, 0x39, 0x72, 0x18,
];

#[test]
fn compact_config_and_all_config_token_boundaries() {
    assert_eq!(std::mem::size_of::<HybridUint>(), 16);
    // The decoder's serialized exponents are at most 15. Also exercise the
    // full shift domain supported by the old test constructor, through 31.
    for split in 0..=31 {
        for msb in 0..=split {
            for lsb in 0..=split - msb {
                let reference = ReferenceConfig { split, msb, lsb };
                let config = HybridUint::new(split, msb, lsb);
                let split_token = 1u32 << split;
                assert_eq!(config.split_token(), split_token);
                assert_eq!(config.is_split_exponent_zero(), split == 0);
                assert_eq!(config.is_config_420(), (split, msb, lsb) == (4, 2, 0));
                if split <= 15 {
                    for symbol in (0..256).chain(split_token..split_token + 256) {
                        compare_read(reference, &config, symbol, &PAYLOAD, (symbol & 63) as usize);
                    }
                }
                for symbol in [0, split_token - 1, split_token, u32::MAX - 1, u32::MAX] {
                    compare_read(reference, &config, symbol, &PAYLOAD, (symbol & 63) as usize);
                }
                // Cross the 31/32 and 63/64 extra-bit boundaries, including
                // every retained token-bit pattern at either end of a bucket.
                for extra in 0..=65u64 {
                    let first = u64::from(split_token) + (extra << (msb + lsb));
                    for symbol in [first, first + (1u64 << (msb + lsb)) - 1] {
                        if let Ok(symbol) = u32::try_from(symbol) {
                            compare_read(
                                reference,
                                &config,
                                symbol,
                                &PAYLOAD,
                                (extra & 63) as usize,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn config_420_matches_reference_and_generic() {
    let reference = ReferenceConfig {
        split: 4,
        msb: 2,
        lsb: 0,
    };
    let config = HybridUint::new(4, 2, 0);
    for bytes in [&PAYLOAD, &[0u8; 24], &[0xffu8; 24]] {
        for symbol in (0..32768).chain([u32::MAX - 1, u32::MAX]) {
            let mut expected_br = BitReader::new(bytes);
            expected_br.skip_bits((symbol & 63) as usize).unwrap();
            let mut actual_br = expected_br.clone();
            let mut specialized_br = expected_br.clone();
            let expected = reference.read(symbol, &mut expected_br);
            assert_eq!(config.read(symbol, &mut actual_br), expected);
            assert_eq!(
                HybridUint::read_config_420(symbol, &mut specialized_br),
                expected
            );
            compare_readers(&actual_br, &expected_br);
            compare_readers(&specialized_br, &expected_br);
        }
    }
    let mut br = BitReader::new(&[0xff; 4]);
    assert_eq!(config.read(127, &mut br), u32::MAX);
    assert_eq!(br.total_bits_read(), 29);
}

#[test]
fn all_decoded_configs_preserve_truncation_and_eof() {
    for split in 0..=15 {
        for msb in 0..=split {
            for lsb in 0..=split - msb {
                let reference = ReferenceConfig { split, msb, lsb };
                let config = HybridUint::new(split, msb, lsb);
                let split_token = 1u32 << split;
                for len in 0..=9 {
                    for offset in 0..=7.min(len * 8) {
                        for symbol in [0, split_token - 1, split_token, split_token + 127, u32::MAX]
                        {
                            compare_read(reference, &config, symbol, &PAYLOAD[..len], offset);
                        }
                    }
                }
            }
        }
    }
    // Zero-bit modulo reads at EOF must remain successful, even when the
    // original unmasked extra-bit count was invalid (32 or 64).
    let config = HybridUint::new(4, 2, 0);
    for symbol in [0, 15, 136, 264] {
        let mut br = BitReader::new(&[]);
        config.read(symbol, &mut br);
        assert_eq!(br.total_bits_read(), 0);
        assert_eq!(overread(&br), None);
    }
    let mut br = BitReader::new(&[0xff]);
    config.read(127, &mut br);
    assert_eq!(br.total_bits_read(), 29);
    assert_eq!(overread(&br), Some(21));
}

fn compare_config_errors(actual: Error, expected: Error) {
    match (actual, expected) {
        (Error::OutOfBounds(actual), Error::OutOfBounds(expected)) => assert_eq!(actual, expected),
        (Error::InvalidUintConfig(a, b, c), Error::InvalidUintConfig(x, y, z)) => {
            assert_eq!((a, b, c), (x, y, z));
        }
        (actual, expected) => panic!("config errors differ: {actual:?} vs {expected:?}"),
    }
}

fn compare_config_decode(log_alpha: usize, bytes: &[u8], offset: usize) {
    let mut expected_br = BitReader::new(bytes);
    expected_br.skip_bits(offset).unwrap();
    let mut actual_br = expected_br.clone();
    let expected = ReferenceConfig::decode(log_alpha, &mut expected_br);
    let actual = HybridUint::decode(log_alpha, &mut actual_br);
    compare_readers(&actual_br, &expected_br);
    match (actual, expected) {
        (Ok(config), Ok(reference)) => {
            assert_eq!(config.split_token(), 1 << reference.split);
            assert_eq!(config.is_split_exponent_zero(), reference.split == 0);
            assert_eq!(
                config.is_config_420(),
                (reference.split, reference.msb, reference.lsb) == (4, 2, 0)
            );
            // Continue in the same readers: test the actual decoded config,
            // not just the constructor, through repeated optimistic EOF reads.
            for symbol in [0, 15, 16, 127, 136, 264, 32767, u32::MAX] {
                assert_eq!(
                    config.read(symbol, &mut actual_br),
                    reference.read(symbol, &mut expected_br)
                );
                compare_readers(&actual_br, &expected_br);
            }
        }
        (Err(actual), Err(expected)) => compare_config_errors(actual, expected),
        (actual, expected) => panic!("config decode differs: {actual:?} vs {expected:?}"),
    }
}

#[test]
fn serialized_configs_and_errors_match_before_optimization() {
    for log_alpha in [5, 6, 7, 8, 15] {
        // A complete config consumes at most twelve bits for these alphabet
        // sizes. Exhausting sixteen bits covers all valid and invalid headers.
        for header in 0..=u16::MAX {
            compare_config_decode(log_alpha, &header.to_le_bytes(), 0);
        }
        for header in (0..=u16::MAX).step_by(257) {
            let bytes = header.to_le_bytes();
            for len in 0..=bytes.len() {
                for offset in 0..=7.min(len * 8) {
                    compare_config_decode(log_alpha, &bytes[..len], offset);
                }
            }
        }
    }
}
