// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::*;
use crate::entropy_coding::huffman::HuffmanCodes;

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    len: usize,
}

impl Bits {
    fn put(&mut self, count: usize, value: u32) {
        for bit in 0..count {
            if self.len % 8 == 0 {
                self.bytes.push(0);
            }
            self.bytes[self.len / 8] |= (((value >> bit) & 1) as u8) << (self.len % 8);
            self.len += 1;
        }
    }

    fn uint(&mut self, value: u32) {
        // HybridUint(0, 0, 0), with a fixed eight-bit prefix code per token.
        let token = if value == 0 { 0 } else { value.ilog2() + 1 };
        self.put(8, u32::from((token as u8).reverse_bits()));
        if token > 1 {
            self.put((token - 1) as usize, value - (1 << (token - 1)));
        }
    }
}

fn prefix_header(lz77: bool) -> Bits {
    let mut bits = Bits::default();
    bits.put(1, u32::from(lz77));
    if lz77 {
        bits.put(2, 0); // min_symbol = 224
        bits.put(2, 0); // min_length = 3
        bits.put(4, 0); // HybridUint(0, 0, 0) for lengths
    }
    bits.put(1, 1); // simple context map
    bits.put(2, 0); // every context uses histogram 0
    bits.put(1, 1); // prefix coding
    bits.put(4, 0); // HybridUint(0, 0, 0)

    // Same 256-symbol histogram fixture as entropy_coding::huffman tests.
    let header = [0xef, 0x3f, 0, 1, 0, 0xa0, 6];
    let mut reader = BitReader::new(&header);
    HuffmanCodes::decode(1, &mut reader).unwrap();
    for bit in 0..reader.total_bits_read() {
        bits.put(1, u32::from((header[bit / 8] >> (bit % 8)) & 1));
    }
    bits
}

fn prefix_values(values: &[u32]) -> Vec<u8> {
    let mut bits = prefix_header(false);
    for &value in values {
        bits.uint(value);
    }
    bits.bytes
}

fn parse(bytes: &[u8]) -> Result<PatchesDictionary> {
    PatchesDictionary::read_with_reference_info(
        &mut BitReader::new(bytes),
        16,
        16,
        0,
        &[Some(((16, 16), true))],
        true,
        &mut Vec::new(),
    )
}

fn empty_ans(state: u32) -> Vec<u8> {
    let mut bits = Bits::default();
    bits.put(1, 0); // no LZ77
    bits.put(1, 1); // simple context map
    bits.put(2, 0); // histogram 0
    bits.put(1, 0); // ANS
    bits.put(2, 0); // log_alpha_size = 5
    bits.put(3, 0); // HybridUint(0, 0, 0)
    bits.put(1, 1); // small histogram
    bits.put(1, 0); // single symbol
    bits.put(1, 0); // symbol 0: no reference patches
    bits.put(32, state);
    bits.bytes
}

#[test]
fn ans_final_state_is_checked() {
    assert!(parse(&empty_ans(0x130000)).unwrap().positions.is_empty());
    assert!(matches!(
        parse(&empty_ans(0x130001)),
        Err(Error::AnsChecksumMismatch)
    ));
}

#[test]
fn lz77_error_is_not_accepted_as_empty_dictionary() {
    let mut bits = prefix_header(true);
    bits.put(8, u32::from(224u8.reverse_bits()));
    assert!(matches!(
        parse(&bits.bytes),
        Err(Error::UnexpectedLz77Repeat)
    ));
}

#[test]
fn valid_prefix_dictionary_and_truncated_payload() {
    let data = prefix_values(&[1, 0, 0, 0, 0, 0, 0, 2, 3, 1]);
    let dictionary = parse(&data).unwrap();
    assert_eq!(dictionary.positions.len(), 1);
    assert_eq!(
        (dictionary.positions[0].x, dictionary.positions[0].y),
        (2, 3)
    );
    assert_eq!(dictionary.blendings[0].mode, PatchBlendMode::Replace);
    assert!(parse(&data[..data.len() - 1]).is_err());
    let ans = empty_ans(0x130000);
    for end in 0..ans.len() {
        assert!(parse(&ans[..end]).is_err(), "accepted truncation at {end}");
    }
}

#[test]
fn encoded_blend_mode_cannot_alias_low_byte() {
    for value in 0..8 {
        assert_eq!(PatchBlendMode::from_encoded(value).unwrap() as u32, value);
    }
    for value in [8, 255, 256, 257, 258, 259, 260, 261, 262, 263, u32::MAX] {
        let data = prefix_values(&[1, 0, 0, 0, 0, 0, 0, 2, 3, value, 0]);
        assert!(
            matches!(
                parse(&data),
                Err(Error::PatchesInvalidBlendMode(..) | Error::InvalidEnum(..))
            ),
            "accepted blend mode {value}"
        );
    }
}

#[test]
fn signed_offsets_do_not_overflow_or_wrap() {
    assert_eq!(checked_patch_offset("x", 8, -8).unwrap(), 0);
    assert_eq!(
        checked_patch_offset("x", 1usize << 31, i32::MIN).unwrap(),
        0
    );
    assert_eq!(
        checked_patch_offset("x", i32::MAX as usize, 1).unwrap(),
        1usize << 31
    );
    assert!(checked_patch_offset("x", 0, i32::MIN).is_err());
    assert!(checked_patch_offset("x", usize::MAX, 1).is_err());
    for (dx, dy) in [(u32::MAX, 0), (0, u32::MAX)] {
        let data = prefix_values(&[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, dx, dy, 1]);
        assert!(matches!(parse(&data), Err(Error::PatchesInvalidDelta(..))));
    }
    // Signed i32::MAX added to 1 exceeds i32, but must still reach bounds checking.
    let data = prefix_values(&[1, 0, 0, 0, 0, 0, 1, 1, 0, 1, u32::MAX - 1, 0, 1]);
    assert!(matches!(parse(&data), Err(Error::PatchesOutOfBounds(..))));
}

#[test]
fn invalid_reference_and_extent_are_errors() {
    assert!(patch_range_fits(3, 5, 8));
    assert!(!patch_range_fits(3, 6, 8));
    assert!(!patch_range_fits(usize::MAX, 1, usize::MAX));
    let missing_slot = prefix_values(&[1, 1, 0, 0, 0, 0]);
    assert!(matches!(
        parse(&missing_slot),
        Err(Error::PatchesInvalidReference(1))
    ));
    let oversized_patch = prefix_values(&[1, 0, 0, 0, u32::MAX, 0]);
    assert!(matches!(
        parse(&oversized_patch),
        Err(Error::PatchesInvalidPosition(..) | Error::SizeOverflow)
    ));
}
