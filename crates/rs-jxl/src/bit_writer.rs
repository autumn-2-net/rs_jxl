// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Direct port of lib/jxl/enc_bit_writer.{h,cc}, b866b3d.
//! Same LSB-first, at-most-56-bit write operation; safe little-endian stores
//! replace C++ unaligned memcpy. Accounting/allotment profiling is not ported.

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitWriterError {
    TooManyBits,
    ValueOutOfRange,
    Unaligned,
    SizeLimit,
    Allocation,
}

impl fmt::Display for BitWriterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TooManyBits => "one write accepts at most 56 bits",
            Self::ValueOutOfRange => "value does not fit its bit count",
            Self::Unaligned => "operation requires byte alignment",
            Self::SizeLimit => "bitstream size limit exceeded",
            Self::Allocation => "bitstream allocation failed",
        })
    }
}
impl std::error::Error for BitWriterError {}
type Result<T> = std::result::Result<T, BitWriterError>;

/// Bounded logical output, with eight bytes of initialized store padding.
/// The limit does not include Vec capacity or that padding.
pub struct BitWriter {
    storage: Vec<u8>,
    bits_written: usize,
    max_bytes: usize,
}

impl BitWriter {
    pub const MAX_BITS_PER_CALL: u8 = 56;
    pub fn new(max_bytes: usize) -> Self {
        Self {
            storage: Vec::new(),
            bits_written: 0,
            max_bytes,
        }
    }
    pub fn bits_written(&self) -> usize {
        self.bits_written
    }

    fn reserve(&mut self, additional_bits: usize) -> Result<()> {
        let end = self
            .bits_written
            .checked_add(additional_bits)
            .ok_or(BitWriterError::SizeLimit)?;
        let bytes = end.div_ceil(8);
        if bytes > self.max_bytes {
            return Err(BitWriterError::SizeLimit);
        }
        let padded = bytes.checked_add(8).ok_or(BitWriterError::SizeLimit)?;
        if padded > self.storage.len() {
            self.storage
                .try_reserve(padded - self.storage.len())
                .map_err(|_| BitWriterError::Allocation)?;
            self.storage.resize(padded, 0);
        }
        Ok(())
    }

    pub fn write(&mut self, n_bits: u8, value: u64) -> Result<()> {
        if n_bits > Self::MAX_BITS_PER_CALL {
            return Err(BitWriterError::TooManyBits);
        }
        if value >> n_bits != 0 {
            return Err(BitWriterError::ValueOutOfRange);
        }
        if n_bits == 0 {
            return Ok(());
        }
        self.reserve(usize::from(n_bits))?;
        let offset = self.bits_written / 8;
        let shift = self.bits_written % 8;
        let word = u64::from(self.storage[offset]) | (value << shift);
        self.storage[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
        self.bits_written += usize::from(n_bits);
        Ok(())
    }

    pub fn zero_pad_to_byte(&mut self) -> Result<()> {
        self.write(((8 - self.bits_written % 8) % 8) as u8, 0)
    }

    pub fn bytes(&self) -> Result<&[u8]> {
        if !self.bits_written.is_multiple_of(8) {
            return Err(BitWriterError::Unaligned);
        }
        Ok(&self.storage[..self.bits_written / 8])
    }

    pub fn into_bytes(mut self) -> Result<Vec<u8>> {
        if !self.bits_written.is_multiple_of(8) {
            return Err(BitWriterError::Unaligned);
        }
        self.storage.truncate(self.bits_written / 8);
        Ok(self.storage)
    }

    pub fn append_byte_aligned(&mut self, data: &[u8]) -> Result<()> {
        if !self.bits_written.is_multiple_of(8) {
            return Err(BitWriterError::Unaligned);
        }
        let bits = data.len().checked_mul(8).ok_or(BitWriterError::SizeLimit)?;
        self.reserve(bits)?;
        let offset = self.bits_written / 8;
        self.storage[offset..offset + data.len()].copy_from_slice(data);
        self.storage[offset + data.len()] = 0;
        self.bits_written += bits;
        Ok(())
    }

    pub fn append_unaligned(&mut self, other: &Self) -> Result<()> {
        self.reserve(other.bits_written)?;
        let full_bytes = other.bits_written / 8;
        for &byte in &other.storage[..full_bytes] {
            self.write(8, u64::from(byte))?;
        }
        let tail = (other.bits_written % 8) as u8;
        if tail != 0 {
            self.write(
                tail,
                u64::from(other.storage[full_bytes]) & ((1 << tail) - 1),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_bitwise_reference_at_every_width_and_offset() {
        for initial in 0..8u8 {
            let mut writer = BitWriter::new(65536);
            let mut expected = Vec::new();
            let mut state = 0x8db3_a598_6642_99a7u64;
            for width in std::iter::once(initial).chain((0..=56).cycle().take(400)) {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let value = state & ((1u64 << width) - 1);
                writer.write(width, value).unwrap();
                expected.extend((0..width).map(|i| ((value >> i) & 1) as u8));
            }
            assert_eq!(writer.bits_written(), expected.len());
            writer.zero_pad_to_byte().unwrap();
            let bytes = writer.into_bytes().unwrap();
            for (i, &bit) in expected.iter().enumerate() {
                assert_eq!((bytes[i / 8] >> (i % 8)) & 1, bit);
            }
            for i in expected.len()..bytes.len() * 8 {
                assert_eq!((bytes[i / 8] >> (i % 8)) & 1, 0);
            }
        }
    }
    #[test]
    fn appends_aligned_and_unaligned_without_corrupting_padding() {
        let mut a = BitWriter::new(16);
        a.append_byte_aligned(&[0xff, 0x81]).unwrap();
        a.write(3, 5).unwrap();
        let mut b = BitWriter::new(16);
        b.write(10, 0x2ab).unwrap();
        a.append_unaligned(&b).unwrap();
        assert_eq!(a.bits_written(), 29);
        a.zero_pad_to_byte().unwrap();
        assert_eq!(a.bytes().unwrap(), &[0xff, 0x81, 0x5d, 0x15]);
        a.append_byte_aligned(&[0x42]).unwrap();
        a.write(8, 0x71).unwrap();
        assert_eq!(
            a.into_bytes().unwrap(),
            &[0xff, 0x81, 0x5d, 0x15, 0x42, 0x71]
        );
    }
    #[test]
    fn rejects_invalid_operations_without_changing_bits() {
        let mut w = BitWriter::new(1);
        assert_eq!(w.write(57, 0), Err(BitWriterError::TooManyBits));
        assert_eq!(w.write(0, 1), Err(BitWriterError::ValueOutOfRange));
        assert_eq!(w.write(3, 8), Err(BitWriterError::ValueOutOfRange));
        w.write(3, 7).unwrap();
        assert_eq!(w.bytes(), Err(BitWriterError::Unaligned));
        assert_eq!(w.append_byte_aligned(&[]), Err(BitWriterError::Unaligned));
        assert_eq!(w.write(6, 0), Err(BitWriterError::SizeLimit));
        assert_eq!(w.bits_written(), 3);
        w.zero_pad_to_byte().unwrap();
        assert_eq!(w.into_bytes().unwrap(), &[7]);
        let mut empty = BitWriter::new(0);
        empty.write(0, 0).unwrap();
        assert!(empty.into_bytes().unwrap().is_empty());
    }
}
