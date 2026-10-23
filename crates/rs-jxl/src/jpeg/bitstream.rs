// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! JPEG entropy primitives ported from lib/jxl/jpeg/enc_jpeg_data_reader.cc
//! (BitReaderState, ProcessRestart and HuffExtend), frozen at b866b3d.
//!
//! JPEG bits are MSB-first, unlike the JXL bitstream. Reads never synthesize
//! bits at EOF or across a marker. Only the current entropy byte is buffered,
//! so marker offsets and padding remain exact without speculative-byte rewind.
//! All fallible operations leave the reader unchanged on error.
//!
//! This is not a marker-segment, coefficient, or pixel decoder. The scan parser
//! owns restart intervals/order, DC predictors, progressive state, and category
//! limits for the selected JPEG process.

use std::fmt;

/// A marker including any leading 0xff fill bytes; offsets index the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Marker {
    pub code: u8,
    /// First 0xff, including fill bytes.
    pub offset: usize,
    /// Position immediately after the non-0xff marker code.
    pub end: usize,
}

impl Marker {
    pub fn restart_index(self) -> Option<u8> {
        (0xd0..=0xd7).contains(&self.code).then(|| self.code - 0xd0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingPolicy {
    /// Require JPEG's prescribed all-one alignment bits.
    RequireOnes,
    /// Preserve nonstandard zero/mixed padding, as the frozen reader does.
    Preserve,
}

/// Zero to seven bits, right-aligned in `value()`, in original MSB-first order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaddingBits {
    value: u8,
    bit_count: u8,
}

impl PaddingBits {
    pub fn value(self) -> u8 {
        self.value
    }

    pub fn bit_count(self) -> u8 {
        self.bit_count
    }

    pub fn is_all_ones(self) -> bool {
        u16::from(self.value) == (1u16 << self.bit_count) - 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyBoundary {
    pub padding: PaddingBits,
    pub marker: Marker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntropyError {
    InvalidPosition { position: usize, length: usize },
    InvalidBitCount(u8),
    UnexpectedEnd { position: usize },
    TruncatedMarker { position: usize },
    InvalidStuffing { position: usize },
    MarkerEncountered(Marker),
    ExpectedMarker { position: usize },
    Unaligned { bits_left: u8 },
    InvalidPadding(PaddingBits),
    InvalidRestartIndex(u8),
    UnexpectedRestart { expected: u8, marker: Marker },
    InvalidCategory(u8),
    InvalidMagnitude { value: u32, category: u8 },
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPosition { position, length } => {
                write!(f, "entropy offset {position} exceeds input length {length}")
            }
            Self::InvalidBitCount(count) => write!(f, "cannot read {count} bits (maximum 32)"),
            Self::UnexpectedEnd { position } => write!(f, "entropy data ends at byte {position}"),
            Self::TruncatedMarker { position } => {
                write!(f, "unterminated 0xff escape/marker at byte {position}")
            }
            Self::InvalidStuffing { position } => {
                write!(
                    f,
                    "0xff fill before stuffed entropy byte at byte {position}"
                )
            }
            Self::MarkerEncountered(marker) => {
                write!(
                    f,
                    "marker ff{:02x} interrupts entropy at byte {}",
                    marker.code, marker.offset
                )
            }
            Self::ExpectedMarker { position } => write!(f, "expected marker at byte {position}"),
            Self::Unaligned { bits_left } => {
                write!(f, "{bits_left} entropy bits remain before marker")
            }
            Self::InvalidPadding(_) => f.write_str("JPEG alignment padding contains zero bits"),
            Self::InvalidRestartIndex(index) => {
                write!(f, "invalid restart index {index} (maximum 7)")
            }
            Self::UnexpectedRestart { expected, marker } => {
                write!(
                    f,
                    "expected RST{expected}, found ff{:02x} at byte {}",
                    marker.code, marker.offset
                )
            }
            Self::InvalidCategory(category) => {
                write!(f, "invalid magnitude category {category} (maximum 16)")
            }
            Self::InvalidMagnitude { value, category } => {
                write!(f, "magnitude bits {value} do not fit category {category}")
            }
        }
    }
}

impl std::error::Error for EntropyError {}

enum Token {
    Byte { value: u8, end: usize },
    Marker(Marker),
}

#[derive(Clone, Copy, Debug)]
pub struct EntropyBitReader<'a> {
    data: &'a [u8],
    position: usize,
    current: u8,
    bits_left: u8,
}

impl<'a> EntropyBitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            current: 0,
            bits_left: 0,
        }
    }

    /// Use a whole-file slice to keep all returned offsets file-relative.
    pub fn with_offset(data: &'a [u8], offset: usize) -> Result<Self, EntropyError> {
        if offset > data.len() {
            return Err(EntropyError::InvalidPosition {
                position: offset,
                length: data.len(),
            });
        }
        Ok(Self {
            position: offset,
            ..Self::new(data)
        })
    }

    /// Next raw byte, after the currently buffered byte (and its stuffing).
    /// At a scan boundary this is the first 0xff of the unconsumed marker.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Unconsumed low bits of the current byte, always in 0..=7.
    pub fn bits_left(&self) -> u8 {
        self.bits_left
    }

    fn next_token(&self) -> Result<Token, EntropyError> {
        let start = self.position;
        let value = *self
            .data
            .get(start)
            .ok_or(EntropyError::UnexpectedEnd { position: start })?;
        if value != 0xff {
            return Ok(Token::Byte {
                value,
                end: start + 1,
            });
        }
        let mut cursor = start + 1;
        loop {
            let code = *self
                .data
                .get(cursor)
                .ok_or(EntropyError::TruncatedMarker { position: start })?;
            if code == 0xff {
                cursor += 1;
            } else if code == 0 {
                if cursor != start + 1 {
                    return Err(EntropyError::InvalidStuffing { position: start });
                }
                return Ok(Token::Byte {
                    value: 0xff,
                    end: cursor + 1,
                });
            } else {
                return Ok(Token::Marker(Marker {
                    code,
                    offset: start,
                    end: cursor + 1,
                }));
            }
        }
    }

    // Call only on a trial reader: an error can follow partial consumption.
    fn read_bits_inner(&mut self, count: u8) -> Result<u32, EntropyError> {
        if count > 32 {
            return Err(EntropyError::InvalidBitCount(count));
        }
        let mut remaining = count;
        let mut value = 0u32;
        while remaining > 0 {
            if self.bits_left == 0 {
                match self.next_token()? {
                    Token::Byte { value, end } => {
                        self.current = value;
                        self.bits_left = 8;
                        self.position = end;
                    }
                    Token::Marker(marker) => return Err(EntropyError::MarkerEncountered(marker)),
                }
            }
            let take = remaining.min(self.bits_left);
            self.bits_left -= take;
            let mask = (1u32 << take) - 1;
            value = (value << take) | ((u32::from(self.current) >> self.bits_left) & mask);
            remaining -= take;
        }
        Ok(value)
    }

    pub fn read_bits(&mut self, count: u8) -> Result<u32, EntropyError> {
        let mut trial = *self;
        let value = trial.read_bits_inner(count)?;
        *self = trial;
        Ok(value)
    }

    pub fn peek_bits(&self, count: u8) -> Result<u32, EntropyError> {
        let mut trial = *self;
        trial.read_bits_inner(count)
    }

    pub fn read_bit(&mut self) -> Result<bool, EntropyError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read the JPEG sign/magnitude suffix, including the zero-bit category.
    /// DCT process-specific DC/AC category restrictions belong to the caller.
    pub fn receive_and_extend(&mut self, category: u8) -> Result<i32, EntropyError> {
        if category > 16 {
            return Err(EntropyError::InvalidCategory(category));
        }
        extend(self.read_bits(category)?, category)
    }

    /// Discard only the remainder of the current byte, retaining its exact bits.
    /// Does not require a following marker; useful for isolated entropy tests.
    pub fn take_padding(&mut self, policy: PaddingPolicy) -> Result<PaddingBits, EntropyError> {
        let mask = ((1u16 << self.bits_left) - 1) as u8;
        let padding = PaddingBits {
            value: self.current & mask,
            bit_count: self.bits_left,
        };
        if policy == PaddingPolicy::RequireOnes && !padding.is_all_ones() {
            return Err(EntropyError::InvalidPadding(padding));
        }
        self.current = 0;
        self.bits_left = 0;
        Ok(padding)
    }

    pub fn peek_marker(&self) -> Result<Marker, EntropyError> {
        if self.bits_left != 0 {
            return Err(EntropyError::Unaligned {
                bits_left: self.bits_left,
            });
        }
        match self.next_token()? {
            Token::Marker(marker) => Ok(marker),
            Token::Byte { .. } => Err(EntropyError::ExpectedMarker {
                position: self.position,
            }),
        }
    }

    /// Align and require a following marker, but leave that marker unconsumed.
    /// The caller must decide whether the marker is legal at this scan boundary.
    pub fn finish_scan(&mut self, policy: PaddingPolicy) -> Result<EntropyBoundary, EntropyError> {
        let mut trial = *self;
        let padding = trial.take_padding(policy)?;
        let marker = trial.peek_marker()?;
        *self = trial;
        Ok(EntropyBoundary { padding, marker })
    }

    /// Explicitly consume RST0..RST7 after alignment. Never skip entropy bytes.
    /// The caller advances the expected index modulo eight and resets predictors.
    pub fn consume_restart(
        &mut self,
        expected_index: u8,
        policy: PaddingPolicy,
    ) -> Result<EntropyBoundary, EntropyError> {
        if expected_index > 7 {
            return Err(EntropyError::InvalidRestartIndex(expected_index));
        }
        let mut trial = *self;
        let boundary = trial.finish_scan(policy)?;
        if boundary.marker.restart_index() != Some(expected_index) {
            return Err(EntropyError::UnexpectedRestart {
                expected: expected_index,
                marker: boundary.marker,
            });
        }
        trial.position = boundary.marker.end;
        *self = trial;
        Ok(boundary)
    }
}

/// JPEG's receive/extend mapping, not two's-complement sign extension.
/// Category 16 is supported for callers that need the full JPEG magnitude range.
pub fn extend(value: u32, category: u8) -> Result<i32, EntropyError> {
    if category > 16 {
        return Err(EntropyError::InvalidCategory(category));
    }
    let limit = 1u32 << category;
    if value >= limit {
        return Err(EntropyError::InvalidMagnitude { value, category });
    }
    if category == 0 || value >= limit / 2 {
        Ok(value as i32)
    } else {
        Ok(value as i32 - limit as i32 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msb_reads_and_stuffing_do_not_consume_markers() {
        let data = [0xa5, 0xff, 0x00, 0x5a, 0xff, 0xd9];
        let mut reader = EntropyBitReader::new(&data);
        assert_eq!(reader.peek_bits(12), Ok(0xa5f));
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.read_bits(4), Ok(0xa));
        assert_eq!(reader.read_bits(12), Ok(0x5ff));
        assert_eq!(reader.position(), 3);
        assert_eq!(reader.read_bits(8), Ok(0x5a));
        let marker = Marker {
            code: 0xd9,
            offset: 4,
            end: 6,
        };
        assert_eq!(
            reader.read_bit(),
            Err(EntropyError::MarkerEncountered(marker))
        );
        assert_eq!(reader.position(), 4);
        assert_eq!(reader.peek_marker(), Ok(marker));
        assert_eq!(reader.read_bits(0), Ok(0));
    }

    #[test]
    fn reads_are_atomic_at_eof_markers_and_invalid_widths() {
        let mut reader = EntropyBitReader::new(&[0x80, 0xff, 0xd0]);
        assert_eq!(reader.read_bit(), Ok(true));
        assert!(matches!(
            reader.read_bits(8),
            Err(EntropyError::MarkerEncountered(_))
        ));
        assert_eq!((reader.position(), reader.bits_left()), (1, 7));
        assert_eq!(reader.read_bits(7), Ok(0));
        let mut reader = EntropyBitReader::new(&[0xa5]);
        assert_eq!(
            reader.read_bits(9),
            Err(EntropyError::UnexpectedEnd { position: 1 })
        );
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.read_bits(33), Err(EntropyError::InvalidBitCount(33)));
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.read_bits(8), Ok(0xa5));
        assert_eq!(reader.read_bits(0), Ok(0));
        assert!(matches!(
            reader.read_bit(),
            Err(EntropyError::UnexpectedEnd { .. })
        ));
        let mut reader = EntropyBitReader::new(&[0x12, 0x34, 0x56, 0x78]);
        assert_eq!(reader.read_bits(32), Ok(0x12345678));
        assert_eq!(reader.position(), 4);
    }

    #[test]
    fn rejects_truncated_escapes_and_fill_before_stuffing() {
        for data in [&[0xff][..], &[0xff, 0xff][..]] {
            let mut reader = EntropyBitReader::new(data);
            assert_eq!(
                reader.read_bit(),
                Err(EntropyError::TruncatedMarker { position: 0 })
            );
            assert_eq!(reader.position(), 0);
        }
        let mut reader = EntropyBitReader::new(&[0xff, 0xff, 0x00]);
        assert_eq!(
            reader.read_bit(),
            Err(EntropyError::InvalidStuffing { position: 0 })
        );
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn explicit_restarts_preserve_fill_and_padding() {
        let data = [0xbf, 0xff, 0xff, 0xd0, 0x7f, 0xff, 0xd1, 0xff, 0xd9];
        let mut reader = EntropyBitReader::new(&data);
        assert_eq!(reader.read_bits(2), Ok(2));
        assert_eq!(
            reader.peek_marker(),
            Err(EntropyError::Unaligned { bits_left: 6 })
        );
        assert!(matches!(
            reader.consume_restart(1, PaddingPolicy::RequireOnes),
            Err(EntropyError::UnexpectedRestart { .. })
        ));
        assert_eq!((reader.position(), reader.bits_left()), (1, 6));
        let boundary = reader
            .consume_restart(0, PaddingPolicy::RequireOnes)
            .unwrap();
        assert_eq!(
            boundary.padding,
            PaddingBits {
                value: 63,
                bit_count: 6
            }
        );
        assert_eq!(
            boundary.marker,
            Marker {
                code: 0xd0,
                offset: 1,
                end: 4
            }
        );
        assert_eq!(reader.position(), 4);
        assert_eq!(reader.read_bit(), Ok(false));
        reader
            .consume_restart(1, PaddingPolicy::RequireOnes)
            .unwrap();
        let boundary = reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
        assert_eq!(boundary.padding.bit_count(), 0);
        assert_eq!(boundary.marker.code, 0xd9);
        assert_eq!(reader.position(), 7);
    }

    #[test]
    fn restart_sequence_wraps_only_under_caller_control() {
        let data: Vec<u8> = (0..10).flat_map(|i| [0x7f, 0xff, 0xd0 + (i & 7)]).collect();
        let mut reader = EntropyBitReader::new(&data);
        for i in 0..10 {
            assert_eq!(reader.read_bit(), Ok(false));
            assert_eq!(
                reader.consume_restart(8, PaddingPolicy::RequireOnes),
                Err(EntropyError::InvalidRestartIndex(8))
            );
            let boundary = reader
                .consume_restart(i & 7, PaddingPolicy::RequireOnes)
                .unwrap();
            assert_eq!(boundary.marker.restart_index(), Some(i & 7));
        }
        assert_eq!(reader.position(), data.len());
    }

    #[test]
    fn padding_can_be_strict_or_preserved_exactly() {
        let mut reader = EntropyBitReader::new(&[0xa5, 0xff, 0xd9]);
        assert_eq!(reader.read_bits(3), Ok(5));
        let padding = PaddingBits {
            value: 5,
            bit_count: 5,
        };
        assert_eq!(
            reader.finish_scan(PaddingPolicy::RequireOnes),
            Err(EntropyError::InvalidPadding(padding))
        );
        assert_eq!((reader.position(), reader.bits_left()), (1, 5));
        let boundary = reader.finish_scan(PaddingPolicy::Preserve).unwrap();
        assert_eq!(boundary.padding, padding);
        assert!(!boundary.padding.is_all_ones());
        assert_eq!(boundary.padding.value(), 5);
        assert_eq!(reader.position(), 1);
        let mut reader = EntropyBitReader::new(&[0xff, 0x00, 0xff, 0xd9]);
        assert_eq!(reader.read_bit(), Ok(true));
        let boundary = reader.finish_scan(PaddingPolicy::RequireOnes).unwrap();
        assert_eq!(
            boundary.padding,
            PaddingBits {
                value: 127,
                bit_count: 7
            }
        );
        assert_eq!(boundary.marker.offset, 2);
    }

    #[test]
    fn finish_requires_marker_without_skipping_data_or_eof() {
        let mut reader = EntropyBitReader::new(&[0x7f, 0x00, 0xff, 0xd9]);
        reader.read_bit().unwrap();
        assert_eq!(
            reader.finish_scan(PaddingPolicy::RequireOnes),
            Err(EntropyError::ExpectedMarker { position: 1 })
        );
        assert_eq!((reader.position(), reader.bits_left()), (1, 7));
        let mut reader = EntropyBitReader::new(&[0x7f]);
        reader.read_bit().unwrap();
        assert_eq!(
            reader.finish_scan(PaddingPolicy::RequireOnes),
            Err(EntropyError::UnexpectedEnd { position: 1 })
        );
        assert_eq!(reader.bits_left(), 7);
        assert_eq!(
            reader
                .take_padding(PaddingPolicy::RequireOnes)
                .unwrap()
                .bit_count(),
            7
        );
    }

    #[test]
    fn offsets_and_empty_input_are_bounded() {
        assert!(matches!(
            EntropyBitReader::with_offset(&[], 1),
            Err(EntropyError::InvalidPosition { .. })
        ));
        let mut empty = EntropyBitReader::with_offset(&[], 0).unwrap();
        assert_eq!(empty.read_bits(0), Ok(0));
        assert_eq!(
            empty.read_bit(),
            Err(EntropyError::UnexpectedEnd { position: 0 })
        );
        let mut reader = EntropyBitReader::with_offset(&[9, 9, 0x7f, 0xff, 0xd9], 2).unwrap();
        reader.read_bit().unwrap();
        assert_eq!(
            reader
                .finish_scan(PaddingPolicy::RequireOnes)
                .unwrap()
                .marker
                .offset,
            3
        );
    }

    #[test]
    fn sign_extension_covers_every_category_and_negative_boundary() {
        assert_eq!(extend(0, 0), Ok(0));
        assert_eq!(extend(0, 1), Ok(-1));
        assert_eq!(extend(1, 1), Ok(1));
        for category in 1..=16 {
            let limit = 1u32 << category;
            assert_eq!(extend(0, category), Ok(1 - limit as i32));
            assert_eq!(extend(limit / 2 - 1, category), Ok(-(limit as i32) / 2));
            assert_eq!(extend(limit / 2, category), Ok((limit / 2) as i32));
            assert_eq!(extend(limit - 1, category), Ok(limit as i32 - 1));
            assert_eq!(
                extend(limit, category),
                Err(EntropyError::InvalidMagnitude {
                    value: limit,
                    category
                })
            );
        }
        assert_eq!(extend(0, 17), Err(EntropyError::InvalidCategory(17)));
        assert_eq!(
            extend(1, 0),
            Err(EntropyError::InvalidMagnitude {
                value: 1,
                category: 0
            })
        );
        let mut reader = EntropyBitReader::new(&[0b0101_1111]);
        assert_eq!(
            reader.receive_and_extend(17),
            Err(EntropyError::InvalidCategory(17))
        );
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.receive_and_extend(0), Ok(0));
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.receive_and_extend(3), Ok(-5));
        assert_eq!(reader.receive_and_extend(3), Ok(7));
    }
}
