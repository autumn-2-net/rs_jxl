// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Bounded scalar port of base/exif.h::{IsExif, FindExifTagPosition,
//! InterpretExif}, frozen b866b3defec13494bfbaf705db93581cda118e37.
//! Only the first matching tag in IFD0 is interpreted; no pointer is followed
//! beyond the initial IFD offset. Malformed data is a no-op, as in the source.

#![forbid(unsafe_code)]

use crate::metadata::Orientation;

#[derive(Clone, Copy)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn from_tiff(tiff: &[u8]) -> Option<Self> {
        if tiff.len() < 12 {
            return None;
        }
        match tiff.get(..4)? {
            b"II\x2a\0" => Some(Self::Little),
            b"MM\0\x2a" => Some(Self::Big),
            _ => None,
        }
    }

    fn word(self, bytes: &[u8], offset: usize) -> Option<u16> {
        let value = bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
        Some(match self {
            Self::Little => u16::from_le_bytes(value),
            Self::Big => u16::from_be_bytes(value),
        })
    }

    fn dword(self, bytes: &[u8], offset: usize) -> Option<u32> {
        let value = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
        Some(match self {
            Self::Little => u32::from_le_bytes(value),
            Self::Big => u32::from_be_bytes(value),
        })
    }
}

fn first_orientation(tiff: &[u8]) -> Option<Orientation> {
    let endian = Endian::from_tiff(tiff)?;
    let offset = usize::try_from(endian.dword(tiff, 4)?).ok()?;
    // Retain the source's stronger-than-TIFF size guard, not just offset+2.
    if offset < 8 || offset.checked_add(14)? > tiff.len() {
        return None;
    }
    let count = endian.word(tiff, offset)?;
    let mut position = offset.checked_add(2)?;
    for _ in 0..count {
        let end = position.checked_add(12)?;
        // Source rejects even an entry ending EXACTLY at the blob boundary.
        if end >= tiff.len() {
            return None;
        }
        if endian.word(tiff, position)? == 274 {
            let kind = endian.word(tiff, position + 2)?;
            let count = endian.dword(tiff, position + 4)?;
            let value = endian.word(tiff, position + 8)?;
            // First matching tag wins even if its type/count/value is invalid.
            if kind != 3 || count != 1 || !(1..=8).contains(&value) {
                return None;
            }
            return Orientation::try_from(value as u8).ok();
        }
        position = end;
    }
    None
}

/// `tiff` excludes JPEG's Exif\0\0 signature and JXL's four-byte offset prefix.
/// Interpretation changes display metadata only; no rotation or TIFF rewrite.
pub(super) fn interpret(tiff: &[u8], initial: Orientation) -> Orientation {
    first_orientation(tiff).unwrap_or(initial)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_word(bytes: &mut [u8], offset: usize, value: u16, big: bool) {
        bytes[offset..offset + 2].copy_from_slice(&if big {
            value.to_be_bytes()
        } else {
            value.to_le_bytes()
        });
    }

    fn put_dword(bytes: &mut [u8], offset: usize, value: u32, big: bool) {
        bytes[offset..offset + 4].copy_from_slice(&if big {
            value.to_be_bytes()
        } else {
            value.to_le_bytes()
        });
    }

    fn tiff(big: bool, offset: usize, entries: &[(u16, u16, u32, u16)]) -> Vec<u8> {
        let mut result = vec![0; offset + 2 + entries.len() * 12 + 4];
        result[..4].copy_from_slice(if big { b"MM\0\x2a" } else { b"II\x2a\0" });
        put_dword(&mut result, 4, offset as u32, big);
        put_word(&mut result, offset, entries.len() as u16, big);
        for (index, &(tag, kind, count, value)) in entries.iter().enumerate() {
            let start = offset + 2 + index * 12;
            put_word(&mut result, start, tag, big);
            put_word(&mut result, start + 2, kind, big);
            put_dword(&mut result, start + 4, count, big);
            put_word(&mut result, start + 8, value, big);
        }
        result
    }

    #[test]
    fn all_eight_orientations_and_both_endian_headers_match_golden_values() {
        for big in [false, true] {
            for value in 1..=8 {
                for offset in [8, 17, 256] {
                    let bytes = tiff(big, offset, &[(274, 3, 1, value)]);
                    assert_eq!(
                        interpret(&bytes, Orientation::FlipVertical),
                        Orientation::try_from(value as u8).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn truncated_prefixes_keep_source_strict_entry_boundary() {
        for big in [false, true] {
            let bytes = tiff(big, 8, &[(274, 3, 1, 6)]);
            for length in 0..=bytes.len() {
                let expected = if length > 22 {
                    Orientation::Rotate90Clockwise
                } else {
                    Orientation::FlipVertical
                };
                assert_eq!(
                    interpret(&bytes[..length], Orientation::FlipVertical),
                    expected
                );
            }
            let bytes = tiff(big, 8, &[(1, 3, 1, 1), (274, 3, 1, 8)]);
            assert_eq!(
                interpret(&bytes[..34], Orientation::Identity),
                Orientation::Identity
            );
            assert_eq!(
                interpret(&bytes[..35], Orientation::Identity),
                Orientation::Rotate90CounterClockwise
            );
        }
    }

    #[test]
    fn invalid_offsets_headers_and_tag_counts_never_escape_the_blob() {
        for big in [false, true] {
            let original = tiff(big, 8, &[(274, 3, 1, 6)]);
            for offset in [0, 1, 7, 13, 25, 26, u32::MAX - 13, u32::MAX] {
                let mut bytes = original.clone();
                put_dword(&mut bytes, 4, offset, big);
                assert_eq!(
                    interpret(&bytes, Orientation::FlipHorizontal),
                    Orientation::FlipHorizontal
                );
            }
            for signature in [*b"II\0\x2a", *b"MM\x2a\0", *b"II\x2b\0", *b"Exif"] {
                let mut bytes = original.clone();
                bytes[..4].copy_from_slice(&signature);
                assert_eq!(
                    interpret(&bytes, Orientation::FlipHorizontal),
                    Orientation::FlipHorizontal
                );
            }
            let mut bytes = tiff(big, 8, &[(1, 3, 1, 6)]);
            put_word(&mut bytes, 8, u16::MAX, big);
            assert_eq!(
                interpret(&bytes, Orientation::FlipHorizontal),
                Orientation::FlipHorizontal
            );
            // Source returns an early match without validating the remaining directory.
            let mut bytes = original.clone();
            put_word(&mut bytes, 8, u16::MAX, big);
            assert_eq!(
                interpret(&bytes, Orientation::Identity),
                Orientation::Rotate90Clockwise
            );
            put_word(&mut bytes, 8, 0, big);
            assert_eq!(
                interpret(&bytes, Orientation::Identity),
                Orientation::Identity
            );
        }
    }

    #[test]
    fn first_matching_tag_blocks_duplicates_even_when_invalid() {
        for big in [false, true] {
            for entry in [
                (274, 1, 1, 6),
                (274, 4, 1, 6),
                (274, 3, 0, 6),
                (274, 3, 2, 6),
                (274, 3, u32::MAX, 6),
                (274, 3, 1, 0),
                (274, 3, 1, 9),
                (274, 3, 1, 0x106),
            ] {
                let bytes = tiff(big, 8, &[entry, (274, 3, 1, 8)]);
                assert_eq!(
                    interpret(&bytes, Orientation::Rotate180),
                    Orientation::Rotate180
                );
            }
            let bytes = tiff(big, 8, &[(1, 3, 1, 8), (274, 3, 1, 2), (274, 3, 1, 6)]);
            assert_eq!(
                interpret(&bytes, Orientation::Identity),
                Orientation::FlipHorizontal
            );
        }
    }

    #[test]
    fn next_ifd_and_subifd_pointers_are_not_followed() {
        for big in [false, true] {
            let mut bytes = tiff(big, 8, &[(0x8769, 4, 1, 0)]);
            bytes.resize(64, 0);
            put_dword(&mut bytes, 18, 32, big);
            put_dword(&mut bytes, 22, 32, big);
            put_word(&mut bytes, 32, 1, big);
            put_word(&mut bytes, 34, 274, big);
            put_word(&mut bytes, 36, 3, big);
            put_dword(&mut bytes, 38, 1, big);
            put_word(&mut bytes, 42, 8, big);
            assert_eq!(
                interpret(&bytes, Orientation::Identity),
                Orientation::Identity
            );
        }
    }

    #[test]
    fn deterministic_bounded_malformed_inputs_are_noops() {
        let mut state = 7u32;
        for length in 0..512 {
            let mut bytes = vec![0; length];
            for byte in &mut bytes {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (state >> 24) as u8;
            }
            // Exercise real headers with random out-of-range directory offsets.
            for big in [false, true] {
                if length >= 8 {
                    bytes[..4].copy_from_slice(if big { b"MM\0\x2a" } else { b"II\x2a\0" });
                    put_dword(&mut bytes, 4, 0xffff0000 | state, big);
                }
                assert_eq!(
                    interpret(&bytes, Orientation::AntiTranspose),
                    Orientation::AntiTranspose
                );
            }
        }
    }
}
