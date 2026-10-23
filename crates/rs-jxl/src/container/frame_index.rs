// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! `jxli` payload primitives from encode_internal.h/encode.cc at b866b3d.
//! Box framing, keyframe eligibility and ordered codestream accounting belong
//! to the caller. SourceFrameIndex deliberately retains the frozen serializer's
//! lagged duration behavior; see docs/FRAME_INDEX.md before integrating it.

#![forbid(unsafe_code)]

use crate::{Error, Result};

pub const BOX_TYPE: [u8; 4] = *b"jxli";
const MAX_FIELD: u64 = i64::MAX as u64;

/// Seconds per index tick, not ticks per second as in Animation metadata.
/// The wire permits a zero numerator, but never a zero denominator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TickDuration {
    pub numerator: u32,
    pub denominator: u32,
}

impl Default for TickDuration {
    fn default() -> Self {
        Self {
            numerator: 1,
            denominator: 1000,
        }
    }
}

impl TickDuration {
    pub fn validate(self) -> Result<()> {
        if self.denominator == 0 {
            return Err(Error::InvalidInput(
                "frame index tick denominator must be nonzero".into(),
            ));
        }
        Ok(())
    }
}

/// One already-aggregated index record. The first frame must be represented.
/// Offsets count concatenated bare codestream bytes, excluding all box framing
/// and the four-byte jxlp index. Later offsets are relative to the previous
/// indexed frame, not to the previous encoded frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameIndexEntry {
    pub offset_delta: u64,
    /// Ticks until the next indexed frame, or until the end for the last entry.
    pub duration: u64,
    /// Presented frames in that interval, including this entry's frame.
    /// Exclude nondisplayed composition frames; include a zero-duration last frame.
    pub frame_count: u64,
}

fn varint_len(mut value: u64) -> usize {
    let mut length = 1;
    while value >= 128 {
        value >>= 7;
        length += 1;
    }
    length
}

fn write_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 128 {
        output.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    output.push(value as u8);
}

/// Serialize NF, big-endian TNUM/TDEN, then unsigned base-128 OFFi/Ti/Fi.
/// Values are bounded to 63 bits, with strictly increasing absolute offsets.
/// `max_payload_bytes` excludes the box header. Validation and size accounting
/// finish before allocating output. No keyframe/timeline inference occurs here.
pub fn encode_payload(
    ticks: TickDuration,
    entries: &[FrameIndexEntry],
    max_payload_bytes: usize,
) -> Result<Vec<u8>> {
    ticks.validate()?;
    let count = u64::try_from(entries.len())
        .ok()
        .filter(|&n| n != 0 && n <= MAX_FIELD)
        .ok_or_else(|| {
            Error::InvalidInput("frame index requires a nonempty bounded index".into())
        })?;
    let mut length = varint_len(count) + 8;
    let mut absolute_offset = 0u64;
    for (i, entry) in entries.iter().enumerate() {
        if entry.offset_delta > MAX_FIELD
            || (i != 0 && entry.offset_delta == 0)
            || entry.duration > MAX_FIELD
            || entry.frame_count == 0
            || entry.frame_count > MAX_FIELD
        {
            return Err(Error::InvalidInput("invalid frame index entry".into()));
        }
        absolute_offset = absolute_offset
            .checked_add(entry.offset_delta)
            .filter(|&n| n <= MAX_FIELD)
            .ok_or(Error::Limit("frame index absolute offset"))?;
        length = length
            .checked_add(
                varint_len(entry.offset_delta)
                    + varint_len(entry.duration)
                    + varint_len(entry.frame_count),
            )
            .filter(|&n| n <= max_payload_bytes)
            .ok_or(Error::Limit("frame index payload bytes"))?;
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| Error::Limit("frame index payload allocation"))?;
    write_varint(count, &mut output);
    output.extend_from_slice(&ticks.numerator.to_be_bytes());
    output.extend_from_slice(&ticks.denominator.to_be_bytes());
    for entry in entries {
        write_varint(entry.offset_delta, &mut output);
        write_varint(entry.duration, &mut output);
        write_varint(entry.frame_count, &mut output);
    }
    debug_assert_eq!(output.len(), length);
    Ok(output)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RecordedFrame {
    offset: u64,
    duration: u32,
    requested: bool,
}

/// Literal frozen AddFrame/StoreFrameIndexBox/EncodeFrameIndexBox lifecycle.
/// This is source-compatibility machinery, NOT correct animation interval
/// aggregation. Use encode_payload with caller-computed entries for that.
#[derive(Debug)]
pub struct SourceFrameIndex {
    ticks: TickDuration,
    frames: Vec<RecordedFrame>,
    max_frames: usize,
}

impl SourceFrameIndex {
    pub fn new(ticks: TickDuration, max_frames: usize) -> Result<Self> {
        ticks.validate()?;
        Ok(Self {
            ticks,
            frames: Vec::new(),
            max_frames,
        })
    }

    /// Record every input frame as the frozen caller does. A second call at the
    /// first offset replaces the sole initial entry, including its request flag.
    /// Otherwise offsets must strictly increase. Failure leaves state unchanged.
    /// Keyframe eligibility cannot be inferred from this structural input.
    pub fn record_frame(&mut self, offset: u64, duration: u32, requested: bool) -> Result<()> {
        if offset > MAX_FIELD {
            return Err(Error::Limit("frame index codestream offset"));
        }
        let frame = RecordedFrame {
            offset,
            duration,
            requested,
        };
        if self.frames.len() == 1 && offset == self.frames[0].offset {
            self.frames[0] = frame;
            return Ok(());
        }
        if self.frames.last().is_some_and(|last| offset <= last.offset) {
            return Err(Error::InvalidInput(
                "frame index offsets must strictly increase".into(),
            ));
        }
        if self.frames.len() >= self.max_frames || self.frames.len() >= i32::MAX as usize {
            return Err(Error::Limit("frame index recorded frames"));
        }
        self.frames
            .try_reserve(1)
            .map_err(|_| Error::Limit("frame index record allocation"))?;
        self.frames.push(frame);
        Ok(())
    }

    pub fn is_requested(&self) -> bool {
        self.frames.iter().any(|frame| frame.requested)
    }

    /// None means the source StoreFrameIndexBox predicate is false, including
    /// an empty recorder. The first frame is implicitly included once requested.
    ///
    /// Frozen Ti uses T_prev, then assigns T_prev = T and adds ONLY the next
    /// selected frame's duration to T. It ignores the first/unselected durations
    /// and counts every recorded frame in Fi. These source defects are retained
    /// explicitly; signed-accumulator overflow is rejected instead of emulated.
    pub fn encode_source_payload(&self, max_payload_bytes: usize) -> Result<Option<Vec<u8>>> {
        if !self.is_requested() {
            return Ok(None);
        }
        let count = self
            .frames
            .iter()
            .enumerate()
            .filter(|(i, frame)| *i == 0 || frame.requested)
            .count();
        count
            .checked_mul(3)
            .and_then(|n| n.checked_add(varint_len(count as u64) + 8))
            .filter(|&n| n <= max_payload_bytes)
            .ok_or(Error::Limit("frame index payload bytes"))?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("frame index entry allocation"))?;
        let mut previous = 0usize;
        let mut previous_offset = 0u64;
        let mut previous_time = 0u32;
        let mut time = 0u32;
        for (i, frame) in self.frames.iter().enumerate().skip(1) {
            if !frame.requested {
                continue;
            }
            let offset = self.frames[previous].offset;
            entries.push(FrameIndexEntry {
                offset_delta: offset - previous_offset,
                duration: u64::from(previous_time),
                frame_count: (i - previous) as u64,
            });
            previous_offset = offset;
            previous = i;
            previous_time = time;
            time = time
                .checked_add(frame.duration)
                .filter(|&n| n <= i32::MAX as u32)
                .ok_or(Error::Limit(
                    "frozen frame index signed duration accumulator",
                ))?;
        }
        entries.push(FrameIndexEntry {
            offset_delta: self.frames[previous].offset - previous_offset,
            duration: u64::from(previous_time),
            frame_count: (self.frames.len() - previous) as u64,
        });
        encode_payload(self.ticks, &entries, max_payload_bytes).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(offset_delta: u64, duration: u64, frame_count: u64) -> FrameIndexEntry {
        FrameIndexEntry {
            offset_delta,
            duration,
            frame_count,
        }
    }

    fn read_varint(bytes: &[u8], position: &mut usize) -> u64 {
        let mut value = 0;
        for shift in (0..70).step_by(7) {
            let byte = bytes[*position];
            *position += 1;
            value |= u64::from(byte & 127) << shift;
            if byte < 128 {
                return value;
            }
        }
        panic!("unterminated fixture varint")
    }

    fn parse(bytes: &[u8]) -> (TickDuration, Vec<FrameIndexEntry>) {
        let mut position = 0;
        let count = read_varint(bytes, &mut position);
        let ticks = TickDuration {
            numerator: u32::from_be_bytes(bytes[position..position + 4].try_into().unwrap()),
            denominator: u32::from_be_bytes(bytes[position + 4..position + 8].try_into().unwrap()),
        };
        position += 8;
        let mut entries = Vec::new();
        for _ in 0..count {
            entries.push(entry(
                read_varint(bytes, &mut position),
                read_varint(bytes, &mut position),
                read_varint(bytes, &mut position),
            ));
        }
        assert_eq!(position, bytes.len());
        (ticks, entries)
    }

    #[test]
    fn literal_wire_golden_uses_big_endian_timebase_and_little_endian_varints() {
        let ticks = TickDuration {
            numerator: 1001,
            denominator: 24000,
        };
        let entries = [entry(5, 150, 2), entry(300, 7, 1)];
        let expected = [
            2, 0, 0, 3, 0xe9, 0, 0, 0x5d, 0xc0, 5, 0x96, 1, 2, 0xac, 2, 7, 1,
        ];
        assert_eq!(
            encode_payload(ticks, &entries, expected.len()).unwrap(),
            expected
        );
        assert_eq!(parse(&expected), (ticks, entries.to_vec()));
        assert!(encode_payload(ticks, &entries, expected.len() - 1).is_err());
        assert_eq!(BOX_TYPE, *b"jxli");
    }

    #[test]
    fn source_varint_goldens_include_all_seven_bit_boundaries() {
        for (value, expected) in [
            (0, vec![0]),
            (127, vec![127]),
            (128, vec![0x80, 1]),
            (16383, vec![0xff, 0x7f]),
            (16384, vec![0x80, 0x80, 1]),
            (
                MAX_FIELD,
                vec![0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f],
            ),
            (
                u64::MAX,
                vec![0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1],
            ),
        ] {
            let mut encoded = Vec::new();
            write_varint(value, &mut encoded);
            assert_eq!(encoded, expected);
            assert_eq!(varint_len(value), expected.len());
        }
        for bits in (7..=63).step_by(7) {
            for value in [(1u64 << bits) - 1, 1u64 << bits] {
                let mut encoded = Vec::new();
                write_varint(value, &mut encoded);
                let mut position = 0;
                assert_eq!(read_varint(&encoded, &mut position), value);
                assert_eq!(position, varint_len(value));
            }
        }
    }

    #[test]
    fn payload_rejects_empty_zero_counts_bad_timebase_and_overflow() {
        let ticks = TickDuration::default();
        assert!(encode_payload(ticks, &[], usize::MAX).is_err());
        assert!(
            encode_payload(
                TickDuration {
                    numerator: 1,
                    denominator: 0
                },
                &[entry(0, 0, 1)],
                32
            )
            .is_err()
        );
        for entries in [
            vec![entry(0, 0, 0)],
            vec![entry(MAX_FIELD + 1, 0, 1)],
            vec![entry(0, MAX_FIELD + 1, 1)],
            vec![entry(0, 0, MAX_FIELD + 1)],
            vec![entry(1, 0, 1), entry(0, 0, 1)],
            vec![entry(MAX_FIELD, 0, 1), entry(1, 0, 1)],
        ] {
            assert!(encode_payload(ticks, &entries, usize::MAX).is_err());
        }
        let entries = [entry(MAX_FIELD, MAX_FIELD, MAX_FIELD)];
        let bytes = encode_payload(ticks, &entries, 36).unwrap();
        assert_eq!(bytes.len(), 36);
        assert_eq!(parse(&bytes), (ticks, entries.to_vec()));
        assert!(encode_payload(ticks, &entries, 35).is_err());
        assert!(encode_payload(ticks, &[entry(0, 0, 1)], 0).is_err());
        let zero = TickDuration {
            numerator: 0,
            denominator: u32::MAX,
        };
        assert_eq!(
            parse(&encode_payload(zero, &[entry(0, 0, 1)], 12).unwrap()).0,
            zero
        );
    }

    #[test]
    fn lifecycle_replaces_only_the_initial_entry_and_keeps_absent_request_absent() {
        let mut source = SourceFrameIndex::new(TickDuration::default(), 3).unwrap();
        assert_eq!(source.encode_source_payload(0).unwrap(), None);
        source.record_frame(5, 10, false).unwrap();
        assert!(!source.is_requested());
        assert_eq!(source.encode_source_payload(0).unwrap(), None);
        source.record_frame(5, 20, true).unwrap();
        assert_eq!(source.frames.len(), 1);
        assert_eq!(source.frames[0].duration, 20);
        assert_eq!(
            source.encode_source_payload(12).unwrap().unwrap(),
            [1, 0, 0, 0, 1, 0, 0, 3, 0xe8, 5, 0, 1]
        );
        source.record_frame(5, 30, false).unwrap();
        assert!(!source.is_requested());
        source.record_frame(10, 40, true).unwrap();
        let bytes = source.encode_source_payload(15).unwrap().unwrap();
        assert_eq!(parse(&bytes).1, [entry(5, 0, 1), entry(5, 0, 1)]);
        assert!(source.record_frame(10, 0, true).is_err());
        assert!(source.record_frame(4, 0, true).is_err());
        assert!(source.record_frame(MAX_FIELD + 1, 0, true).is_err());
        assert_eq!(source.encode_source_payload(15).unwrap(), Some(bytes));
    }

    #[test]
    fn frozen_timing_is_lagged_and_ignores_unselected_durations() {
        let mut source = SourceFrameIndex::new(TickDuration::default(), 5).unwrap();
        for (offset, duration, requested) in [
            (10, 11, false),
            (20, 22, false),
            (30, 33, true),
            (40, 44, false),
            (50, 55, true),
        ] {
            source.record_frame(offset, duration, requested).unwrap();
        }
        let bytes = source.encode_source_payload(18).unwrap().unwrap();
        // Frozen T_prev outputs 0, 0, 33, NOT interval durations 33, 77, 55.
        assert_eq!(
            parse(&bytes).1,
            [entry(10, 0, 2), entry(20, 0, 2), entry(20, 33, 1)]
        );
        assert!(source.encode_source_payload(17).is_err());
        source.frames[0].duration = u32::MAX;
        source.frames[1].duration = u32::MAX;
        source.frames[3].duration = u32::MAX;
        assert_eq!(source.encode_source_payload(18).unwrap(), Some(bytes));
    }

    #[test]
    fn source_limits_and_signed_duration_overflow_are_checked() {
        assert!(
            SourceFrameIndex::new(
                TickDuration {
                    numerator: 1,
                    denominator: 0
                },
                3
            )
            .is_err()
        );
        let mut empty = SourceFrameIndex::new(TickDuration::default(), 0).unwrap();
        assert!(empty.record_frame(0, 0, true).is_err());
        let mut source = SourceFrameIndex::new(TickDuration::default(), 3).unwrap();
        source.record_frame(0, u32::MAX, true).unwrap(); // First duration is unused.
        source.record_frame(10, i32::MAX as u32, true).unwrap();
        assert!(source.encode_source_payload(32).is_ok());
        source.record_frame(20, 1, true).unwrap();
        let before = source.frames.clone();
        assert!(source.record_frame(30, 0, false).is_err());
        assert_eq!(source.frames, before);
        assert!(source.encode_source_payload(100).is_err());
        assert_eq!(source.frames, before);
    }

    #[test]
    fn raw_entries_allow_correct_timeline_and_display_frame_accounting() {
        let ticks = TickDuration {
            numerator: 1001,
            denominator: 30000,
        };
        // Caller has aggregated 11+22 and 33+44 ticks, independently of the
        // frozen source bug. A nondisplayed composition frame contributes no Fi.
        let entries = [entry(10, 33, 2), entry(20, 77, 1), entry(20, 55, 1)];
        assert_eq!(
            parse(&encode_payload(ticks, &entries, 64).unwrap()),
            (ticks, entries.to_vec())
        );
    }
}
