//! Incremental equivalent of streaming::index, retaining selected entries only.
use super::FrameIndexOptions;
use crate::container::frame_index::{self, FrameIndexEntry, TickDuration};
use crate::frame::{BlendMode, FrameOptions, Geometry};
use crate::{EncodeOptions, Error, Result};

pub(super) struct IndexState {
    config: FrameIndexOptions,
    ticks: TickDuration,
    entries: Vec<FrameIndexEntry>,
    previous_offset: u64,
    references: [bool; 4],
    payload_bytes: usize,
}

pub(super) struct IndexUpdate {
    selected: bool,
    entry: FrameIndexEntry,
    offset: u64,
    references: [bool; 4],
    pub payload_bytes: usize,
}

impl IndexState {
    pub fn new(config: FrameIndexOptions, canvas: Geometry<'_>) -> Result<Self> {
        if config.max_entries == 0 || config.max_payload_bytes < 12 {
            return Err(Error::InvalidInput(
                "frame index limits must allow at least one entry".into(),
            ));
        }
        let ticks = canvas
            .metadata
            .animation
            .map_or(TickDuration::default(), |a| TickDuration {
                numerator: a.ticks_per_second_denominator,
                denominator: a.ticks_per_second_numerator,
            });
        Ok(Self {
            config,
            ticks,
            entries: Vec::new(),
            previous_offset: 0,
            references: [false; 4],
            payload_bytes: 9,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        canvas: Geometry<'_>,
        image: Geometry<'_>,
        frame: &FrameOptions,
        options: &EncodeOptions,
        last: bool,
        selected: bool,
        offset: u64,
    ) -> Result<IndexUpdate> {
        if options.patches.is_some() || options.source_patches.is_some() {
            return Err(Error::InvalidInput(
                "indexed output does not support generated patch reference frames".into(),
            ));
        }
        if self.entries.is_empty() && !selected {
            return Err(Error::InvalidInput(
                "indexed output must select its first frame".into(),
            ));
        }
        let displayed = !frame.reference_only && (frame.duration != 0 || last);
        let partial = frame.partial_geometry(canvas, image);
        let mut references = self.references;
        if selected {
            if !displayed
                || partial
                || std::iter::once(frame.blend)
                    .chain((0..canvas.metadata.extra_channels.len()).map(|e| frame.extra(e)))
                    .any(|blend| blend.mode != BlendMode::Replace)
            {
                return Err(Error::InvalidInput(
                    "indexed frames must independently replace and display the complete canvas"
                        .into(),
                ));
            }
            if self.entries.len() >= self.config.max_entries {
                return Err(Error::Limit("frame index entries"));
            }
            references = [false; 4];
        }
        if !frame.reference_only {
            for blend in std::iter::once(frame.blend)
                .chain((0..canvas.metadata.extra_channels.len()).map(|e| frame.extra(e)))
            {
                if (partial || blend.mode != BlendMode::Replace)
                    && !references[usize::from(blend.source)]
                {
                    return Err(Error::InvalidInput(
                        "frame index interval depends on an earlier reference".into(),
                    ));
                }
            }
        }
        if !last && (frame.reference_only || frame.duration == 0 || frame.save_as_reference != 0) {
            references[usize::from(frame.save_as_reference)] = true;
        }
        if offset > i64::MAX as u64 {
            return Err(Error::Limit("frame index offset"));
        }
        let mut entry = if selected {
            let delta = offset
                .checked_sub(self.previous_offset)
                .filter(|&delta| self.entries.is_empty() || delta != 0)
                .ok_or(Error::Limit("frame index offset ordering"))?;
            FrameIndexEntry {
                offset_delta: delta,
                duration: 0,
                frame_count: 0,
            }
        } else {
            *self.entries.last().expect("validated first selection")
        };
        entry.duration = entry
            .duration
            .checked_add(u64::from(frame.duration))
            .filter(|&n| n <= i64::MAX as u64)
            .ok_or(Error::Limit("frame index duration"))?;
        entry.frame_count = entry
            .frame_count
            .checked_add(u64::from(displayed))
            .filter(|&n| n <= i64::MAX as u64)
            .ok_or(Error::Limit("frame index display count"))?;
        let payload_bytes = if selected {
            self.payload_bytes
                .checked_add(entry_len(entry))
                .and_then(|n| {
                    n.checked_add(
                        varint_len(self.entries.len() as u64 + 1)
                            - varint_len(self.entries.len() as u64),
                    )
                })
        } else {
            self.payload_bytes
                .checked_sub(entry_len(*self.entries.last().unwrap()))
                .and_then(|n| n.checked_add(entry_len(entry)))
        }
        .filter(|&n| n <= self.config.max_payload_bytes)
        .ok_or(Error::Limit("frame index payload bytes"))?;
        if selected {
            self.entries
                .try_reserve_exact(1)
                .map_err(|_| Error::Limit("frame index entries"))?;
        }
        Ok(IndexUpdate {
            selected,
            entry,
            offset,
            references,
            payload_bytes,
        })
    }

    pub fn commit(&mut self, update: IndexUpdate) {
        if update.selected {
            self.entries.push(update.entry);
            self.previous_offset = update.offset;
        } else {
            *self.entries.last_mut().expect("validated index") = update.entry;
        }
        self.references = update.references;
        self.payload_bytes = update.payload_bytes;
    }

    pub fn payload(&self) -> Result<Vec<u8>> {
        frame_index::encode_payload(self.ticks, &self.entries, self.config.max_payload_bytes)
    }

    pub fn reserved_bytes(&self) -> Result<usize> {
        self.payload_bytes
            .checked_add(16)
            .ok_or(Error::Limit("frame index box bytes"))
    }
}

fn varint_len(mut value: u64) -> usize {
    let mut length = 1;
    while value >= 128 {
        value >>= 7;
        length += 1;
    }
    length
}
fn entry_len(entry: FrameIndexEntry) -> usize {
    varint_len(entry.offset_delta) + varint_len(entry.duration) + varint_len(entry.frame_count)
}
