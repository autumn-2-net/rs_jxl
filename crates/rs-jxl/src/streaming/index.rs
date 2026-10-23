//! Safe timeline/offset integration for the source jxli wire format.
use crate::{
    Error, Image, Result,
    container::frame_index::{self, FrameIndexEntry, TickDuration},
    frame::{BlendMode, Frame},
};

pub(super) struct IndexPlan {
    selected: Vec<usize>,
    entries: Vec<FrameIndexEntry>,
    ticks: TickDuration,
    next: usize,
    previous_offset: u64,
}

impl IndexPlan {
    pub(super) fn new(canvas: &Image, frames: &[Frame<'_>], selected: &[usize]) -> Result<Self> {
        if selected.first() != Some(&0)
            || selected.last().is_none_or(|&i| i >= frames.len())
            || selected.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(Error::InvalidInput("frame index selections must start at zero and strictly increase within the sequence".into()));
        }
        let mut choices = Vec::new();
        choices
            .try_reserve_exact(selected.len())
            .map_err(|_| Error::Limit("frame index selections"))?;
        choices.extend_from_slice(selected);
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(selected.len())
            .map_err(|_| Error::Limit("frame index entries"))?;
        let mut next = 0;
        let mut available = [false; 4];
        for (i, frame) in frames.iter().enumerate() {
            let settings = &frame.options;
            let displayed =
                !settings.reference_only && (settings.duration != 0 || i + 1 == frames.len());
            if selected.get(next) == Some(&i) {
                if !displayed
                    || settings.partial(canvas, frame.image)
                    || std::iter::once(settings.blend)
                        .chain(
                            (0..canvas.metadata().extra_channels.len()).map(|e| settings.extra(e)),
                        )
                        .any(|blend| blend.mode != BlendMode::Replace)
                {
                    return Err(Error::InvalidInput(
                        "indexed frames must independently replace and display the complete canvas"
                            .into(),
                    ));
                }
                available = [false; 4];
                entries.push(FrameIndexEntry {
                    offset_delta: 0,
                    duration: 0,
                    frame_count: 0,
                });
                next += 1;
            }
            // A keyframe must also start an independently decodable interval:
            // later frames cannot reach back into references from the old interval.
            if !settings.reference_only {
                for blend in std::iter::once(settings.blend)
                    .chain((0..canvas.metadata().extra_channels.len()).map(|e| settings.extra(e)))
                {
                    if (settings.partial(canvas, frame.image) || blend.mode != BlendMode::Replace)
                        && !available[usize::from(blend.source)]
                    {
                        return Err(Error::InvalidInput(
                            "frame index interval depends on an earlier reference".into(),
                        ));
                    }
                }
            }
            if i + 1 < frames.len()
                && (settings.reference_only
                    || settings.duration == 0
                    || settings.save_as_reference != 0)
            {
                available[usize::from(settings.save_as_reference)] = true;
            }
            let current = entries.last_mut().expect("validated first selection");
            current.duration = current
                .duration
                .checked_add(u64::from(settings.duration))
                .filter(|&n| n <= i64::MAX as u64)
                .ok_or(Error::Limit("frame index duration"))?;
            current.frame_count = current
                .frame_count
                .checked_add(u64::from(displayed))
                .ok_or(Error::Limit("frame index display count"))?;
        }
        let ticks = canvas
            .metadata()
            .animation
            .map_or(TickDuration::default(), |a| TickDuration {
                numerator: a.ticks_per_second_denominator,
                denominator: a.ticks_per_second_numerator,
            });
        Ok(Self {
            selected: choices,
            entries,
            ticks,
            next: 0,
            previous_offset: 0,
        })
    }

    pub(super) fn record(&mut self, frame: usize, offset: u64) -> Result<()> {
        if self.selected.get(self.next) == Some(&frame) {
            self.entries[self.next].offset_delta = offset
                .checked_sub(self.previous_offset)
                .ok_or(Error::Limit("frame index offset ordering"))?;
            self.previous_offset = offset;
            self.next += 1;
        }
        Ok(())
    }

    pub(super) fn payload(&self, maximum: usize) -> Result<Vec<u8>> {
        if self.next != self.selected.len() {
            return Err(Error::Encoding("frame index is incomplete".into()));
        }
        frame_index::encode_payload(self.ticks, &self.entries, maximum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout, frame::FrameOptions, metadata::Animation};

    fn image() -> Image {
        let image = Image::new(2, 2, PixelLayout::Gray, PixelData::U8(vec![5; 4])).unwrap();
        let mut metadata = image.metadata().clone();
        metadata.animation = Some(Animation {
            ticks_per_second_numerator: 24,
            ..Default::default()
        });
        image.with_metadata(metadata, vec![]).unwrap()
    }

    #[test]
    fn intervals_count_displays_and_use_actual_timebase_and_offsets() {
        let image = image();
        let frames: Vec<_> = [11, 22, 33, 44, 0]
            .into_iter()
            .map(|duration| Frame {
                image: &image,
                options: FrameOptions {
                    duration,
                    ..Default::default()
                },
            })
            .collect();
        let mut index = IndexPlan::new(&image, &frames, &[0, 2, 4]).unwrap();
        for (i, offset) in [17, 91, 150, 390, 700].into_iter().enumerate() {
            index.record(i, offset).unwrap();
        }
        assert_eq!(
            index.ticks,
            TickDuration {
                numerator: 1,
                denominator: 24
            }
        );
        assert_eq!(
            index.entries,
            [
                FrameIndexEntry {
                    offset_delta: 17,
                    duration: 33,
                    frame_count: 2
                },
                FrameIndexEntry {
                    offset_delta: 133,
                    duration: 77,
                    frame_count: 2
                },
                FrameIndexEntry {
                    offset_delta: 550,
                    duration: 0,
                    frame_count: 1
                },
            ]
        );
        assert!(index.payload(100).is_ok());
        assert!(index.payload(8).is_err());
    }

    #[test]
    fn rejects_later_reference_crossing_index_boundary_before_output() {
        let image = image();
        let mut frames: Vec<_> = (0..3)
            .map(|_| Frame {
                image: &image,
                options: FrameOptions {
                    duration: 1,
                    ..Default::default()
                },
            })
            .collect();
        frames[0].options.save_as_reference = 1;
        frames[2].options.blend.mode = BlendMode::Add;
        frames[2].options.blend.source = 1;
        assert!(IndexPlan::new(&image, &frames, &[0]).is_ok());
        assert!(IndexPlan::new(&image, &frames, &[0, 1]).is_err());
        frames[1].options.save_as_reference = 1;
        assert!(IndexPlan::new(&image, &frames, &[0, 1]).is_ok());
        for selections in [&[][..], &[1], &[0, 0], &[0, 3], &[0, 2]] {
            assert!(IndexPlan::new(&image, &frames, selections).is_err());
        }
    }
}
