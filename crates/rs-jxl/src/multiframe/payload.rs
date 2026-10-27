//! Share repeated unmasked crops only within one logical input's candidate set.
use super::{Rect, candidates::Candidate};
use crate::{
    EncodeOptions, Image, Result,
    encode::{
        self, LogicalFrameCache,
        budget::{SectionBudget, TrialEncode},
    },
    frame::Frame,
};
use std::collections::BTreeMap;

type Key = (u32, u32, u32, u32);

#[derive(Default)]
pub(super) struct Payloads {
    entries: BTreeMap<Key, (usize, LogicalFrameCache<()>)>,
    pub(super) hits: usize,
    pub(super) pixels: u64,
}

impl Payloads {
    pub(super) fn new<'a>(candidates: impl Iterator<Item = &'a Candidate>) -> Self {
        let mut result = Self::default();
        for candidate in candidates.filter(|c| !c.mask) {
            for &rect in &candidate.rects {
                result.entries.entry(key(rect)).or_default().0 += 1;
            }
        }
        result.entries.retain(|_, (uses, _)| *uses > 1);
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &mut self,
        rect: Rect,
        canvas: &Image,
        frame: &Frame<'_>,
        options: &EncodeOptions,
        last: bool,
        budget: Option<&SectionBudget>,
    ) -> Result<TrialEncode<Vec<u8>>> {
        let Some((uses, cache)) = self.entries.get_mut(&key(rect)) else {
            return encode::encode_logical_frame_trial(canvas, frame, options, last, budget);
        };
        if cache.contains(&()) {
            self.hits += 1;
            self.pixels += u64::from(rect.width) * u64::from(rect.height);
        }
        let output = encode::encode_logical_frame_cached_trial(
            canvas,
            frame,
            options,
            last,
            budget,
            cache,
            (),
        )?;
        *uses -= 1;
        if *uses == 0 {
            self.entries.remove(&key(rect));
        }
        Ok(output)
    }

    pub(super) fn skip(&mut self, rects: &[Rect]) {
        for &rect in rects {
            if let Some((uses, _)) = self.entries.get_mut(&key(rect)) {
                *uses -= 1;
                if *uses == 0 {
                    self.entries.remove(&key(rect));
                }
            }
        }
    }
}

fn key(rect: Rect) -> Key {
    (rect.x, rect.y, rect.width, rect.height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PixelData, PixelLayout, codestream, decode::decode, frame::FrameOptions};

    #[test]
    fn pruning_releases_unvisited_cache_uses() {
        let rect = Rect {
            x: 1,
            y: 2,
            width: 3,
            height: 4,
        };
        let mut payloads = Payloads::default();
        payloads
            .entries
            .insert(key(rect), (2, LogicalFrameCache::default()));
        payloads.skip(&[rect]);
        assert_eq!(payloads.entries[&key(rect)].0, 1);
        payloads.skip(&[rect]);
        assert!(payloads.entries.is_empty());
        payloads.skip(&[rect]);
        assert_eq!((payloads.hits, payloads.pixels), (0, 0));
    }

    #[test]
    fn populated_caches_release_on_interleaved_skips_and_last_hit() {
        let image = Image::new(
            8,
            4,
            PixelLayout::Gray,
            PixelData::U8((0..32u8).map(|i| i.wrapping_mul(17)).collect()),
        )
        .unwrap();
        let options = EncodeOptions::default();
        let a = Rect {
            x: 0,
            y: 0,
            width: 3,
            height: 2,
        };
        let b = Rect { x: 4, y: 2, ..a };
        let mut payloads = Payloads::default();
        for rect in [a, b] {
            payloads
                .entries
                .insert(key(rect), (3, LogicalFrameCache::default()));
        }
        let settings = |rect: Rect, last| FrameOptions {
            origin: (rect.x as i32, rect.y as i32),
            save_as_reference: if last { 0 } else { 1 },
            blend: crate::frame::BlendInfo {
                source: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let initial = Frame {
            image: &image,
            options: FrameOptions {
                save_as_reference: 1,
                ..Default::default()
            },
        };
        let mut stream = codestream::image_header(&image, options.max_output_bytes).unwrap();
        stream.extend(
            encode::encode_logical_frame_trial(&image, &initial, &options, false, None)
                .unwrap()
                .into_complete()
                .unwrap(),
        );
        for rect in [a, b] {
            let crop = crate::multiframe::crop(&image, rect).unwrap();
            let frame = Frame {
                image: &crop,
                options: settings(rect, false),
            };
            stream.extend(
                payloads
                    .encode(rect, &image, &frame, &options, false, None)
                    .unwrap()
                    .into_complete()
                    .unwrap(),
            );
            assert_eq!(payloads.entries[&key(rect)].0, 2);
            assert!(payloads.entries[&key(rect)].1.contains(&()));
        }
        assert_eq!((payloads.hits, payloads.pixels), (0, 0));
        payloads.skip(&[a]);
        assert_eq!(payloads.entries[&key(a)].0, 1);
        assert!(payloads.entries[&key(a)].1.contains(&()));
        for last in [false, true] {
            let crop = crate::multiframe::crop(&image, b).unwrap();
            let frame = Frame {
                image: &crop,
                options: settings(b, last),
            };
            assert!(payloads.entries[&key(b)].1.contains(&()));
            let hit = payloads
                .encode(b, &image, &frame, &options, last, None)
                .unwrap()
                .into_complete()
                .unwrap();
            let ordinary = encode::encode_logical_frame_trial(&image, &frame, &options, last, None)
                .unwrap()
                .into_complete()
                .unwrap();
            assert_eq!(hit.len(), ordinary.len());
            stream.extend(hit);
            if !last {
                assert_eq!(payloads.entries[&key(b)].0, 1);
                payloads.skip(&[a]);
                assert!(!payloads.entries.contains_key(&key(a)));
                assert_eq!(payloads.entries.len(), 1);
                assert!(payloads.entries[&key(b)].1.contains(&()));
            }
        }
        assert!(payloads.entries.is_empty());
        assert_eq!((payloads.hits, payloads.pixels), (2, 12));
        payloads.skip(&[a, b]);
        assert_eq!((payloads.hits, payloads.pixels), (2, 12));
        let bytes = encode::wrap_stream(&image, stream, options.max_output_bytes).unwrap();
        let decoded = decode(&bytes, &Default::default()).unwrap();
        assert!(decoded.complete);
        assert_eq!(decoded.frames.len(), 1);
        assert_eq!(&decoded.frames[0].pixels, image.pixels());
    }
}
