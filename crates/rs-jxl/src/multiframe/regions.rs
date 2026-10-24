//! Bounded exact change regions; no color or alpha approximation.
use super::{Rect, crop};
use crate::{Image, PixelData, Result};

pub(super) fn changed(a: &Image, b: &Image, x: u32, y: u32) -> bool {
    let pixel = y as usize * a.width() as usize + x as usize;
    (0..a.channel_count()).any(|c| a.modular_sample(c, pixel) != b.modular_sample(c, pixel))
}

/// Tight boxes per occupied tile, merging only directly adjacent equal spans.
/// Stop at the configured bound instead of retaining an image-sized region list.
pub(super) fn tiles(a: &Image, b: &Image, tile: u32, maximum: usize) -> Option<Vec<Rect>> {
    let mut boxes: Vec<Rect> = Vec::new();
    for ty in (0..a.height()).step_by(tile as usize) {
        for tx in (0..a.width()).step_by(tile as usize) {
            let (mut left, mut top, mut right, mut bottom) = (a.width(), a.height(), 0, 0);
            for y in ty..ty.saturating_add(tile).min(a.height()) {
                for x in tx..tx.saturating_add(tile).min(a.width()) {
                    if changed(a, b, x, y) {
                        left = left.min(x);
                        top = top.min(y);
                        right = right.max(x + 1);
                        bottom = bottom.max(y + 1);
                    }
                }
            }
            if right == 0 {
                continue;
            }
            let next = Rect {
                x: left,
                y: top,
                width: right - left,
                height: bottom - top,
            };
            if let Some(previous) = boxes.iter_mut().rev().find(|r| {
                (r.x == next.x && r.width == next.width && r.y + r.height == next.y)
                    || (r.y == next.y && r.height == next.height && r.x + r.width == next.x)
            }) {
                previous.width =
                    (next.x + next.width).max(previous.x + previous.width) - previous.x;
                previous.height =
                    (next.y + next.height).max(previous.y + previous.height) - previous.y;
            } else {
                if boxes.len() == maximum {
                    return None;
                }
                boxes.push(next);
            }
        }
    }
    Some(boxes)
}

/// A binary opacity mask is exact only for already-opaque integer GA/RGBA.
/// Never add an alpha channel, reinterpret partial alpha or drop hidden samples.
pub(super) fn mask_eligible(image: &Image) -> bool {
    if !image.layout().has_alpha() || image.metadata().extra_channels.len() != 1 {
        return false;
    }
    let alpha = &image.metadata().extra_channels[0];
    if alpha.alpha_associated
        || alpha.dim_shift != 0
        || alpha.bit_depth.is_float()
        || alpha.bit_depth.bits_per_sample > 16
        || !matches!(image.pixels(), PixelData::U8(_) | PixelData::U16(_))
    {
        return false;
    }
    let channels = image.layout().channels();
    let maximum = ((1u32 << alpha.bit_depth.bits_per_sample) - 1) as i32;
    (0..image.width() as usize * image.height() as usize)
        .all(|p| image.modular_sample(channels - 1, p) == maximum)
}

pub(super) fn mask(a: &Image, b: &Image, rect: Rect) -> Result<Image> {
    let part = crop(b, rect)?;
    let channels = b.layout().channels();
    let mut pixels = part.pixels().clone();
    for y in 0..rect.height {
        for x in 0..rect.width {
            if !changed(a, b, x + rect.x, y + rect.y) {
                let start = (y as usize * rect.width as usize + x as usize) * channels;
                match &mut pixels {
                    PixelData::U8(v) => v[start..start + channels].fill(0),
                    PixelData::U16(v) => v[start..start + channels].fill(0),
                    _ => unreachable!("mask eligibility checked"),
                }
            }
        }
    }
    Image::new(rect.width, rect.height, b.layout(), pixels)?
        .with_metadata(b.metadata().clone(), Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelLayout;

    #[test]
    fn bounded_tiles_cover_changes_without_overlapping() {
        let a = Image::new(33, 25, PixelLayout::Gray, PixelData::U8(vec![0; 33 * 25])).unwrap();
        let mut samples = vec![0; 33 * 25];
        for (i, value) in samples.iter_mut().enumerate() {
            if i % 29 == 0 || i % 31 == 0 {
                *value = 1;
            }
        }
        let b = Image::new(33, 25, PixelLayout::Gray, PixelData::U8(samples)).unwrap();
        for size in [1, 8, 32, 128] {
            let boxes = tiles(&a, &b, size, 4096).unwrap();
            for y in 0..25 {
                for x in 0..33 {
                    let count = boxes
                        .iter()
                        .filter(|r| x >= r.x && y >= r.y && x < r.x + r.width && y < r.y + r.height)
                        .count();
                    assert!(count <= 1);
                    if changed(&a, &b, x, y) {
                        assert_eq!(count, 1);
                    }
                }
            }
        }
        assert!(tiles(&a, &b, 1, 1).is_none());
    }
}
