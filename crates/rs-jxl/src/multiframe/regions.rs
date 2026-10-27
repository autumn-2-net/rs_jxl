//! Bounded exact change regions; no color or alpha approximation.
use super::{
    Rect,
    diff::{DiffMap, mask_alpha_max},
};
use crate::{Error, Image, PixelData, Result};

/// Preserve the legacy raster/equal-span candidate, including its early cap.
pub(super) fn tiles_from_map(
    map: &DiffMap,
    tile: u32,
    maximum: usize,
) -> Result<Option<Vec<Rect>>> {
    let mut boxes: Vec<Rect> = Vec::new();
    for next in map.occupied_tiles(tile)? {
        if let Some(previous) = boxes.iter_mut().rev().find(|r| {
            (r.x == next.x && r.width == next.width && r.y + r.height == next.y)
                || (r.y == next.y && r.height == next.height && r.x + r.width == next.x)
        }) {
            previous.width = (next.x + next.width).max(previous.x + previous.width) - previous.x;
            previous.height = (next.y + next.height).max(previous.y + previous.height) - previous.y;
        } else {
            if boxes.len() == maximum {
                return Ok(None);
            }
            boxes
                .try_reserve(1)
                .map_err(|_| Error::Limit("difference region allocation"))?;
            boxes.push(next);
        }
    }
    Ok(Some(boxes))
}

/// `current` must be the current input used to construct `map`. The shared map
/// already checked both opacity planes; no reference pixels are read here.
pub(super) fn mask_from_map(current: &Image, rect: Rect, map: &DiffMap) -> Result<Image> {
    if map.width() != current.width()
        || map.height() != current.height()
        || !map.mask_eligible()
        || mask_alpha_max(current).is_none()
    {
        return Err(Error::InvalidInput(
            "mask requires matching opaque integer GA/RGBA inputs".into(),
        ));
    }
    if rect.width == 0
        || rect.height == 0
        || rect
            .x
            .checked_add(rect.width)
            .is_none_or(|x| x > current.width())
        || rect
            .y
            .checked_add(rect.height)
            .is_none_or(|y| y > current.height())
    {
        return Err(Error::InvalidInput("mask crop is outside image".into()));
    }
    let channels = current.layout().channels();
    let count = (rect.width as usize)
        .checked_mul(rect.height as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or(Error::Limit("mask sample count"))?;
    macro_rules! masked {
        ($values:expr, $variant:ident) => {{
            let mut pixels = Vec::new();
            pixels
                .try_reserve_exact(count)
                .map_err(|_| Error::Limit("mask allocation"))?;
            pixels.resize(count, 0);
            for y in 0..rect.height {
                let source =
                    ((y + rect.y) as usize * current.width() as usize + rect.x as usize) * channels;
                let target = y as usize * rect.width as usize * channels;
                for x in 0..rect.width {
                    if map.changed(x + rect.x, y + rect.y) {
                        let offset = x as usize * channels;
                        pixels[target + offset..target + offset + channels]
                            .copy_from_slice(&$values[source + offset..source + offset + channels]);
                    }
                }
            }
            PixelData::$variant(pixels)
        }};
    }
    let pixels = match current.pixels() {
        PixelData::U8(values) => masked!(values, U8),
        PixelData::U16(values) => masked!(values, U16),
        _ => unreachable!("mask structure checked"),
    };
    Image::new(rect.width, rect.height, current.layout(), pixels)?
        .with_metadata(current.metadata().clone(), Vec::new())
}

#[cfg(test)]
use super::crop;

#[cfg(test)]
pub(super) fn changed(a: &Image, b: &Image, x: u32, y: u32) -> bool {
    let pixel = y as usize * a.width() as usize + x as usize;
    (0..a.channel_count()).any(|c| a.modular_sample(c, pixel) != b.modular_sample(c, pixel))
}

/// Tight boxes per occupied tile, merging only directly adjacent equal spans.
/// Stop at the configured bound instead of retaining an image-sized region list.
#[cfg(test)]
fn tiles(a: &Image, b: &Image, tile: u32, maximum: usize) -> Option<Vec<Rect>> {
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
#[cfg(test)]
fn mask_eligible(image: &Image) -> bool {
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

#[cfg(test)]
fn mask(a: &Image, b: &Image, rect: Rect) -> Result<Image> {
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

    #[test]
    fn map_tiles_match_legacy_order_merges_and_early_caps() {
        let a = Image::new(3, 3, PixelLayout::Gray, PixelData::U8(vec![0; 9])).unwrap();
        for pattern in 0..1u16 << 9 {
            let pixels = (0..9).map(|i| ((pattern >> i) & 1) as u8).collect();
            let b = Image::new(3, 3, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
            let map = DiffMap::between(&a, &b).unwrap();
            for tile in [1, 2, 3, 4, u32::MAX] {
                for maximum in 0..=9 {
                    assert_eq!(
                        tiles_from_map(&map, tile, maximum).unwrap(),
                        tiles(&a, &b, tile, maximum),
                        "pattern {pattern}, tile {tile}, maximum {maximum}"
                    );
                }
            }
        }
        let map = DiffMap::between(&a, &a).unwrap();
        assert_eq!(tiles_from_map(&map, 1, 0).unwrap(), Some(vec![]));
        assert!(tiles_from_map(&map, 0, 1).is_err());
    }

    #[test]
    fn map_tiles_match_legacy_for_odd_multibyte_images_and_extra_planes() {
        use crate::metadata::{ExtraChannelInfo, ExtraChannelType};
        let base = Image::new(67, 5, PixelLayout::Gray, PixelData::U16(vec![0; 335])).unwrap();
        let mut metadata = base.metadata().clone();
        metadata
            .extra_channels
            .push(ExtraChannelInfo::new(ExtraChannelType::Depth, 16).unwrap());
        let a = base
            .clone()
            .with_metadata(metadata.clone(), vec![PixelData::U16(vec![0; 335])])
            .unwrap();
        let values = (0..335)
            .map(|i| u16::from(i % 13 == 0 || i % 31 == 0))
            .collect();
        let b = base
            .with_metadata(metadata, vec![PixelData::U16(values)])
            .unwrap();
        let map = DiffMap::between(&a, &b).unwrap();
        for tile in [1, 2, 3, 8, 32, 64, u32::MAX] {
            for maximum in [0, 1, 2, 4, 16, 128, usize::MAX] {
                assert_eq!(
                    tiles_from_map(&map, tile, maximum).unwrap(),
                    tiles(&a, &b, tile, maximum)
                );
            }
        }
    }

    #[test]
    fn direct_masks_match_legacy_for_opaque_ga_rgba_and_reduced_depths() {
        use crate::metadata::BitDepth;
        for layout in [PixelLayout::GrayAlpha, PixelLayout::Rgba] {
            for wide in [false, true] {
                let channels = layout.channels();
                let values = |changed: bool| {
                    let mut pixels = vec![0u16; 35 * channels];
                    for (i, pixel) in pixels.chunks_exact_mut(channels).enumerate() {
                        pixel[0] = if changed && i % 3 == 1 { 9 } else { 7 };
                        pixel[channels - 1] = if wide { 1023 } else { 255 };
                    }
                    if wide {
                        PixelData::U16(pixels)
                    } else {
                        PixelData::U8(pixels.into_iter().map(|p| p as u8).collect())
                    }
                };
                let image = |changed| {
                    let image = Image::new(7, 5, layout, values(changed)).unwrap();
                    if wide {
                        let mut metadata = image.metadata().clone();
                        metadata.bit_depth = BitDepth::new(12).unwrap();
                        metadata.extra_channels[0].bit_depth = BitDepth::new(10).unwrap();
                        image.with_metadata(metadata, vec![]).unwrap()
                    } else {
                        image
                    }
                };
                let a = image(false);
                let b = image(true);
                assert!(mask_eligible(&a) && mask_eligible(&b));
                for current in [&a, &b] {
                    let map = DiffMap::between(&a, current).unwrap();
                    assert!(map.mask_eligible());
                    for rect in [
                        Rect {
                            x: 0,
                            y: 0,
                            width: 7,
                            height: 5,
                        },
                        Rect {
                            x: 1,
                            y: 1,
                            width: 5,
                            height: 3,
                        },
                        Rect {
                            x: 6,
                            y: 4,
                            width: 1,
                            height: 1,
                        },
                    ] {
                        assert_eq!(
                            mask_from_map(current, rect, &map).unwrap(),
                            mask(&a, current, rect).unwrap()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn direct_masks_reject_invalid_geometry_and_ineligible_inputs() {
        let a = Image::new(2, 1, PixelLayout::Rgba, PixelData::U8(vec![255; 8])).unwrap();
        let map = DiffMap::between(&a, &a).unwrap();
        for rect in [
            Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 1,
            },
            Rect {
                x: u32::MAX,
                y: 0,
                width: 2,
                height: 1,
            },
            Rect {
                x: 0,
                y: 1,
                width: 2,
                height: 1,
            },
        ] {
            assert!(mask_from_map(&a, rect, &map).is_err());
        }
        let rect = Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        let wrong = Image::new(1, 1, PixelLayout::Rgba, PixelData::U8(vec![255; 4])).unwrap();
        assert!(mask_from_map(&wrong, rect, &map).is_err());
        let partial = Image::new(2, 1, PixelLayout::Rgba, PixelData::U8(vec![254; 8])).unwrap();
        let partial_map = DiffMap::between(&a, &partial).unwrap();
        assert!(mask_from_map(&partial, rect, &partial_map).is_err());
        let rgb = Image::new(2, 1, PixelLayout::Rgb, PixelData::U8(vec![255; 6])).unwrap();
        let rgb_map = DiffMap::between(&rgb, &rgb).unwrap();
        assert!(!mask_eligible(&rgb));
        assert!(mask_from_map(&rgb, rect, &rgb_map).is_err());
    }
}
