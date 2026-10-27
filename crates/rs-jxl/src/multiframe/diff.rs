//! Exact shared change coverage in canvas coordinates, including extra planes.
use super::{Rect, compatible};
use crate::{Error, Image, PixelData, Result, metadata::ExtraChannelType};
use std::sync::atomic::{AtomicBool, Ordering};

const WORD_BITS: usize = u64::BITS as usize;
const BLOCK_WORDS: usize = 1024;
const PARALLEL_PIXELS: usize = 256 * 1024;

pub(super) struct DiffMap {
    width: u32,
    height: u32,
    words: Vec<u64>,
    bounds: Option<Rect>,
    changed_pixels: u64,
    mask_eligible: bool,
}

impl DiffMap {
    pub(super) fn between(a: &Image, b: &Image) -> Result<Self> {
        compatible(a, b)?;
        let count = (a.width() as usize)
            .checked_mul(a.height() as usize)
            .ok_or(Error::Limit("difference pixel count"))?;
        let mut words = Vec::new();
        let word_count = count.div_ceil(WORD_BITS);
        words
            .try_reserve_exact(word_count)
            .map_err(|_| Error::Limit("difference map allocation"))?;
        words.resize(word_count, 0);
        let alpha = mask_alpha_max(a).zip(mask_alpha_max(b)).map(|(a, b)| {
            debug_assert_eq!(a, b);
            a
        });
        let opaque = scan_main(a, b, alpha, count, &mut words)?;
        let mut map = Self {
            width: a.width(),
            height: a.height(),
            words,
            bounds: None,
            changed_pixels: 0,
            mask_eligible: alpha.is_some() && opaque,
        };
        for channel in a.layout().channels()..a.channel_count() {
            map.scan_extra(a, b, channel);
        }
        map.changed_pixels = map
            .words
            .iter()
            .map(|word| u64::from(word.count_ones()))
            .sum();
        if map.changed_pixels != 0 {
            let mut bounds = map
                .tight_bounds(0, 0, map.width, map.height)
                .expect("nonempty difference map");
            let shift = a
                .metadata()
                .extra_channels
                .iter()
                .map(|info| info.dim_shift)
                .max()
                .unwrap_or(0);
            let align = 1u64 << shift;
            let right = (u64::from(bounds.x) + u64::from(bounds.width))
                .div_ceil(align)
                .saturating_mul(align)
                .min(u64::from(map.width));
            let bottom = (u64::from(bounds.y) + u64::from(bounds.height))
                .div_ceil(align)
                .saturating_mul(align)
                .min(u64::from(map.height));
            bounds.x = (u64::from(bounds.x) / align * align) as u32;
            bounds.y = (u64::from(bounds.y) / align * align) as u32;
            bounds.width = right as u32 - bounds.x;
            bounds.height = bottom as u32 - bounds.y;
            map.bounds = Some(bounds);
        }
        Ok(map)
    }

    pub(super) fn width(&self) -> u32 {
        self.width
    }

    pub(super) fn height(&self) -> u32 {
        self.height
    }

    /// The same extra-grid-aligned bounds as `difference_bounds`.
    pub(super) fn bounds(&self) -> Option<Rect> {
        self.bounds
    }

    /// Union of changed main pixels and covered extra-sample cells, not padding.
    pub(super) fn changed_pixels(&self) -> u64 {
        self.changed_pixels
    }

    /// Out-of-canvas coordinates are never changed.
    pub(super) fn changed(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let pixel = y as usize * self.width as usize + x as usize;
        self.words[pixel / WORD_BITS] & (1u64 << (pixel % WORD_BITS)) != 0
    }

    pub(super) fn mask_eligible(&self) -> bool {
        self.mask_eligible
    }

    /// Tight occupied-tile boxes in raster order; unlike `bounds`, these are not
    /// expanded to extra-channel grids and do not imply safe subsampled crops.
    pub(super) fn tile_bounds(&self, tile: u32) -> Result<Vec<Rect>> {
        let mut result = Vec::new();
        for rect in self.occupied_tiles(tile)? {
            result
                .try_reserve(1)
                .map_err(|_| Error::Limit("difference tile allocation"))?;
            result.push(rect);
        }
        Ok(result)
    }

    /// Streaming counterpart keeps legacy region caps from allocating all tiles.
    pub(super) fn occupied_tiles(&self, tile: u32) -> Result<impl Iterator<Item = Rect> + '_> {
        if tile == 0 {
            return Err(Error::InvalidInput(
                "difference tile size must be positive".into(),
            ));
        }
        let (left, top, right, bottom) = self.bounds.map_or((0, 0, 0, 0), |r| {
            (
                r.x / tile * tile,
                r.y / tile * tile,
                r.x + r.width,
                r.y + r.height,
            )
        });
        Ok((top..bottom).step_by(tile as usize).flat_map(move |y| {
            (left..right).step_by(tile as usize).filter_map(move |x| {
                self.tight_bounds(x, y, tile.min(self.width - x), tile.min(self.height - y))
            })
        }))
    }

    fn tight_bounds(&self, x: u32, y: u32, width: u32, height: u32) -> Option<Rect> {
        let (mut left, mut top, mut right, mut bottom) = (self.width, self.height, 0, 0);
        for row in y..y + height {
            let offset = row as usize * self.width as usize;
            let start = offset + x as usize;
            let end = start + width as usize;
            for (index, bits) in masked_words(&self.words, start, end) {
                if bits == 0 {
                    continue;
                }
                left =
                    left.min((index * WORD_BITS + bits.trailing_zeros() as usize - offset) as u32);
                right = right.max(
                    (index * WORD_BITS + WORD_BITS - bits.leading_zeros() as usize - offset) as u32,
                );
                top = top.min(row);
                bottom = row + 1;
            }
        }
        (right > left).then(|| Rect {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        })
    }

    fn scan_extra(&mut self, a: &Image, b: &Image, channel: usize) {
        let (width, height) = a.channel_dimensions(channel);
        let shift = a.channel_shift(channel);
        macro_rules! scan {
            ($different:expr) => {
                self.scan_extra_samples(width, height, shift, $different)
            };
        }
        if a.has_modular_planes() || b.has_modular_planes() {
            scan!(|i| a.modular_sample(channel, i) != b.modular_sample(channel, i));
            return;
        }
        let plane = channel - a.layout().channels();
        match (&a.extra_planes()[plane], &b.extra_planes()[plane]) {
            (PixelData::U8(a), PixelData::U8(b)) => scan!(|i| a[i] != b[i]),
            (PixelData::U16(a), PixelData::U16(b)) | (PixelData::F16(a), PixelData::F16(b)) => {
                scan!(|i| a[i] != b[i])
            }
            (PixelData::U32(a), PixelData::U32(b)) => scan!(|i| a[i] != b[i]),
            (PixelData::F32(a), PixelData::F32(b)) => scan!(|i| a[i].to_bits() != b[i].to_bits()),
            _ => scan!(|i| a.modular_sample(channel, i) != b.modular_sample(channel, i)),
        }
    }

    fn scan_extra_samples(
        &mut self,
        width: usize,
        height: usize,
        shift: u8,
        different: impl Fn(usize) -> bool,
    ) {
        let scale = 1u64 << shift;
        for y in 0..height {
            for x in 0..width {
                if !different(y * width + x) {
                    continue;
                }
                let left = (x as u64 * scale) as u32;
                let top = (y as u64 * scale) as u32;
                let right = ((x as u64 + 1) * scale).min(u64::from(self.width)) as u32;
                let bottom = ((y as u64 + 1) * scale).min(u64::from(self.height)) as u32;
                for row in top..bottom {
                    let offset = row as usize * self.width as usize;
                    set_range(
                        &mut self.words,
                        offset + left as usize,
                        offset + right as usize,
                    );
                }
            }
        }
    }
}

fn masked_words(
    words: &[u64],
    start: usize,
    end: usize,
) -> impl Iterator<Item = (usize, u64)> + '_ {
    let first = start / WORD_BITS;
    let last = (end - 1) / WORD_BITS;
    (first..=last).map(move |i| {
        let mut bits = words[i];
        if i == first {
            bits &= u64::MAX << (start % WORD_BITS);
        }
        if i == last {
            bits &= u64::MAX >> ((WORD_BITS - end % WORD_BITS) % WORD_BITS);
        }
        (i, bits)
    })
}

fn set_range(words: &mut [u64], start: usize, end: usize) {
    let first = start / WORD_BITS;
    let last = (end - 1) / WORD_BITS;
    let left = u64::MAX << (start % WORD_BITS);
    let right = u64::MAX >> ((WORD_BITS - end % WORD_BITS) % WORD_BITS);
    if first == last {
        words[first] |= left & right;
    } else {
        words[first] |= left;
        words[first + 1..last].fill(u64::MAX);
        words[last] |= right;
    }
}

/// Structural eligibility only; opacity of both inputs is checked during scan.
pub(super) fn mask_alpha_max(image: &Image) -> Option<u32> {
    let metadata = image.metadata();
    if image.has_modular_planes()
        || !image.layout().has_alpha()
        || metadata.extra_channels.len() != 1
        || !image.extra_planes().is_empty()
        || metadata.bit_depth.is_float()
        || metadata.bit_depth.bits_per_sample > 16
        || !matches!(image.pixels(), PixelData::U8(_) | PixelData::U16(_))
    {
        return None;
    }
    let alpha = &metadata.extra_channels[0];
    if alpha.channel_type != ExtraChannelType::Alpha
        || alpha.alpha_associated
        || alpha.dim_shift != 0
        || alpha.bit_depth.is_float()
        || alpha.bit_depth.bits_per_sample > 16
    {
        return None;
    }
    Some((1u32 << alpha.bit_depth.bits_per_sample) - 1)
}

fn scan_main(
    a: &Image,
    b: &Image,
    alpha: Option<u32>,
    count: usize,
    words: &mut [u64],
) -> Result<bool> {
    if !a.has_modular_planes() && !b.has_modular_planes() {
        macro_rules! typed {
            ($a:expr, $b:expr, $word:expr) => {
                match a.layout().channels() {
                    1 => scan_typed::<_, _, 1>($a, $b, $word, alpha, count, words),
                    2 => scan_typed::<_, _, 2>($a, $b, $word, alpha, count, words),
                    3 => scan_typed::<_, _, 3>($a, $b, $word, alpha, count, words),
                    4 => scan_typed::<_, _, 4>($a, $b, $word, alpha, count, words),
                    _ => unreachable!("pixel layout channel count"),
                }
            };
        }
        match (a.pixels(), b.pixels()) {
            (PixelData::U8(a_words), PixelData::U8(b_words)) => {
                return typed!(a_words, b_words, u32::from);
            }
            (PixelData::U16(a_words), PixelData::U16(b_words))
            | (PixelData::F16(a_words), PixelData::F16(b_words)) => {
                return typed!(a_words, b_words, u32::from);
            }
            (PixelData::U32(a_words), PixelData::U32(b_words)) => {
                return typed!(a_words, b_words, |v| v);
            }
            (PixelData::F32(a_words), PixelData::F32(b_words)) => {
                // Validated custom floats round-trip bit-exactly, so comparing
                // their f32 words is equivalent to comparing encoded words.
                return typed!(a_words, b_words, f32::to_bits);
            }
            _ => {}
        }
    }
    let channels = a.layout().channels();
    scan_words(words, count, |pixel| {
        let changed =
            (0..channels).any(|c| a.modular_sample(c, pixel) != b.modular_sample(c, pixel));
        let opaque = alpha.is_none_or(|max| {
            a.modular_sample(channels - 1, pixel) as u32 == max
                && b.modular_sample(channels - 1, pixel) as u32 == max
        });
        (changed, opaque)
    })
}

fn scan_typed<T: Copy + Sync, F: Fn(T) -> u32 + Sync, const CHANNELS: usize>(
    a: &[T],
    b: &[T],
    word: F,
    alpha: Option<u32>,
    count: usize,
    words: &mut [u64],
) -> Result<bool> {
    scan_words(words, count, |pixel| {
        let start = pixel * CHANNELS;
        let changed = (0..CHANNELS).any(|c| word(a[start + c]) != word(b[start + c]));
        let opaque = alpha.is_none_or(|max| {
            word(a[start + CHANNELS - 1]) == max && word(b[start + CHANNELS - 1]) == max
        });
        (changed, opaque)
    })
}

fn scan_words(
    words: &mut [u64],
    count: usize,
    sample: impl Fn(usize) -> (bool, bool) + Sync,
) -> Result<bool> {
    let fill = |offset: usize, words: &mut [u64]| {
        let mut opaque = true;
        for (index, word) in words.iter_mut().enumerate() {
            let start = offset + index * WORD_BITS;
            let mut bits = 0;
            for pixel in start..start.saturating_add(WORD_BITS).min(count) {
                let (changed, pixel_opaque) = sample(pixel);
                bits |= u64::from(changed) << (pixel - start);
                opaque &= pixel_opaque;
            }
            *word = bits;
        }
        opaque
    };
    if count < PARALLEL_PIXELS
        || crate::execution::is_serial()
        || rayon::current_thread_index().is_none()
        || crate::execution::worker_count() == 1
    {
        return Ok(fill(0, words));
    }
    // Disjoint word-aligned chunks avoid atomic pixel updates and preserve the
    // installed execution policy, including serial scopes/cooperative arenas.
    let mut blocks = Vec::new();
    blocks
        .try_reserve_exact(words.len().div_ceil(BLOCK_WORDS))
        .map_err(|_| Error::Limit("difference scan blocks"))?;
    blocks.extend(words.chunks_mut(BLOCK_WORDS));
    let opaque = AtomicBool::new(true);
    crate::execution::for_each_mut(&mut blocks, |index, block| {
        if !fill(index * BLOCK_WORDS * WORD_BITS, block) {
            opaque.store(false, Ordering::Relaxed);
        }
    });
    Ok(opaque.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PixelLayout,
        metadata::{BitDepth, ExtraChannelInfo},
        multiframe::difference_bounds,
    };

    fn assert_oracle(a: &Image, b: &Image) -> DiffMap {
        let map = DiffMap::between(a, b).unwrap();
        assert_eq!((map.width(), map.height()), (a.width(), a.height()));
        assert_eq!(map.bounds(), difference_bounds(a, b).unwrap());
        let mut count = 0;
        for y in 0..a.height() {
            for x in 0..a.width() {
                let changed = (0..a.channel_count()).any(|channel| {
                    let shift = a.channel_shift(channel);
                    let (width, _) = a.channel_dimensions(channel);
                    let pixel = (y as usize >> shift) * width + (x as usize >> shift);
                    a.modular_sample(channel, pixel) != b.modular_sample(channel, pixel)
                });
                assert_eq!(map.changed(x, y), changed, "pixel ({x}, {y})");
                count += u64::from(changed);
            }
        }
        assert_eq!(map.changed_pixels(), count);
        assert_eq!(
            map.words.len(),
            (a.width() as usize * a.height() as usize).div_ceil(WORD_BITS)
        );
        assert!(!map.changed(a.width(), 0));
        assert!(!map.changed(0, a.height()));
        assert!(!map.changed(u32::MAX, u32::MAX));
        map
    }

    fn samples(kind: usize, count: usize, changes: &[usize]) -> PixelData {
        macro_rules! values {
            ($variant:ident, $base:expr, $changed:expr) => {{
                let mut values = vec![$base; count];
                for &i in changes {
                    values[i] = $changed;
                }
                PixelData::$variant(values)
            }};
        }
        match kind {
            0 => values!(U8, 73, 74),
            1 => values!(U16, 42000, 52000),
            2 => values!(U32, 0x4000_0001, 0x4000_0002),
            3 => values!(F16, 0x7e15, 0x7e16),
            4 => values!(
                F32,
                f32::from_bits(0x7fc1_2345),
                f32::from_bits(0x7fc1_2346)
            ),
            _ => unreachable!(),
        }
    }

    #[test]
    fn all_main_words_and_layouts_match_bounds_and_pixel_oracles() {
        for (width, height) in [(1, 1), (7, 5), (33, 3), (65, 3)] {
            for layout in [
                PixelLayout::Gray,
                PixelLayout::GrayAlpha,
                PixelLayout::Rgb,
                PixelLayout::Rgba,
            ] {
                for kind in 0..5 {
                    let count = width as usize * height as usize * layout.channels();
                    let a = Image::new(width, height, layout, samples(kind, count, &[])).unwrap();
                    let b = Image::new(
                        width,
                        height,
                        layout,
                        samples(kind, count, &[0, count / 2, count - 1]),
                    )
                    .unwrap();
                    assert_oracle(&a, &b);
                    let same = assert_oracle(&a, &a);
                    assert_eq!(same.bounds(), None);
                    assert!(same.tile_bounds(1).unwrap().is_empty());
                }
            }
        }
    }

    #[test]
    fn hidden_rgb_signed_zeros_and_nan_payloads_are_changes() {
        let a = Image::new(3, 1, PixelLayout::Rgba, PixelData::U8(vec![0; 12])).unwrap();
        let b = Image::new(
            3,
            1,
            PixelLayout::Rgba,
            PixelData::U8(vec![0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0]),
        )
        .unwrap();
        let map = assert_oracle(&a, &b);
        assert_eq!(map.changed_pixels(), 1);
        assert!(!map.mask_eligible());
        for (a, b) in [
            (
                PixelData::F16(vec![0, 0x7e15, 0x7e15, 0x7c01, 0x8000]),
                PixelData::F16(vec![0x8000, 0x7e16, 0x7e15, 0x7c02, 0]),
            ),
            (
                PixelData::F32(vec![
                    0.0,
                    f32::from_bits(0x7fc1_2345),
                    f32::from_bits(0x7fc1_2345),
                    f32::from_bits(0x7f80_0001),
                    -0.0,
                ]),
                PixelData::F32(vec![
                    -0.0,
                    f32::from_bits(0x7fc1_2346),
                    f32::from_bits(0x7fc1_2345),
                    f32::from_bits(0x7f80_0002),
                    0.0,
                ]),
            ),
        ] {
            let a = Image::new(5, 1, PixelLayout::Gray, a).unwrap();
            let b = Image::new(5, 1, PixelLayout::Gray, b).unwrap();
            assert_eq!(assert_oracle(&a, &b).changed_pixels(), 4);
        }
    }

    #[test]
    fn all_extra_words_cover_subsampled_cells_and_clip_odd_edges() {
        for (width, height) in [(7, 5), (65, 3)] {
            for kind in 0..5 {
                for shift in [0, 1, 2, 3] {
                    let scale = 1usize << shift;
                    let count =
                        (width as usize).div_ceil(scale) * (height as usize).div_ceil(scale);
                    let a_plane = samples(kind, count, &[]);
                    let b_plane = samples(kind, count, &[count / 2, count - 1]);
                    let base = Image::new(
                        width,
                        height,
                        PixelLayout::Rgba,
                        PixelData::U8(vec![0; width as usize * height as usize * 4]),
                    )
                    .unwrap();
                    let mut metadata = base.metadata().clone();
                    let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
                    info.bit_depth = a_plane.bit_depth();
                    info.dim_shift = shift;
                    metadata.extra_channels.push(info);
                    let a = base
                        .clone()
                        .with_metadata(metadata.clone(), vec![a_plane])
                        .unwrap();
                    let b = base.with_metadata(metadata, vec![b_plane]).unwrap();
                    let map = assert_oracle(&a, &b);
                    assert!(map.changed(width - 1, height - 1));
                    assert!(!map.mask_eligible());
                    assert_oracle(&a, &a);
                }
            }
        }
    }

    #[test]
    fn extra_signed_zero_and_nan_changes_are_exact() {
        for (a_plane, b_plane) in [
            (
                PixelData::F16(vec![0, 0x7e01, 0x7c01, 0x8000]),
                PixelData::F16(vec![0x8000, 0x7e01, 0x7c02, 0]),
            ),
            (
                PixelData::F32(vec![
                    0.0,
                    f32::from_bits(0x7fc0_0001),
                    f32::from_bits(0x7f80_0001),
                    -0.0,
                ]),
                PixelData::F32(vec![
                    -0.0,
                    f32::from_bits(0x7fc0_0001),
                    f32::from_bits(0x7f80_0002),
                    0.0,
                ]),
            ),
        ] {
            let base = Image::new(3, 3, PixelLayout::Gray, PixelData::U8(vec![0; 9])).unwrap();
            let mut metadata = base.metadata().clone();
            let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
            info.bit_depth = a_plane.bit_depth();
            info.dim_shift = 1;
            metadata.extra_channels.push(info);
            let a = base
                .clone()
                .with_metadata(metadata.clone(), vec![a_plane])
                .unwrap();
            let b = base.with_metadata(metadata, vec![b_plane]).unwrap();
            assert_eq!(assert_oracle(&a, &b).changed_pixels(), 7);
        }
    }

    #[test]
    fn overlapping_planes_count_once_and_bounds_padding_is_not_changed() {
        let base = Image::new(7, 5, PixelLayout::Gray, PixelData::U8(vec![0; 35])).unwrap();
        let mut metadata = base.metadata().clone();
        for shift in [0, 1, 2] {
            let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
            info.dim_shift = shift;
            metadata.extra_channels.push(info);
        }
        let a = base
            .clone()
            .with_metadata(
                metadata.clone(),
                vec![
                    PixelData::U8(vec![0; 35]),
                    PixelData::U8(vec![0; 12]),
                    PixelData::U8(vec![0; 4]),
                ],
            )
            .unwrap();
        let mut full = vec![0; 35];
        full[17] = 1;
        let mut half = vec![0; 12];
        half[5] = 1;
        let b = base
            .with_metadata(
                metadata,
                vec![
                    PixelData::U8(full),
                    PixelData::U8(half),
                    PixelData::U8(vec![0; 4]),
                ],
            )
            .unwrap();
        let map = assert_oracle(&a, &b);
        assert_eq!(map.changed_pixels(), 4);
        assert_eq!(
            map.bounds(),
            Some(Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4
            })
        );
        assert!(!map.changed(0, 0));
        assert_eq!(
            map.tile_bounds(u32::MAX).unwrap(),
            [Rect {
                x: 2,
                y: 2,
                width: 2,
                height: 2
            }]
        );
    }

    #[test]
    fn mixed_storage_custom_floats_and_working_planes_use_exact_words() {
        let a = Image::new(3, 1, PixelLayout::Gray, PixelData::U8(vec![0, 73, 255])).unwrap();
        for storage in [
            PixelData::U16(vec![0, 74, 255]),
            PixelData::U32(vec![0, 74, 255]),
        ] {
            let b = Image::new(3, 1, PixelLayout::Gray, storage)
                .unwrap()
                .with_metadata(a.metadata().clone(), vec![])
                .unwrap();
            assert_eq!(assert_oracle(&a, &b).changed_pixels(), 1);
        }
        let a = Image::new(
            3,
            1,
            PixelLayout::Gray,
            PixelData::F16(vec![0, 0x7e01, 0x3c00]),
        )
        .unwrap();
        let b = Image::new(
            3,
            1,
            PixelLayout::Gray,
            PixelData::F32(vec![-0.0, f32::from_bits(0x7fc0_2000), 1.0]),
        )
        .unwrap()
        .with_metadata(a.metadata().clone(), vec![])
        .unwrap();
        assert_eq!(assert_oracle(&a, &b).changed_pixels(), 1);
        assert_eq!(assert_oracle(&b, &b).changed_pixels(), 0);
        let c = Image::new(
            3,
            1,
            PixelLayout::Gray,
            PixelData::F32(vec![0.0, f32::from_bits(0x7fc0_4000), 1.0]),
        )
        .unwrap()
        .with_metadata(a.metadata().clone(), vec![])
        .unwrap();
        assert_eq!(assert_oracle(&b, &c).changed_pixels(), 2);
        let working = a
            .with_modular_planes(3, 1, vec![vec![0x8000, 0x7e02, 0x3c00]])
            .unwrap();
        assert_eq!(assert_oracle(&a, &working).changed_pixels(), 2);
        assert_eq!(assert_oracle(&b, &working).changed_pixels(), 1);
        assert!(!assert_oracle(&working, &working).mask_eligible());
    }

    #[test]
    fn mixed_extra_storage_working_planes_and_main_grid_alignment() {
        let base = Image::new(7, 5, PixelLayout::Gray, PixelData::U8(vec![0; 35])).unwrap();
        let mut metadata = base.metadata().clone();
        let mut info = ExtraChannelInfo::new(ExtraChannelType::Depth, 8).unwrap();
        info.dim_shift = 2;
        metadata.extra_channels.push(info);
        let a = base
            .clone()
            .with_metadata(metadata.clone(), vec![PixelData::U8(vec![0; 4])])
            .unwrap();
        let b = base
            .with_metadata(metadata.clone(), vec![PixelData::U16(vec![0, 0, 0, 1])])
            .unwrap();
        assert_eq!(assert_oracle(&a, &b).changed_pixels(), 3);
        let working = a
            .with_modular_planes(7, 5, vec![vec![0; 35], vec![0, 0, 0, 2]])
            .unwrap();
        assert_eq!(assert_oracle(&b, &working).changed_pixels(), 3);
        let mut main = vec![0; 35];
        main[17] = 1;
        let c = Image::new(7, 5, PixelLayout::Gray, PixelData::U8(main))
            .unwrap()
            .with_metadata(metadata, vec![PixelData::U8(vec![0; 4])])
            .unwrap();
        let map = assert_oracle(&a, &c);
        assert_eq!(map.changed_pixels(), 1);
        assert_eq!(
            map.bounds(),
            Some(Rect {
                x: 0,
                y: 0,
                width: 4,
                height: 4
            })
        );
        assert!(!map.changed(0, 0));
        assert_eq!(assert_oracle(&c, &b).changed_pixels(), 4);
    }

    #[test]
    fn mask_eligibility_requires_both_opaque_and_exact_structure() {
        for layout in [PixelLayout::GrayAlpha, PixelLayout::Rgba] {
            for wide in [false, true] {
                let channels = layout.channels();
                let values = |alpha| {
                    let mut v = vec![1u16; channels * 3];
                    for pixel in v.chunks_exact_mut(channels) {
                        pixel[channels - 1] = alpha;
                    }
                    if wide {
                        PixelData::U16(v)
                    } else {
                        PixelData::U8(v.into_iter().map(|v| v as u8).collect())
                    }
                };
                let max = if wide { 65535 } else { 255 };
                let a = Image::new(3, 1, layout, values(max)).unwrap();
                let partial = Image::new(3, 1, layout, values(max - 1)).unwrap();
                assert!(assert_oracle(&a, &a).mask_eligible());
                assert!(!assert_oracle(&a, &partial).mask_eligible());
                assert!(!assert_oracle(&partial, &a).mask_eligible());
                let mut metadata = a.metadata().clone();
                metadata.extra_channels[0].alpha_associated = true;
                let associated = a.with_metadata(metadata, vec![]).unwrap();
                assert!(!assert_oracle(&associated, &associated).mask_eligible());
            }
        }
        for pixels in [
            PixelData::U32(vec![1, 0x7fff_ffff]),
            PixelData::F16(vec![0x3c00, 0x3c00]),
            PixelData::F32(vec![1.0, 1.0]),
        ] {
            let image = Image::new(1, 1, PixelLayout::GrayAlpha, pixels).unwrap();
            assert!(!assert_oracle(&image, &image).mask_eligible());
        }
        let base = Image::new(
            1,
            1,
            PixelLayout::GrayAlpha,
            PixelData::U16(vec![1023, 255]),
        )
        .unwrap();
        let mut metadata = base.metadata().clone();
        metadata.bit_depth = BitDepth::new(10).unwrap();
        metadata.extra_channels[0].bit_depth = BitDepth::new(8).unwrap();
        let image = base.with_metadata(metadata, vec![]).unwrap();
        assert!(assert_oracle(&image, &image).mask_eligible());
    }

    #[test]
    fn tile_bounds_are_tight_ordered_and_cover_each_change_once() {
        let width = 67;
        let height = 5;
        let a = Image::new(
            width,
            height,
            PixelLayout::Gray,
            PixelData::U8(vec![0; width as usize * height as usize]),
        )
        .unwrap();
        let mut pixels = vec![0; width as usize * height as usize];
        for (i, pixel) in pixels.iter_mut().enumerate() {
            *pixel = u8::from(i % 11 == 0 || i % 17 == 0);
        }
        let b = Image::new(width, height, PixelLayout::Gray, PixelData::U8(pixels)).unwrap();
        let map = assert_oracle(&a, &b);
        for tile in [1, 2, 3, 32, 64, u32::MAX] {
            let mut expected = Vec::new();
            for ty in (0..height).step_by(tile as usize) {
                for tx in (0..width).step_by(tile as usize) {
                    let (mut left, mut top, mut right, mut bottom) = (width, height, 0, 0);
                    for y in ty..ty.saturating_add(tile).min(height) {
                        for x in tx..tx.saturating_add(tile).min(width) {
                            if a.modular_sample(0, (y * width + x) as usize)
                                != b.modular_sample(0, (y * width + x) as usize)
                            {
                                left = left.min(x);
                                top = top.min(y);
                                right = right.max(x + 1);
                                bottom = bottom.max(y + 1);
                            }
                        }
                    }
                    if right > left {
                        expected.push(Rect {
                            x: left,
                            y: top,
                            width: right - left,
                            height: bottom - top,
                        });
                    }
                }
            }
            assert_eq!(map.tile_bounds(tile).unwrap(), expected);
        }
        assert!(map.tile_bounds(0).is_err());
    }

    #[test]
    fn parallel_main_scan_matches_serial_including_opacity_and_tail() {
        let (width, height) = (1025, 257);
        let count = width as usize * height as usize;
        let a = Image::new(
            width,
            height,
            PixelLayout::GrayAlpha,
            PixelData::U8(vec![255; count * 2]),
        )
        .unwrap();
        let mut pixels = vec![255; count * 2];
        for i in (0..count).step_by(991) {
            pixels[i * 2] = 17;
        }
        pixels[count * 2 - 1] = 254;
        let b = Image::new(width, height, PixelLayout::GrayAlpha, PixelData::U8(pixels)).unwrap();
        let serial = crate::execution::serial(|| assert_oracle(&a, &b));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let parallel = pool.install(|| DiffMap::between(&a, &b).unwrap());
        assert_eq!(serial.words, parallel.words);
        assert_eq!(serial.bounds(), parallel.bounds());
        assert_eq!(serial.changed_pixels(), parallel.changed_pixels());
        assert!(!parallel.mask_eligible());
        assert!(pool.install(|| DiffMap::between(&a, &a).unwrap().mask_eligible()));
        pool.install(|| {
            crate::execution::serial(|| {
                let nested = DiffMap::between(&a, &b).unwrap();
                assert_eq!(serial.words, nested.words);
            })
        });
    }

    #[test]
    fn incompatible_inputs_are_rejected() {
        let a = Image::new(1, 1, PixelLayout::Gray, PixelData::U8(vec![0])).unwrap();
        let b = Image::new(2, 1, PixelLayout::Gray, PixelData::U8(vec![0; 2])).unwrap();
        let c = Image::new(1, 1, PixelLayout::Gray, PixelData::U16(vec![0])).unwrap();
        assert!(DiffMap::between(&a, &b).is_err());
        assert!(DiffMap::between(&a, &c).is_err());
    }
}
