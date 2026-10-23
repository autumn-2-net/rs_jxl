// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! `enc_lossless_scan.h/.cc`: exact integer-sample coverage and literal evidence.
//! Scratch is bounded by 4096 tiles, 1024 hashes and (optionally) 70190 RGB keys.
//! Coverage visits every pixel; only literal evidence is a fixed-size sample.

use crate::{Error, Image, PixelData, Result};
use std::collections::HashSet;

pub const IMPLICIT_PALETTE_SIZE: usize = 4 * 4 * 4 + 5 * 5 * 5;
pub const PALETTE_SCOPE_LIMIT: usize = 70000 + IMPLICIT_PALETTE_SIZE;

/// `EncodeLosslessSearch` rejects floating-point or >16-bit color/extra
/// channels. This port additionally requires the source's U8/U16 input storage;
/// U32 storage with a <=16-bit declaration has no mapped source input format.
/// Cropping is storage-preserving and does not have this scheduling restriction.
pub fn validate_source_input(image: &Image) -> Result<()> {
    for depth in std::iter::once(image.metadata().bit_depth).chain(
        image
            .metadata()
            .extra_channels
            .iter()
            .map(|ec| ec.bit_depth),
    ) {
        if depth.is_float() {
            return Err(Error::InvalidInput(
                "source pixel scheduling excludes floating-point color or extra channels".into(),
            ));
        }
        if depth.bits_per_sample > 16 {
            return Err(Error::InvalidInput(
                "source pixel scheduling excludes integer color or extra channels above 16 bits"
                    .into(),
            ));
        }
    }
    for plane in std::iter::once(image.pixels()).chain(image.extra_planes()) {
        match plane {
            PixelData::U8(_) | PixelData::U16(_) => {}
            PixelData::U32(_) => {
                return Err(Error::InvalidInput(
                    "U32 storage is not mapped to a source scheduler input format".into(),
                ));
            }
            PixelData::F16(_) | PixelData::F32(_) => {
                return Err(Error::InvalidInput(
                    "source pixel scheduling excludes floating-point sample storage".into(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScanOptions {
    /// C++ JXL_LOSSLESS_PALETTE_SCOPE=1, off by default.
    pub palette_scope: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScanRegion {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScanTile {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub anchor_x: usize,
    pub anchor_y: usize,
    pub horizontal: u64,
    pub vertical: u64,
    pub observations: u64,
    pub peak_row: usize,
}

impl ScanTile {
    pub fn activity(&self) -> f64 {
        if self.observations == 0 {
            0.0
        } else {
            (self.horizontal + self.vertical) as f64 / self.observations as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LosslessScan {
    pub width: usize,
    pub height: usize,
    pub pixels: u64,
    pub literal_candidate: bool,
    pub sampled_colors: usize,
    pub sampled_flat: usize,
    pub sampled_pixels: usize,
    /// RGB only, ignoring alpha: a lower bound, never an RGBA upper bound.
    pub palette_color_lower_bound: usize,
    pub tiles: Vec<ScanTile>,
}

impl Default for LosslessScan {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixels: 0,
            literal_candidate: true,
            sampled_colors: 0,
            sampled_flat: 0,
            sampled_pixels: 0,
            palette_color_lower_bound: 0,
            tiles: Vec::new(),
        }
    }
}

impl LosslessScan {
    pub fn available(&self) -> bool {
        !self.tiles.is_empty()
    }
    pub fn likely_continuous_tone(&self) -> bool {
        self.sampled_pixels != 0
            && self.sampled_colors > self.sampled_pixels / 2
            && self.sampled_flat < self.sampled_pixels / 2
    }
    pub fn likely_sparse_continuous_tone(&self) -> bool {
        if self.sampled_flat >= self.sampled_pixels {
            return false;
        }
        let active = self.sampled_pixels - self.sampled_flat;
        active >= 64 && self.sampled_colors > active - active / 4
    }
    pub fn prefer_compact_band_atlas(&self) -> bool {
        self.pixels >= 1 << 19
            && self.pixels < 2 << 20
            && self.sampled_pixels != 0
            && self.sampled_flat < self.sampled_pixels / 2
    }
    pub fn likely_dense_texture(&self) -> bool {
        self.sampled_pixels != 0
            && self.sampled_flat < self.sampled_pixels / 3
            && self.sampled_colors > self.sampled_pixels / 8
    }
    pub fn regions(&self) -> Vec<ScanRegion> {
        if !self.available() {
            return Vec::new();
        }
        let mut active: Vec<_> = self
            .tiles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.horizontal != 0 || t.vertical != 0)
            .map(|(i, _)| i)
            .collect();
        if active.is_empty() {
            active.push(self.tiles.len() / 2);
        }
        active.sort_by(|&a, &b| {
            self.tiles[a]
                .activity()
                .total_cmp(&self.tiles[b].activity())
        });
        let mut result: Vec<ScanRegion> = Vec::new();
        for i in 0..3 {
            let lo = (2 - i) * active.len() / 3;
            let hi = (lo + 1).max((3 - i) * active.len() / 3);
            let mut chosen = active[lo + (hi - lo) / 2];
            let mut farthest = -1.0;
            for &index in &active[lo..hi] {
                let tile = &self.tiles[index];
                let mut distance: f64 = 2.0;
                for r in &result {
                    let dx =
                        (tile.anchor_x as f64 - (r.x + r.width / 2) as f64) / self.width as f64;
                    let dy =
                        (tile.anchor_y as f64 - (r.y + r.height / 2) as f64) / self.height as f64;
                    distance = distance.min(dx * dx + dy * dy);
                }
                if distance > farthest {
                    chosen = index;
                    farthest = distance;
                }
            }
            let tile = &self.tiles[chosen];
            let width = self.width.min(if i == 2 { 256 } else { 1024 });
            let height = self.height.min(if i == 2 {
                256
            } else if self.pixels >= 2 << 20 {
                64
            } else {
                32
            });
            let region = centered_region(
                self.width,
                self.height,
                tile.anchor_x,
                tile.anchor_y,
                width,
                height,
            );
            if !result.contains(&region) {
                result.push(region);
            }
        }
        result
    }
}

pub(crate) fn centered_region(
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> ScanRegion {
    ScanRegion {
        x: (w - width).min(x.saturating_sub(width / 2)),
        y: (h - height).min(y.saturating_sub(height / 2)),
        width,
        height,
    }
}

trait Sample: Copy + Eq + Sync {
    fn hash(self, hash: u64) -> u64;
    fn key(self) -> u64;
    const BITS: usize;
}
impl Sample for u8 {
    fn hash(self, hash: u64) -> u64 {
        (hash ^ u64::from(self)).wrapping_mul(1099511628211)
    }
    fn key(self) -> u64 {
        u64::from(self)
    }
    const BITS: usize = 8;
}
impl Sample for u16 {
    fn hash(self, mut hash: u64) -> u64 {
        // The typed Rust input is native endian, matching the C++ native input.
        for byte in self.to_ne_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(1099511628211);
        }
        hash
    }
    fn key(self) -> u64 {
        u64::from(self)
    }
    const BITS: usize = 16;
}

fn literal_statistics<T: Sample>(
    data: &[T],
    w: usize,
    h: usize,
    channels: usize,
) -> (usize, usize) {
    let mut hashes = [0u64; 1024];
    let mut flat = 0;
    let pixel = |x: usize, y: usize| &data[(y * w + x) * channels..(y * w + x + 1) * channels];
    for y in 0..32 {
        let sy = (y as u64 * (h - 1) as u64 / 31) as usize;
        for x in 0..32 {
            let sx = (x as u64 * (w - 1) as u64 / 31) as usize;
            let p = pixel(sx, sy);
            hashes[32 * y + x] = p.iter().fold(14695981039346656037, |hash, &v| v.hash(hash));
            flat += usize::from(
                p == pixel((sx + 1).min(w - 1), sy) && p == pixel(sx, (sy + 1).min(h - 1)),
            );
        }
    }
    hashes.sort_unstable();
    let colors = 1 + hashes.windows(2).filter(|p| p[0] != p[1]).count();
    (colors, flat)
}

/// Fixed 1024-observation evidence, without allocating coverage tiles or
/// visiting the full image. The search-level tiny-image exception is separate.
/// Unsupported source inputs provide no evidence; use validate_source_input
/// to obtain the rejection reason before making scheduling decisions.
pub fn literal_evidence(image: &Image) -> bool {
    if validate_source_input(image).is_err() {
        return false;
    }
    let w = image.width() as usize;
    let h = image.height() as usize;
    let (colors, flat) = match image.pixels() {
        PixelData::U8(data) => literal_statistics(data, w, h, image.layout().channels()),
        PixelData::U16(data) => literal_statistics(data, w, h, image.layout().channels()),
        PixelData::U32(_) | PixelData::F16(_) | PixelData::F32(_) => return false,
    };
    flat >= 128 || colors <= 256
}

fn scan_samples<T: Sample>(
    data: &[T],
    w: usize,
    h: usize,
    channels: usize,
    options: ScanOptions,
    pool: Option<&rayon::ThreadPool>,
) -> LosslessScan {
    let (colors, flat) = literal_statistics(data, w, h, channels);
    let mut result = LosslessScan {
        width: w,
        height: h,
        pixels: w as u64 * h as u64,
        literal_candidate: flat >= 128 || colors <= 256,
        sampled_colors: colors,
        sampled_flat: flat,
        sampled_pixels: 1024,
        palette_color_lower_bound: 0,
        tiles: Vec::new(),
    };
    if options.palette_scope && channels >= 3 {
        let mut colors = HashSet::with_capacity(PALETTE_SCOPE_LIMIT + 1);
        let mut previous = u64::MAX;
        for p in data.chunks_exact(channels) {
            let key = p[0].key() | (p[1].key() << T::BITS) | (p[2].key() << (2 * T::BITS));
            if key == previous {
                continue;
            }
            previous = key;
            colors.insert(key);
            if colors.len() > PALETTE_SCOPE_LIMIT {
                break;
            }
        }
        result.palette_color_lower_bound = colors.len();
    }
    let tw = 64.max(w.div_ceil(64));
    let th = 64.max(h.div_ceil(64));
    result.tiles.reserve(w.div_ceil(tw) * h.div_ceil(th));
    for ty in (0..h).step_by(th) {
        for tx in (0..w).step_by(tw) {
            let width = tw.min(w - tx);
            let height = th.min(h - ty);
            result.tiles.push(ScanTile {
                x: tx,
                y: ty,
                width,
                height,
                anchor_x: tx + width / 2,
                anchor_y: ty + height / 2,
                ..Default::default()
            });
        }
    }
    scan_plane(data, w, channels, &mut result.tiles, pool);
    result
}

fn scan_plane<T: Sample>(
    data: &[T],
    w: usize,
    channels: usize,
    tiles: &mut [ScanTile],
    pool: Option<&rayon::ThreadPool>,
) {
    if let Some(pool) = pool {
        let columns = w.div_ceil(64.max(w.div_ceil(64)));
        // ScanPlane assigns whole tile rows. Each plane completes before the
        // next one, preserving strict peak comparisons and first-plane ties.
        crate::execution::with_pool(pool, || {
            let mut rows: Vec<_> = tiles.chunks_mut(columns).collect();
            crate::execution::for_each_mut(&mut rows, |_, row| {
                accumulate_plane(data, w, channels, row);
            });
        });
    } else {
        accumulate_plane(data, w, channels, tiles);
    }
}

fn accumulate_plane<T: Sample>(data: &[T], w: usize, channels: usize, tiles: &mut [ScanTile]) {
    let pixel = |x: usize, y: usize| &data[(y * w + x) * channels..(y * w + x + 1) * channels];
    for tile in tiles {
        for y in tile.y..tile.y + tile.height {
            let mut activity = 0;
            let mut first_x = tile.x;
            for x in tile.x..tile.x + tile.width {
                let p = pixel(x, y);
                let horizontal = x != 0 && p != pixel(x - 1, y);
                let vertical = y != 0 && p != pixel(x, y - 1);
                if (horizontal || vertical) && activity == 0 {
                    first_x = x;
                }
                activity += usize::from(horizontal) + usize::from(vertical);
                tile.horizontal += u64::from(horizontal);
                tile.vertical += u64::from(vertical);
            }
            if activity > tile.peak_row {
                tile.peak_row = activity;
                tile.anchor_x = first_x;
                tile.anchor_y = y;
            }
        }
        tile.observations += 2 * tile.width as u64 * tile.height as u64;
    }
}

/// Includes hidden RGB, alpha, and all bits of 16-bit samples. No thumbnail.
/// Unsupported inputs return an unavailable scan, never integer statistics on
/// float bits. Source scheduling entry points call validate_source_input first.
pub fn scan_lossless_input(image: &Image, options: ScanOptions) -> LosslessScan {
    scan_input(image, options, None)
}

/// Source ScanPlane tile-row dispatch using only the caller's pool. Literal
/// evidence and bounded palette counting retain their original serial order.
/// There is no image-sized scratch or cross-worker accumulation/reduction.
pub fn scan_lossless_input_with_pool(
    image: &Image,
    options: ScanOptions,
    pool: &rayon::ThreadPool,
) -> LosslessScan {
    scan_input(image, options, Some(pool))
}

fn scan_input(
    image: &Image,
    options: ScanOptions,
    pool: Option<&rayon::ThreadPool>,
) -> LosslessScan {
    if validate_source_input(image).is_err() {
        return LosslessScan {
            literal_candidate: false,
            ..Default::default()
        };
    }
    if image.metadata().extra_channels.len() > 8
        || image
            .metadata()
            .extra_channels
            .iter()
            .any(|ec| ec.dim_shift != 0)
    {
        return LosslessScan::default();
    }
    let w = image.width() as usize;
    let h = image.height() as usize;
    let channels = image.layout().channels();
    let options = ScanOptions {
        palette_scope: options.palette_scope && image.pixels().bits() == image.bits(),
    };
    let mut result = match image.pixels() {
        PixelData::U8(data) => scan_samples(data, w, h, channels, options, pool),
        PixelData::U16(data) => scan_samples(data, w, h, channels, options, pool),
        PixelData::U32(_) | PixelData::F16(_) | PixelData::F32(_) => {
            unreachable!("validated source storage")
        }
    };
    for plane in image.extra_planes() {
        match plane {
            PixelData::U8(data) => scan_plane(data, w, 1, &mut result.tiles, pool),
            PixelData::U16(data) => scan_plane(data, w, 1, &mut result.tiles, pool),
            PixelData::U32(_) | PixelData::F16(_) | PixelData::F32(_) => {
                unreachable!("validated extra storage")
            }
        }
    }
    result
}
