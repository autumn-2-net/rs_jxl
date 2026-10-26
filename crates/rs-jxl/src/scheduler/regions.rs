// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Translated portions are governed by the upstream BSD-3-Clause license.
//! Original-pixel rectangles and optional band atlas from
//! `enc_lossless_probe.cc::LosslessCandidateScreen::Impl::Init`.

use super::scan::{IMPLICIT_PALETTE_SIZE, LosslessScan, ScanRegion, centered_region};
use crate::{Error, Image, PixelData, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AtlasMode {
    #[default]
    Disabled,
    /// Source setting 1.
    Always,
    /// Source setting 2.
    PaletteIneligible,
    /// Source setting 3.
    Uniform,
    /// Source setting 4.
    Routed,
}
#[derive(Clone, Debug)]
pub struct RegionOptions {
    pub context_side: usize,
    pub full_width_strips: bool,
    /// Explicit experimental full-width calibration height; default source geometry is unchanged.
    pub strip_rows: Option<usize>,
    pub compact_context_side: usize,
    pub atlas: AtlasMode,
    /// None is source default (automatic for Routed, 65536 otherwise).
    pub band_pixels: Option<usize>,
    pub automatic_band_pixels: bool,
}
impl Default for RegionOptions {
    fn default() -> Self {
        Self {
            context_side: 0,
            full_width_strips: false,
            strip_rows: None,
            compact_context_side: 256,
            atlas: AtlasMode::Disabled,
            band_pixels: None,
            automatic_band_pixels: false,
        }
    }
}

#[cfg(test)]
mod strip_tests {
    use super::*;
    use crate::PixelLayout;

    #[test]
    fn explicit_strip_height_is_bounded_and_preserves_source_samples() {
        let image = Image::new(
            67,
            37,
            PixelLayout::Gray,
            PixelData::U8((0..67 * 37).map(|p| (p % 251) as u8).collect()),
        )
        .unwrap();
        for height in [1, 17, 128, 1024] {
            let options = RegionOptions {
                full_width_strips: true,
                strip_rows: Some(height),
                ..Default::default()
            };
            for spec in screen_regions(&image, None, &options).unwrap() {
                assert_eq!(spec.regions.len(), 1);
                let r = spec.regions[0];
                assert_eq!(r.width, 67);
                assert_eq!(r.height, height.min(37));
                assert!(r.y + r.height <= 37);
                let sample = copy_sample(&image, &spec, 67 * 37).unwrap();
                let PixelData::U8(values) = sample.pixels() else {
                    panic!("sample format changed")
                };
                let PixelData::U8(original) = image.pixels() else {
                    unreachable!()
                };
                assert_eq!(values, &original[r.y * 67..(r.y + r.height) * 67]);
            }
        }
        for (wide, height) in [(false, 8), (true, 0), (true, 1025)] {
            let options = RegionOptions {
                full_width_strips: wide,
                strip_rows: Some(height),
                ..Default::default()
            };
            assert!(screen_regions(&image, None, &options).is_err());
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleSpec {
    /// Multiple rectangles are stacked vertically with no resampling.
    pub regions: Vec<ScanRegion>,
}
impl SampleSpec {
    pub fn pixels(&self) -> Result<usize> {
        self.regions.iter().try_fold(0usize, |sum, r| {
            r.width
                .checked_mul(r.height)
                .and_then(|n| sum.checked_add(n))
                .ok_or(Error::Limit("probe pixel count"))
        })
    }
    pub fn is_atlas(&self) -> bool {
        self.regions.len() > 1
    }
}

pub fn screen_regions(
    image: &Image,
    scan: Option<&LosslessScan>,
    options: &RegionOptions,
) -> Result<Vec<SampleSpec>> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let pixels = w as u64 * h as u64;
    if scan.is_some_and(|s| s.available() && (s.width != w || s.height != h)) {
        return Err(Error::InvalidInput(
            "coverage dimensions do not match probe image".into(),
        ));
    }
    if options.compact_context_side == 0
        || options.compact_context_side > 256
        || options.band_pixels.is_some_and(|p| p == 0 || p > 262144)
        || options
            .strip_rows
            .is_some_and(|rows| rows == 0 || rows > 1024 || !options.full_width_strips)
    {
        return Err(Error::InvalidInput("invalid probe region limits".into()));
    }
    let scan = scan.filter(|s| s.available());
    let large = pixels >= 2 << 20;
    let small = pixels < 1 << 19;
    let compact = pixels < 1 << 18;
    let width = w.min(if compact {
        128
    } else if small {
        256
    } else {
        1024
    });
    let height = h.min(if compact {
        8
    } else if small {
        16
    } else if large {
        64
    } else {
        32
    });
    let mut regions: Vec<ScanRegion> = Vec::new();
    if options.full_width_strips {
        let ch = h.min(options.strip_rows.unwrap_or((65536 / w).clamp(8, 32)));
        if let Some(scan) = scan {
            let mut active: Vec<_> = scan
                .tiles
                .iter()
                .enumerate()
                .filter(|(_, t)| t.horizontal != 0 || t.vertical != 0)
                .map(|(i, _)| i)
                .collect();
            if active.is_empty() {
                active.push(scan.tiles.len() / 2);
            }
            active.sort_by(|&a, &b| {
                scan.tiles[b]
                    .activity()
                    .total_cmp(&scan.tiles[a].activity())
            });
            for i in 0..3 {
                let lo = i * active.len() / 3;
                let hi = (lo + 1).max((i + 1) * active.len() / 3);
                let mut best = active[lo];
                let mut separation = 0;
                for &j in &active[lo..hi] {
                    let y = scan.tiles[j].anchor_y;
                    let distance = regions
                        .iter()
                        .fold(h, |d, r| d.min(y.abs_diff(r.y + ch / 2)));
                    if distance > separation {
                        best = j;
                        separation = distance;
                    }
                }
                regions.push(centered_region(
                    w,
                    h,
                    w / 2,
                    scan.tiles[best].anchor_y,
                    w,
                    ch,
                ));
            }
        } else {
            for i in 0..3 {
                regions.push(ScanRegion {
                    x: 0,
                    y: i * (h - ch) / 2,
                    width: w,
                    height: ch,
                });
            }
        }
    } else if options.context_side != 0 {
        let cw = w.min(options.context_side);
        let ch = h.min(options.context_side);
        for i in 0..3 {
            regions.push(ScanRegion {
                x: i * (w - cw) / 2,
                y: i * (h - ch) / 2,
                width: cw,
                height: ch,
            });
        }
    } else if let Some(scan) = scan {
        let limit = if large {
            256
        } else {
            options.compact_context_side
        };
        for mut r in scan.regions() {
            if r.width == r.height && r.width > limit {
                r.x += (r.width - limit) / 2;
                r.y += (r.height - limit) / 2;
                r.width = limit;
                r.height = limit;
            }
            regions.push(r);
        }
    } else {
        for i in 0..3 {
            regions.push(ScanRegion {
                x: i * (w - width) / 2,
                y: i * (h - height) / 2,
                width,
                height,
            });
        }
        let side = if compact {
            32
        } else if small {
            64
        } else {
            256
        };
        let sw = w.min(side);
        let sh = h.min(side);
        regions.push(ScanRegion {
            x: (w - sw) / 2,
            y: (h - sh) / 2,
            width: sw,
            height: sh,
        });
    }
    regions.sort();
    regions.dedup();
    let eight_bit = image.bits() <= 8
        && image
            .metadata()
            .extra_channels
            .iter()
            .all(|ec| ec.bit_depth.bits_per_sample <= 8);
    let compact_bands = eight_bit && scan.is_some_and(LosslessScan::prefer_compact_band_atlas);
    let ineligible =
        scan.is_some_and(|s| s.palette_color_lower_bound > 10000 + IMPLICIT_PALETTE_SIZE);
    let atlas_enabled = match options.atlas {
        AtlasMode::Disabled => false,
        AtlasMode::Always | AtlasMode::Uniform => true,
        AtlasMode::PaletteIneligible => ineligible,
        AtlasMode::Routed => compact_bands || ineligible,
    };
    if options.context_side == 0 && !options.full_width_strips && atlas_enabled {
        let uniform = options.atlas == AtlasMode::Uniform;
        let count = if uniform { h.min(8) } else { 1 };
        let automatic = eight_bit
            && (options.automatic_band_pixels
                || (options.band_pixels.is_none() && options.atlas == AtlasMode::Routed));
        let tone = scan.is_some_and(|s| {
            s.likely_continuous_tone()
                || (large
                    && s.likely_sparse_continuous_tone()
                    && s.palette_color_lower_bound > 70000 + IMPLICIT_PALETTE_SIZE)
        });
        let band_pixels = if automatic {
            if compact_bands {
                8192
            } else if large && tone {
                32768
            } else {
                65536
            }
        } else {
            options.band_pixels.unwrap_or(65536)
        };
        let bh = h.min(if uniform {
            (131072 / w / count).clamp(1, 16)
        } else {
            (band_pixels / w).clamp(8, 32)
        });
        let mut rows = Vec::new();
        if uniform {
            for i in 0..count {
                let center = ((2 * i + 1) as u64 * h as u64 / (2 * count) as u64) as usize;
                rows.push((h - bh).min(center.saturating_sub(bh / 2)));
            }
        } else {
            for r in &regions {
                rows.push((h - bh).min((r.y + r.height / 2).saturating_sub(bh / 2)));
            }
        }
        rows.sort_unstable();
        rows.dedup();
        return Ok(vec![SampleSpec {
            regions: rows
                .into_iter()
                .map(|y| ScanRegion {
                    x: 0,
                    y,
                    width: w,
                    height: bh,
                })
                .collect(),
        }]);
    }
    Ok(regions
        .into_iter()
        .map(|r| SampleSpec { regions: vec![r] })
        .collect())
}

/// The nine independent 64x64 witnesses used by ProbeLosslessCandidates.
pub fn small_probe_regions(image: &Image) -> Vec<SampleSpec> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    let width = w.min(64);
    let height = h.min(64);
    let mut result = Vec::new();
    for gy in 0..3 {
        for gx in 0..3 {
            let r = ScanRegion {
                x: ((2 * gx + 1) as u64 * (w - width) as u64 / 6) as usize,
                y: ((2 * gy + 1) as u64 * (h - height) as u64 / 6) as usize,
                width,
                height,
            };
            let spec = SampleSpec { regions: vec![r] };
            if !result.contains(&spec) {
                result.push(spec);
            }
        }
    }
    result
}

/// Validates before allocating; retains every sample, including invisible RGB.
/// U32 integers and F16/F32 raw bits (NaN payloads and signed zero included)
/// are copied without conversion. This does not imply source search support.
pub fn copy_sample(image: &Image, spec: &SampleSpec, max_pixels: usize) -> Result<Image> {
    if image
        .metadata()
        .extra_channels
        .iter()
        .any(|ec| ec.dim_shift != 0)
    {
        return Err(Error::InvalidInput(
            "subsampled extra channels are not supported by probe cropping".into(),
        ));
    }
    let first = spec
        .regions
        .first()
        .ok_or_else(|| Error::InvalidInput("empty probe atlas".into()))?;
    let width = first.width;
    let mut height = 0usize;
    for r in &spec.regions {
        if width == 0
            || r.width != width
            || r.height == 0
            || r.x
                .checked_add(r.width)
                .is_none_or(|x| x > image.width() as usize)
            || r.y
                .checked_add(r.height)
                .is_none_or(|y| y > image.height() as usize)
        {
            return Err(Error::InvalidInput(
                "probe rectangle outside image or unequal atlas widths".into(),
            ));
        }
        height = height
            .checked_add(r.height)
            .ok_or(Error::Limit("atlas height"))?;
    }
    let pixels = spec.pixels()?;
    if pixels > max_pixels {
        return Err(Error::Limit("probe sample pixels"));
    }
    let channels = image.layout().channels();
    let count = pixels
        .checked_mul(channels)
        .ok_or(Error::Limit("probe sample count"))?;
    fn copy<T: Copy>(
        source: &[T],
        spec: &SampleSpec,
        stride: usize,
        channels: usize,
        count: usize,
    ) -> Result<Vec<T>> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("probe allocation"))?;
        for r in &spec.regions {
            for y in r.y..r.y + r.height {
                let begin = y * stride + r.x * channels;
                output.extend_from_slice(&source[begin..begin + r.width * channels]);
            }
        }
        Ok(output)
    }
    let stride = image.width() as usize * channels;
    let data = match image.pixels() {
        PixelData::U8(v) => PixelData::U8(copy(v, spec, stride, channels, count)?),
        PixelData::U16(v) => PixelData::U16(copy(v, spec, stride, channels, count)?),
        PixelData::U32(v) => PixelData::U32(copy(v, spec, stride, channels, count)?),
        PixelData::F16(v) => PixelData::F16(copy(v, spec, stride, channels, count)?),
        PixelData::F32(v) => PixelData::F32(copy(v, spec, stride, channels, count)?),
    };
    let mut extras = Vec::new();
    extras
        .try_reserve_exact(image.extra_planes().len())
        .map_err(|_| Error::Limit("probe extra channels"))?;
    for plane in image.extra_planes() {
        extras.push(match plane {
            PixelData::U8(v) => PixelData::U8(copy(v, spec, image.width() as usize, 1, pixels)?),
            PixelData::U16(v) => PixelData::U16(copy(v, spec, image.width() as usize, 1, pixels)?),
            PixelData::U32(v) => PixelData::U32(copy(v, spec, image.width() as usize, 1, pixels)?),
            PixelData::F16(v) => PixelData::F16(copy(v, spec, image.width() as usize, 1, pixels)?),
            PixelData::F32(v) => PixelData::F32(copy(v, spec, image.width() as usize, 1, pixels)?),
        });
    }
    Image::new(
        u32::try_from(width).map_err(|_| Error::Limit("probe width"))?,
        u32::try_from(height).map_err(|_| Error::Limit("probe height"))?,
        image.layout(),
        data,
    )?
    .with_metadata(image.metadata().clone(), extras)
}
