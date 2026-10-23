// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Modular sample transforms from the supplied read-only libjxl baseline
//! `b866b3defec13494bfbaf705db93581cda118e37`. Source map:
//!
//! - `lib/jxl/modular/transform/enc_rct.cc`: `RctPermute`, `FwdRctRow`,
//!   `FwdRctImpl`; all 42 RCTs. `rct.cc::InvRCTRow` is the test inverse.
//! - `enc_squeeze.cc`: `AVERAGE`, `FwdHSqueeze`, `FwdVSqueeze`;
//!   `squeeze.h::SmoothTendency`; `squeeze.cc` supplies test inverses.
//! - `enc_palette.cc::FwdPaletteIteration`: single-channel sorted palette with
//!   the default <=65536-span dense LUT (`palette.h` cap) and large-range tree,
//!   multi-channel lossless palette, implicit-color promotion (>10), image
//!   order and default luma ordering. `palette.h::GetPaletteValue` supplies
//!   the two nonnegative implicit cubes; `palette.cc` supplies test expansion.
//! - `enc_palette_order.h`, `enc_palette_map.h` and enc_palette.cc optional
//!   modes: explicit `PaletteTools`, implemented in the palette_order child.
//! - `enc_palette_delta.h`: `FwdExactDeltaPalette`, `ExactZeroDeltaMask`, and
//!   `AddExactZeroDelta`, preserving residual counts, lexicographic tie breaks,
//!   wide prediction differences, and zero-entry/index remapping.
//!
//! Integration boundary: these APIs only transform tightly packed samples.
//! Callers own equal channel dimensions/shifts, begin-channel metadata,
//! transform headers, palette meta-channel insertion, squeeze channel ordering
//! and hshift/vshift changes. Equal flattened lengths do NOT establish equal
//! image dimensions. RCT type zero is a successful no-op (C++ returns false
//! to signal 'not applied'). RCT uses wrapping i32 operations, matching Highway
//! lanes and the inverse's PixelAdd, including the sum BEFORE signed shifts.
//!
//! RCT dispatches to native Rust AVX2/SSE2 or portable scalar kernels.
//! This module does not own the default squeeze schedule, meta-transform machinery,
//! lossy palettes or negative implicit delta tables. Optional palette ordering
//! and mapping use explicit options, never process environment. Exact delta
//! palettes retain the source's
//! 3/4-channel, <=16-bit, non-weighted predictor eligibility. Squeeze rejects
//! signed-overflow cases in the scalar C++ forward
//! operations rather than inventing a non-reversible wrapping transform.
//! Multi-channel palette declines depths >24: the source's forward implicit
//! cube uses input depth while its inverse caps depth at 24. Single-channel
//! explicit palettes support depths 1..31. No codec or compression claim.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use super::predictor::{Neighbors, Predictor, predict_one};
use crate::{Error, Result};

#[path = "palette_order.rs"]
pub mod palette_order;

pub use palette_order::{PaletteGeometry, PaletteMapping, PaletteOrder, PaletteTools};

fn channel_range(planes: &[Vec<i32>], begin: usize, count: usize) -> Result<usize> {
    let end = begin
        .checked_add(count)
        .ok_or(Error::Limit("transform channel range"))?;
    if count == 0 || end > planes.len() {
        return Err(Error::InvalidInput(
            "invalid transform channel range".into(),
        ));
    }
    let len = planes[begin].len();
    if planes[begin..end].iter().any(|p| p.len() != len) {
        return Err(Error::InvalidInput(
            "transform planes must have equal lengths".into(),
        ));
    }
    Ok(end)
}

/// All six channel permutations times seven reversible color transforms.
/// Validation precedes mutation. Other planes (including alpha) are untouched.
pub fn rct_forward(planes: &mut [Vec<i32>], begin: usize, rct_type: u32) -> Result<()> {
    super::rct_simd::forward(planes, begin, rct_type)
}

/// `squeeze.h::SmoothTendency`. Division is truncation toward zero, NOT shift.
pub fn smooth_tendency(previous: i32, average: i32, next_average: i32) -> i64 {
    let (b, a, n) = (
        i64::from(previous),
        i64::from(average),
        i64::from(next_average),
    );
    let mut diff = 0;
    if b >= a && a >= n {
        diff = (4 * b - 3 * n - a + 6) / 12;
        if diff - (diff & 1) > 2 * (b - a) {
            diff = 2 * (b - a) + 1;
        }
        if diff + (diff & 1) > 2 * (a - n) {
            diff = 2 * (a - n);
        }
    } else if b <= a && a <= n {
        diff = (4 * b - 3 * n - a - 6) / 12;
        if diff + (diff & 1) < 2 * (b - a) {
            diff = 2 * (b - a) - 1;
        }
        if diff - (diff & 1) < 2 * (a - n) {
            diff = 2 * (a - n);
        }
    }
    diff
}

fn squeeze_average(a: i32, b: i32) -> Result<i32> {
    a.checked_add(b)
        .and_then(|sum| sum.checked_add(i32::from(a > b)))
        .map(|sum| sum >> 1)
        .ok_or(Error::Limit(
            "squeeze average overflows source i32 arithmetic",
        ))
}

fn zeroed_samples(len: usize) -> Result<Vec<i32>> {
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(len)
        .map_err(|_| Error::Limit("transform allocation"))?;
    samples.resize(len, 0);
    Ok(samples)
}

/// One source channel split into its average and residual channels.
/// No metadata or channel-vector edits are performed on the caller's behalf.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SqueezedPlane {
    pub average: Vec<i32>,
    pub residual: Vec<i32>,
    pub average_width: usize,
    pub average_height: usize,
    pub residual_width: usize,
    pub residual_height: usize,
}

/// Proposed per-channel squeeze interface. An odd trailing sample is copied
/// unchanged to the average channel. Length-one axes have an empty residual.
pub fn squeeze_forward(
    samples: &[i32],
    width: usize,
    height: usize,
    horizontal: bool,
) -> Result<SqueezedPlane> {
    let len = width
        .checked_mul(height)
        .ok_or(Error::Limit("squeeze dimensions"))?;
    if width == 0 || height == 0 || samples.len() != len {
        return Err(Error::InvalidInput(
            "squeeze requires a nonempty tightly packed plane".into(),
        ));
    }
    let (axis, lines) = if horizontal {
        (width, height)
    } else {
        (height, width)
    };
    let pairs = axis / 2;
    let averages = pairs + axis % 2;
    let (aw, ah, rw, rh) = if horizontal {
        (averages, height, pairs, height)
    } else {
        (width, averages, width, pairs)
    };
    let mut output = SqueezedPlane {
        average: zeroed_samples(aw * ah)?,
        residual: zeroed_samples(rw * rh)?,
        average_width: aw,
        average_height: ah,
        residual_width: rw,
        residual_height: rh,
    };
    for line in 0..lines {
        let input = |k| {
            samples[if horizontal {
                line * width + k
            } else {
                k * width + line
            }]
        };
        for k in 0..pairs {
            let a = input(2 * k);
            let b = input(2 * k + 1);
            let average = squeeze_average(a, b)?;
            let diff = a
                .checked_sub(b)
                .ok_or(Error::Limit("squeeze difference overflows i32"))?;
            let next_average = if k + 1 < pairs {
                squeeze_average(input(2 * k + 2), input(2 * k + 3))?
            } else if axis & 1 != 0 {
                input(2 * k + 2)
            } else {
                average
            };
            let previous = if k > 0 { input(2 * k - 1) } else { average };
            let tendency = i32::try_from(smooth_tendency(previous, average, next_average))
                .map_err(|_| Error::Limit("squeeze tendency exceeds i32"))?;
            let residual = diff
                .checked_sub(tendency)
                .ok_or(Error::Limit("squeeze residual overflows i32"))?;
            let ai = if horizontal {
                line * aw + k
            } else {
                k * width + line
            };
            let ri = if horizontal {
                line * rw + k
            } else {
                k * width + line
            };
            output.average[ai] = average;
            output.residual[ri] = residual;
        }
        if axis & 1 != 0 {
            let ai = if horizontal {
                line * aw + pairs
            } else {
                pairs * width + line
            };
            output.average[ai] = input(axis - 1);
        }
    }
    Ok(output)
}

const IMPLICIT_COLORS: usize = 64 + 125;
// palette.h::palette_internal::kMaxPaletteLookupTableSize.
const MAX_PALETTE_LOOKUP_TABLE_SIZE: usize = 1 << 16;

fn implicit_palette_value(index: usize, channel: usize, bit_depth: u32) -> i32 {
    if channel >= 3 {
        return 0;
    }
    let max = (1u64 << bit_depth) - 1;
    if index < 64 {
        let digit = (index >> (2 * channel)) % 4;
        ((digit as u64 * max) >> 2) as i32 + (1i32 << bit_depth.saturating_sub(3))
    } else {
        let digit = ((index - 64) / 5usize.pow(channel as u32)) % 5;
        ((digit as u64 * max) >> 2) as i32
    }
}

/// A zero-delta palette (`predictor = Zero`, `nb_deltas = 0`).
/// `palette[c][index]` is row c of the new meta-channel; its width is
/// `palette[0].len()` and its height is the number of replaced channels.
/// Replace the selected channels with `indices`, then insert the palette at
/// channel zero and increment nb_meta_channels, as in FwdPaletteIteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaletteOutput {
    pub palette: Vec<Vec<i32>>,
    pub indices: Vec<i32>,
}

/// Proposed lossless palette interface; `None` means the source eligibility or
/// color-count test declined the transform. Input planes are never mutated.
/// `ordered` selects the source's default frequency/luma sort for >=3 channels;
/// otherwise explicit colors retain first appearance order. A single channel
/// always uses the source's sorted explicit palette. The maximum limits initial
/// explicit candidates, NOT the later promotion of frequent implicit colors.
#[allow(clippy::too_many_arguments)]
pub fn palette_forward(
    planes: &[Vec<i32>],
    begin: usize,
    num_channels: usize,
    max_colors: u32,
    bit_depth: u32,
    ordered: bool,
) -> Result<Option<PaletteOutput>> {
    let end = channel_range(planes, begin, num_channels)?;
    palette_forward_with_options(
        planes,
        begin,
        end - begin,
        planes[begin].len(),
        1,
        max_colors,
        bit_depth,
        ordered,
        PaletteTools::default(),
    )
}

/// Dimension-aware lossless palette entry point. Ordering is applied before
/// building the table and remapping every index, including implicit promotions.
/// Adjacency uses these actual dimensions, not the flattened-length row shape.
/// Geometry/order do not affect single-channel or `ordered=false` palettes.
#[allow(clippy::too_many_arguments)]
pub fn palette_forward_with_options(
    planes: &[Vec<i32>],
    begin: usize,
    num_channels: usize,
    width: usize,
    height: usize,
    max_colors: u32,
    bit_depth: u32,
    ordered: bool,
    tools: PaletteTools,
) -> Result<Option<PaletteOutput>> {
    let end = spatial_channel_range(planes, begin, num_channels, width, height)?;
    if bit_depth == 0 || bit_depth > 32 {
        return Err(Error::InvalidInput("palette depth must be 1..=32".into()));
    }
    if bit_depth == 32
        || max_colors < 2
        || planes[begin].is_empty()
        || (num_channels > 1 && bit_depth > 24)
    {
        return Ok(None);
    }
    let planes = &planes[begin..end];
    let len = planes[0].len();
    if num_channels == 1 {
        let samples = &planes[0];
        let (min, max) = samples
            .iter()
            .fold((samples[0], samples[0]), |(min, max), &v| {
                (min.min(v), max.max(v))
            });
        let range = i64::from(max) - i64::from(min) + 1;
        if range <= MAX_PALETTE_LOOKUP_TABLE_SIZE as i64 {
            let mut lookup = zeroed_samples(range as usize)?;
            let offset = |sample: i32| (i64::from(sample) - i64::from(min)) as usize;
            let mut count = 0i32;
            for &sample in samples {
                let present = &mut lookup[offset(sample)];
                if *present == 0 {
                    *present = 1;
                    count += 1;
                    if count as u32 > max_colors {
                        return Ok(None);
                    }
                }
            }
            let mut entries = zeroed_samples(count as usize)?;
            let mut index = 0i32;
            // Count first, then replace presence flags by indices. Index zero
            // is valid and must not be treated as unseen in a later count pass.
            for (i, entry) in lookup.iter_mut().enumerate() {
                if *entry != 0 {
                    entries[index as usize] = i32::try_from(i64::from(min) + i as i64)
                        .map_err(|_| Error::Limit("palette value exceeds i32"))?;
                    *entry = index;
                    index += 1;
                }
            }
            let mut indices = zeroed_samples(len)?;
            for (index, &sample) in indices.iter_mut().zip(samples) {
                *index = lookup[offset(sample)];
            }
            return Ok(Some(PaletteOutput {
                palette: vec![entries],
                indices,
            }));
        }
        // Source's large-range std::set path, with exact sorted indices.
        let mut palette = BTreeMap::new();
        for &sample in &planes[0] {
            palette.insert(sample, 0i32);
            if palette.len() > max_colors as usize {
                return Ok(None);
            }
        }
        let mut entries = zeroed_samples(palette.len())?;
        for (i, (&value, index)) in palette.iter_mut().enumerate() {
            *index = i32::try_from(i).map_err(|_| Error::Limit("palette index"))?;
            entries[i] = value;
        }
        let mut indices = zeroed_samples(len)?;
        for (index, value) in indices.iter_mut().zip(&planes[0]) {
            *index = palette[value];
        }
        return Ok(Some(PaletteOutput {
            palette: vec![entries],
            indices,
        }));
    }

    let implicit: Vec<Vec<i32>> = (0..IMPLICIT_COLORS)
        .map(|k| {
            (0..num_channels)
                .map(|c| implicit_palette_value(k, c, bit_depth))
                .collect()
        })
        .collect();
    let mut colors = palette_order::PaletteMap::new(num_channels, tools.mapping);
    for color in &implicit {
        colors.get(color)?.implicit = true;
    }
    let mut candidates = Vec::new();
    let mut implicit_used = 0;
    let mut color = vec![0; num_channels];
    for (x, &first) in planes[0].iter().enumerate() {
        color[0] = first;
        for (value, plane) in color[1..].iter_mut().zip(&planes[1..]) {
            *value = plane[x];
        }
        let entry = colors.get(&color)?;
        if !entry.seen {
            entry.seen = true;
            if entry.implicit {
                implicit_used += 1;
            } else {
                candidates.push(color.clone());
                if candidates.len() > max_colors as usize {
                    return Ok(None);
                }
            }
        }
        entry.count += 1;
    }
    if candidates.len() + implicit_used == 1 {
        return Ok(None);
    }
    for color in &implicit {
        if colors.get(color)?.count > 10 {
            // Preserve source duplicate promotion at low depths / 2 channels.
            candidates.push(color.clone());
        }
    }
    let palette_size = candidates.len();
    if palette_size > i32::MAX as usize - IMPLICIT_COLORS {
        return Err(Error::Limit("palette index exceeds i32"));
    }
    for (k, color) in implicit.iter().enumerate() {
        // Low depths and <3 channels have duplicate colors: LAST index wins.
        colors.get(color)?.index = palette_size + k;
    }
    if ordered {
        palette_order::reorder(&mut candidates, &mut colors, planes, width, height, tools)?;
    }
    let mut palette = Vec::new();
    palette
        .try_reserve_exact(num_channels)
        .map_err(|_| Error::Limit("palette allocation"))?;
    for _ in 0..num_channels {
        palette.push(zeroed_samples(palette_size)?);
    }
    for (index, color) in candidates.into_iter().enumerate() {
        for c in 0..num_channels {
            palette[c][index] = color[c];
        }
        colors.get(&color)?.index = index;
    }
    let mut indices = zeroed_samples(len)?;
    for (x, index) in indices.iter_mut().enumerate() {
        for (value, plane) in color.iter_mut().zip(planes) {
            *value = plane[x];
        }
        *index = colors.get(&color)?.index as i32;
    }
    Ok(Some(PaletteOutput { palette, indices }))
}

/// Sample-level output of `FwdExactDeltaPalette` or `AddExactZeroDelta`.
/// The first `num_deltas` columns are added to the prediction; the following
/// `num_colors` columns are absolute colors. Their sum is the palette width.
/// Implicit absolute cube indices are relative to that TOTAL palette width.
/// The caller serializes these parameters and performs meta-channel insertion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeltaPaletteOutput {
    pub palette: Vec<Vec<i32>>,
    pub indices: Vec<i32>,
    pub num_colors: u32,
    pub num_deltas: u32,
    pub predictor: Predictor,
}

fn spatial_channel_range(
    planes: &[Vec<i32>],
    begin: usize,
    num_channels: usize,
    width: usize,
    height: usize,
) -> Result<usize> {
    let end = channel_range(planes, begin, num_channels)?;
    let len = width
        .checked_mul(height)
        .ok_or(Error::Limit("palette dimensions"))?;
    if planes[begin].len() != len {
        return Err(Error::InvalidInput(
            "palette dimensions must match tightly packed planes".into(),
        ));
    }
    Ok(end)
}

fn palette_prediction(
    plane: &[i32],
    width: usize,
    pos: usize,
    predictor: Predictor,
) -> Result<i64> {
    let neighbors = Neighbors::from_plane(plane, width, width, pos % width, pos / width)
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    predict_one(predictor, neighbors, None).map_err(|e| Error::InvalidInput(e.to_string()))
}

/// `enc_palette_delta.h::FwdExactDeltaPalette`, with no channel metadata edits.
/// `max_colors` is the source's INPUT color budget; the successful output has
/// zero absolute colors and <=max_colors exact residual tuples. `None` means
/// ineligible inputs, an unrepresentable i32 difference, or budget exhaustion;
/// malformed ranges/dimensions are errors. Inputs are unchanged in every case.
#[allow(clippy::too_many_arguments)]
pub fn delta_palette_forward(
    planes: &[Vec<i32>],
    begin: usize,
    num_channels: usize,
    width: usize,
    height: usize,
    max_colors: u32,
    bit_depth: u32,
    predictor: Predictor,
) -> Result<Option<DeltaPaletteOutput>> {
    let end = spatial_channel_range(planes, begin, num_channels, width, height)?;
    if !(3..=4).contains(&num_channels)
        || !(1..=16).contains(&bit_depth)
        || !(1..=65536).contains(&max_colors)
        || predictor.has_weighted()
        || width == 0
        || height == 0
    {
        return Ok(None);
    }
    let planes = &planes[begin..end];
    let mut indices = zeroed_samples(planes[0].len())?;
    let mut residual = zeroed_samples(num_channels)?;
    let mut lookup: BTreeMap<Vec<i32>, usize> = BTreeMap::new();
    let mut entries: Vec<(Vec<i32>, usize)> = Vec::new();
    for (pos, index) in indices.iter_mut().enumerate() {
        for (c, plane) in planes.iter().enumerate() {
            let prediction = palette_prediction(plane, width, pos, predictor)?;
            let difference = i64::from(plane[pos]) - prediction;
            let Ok(difference) = i32::try_from(difference) else {
                return Ok(None);
            };
            residual[c] = difference;
        }
        let entry = if let Some(&entry) = lookup.get(residual.as_slice()) {
            entry
        } else {
            if entries.len() == max_colors as usize {
                return Ok(None);
            }
            let entry = entries.len();
            lookup.insert(residual.clone(), entry);
            entries.push((residual.clone(), 0));
            entry
        };
        entries[entry].1 += 1;
        *index = entry as i32;
    }
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        entries[b]
            .1
            .cmp(&entries[a].1)
            .then_with(|| entries[a].0.cmp(&entries[b].0))
    });
    let mut remap = zeroed_samples(entries.len())?;
    let mut palette = Vec::new();
    palette
        .try_reserve_exact(num_channels)
        .map_err(|_| Error::Limit("delta palette allocation"))?;
    for _ in 0..num_channels {
        palette.push(zeroed_samples(entries.len())?);
    }
    for (index, &entry) in order.iter().enumerate() {
        remap[entry] = index as i32;
        for (channel, &value) in palette.iter_mut().zip(&entries[entry].0) {
            channel[index] = value;
        }
    }
    for index in &mut indices {
        *index = remap[*index as usize];
    }
    Ok(Some(DeltaPaletteOutput {
        palette,
        indices,
        num_colors: 0,
        num_deltas: entries.len() as u32,
        predictor,
    }))
}

/// `enc_palette_delta.h::ExactZeroDeltaMask` on ORIGINAL group-local planes.
/// A 1 means every selected sample equals its (wide) prediction exactly.
/// No i32 narrowing or residual wrapping is used when deciding equality.
#[allow(clippy::too_many_arguments)]
pub fn exact_zero_delta_mask(
    planes: &[Vec<i32>],
    begin: usize,
    num_channels: usize,
    width: usize,
    height: usize,
    bit_depth: u32,
    predictor: Predictor,
) -> Result<Option<Vec<u8>>> {
    let end = spatial_channel_range(planes, begin, num_channels, width, height)?;
    if !(3..=4).contains(&num_channels)
        || !(1..=16).contains(&bit_depth)
        || predictor.has_weighted()
        || width == 0
        || height == 0
    {
        return Ok(None);
    }
    let planes = &planes[begin..end];
    let mut mask = Vec::new();
    mask.try_reserve_exact(planes[0].len())
        .map_err(|_| Error::Limit("zero delta mask allocation"))?;
    mask.resize(planes[0].len(), 1);
    for (pos, entry) in mask.iter_mut().enumerate() {
        for plane in planes {
            if i64::from(plane[pos]) != palette_prediction(plane, width, pos, predictor)? {
                *entry = 0;
                break;
            }
        }
    }
    Ok(Some(mask))
}

/// `enc_palette_delta.h::AddExactZeroDelta` for a zero-delta ordinary palette.
/// The mask MUST come from the same original channels, geometry and predictor.
/// Nonzero mask entries select the new all-zero delta column; other indices
/// (including implicit cube indices) shift by one. Even an all-zero mask adds
/// the column, matching the source. All validation occurs before allocation.
pub fn add_exact_zero_delta(
    input: &PaletteOutput,
    predictor: Predictor,
    mask: &[u8],
) -> Result<DeltaPaletteOutput> {
    let Some(first) = input.palette.first() else {
        return Err(Error::InvalidInput(
            "zero delta requires a palette table".into(),
        ));
    };
    if input.palette.iter().any(|row| row.len() != first.len())
        || mask.len() != input.indices.len()
        || predictor.has_weighted()
        || input
            .indices
            .iter()
            .any(|&index| index < 0 || index == i32::MAX)
    {
        return Err(Error::InvalidInput(
            "invalid zero delta palette, predictor or mask".into(),
        ));
    }
    let num_colors =
        u32::try_from(first.len()).map_err(|_| Error::Limit("zero delta palette width"))?;
    let width = first
        .len()
        .checked_add(1)
        .ok_or(Error::Limit("zero delta palette width"))?;
    if width > i32::MAX as usize {
        return Err(Error::Limit("zero delta palette index"));
    }
    let mut palette = Vec::new();
    palette
        .try_reserve_exact(input.palette.len())
        .map_err(|_| Error::Limit("zero delta palette allocation"))?;
    for row in &input.palette {
        let mut extended = zeroed_samples(width)?;
        extended[1..].copy_from_slice(row);
        palette.push(extended);
    }
    let mut indices = zeroed_samples(input.indices.len())?;
    for (pos, index) in indices.iter_mut().enumerate() {
        *index = if mask[pos] != 0 {
            0
        } else {
            input.indices[pos] + 1
        };
    }
    Ok(DeltaPaletteOutput {
        palette,
        indices,
        num_colors,
        num_deltas: 1,
        predictor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Direct scalar inverse from rct.cc::InvRCTRow, independent of the forward
    // permutation formula. Full-range wrapping matches its PixelAdd operations.
    fn inverse_rct(planes: &mut [Vec<i32>], begin: usize, rct_type: u32) {
        const PERMUTATIONS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
            [1, 0, 2],
            [2, 1, 0],
        ];
        let permutation = PERMUTATIONS[(rct_type / 7) as usize];
        let transform = rct_type % 7;
        for x in 0..planes[begin].len() {
            let first = planes[begin][x];
            let mut second = planes[begin + 1][x];
            let mut third = planes[begin + 2][x];
            let values = if transform == 6 {
                let tmp = first.wrapping_sub(third >> 1);
                let g = third.wrapping_add(tmp);
                let b = tmp.wrapping_sub(second >> 1);
                [b.wrapping_add(second), g, b]
            } else {
                if transform & 1 != 0 {
                    third = third.wrapping_add(first);
                }
                if transform >> 1 == 1 {
                    second = second.wrapping_add(first);
                }
                if transform >> 1 == 2 {
                    second = second.wrapping_add(first.wrapping_add(third) >> 1);
                }
                [first, second, third]
            };
            for i in 0..3 {
                planes[begin + permutation[i]][x] = values[i];
            }
        }
    }

    #[test]
    fn all_42_rct_source_equation_goldens() {
        // Evaluated from the source equations, not captured from a C++ binary.
        // Negative and odd components distinguish shifts from truncating divide.
        let expected = [
            [17, -9, 4],
            [17, -9, -13],
            [17, -26, 4],
            [17, -26, -13],
            [17, -19, 4],
            [17, -19, -13],
            [0, 13, -19],
            [-9, 4, 17],
            [-9, 4, 26],
            [-9, 13, 17],
            [-9, 13, 26],
            [-9, 0, 17],
            [-9, 0, 26],
            [4, -26, 0],
            [4, 17, -9],
            [4, 17, -13],
            [4, 13, -9],
            [4, 13, -13],
            [4, 20, -9],
            [4, 20, -13],
            [7, 13, 20],
            [17, 4, -9],
            [17, 4, -26],
            [17, -13, -9],
            [17, -13, -26],
            [17, 0, -9],
            [17, 0, -26],
            [4, 26, 0],
            [-9, 17, 4],
            [-9, 17, 13],
            [-9, 26, 4],
            [-9, 26, 13],
            [-9, 20, 4],
            [-9, 20, 13],
            [7, -13, 20],
            [4, -9, 17],
            [4, -9, 13],
            [4, -13, 17],
            [4, -13, 13],
            [4, -19, 17],
            [4, -19, 13],
            [0, -13, -19],
        ];
        for (rct_type, golden) in expected.iter().enumerate() {
            let mut planes = vec![vec![999], vec![17], vec![-9], vec![4], vec![0]];
            let original = planes.clone();
            rct_forward(&mut planes, 1, rct_type as u32).unwrap();
            assert_eq!(
                [planes[1][0], planes[2][0], planes[3][0]],
                *golden,
                "RCT {rct_type}"
            );
            assert_eq!(planes[0], original[0]);
            assert_eq!(planes[4], original[4]);
            inverse_rct(&mut planes, 1, rct_type as u32);
            assert_eq!(planes, original);
        }
    }

    #[test]
    fn rct_wraps_before_shifting_and_subtracting() {
        for (input, rct_type, expected) in [
            ([i32::MAX, 0, i32::MAX], 4, [i32::MAX, 1, i32::MAX]),
            ([i32::MIN, 0, i32::MAX], 5, [i32::MIN, 1, -1]),
            ([i32::MAX, i32::MIN, i32::MIN], 6, [i32::MAX, -1, 1]),
            (
                [i32::MIN, i32::MAX, 0],
                6,
                [-1610612737, i32::MIN, -1073741825],
            ),
        ] {
            let mut planes = input.map(|v| vec![v]);
            rct_forward(&mut planes, 0, rct_type).unwrap();
            assert_eq!(planes.each_ref().map(|p| p[0]), expected);
            inverse_rct(&mut planes, 0, rct_type);
            assert_eq!(planes.each_ref().map(|p| p[0]), input);
        }
    }

    #[test]
    fn every_rct_roundtrips_full_i32_domain_samples_and_preserves_alpha() {
        let values = [
            i32::MIN,
            i32::MIN + 1,
            -65535,
            -9,
            -1,
            0,
            1,
            7,
            65535,
            i32::MAX,
        ];
        let mut planes = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    planes[0].push(a);
                    planes[1].push(b);
                    planes[2].push(c);
                    planes[3].push(0); // invisible RGB remains reversible
                }
            }
        }
        for rct_type in 0..42 {
            let mut transformed = planes.clone();
            rct_forward(&mut transformed, 0, rct_type).unwrap();
            assert_eq!(transformed[3], planes[3]);
            inverse_rct(&mut transformed, 0, rct_type);
            assert_eq!(transformed, planes, "RCT {rct_type}");
        }
    }

    #[test]
    fn rct_range_and_shape_validation_is_atomic_including_noop() {
        let input = vec![vec![1], vec![2], vec![3]];
        for (begin, rct_type) in [(0, 42), (0, u32::MAX), (1, 6), (usize::MAX, 0)] {
            let mut planes = input.clone();
            assert!(rct_forward(&mut planes, begin, rct_type).is_err());
            assert_eq!(planes, input);
        }
        let mut mismatch = vec![vec![1], vec![2, 3], vec![4]];
        let original = mismatch.clone();
        for rct_type in [0, 6] {
            assert!(rct_forward(&mut mismatch, 0, rct_type).is_err());
            assert_eq!(mismatch, original);
        }
        for rct_type in 0..42 {
            rct_forward(&mut [vec![], vec![], vec![]], 0, rct_type).unwrap();
        }
    }

    // Scalar inverse equations in squeeze.cc. Inverse uses division toward
    // zero for diff/2; forward AVERAGE instead uses arithmetic right shift.
    fn inverse_squeeze(s: &SqueezedPlane, horizontal: bool) -> Vec<i32> {
        let (width, height) = if horizontal {
            (s.average_width + s.residual_width, s.average_height)
        } else {
            (s.average_width, s.average_height + s.residual_height)
        };
        let mut out = vec![0; width * height];
        for y in 0..s.residual_height {
            for x in 0..s.residual_width {
                let ai = y * s.average_width + x;
                let avg = s.average[ai];
                let next = if horizontal && x + 1 < s.average_width {
                    s.average[ai + 1]
                } else if !horizontal && y + 1 < s.average_height {
                    s.average[ai + s.average_width]
                } else {
                    avg
                };
                let oi = if horizontal {
                    y * width + 2 * x
                } else {
                    2 * y * width + x
                };
                let step = if horizontal { 1 } else { width };
                let previous = if (horizontal && x > 0) || (!horizontal && y > 0) {
                    out[oi - step]
                } else {
                    avg
                };
                let diff = i64::from(s.residual[y * s.residual_width + x])
                    + smooth_tendency(previous, avg, next);
                let a = i64::from(avg) + diff / 2;
                out[oi] = i32::try_from(a).unwrap();
                out[oi + step] = i32::try_from(a - diff).unwrap();
            }
        }
        if horizontal && width & 1 != 0 {
            for y in 0..height {
                out[y * width + width - 1] = s.average[y * s.average_width + s.average_width - 1];
            }
        }
        if !horizontal && height & 1 != 0 {
            out[(height - 1) * width..]
                .copy_from_slice(&s.average[(s.average_height - 1) * width..]);
        }
        out
    }

    #[test]
    fn squeeze_source_equation_goldens_and_negative_rounding() {
        let pixels = [-9, -4, -1, 4, 12, 6, 2, -3, -8, -13, 10, 8, 6, 4, 2];
        let horizontal = squeeze_forward(&pixels, 5, 3, true).unwrap();
        assert_eq!(horizontal.average, [-7, 1, 12, 4, -5, -13, 9, 5, 2]);
        assert_eq!(horizontal.residual, [-4, -1, 3, 1, 1, 0]);
        assert_eq!(inverse_squeeze(&horizontal, true), pixels);
        let vertical = squeeze_forward(&pixels, 5, 3, false).unwrap();
        assert_eq!(vertical.average, [-2, -1, -2, -2, 0, 10, 8, 6, 4, 2]);
        assert_eq!(vertical.residual, [-14, -5, 3, 13, 26]);
        assert_eq!(inverse_squeeze(&vertical, false), pixels);
        assert_eq!(squeeze_average(-9, -4).unwrap(), -7);
        assert_eq!(squeeze_average(-4, -9).unwrap(), -6);
        assert_eq!(squeeze_average(-1, 0).unwrap(), -1);
        assert_eq!(squeeze_average(0, -1).unwrap(), 0);
        for (b, a, n, expected) in [
            (0, 0, 1, 0),
            (0, 1, 1, 0),
            (0, 1, 2, -1),
            (8, 5, 0, 2),
            (-8, -5, 0, -2),
            (0, 5, 8, -2),
            (-1, -3, -8, 2),
            (9, 0, 4, 0),
            (0, 9, 4, 0),
        ] {
            assert_eq!(smooth_tendency(b, a, n), expected);
        }
    }

    #[test]
    fn squeeze_roundtrips_small_dimensions_and_odd_boundaries() {
        let mut random = 37u32;
        for width in 1..=13 {
            for height in 1..=11 {
                let samples: Vec<i32> = (0..width * height)
                    .map(|_| {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        (random % 131071) as i32 - 65535
                    })
                    .collect();
                for horizontal in [true, false] {
                    let split = squeeze_forward(&samples, width, height, horizontal).unwrap();
                    assert_eq!(split.average.len() + split.residual.len(), samples.len());
                    assert_eq!(inverse_squeeze(&split, horizontal), samples);
                }
            }
        }
        for value in [i32::MIN, i32::MAX] {
            for horizontal in [true, false] {
                let split = squeeze_forward(&[value], 1, 1, horizontal).unwrap();
                assert_eq!(split.average, [value]);
                assert!(split.residual.is_empty());
            }
        }
    }

    #[test]
    fn squeeze_declines_undefined_scalar_overflow_and_invalid_shapes() {
        assert!(squeeze_forward(&[], 0, 1, true).is_err());
        assert!(squeeze_forward(&[1], 1, 0, false).is_err());
        assert!(squeeze_forward(&[1], 2, 1, true).is_err());
        assert!(squeeze_forward(&[], usize::MAX, 2, true).is_err());
        assert!(squeeze_forward(&[i32::MAX, i32::MAX], 2, 1, true).is_err());
        assert!(squeeze_forward(&[i32::MIN, i32::MIN], 2, 1, true).is_err());
        assert!(squeeze_forward(&[i32::MIN, i32::MAX], 2, 1, true).is_err());
    }

    fn expand_palette(output: &PaletteOutput, depth: u32) -> Vec<Vec<i32>> {
        let size = output.palette[0].len();
        output
            .palette
            .iter()
            .enumerate()
            .map(|(c, entries)| {
                output
                    .indices
                    .iter()
                    .map(|&i| {
                        assert!(i >= 0);
                        if (i as usize) < size {
                            entries[i as usize]
                        } else {
                            implicit_palette_value(i as usize - size, c, depth)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn planes(colors: &[[i32; 3]]) -> Vec<Vec<i32>> {
        (0..3)
            .map(|c| colors.iter().map(|color| color[c]).collect())
            .collect()
    }

    #[test]
    fn single_channel_palette_is_sorted_and_does_not_discard_extremes() {
        let samples = vec![vec![999], vec![-8, 5, -8, 2, 5], vec![42]];
        let result = palette_forward(&samples, 1, 1, 3, 16, false)
            .unwrap()
            .unwrap();
        assert_eq!(result.palette, [vec![-8, 2, 5]]);
        assert_eq!(result.indices, [0, 2, 0, 1, 2]);
        assert_eq!(expand_palette(&result, 16), samples[1..2]);
        let samples = vec![vec![i32::MIN, i32::MAX, 0, i32::MIN]];
        let result = palette_forward(&samples, 0, 1, 3, 31, true)
            .unwrap()
            .unwrap();
        assert_eq!(result.palette, [vec![i32::MIN, 0, i32::MAX]]);
        assert_eq!(expand_palette(&result, 31), samples);
        // The source single-channel branch accepts constant channels.
        assert!(
            palette_forward(&[vec![7; 10]], 0, 1, 2, 8, false)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn single_channel_dense_lut_signed_holes_have_exact_dictionary_and_indices() {
        let samples = vec![3, -5, 1, -5, -4, 3, 0, -2, 1];
        let padded = vec![vec![999], samples.clone(), vec![-999; 2]];
        let expected = PaletteOutput {
            palette: vec![vec![-5, -4, -2, 0, 1, 3]],
            indices: vec![5, 0, 4, 0, 1, 5, 3, 2, 4],
        };
        for ordered in [false, true] {
            let result = palette_forward_with_options(
                &padded,
                1,
                1,
                3,
                3,
                6,
                16,
                ordered,
                PaletteTools {
                    order: PaletteOrder::Adjacency,
                    geometry: PaletteGeometry::FrequencyBandsPrincipal,
                    mapping: PaletteMapping::Flat,
                },
            )
            .unwrap()
            .unwrap();
            assert_eq!(result, expected);
            assert_eq!(expand_palette(&result, 16), std::slice::from_ref(&samples));
            assert!(
                palette_forward(&padded, 1, 1, 5, 16, ordered)
                    .unwrap()
                    .is_none()
            );
        }
        assert_eq!(padded, [vec![999], samples, vec![-999; 2]]);
    }

    #[test]
    fn single_channel_lut_span_cutoff_and_extreme_offsets_do_not_overflow() {
        assert_eq!(MAX_PALETTE_LOOKUP_TABLE_SIZE, 65536);
        // The first two spans use the LUT; the third must use the tree even
        // though all three have only four distinct colors. Include all-negative,
        // cross-zero and near-MAX ranges with the same dictionary/index pattern.
        for span in [65535, 65536, 65537] {
            for min in [i32::MIN, -32769, i32::MAX - (span - 1)] {
                let max = i32::try_from(i64::from(min) + i64::from(span) - 1).unwrap();
                let input = vec![vec![max, min, min + 1, max - 1, min, max, min + 1]];
                let result = palette_forward_with_options(
                    &input,
                    0,
                    1,
                    1,
                    7,
                    4,
                    31,
                    false,
                    PaletteTools::default(),
                )
                .unwrap()
                .unwrap();
                assert_eq!(result.palette, [vec![min, min + 1, max - 1, max]]);
                assert_eq!(result.indices, [3, 0, 1, 2, 0, 3, 1]);
                assert_eq!(expand_palette(&result, 31), input);
                assert!(
                    palette_forward(&input, 0, 1, 3, 31, true)
                        .unwrap()
                        .is_none()
                );
            }
        }
        // The full signed span is 2^32, not zero, even on a 32-bit target.
        let input = vec![vec![
            i32::MAX,
            i32::MIN,
            i32::MIN + 1,
            i32::MAX - 1,
            i32::MIN,
        ]];
        let result = palette_forward(&input, 0, 1, 4, 31, true).unwrap().unwrap();
        assert_eq!(
            result.palette,
            [vec![i32::MIN, i32::MIN + 1, i32::MAX - 1, i32::MAX]]
        );
        assert_eq!(result.indices, [3, 0, 1, 2, 0]);
        assert_eq!(expand_palette(&result, 31), input);
    }

    #[test]
    fn single_channel_lut_full_cardinality_and_tree_fallback_obey_exact_caps() {
        for count in [65535usize, 65536, 65537] {
            let dictionary: Vec<_> = (0..count).map(|i| -32769 + i as i32).collect();
            let mut samples: Vec<_> = dictionary.iter().rev().copied().collect();
            samples.extend([dictionary[0], dictionary[count - 1], dictionary[0]]);
            let mut indices: Vec<_> = (0..count as i32).rev().collect();
            indices.extend([0, count as i32 - 1, 0]);
            let input = vec![samples];
            assert!(
                palette_forward(&input, 0, 1, count as u32 - 1, 31, true)
                    .unwrap()
                    .is_none()
            );
            for cap in [count as u32, count as u32 + 1] {
                let result = palette_forward(&input, 0, 1, cap, 31, true)
                    .unwrap()
                    .unwrap();
                assert_eq!(result.palette[0], dictionary);
                assert_eq!(result.indices, indices);
                assert_eq!(expand_palette(&result, 31), input);
            }
        }
    }

    #[test]
    fn single_channel_lut_constant_empty_and_depth_gates_match_source() {
        for sample in [i32::MIN, -9, 0, i32::MAX] {
            let input = vec![vec![sample; 5]];
            for cap in [0, 1] {
                assert!(
                    palette_forward(&input, 0, 1, cap, 31, false)
                        .unwrap()
                        .is_none()
                );
            }
            let result = palette_forward(&input, 0, 1, 2, 31, false)
                .unwrap()
                .unwrap();
            assert_eq!(result.palette, [vec![sample]]);
            assert_eq!(result.indices, [0; 5]);
            assert_eq!(expand_palette(&result, 31), input);
            assert!(
                palette_forward(&input, 0, 1, 2, 32, true)
                    .unwrap()
                    .is_none()
            );
            for depth in [0, 33] {
                assert!(palette_forward(&input, 0, 1, 2, depth, true).is_err());
            }
        }
        assert!(
            palette_forward(&[vec![]], 0, 1, 2, 31, true)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn palette_image_order_and_both_implicit_cubes_have_fixed_indices() {
        let input = planes(&[
            [1, 2, 3],
            [0, 0, 0],
            [255, 255, 255],
            [1, 2, 3],
            [32, 32, 32],
            [7, 11, 13],
        ]);
        let result = palette_forward(&input, 0, 3, 2, 8, false).unwrap().unwrap();
        assert_eq!(result.palette, [vec![1, 7], vec![2, 11], vec![3, 13]]);
        assert_eq!(result.indices, [0, 66, 190, 0, 2, 1]);
        assert_eq!(expand_palette(&result, 8), input);
        assert_eq!(implicit_palette_value(0, 0, 8), 32);
        assert_eq!(implicit_palette_value(63, 2, 8), 223);
        assert_eq!(implicit_palette_value(64, 0, 8), 0);
        assert_eq!(implicit_palette_value(188, 2, 8), 255);
        assert_eq!(implicit_palette_value(0, 3, 8), 0);
    }

    #[test]
    fn frequent_implicit_colors_promote_only_above_ten() {
        for count in [10, 11] {
            let mut colors = vec![[0, 0, 0]; count];
            colors.push([1, 2, 3]);
            let input = planes(&colors);
            let result = palette_forward(&input, 0, 3, 2, 8, false).unwrap().unwrap();
            assert_eq!(result.palette[0].len(), if count == 10 { 1 } else { 2 });
            assert_eq!(result.indices[0], if count == 10 { 65 } else { 1 });
            assert_eq!(result.indices[count], 0);
            assert_eq!(expand_palette(&result, 8), input);
        }
    }

    #[test]
    fn default_luma_order_matches_source_and_transparent_rgb_survives() {
        let mut colors = vec![[120, 120, 120], [10, 10, 10]];
        colors.extend([[90, 90, 90]; 5]);
        let input = planes(&colors);
        let result = palette_forward(&input, 0, 3, 3, 8, true).unwrap().unwrap();
        assert_eq!(result.palette[0], [90, 10, 120]);
        assert_eq!(expand_palette(&result, 8), input);
        let mut rgba = planes(&[[1, 2, 3], [7, 11, 13], [0, 0, 0], [0, 0, 0]]);
        rgba.push(vec![0, 0, 0, 255]);
        for ordered in [false, true] {
            let result = palette_forward(&rgba, 0, 4, 4, 8, ordered)
                .unwrap()
                .unwrap();
            assert_eq!(expand_palette(&result, 8), rgba);
        }
    }

    #[test]
    fn implicit_cube_low_depth_aliases_and_empty_explicit_palettes_roundtrip() {
        for depth in [1, 2, 3, 8, 16, 24] {
            for channels in [2, 3, 4] {
                let input: Vec<Vec<i32>> = (0..channels)
                    .map(|c| {
                        (0..IMPLICIT_COLORS)
                            .map(|k| implicit_palette_value(k, c, depth))
                            .collect()
                    })
                    .collect();
                let result = palette_forward(&input, 0, channels, 2, depth, false)
                    .unwrap()
                    .unwrap();
                assert_eq!(expand_palette(&result, depth), input);
                if depth == 8 && channels == 3 {
                    assert!(result.palette[0].is_empty());
                }
            }
        }
    }

    #[test]
    fn palette_declines_ineligible_inputs_and_rejects_bad_ranges() {
        let input = planes(&[[1, 2, 3], [7, 11, 13], [14, 15, 16]]);
        for (limit, depth) in [(1, 8), (2, 8), (3, 25), (3, 32)] {
            assert!(
                palette_forward(&input, 0, 3, limit, depth, false)
                    .unwrap()
                    .is_none()
            );
        }
        assert!(
            palette_forward(&planes(&[[1, 2, 3]; 5]), 0, 3, 2, 8, false)
                .unwrap()
                .is_none()
        );
        for (begin, count, depth) in [
            (0, 0, 8),
            (2, 3, 8),
            (usize::MAX, 3, 8),
            (0, 3, 0),
            (0, 3, 33),
        ] {
            assert!(palette_forward(&input, begin, count, 3, depth, false).is_err());
        }
        assert!(palette_forward(&[vec![1], vec![1, 2]], 0, 2, 3, 8, false).is_err());
        assert!(
            palette_forward(&[vec![]], 0, 1, 3, 8, false)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn every_optional_palette_order_and_map_reconstructs_signed_and_hidden_samples() {
        let tuples = [
            [0, 0, 0, 0, 0],
            [i32::MIN, 7, i32::MAX, 0, 3],
            [7, 11, 13, 0, 9],
            [7, 11, 13, 255, -8],
            [60001, -30000, 501, 0, 0],
            [65535, 65535, 65535, 0, 0],
        ];
        for count in [3, 4, 5] {
            let input: Vec<Vec<i32>> = (0..count)
                .map(|c| {
                    (0..70)
                        .map(|i| {
                            tuples[if i == 69 {
                                5
                            } else {
                                [0, 0, 1, 2, 3, 0, 4][i % 7]
                            }][c]
                        })
                        .collect()
                })
                .collect();
            for source_order in 0..=9 {
                for source_geometry in 0..=2 {
                    for ordered in [false, true] {
                        let tools = PaletteTools {
                            order: PaletteOrder::from_source_mode(source_order).unwrap(),
                            geometry: PaletteGeometry::from_source_mode(source_geometry).unwrap(),
                            mapping: PaletteMapping::Tree,
                        };
                        let tree = palette_forward_with_options(
                            &input, 0, count, 7, 10, 16, 16, ordered, tools,
                        )
                        .unwrap()
                        .unwrap();
                        let flat = palette_forward_with_options(
                            &input,
                            0,
                            count,
                            7,
                            10,
                            16,
                            16,
                            ordered,
                            PaletteTools {
                                mapping: PaletteMapping::Flat,
                                ..tools
                            },
                        )
                        .unwrap()
                        .unwrap();
                        assert_eq!(
                            tree, flat,
                            "order={source_order} geometry={source_geometry} channels={count}"
                        );
                        assert_eq!(expand_palette(&flat, 16), input);
                        if !ordered {
                            assert_eq!(
                                flat,
                                palette_forward(&input, 0, count, 16, 16, false)
                                    .unwrap()
                                    .unwrap()
                            );
                        }
                        if count > 4 && source_order >= 6 {
                            let morton = palette_forward_with_options(
                                &input,
                                0,
                                count,
                                7,
                                10,
                                16,
                                16,
                                ordered,
                                PaletteTools {
                                    order: PaletteOrder::Morton,
                                    ..PaletteTools::default()
                                },
                            )
                            .unwrap()
                            .unwrap();
                            assert_eq!(flat, morton);
                        } else if count <= 4 && source_geometry != 0 {
                            let principal = palette_forward_with_options(
                                &input,
                                0,
                                count,
                                7,
                                10,
                                16,
                                16,
                                ordered,
                                PaletteTools {
                                    order: PaletteOrder::Default,
                                    ..tools
                                },
                            )
                            .unwrap()
                            .unwrap();
                            assert_eq!(flat, principal);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn optional_tools_preserve_implicit_aliases_promotions_and_empty_tables() {
        for depth in [1, 2, 8, 24] {
            for count in [2, 3, 4] {
                for mode in 0..=9 {
                    let input: Vec<Vec<i32>> = (0..count)
                        .map(|c| {
                            (0..IMPLICIT_COLORS)
                                .map(|k| implicit_palette_value(k, c, depth))
                                .collect()
                        })
                        .collect();
                    let tools = PaletteTools {
                        order: PaletteOrder::from_source_mode(mode).unwrap(),
                        ..PaletteTools::default()
                    };
                    let tree = palette_forward_with_options(
                        &input, 0, count, 21, 9, 2, depth, true, tools,
                    )
                    .unwrap()
                    .unwrap();
                    let flat = palette_forward_with_options(
                        &input,
                        0,
                        count,
                        21,
                        9,
                        2,
                        depth,
                        true,
                        PaletteTools {
                            mapping: PaletteMapping::Flat,
                            ..tools
                        },
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(tree, flat);
                    assert_eq!(expand_palette(&flat, depth), input);
                    if depth >= 8 && count >= 3 {
                        assert!(flat.palette[0].is_empty());
                    }
                    if count == 2 {
                        assert_eq!(
                            flat,
                            palette_forward(&input, 0, count, 2, depth, false)
                                .unwrap()
                                .unwrap()
                        );
                    }
                }
            }
        }
        for count in [10, 11] {
            let mut input = planes(&vec![[0, 0, 0]; count]);
            for (c, row) in input.iter_mut().enumerate() {
                row.push(c as i32 + 1);
            }
            for mode in 0..=9 {
                let result = palette_forward_with_options(
                    &input,
                    0,
                    3,
                    count + 1,
                    1,
                    2,
                    8,
                    true,
                    PaletteTools {
                        order: PaletteOrder::from_source_mode(mode).unwrap(),
                        mapping: PaletteMapping::Flat,
                        ..PaletteTools::default()
                    },
                )
                .unwrap()
                .unwrap();
                assert_eq!(result.palette[0].len(), if count == 10 { 1 } else { 2 });
                assert_eq!(expand_palette(&result, 8), input);
            }
        }
    }

    #[test]
    fn adjacency_uses_true_rows_and_source_sampling_stride() {
        let input = planes(&[[10; 3], [20; 3], [30; 3], [40; 3]]);
        let tools = PaletteTools {
            order: PaletteOrder::Adjacency,
            mapping: PaletteMapping::Flat,
            ..PaletteTools::default()
        };
        for (width, height, expected) in [
            (2, 2, [30, 10, 20, 40]),
            (4, 1, [10, 20, 30, 40]),
            (1, 4, [10, 20, 30, 40]),
        ] {
            let result =
                palette_forward_with_options(&input, 0, 3, width, height, 4, 8, true, tools)
                    .unwrap()
                    .unwrap();
            assert_eq!(result.palette[0], expected);
            assert_eq!(expand_palette(&result, 8), input);
        }
        // At 262145 pixels source samples every second position. The B-C and
        // C-A edges are at odd positions and must not enter the forest.
        let mut input = vec![vec![10; 262145]; 3];
        for plane in &mut input {
            plane[1] = 20;
            plane[2] = 30;
            plane[3] = 30;
        }
        let result = palette_forward_with_options(&input, 0, 3, 262145, 1, 3, 8, true, tools)
            .unwrap()
            .unwrap();
        assert_eq!(result.palette[0], [10, 20, 30]);
        assert_eq!(expand_palette(&result, 8), input);
    }

    #[test]
    fn optional_palette_dimension_errors_and_single_channel_bypass() {
        let input = planes(&[[1, 2, 3], [7, 8, 9]]);
        let tools = PaletteTools {
            order: PaletteOrder::Adjacency,
            geometry: PaletteGeometry::Principal,
            mapping: PaletteMapping::Flat,
        };
        for (width, height) in [(1, 1), (0, 2), (usize::MAX, 2)] {
            assert!(
                palette_forward_with_options(&input, 0, 3, width, height, 3, 8, true, tools)
                    .is_err()
            );
        }
        assert!(
            palette_forward_with_options(&input, 0, 3, 2, 1, 3, 32, true, tools)
                .unwrap()
                .is_none()
        );
        assert!(palette_forward_with_options(&[], 0, 0, 0, 0, 3, 8, true, tools).is_err());
        let single = vec![vec![i32::MAX, i32::MIN, i32::MAX, 0]];
        let result = palette_forward_with_options(&single, 0, 1, 2, 2, 3, 31, true, tools)
            .unwrap()
            .unwrap();
        assert_eq!(result.palette[0], [i32::MIN, 0, i32::MAX]);
        assert_eq!(expand_palette(&result, 31), single);
        // With alpha=-1 every default luma key is zero. Preserve image-order
        // ties rather than inventing lexicographic ties absent from C++.
        let mut tied = planes(&[[90; 3], [10; 3], [70; 3]]);
        tied.push(vec![-1; 3]);
        let result = palette_forward(&tied, 0, 4, 3, 8, true).unwrap().unwrap();
        assert_eq!(result.palette[0], [90, 10, 70]);
    }

    // palette.cc's zero/nonzero-delta split, on reconstructed neighbors rather
    // than the original planes used by the encoder. No metadata is involved.
    fn expand_delta_palette(
        output: &DeltaPaletteOutput,
        width: usize,
        depth: u32,
    ) -> Vec<Vec<i32>> {
        let palette_size = output.num_colors as usize + output.num_deltas as usize;
        assert!(output.palette.iter().all(|row| row.len() == palette_size));
        let mut planes = vec![vec![0; output.indices.len()]; output.palette.len()];
        for (c, plane) in planes.iter_mut().enumerate() {
            for (pos, &index) in output.indices.iter().enumerate() {
                assert!(index >= 0);
                let index = index as usize;
                let entry = if index < palette_size {
                    output.palette[c][index]
                } else {
                    implicit_palette_value(index - palette_size, c, depth)
                };
                plane[pos] = if index < output.num_deltas as usize {
                    let prediction =
                        palette_prediction(plane, width, pos, output.predictor).unwrap();
                    i32::try_from(i64::from(entry) + prediction).unwrap()
                } else {
                    entry
                };
            }
        }
        planes
    }

    #[test]
    fn exact_delta_palette_source_equation_golden_sorts_and_remaps() {
        let input = vec![
            vec![3, 5, 7, 6, 3, 5, 7, 6],
            vec![-3, -1, 1, 0, -3, -1, 1, 0],
            vec![5, 7, 9, 8, 5, 7, 9, 8],
        ];
        let original = input.clone();
        let result = delta_palette_forward(&input, 0, 3, 4, 2, 4, 16, Predictor::Left)
            .unwrap()
            .unwrap();
        assert_eq!(result.num_colors, 0);
        assert_eq!(result.num_deltas, 4);
        assert_eq!(result.predictor, Predictor::Left);
        assert_eq!(
            result.palette,
            [vec![2, -1, 0, 3], vec![2, -1, 0, -3], vec![2, -1, 0, 5]]
        );
        assert_eq!(result.indices, [3, 0, 0, 1, 2, 0, 0, 1]);
        assert_eq!(expand_delta_palette(&result, 4, 16), input);
        assert!(
            delta_palette_forward(&input, 0, 3, 4, 2, 3, 16, Predictor::Left)
                .unwrap()
                .is_none()
        );
        assert_eq!(input, original);

        let tied = planes(&[[1, 2, -9], [1, -2, 9], [0, 10, 10]]);
        let result = delta_palette_forward(&tied, 0, 3, 3, 1, 3, 8, Predictor::Zero)
            .unwrap()
            .unwrap();
        assert_eq!(
            result.palette,
            [vec![0, 1, 1], vec![10, -2, 2], vec![10, 9, -9]]
        );
        assert_eq!(result.indices, [2, 1, 0]);
    }

    #[test]
    fn delta_palette_roundtrips_all_source_predictors_and_group_boundaries() {
        let mut random = 0xe349_16a5u32;
        for width in [1usize, 2, 5, 9] {
            for height in [1usize, 2, 7] {
                for channels in [3usize, 4] {
                    let input: Vec<Vec<i32>> = (0..channels)
                        .map(|c| {
                            (0..width * height)
                                .map(|_| {
                                    random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                                    if c == 3 {
                                        0
                                    } else {
                                        (random % 131071) as i32 - 65535
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    let mut padded = vec![vec![123]];
                    padded.extend(input.clone());
                    padded.push(vec![456, 789]);
                    let original = padded.clone();
                    for id in 0..14 {
                        let predictor = Predictor::try_from(id).unwrap();
                        if predictor == Predictor::Weighted {
                            continue;
                        }
                        let result = delta_palette_forward(
                            &padded, 1, channels, width, height, 65536, 16, predictor,
                        )
                        .unwrap()
                        .unwrap();
                        assert_eq!(
                            expand_delta_palette(&result, width, 16),
                            input,
                            "{width}x{height}, channels={channels}, predictor={predictor:?}"
                        );
                        assert_eq!(padded, original);
                    }
                }
            }
        }
        let zeros = vec![vec![0; 7]; 4];
        let result = delta_palette_forward(&zeros, 0, 4, 7, 1, 1, 8, Predictor::Gradient)
            .unwrap()
            .unwrap();
        assert_eq!(result.num_deltas, 1);
        assert_eq!(result.indices, [0; 7]);
        assert_eq!(expand_delta_palette(&result, 7, 8), zeros);
    }

    #[test]
    fn zero_delta_mask_and_insertion_preserve_implicit_index_offsets() {
        let input = vec![vec![0, 0, 3, 0, 0, 0, 3, 3]; 3];
        let mask = exact_zero_delta_mask(&input, 0, 3, 4, 2, 8, Predictor::Left)
            .unwrap()
            .unwrap();
        assert_eq!(mask, [1, 1, 0, 0, 1, 1, 0, 1]);
        let ordinary = palette_forward(&input, 0, 3, 2, 8, false).unwrap().unwrap();
        let original = ordinary.clone();
        assert_eq!(ordinary.indices, [65, 65, 0, 65, 65, 65, 0, 0]);
        let output = add_exact_zero_delta(&ordinary, Predictor::Left, &mask).unwrap();
        assert_eq!(output.num_colors, 1);
        assert_eq!(output.num_deltas, 1);
        assert_eq!(output.palette, [vec![0, 3], vec![0, 3], vec![0, 3]]);
        assert_eq!(output.indices, [0, 0, 1, 66, 0, 0, 1, 0]);
        assert_eq!(expand_delta_palette(&output, 4, 8), input);
        assert_eq!(ordinary, original);

        let implicit_only = PaletteOutput {
            palette: vec![vec![]; 3],
            indices: vec![0, 64, 188],
        };
        let output = add_exact_zero_delta(&implicit_only, Predictor::Left, &[0; 3]).unwrap();
        assert_eq!(output.num_colors, 0);
        assert_eq!(output.palette, [vec![0], vec![0], vec![0]]);
        assert_eq!(output.indices, [1, 65, 189]);
        assert_eq!(
            expand_delta_palette(&output, 3, 8),
            vec![vec![32, 0, 255]; 3]
        );
    }

    #[test]
    fn zero_delta_requires_every_channel_and_wide_prediction_equality() {
        let input = vec![vec![0, 0], vec![0, 1], vec![0, 0], vec![0, 0]];
        let mask = exact_zero_delta_mask(&input, 0, 4, 2, 1, 8, Predictor::Left)
            .unwrap()
            .unwrap();
        assert_eq!(mask, [1, 0]);

        let mut sample = vec![0, i32::MIN, 0, i32::MAX, i32::MAX, i32::MAX, i32::MAX, 0, 0];
        let prediction = palette_prediction(&sample, 3, 7, Predictor::Average4).unwrap();
        assert!(prediction > i64::from(i32::MAX));
        sample[7] = prediction as i32;
        let mask = exact_zero_delta_mask(&vec![sample; 3], 0, 3, 3, 3, 16, Predictor::Average4)
            .unwrap()
            .unwrap();
        assert_eq!(mask[7], 0);
    }

    #[test]
    fn delta_palette_declines_unrepresentable_residuals_and_source_ineligibility() {
        let extremes = vec![vec![i32::MIN, i32::MAX]; 3];
        assert!(
            delta_palette_forward(&extremes, 0, 3, 2, 1, 65536, 16, Predictor::Left)
                .unwrap()
                .is_none()
        );
        let input = vec![vec![0, 1]; 3];
        for predictor in [Predictor::Weighted, Predictor::Best, Predictor::Variable] {
            assert!(
                delta_palette_forward(&input, 0, 3, 2, 1, 2, 8, predictor)
                    .unwrap()
                    .is_none()
            );
            assert!(
                exact_zero_delta_mask(&input, 0, 3, 2, 1, 8, predictor)
                    .unwrap()
                    .is_none()
            );
        }
        for channels in [1usize, 2, 5] {
            let input = vec![vec![0, 1]; channels];
            assert!(
                delta_palette_forward(&input, 0, channels, 2, 1, 2, 8, Predictor::Left)
                    .unwrap()
                    .is_none()
            );
            assert!(
                exact_zero_delta_mask(&input, 0, channels, 2, 1, 8, Predictor::Left)
                    .unwrap()
                    .is_none()
            );
        }
        for depth in [0, 17, 32] {
            assert!(
                delta_palette_forward(&input, 0, 3, 2, 1, 2, depth, Predictor::Left)
                    .unwrap()
                    .is_none()
            );
            assert!(
                exact_zero_delta_mask(&input, 0, 3, 2, 1, depth, Predictor::Left)
                    .unwrap()
                    .is_none()
            );
        }
        for limit in [0, 65537, u32::MAX] {
            assert!(
                delta_palette_forward(&input, 0, 3, 2, 1, limit, 8, Predictor::Left)
                    .unwrap()
                    .is_none()
            );
        }
        for (begin, channels, width, height) in [
            (0, 0, 2, 1),
            (1, 3, 2, 1),
            (usize::MAX, 3, 2, 1),
            (0, 3, usize::MAX, 2),
            (0, 3, 1, 1),
        ] {
            assert!(
                delta_palette_forward(
                    &input,
                    begin,
                    channels,
                    width,
                    height,
                    2,
                    8,
                    Predictor::Left
                )
                .is_err()
            );
            assert!(
                exact_zero_delta_mask(&input, begin, channels, width, height, 8, Predictor::Left)
                    .is_err()
            );
        }
        let empty = vec![vec![]; 3];
        assert!(
            delta_palette_forward(&empty, 0, 3, 0, 7, 2, 8, Predictor::Left)
                .unwrap()
                .is_none()
        );
        assert!(
            exact_zero_delta_mask(&empty, 0, 3, 0, 7, 8, Predictor::Left)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn zero_delta_insertion_rejects_invalid_tables_masks_and_indices_atomically() {
        let ordinary = PaletteOutput {
            palette: vec![vec![7]; 3],
            indices: vec![0, 64],
        };
        let original = ordinary.clone();
        assert!(add_exact_zero_delta(&ordinary, Predictor::Left, &[0]).is_err());
        for predictor in [Predictor::Weighted, Predictor::Best, Predictor::Variable] {
            assert!(add_exact_zero_delta(&ordinary, predictor, &[0, 1]).is_err());
        }
        for bad in [
            PaletteOutput {
                palette: vec![],
                indices: vec![0, 0],
            },
            PaletteOutput {
                palette: vec![vec![7], vec![]],
                indices: vec![0, 0],
            },
            PaletteOutput {
                palette: vec![vec![7]; 3],
                indices: vec![-1, 0],
            },
            PaletteOutput {
                palette: vec![vec![7]; 3],
                indices: vec![i32::MAX, 0],
            },
        ] {
            let copy = bad.clone();
            assert!(add_exact_zero_delta(&bad, Predictor::Left, &[1, 1]).is_err());
            assert_eq!(bad, copy);
        }
        assert_eq!(ordinary, original);
        let maximum = PaletteOutput {
            palette: vec![vec![]; 3],
            indices: vec![i32::MAX - 1],
        };
        assert_eq!(
            add_exact_zero_delta(&maximum, Predictor::Left, &[0])
                .unwrap()
                .indices,
            [i32::MAX]
        );
    }
}
