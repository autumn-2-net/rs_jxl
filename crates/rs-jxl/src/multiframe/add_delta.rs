//! Single-layer signed temporal residuals for 8-bit integer animation.
use super::{Rect, diff::DiffMap};
use crate::{Error, Image, PixelData, Result};

// The normalized f32 error from the initial conversion and each separate
// addition is conservatively below 4 * 2^-23 per step (all exact target
// values are in [0, 1]). Even 33 such errors scaled by 255 are below 0.005,
// inside the 0.00781 sample margin of the 8-bit output dither.
pub(super) const MAX_DEPTH: u8 = 32;

pub(super) fn promising(
    previous: &Image,
    current: &Image,
    rect: Rect,
    diff: &DiffMap,
    last: bool,
) -> bool {
    let (PixelData::U8(a), PixelData::U8(b)) = (previous.pixels(), current.pixels()) else {
        return false;
    };
    let area = u64::from(rect.width) * u64::from(rect.height);
    if area < 256
        || current.bits() != 8
        || (current.layout().has_alpha() && !last)
        || !super::multiply_mask::eligible(previous, current)
        || (0..current.channel_count()).any(|c| current.channel_depth(c).bits_per_sample != 8)
    {
        return false;
    }
    if diff.changed_pixels() <= area / 8 {
        return true;
    }
    // Screen dense updates on at most 16x16 pixels. A small residual alphabet
    // is useful for changing tints and fine edits even on a noisy background.
    let channels = current.layout().channels();
    let mut targets = [false; 256];
    let mut deltas = [false; 511];
    for y in (0..rect.height).step_by(rect.height.div_ceil(16) as usize) {
        for x in (0..rect.width).step_by(rect.width.div_ceil(16) as usize) {
            let start = ((rect.y + y) as usize * current.width() as usize + (rect.x + x) as usize)
                * channels;
            for c in start..start + channels {
                targets[usize::from(b[c])] = true;
                deltas[255 + usize::from(b[c]) - usize::from(a[c])] = true;
            }
        }
    }
    let distinct = |h: &[bool]| h.iter().filter(|&&v| v).count();
    distinct(&deltas) * 5 <= distinct(&targets) * 4
}

pub(super) fn image(
    previous: &Image,
    current: &Image,
    rect: Rect,
    diff: &DiffMap,
) -> Result<Image> {
    let (PixelData::U8(a), PixelData::U8(b)) = (previous.pixels(), current.pixels()) else {
        return Err(Error::InvalidInput(
            "temporal Add requires 8-bit samples".into(),
        ));
    };
    let count = (rect.width as usize)
        .checked_mul(rect.height as usize)
        .ok_or(Error::Limit("temporal Add pixels"))?;
    let channels = current.layout().channels();
    let mut planes = Vec::with_capacity(channels);
    for _ in 0..channels {
        let mut plane = Vec::new();
        plane
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("temporal Add allocation"))?;
        plane.resize(count, 0);
        planes.push(plane);
    }
    diff.changed_spans(rect, |src, dst, len| {
        for (c, plane) in planes.iter_mut().enumerate() {
            for (i, value) in plane[dst..dst + len].iter_mut().enumerate() {
                let pos = (src + i) * channels + c;
                *value = i32::from(b[pos]) - i32::from(a[pos]);
            }
        }
    });
    current.with_modular_planes(rect.width, rect.height, planes)
}
